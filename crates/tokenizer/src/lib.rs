//! Text to token ids and back for the BPE vocabularies this engine runs
//! (GPT-2 byte-level, and gemma4's metaspace), bit-identical to the
//! reference's `llama_tokenize` and `llama_token_to_piece`.
//!
//! The vocabulary comes from the GGUF header (`tokenizer.ggml.*`). Encoding
//! is the reference's pipeline: special tokens are cut out of the text as
//! whole pieces first (longest text first), each remaining fragment is
//! respelled for a metaspace vocabulary (each space becomes U+2581) and split
//! by the pre-tokenizer ([`pretok`]), and each word is merged by BPE rank
//! ([`bpe`]), unless the word is itself a token the pre-tokenizer takes
//! whole. Decoding concatenates pieces as `llama_token_to_piece` renders
//! them; neither spelling adds a space prefix, so decoding strips nothing.

mod bpe;
mod collapse_table;
mod decode;
mod pretok;
mod unicode;
mod vocab;

use std::path::Path;

use memchr::memmem::Finder;

pub use decode::Decoder;
pub use vocab::{Specials, attr};

use vocab::Vocab;

/// Why a vocabulary could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Gguf(String, gguf::LoadError),
    #[error("missing metadata key {0}")]
    Missing(&'static str),
    #[error("metadata key {0} has an unexpected type")]
    WrongType(&'static str),
    #[error("tokenizer model {0:?} is not supported ({known})", known = names(&vocab::Model::NAMES))]
    UnsupportedModel(String),
    #[error("pre-tokenizer {0:?} is not supported ({known})", known = names(&pretok::Pre::NAMES))]
    UnsupportedPre(String),
    #[error("tokenizer.ggml.token_type has {types} entries for {tokens} tokens")]
    TokenTypesShort { types: usize, tokens: usize },
    #[error("a vocabulary of {0} tokens does not fit u32 ids")]
    TooManyTokens(usize),
    #[error("token text {0:?} appears twice in the vocabulary")]
    DuplicateToken(String),
    #[error("the vocabulary has no byte token <0x{0:02X}>")]
    MissingByteToken(u8),
}

/// Whether this crate runs the pre-tokenizer `tokenizer.ggml.pre` names
/// `name`.
#[must_use]
pub fn runs_pre_tokenizer(name: &str) -> bool {
    named(&pretok::Pre::NAMES, name).is_some()
}

/// The value `name` stands for in a table of names this crate runs; `None`
/// for a name it does not run.
fn named<T: Copy>(table: &[(&str, T)], name: &str) -> Option<T> {
    table.iter().find(|(n, _)| *n == name).map(|&(_, v)| v)
}

/// A name table's names, comma-separated.
fn names<T>(table: &[(&str, T)]) -> String {
    let names: Vec<&str> = table.iter().map(|&(n, _)| n).collect();
    names.join(", ")
}

/// A loaded vocabulary with its encoder and decoder tables.
pub struct Tokenizer {
    vocab: Vocab,
    bpe: bpe::Bpe,
    /// Partition order: longest text first.
    specials: Vec<u32>,
    finders: Vec<Finder<'static>>,
    pieces: Vec<Box<[u8]>>,
}

/// One stretch of the input after the special-token partition.
#[derive(Clone, Copy)]
enum Frag {
    Text { start: usize, len: usize },
    Token(u32),
}

impl Tokenizer {
    /// Load the vocabulary from a GGUF file's header; for a split set, the
    /// first shard (the one that carries `tokenizer.ggml.*`).
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Tokenizer, Error> {
        let vocab = Vocab::load(path.as_ref())?;
        let spelling = vocab.pretok.spelling();
        let bpe = bpe::Bpe::new(
            &vocab.texts,
            &vocab.attrs,
            &vocab.merges,
            spelling,
            vocab.pretok.ignore_merges(),
        )?;
        let specials = vocab.special_order();
        let finders = specials
            .iter()
            .map(|&id| Finder::new(vocab.texts[id as usize].as_bytes()).into_owned())
            .collect();
        let pieces = decode::pieces(&vocab.texts, &vocab.attrs, spelling);
        Ok(Tokenizer {
            vocab,
            bpe,
            specials,
            finders,
            pieces,
        })
    }

    /// Token ids of `text`. `add_special` adds BOS/EOS where the vocabulary
    /// asks for them ([`Tokenizer::add_bos`], [`Tokenizer::add_eos`]);
    /// `parse_special` lets CONTROL tokens written in the text become their
    /// ids (USER_DEFINED tokens always do, as in the reference).
    pub fn encode(
        &self,
        text: impl AsRef<[u8]>,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let text = text.as_ref();
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special
            && self.vocab.add_bos
            && let Some(bos) = self.vocab.specials.bos
        {
            out.push(bos);
        }

        let mut cpts = Vec::new();
        let mut word = Vec::new();
        let mut pre = pretok::Scratch::default();
        let mut bpe = bpe::Scratch::default();
        let metaspace = self.vocab.pretok.spelling() == bpe::Spelling::Metaspace;
        for frag in self.partition(text, parse_special) {
            match frag {
                Frag::Token(id) => out.push(id),
                Frag::Text { start, len } => {
                    unicode::decode_lenient(&text[start..start + len], &mut cpts);
                    if metaspace {
                        for c in cpts.iter_mut().filter(|c| **c == u32::from(b' ')) {
                            *c = 0x2581;
                        }
                    }
                    let mut at = 0;
                    for &n in pretok::split(self.vocab.pretok, &cpts, &mut pre) {
                        word.clear();
                        for &cpt in &cpts[at..at + n] {
                            unicode::encode_cpt(cpt, &mut word);
                        }
                        at += n;
                        self.bpe
                            .word(&word, &mut bpe, &mut out, &self.vocab.token_to_id);
                    }
                }
            }
        }

        if add_special
            && self.vocab.add_eos
            && let Some(eos) = self.vocab.specials.eos
        {
            out.push(eos);
        }
        out
    }

    /// `tokenizer_st_partition`: for each special token in order, cut every
    /// occurrence out of every text fragment still unsplit.
    fn partition(&self, text: &[u8], parse_special: bool) -> Vec<Frag> {
        let mut frags = Vec::new();
        if text.is_empty() {
            return frags;
        }
        frags.push(Frag::Text {
            start: 0,
            len: text.len(),
        });
        let mut next = Vec::new();
        for (&id, finder) in self.specials.iter().zip(&self.finders) {
            if !parse_special
                && self.vocab.attrs[id as usize] & (attr::CONTROL | attr::UNKNOWN) != 0
            {
                continue;
            }
            let tlen = finder.needle().len();
            next.clear();
            let mut changed = false;
            for &f in &frags {
                let Frag::Text { start, len } = f else {
                    next.push(f);
                    continue;
                };
                let end = start + len;
                let mut at = start;
                while let Some(m) = finder.find(&text[at..end]) {
                    changed = true;
                    if m > 0 {
                        next.push(Frag::Text { start: at, len: m });
                    }
                    next.push(Frag::Token(id));
                    at += m + tlen;
                }
                if at < end {
                    next.push(Frag::Text {
                        start: at,
                        len: end - at,
                    });
                }
            }
            if changed {
                std::mem::swap(&mut frags, &mut next);
            }
        }
        frags
    }

    /// The bytes token `id` stands for; `special` renders CONTROL and UNKNOWN
    /// tokens as their text instead of nothing. An id outside the vocabulary
    /// is empty.
    pub fn piece(&self, id: u32, special: bool) -> &[u8] {
        let Some(p) = self.pieces.get(id as usize) else {
            return &[];
        };
        if !special && self.vocab.attrs[id as usize] & (attr::CONTROL | attr::UNKNOWN) != 0 {
            return &[];
        }
        p
    }

    /// The concatenated pieces of `ids`, exact bytes.
    pub fn decode_bytes(&self, ids: &[u32], special: bool) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| self.piece(id, special))
            .copied()
            .collect()
    }

    /// The text of `ids` with special tokens rendered; bytes that do not form
    /// UTF-8 become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, true)).into_owned()
    }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> usize {
        self.vocab.texts.len()
    }

    /// Token `id`'s text as the vocabulary writes it.
    pub fn text(&self, id: u32) -> Option<&str> {
        self.vocab.texts.get(id as usize).map(String::as_str)
    }

    /// Token `id`'s attribute bits ([`attr`]).
    pub fn attr(&self, id: u32) -> Option<u32> {
        self.vocab.attrs.get(id as usize).copied()
    }

    /// The special ids after the reference's defaults and text-based
    /// detection.
    pub fn specials(&self) -> &Specials {
        &self.vocab.specials
    }

    /// Whether the vocabulary asks for a BOS / an EOS around encoded text:
    /// `tokenizer.ggml.add_bos_token` / `add_eos_token`, and a BOS always for
    /// the `gemma4` pre-tokenizer, as the reference forces it.
    pub fn add_bos(&self) -> bool {
        self.vocab.add_bos
    }

    /// See [`Tokenizer::add_bos`].
    pub fn add_eos(&self) -> bool {
        self.vocab.add_eos
    }

    /// End-of-generation ids, ascending.
    pub fn eog(&self) -> &[u32] {
        &self.vocab.eog
    }

    /// The pre-tokenizer's name: `tokenizer.ggml.pre`, or `gemma4` for a
    /// gemma4 model, which names its own.
    pub fn pre(&self) -> &str {
        &self.vocab.pre
    }

    /// The special tokens in partition order.
    pub fn special_tokens(&self) -> &[u32] {
        &self.specials
    }

    /// Every metadata key the load read, with its value as shown text.
    pub fn metadata_read(&self) -> &[(String, String)] {
        &self.vocab.keys_read
    }

    /// Merges whose sides or result are not vocabulary tokens.
    pub fn merges_off_vocab(&self) -> usize {
        self.bpe.merges_off_vocab
    }

    /// Roles (eot, fim_pre, ...) filled by text where several candidate
    /// texts exist; the reference's choice among them is hash-map order.
    pub fn ambiguous_roles(&self) -> &[&'static str] {
        &self.vocab.ambiguous_roles
    }

    /// The reference sorts special tokens by length with an unstable sort, so
    /// the order among equal-length tokens is unspecified; it can change a
    /// partition only if two such tokens can overlap in text. Returns the
    /// first pair that can, if any.
    pub fn special_overlap(&self) -> Option<(u32, u32)> {
        let text = |id: u32| self.vocab.texts[id as usize].as_bytes();
        let s = &self.specials;
        let mut i = 0;
        while i < s.len() {
            let len = text(s[i]).len();
            let j = i + s[i..]
                .iter()
                .take_while(|&&id| text(id).len() == len)
                .count();
            for (x, &a) in s[i..j].iter().enumerate() {
                for &b in &s[i + x + 1..j] {
                    let (ta, tb) = (text(a), text(b));
                    let overlaps =
                        (1..len).any(|k| ta[len - k..] == tb[..k] || tb[len - k..] == ta[..k]);
                    if overlaps {
                        return Some((a, b));
                    }
                }
            }
            i = j;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use gguf::Value;
    use gguf::write::{Layout, Writer};

    use super::{Error, Tokenizer};

    /// A gemma4 vocabulary of a few tokens: six CONTROL tokens, one
    /// USER_DEFINED token, the 256 `<0xXX>` byte tokens but `missing_byte`,
    /// then the NORMAL tokens below. Like the released file's, the header
    /// names no pre-tokenizer; unlike it, it sets `add_bos_token` false, so a
    /// BOS can come only from the gemma4 rule. `header` adds the special ids
    /// and `add_eos_token`.
    fn gemma4_file(header: &[(&str, Value)], missing_byte: Option<u8>) -> std::path::PathBuf {
        let control = ["<pad>", "<eos>", "<bos>", "<unk>", "<turn|>", "<|channel|>"];
        let normal = [
            "a",
            "b",
            "\u{2581}",
            "\u{2581}b",
            "\u{2581}\u{2581}",
            "\u{2581}\u{2581}\u{2581}\u{2581}",
            "\n",
            "\n\n\n",
            "x",
            "y",
            "한",
            "국",
            "한국",
            "é",
            "😀",
        ];
        let mut texts: Vec<String> = control.iter().map(|t| t.to_string()).collect();
        let mut types = vec![3; control.len()];
        texts.push("<mark>".to_string());
        types.push(4);
        let bytes = (0..=u8::MAX).filter(|&b| Some(b) != missing_byte);
        texts.extend(bytes.map(|b| format!("<0x{b:02X}>")));
        types.resize(texts.len(), 6);
        texts.extend(normal.iter().map(|t| t.to_string()));
        types.resize(texts.len(), 1);
        let merges = [
            "\u{2581} b",
            "x y",
            "한 국",
            "\u{2581} \u{2581}",
            "\u{2581}\u{2581} \u{2581}\u{2581}",
        ];
        let kvs: Vec<(String, Value)> = [
            ("tokenizer.ggml.model", Value::String("gemma4".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(texts.into_iter().map(Value::String).collect()),
            ),
            (
                "tokenizer.ggml.token_type",
                Value::Array(types.into_iter().map(Value::I32).collect()),
            ),
            (
                "tokenizer.ggml.merges",
                Value::Array(
                    merges
                        .iter()
                        .map(|m| Value::String(m.to_string()))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ]
        .into_iter()
        .chain(header.iter().cloned())
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let path =
            std::env::temp_dir().join(format!("tokenizer-gemma4-{}.gguf", std::process::id()));
        let layout = Layout::new(&kvs, Vec::new()).unwrap();
        let file = std::fs::File::create(&path).unwrap();
        Writer::new(file, layout).and_then(Writer::finish).unwrap();
        path
    }

    fn open(header: &[(&str, Value)], missing_byte: Option<u8>) -> Result<Tokenizer, Error> {
        let path = gemma4_file(header, missing_byte);
        let tok = Tokenizer::from_gguf(&path);
        std::fs::remove_file(&path).unwrap();
        tok
    }

    /// The ids of the tokens whose texts are `texts`.
    fn ids(tok: &Tokenizer, texts: &[&str]) -> Vec<u32> {
        let id = |t: &str| {
            (0..tok.n_vocab() as u32)
                .find(|&i| tok.text(i) == Some(t))
                .unwrap()
        };
        texts.iter().map(|t| id(t)).collect()
    }

    #[test]
    fn gpt2_defaults_stay_in_vocab() {
        for n in [1, 11] {
            for (add_bos, add_eos, want) in [
                (
                    true,
                    false,
                    Some(Error::Missing("tokenizer.ggml.bos_token_id")),
                ),
                (
                    false,
                    true,
                    Some(Error::Missing("tokenizer.ggml.eos_token_id")),
                ),
                (false, false, None),
            ] {
                let kvs = [
                    ("tokenizer.ggml.model", Value::String("gpt2".into())),
                    ("tokenizer.ggml.pre", Value::String("qwen2".into())),
                    (
                        "tokenizer.ggml.tokens",
                        Value::Array(
                            (b'a'..)
                                .take(n)
                                .map(|b| Value::String(char::from(b).to_string()))
                                .collect(),
                        ),
                    ),
                    (
                        "tokenizer.ggml.merges",
                        Value::Array(vec![Value::String("a a".into())]),
                    ),
                    ("tokenizer.ggml.add_bos_token", Value::Bool(add_bos)),
                    ("tokenizer.ggml.add_eos_token", Value::Bool(add_eos)),
                ]
                .map(|(k, v)| (k.to_string(), v));
                let path = std::env::temp_dir().join(format!(
                    "tokenizer-gpt2-defaults-{}.gguf",
                    std::process::id()
                ));
                let layout = Layout::new(&kvs, Vec::new()).unwrap();
                let file = std::fs::File::create(&path).unwrap();
                Writer::new(file, layout).and_then(Writer::finish).unwrap();
                let got = Tokenizer::from_gguf(&path);
                std::fs::remove_file(&path).unwrap();
                if let Some(want) = want {
                    assert_eq!(got.err().map(|e| e.to_string()), Some(want.to_string()));
                } else {
                    let tok = got.unwrap();
                    assert_eq!(tok.specials().bos, None);
                    assert_eq!(tok.specials().eos, None);
                    assert!(tok.eog().is_empty());
                }
            }
        }
    }

    /// Every `<0xXX>` fallback must carry BYTE even when all 256 texts exist.
    #[test]
    fn gemma4_requires_byte_token_attributes() {
        for (token_type, byte) in [(None, 0), (Some(1), 0xBC), (Some(5), 0xBC)] {
            let texts: Vec<Value> = std::iter::once(Value::String("<bos>".into()))
                .chain((0..=u8::MAX).map(|b| Value::String(format!("<0x{b:02X}>"))))
                .collect();
            let mut types = vec![Value::I32(6); texts.len()];
            types[0] = Value::I32(3);
            let mut kvs: Vec<(String, Value)> = [
                ("tokenizer.ggml.model", Value::String("gemma4".into())),
                ("tokenizer.ggml.tokens", Value::Array(texts)),
                (
                    "tokenizer.ggml.merges",
                    Value::Array(vec![Value::String("<bos> <bos>".into())]),
                ),
                ("tokenizer.ggml.bos_token_id", Value::U32(0)),
            ]
            .map(|(k, v)| (k.to_string(), v))
            .into();
            if let Some(token_type) = token_type {
                types[usize::from(byte) + 1] = Value::I32(token_type);
                kvs.push(("tokenizer.ggml.token_type".into(), Value::Array(types)));
            }
            let path = std::env::temp_dir().join(format!(
                "tokenizer-gemma4-byte-attrs-{}.gguf",
                std::process::id()
            ));
            let layout = Layout::new(&kvs, Vec::new()).unwrap();
            let file = std::fs::File::create(&path).unwrap();
            Writer::new(file, layout).and_then(Writer::finish).unwrap();
            let got = Tokenizer::from_gguf(&path);
            std::fs::remove_file(&path).unwrap();
            assert_eq!(
                got.err().map(|e| e.to_string()),
                Some(Error::MissingByteToken(byte).to_string())
            );
        }
    }

    /// The gemma4 pipeline, one row per step the reference takes: a space is
    /// U+2581 and merges into the word after it, a word of newlines that is a
    /// token stays whole while one that is not merges as any word, symbols
    /// start as UTF-8 characters, a symbol that is no token falls back to
    /// `<0xXX>` tokens, USER_DEFINED tokens (and `<|channel|>`, which the
    /// reference makes one) are cut out without special parsing while
    /// CONTROL ones are not, and BOS is added though the file does not ask.
    /// Only U+0020 becomes U+2581; a literal U+2581 reads as a space, so it
    /// does not round-trip; invalid UTF-8 is one U+FFFD per bad byte. A
    /// header that asks for a BOS or an EOS it cannot name is refused, as the
    /// reference asserts one, and so is a vocabulary missing a byte token,
    /// where the fallback would drop that byte.
    ///
    /// FAIL-first: without the U+2581 respelling "a b" is `a <0x20> b`;
    /// without the newline-word lookup "\n\n\n" is three `\n`; with one symbol
    /// per byte "a b" falls back to the three bytes of U+2581; with the file's
    /// `add_bos_token` the BOS is missing; without the BOS-id, EOS-id or
    /// byte-token check that header loads.
    /// With a wrong two- or four-byte symbol length, "é" or "😀" falls back
    /// to bytes; with the file's `qwen2` pre-tokenizer, `pre()` is not gemma4.
    #[test]
    fn gemma4_encodes_as_the_reference() {
        let bos = ("tokenizer.ggml.bos_token_id", Value::U32(2));
        let eos = ("tokenizer.ggml.eos_token_id", Value::U32(1));
        let eos_out_of_range = ("tokenizer.ggml.eos_token_id", Value::U32(100_000));
        let add_eos = |on| ("tokenizer.ggml.add_eos_token", Value::Bool(on));
        let refused = [
            (
                vec![eos.clone(), add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.bos_token_id"),
            ),
            (
                vec![bos.clone(), add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.eos_token_id"),
            ),
            (
                vec![bos.clone(), eos_out_of_range, add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.eos_token_id"),
            ),
            (
                vec![bos.clone(), eos.clone(), add_eos(true)],
                Some(0xBC),
                Error::MissingByteToken(0xBC),
            ),
        ];
        for (header, missing_byte, want) in refused {
            let got = open(&header, missing_byte).err().map(|e| e.to_string());
            assert_eq!(got, Some(want.to_string()));
        }

        let tok = open(&[bos.clone(), add_eos(false)], None).unwrap();
        assert_eq!(tok.encode("a", true, true), ids(&tok, &["<bos>", "a"]));

        let tok = open(
            &[
                bos,
                eos,
                add_eos(true),
                ("tokenizer.ggml.pre", Value::String("qwen2".into())),
            ],
            None,
        )
        .unwrap();
        assert_eq!(tok.pre(), "gemma4");
        let fffd = ["<0xEF>", "<0xBF>", "<0xBD>"];
        let cases: &[(&[u8], bool, &[&str])] = &[
            (b"a b", true, &["a", "\u{2581}b"]),
            ("a\u{2581}b".as_bytes(), true, &["a", "\u{2581}b"]),
            (b"a\n\n\nb", true, &["a", "\n\n\n", "b"]),
            (b"\n\n", true, &["\n", "\n"]),
            (b"\n\n\n\n", true, &["\n", "\n", "\n", "\n"]),
            (b"a\r\n\n\nb", true, &["a", "<0x0D>", "\n\n\n", "b"]),
            (
                b"     ",
                true,
                &["\u{2581}\u{2581}\u{2581}\u{2581}", "\u{2581}"],
            ),
            ("한국".as_bytes(), true, &["한국"]),
            ("é".as_bytes(), true, &["é"]),
            ("😀".as_bytes(), true, &["😀"]),
            (b"xy", true, &["<0x78>", "<0x79>"]),
            ("ü".as_bytes(), true, &["<0xC3>", "<0xBC>"]),
            ("a\u{a0}b".as_bytes(), true, &["a", "<0xC2>", "<0xA0>", "b"]),
            (
                "a\u{3000}b".as_bytes(),
                true,
                &["a", "<0xE3>", "<0x80>", "<0x80>", "b"],
            ),
            (
                b"a\xF0\x9F b",
                true,
                &[
                    "a",
                    fffd[0],
                    fffd[1],
                    fffd[2],
                    fffd[0],
                    fffd[1],
                    fffd[2],
                    "\u{2581}b",
                ],
            ),
            (b"a<|channel|>b", false, &["a", "<|channel|>", "b"]),
            (b"<mark> b", false, &["<mark>", "\u{2581}b"]),
            (b"a <turn|>b", true, &["a", "\u{2581}", "<turn|>", "b"]),
            (
                b"a <turn|>b",
                false,
                &[
                    "a", "\u{2581}", "<0x3C>", "<0x74>", "<0x75>", "<0x72>", "<0x6E>", "<0x7C>",
                    "<0x3E>", "b",
                ],
            ),
        ];
        for &(text, parse_special, want) in cases {
            let got = tok.encode(text, false, parse_special);
            let text = String::from_utf8_lossy(text);
            assert_eq!(got, ids(&tok, want), "{text:?} {parse_special}");
        }
        assert_eq!(
            tok.encode("a", true, true),
            ids(&tok, &["<bos>", "a", "<eos>"])
        );

        let decode = |text: &[u8]| {
            let round = tok.decode_bytes(&tok.encode(text, true, true), false);
            String::from_utf8(round).unwrap()
        };
        assert_eq!(decode("a\u{2581}b".as_bytes()), "a b");
        assert_eq!(decode(b"a\xF0\x9F b"), "a\u{FFFD}\u{FFFD} b");
        let text = "a b\n\n\nü 한국";
        assert_eq!(decode(text.as_bytes()), text);
    }
}
