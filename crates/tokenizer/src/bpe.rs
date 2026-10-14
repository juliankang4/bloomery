//! BPE on one pre-tokenizer word, as `llm_tokenizer_bpe_session` runs it,
//! in either of the two spellings the reference's vocabularies use
//! ([`Spelling`]).
//!
//! Every unit of the word starts as one symbol: a byte, written in the
//! vocabulary as its GPT-2 alphabet character, or a UTF-8 character as it
//! is. The adjacent pair with the lowest merge rank merges first, ties going
//! to the leftmost pair; a queued pair whose sides have changed since it was
//! queued is skipped. A finished symbol whose text is not a token falls back
//! to byte tokens (the spelling says which). A byte-level byte with no such
//! token is dropped, as in the reference; a metaspace vocabulary without all
//! 256 `<0xXX>` tokens carrying BYTE is refused ([`Bpe::new`]).
//!
//! A pre-tokenizer may ask that a word whose spelled text is itself a
//! vocabulary token be that token, with no merges ([`Bpe::new`]'s
//! `ignore_merges`). The flag is the pre-tokenizer's, not this module's.
//!
//! Strings are interned so the merge loop compares integers: token texts
//! first, so a token's intern id is its token id, then any merge side or
//! merge result that is not a token. An interned id names one exact string,
//! so comparing ids is comparing the reference's strings.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::Error;
use crate::unicode::{byte_to_cpt_table, encode_cpt};

const NONE: u32 = u32::MAX;

/// How a vocabulary spells a word's text: the reference's `byte_encode`
/// and `escape_whitespaces` flags, which its pre-tokenizer names set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spelling {
    /// GPT-2 byte-level: each byte is written as its alphabet character, and
    /// a symbol that is not a token falls back to the tokens whose text is
    /// one byte of those characters.
    ByteLevel,
    /// Metaspace: each space is written as U+2581, symbols start as raw UTF-8
    /// characters, a word of newlines that is a token is that token, and a
    /// symbol that is not a token falls back to the `<0xXX>` token of each of
    /// its bytes.
    Metaspace,
}

/// Each spelling's starting symbols and byte fallback; the tables are boxed
/// so the two variants are the same small size.
enum Units {
    ByteLevel {
        /// The intern id of each byte's one-character string, or `NONE`.
        byte_iid: Box<[u32; 256]>,
        /// Each byte's GPT-2 alphabet character, UTF-8.
        byte_text: Box<[Vec<u8>; 256]>,
        /// The token whose text is exactly this one byte, or `NONE`.
        one_byte_token: Box<[u32; 256]>,
    },
    Metaspace {
        /// The intern id of every one-character string a token or a merge
        /// names.
        char_iid: HashMap<Box<[u8]>, u32>,
        /// The `<0xXX>` token of each byte.
        hex_token: Box<[u32; 256]>,
    },
}

pub(crate) struct Bpe {
    n_tokens: u32,
    units: Units,
    /// `(left, right)` intern ids -> `(rank, merged intern id)`; the first
    /// listing of a pair keeps its rank.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Merges whose sides or result are not tokens (reported, not needed).
    pub(crate) merges_off_vocab: usize,
    ignore_merges: bool,
}

struct Interner {
    ids: HashMap<Vec<u8>, u32>,
}

impl Interner {
    fn get_or_add(&mut self, s: &[u8]) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = u32::try_from(self.ids.len()).expect("fewer than 2^32 distinct strings");
        self.ids.insert(s.to_vec(), id);
        id
    }
}

impl Bpe {
    /// `ignore_merges`: a word whose spelled text is a token is that token
    /// ([`Bpe::word`]), found in the vocabulary's own lookup. A metaspace
    /// vocabulary must have the `<0xXX>` token of every byte carrying BYTE
    /// ([`Error::MissingByteToken`]), so a symbol that is no token never
    /// loses a byte.
    pub(crate) fn new(
        texts: &[String],
        attrs: &[u32],
        merges: &[String],
        spelling: Spelling,
        ignore_merges: bool,
    ) -> Result<Bpe, Error> {
        let n_tokens = u32::try_from(texts.len()).expect("the vocabulary loader bounds the count");
        let mut interner = Interner {
            ids: HashMap::with_capacity(texts.len() + merges.len()),
        };
        for t in texts {
            interner.get_or_add(t.as_bytes());
        }

        let mut off_vocab = 0;
        let mut table = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let m = m.as_bytes();
            // The reference splits at the first space after the first byte;
            // a merge with no such space is the pair ("", "").
            let (first, second) = match m.iter().skip(1).position(|&b| b == b' ') {
                Some(p) => (&m[..p + 1], &m[p + 2..]),
                None => (&m[..0], &m[..0]),
            };
            let merged = [first, second].concat();
            let (l, r, j) = (
                interner.get_or_add(first),
                interner.get_or_add(second),
                interner.get_or_add(&merged),
            );
            if l >= n_tokens || r >= n_tokens || j >= n_tokens {
                off_vocab += 1;
            }
            let rank = u32::try_from(rank).expect("fewer than 2^32 merges");
            table.entry((l, r)).or_insert((rank, j));
        }

        let token = |s: &[u8]| {
            let id = interner.ids.get(s).copied().unwrap_or(NONE);
            if id < n_tokens { id } else { NONE }
        };
        let units = match spelling {
            Spelling::ByteLevel => {
                let cpts = byte_to_cpt_table();
                let byte_text: Box<[Vec<u8>; 256]> = Box::new(std::array::from_fn(|b| {
                    let mut s = Vec::with_capacity(2);
                    encode_cpt(cpts[b], &mut s);
                    s
                }));
                let byte_iid = Box::new(std::array::from_fn(|b| {
                    interner.ids.get(&byte_text[b]).copied().unwrap_or(NONE)
                }));
                let one_byte_token = Box::new(std::array::from_fn(|b| token(&[b as u8])));
                Units::ByteLevel {
                    byte_iid,
                    byte_text,
                    one_byte_token,
                }
            }
            Spelling::Metaspace => {
                let char_iid = interner
                    .ids
                    .iter()
                    .filter(|(s, _)| {
                        s.len() <= 4 && std::str::from_utf8(s).is_ok_and(|s| s.chars().count() == 1)
                    })
                    .map(|(s, &id)| (s.as_slice().into(), id))
                    .collect();
                let mut hex_token = Box::new([NONE; 256]);
                for b in 0..=u8::MAX {
                    let t = token(format!("<0x{b:02X}>").as_bytes());
                    if t == NONE || attrs[t as usize] & crate::vocab::attr::BYTE == 0 {
                        return Err(Error::MissingByteToken(b));
                    }
                    hex_token[usize::from(b)] = t;
                }
                Units::Metaspace {
                    char_iid,
                    hex_token,
                }
            }
        };
        Ok(Bpe {
            n_tokens,
            units,
            merges: table,
            merges_off_vocab: off_vocab,
            ignore_merges,
        })
    }

    /// The token whose text is `word` as the vocabulary spells it, if the
    /// vocabulary has that text.
    fn whole_token(
        &self,
        word: &[u8],
        s: &mut Scratch,
        tokens: &HashMap<String, u32>,
    ) -> Option<u32> {
        let text = match &self.units {
            Units::ByteLevel { byte_text, .. } => {
                s.encoded.clear();
                for &b in word {
                    s.encoded.extend_from_slice(&byte_text[b as usize]);
                }
                std::str::from_utf8(&s.encoded).expect("the GPT-2 byte alphabet is valid UTF-8")
            }
            // A word that is not UTF-8 is no token's text.
            Units::Metaspace { .. } => std::str::from_utf8(word).ok()?,
        };
        tokens.get(text).copied()
    }

    /// Append the tokens of one word (raw bytes, before the vocabulary's
    /// spelling) to `out`. Under `ignore_merges`, and for a metaspace word of
    /// newlines, a word whose spelled text is a token is that token.
    ///
    /// Panics on a word of more than `u32::MAX` bytes, which the `u32`
    /// symbol links cannot index.
    pub(crate) fn word(
        &self,
        word: &[u8],
        s: &mut Scratch,
        out: &mut Vec<u32>,
        tokens: &HashMap<String, u32>,
    ) {
        let newlines =
            matches!(self.units, Units::Metaspace { .. }) && word.iter().all(|&b| b == b'\n');
        if (self.ignore_merges || newlines)
            && let Some(id) = self.whole_token(word, s, tokens)
        {
            out.push(id);
            return;
        }
        let n = word.len();
        assert!(
            u32::try_from(n).is_ok(),
            "a pre-tokenizer word of {n} bytes does not fit the u32 symbol links"
        );
        let at = |i: usize| u32::try_from(i).expect("at most the word length, checked above");
        s.syms.clear();
        match &self.units {
            Units::ByteLevel { byte_iid, .. } => {
                s.syms.extend(word.iter().enumerate().map(|(i, &b)| Sym {
                    len: 1,
                    iid: byte_iid[b as usize],
                    prev: if i == 0 { NONE } else { at(i - 1) },
                    next: if i + 1 == n { NONE } else { at(i + 1) },
                }));
            }
            Units::Metaspace { char_iid, .. } => {
                // One symbol per character, at its lead byte; the bytes after
                // it start dead.
                s.syms.resize(
                    n,
                    Sym {
                        len: 0,
                        iid: NONE,
                        prev: NONE,
                        next: NONE,
                    },
                );
                let (mut i, mut prev) = (0, NONE);
                while i < n {
                    let len = utf8_len(word[i]).min(n - i);
                    s.syms[i] = Sym {
                        len: at(len),
                        iid: char_iid.get(&word[i..i + len]).copied().unwrap_or(NONE),
                        prev,
                        next: if i + len == n { NONE } else { at(i + len) },
                    };
                    prev = at(i);
                    i += len;
                }
            }
        }
        s.heap.clear();
        for i in 0..n {
            let next = s.syms[i].next;
            if s.syms[i].len > 0 && next != NONE {
                self.push(&s.syms, &mut s.heap, at(i), next);
            }
        }

        while let Some(Reverse((_, left, right, size, merged))) = s.heap.pop() {
            let (l, r) = (s.syms[left as usize], s.syms[right as usize]);
            if l.len == 0 || r.len == 0 || l.len + r.len != size {
                continue;
            }
            let next = r.next;
            s.syms[left as usize] = Sym {
                len: size,
                iid: merged,
                next,
                ..l
            };
            s.syms[right as usize].len = 0;
            if next != NONE {
                s.syms[next as usize].prev = left;
            }
            if l.prev != NONE {
                self.push(&s.syms, &mut s.heap, l.prev, left);
            }
            if next != NONE {
                self.push(&s.syms, &mut s.heap, left, next);
            }
        }

        // A live symbol starts at the byte whose index it has.
        for (start, sym) in s.syms.iter().enumerate() {
            let len = sym.len as usize;
            if len == 0 {
                continue;
            }
            if sym.iid < self.n_tokens {
                out.push(sym.iid);
                continue;
            }
            let bytes = &word[start..start + len];
            match &self.units {
                Units::ByteLevel {
                    byte_text,
                    one_byte_token,
                    ..
                } => {
                    for &b in bytes {
                        for &e in &byte_text[b as usize] {
                            let t = one_byte_token[e as usize];
                            if t != NONE {
                                out.push(t);
                            }
                        }
                    }
                }
                Units::Metaspace { hex_token, .. } => {
                    out.extend(bytes.iter().map(|&b| hex_token[usize::from(b)]));
                }
            }
        }
    }

    fn push(&self, syms: &[Sym], heap: &mut Heap, left: u32, right: u32) {
        let (l, r) = (syms[left as usize], syms[right as usize]);
        if l.iid == NONE || r.iid == NONE {
            return;
        }
        if let Some(&(rank, merged)) = self.merges.get(&(l.iid, r.iid)) {
            heap.push(Reverse((rank, left, right, l.len + r.len, merged)));
        }
    }
}

/// `unicode_len_utf8`: a character's length from its lead byte; a
/// continuation byte counts as one.
fn utf8_len(lead: u8) -> usize {
    match lead >> 4 {
        0xC | 0xD => 2,
        0xE => 3,
        0xF => 4,
        _ => 1,
    }
}

#[derive(Clone, Copy)]
struct Sym {
    /// Bytes this symbol covers; 0 once merged into its left neighbour.
    len: u32,
    iid: u32,
    prev: u32,
    next: u32,
}

/// `(rank, left, right, size at queueing, merged intern id)`, smallest first.
type Heap = BinaryHeap<Reverse<(u32, u32, u32, u32, u32)>>;

/// Reusable buffers for [`Bpe::word`].
#[derive(Default)]
pub(crate) struct Scratch {
    syms: Vec<Sym>,
    heap: Heap,
    /// The GPT-2 byte encoding of one word, for the whole-token lookup.
    encoded: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::{Bpe, Scratch, Spelling};

    /// A word that is a token but that no merge reaches: under
    /// `ignore_merges` it is that token, otherwise its bytes' tokens.
    #[test]
    fn ignore_merges_takes_a_whole_token_no_merge_reaches() {
        let texts = ["Ġ", "a", "Ġa"].map(String::from);
        let tokens = texts.iter().cloned().zip(0u32..).collect();
        let ids = |ignore_merges: bool| {
            let bpe = Bpe::new(
                &texts,
                &[crate::vocab::attr::NORMAL; 3],
                &[],
                Spelling::ByteLevel,
                ignore_merges,
            )
            .unwrap();
            let (mut s, mut out) = (Scratch::default(), Vec::new());
            bpe.word(b" a", &mut s, &mut out, &tokens);
            out
        };
        assert_eq!(ids(true), [2]);
        assert_eq!(ids(false), [0, 1]);
    }
}
