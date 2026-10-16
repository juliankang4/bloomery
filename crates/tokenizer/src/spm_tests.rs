//! Fixed SentencePiece cases from the reference's score and fragment rules.

use gguf::Value;

use crate::tests::{from_header, ids};
use crate::{Decoder, Error};

fn header() -> Vec<(String, Value)> {
    let mut texts = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "<mark>".into()];
    let mut types = vec![2, 3, 3, 4];
    texts.extend((0..=u8::MAX).map(|b| format!("<0x{b:02X}>")));
    types.resize(texts.len(), 6);
    let normal = [
        "a", "b", "c", "ab", "bc", "▁", "▁a", "▁b", "한", "국", "한국", "é", "😀", "\n", "\n\n\n",
    ];
    texts.extend(normal.map(String::from));
    types.resize(texts.len(), 1);
    let scores = texts
        .iter()
        .map(|text| {
            Value::F32(match text.as_str() {
                "ab" | "▁a" | "▁b" => 1.0,
                "bc" | "한국" => 2.0,
                _ => -100.0,
            })
        })
        .collect();
    [
        ("tokenizer.ggml.model", Value::String("llama".into())),
        (
            "tokenizer.ggml.tokens",
            Value::Array(texts.into_iter().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type",
            Value::Array(types.into_iter().map(Value::I32).collect()),
        ),
        ("tokenizer.ggml.scores", Value::Array(scores)),
        ("tokenizer.ggml.add_space_prefix", Value::Bool(false)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

fn set(kvs: &mut Vec<(String, Value)>, key: &str, value: Value) {
    if let Some((_, old)) = kvs.iter_mut().find(|(k, _)| k == key) {
        *old = value;
    } else {
        kvs.push((key.to_string(), value));
    }
}

/// Highest score first, leftmost on ties, no pre-tokenizer or whole-word shortcut.
#[test]
fn sentencepiece_scores_and_bytes() {
    let mut kvs = header();
    set(
        &mut kvs,
        "tokenizer.ggml.pre",
        Value::String("qwen2".into()),
    );
    let tok = from_header(&kvs).unwrap();
    assert_eq!(tok.pre(), "default");
    assert!(crate::runs_pre_tokenizer("llama"));
    assert!(!crate::runs_pre_tokenizer("default"));
    let cases: &[(&[u8], &[&str])] = &[
        (b"", &[]),
        (b"abc", &["a", "bc"]),
        (b"ababc", &["ab", "a", "bc"]),
        (b"a b", &["a", "▁b"]),
        (b"  a", &["▁", "▁a"]),
        ("a▁b".as_bytes(), &["a", "▁b"]),
        (b"\n\n\n", &["\n", "\n", "\n"]),
        (b"a\r\nb", &["a", "<0x0D>", "\n", "b"]),
        ("한국é😀".as_bytes(), &["한국", "é", "😀"]),
        ("ü".as_bytes(), &["<0xC3>", "<0xBC>"]),
        ("a\u{a0}b".as_bytes(), &["a", "<0xC2>", "<0xA0>", "b"]),
        (b"a\xFFb", &["a", "<0xFF>", "<0x62>"]),
        (b"\xF0\x9F", &["<0xF0>", "<0x9F>"]),
        (b"\xED\xA0\x80", &["<0xED>", "<0xA0>", "<0x80>"]),
        (b"\0", &["<0x00>"]),
    ];
    for &(text, want) in cases {
        let got = tok.encode(text, false, true);
        assert_eq!(got, ids(&tok, want), "{text:?}");
        let decoded = tok.decode_bytes(&got, true);
        if text != "a▁b".as_bytes() {
            assert_eq!(decoded, text, "{text:?}");
        } else {
            assert_eq!(decoded, b"a b");
        }
    }
    let text = "a b 한국é😀ü";
    let mut decoder = Decoder::new(&tok, true);
    let mut streamed = String::new();
    for id in tok.encode(text, false, true) {
        if let Some(piece) = decoder.push(id) {
            streamed.push_str(piece);
        }
    }
    streamed.extend(decoder.finish());
    assert_eq!(streamed, text);

    for integer in [false, true] {
        let mut kvs = header();
        let Value::Array(scores) = &mut kvs[3].1 else {
            panic!("scores array")
        };
        for (id, score) in scores.iter_mut().enumerate() {
            *score = if integer {
                Value::I32(0)
            } else if id == 263 {
                Value::F32(-0.0)
            } else {
                Value::F32(0.0)
            };
        }
        let tok = from_header(&kvs).unwrap();
        assert_eq!(tok.encode("abc", false, true), ids(&tok, &["ab", "c"]));
    }
    let mut literal = header();
    let Value::Array(texts) = &mut literal[1].1 else {
        panic!("tokens array")
    };
    texts[4 + usize::from(b'a')] = Value::String("<missing-a>".into());
    for token_type in [1, 4] {
        let Value::Array(types) = &mut literal[2].1 else {
            panic!("token types array")
        };
        types[260] = Value::I32(token_type);
        let tok = from_header(&literal).unwrap();
        let encoded = tok.encode(b"\xC0a", false, true);
        assert_eq!(encoded, ids(&tok, &["<0xC0>", "a"]));
        assert_eq!(tok.decode_bytes(&encoded, true), b"\xC0a");
    }

    kvs.retain(|(key, _)| key != "tokenizer.ggml.scores");
    let tok = from_header(&kvs).unwrap();
    assert_eq!(tok.encode("abc", false, true), ids(&tok, &["ab", "c"]));
}

/// A token marked unused is still a merge candidate, as in llama-vocab's SentencePiece session.
#[test]
fn sentencepiece_merges_unused_tokens() {
    let mut kvs = header();
    let Some((_, Value::Array(types))) = kvs
        .iter_mut()
        .find(|(k, _)| k == "tokenizer.ggml.token_type")
    else {
        unreachable!()
    };
    types[4 + 256 + 3] = Value::I32(5);
    let tok = from_header(&kvs).unwrap();
    assert_eq!(tok.encode("ab", false, true), ids(&tok, &["ab"]));
}

/// An out-of-range special id in the header keeps the default, as llama-vocab does.
#[test]
fn sentencepiece_keeps_defaults_for_out_of_range_specials() {
    for (key, value) in [
        ("tokenizer.ggml.bos_token_id", Value::U32(100_000)),
        ("tokenizer.ggml.eos_token_id", Value::U32(u32::MAX)),
    ] {
        let mut kvs = header();
        set(&mut kvs, key, value);
        let tok = from_header(&kvs).unwrap();
        assert_eq!(
            (tok.specials().bos, tok.specials().eos),
            (Some(1), Some(2)),
            "{key}"
        );
    }
}

/// Prefix spaces follow every parsed special, and BOS/EOS follow header flags.
#[test]
fn sentencepiece_special_fragments() {
    let mut kvs = header();
    kvs.retain(|(key, _)| key != "tokenizer.ggml.add_space_prefix");
    let tok = from_header(&kvs).unwrap();
    assert!(tok.add_bos());
    assert!(!tok.add_eos());
    assert_eq!(tok.specials().unk, Some(0));
    assert_eq!(tok.encode("", true, true), ids(&tok, &["<s>"]));
    for parse in [false, true] {
        assert_eq!(
            tok.encode("a<mark>b", false, parse),
            ids(&tok, &["▁a", "<mark>", "▁b"])
        );
    }
    assert_eq!(
        tok.encode("a<s>b", false, true),
        ids(&tok, &["▁a", "<s>", "▁b"])
    );
    assert_eq!(
        tok.encode("<unk>b", false, true),
        ids(&tok, &["<unk>", "▁b"])
    );
    assert_eq!(
        tok.encode("<s>b", false, false),
        ids(&tok, &["▁", "<0x3C>", "<0x73>", "<0x3E>", "b"])
    );
    assert_eq!(
        tok.decode_bytes(&tok.encode("a<s>b", true, true), false),
        b" a b"
    );

    set(
        &mut kvs,
        "tokenizer.ggml.add_space_prefix",
        Value::Bool(false),
    );
    set(&mut kvs, "tokenizer.ggml.add_eos_token", Value::Bool(true));
    let tok = from_header(&kvs).unwrap();
    assert_eq!(
        tok.encode("a<mark>b", true, false),
        ids(&tok, &["<s>", "a", "<mark>", "b", "</s>"])
    );
    assert_eq!(tok.encode("", true, true), ids(&tok, &["<s>", "</s>"]));
    set(&mut kvs, "tokenizer.ggml.add_bos_token", Value::Bool(false));
    let tok = from_header(&kvs).unwrap();
    assert_eq!(tok.encode("a", true, true), ids(&tok, &["a", "</s>"]));
}

/// Invalid scores and missing byte fallback fail by name before encoding.
#[test]
fn sentencepiece_refuses_invalid_tables() {
    let refused = [
        (
            Value::Array(vec![Value::F32(0.0)]),
            Error::TokenScoresShort {
                scores: 1,
                tokens: 275,
            },
        ),
        (
            Value::String("scores".into()),
            Error::WrongType("tokenizer.ggml.scores"),
        ),
        (
            Value::Array(vec![Value::U32(0)]),
            Error::WrongType("tokenizer.ggml.scores"),
        ),
    ];
    for (value, want) in refused {
        let mut kvs = header();
        set(&mut kvs, "tokenizer.ggml.scores", value);
        let got = from_header(&kvs).err().unwrap();
        assert_eq!(got.to_string(), want.to_string());
    }
    for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut kvs = header();
        let Value::Array(scores) = &mut kvs[3].1 else {
            panic!("scores array")
        };
        scores[0] = Value::F32(score);
        assert!(matches!(
            from_header(&kvs),
            Err(Error::InvalidTokenScore { id: 0, .. })
        ));
    }
    for byte in [0xC3, 0xFF] {
        for token_type in [1, 5] {
            let mut kvs = header();
            let Value::Array(types) = &mut kvs[2].1 else {
                panic!("token types array")
            };
            types[4 + usize::from(byte)] = Value::I32(token_type);
            assert!(matches!(
                from_header(&kvs),
                Err(Error::MissingByteToken(b)) if b == byte
            ));
        }
    }
    for token_type in [0, 2, 3, 5, 6] {
        let mut kvs = header();
        let Value::Array(texts) = &mut kvs[1].1 else {
            panic!("token array")
        };
        texts[4 + usize::from(b'a')] = Value::String("<missing>".into());
        let Value::Array(types) = &mut kvs[2].1 else {
            panic!("token types array")
        };
        types[260] = Value::I32(token_type);
        assert!(matches!(
            from_header(&kvs),
            Err(Error::MissingByteToken(b'a'))
        ));
    }
    for n in 1..=3 {
        for add_eos in [false, true] {
            let mut kvs = header();
            for (_, value) in &mut kvs[1..=3] {
                let Value::Array(values) = value else {
                    panic!("vocabulary array")
                };
                values.truncate(n);
            }
            set(
                &mut kvs,
                "tokenizer.ggml.add_eos_token",
                Value::Bool(add_eos),
            );
            let error = from_header(&kvs).err().unwrap();
            match (n, add_eos) {
                (1, _) => assert!(matches!(
                    error,
                    Error::Missing("tokenizer.ggml.bos_token_id")
                )),
                (2, true) => assert!(matches!(
                    error,
                    Error::Missing("tokenizer.ggml.eos_token_id")
                )),
                _ => assert!(matches!(error, Error::MissingByteToken(0))),
            }
        }
    }
    let mut kvs = header();
    let Value::Array(texts) = &mut kvs[1].1 else {
        panic!("token array")
    };
    texts[4] = Value::String("<missing>".into());
    assert!(matches!(from_header(&kvs), Err(Error::MissingByteToken(0))));
}
