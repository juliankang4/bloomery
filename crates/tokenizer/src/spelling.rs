//! Shared text spelling and byte fallback for vocabulary encoders.

use crate::Error;

/// How a vocabulary spells a word's text: the reference's `byte_encode`
/// and `escape_whitespaces` flags, which its pre-tokenizer names set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spelling {
    /// GPT-2 byte-level: each byte is written as its alphabet character, and
    /// a symbol that is not a token falls back to the tokens whose text is
    /// one byte of those characters.
    ByteLevel,
    /// Metaspace: each space is written as U+2581, symbols start as raw UTF-8
    /// characters, and a
    /// symbol that is not a token falls back to the `<0xXX>` token of each of
    /// its bytes.
    Metaspace,
}

/// Write spaces as U+2581, preserving every other byte.
pub(crate) fn escape_whitespace(text: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for (i, part) in text.split(|&b| b == b' ').enumerate() {
        if i > 0 {
            out.extend_from_slice("▁".as_bytes());
        }
        out.extend_from_slice(part);
    }
}

/// The byte fallback table; every byte must have a token.
pub(crate) fn byte_tokens(
    mut token: impl FnMut(u8, &str) -> Option<u32>,
) -> Result<Box<[u32; 256]>, Error> {
    let mut ids = Box::new([0; 256]);
    for b in 0..=u8::MAX {
        ids[usize::from(b)] =
            token(b, &format!("<0x{b:02X}>")).ok_or(Error::MissingByteToken(b))?;
    }
    Ok(ids)
}
