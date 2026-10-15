//! EmbeddingGemma 2 (`gemma-embedding2`, its image tower `gemma4v`). Every set
//! is llama.cpp mainline's, dumped by the tree [`crate::emb2::LCPP_BUILD`]
//! (`tools/ref/emb2/emb2.sh`, `just dump-ref-emb2`) from the Q8_0 file
//! [`MODEL`] and the bf16 projector [`MMPROJ_BF16`]:
//!
//! - [`TEXT`]: `dump_emb2 text` over the text cases (the tokenizer oracle's
//!   gemma4 set, `ref-tokenizer-gemma4`):
//!   per case the token rows (`embd`, the output projection of every
//!   position), the mean-pooled vector before normalization (`pooled`), and
//!   the taps `inp_scaled`, `inp_per_layer`, `l_out-<il>` of every layer,
//!   `result_norm` and `result_embd`. [`TEXT_CPU`] is its CPU twin.
//! - [`TOWER`]: `dump_emb2 tower` over the image cases: per image mtmd's rows
//!   (`embd`) and the tower's taps. As the other image-input seats' tower sets,
//!   its `# model` is the projector file and its `# arch` the projector's
//!   (`clip`). [`TOWER_CPU`] is its CPU twin.
//! - [`SERVER`]: llama-server's `/v1/embeddings` vectors of every text case and
//!   image, normalized and raw (`tools/ref/emb2/server_vectors.py`).
//!
//! Each family opens its sets through [`crate::emb2::Emb2Set`] with
//! [`PROFILE`]: the node dumps' checks (the trailer, `# model`, `# arch`,
//! `# build`), then the kind of set, the device and the files' sha256.

use crate::RefError;
use crate::emb2::{Kind, Profile, family};
use crate::family::Family;

/// The text file every set is dumped from: `ggml-org/embeddinggemma-2-GGUF`'s
/// Q8_0 at revision bfcd2987 (309,855,456 bytes).
pub const MODEL: &str = "/models/embeddinggemma-2/embeddinggemma-2-Q8_0.gguf";

/// The sha256 of [`MODEL`].
// PIN(2026-10-14): `sha256sum` of the file, equal on the DGX Spark and on the
// copy it was taken from.
pub const MODEL_SHA256: &str = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135";

/// The projector of the tower and server sets: the same repository's bf16
/// `mmproj-embeddinggemma-2-BF16.gguf` (982,074,784 bytes, projector
/// `gemma4v`).
pub const MMPROJ_BF16: &str = "/models/embeddinggemma-2/mmproj-embeddinggemma-2-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`].
// PIN(2026-10-14): `sha256sum` of the file, equal on the DGX Spark and on the
// copy it was taken from.
pub const MMPROJ_BF16_SHA256: &str =
    "d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766";

/// The architecture the text and server sets name in their `# arch` line.
pub const ARCH: &str = "gemma-embedding2";

/// The text sets.
pub const TEXT_SET: &str = "ref_gemma_embedding2_text";
pub const TEXT_CPU_SET: &str = "ref_gemma_embedding2_text.cpu";

/// The image tower sets.
pub const TOWER_SET: &str = "ref_gemma_embedding2_tower";
pub const TOWER_CPU_SET: &str = "ref_gemma_embedding2_tower.cpu";

/// llama-server's vectors.
pub const SERVER_SET: &str = "ref_gemma_embedding2_server";

const RECIPE: &str = "just dump-ref-emb2";

/// The files every set is dumped from.
pub static PROFILE: Profile = Profile {
    model: MODEL,
    model_sha256: MODEL_SHA256,
    mmproj: Some((MMPROJ_BF16, MMPROJ_BF16_SHA256)),
};

/// [`MODEL`], as a text or server family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// The text cases' token rows, pooled vectors and layer taps on the card.
pub static TEXT: Family = family(
    "text-gemma-embedding2",
    &[TEXT_SET],
    Kind::Text,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The same, with no card in view.
pub static TEXT_CPU: Family = family(
    "text-gemma-embedding2-cpu",
    &[TEXT_CPU_SET],
    Kind::TextCpu,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The image cases' tower taps and rows on the card.
pub static TOWER: Family = family(
    "tower-gemma4v",
    &[TOWER_SET],
    Kind::Tower,
    &PROFILE,
    crate::clefvis::TOWER_ARCH,
    mmproj,
    RECIPE,
);
/// The same, with no card in view.
pub static TOWER_CPU: Family = family(
    "tower-gemma4v-cpu",
    &[TOWER_CPU_SET],
    Kind::TowerCpu,
    &PROFILE,
    crate::clefvis::TOWER_ARCH,
    mmproj,
    RECIPE,
);
/// llama-server's `/v1/embeddings` vectors.
pub static SERVER: Family = family(
    "server-gemma-embedding2",
    &[SERVER_SET],
    Kind::Server,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&TEXT, &TEXT_CPU, &TOWER, &TOWER_CPU, &SERVER];

#[cfg(test)]
mod tests;
