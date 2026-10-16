//! EmbeddingGemma v1 text rows, pooled vectors and llama-server vectors.
use crate::RefError;
use crate::emb2::{Kind, Profile, family};
use crate::family::{Build, Family, Identity};

/// The Q8_0 file from ggml-org/embeddinggemma-300M-GGUF, revision 0f741b5a.
pub const MODEL: &str = "/models/embeddinggemma-300M/embeddinggemma-300M-Q8_0.gguf";
/// The file's SHA256, checked against the Hub LFS digest.
pub const MODEL_SHA256: &str = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63";
/// The architecture in the model and its sets.
pub const ARCH: &str = "gemma-embedding";
/// The text-only model has no projector.
pub static PROFILE: Profile = Profile {
    model: MODEL,
    model_sha256: MODEL_SHA256,
    mmproj: None,
};
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}
/// The vocabulary oracle's executable and library, built at mainline 6868a52cb.
// PIN(2026-10-16): the mainline tokenizer used to dump the text-only model's ids.
pub const TOKENIZER_BUILD: &str = "llama-tokenize be4957f2c2f96704054e0557c48b9b09 \
                                 libllama.so 10eff38604798775e975958c13b3b385";

/// The SentencePiece ids, including BOS and EOS in both parse modes.
pub static TOKENIZER: Family = Family {
    name: "tokenizer-llama",
    sets: &["ref-tokenizer-llama"],
    resolve: None,
    recipe: "just dump-ref-tokenizer llama",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(TOKENIZER_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

const RECIPE: &str = "just dump-ref-emb1";
/// Token rows, pooled vectors and layer taps on the card.
pub static TEXT: Family = family(
    "text-gemma-embedding",
    &["ref_gemma_embedding_text"],
    Kind::Text,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The text set with no card in view.
pub static TEXT_CPU: Family = family(
    "text-gemma-embedding-cpu",
    &["ref_gemma_embedding_text.cpu"],
    Kind::TextCpu,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// llama-server's normalized and raw vectors.
pub static SERVER: Family = family(
    "server-gemma-embedding",
    &["ref_gemma_embedding_server"],
    Kind::Server,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The architecture's families in reader order.
pub static FAMILIES: &[&Family] = &[&TEXT, &TEXT_CPU, &SERVER];
#[cfg(test)]
mod tests;
