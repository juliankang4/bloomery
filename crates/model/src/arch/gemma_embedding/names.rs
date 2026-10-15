//! EmbeddingGemma's model-level tensor names and pooled dense head.

pub use crate::arch::names::{output, output_norm, token_embd};

/// The first linear map after pooling, from the residual width.
pub const DENSE_2: &str = "dense_2.weight";
/// The second linear map, back to the embedding width.
pub const DENSE_3: &str = "dense_3.weight";
