//! Every tensor name a `gemma-embedding2` file spells. The model-level names
//! are the common ones and the two of the per-layer input projection; a
//! layer's names are `blk.{l}.{stem}`.

pub use crate::arch::names::{output, output_norm, token_embd};

/// The projection of the scaled token rows to every layer's own input.
pub const PER_LAYER_MODEL_PROJ: &str = "per_layer_model_proj.weight";
/// The RMS gain on each layer's slice of that projection.
pub const PER_LAYER_PROJ_NORM: &str = "per_layer_proj_norm.weight";

/// The RMS gain before the attention.
pub const ATTN_NORM: &str = "attn_norm.weight";
/// The query projection.
pub const ATTN_Q: &str = "attn_q.weight";
/// The key projection.
pub const ATTN_K: &str = "attn_k.weight";
/// The value projection. Its rows are normed with no gain.
pub const ATTN_V: &str = "attn_v.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";
/// The per-head RMS gain on the queries.
pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
/// The per-head RMS gain on the keys.
pub const ATTN_K_NORM: &str = "attn_k_norm.weight";
/// The RMS gain on the attention's output, before the residual.
pub const POST_ATTENTION_NORM: &str = "post_attention_norm.weight";

/// The RMS gain before the feed-forward block.
pub const FFN_NORM: &str = "ffn_norm.weight";
/// The GELU GLU's gate.
pub const FFN_GATE: &str = "ffn_gate.weight";
/// The GELU GLU's up projection.
pub const FFN_UP: &str = "ffn_up.weight";
/// The down projection.
pub const FFN_DOWN: &str = "ffn_down.weight";
/// The RMS gain on the feed-forward output, before the residual.
pub const POST_FFW_NORM: &str = "post_ffw_norm.weight";

/// The per-layer input's gate, from the layer's residual row.
pub const INP_GATE: &str = "inp_gate.weight";
/// The per-layer input's projection back to the residual width.
pub const PROJ: &str = "proj.weight";
/// The RMS gain on that projection.
pub const POST_NORM: &str = "post_norm.weight";
/// The scalar the layer's output is multiplied by.
pub const LAYER_OUTPUT_SCALE: &str = "layer_output_scale.weight";
