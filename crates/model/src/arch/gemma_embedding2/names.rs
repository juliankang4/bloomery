//! Every tensor name a `gemma-embedding2` file spells. The model-level names
//! are the common ones and the two of the per-layer input projection; a
//! layer's names are `blk.{l}.{stem}`.

pub use crate::arch::names::{output, output_norm, token_embd};

/// The projection of the scaled token rows to every layer's own input.
pub const PER_LAYER_MODEL_PROJ: &str = "per_layer_model_proj.weight";
/// The RMS gain on each layer's slice of that projection.
pub const PER_LAYER_PROJ_NORM: &str = "per_layer_proj_norm.weight";

pub use crate::arch::gemma_encoder::names::*;

/// The per-layer input's gate, from the layer's residual row.
pub const INP_GATE: &str = "inp_gate.weight";
/// The per-layer input's projection back to the residual width.
pub const PROJ: &str = "proj.weight";
/// The RMS gain on that projection.
pub const POST_NORM: &str = "post_norm.weight";
/// The scalar the layer's output is multiplied by.
pub const LAYER_OUTPUT_SCALE: &str = "layer_output_scale.weight";
