//! The role of every tensor in a `gemma-embedding2` file, by exact name.
//! Every layer carries the same stems, all of them required (llama.cpp
//! creates each with no `TENSOR_NOT_REQUIRED`). The per-layer input's
//! weights are dense weights every token reads in full. A name the table
//! does not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_V, FFN_DOWN, FFN_GATE,
    FFN_NORM, FFN_UP, INP_GATE, LAYER_OUTPUT_SCALE, PER_LAYER_MODEL_PROJ, PER_LAYER_PROJ_NORM,
    POST_ATTENTION_NORM, POST_FFW_NORM, POST_NORM, PROJ,
};
use crate::arch::classify::{Counts, classify_with};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A layer's stems.
pub(super) const LAYER: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(ATTN_Q, Role::Attention),
    req(ATTN_K, Role::Attention),
    req(ATTN_V, Role::Attention),
    req(ATTN_OUTPUT, Role::Attention),
    req(ATTN_Q_NORM, Role::Attention),
    req(ATTN_K_NORM, Role::Attention),
    req(POST_ATTENTION_NORM, Role::Attention),
    req(FFN_NORM, Role::FfnNorm),
    req(FFN_GATE, Role::DenseFfn),
    req(FFN_UP, Role::DenseFfn),
    req(FFN_DOWN, Role::DenseFfn),
    req(POST_FFW_NORM, Role::DenseFfn),
    req(INP_GATE, Role::EngramDense),
    req(PROJ, Role::EngramDense),
    req(POST_NORM, Role::EngramDense),
    req(LAYER_OUTPUT_SCALE, Role::DenseFfn),
];

/// A tensor's role and layer by its name; `None` when no table holds it.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        PER_LAYER_MODEL_PROJ | PER_LAYER_PROJ_NORM => return Some((Role::EngramDense, None)),
        "output_norm.weight" | "output.weight" => return Some((Role::Head, None)),
        _ => {}
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < hp.n_layer)?;
    LAYER
        .iter()
        .find(|s| s.name == stem)
        .map(|s| (s.role, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role(name, hp),
        Counts {
            layers: hp.n_layer,
            experts: 0,
            experts_used: 0,
        },
    )
}
