//! Geometry and graph facts resolved by the architecture readers for the shared pass.

use gguf::Split;
use model::arch::gemma_embedding2::hparams::Attn;
use model::placement::PlacementError;

#[derive(Clone, Copy)]
pub(crate) struct DenseHead {
    pub width: usize,
    pub up: &'static str,
    pub down: &'static str,
}

#[derive(Clone)]
pub(crate) struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub rms_eps: f32,
    pub n_head: usize,
    pub n_ff: usize,
    pub kv: Vec<usize>,
    pub sliding: Vec<bool>,
    pub full: Attn,
    pub swa: Attn,
    pub per_layer: usize,
    pub n_out: usize,
    pub dense: Option<DenseHead>,
}

impl Hparams {
    pub fn read(split: &Split) -> Result<Self, PlacementError> {
        match split.architecture() {
            Some(model::arch::GEMMA_EMBEDDING) => {
                let hp = model::arch::gemma_embedding::hparams::Hparams::read(split)?;
                Ok(Self {
                    n_layer: hp.n_layer,
                    n_embd: hp.n_embd,
                    n_ctx: hp.n_ctx,
                    n_vocab: hp.n_vocab,
                    rms_eps: hp.rms_eps,
                    n_head: hp.n_head,
                    n_ff: hp.n_ff,
                    kv: hp.kv,
                    sliding: hp.sliding,
                    full: hp.full,
                    swa: hp.swa,
                    per_layer: 0,
                    n_out: hp.n_embd,
                    dense: Some(DenseHead {
                        width: hp.n_dense,
                        up: model::arch::gemma_embedding::names::DENSE_2,
                        down: model::arch::gemma_embedding::names::DENSE_3,
                    }),
                })
            }
            Some(model::arch::GEMMA_EMBEDDING2) => {
                let hp = model::arch::gemma_embedding2::hparams::Hparams::read(split)?;
                Ok(Self {
                    n_layer: hp.n_layer,
                    n_embd: hp.n_embd,
                    n_ctx: hp.n_ctx,
                    n_vocab: hp.n_vocab,
                    rms_eps: hp.rms_eps,
                    n_head: hp.n_head,
                    n_ff: hp.n_ff,
                    kv: hp.kv,
                    sliding: hp.sliding,
                    full: hp.full,
                    swa: hp.swa,
                    per_layer: hp.per_layer,
                    n_out: hp.n_out,
                    dense: None,
                })
            }
            arch => Err(PlacementError::Metadata {
                key: gguf::GENERAL_ARCHITECTURE.into(),
                detail: format!("{arch:?}: expected an EmbeddingGemma encoder"),
            }),
        }
    }

    pub fn attn(&self, l: usize) -> Attn {
        if self.sliding[l] { self.swa } else { self.full }
    }
}
