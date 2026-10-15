//! Every `gemma-embedding2.*` key the description reads, and each layer's
//! kind from `attention.sliding_window_pattern`: a sliding layer takes the
//! `_swa` head and rope keys, a full layer the plain ones (llama.cpp's
//! `n_embd_head_k(il)`, `n_rot(il)`). A key llama.cpp would default is
//! required here, so the description is the file's; the two that only
//! confirm the model's kind, `attention.causal` and
//! `attention.shared_kv_layers`, may be absent. The model pools by the mean
//! and attends both ways; a file that says otherwise is refused.

use gguf::Split;

use super::names;
use crate::arch::names::blk;
use crate::arch::{
    finite_positive, metadata, mul, n_vocab, per_layer_count, per_layer_items, positive,
};
use crate::placement::PlacementError;

pub use crate::arch::gemma_encoder::Attn;
use crate::arch::gemma_encoder::encoder_keys;

/// The hyperparameters of one `gemma-embedding2` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `context_length`.
    pub n_ctx: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `feed_forward_length`.
    pub n_ff: usize,
    /// `attention.head_count_kv`, shared or per layer, resolved per layer.
    pub kv: Vec<usize>,
    /// `attention.sliding_window_pattern`: whether each layer slides.
    pub sliding: Vec<bool>,
    /// The full layers' head and rope.
    pub full: Attn,
    /// The sliding layers' head and rope.
    pub swa: Attn,
    /// `attention.sliding_window`: the window's whole span, both sides.
    pub window: usize,
    /// `embedding_length_per_layer_input`.
    pub per_layer: usize,
    /// `embedding_length_out`: the output projection's width.
    pub n_out: usize,
}

impl Hparams {
    /// Every value from `split`'s headers. A missing key, a key of another
    /// value than the model's, a per-layer array of the wrong length, or a
    /// tensor missing or of other dims than the keys give is an error naming
    /// the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let n_ff = positive(split, "feed_forward_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        encoder_keys(split)?;
        let full = attn(split, "")?;
        let swa = attn(split, "_swa")?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        for (l, &heads) in kv.iter().enumerate() {
            crate::arch::check_kv(split, n_head, l, heads)?;
        }
        let sliding = sliding(split, n_layer)?;
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab: n_vocab(split)?,
            rms_eps,
            n_head,
            n_ff,
            kv,
            sliding,
            full,
            swa,
            window: positive(split, "attention.sliding_window")?,
            per_layer: positive(split, "embedding_length_per_layer_input")?,
            n_out: positive(split, "embedding_length_out")?,
        };
        crate::arch::tensors_agree(split, hp.want(split)?)?;
        Ok(hp)
    }

    /// Layer `l`'s head and rope.
    #[must_use]
    pub fn attn(&self, l: usize) -> Attn {
        if self.sliding[l] { self.swa } else { self.full }
    }

    /// Every tensor the model carries with the dims the keys give it, ggml
    /// order. A product that does not fit is an error naming the key.
    fn want(&self, split: &Split) -> Result<Vec<(String, Vec<u64>)>, PlacementError> {
        let w = |n: usize| n as u64;
        let e = w(self.n_embd);
        let ple = w(self.per_layer);
        let mut out = vec![
            (names::token_embd(), vec![e, w(self.n_vocab)]),
            (
                names::PER_LAYER_MODEL_PROJ.to_string(),
                vec![
                    e,
                    w(mul(
                        split,
                        "embedding_length_per_layer_input",
                        self.per_layer,
                        self.n_layer,
                    )?),
                ],
            ),
            (names::PER_LAYER_PROJ_NORM.to_string(), vec![ple]),
            (names::output_norm(), vec![e]),
            (names::output(), vec![e, w(self.n_out)]),
        ];
        for l in 0..self.n_layer {
            let a = self.attn(l);
            let key = if self.sliding[l] {
                "attention.key_length_swa"
            } else {
                "attention.key_length"
            };
            out.extend(
                crate::arch::gemma_encoder::layer_tensors(
                    split,
                    self.n_embd,
                    self.n_ff,
                    self.n_head,
                    self.kv[l],
                    a,
                    key,
                )?
                .into_iter()
                .map(|(stem, dims)| (blk(l, stem), dims)),
            );
            out.extend(
                [
                    (names::INP_GATE, vec![e, ple]),
                    (names::PROJ, vec![ple, e]),
                    (names::POST_NORM, vec![e]),
                    (names::LAYER_OUTPUT_SCALE, vec![1]),
                ]
                .map(|(stem, dims)| (blk(l, stem), dims)),
            );
        }
        Ok(out)
    }
}

/// Each layer kind declares its rope width in the header.
fn attn(split: &Split, suffix: &str) -> Result<Attn, PlacementError> {
    let dims = positive(split, &format!("rope.dimension_count{suffix}"))?;
    crate::arch::gemma_encoder::attn(split, suffix, dims)
}

/// `attention.sliding_window_pattern` as one bool per layer. A scalar, a
/// length other than `n_layer` or a value that is no bool is refused by
/// name: the file's own form is the per-layer array.
fn sliding(split: &Split, n_layer: usize) -> Result<Vec<bool>, PlacementError> {
    let key = "attention.sliding_window_pattern";
    per_layer_items(split, key, n_layer)?
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_bool()
                .ok_or_else(|| metadata(split, key, format!("has no bool at layer {l}")))
        })
        .collect()
}
