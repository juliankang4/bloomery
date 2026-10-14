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
    meta_bool, meta_f32, metadata, mul, n_vocab, optional_usize, per_layer_count, per_layer_items,
    positive,
};
use crate::placement::PlacementError;

/// `pooling_type`'s mean (llama.h `LLAMA_POOLING_TYPE_MEAN`).
const POOLING_MEAN: usize = 1;

/// The head and rope of one kind of layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attn {
    /// `attention.key_length{,_swa}`, which `value_length{,_swa}` equals.
    pub head: usize,
    /// `rope.dimension_count{,_swa}`: the values rotated, NeoX halves.
    pub rope_dims: usize,
    /// `rope.freq_base{,_swa}`.
    pub rope_base: f32,
}

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
    /// `attention.head_count_kv`, per layer.
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
        let rms_eps = meta_f32(split, "attention.layer_norm_rms_epsilon")?;
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(metadata(
                split,
                "attention.layer_norm_rms_epsilon",
                format!("is {rms_eps}, not a finite positive epsilon"),
            ));
        }
        encoder_keys(split)?;
        let full = attn(split, "")?;
        let swa = attn(split, "_swa")?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        for (l, &heads) in kv.iter().enumerate() {
            if heads == 0 || !n_head.is_multiple_of(heads) {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!("is {heads} at layer {l} for {n_head} query heads"),
                ));
            }
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
        let ff = w(self.n_ff);
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
            let q = w(mul(split, key, self.n_head, a.head)?);
            let k = w(mul(split, key, self.kv[l], a.head)?);
            let d = w(a.head);
            out.extend(
                [
                    (names::ATTN_NORM, vec![e]),
                    (names::ATTN_Q, vec![e, q]),
                    (names::ATTN_K, vec![e, k]),
                    (names::ATTN_V, vec![e, k]),
                    (names::ATTN_OUTPUT, vec![q, e]),
                    (names::ATTN_Q_NORM, vec![d]),
                    (names::ATTN_K_NORM, vec![d]),
                    (names::POST_ATTENTION_NORM, vec![e]),
                    (names::FFN_NORM, vec![e]),
                    (names::FFN_GATE, vec![e, ff]),
                    (names::FFN_UP, vec![e, ff]),
                    (names::FFN_DOWN, vec![ff, e]),
                    (names::POST_FFW_NORM, vec![e]),
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

/// The head and rope keys of one kind of layer, `suffix` `""` or `"_swa"`.
/// The value head is the key head (llama.cpp throws otherwise), and the
/// rope turns an even count of values within it.
fn attn(split: &Split, suffix: &str) -> Result<Attn, PlacementError> {
    let key = format!("attention.key_length{suffix}");
    let value = format!("attention.value_length{suffix}");
    let dims = format!("rope.dimension_count{suffix}");
    let base = format!("rope.freq_base{suffix}");
    let head = positive(split, &key)?;
    let v = positive(split, &value)?;
    if v != head {
        return Err(metadata(split, &value, format!("is {v}, not {key} {head}")));
    }
    let rope_dims = positive(split, &dims)?;
    if !rope_dims.is_multiple_of(2) || rope_dims > head {
        return Err(metadata(
            split,
            &dims,
            format!("is {rope_dims}, for a head of {head}"),
        ));
    }
    let rope_base = meta_f32(split, &base)?;
    if !rope_base.is_finite() || rope_base <= 0.0 {
        return Err(metadata(
            split,
            &base,
            format!("is {rope_base}, not a finite positive base"),
        ));
    }
    Ok(Attn {
        head,
        rope_dims,
        rope_base,
    })
}

/// `pooling_type` is the mean, `attention.causal` is absent or false and
/// `attention.shared_kv_layers` is absent or 0: the model this reader
/// describes. Any other value is refused by name.
fn encoder_keys(split: &Split) -> Result<(), PlacementError> {
    let pooling = positive(split, "pooling_type")?;
    if pooling != POOLING_MEAN {
        return Err(metadata(
            split,
            "pooling_type",
            format!("is {pooling}; this reader pools by the mean ({POOLING_MEAN})"),
        ));
    }
    let causal = "attention.causal";
    if split.value(&split.arch_key(causal)).is_some() && meta_bool(split, causal)? {
        return Err(metadata(
            split,
            causal,
            "is true; the encoder attends both ways",
        ));
    }
    let shared = "attention.shared_kv_layers";
    if let Some(n) = optional_usize(split, shared)?
        && n != 0
    {
        return Err(metadata(
            split,
            shared,
            format!("is {n}; every layer here projects its own keys and values"),
        ));
    }
    Ok(())
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
