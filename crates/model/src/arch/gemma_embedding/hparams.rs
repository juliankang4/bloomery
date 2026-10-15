//! EmbeddingGemma's header facts. The architecture fixes non-causal attention
//! and one global layer in every six; the header carries the window and widths.

use gguf::{Split, Value};

pub use crate::arch::gemma_encoder::Attn;
use crate::arch::gemma_encoder::{attn, encoder_keys};
use crate::arch::{
    finite_positive, metadata, n_vocab, optional_usize, per_layer_count, per_layer_items, positive,
};
use crate::placement::PlacementError;

use super::names;

/// The hyperparameters of one `gemma-embedding` file with its dense head.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// The trunk's layer count.
    pub n_layer: usize,
    /// Values per residual row.
    pub n_embd: usize,
    /// The training context length.
    pub n_ctx: usize,
    /// Entries in the token vocabulary.
    pub n_vocab: usize,
    /// The RMS epsilon of every block norm.
    pub rms_eps: f32,
    /// Query heads on every layer.
    pub n_head: usize,
    /// The GELU-gated feed-forward width.
    pub n_ff: usize,
    /// Key and value heads, one count per layer.
    pub kv: Vec<usize>,
    /// Every sixth layer is global, the others slide.
    pub sliding: Vec<bool>,
    /// The head and rope of a global layer.
    pub full: Attn,
    /// The same head and rope dimensions, with the sliding rope base.
    pub swa: Attn,
    /// The full symmetric window span.
    pub window: usize,
    /// The first dense map's output and the second's input width.
    pub n_dense: usize,
    /// Architecture facts and omitted keys, with their authority.
    pub defaults: Vec<String>,
}

impl Hparams {
    /// Read the header and hold every required tensor to its declared shape.
    /// The dense head, rope base, key/value lengths and KV-head count are required here although llama.cpp defaults them.
    pub fn read(split: &Split) -> Result<Self, PlacementError> {
        encoder_keys(split)?;
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let n_ff = positive(split, "feed_forward_length")?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        for (l, &heads) in kv.iter().enumerate() {
            crate::arch::check_kv(split, n_head, l, heads)?;
        }
        let mut defaults = Vec::new();
        let rope_dims = match optional_usize(split, "rope.dimension_count")? {
            Some(dims) => dims,
            None => {
                defaults.push(
                    "rope.dimension_count = attention.key_length (llama.cpp's load_hparams)".into(),
                );
                positive(split, "attention.key_length")?
            }
        };
        let full = attn(split, "", rope_dims)?;
        for (key, want) in [
            ("attention.key_length_swa", full.head),
            ("attention.value_length_swa", full.head),
            ("rope.dimension_count_swa", full.rope_dims),
        ] {
            if let Some(got) = optional_usize(split, key)?
                && got != want
            {
                return Err(metadata(split, key, format!("is {got}, not {want}")));
            }
        }
        if split.value(&split.arch_key("attention.causal")).is_none() {
            defaults.push(
                "attention.causal = false (gemma-embedding, llama.cpp's load_arch_hparams)".into(),
            );
        }
        let sliding: Vec<_> = (0..n_layer).map(|l| (l + 1) % 6 != 0).collect();
        let pattern = "attention.sliding_window_pattern";
        match split.value(&split.arch_key(pattern)) {
            Some(Value::Array(_)) => {
                for (l, value) in per_layer_items(split, pattern, n_layer)?.iter().enumerate() {
                    if value.as_bool() != Some(sliding[l]) {
                        return Err(metadata(
                            split,
                            pattern,
                            format!("differs from gemma-embedding's six-layer pattern at layer {l}"),
                        ));
                    }
                }
            }
            Some(_) => {
                if optional_usize(split, pattern)? != Some(6) {
                    return Err(metadata(
                        split,
                        pattern,
                        "differs from gemma-embedding's six-layer pattern (want 6)",
                    ));
                }
            }
            None => defaults.push("attention.sliding_window_pattern = five sliding, one global (gemma-embedding, llama.cpp's load_swa_pattern(ml, 6))".into()),
        }
        let base_key = "rope.freq_base_swa";
        let swa_base = if split.value(&split.arch_key(base_key)).is_some() {
            finite_positive(split, base_key)?
        } else {
            defaults.push("rope.freq_base_swa = 10000 (llama.cpp's llama_hparams)".into());
            10_000.0
        };
        let n_dense = positive(split, "dense_2_feat_out")?;
        for (key, want) in [
            ("dense_2_feat_in", n_embd),
            ("dense_3_feat_in", n_dense),
            ("dense_3_feat_out", n_embd),
        ] {
            let got = positive(split, key)?;
            if got != want {
                return Err(metadata(split, key, format!("is {got}, not {want}")));
            }
        }
        let hp = Self {
            n_layer,
            n_embd,
            n_head,
            n_ff,
            kv,
            sliding,
            full,
            swa: Attn {
                rope_base: swa_base,
                ..full
            },
            n_ctx: positive(split, "context_length")?,
            n_vocab: n_vocab(split)?,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            window: positive(split, "attention.sliding_window")?,
            n_dense,
            defaults,
        };
        crate::arch::tensors_agree(split, hp.want(split)?)?;
        Ok(hp)
    }

    /// Layer `l`'s head and rope.
    #[must_use]
    pub fn attn(&self, l: usize) -> Attn {
        if self.sliding[l] { self.swa } else { self.full }
    }

    fn want(&self, split: &Split) -> Result<Vec<(String, Vec<u64>)>, PlacementError> {
        let e = self.n_embd as u64;
        let dense = self.n_dense as u64;
        let mut out = vec![
            (names::token_embd(), vec![e, self.n_vocab as u64]),
            (names::output_norm(), vec![e]),
            (names::DENSE_2.to_string(), vec![e, dense]),
            (names::DENSE_3.to_string(), vec![dense, e]),
        ];
        if split.find(&names::output()).is_some() {
            out.push((names::output(), vec![e, self.n_vocab as u64]));
        }
        for l in 0..self.n_layer {
            out.extend(
                crate::arch::gemma_encoder::layer_tensors(
                    split,
                    self.n_embd,
                    self.n_ff,
                    self.n_head,
                    self.kv[l],
                    self.attn(l),
                    "attention.key_length",
                )?
                .into_iter()
                .map(|(stem, dims)| (crate::arch::names::blk(l, stem), dims)),
            );
        }
        Ok(out)
    }
}
