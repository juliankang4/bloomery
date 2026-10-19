//! GLM-5.3-Flash's Hugging Face checkpoint in the names and layouts the glm5next program reads,
//! as llama.cpp's converter (`conversion/glm.py`, `Glm5NextModel`) writes them into a GGUF.
//!
//! A checkpoint tensor maps to one glm5next tensor ([`names`]) and a [`Layout`]: as stored; the
//! routed experts' tensors of one projection stacked by expert id; `kv_b_proj` split per head
//! into the key half, transposed, and the value half; the KDA conv taps as `[kernel, 1,
//! channels]`; and `A_log` as `-exp`. `dt_bias` becomes the `ssm_dt` bias, and both
//! `model.language_model.` and `model.` prefixes are read. The next-token layers (index
//! `num_hidden_layers` and up) and the vision tower are left out; any other name is refused.
//!
//! Shapes are logical element counts in ggml's order, the fastest axis first; a packed
//! weight's caller gives its unpacked shape.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::names::{self, Sub};

/// Why the checkpoint was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("glm5next hf: config.json: {0}")]
    Config(String),
    #[error("glm5next hf: {name}: {what}")]
    Tensor { name: String, what: String },
}

fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// The config values the map reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Trunk layers; the next-token layers follow.
    pub layers: usize,
    pub heads: usize,
    pub nope: usize,
    pub v: usize,
    pub experts: usize,
}

impl Geometry {
    /// `config.json`'s values, from `text_config` when the checkpoint nests them there.
    pub fn of(config: &Value) -> Result<Geometry, Error> {
        let c = config.get("text_config").unwrap_or(config);
        let n = |key: &str| {
            c.get(key)
                .and_then(Value::as_u64)
                .filter(|&n| n > 0)
                .map(|n| n as usize)
                .ok_or_else(|| Error::Config(format!("{key} is not a count")))
        };
        Ok(Geometry {
            layers: n("num_hidden_layers")?,
            heads: n("num_attention_heads")?,
            nope: n("qk_nope_head_dim")?,
            v: n("v_head_dim")?,
            experts: n("n_routed_experts")?,
        })
    }
}

/// How a glm5next tensor is made from the checkpoint's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The tensor's elements in stored order.
    Same,
    /// The expert tensors named, in expert id order, one after another.
    Stack(Vec<String>),
    /// `kv_b_proj` `[heads · (nope + v), rank]`: each head's first `nope` rows, transposed to
    /// `[rank, nope]`.
    KeyHalf { heads: usize, nope: usize, v: usize },
    /// Each head's last `v` rows, `[v, rank]`.
    ValueHalf { heads: usize, nope: usize, v: usize },
    /// `-exp(x)` of the first `heads` values, as f32, computed in f64 and rounded once.
    NegExp { heads: usize },
}

/// One glm5next tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The checkpoint tensor it is made from (a stack's first expert).
    pub source: String,
    pub layout: Layout,
    /// Logical element counts in ggml's order.
    pub dims: Vec<usize>,
}

/// The glm5next tensors, by name, of the checkpoint tensors `tensors` (name and logical shape,
/// row-major) under `g`. A layer's routed experts not all there or not matrices, and a layer
/// holding part of a set of tensors that come together (`parts`), are refused by name.
pub fn map<'a>(
    g: Geometry,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
) -> Result<BTreeMap<String, Mapped>, Error> {
    let mut out = BTreeMap::new();
    // A layer's experts: (layer, projection) to each expert's tensor name and shape.
    type ById<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;
    let mut experts: BTreeMap<(usize, &str), ById> = BTreeMap::new();
    let mut put = |name: String, m: Mapped| match out.insert(name, m) {
        Some(prev) => Err(refuse(
            &prev.source,
            "maps to a glm5next name another tensor has",
        )),
        None => Ok(()),
    };
    for (name, shape) in tensors {
        let short = name
            .strip_prefix("model.language_model.")
            .or_else(|| name.strip_prefix("model."))
            .unwrap_or(name);
        let same = |dims: Vec<usize>| Mapped {
            source: name.to_string(),
            layout: Layout::Same,
            dims,
        };
        let reversed = || same(shape.iter().rev().copied().collect());
        match short {
            "embed_tokens.weight" => put(names::token_embd(), reversed())?,
            "norm.weight" => put(names::output_norm(), reversed())?,
            "lm_head.weight" => put(names::output(), reversed())?,
            _ if short.starts_with("visual.") || name.starts_with("model.visual.") => {}
            _ => {
                let (l, rest) = short
                    .strip_prefix("layers.")
                    .and_then(|r| r.split_once('.'))
                    .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                    .ok_or_else(|| refuse(name, "not a GLM-5.3-Flash tensor"))?;
                if l >= g.layers {
                    continue;
                }
                if let Some(e) = rest.strip_prefix("mlp.experts.") {
                    let (id, proj) = e
                        .split_once('.')
                        .and_then(|(id, p)| {
                            Some((id.parse::<usize>().ok()?, p.strip_suffix(".weight")?))
                        })
                        .filter(|(_, p)| ["gate_proj", "up_proj", "down_proj"].contains(p))
                        .ok_or_else(|| refuse(name, "not a routed expert projection"))?;
                    if let Some((other, _)) = experts
                        .entry((l, proj))
                        .or_default()
                        .insert(id, (name, shape))
                    {
                        return Err(refuse(
                            name,
                            format!("layer {l} {proj} expert {id} is also supplied by {other}"),
                        ));
                    }
                    continue;
                }
                for (glm, m) in layer(g, l, rest, name, shape)? {
                    put(glm, m)?;
                }
            }
        }
    }
    for ((l, proj), by_id) in experts {
        let (first, shape) = by_id[by_id.keys().next().expect("an entry holds an expert")];
        if by_id.len() != g.experts || by_id.keys().next_back() != Some(&(g.experts - 1)) {
            return Err(refuse(
                first,
                format!(
                    "layer {l} {proj} has {} of {} experts",
                    by_id.len(),
                    g.experts
                ),
            ));
        }
        if let Some((name, _)) = by_id.values().find(|(_, s)| *s != shape) {
            return Err(refuse(
                name,
                format!("shape differs from {first}'s {shape:?}"),
            ));
        }
        if shape.len() != 2 {
            return Err(refuse(
                first,
                format!("expert projection {shape:?} is not 2-D"),
            ));
        }
        let glm = match proj {
            "gate_proj" => names::ffn_gate_exps(l),
            "up_proj" => names::ffn_up_exps(l),
            _ => names::ffn_down_exps(l),
        };
        let mut dims: Vec<usize> = shape.iter().rev().copied().collect();
        dims.push(g.experts);
        let sources = by_id.values().map(|(n, _)| n.to_string()).collect();
        put(
            glm,
            Mapped {
                source: first.to_string(),
                layout: Layout::Stack(sources),
                dims,
            },
        )?;
    }
    let layers: BTreeSet<usize> = out
        .keys()
        .filter_map(|n| n.strip_prefix("blk.")?.split_once('.')?.0.parse().ok())
        .collect();
    for l in layers {
        for set in parts(l) {
            if let Some(have) = set.iter().find(|n| out.contains_key(*n))
                && let Some(missing) = set.iter().find(|n| !out.contains_key(*n))
            {
                return Err(refuse(
                    &out[have].source,
                    format!("layer {l} has {have} but not {missing}"),
                ));
            }
        }
    }
    Ok(out)
}

/// The required glm5next tensors of layer `l` that come together: a layer holding one of a set
/// holds all of it.
fn parts(l: usize) -> [Vec<String>; 8] {
    use super::roles;
    let group = |stems: &[roles::Stem], keep: fn(&str) -> bool| {
        roles::required(stems)
            .filter(|s| keep(s))
            .map(|s| format!("blk.{l}.{s}"))
            .collect()
    };
    let mixer =
        |s: &str| s != "attn_norm.weight" && s != "attn_output.weight" && !s.starts_with("indexer");
    [
        group(roles::KDA, mixer),
        group(roles::LATENT, mixer),
        group(roles::LATENT, |s| s.starts_with("indexer")),
        group(roles::DENSE, |s| s != "ffn_norm.weight"),
        group(roles::SHARED, |_| true),
        group(roles::MOE, |s| s != "ffn_norm.weight"),
        group(roles::HC, |s| s.starts_with("hc_attn_")),
        group(roles::HC, |s| s.starts_with("hc_ffn_")),
    ]
}

/// The glm5next tensors of `rest`, a tensor of layer `l` other than a routed expert.
fn layer(
    g: Geometry,
    l: usize,
    rest: &str,
    name: &str,
    shape: &[usize],
) -> Result<Vec<(String, Mapped)>, Error> {
    let mapped = |layout: Layout, dims: Vec<usize>| Mapped {
        source: name.to_string(),
        layout,
        dims,
    };
    let reversed = || mapped(Layout::Same, shape.iter().rev().copied().collect());
    let one = |glm: String| Ok(vec![(glm, reversed())]);
    match rest {
        "input_layernorm.weight" => one(names::attn_norm(l)),
        "post_attention_layernorm.weight" => one(names::ffn_norm(l)),
        "self_attn.o_proj.weight" => one(names::attn_output(l)),
        "self_attn.q_proj.weight" => one(names::attn_q(l)),
        "self_attn.k_proj.weight" => one(names::attn_k(l)),
        "self_attn.v_proj.weight" => one(names::attn_v(l)),
        "self_attn.f_a_proj.weight" => one(names::ssm_f_a(l)),
        "self_attn.f_b_proj.weight" => one(names::ssm_f_b(l)),
        "self_attn.g_a_proj.weight" => one(names::ssm_g_a(l)),
        "self_attn.g_b_proj.weight" => one(names::ssm_g_b(l)),
        "self_attn.b_proj.weight" => one(names::ssm_beta(l)),
        "self_attn.o_norm.weight" => one(names::ssm_norm(l)),
        "self_attn.dt_bias" => one(names::ssm_dt_bias(l)),
        "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight" => {
            let (channels, kernel) = match *shape {
                [c, 1, k] | [c, k] => (c, k),
                _ => {
                    return Err(refuse(
                        name,
                        format!("conv taps {shape:?} are not [channels, 1, kernel]"),
                    ));
                }
            };
            let part = rest.as_bytes()["self_attn.".len()] as char;
            Ok(vec![(
                names::ssm_conv1d(l, part),
                mapped(Layout::Same, vec![kernel, 1, channels]),
            )])
        }
        "self_attn.A_log" => {
            let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            if count.is_none_or(|n| n < g.heads) {
                return Err(refuse(
                    name,
                    format!("{shape:?} holds fewer than {} heads", g.heads),
                ));
            }
            Ok(vec![(
                names::ssm_a(l),
                mapped(Layout::NegExp { heads: g.heads }, vec![g.heads]),
            )])
        }
        "self_attn.q_a_proj.weight" => one(names::attn_q_a(l)),
        "self_attn.q_a_layernorm.weight" => one(names::attn_q_a_norm(l)),
        "self_attn.q_b_proj.weight" => one(names::attn_q_b(l)),
        "self_attn.kv_a_proj_with_mqa.weight" => one(names::attn_kv_a_mqa(l)),
        "self_attn.kv_a_layernorm.weight" => one(names::attn_kv_a_norm(l)),
        "self_attn.kv_b_proj.weight" => {
            let (heads, nope, v) = (g.heads, g.nope, g.v);
            let &[rows, rank] = shape else {
                return Err(refuse(name, format!("{shape:?} is not 2-D")));
            };
            if nope.checked_add(v).and_then(|w| w.checked_mul(heads)) != Some(rows) {
                return Err(refuse(
                    name,
                    format!("{rows} rows, not {heads} heads of {nope} + {v}"),
                ));
            }
            Ok(vec![
                (
                    names::attn_k_b(l),
                    mapped(Layout::KeyHalf { heads, nope, v }, vec![nope, rank, heads]),
                ),
                (
                    names::attn_v_b(l),
                    mapped(Layout::ValueHalf { heads, nope, v }, vec![rank, v, heads]),
                ),
            ])
        }
        "self_attn.indexer.wq_b.weight" => one(names::indexer_attn_q_b(l)),
        "self_attn.indexer.wk.weight" => one(names::indexer_attn_k(l)),
        "self_attn.indexer.k_norm.weight" => one(names::indexer_k_norm(l)),
        "self_attn.indexer.k_norm.bias" => one(names::indexer_k_norm_bias(l)),
        "self_attn.indexer.weights_proj.weight" => one(names::indexer_proj(l)),
        "self_attn.indexer.index_kpool_compress_ape" => one(names::indexer_compressor_ape(l)),
        "self_attn.indexer.index_kpool_compress_gate" => one(names::indexer_compressor_gate(l)),
        "mlp.gate.weight" => one(names::ffn_gate_inp(l)),
        "mlp.gate.e_score_correction_bias" => one(names::exp_probs_b(l)),
        "mlp.gate_proj.weight" => one(names::ffn_gate(l)),
        "mlp.up_proj.weight" => one(names::ffn_up(l)),
        "mlp.down_proj.weight" => one(names::ffn_down(l)),
        "mlp.shared_experts.gate_proj.weight" => one(names::ffn_gate_shexp(l)),
        "mlp.shared_experts.up_proj.weight" => one(names::ffn_up_shexp(l)),
        "mlp.shared_experts.down_proj.weight" => one(names::ffn_down_shexp(l)),
        "hc_attn_fn" => one(names::hc_fn(l, Sub::Attn)),
        "hc_attn_base" => one(names::hc_base(l, Sub::Attn)),
        "hc_attn_scale" => one(names::hc_scale(l, Sub::Attn)),
        "hc_ffn_fn" => one(names::hc_fn(l, Sub::Ffn)),
        "hc_ffn_base" => one(names::hc_base(l, Sub::Ffn)),
        "hc_ffn_scale" => one(names::hc_scale(l, Sub::Ffn)),
        _ => Err(refuse(name, "not a GLM-5.3-Flash tensor")),
    }
}

#[cfg(test)]
impl Layout {
    /// The glm5next tensor's elements from the source's `src`, `size` bytes an element, row-major
    /// `shape`; for [`Layout::NegExp`], `src` is f32. A stack's experts are joined by the caller.
    pub fn apply(&self, src: &[u8], shape: &[usize], size: usize) -> Vec<u8> {
        match *self {
            Layout::Same | Layout::Stack(_) => src.to_vec(),
            Layout::KeyHalf { heads, nope, v } => {
                let rank = shape[1];
                let mut out = Vec::with_capacity(heads * rank * nope * size);
                for h in 0..heads {
                    let head = &src[h * (nope + v) * rank * size..];
                    for c in 0..rank {
                        for r in 0..nope {
                            out.extend_from_slice(&head[(r * rank + c) * size..][..size]);
                        }
                    }
                }
                out
            }
            Layout::ValueHalf { heads, nope, v } => {
                let row = shape[1] * size;
                (0..heads)
                    .flat_map(|h| &src[(h * (nope + v) + nope) * row..][..v * row])
                    .copied()
                    .collect()
            }
            Layout::NegExp { heads } => src
                .as_chunks::<4>()
                .0
                .iter()
                .take(heads)
                .flat_map(|&b| {
                    let x = f32::from_le_bytes(b);
                    ((-f64::from(x).exp()) as f32).to_le_bytes()
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use hf::quant_config::{Config, Format, plan};
    use hf::safetensors::Dtype;

    use super::*;

    const GLM: Geometry = Geometry {
        layers: 45,
        heads: 64,
        nope: 256,
        v: 256,
        experts: 288,
    };

    fn gunzip(path: &str) -> Value {
        let gz = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut text = Vec::new();
        flate2::read::GzDecoder::new(gz.as_slice())
            .read_to_end(&mut text)
            .unwrap();
        serde_json::from_slice(&text).unwrap()
    }

    /// The checkpoint's placed tensors with their logical shapes: an NVFP4 weight holds two values
    /// a byte.
    fn logical(
        config: &Config,
        table: &[(String, Dtype, Vec<usize>)],
    ) -> Vec<(String, Vec<usize>)> {
        let p = plan(
            config,
            table.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        );
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        table
            .iter()
            .filter_map(|(n, _, s)| {
                let mut s = s.clone();
                match p.placed.get(n)?.format {
                    Format::Nvfp4 => *s.last_mut()? *= 2,
                    Format::Bf16 | Format::F32 => {}
                    f => panic!("{n}: {f} in GLM NVFP4"),
                }
                Some((n.clone(), s))
            })
            .collect()
    }

    fn table(header: &Value) -> Vec<(String, Dtype, Vec<usize>)> {
        header
            .as_object()
            .unwrap()
            .iter()
            .filter(|(n, _)| *n != "__metadata__")
            .map(|(n, e)| {
                let shape = e["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                let dtype = match e["dtype"].as_str().unwrap() {
                    "BF16" => Dtype::Bf16,
                    "F32" => Dtype::F32,
                    "U8" => Dtype::U8,
                    "F8_E4M3" => Dtype::F8E4M3,
                    d => panic!("{n}: {d}"),
                };
                (n.clone(), dtype, shape)
            })
            .collect()
    }

    /// The unsloth GGUF's names and ggml dims from a `tensor`, `dimensions` table.
    fn gguf_table(text: &str, keep: impl Fn(&str) -> bool) -> BTreeMap<String, Vec<usize>> {
        let cols: Vec<&str> = text.lines().next().unwrap().split('\t').collect();
        let (t, d) = (
            cols.iter().position(|c| *c == "tensor").unwrap(),
            cols.iter().position(|c| *c == "dimensions").unwrap(),
        );
        text.lines()
            .skip(1)
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|r| keep(r[t]))
            .map(|r| {
                (
                    r[t].to_string(),
                    r[d].split('x').map(|n| n.parse().unwrap()).collect(),
                )
            })
            .collect()
    }

    /// Layers 0 (KDA, dense) and 3 (latent attention, 288 experts) and the unlayered tensors of
    /// nvidia/GLM-5.3-Flash-NVFP4 map to exactly the names and ggml dims of unsloth's
    /// GLM-5.3-Flash UD-Q4_K_XL for the same layers.
    #[test]
    fn layers_0_and_3_map_to_the_unsloth_gguf_names_and_dims() {
        let hf = concat!(env!("CARGO_MANIFEST_DIR"), "/../hf/tests/fixtures");
        let config = Config::parse(
            &gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__config.json.gz"
            )),
            Some(&gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__hf_quant_config.json.gz"
            ))),
        )
        .unwrap();
        let table = table(&gunzip(&format!(
            "{hf}/checkpoint-headers/glm-5.3-flash-nvfp4-layers-0-3.tensors.json.gz"
        )));
        let tensors = logical(&config, &table);
        let got = map(GLM, tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice()))).unwrap();
        let got: BTreeMap<String, Vec<usize>> = got.into_iter().map(|(n, m)| (n, m.dims)).collect();
        let want = gguf_table(
            include_str!("../../../../hf/tests/fixtures/glm-5.3-flash-ud-q4_k_xl-layers-0-3.tsv"),
            |_| true,
        );
        assert_eq!(got, want);
    }

    /// kv_b_proj splits per head as llama.cpp's converter does (`view(heads, nope + v, rank)`,
    /// split, key half transposed); `A_log` becomes `-exp` of its first `heads` values; a missing
    /// expert, an unknown name and a zero count in the config are refused by name.
    #[test]
    fn layouts_move_elements_as_the_converter_does() {
        // Two heads of nope 2 and v 1 over rank 3: element (row, col) holds 10 * row + col.
        let (heads, nope, v, rank) = (2, 2, 1, 3);
        let src: Vec<u8> = (0..heads * (nope + v))
            .flat_map(|r| (0..rank).map(move |c| (10 * r + c) as u8))
            .collect();
        let shape = [heads * (nope + v), rank];
        let kv = layer(
            Geometry {
                heads,
                nope,
                v,
                ..GLM
            },
            0,
            "self_attn.kv_b_proj.weight",
            "kv_b_proj",
            &shape,
        )
        .unwrap();
        let k = kv[0].1.layout.apply(&src, &shape, 1);
        assert_eq!(k, [0, 10, 1, 11, 2, 12, 30, 40, 31, 41, 32, 42]);
        let v_half = kv[1].1.layout.apply(&src, &shape, 1);
        assert_eq!(v_half, [20, 21, 22, 50, 51, 52]);
        let a: Vec<u8> = [0.0f32, 1.0, 9.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let neg = Layout::NegExp { heads: 2 }.apply(&a, &[3], 4);
        let neg: Vec<f32> = neg
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        assert_eq!(neg, [-1.0, -1.0f32.exp()]);

        let g = Geometry {
            layers: 2,
            heads: 2,
            nope: 2,
            v: 1,
            experts: 2,
        };
        let lone = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            &[4usize, 8][..],
        )];
        let e = map(g, lone).unwrap_err().to_string();
        assert!(
            e.contains("experts.0.up_proj.weight: layer 0 up_proj has 1 of 2 experts"),
            "{e}"
        );
        let twice = [
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                &[4usize, 8][..],
            ),
            ("model.layers.0.mlp.experts.1.up_proj.weight", &[4, 8]),
            (
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
                &[4, 8],
            ),
        ];
        let e = map(g, twice).unwrap_err().to_string();
        assert!(
            e.contains("layer 0 up_proj expert 0 is also supplied by model.layers.0.mlp"),
            "{e}"
        );
        let e = map(
            g,
            [("model.layers.0.self_attn.rotary.inv_freq", &[4usize][..])],
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("rotary.inv_freq: not a GLM-5.3-Flash tensor"),
            "{e}"
        );
        let vectors = [
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                &[4usize][..],
            ),
            ("model.layers.0.mlp.experts.1.gate_proj.weight", &[4]),
        ];
        let e = map(g, vectors).unwrap_err().to_string();
        assert!(e.contains("expert projection [4] is not 2-D"), "{e}");
        let e = map(
            g,
            [("model.layers.0.mlp.gate_proj.weight", &[8usize, 4][..])],
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("layer 0 has blk.0.ffn_gate.weight but not blk.0.ffn_up.weight"),
            "{e}"
        );
        let wide = Geometry {
            heads: usize::MAX,
            ..g
        };
        let e = map(
            wide,
            [(
                "model.layers.0.self_attn.kv_b_proj.weight",
                &[6usize, 3][..],
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("6 rows, not"), "{e}");
        let skipped = [
            ("model.layers.2.eh_proj.weight", &[4usize, 8][..]),
            ("model.visual.blocks.0.attn.qkv.weight", &[4, 8]),
        ];
        assert!(map(g, skipped).unwrap().is_empty());
        let zero = serde_json::json!({ "num_hidden_layers": 2, "num_attention_heads": 0,
            "qk_nope_head_dim": 2, "v_head_dim": 1, "n_routed_experts": 2 });
        let e = Geometry::of(&zero).unwrap_err().to_string();
        assert!(e.contains("num_attention_heads is not a count"), "{e}");
    }

    /// Every layer of nvidia/GLM-5.3-Flash-NVFP4 (all 33 shard headers in
    /// `$GLM_MAP_ORACLE/headers`) maps to the names and dims of unsloth's UD-Q4_K_XL table
    /// (`tensors.tsv`) for layers 0 to 44 and the unlayered tensors; and the small F32 tensors
    /// range-read from both (`hf/`, `gguf.safetensors`, written by the evidence's `oracle.py`)
    /// are byte-equal after the map's layout (`ssm_a` within one unit in the last place), or for
    /// the BF16 ones equal to the GGUF's rounded.
    #[test]
    #[ignore = "reads $GLM_MAP_ORACLE"]
    fn hw_glm_map_matches_the_unsloth_gguf() {
        let dir =
            std::path::PathBuf::from(std::env::var("GLM_MAP_ORACLE").expect("GLM_MAP_ORACLE"));
        let hf = concat!(env!("CARGO_MANIFEST_DIR"), "/../hf/tests/fixtures");
        let config = Config::parse(
            &gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__config.json.gz"
            )),
            Some(&gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__hf_quant_config.json.gz"
            ))),
        )
        .unwrap();
        let mut all = Vec::new();
        for f in std::fs::read_dir(dir.join("headers")).unwrap() {
            all.extend(table(&gunzip(f.unwrap().path().to_str().unwrap())));
        }
        let tensors = logical(&config, &all);
        let got = map(GLM, tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice()))).unwrap();
        let dims: BTreeMap<String, Vec<usize>> = got
            .iter()
            .map(|(n, m)| (n.clone(), m.dims.clone()))
            .collect();
        let tsv = std::fs::read_to_string(dir.join("tensors.tsv")).unwrap();
        let want = gguf_table(&tsv, |n| {
            n.strip_prefix("blk.")
                .is_none_or(|r| r.split('.').next().unwrap().parse::<usize>().unwrap() < 45)
        });
        assert_eq!(dims.len(), want.len());
        assert_eq!(dims, want);
        println!("names and dims equal: {} tensors", dims.len());

        let small = hf::safetensors::Folder::open(&dir.join("hf")).unwrap();
        let gguf = hf::safetensors::Safetensors::open(&dir.join("gguf.safetensors")).unwrap();
        let (mut equal, mut rounded, mut last_place) = (0, 0, 0);
        for name in gguf.names() {
            let m = &got[name];
            let raw = small.raw(&m.source).unwrap();
            let want = gguf.raw(name).unwrap().bytes;
            match raw.dtype {
                // The converter's `torch.exp` is not correctly rounded, so `-exp` may differ in
                // the last place: `ssm_a` is held to one unit there.
                Dtype::F32 if m.layout != Layout::Same => {
                    let out = m.layout.apply(raw.bytes, raw.shape, 4);
                    assert_eq!(out.len(), want.len(), "{name}: converted byte count");
                    for (&o, &w) in out.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0) {
                        let o = i32::from_le_bytes(o);
                        let w = i32::from_le_bytes(w);
                        assert!((o - w).abs() <= 1, "{name}: {o:#x} against {w:#x}");
                        last_place += usize::from(o != w);
                    }
                    equal += 1;
                }
                Dtype::F32 => {
                    assert_eq!(
                        m.layout.apply(raw.bytes, raw.shape, 4),
                        want,
                        "{name} from {}",
                        m.source
                    );
                    equal += 1;
                }
                // NVIDIA's export stores these BF16 where the GGUF's source kept F32: each BF16
                // value is the GGUF's F32 rounded to nearest even.
                Dtype::Bf16 => {
                    let out = m.layout.apply(raw.bytes, raw.shape, 2);
                    let round = |b: [u8; 4]| {
                        let x = u32::from_le_bytes(b);
                        ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16
                    };
                    let want: Vec<u8> = want
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|&b| round(b).to_le_bytes())
                        .collect();
                    assert_eq!(out, want, "{name} from {}", m.source);
                    rounded += 1;
                }
                d => panic!("{}: {}", m.source, d.name()),
            }
        }
        println!(
            "F32 equal: {equal} tensors, {last_place} ssm_a values one unit apart; \
             BF16 equal to the GGUF's F32 rounded: {rounded}"
        );
    }
}
