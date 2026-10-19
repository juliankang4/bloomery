//! A ModelOpt checkpoint's tensors in GGUF's types: one table ([`normalize`]) from the format a
//! tensor is placed in ([`quant_config::Format`]) to the GGUF type it is written as and the
//! repack that takes it there. A model's map supplies only names, roles and shapes; the type
//! follows from the format and from how the role reads the tensor ([`Use`]).
//!
//! ModelOpt stores an NVFP4 linear `<stem>` as a triple: `<stem>.weight`, U8 `[rows, k/2]`, two
//! E2M1 codes a byte (value `2i` in the low nibble, `2i + 1` in the high); `<stem>.weight_scale`,
//! F8_E4M3 `[rows, k/16]`, one scale a 16-value group; and `<stem>.weight_scale_2`, one F32, the
//! tensor's scale. A value is `e2m1 · weight_scale · weight_scale_2`. The plan names the two
//! scales a weight reads ([`Placed::companions`]); a module that stores no tensor scale reads the
//! one its parent module shares.
//!
//! GGUF's NVFP4 block (type 40, [`gguf::quant::dequant_row`]) holds 64 values in 36 bytes: the four
//! groups' scale bytes, then 32 code bytes, group `s`'s value `j` in the low nibble of byte `8s + j`
//! and value `j + 8` in its high nibble. The reader moves the nibbles into that order and copies
//! the scale bytes as they are; the tensor scale becomes the f32 the GGUF file keeps beside the
//! weight as `<stem>.scale`. A routed stack (`experts`) has one tensor scale per expert, so its
//! `.scale` holds one value per expert, in expert order.
//!
//! A scale byte UE4M3 does not define as a scale (bit 7 set, or `0x7F`) is refused by name, as the
//! block's decode refuses it, rather than stripped of bit 7. `<stem>.input_scale`, the activation
//! scale ModelOpt calibrates for 4-bit activations, has no place in the block; the reader leaves it.

use gguf::quant::{E4m3Rows, GgmlType, Q8Block, QuantError, fp4_pair, ue4m3_to_f32_half};

use crate::quant_config::{Format, Placed};
use crate::safetensors::{self, Dtype, Raw};

/// Why a ModelOpt tensor was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Safetensors(#[from] safetensors::Error),
    #[error("modelopt nvfp4: {name}: {what}")]
    Shape { name: String, what: String },
    #[error("{name}: {format} {what}")]
    Format {
        name: String,
        format: Format,
        what: String,
    },
    #[error("modelopt nvfp4: {name}: row {row} group {group}: {source}")]
    Scale {
        name: String,
        row: usize,
        group: usize,
        source: QuantError,
    },
}

/// One NVFP4 tensor, or a routed stack of them, in GGUF's layout.
#[derive(Debug, Default)]
struct Nvfp4 {
    /// The NVFP4 blocks, row-major, a stack expert by expert: `rows · k / 64` blocks of 36 bytes
    /// an expert.
    blocks: Vec<u8>,
    /// ggml's dims: `[k, rows]`, or `[k, rows, n_expert]` for a stack.
    dims: Vec<u64>,
    /// The `<stem>.scale` values: the tensor scale, or one per expert for a stack.
    scale: Vec<f32>,
}

/// What a role makes of a weight its checkpoint stores unquantized (BF16 or F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plain {
    /// ggml's reference Q8_0 ([`Q8Block::quantize_ref`]), as llama.cpp's converters write it.
    Q8_0,
    F32,
    /// BF16 as stored; an F32 tensor is refused.
    Bf16,
}

/// How a role reads its tensor, which with the tensor's format picks its GGUF type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// A matrix or vector a kernel reads: a quantized format keeps its blocks; an unquantized
    /// one is written as the role asks.
    Matrix(Plain),
    /// A table the host reads a row at a time ([`GgmlType::F8E4M3_ROWS`]).
    Rows,
}

/// One tensor in its GGUF type.
#[derive(Debug, Default, PartialEq)]
pub struct Normal {
    pub type_id: u32,
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
    /// Row-major in the type's blocks.
    pub bytes: Vec<u8>,
    /// The values of the `<stem>.scale` the type reads beside it; empty for a type that reads
    /// none.
    pub scale: Vec<f32>,
}

/// The GGUF type id `format` is written as when `role` reads it: the table [`normalize`] runs.
///
/// | format | [`Use::Matrix`] | [`Use::Rows`] |
/// |---|---|---|
/// | NVFP4 (ModelOpt) | NVFP4 (40), `.scale` a tensor or an expert (`linear`, `experts`) | none |
/// | BF16, F32 | Q8_0, F32 or BF16 as [`Plain`] says | none |
/// | FP8 with one tensor scale (its companion) | none | [`GgmlType::F8E4M3_ROWS`], bytes as stored |
///
/// Every other pair has no GGUF type here yet.
#[must_use]
pub fn type_of(format: Format, role: Use) -> Option<u32> {
    match (format, role) {
        (Format::Nvfp4, Use::Matrix(_)) => Some(GgmlType::NVFP4.as_u32()),
        (Format::Bf16 | Format::F32, Use::Matrix(Plain::Q8_0)) => Some(GgmlType::Q8_0.as_u32()),
        (Format::Bf16 | Format::F32, Use::Matrix(Plain::F32)) => Some(GgmlType::F32.as_u32()),
        (Format::Bf16, Use::Matrix(Plain::Bf16)) => Some(GgmlType::BF16.as_u32()),
        (Format::Fp8Tensor, Use::Rows) => Some(GgmlType::F8E4M3_ROWS.as_u32()),
        _ => None,
    }
}

fn finite(name: &str, raw: &Raw<'_>) -> Result<(), Error> {
    let at = match raw.dtype {
        Dtype::F32 => raw
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .position(|b| !f32::from_le_bytes(*b).is_finite()),
        Dtype::Bf16 => raw
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .position(|b| u16::from_le_bytes(*b) & 0x7f80 == 0x7f80),
        _ => {
            return Err(safetensors::Error::Dtype {
                name: name.into(),
                dtype: raw.dtype.name().into(),
            }
            .into());
        }
    };
    if let Some(at) = at {
        return Err(Error::Shape {
            name: name.into(),
            what: format!("non-finite value at {at}"),
        });
    }
    Ok(())
}

/// The one format of `sources` (one weight, or a routed stack's experts in expert order), each
/// with its place in the plan.
///
/// # Errors
///
/// No source, and a source placed in another format than the first, by name.
pub fn format_of(sources: &[(&str, &Placed)]) -> Result<Format, Error> {
    let &[(first, placed), ..] = sources else {
        return Err(Error::Shape {
            name: "a tensor".into(),
            what: "has no source".into(),
        });
    };
    if let Some((name, p)) = sources.iter().find(|(_, p)| p.format != placed.format) {
        return Err(Error::Format {
            name: name.to_string(),
            format: p.format,
            what: format!("in a stack whose first source {first} is {}", placed.format),
        });
    }
    Ok(placed.format)
}

/// The checkpoint tensors `sources` (one weight, or a routed stack's experts in expert order),
/// each read through the companions its place in the plan names, in the GGUF type [`type_of`]
/// gives for `role`.
///
/// # Errors
///
/// A refusal of [`format_of`]; a pair [`type_of`] has no type for; a tensor `get` refuses or
/// lacks; a refusal of `linear` or `experts`; a stack of a format that is not NVFP4; a shape or
/// dtype the format does not hold; a non-finite value or scaled value; and a row a
/// [`Plain::Q8_0`] block does not divide, each by name.
pub fn normalize<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
) -> Result<Normal, Error> {
    let format = format_of(sources)?;
    let (name, placed) = sources[0];
    let refuse = |what: String| Error::Format {
        name: name.to_string(),
        format,
        what,
    };
    let type_id = type_of(format, role)
        .ok_or_else(|| refuse(format!("read as {role:?} has no GGUF type here")))?;
    if sources.len() != 1 && format != Format::Nvfp4 {
        return Err(refuse(format!(
            "in a stack of {}: only NVFP4 experts stack here",
            sources.len()
        )));
    }
    if format == Format::Nvfp4 {
        let n = if let [one] = sources {
            linear(get, *one)?
        } else {
            experts(get, sources)?
        };
        return Ok(Normal {
            type_id,
            dims: n.dims,
            bytes: n.blocks,
            scale: n.scale,
        });
    }
    let raw = get(name)?;
    let dims: Vec<u64> = raw.shape.iter().rev().map(|&d| d as u64).collect();
    if format == Format::Fp8Tensor {
        if raw.dtype != Dtype::F8E4M3 || raw.shape.len() != 2 {
            return Err(refuse(format!(
                "is {} {:?}, not F8_E4M3 rows",
                raw.dtype.name(),
                raw.shape
            )));
        }
        let [scale_name] = placed.companions.as_slice() else {
            return Err(refuse(format!(
                "reads {:?}, not one tensor scale",
                placed.companions
            )));
        };
        let s = get(scale_name)?;
        finite(scale_name, &s)?;
        let scale = match (s.dtype, s.bytes) {
            (Dtype::F32, &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]),
            (Dtype::Bf16, &[a, b]) => f32::from_bits(u32::from(u16::from_le_bytes([a, b])) << 16),
            _ => {
                return Err(refuse(format!(
                    "scale {scale_name} is {} {:?}, not one F32 or BF16",
                    s.dtype.name(),
                    s.shape
                )));
            }
        };
        let decoder = E4m3Rows::new(scale);
        let mut values = [0.0; 256];
        for (chunk, bytes) in raw.bytes.chunks(values.len()).enumerate() {
            let values = &mut values[..bytes.len()];
            decoder
                .decode(bytes, values)
                .map_err(|e| refuse(e.to_string()))?;
            if let Some(at) = values.iter().position(|v| !v.is_finite()) {
                return Err(refuse(format!(
                    "value {} times scale {scale_name} is not finite",
                    chunk * 256 + at
                )));
            }
        }
        return Ok(Normal {
            type_id,
            dims,
            bytes: raw.bytes.to_vec(),
            scale: vec![scale],
        });
    }
    finite(name, &raw)?;
    let bytes = match role {
        Use::Matrix(Plain::Bf16) => raw.bytes.to_vec(),
        Use::Matrix(Plain::F32) => safetensors::f32_values(name, &raw)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => {
            if !raw.shape.last().is_some_and(|k| k.is_multiple_of(32)) {
                return Err(refuse(format!(
                    "shape {:?}: a row is not whole 32-value Q8_0 blocks",
                    raw.shape
                )));
            }
            let values = safetensors::f32_values(name, &raw)?;
            let capacity = (values.len() / 32)
                .checked_mul(34)
                .ok_or_else(|| refuse("Q8_0 byte count overflows usize".into()))?;
            let mut bytes = Vec::with_capacity(capacity);
            for (i, b) in values.as_chunks::<32>().0.iter().enumerate() {
                let q = Q8Block::quantize_ref(b).ok_or_else(|| {
                    refuse(format!(
                        "Q8_0 block {i} has non-finite values or an unrepresentable scale"
                    ))
                })?;
                bytes.extend(q.to_bytes());
            }
            bytes
        }
    };
    Ok(Normal {
        type_id,
        dims,
        bytes,
        scale: vec![],
    })
}

/// The NVFP4 linear `source`: its weight, and the block and tensor scales its place names,
/// looked up through `get`.
///
/// # Errors
///
/// A place that does not name the two scales; a tensor `get` refuses or lacks, or whose bytes do
/// not fill its shape; a weight that is not U8 `[rows, k/2]` with `k` a multiple of 64, a scale
/// that is not F8_E4M3 `[rows, k/16]`, a tensor scale that is not one finite F32, and a scale
/// byte that is no UE4M3 scale, each by name.
fn linear<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    source: (&str, &Placed),
) -> Result<Nvfp4, Error> {
    let mut out = Nvfp4::default();
    let [rows, k] = append(&get, source, &mut out)?;
    out.dims = vec![k as u64, rows as u64];
    Ok(out)
}

/// A routed stack: the NVFP4 linears `sources`, one an expert in expert order, as one tensor of
/// `[k, rows, n_expert]` with one tensor scale an expert, each the one its expert's place names.
///
/// # Errors
///
/// As [`linear`] for each expert; an empty stack, and an expert whose shape differs from the
/// first's, by name.
fn experts<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
) -> Result<Nvfp4, Error> {
    let &[first, ..] = sources else {
        return Err(Error::Shape {
            name: "routed stack".into(),
            what: "no expert".into(),
        });
    };
    let mut out = Nvfp4::default();
    let shape = append(&get, first, &mut out)?;
    for &source in &sources[1..] {
        let have = append(&get, source, &mut out)?;
        if have != shape {
            return Err(Error::Shape {
                name: source.0.to_string(),
                what: format!(
                    "rows and k {have:?}; the stack's first expert {} has {shape:?}",
                    first.0
                ),
            });
        }
    }
    let [rows, k] = shape;
    out.dims = vec![k as u64, rows as u64, sources.len() as u64];
    Ok(out)
}

/// Reads the weight of `source` and the two scales its place names, and appends its blocks and
/// tensor scale to `out`. Returns `[rows, k]`.
fn append<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    out: &mut Nvfp4,
) -> Result<[usize; 2], Error> {
    let [block_scale, tensor_scale] = placed.companions.as_slice() else {
        return Err(Error::Shape {
            name: weight.to_string(),
            what: format!(
                "its place names {:?}, not a block scale and a tensor scale",
                placed.companions
            ),
        });
    };
    let typed = |name: &str, want: Dtype| -> Result<(String, Raw<'a>), Error> {
        let name = name.to_string();
        let raw = get(&name)?;
        if raw.dtype != want {
            return Err(safetensors::Error::Kind {
                name,
                have: raw.dtype.name(),
                want: want.name(),
            }
            .into());
        }
        let bytes = raw
            .shape
            .iter()
            .try_fold(want.size(), |n, &d| n.checked_mul(d));
        if bytes != Some(raw.bytes.len()) {
            return Err(Error::Shape {
                name,
                what: format!(
                    "shape {:?} does not match {} stored bytes",
                    raw.shape,
                    raw.bytes.len()
                ),
            });
        }
        Ok((name, raw))
    };
    let shape = |name: &str, what: String| Error::Shape {
        name: name.to_string(),
        what,
    };
    let (wname, w) = typed(weight, Dtype::U8)?;
    let &[rows, half] = w.shape else {
        return Err(shape(
            &wname,
            format!("shape {:?}, not [rows, k/2]", w.shape),
        ));
    };
    let k = half
        .checked_mul(2)
        .ok_or_else(|| shape(&wname, format!("k/2 = {half} overflows k")))?;
    if k == 0 || !k.is_multiple_of(64) {
        return Err(shape(
            &wname,
            format!("k = {k}, not a positive multiple of the 64 values of a block"),
        ));
    }
    let (sname, s) = typed(block_scale, Dtype::F8E4M3)?;
    if s.shape != [rows, k / 16] {
        return Err(shape(
            &sname,
            format!(
                "shape {:?}; its weight's 16-value groups are [{rows}, {}]",
                s.shape,
                k / 16
            ),
        ));
    }
    let (tname, t) = typed(tensor_scale, Dtype::F32)?;
    let scale = <[u8; 4]>::try_from(t.bytes)
        .map(f32::from_le_bytes)
        .map_err(|_| shape(&tname, format!("shape {:?}, not one value", t.shape)))?;
    finite(&tname, &t)?;

    let bytes = rows
        .checked_mul(k / 64)
        .and_then(|n| n.checked_mul(36))
        .ok_or_else(|| shape(&wname, "NVFP4 byte count overflows usize".into()))?;
    out.blocks
        .try_reserve(bytes)
        .map_err(|e| shape(&wname, format!("cannot reserve {bytes} NVFP4 bytes: {e}")))?;
    for (row, (w, s)) in w
        .bytes
        .chunks_exact(half)
        .zip(s.bytes.chunks_exact(k / 16))
        .enumerate()
    {
        for (block, (w, d)) in w
            .as_chunks::<32>()
            .0
            .iter()
            .zip(s.as_chunks::<4>().0)
            .enumerate()
        {
            for (sub, &byte) in d.iter().enumerate() {
                let half_scale = ue4m3_to_f32_half(byte).map_err(|source| Error::Scale {
                    name: sname.clone(),
                    row,
                    group: 4 * block + sub,
                    source,
                })?;
                if !(half_scale * 12.0 * scale).is_finite()
                    && w[sub * 8..][..8]
                        .iter()
                        .flat_map(|&b| fp4_pair(b))
                        .any(|v| !(half_scale * f32::from(v) * scale).is_finite())
                {
                    return Err(shape(
                        &wname,
                        format!(
                            "row {row} group {} times scale {tname} is not finite",
                            4 * block + sub
                        ),
                    ));
                }
            }
            out.blocks.extend_from_slice(d);
            for g in w.as_chunks::<8>().0 {
                // Value j is the (j % 2) nibble of byte j / 2; the block's byte j holds value j
                // low and value j + 8 high.
                out.blocks.extend((0..8).map(|j| {
                    let nib = |b: u8| (b >> (4 * (j % 2))) & 0x0f;
                    nib(g[j / 2]) | nib(g[4 + j / 2]) << 4
                }));
            }
        }
    }
    out.scale.push(scale);
    Ok([rows, k])
}

#[cfg(test)]
mod tests {
    use gguf::quant::{GgmlType, dequant_row};
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::safetensors::{Safetensors, write};

    const STEMS: [&str; 2] = [
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.1.gate_proj",
    ];

    fn fixture() -> Safetensors {
        Safetensors::parse(include_bytes!("../tests/fixtures/modelopt-nvfp4.safetensors").to_vec())
            .unwrap()
    }

    /// The place the plan gives the triple `stem`: its weight with its own block and tensor
    /// scales.
    fn own(stem: &str) -> (String, Placed) {
        let companions = ["weight_scale", "weight_scale_2"].map(|s| format!("{stem}.{s}"));
        (
            format!("{stem}.weight"),
            Placed {
                format: Format::Nvfp4,
                companions: companions.to_vec(),
            },
        )
    }

    /// The pairs [`normalize`] reads from owned places.
    fn sources(owned: &[(String, Placed)]) -> Vec<(&str, &Placed)> {
        owned.iter().map(|(w, p)| (w.as_str(), p)).collect()
    }

    fn lin<'a>(
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
        stem: &str,
    ) -> Result<Nvfp4, Error> {
        let (w, p) = own(stem);
        linear(get, (&w, &p))
    }

    fn stack<'a>(
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
        stems: &[&str],
    ) -> Result<Nvfp4, Error> {
        let owned: Vec<_> = stems.iter().map(|s| own(s)).collect();
        experts(get, &sources(&owned))
    }

    /// The reference's row digests and tensor scale of `stem`.
    fn reference(stem: &str) -> (Vec<String>, f64) {
        let v: Value =
            serde_json::from_str(include_str!("../tests/fixtures/modelopt-nvfp4-ref.json"))
                .unwrap();
        let t = &v["tensors"][stem];
        let rows = t["rows"].as_array().unwrap();
        (
            rows.iter()
                .map(|r| r.as_str().unwrap().to_string())
                .collect(),
            t["scale"].as_f64().unwrap(),
        )
    }

    /// The sha256 of each row of `blocks` decoded by GGUF's NVFP4 decode and multiplied by
    /// `scale`, its f32 bits little-endian.
    fn decoded(blocks: &[u8], k: usize, scale: f32) -> Vec<String> {
        blocks
            .chunks_exact(k / 64 * 36)
            .map(|row| {
                let mut v = vec![0.0f32; k];
                dequant_row(GgmlType::NVFP4, row, &mut v).unwrap();
                let bits: Vec<u8> = v.iter().flat_map(|x| (x * scale).to_le_bytes()).collect();
                format!("{:x}", Sha256::digest(&bits))
            })
            .collect()
    }

    /// The oracle: every row of the synthetic tensor (every defined scale byte, every code at
    /// every position of a group) and of the real `gate_proj` rows, read here and decoded by
    /// GGUF's NVFP4 decode times the tensor scale, has the f32 bits the ModelOpt rule gives in
    /// `modelopt_nvfp4.py ref`. A nibble order other than the block's, a scale byte at another
    /// group, or a tensor scale from another expert changes a row's digest.
    #[test]
    fn linears_decode_to_the_modelopt_rule() {
        let st = fixture();
        for stem in ["synthetic", STEMS[0], STEMS[1]] {
            let t = lin(|n| st.raw(n), stem).unwrap();
            let (rows, scale) = reference(stem);
            let k = t.dims[0] as usize;
            assert_eq!(t.dims, [k as u64, rows.len() as u64], "{stem}");
            assert_eq!(t.scale, [scale as f32], "{stem}");
            assert_eq!(t.blocks.len(), rows.len() * k / 64 * 36, "{stem}");
            assert_eq!(decoded(&t.blocks, k, t.scale[0]), rows, "{stem}");
        }
    }

    /// A routed stack of two real experts with different tensor scales: the blocks are each
    /// expert's in expert order, `.scale` holds each expert's own scale, and every row decodes
    /// to the reference under its expert's scale.
    #[test]
    fn a_routed_stack_keeps_one_scale_an_expert() {
        let st = fixture();
        let t = stack(|n| st.raw(n), &STEMS).unwrap();
        assert_eq!(t.dims, [2560, 4, 2]);
        let per = t.blocks.len() / 2;
        for (e, stem) in STEMS.iter().enumerate() {
            let (rows, scale) = reference(stem);
            assert_eq!(t.scale[e], scale as f32, "{stem}");
            assert_eq!(
                decoded(&t.blocks[e * per..(e + 1) * per], 2560, t.scale[e]),
                rows,
                "{stem}"
            );
        }
        assert_ne!(t.scale[0], t.scale[1]);
    }

    /// Each expert reads the tensor scale its place names: one whose module stores none reads
    /// the one its parent module shares, beside one with its own, each in its own slot of the
    /// stack's `.scale`.
    ///
    /// FAIL-first: reading `<stem>.weight_scale_2` by name refuses expert 0 as missing.
    #[test]
    fn a_stack_reads_each_expert_s_planned_tensor_scale() {
        let codes = [0x11u8; 32];
        let scales = [0x38u8; 4];
        let (shared, local) = (2.0f32.to_le_bytes(), 3.0f32.to_le_bytes());
        let st = Safetensors::parse(write(&[
            ("e.0.weight", "U8", &[1, 32][..], &codes[..]),
            ("e.0.weight_scale", "F8_E4M3", &[1, 4], &scales),
            ("e.weight_scale_2", "F32", &[], &shared),
            ("e.1.weight", "U8", &[1, 32], &codes),
            ("e.1.weight_scale", "F8_E4M3", &[1, 4], &scales),
            ("e.1.weight_scale_2", "F32", &[], &local),
        ]))
        .unwrap();
        let config = crate::quant_config::Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": null,
                "quantized_layers": { "e": { "quant_algo": "NVFP4", "group_size": 16 } },
            }})),
        )
        .unwrap();
        let raws: Vec<_> = st.names().map(|n| (n, st.raw(n).unwrap())).collect();
        let plan =
            crate::quant_config::plan(&config, raws.iter().map(|(n, r)| (*n, r.dtype, r.shape)));
        let experts = ["e.0.weight", "e.1.weight"].map(|n| (n, &plan.placed[n]));
        let got = normalize(|n| st.raw(n), &experts, Use::Matrix(Plain::Q8_0)).unwrap();
        assert_eq!(got.scale, [2.0, 3.0]);
        assert_eq!(got.dims, [64, 1, 2]);
        let alone = normalize(|n| st.raw(n), &experts[..1], Use::Matrix(Plain::Q8_0)).unwrap();
        assert_eq!((alone.dims, alone.scale), (vec![64, 1], vec![2.0]));
    }

    /// The table: NVFP4 linears and stacks come out as [`linear`] and [`experts`] give them;
    /// BF16 becomes ggml's reference Q8_0 of its values, their f32 widening, or itself; an F8
    /// row table keeps its bytes and reads its one BF16 scale; every pair without a line, a
    /// stack of anything but NVFP4, an F32 asked for BF16 and a row Q8_0 does not divide are
    /// refused by name.
    #[test]
    fn the_table_writes_each_format_as_its_type() {
        let st = fixture();
        let get = |n: &str| st.raw(n);
        let placed = |format, companions: &[&str]| Placed {
            format,
            companions: companions.iter().map(|c| c.to_string()).collect(),
        };
        let owned = STEMS.map(own);
        let both = sources(&owned);
        let m = Use::Matrix(Plain::Q8_0);
        let one = normalize(get, &both[..1], m).unwrap();
        let first = lin(get, STEMS[0]).unwrap();
        assert_eq!(
            (one.type_id, &one.dims, &one.bytes, &one.scale),
            (40, &first.dims, &first.blocks, &first.scale)
        );
        let two = normalize(get, &both, m).unwrap();
        let exp = stack(get, &STEMS).unwrap();
        assert_eq!(
            (&two.dims, &two.bytes, &two.scale),
            (&exp.dims, &exp.blocks, &exp.scale)
        );
        let bf16_first = placed(Format::Bf16, &[]);
        let e = normalize(get, &[("b.weight", &bf16_first), both[0]], m).unwrap_err();
        assert!(
            e.to_string()
                .contains("NVFP4-ModelOpt in a stack whose first source b.weight is BF16"),
            "{e}"
        );
        let e = normalize(get, &[], m).unwrap_err();
        assert!(e.to_string().contains("has no source"), "{e}");
        let bare = placed(Format::Nvfp4, &[]);
        let e = normalize(get, &[(both[0].0, &bare)], m).unwrap_err();
        assert!(
            e.to_string()
                .contains("its place names [], not a block scale and a tensor scale"),
            "{e}"
        );

        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 20.5) / 7.0).collect();
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let wide: Vec<f32> = bf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect();
        let f8 = [0x38u8, 0x40, 0x30, 0x00, 0xB8, 0x7E];
        let f32s: Vec<u8> = wide.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale = 0x3a00u16.to_le_bytes();
        let mut nan = bf16[..128].to_vec();
        nan[2 * 37..][..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
        let file = write(&[
            ("b.weight", "BF16", &[2, 32][..], &bf16[..]),
            ("f.weight", "F32", &[2, 32], &f32s),
            ("odd.weight", "BF16", &[4, 16], &bf16),
            ("t.weight", "F8_E4M3", &[3, 2], &f8),
            ("t.scale", "BF16", &[1], &scale),
            ("nan.weight", "BF16", &[1, 64], &nan),
        ]);
        let st = Safetensors::parse(file).unwrap();
        let get = |n: &str| st.raw(n);
        let b = placed(Format::Bf16, &[]);
        let q = normalize(get, &[("b.weight", &b)], m).unwrap();
        let want: Vec<u8> = wide
            .as_chunks::<32>()
            .0
            .iter()
            .flat_map(|blk| Q8Block::quantize_ref(blk).unwrap().to_bytes())
            .collect();
        assert_eq!((q.type_id, &q.dims, &q.bytes), (8, &vec![32, 2], &want));
        let f = normalize(get, &[("b.weight", &b)], Use::Matrix(Plain::F32)).unwrap();
        assert_eq!((f.type_id, f.bytes.len()), (0, 256));
        assert!(
            f.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(&wide)
                .all(|(x, w)| f32::from_le_bytes(*x) == *w)
        );
        let kept = normalize(get, &[("b.weight", &b)], Use::Matrix(Plain::Bf16)).unwrap();
        assert_eq!((kept.type_id, &kept.bytes), (30, &bf16));
        let rows = normalize(
            get,
            &[("t.weight", &placed(Format::Fp8Tensor, &["t.scale"]))],
            Use::Rows,
        )
        .unwrap();
        assert_eq!(
            (rows.type_id, &rows.dims, &rows.bytes[..], &rows.scale[..]),
            (
                2000,
                &vec![2, 3],
                &f8[..],
                &[f32::from_bits(0x3a00u32 << 16)][..]
            )
        );

        let refused = |names: &[&str], p: Placed, role: Use, want: &str| {
            let names: Vec<_> = names.iter().map(|n| (*n, &p)).collect();
            let e = normalize(get, &names, role).unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        };
        refused(
            &["b.weight"],
            placed(Format::Mxfp8, &[]),
            m,
            "b.weight: MXFP8 read as Matrix(Q8_0) has no GGUF type here",
        );
        refused(
            &["b.weight"],
            placed(
                Format::Fp8Block {
                    rows: 128,
                    cols: 128,
                },
                &[],
            ),
            m,
            "FP8-block 128x128 read as",
        );
        refused(
            &["t.weight"],
            placed(Format::Nvfp4, &[]),
            Use::Rows,
            "NVFP4-ModelOpt read as Rows",
        );
        refused(
            &["b.weight", "b.weight"],
            b,
            m,
            "in a stack of 2: only NVFP4 experts stack here",
        );
        refused(
            &["f.weight"],
            placed(Format::F32, &[]),
            Use::Matrix(Plain::Bf16),
            "has no GGUF type here",
        );
        refused(
            &["odd.weight"],
            placed(Format::Bf16, &[]),
            m,
            "a row is not whole 32-value Q8_0 blocks",
        );
        refused(&["nan.weight"], placed(Format::Bf16, &[]), m, "nan.weight");
        let mut missed = vec![];
        for (dtype, format) in [(Dtype::F32, Format::F32), (Dtype::Bf16, Format::Bf16)] {
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let bits = x.to_bits();
                let bytes = if dtype == Dtype::F32 {
                    bits.to_le_bytes().to_vec()
                } else {
                    ((bits >> 16) as u16).to_le_bytes().to_vec()
                }
                .repeat(32);
                let st =
                    Safetensors::parse(write(&[("bad.weight", dtype.name(), &[1, 32], &bytes)]))
                        .unwrap();
                for plain in [Plain::F32, Plain::Bf16, Plain::Q8_0] {
                    if type_of(format, Use::Matrix(plain)).is_none() {
                        continue;
                    }
                    let got = normalize(
                        |n| st.raw(n),
                        &[("bad.weight", &placed(format, &[]))],
                        Use::Matrix(plain),
                    );
                    if !got
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("bad.weight"))
                    {
                        missed.push(format!("{dtype:?} {plain:?} {x}: {got:?}"));
                    }
                }
            }
        }
        for (byte, scale, dtype, want) in [
            (0x38, f32::INFINITY, Dtype::F32, "t.scale"),
            (0x38, f32::NEG_INFINITY, Dtype::F32, "t.scale"),
            (0x38, f32::NAN, Dtype::F32, "t.scale"),
            (0x38, f32::INFINITY, Dtype::Bf16, "t.scale"),
            (0x38, f32::NEG_INFINITY, Dtype::Bf16, "t.scale"),
            (0x38, f32::NAN, Dtype::Bf16, "t.scale"),
            (0x7f, 1.0, Dtype::F32, "t.weight"),
            (0xff, 1.0, Dtype::F32, "t.weight"),
            (0x7e, f32::MAX, Dtype::F32, "t.weight"),
        ] {
            let bytes = if dtype == Dtype::F32 {
                scale.to_le_bytes().to_vec()
            } else {
                ((scale.to_bits() >> 16) as u16).to_le_bytes().to_vec()
            };
            let st = Safetensors::parse(write(&[
                ("t.weight", "F8_E4M3", &[1, 1], &[byte]),
                ("t.scale", dtype.name(), &[1], &bytes),
            ]))
            .unwrap();
            let got = normalize(
                |n| st.raw(n),
                &[("t.weight", &placed(Format::Fp8Tensor, &["t.scale"]))],
                Use::Rows,
            );
            if !got.as_ref().is_err_and(|e| e.to_string().contains(want)) {
                missed.push(format!("FP8 {byte:#x} scale {scale}: {got:?}"));
            }
        }
        for (code, scale, ok) in [
            (0x77, f32::MAX, false),
            (0x00, f32::MAX, true),
            (0x77, 0.0, true),
        ] {
            let st = Safetensors::parse(write(&[
                ("t.weight", "U8", &[1, 32], &[code; 32]),
                ("t.weight_scale", "F8_E4M3", &[1, 4], &[0x7e; 4]),
                ("t.weight_scale_2", "F32", &[], &scale.to_le_bytes()),
            ]))
            .unwrap();
            let t = own("t");
            let got = normalize(|n| st.raw(n), &sources(std::slice::from_ref(&t)), m);
            if got.is_ok() != ok
                || (!ok
                    && !got
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("t.weight_scale_2")))
            {
                missed.push(format!("NVFP4 {code:#x} scale {scale}: {got:?}"));
            }
        }
        assert!(missed.is_empty(), "accepted: {missed:?}");
    }

    /// A one-row triple of `k` values: the scale bytes, codes 1 (0.5) throughout, a tensor scale
    /// of 2 and, when `input` is given, an `input_scale` of that value.
    fn triple(k: usize, scales: &[u8], input: Option<f32>) -> Vec<u8> {
        let codes = vec![0x11u8; k / 2];
        let two = 2.0f32.to_le_bytes();
        let input = input.map(f32::to_le_bytes);
        let (wshape, sshape) = ([1, k / 2], [1, k / 16]);
        let mut list: Vec<(&str, &str, &[usize], &[u8])> = vec![
            ("t.weight", "U8", &wshape, &codes),
            ("t.weight_scale", "F8_E4M3", &sshape, scales),
            ("t.weight_scale_2", "F32", &[][..], &two),
        ];
        if let Some(b) = &input {
            list.push(("t.input_scale", "F32", &[][..], b));
        }
        write(&list)
    }

    /// `input_scale` is never read: two values, or none, give the same blocks and scale.
    #[test]
    fn input_scale_is_not_read() {
        let get = |file: Vec<u8>| {
            let st = Safetensors::parse(file).unwrap();
            lin(|n| st.raw(n), "t").unwrap()
        };
        let a = get(triple(64, &[0x38; 4], Some(0.25)));
        let b = get(triple(64, &[0x38; 4], Some(1e30)));
        let c = get(triple(64, &[0x38; 4], None));
        assert_eq!((&a.blocks, &a.scale), (&b.blocks, &b.scale));
        assert_eq!((&a.blocks, &a.scale), (&c.blocks, &c.scale));
    }

    /// A scale byte with bit 7 set, or `0x7F`, is refused naming the tensor, the row and the
    /// group, with the reason the block's decode gives; mainline's conversion strips bit 7.
    #[test]
    fn a_scale_byte_that_is_no_ue4m3_scale_is_refused_by_name() {
        for byte in [0x7Fu8, 0x80, 0xB8, 0xFF] {
            let mut scales = [0x38u8; 8];
            scales[6] = byte;
            let st = Safetensors::parse(triple(128, &scales, None)).unwrap();
            let e = lin(|n| st.raw(n), "t").unwrap_err();
            let why = ue4m3_to_f32_half(byte).unwrap_err();
            assert!(
                matches!(&e, Error::Scale { name, row: 0, group: 6, source }
                    if name == "t.weight_scale" && *source == why),
                "{e:?}"
            );
            assert!(e.to_string().contains("row 0 group 6"), "{e}");
        }
    }

    /// Each malformed triple and stack is refused by name.
    #[test]
    fn malformed_triples_and_stacks_are_refused_by_name() {
        let one = 1.0f32.to_le_bytes();
        let nan = f32::NAN.to_le_bytes();
        let inf = f32::INFINITY.to_le_bytes();
        let neg_inf = f32::NEG_INFINITY.to_le_bytes();
        let two = [0u8; 8];
        let codes = [0u8; 48];
        let s3 = [0x38u8; 6];
        let s4 = [0x38u8; 4];
        let shape = |list: &[(&str, &str, &[usize], &[u8])], name: &str| {
            let st = Safetensors::parse(write(list)).unwrap();
            let e = lin(|n| st.raw(n), "t").unwrap_err();
            assert!(
                matches!(&e, Error::Shape { name: n, .. } if n == name),
                "{e:?}"
            );
        };
        // k = 96 is not a multiple of 64.
        shape(
            &[
                ("t.weight", "U8", &[1, 48], &codes),
                ("t.weight_scale", "F8_E4M3", &[1, 6], &s3),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight",
        );
        // The scale's groups do not match the weight's.
        shape(
            &[
                ("t.weight", "U8", &[1, 32], &codes[..32]),
                ("t.weight_scale", "F8_E4M3", &[2, 2], &s4),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight_scale",
        );
        // Two tensor scales, and one that is NaN.
        for (shape2, bytes) in [
            (&[2][..], &two[..]),
            (&[][..], &nan[..]),
            (&[], &inf),
            (&[], &neg_inf),
        ] {
            shape(
                &[
                    ("t.weight", "U8", &[1, 32], &codes[..32]),
                    ("t.weight_scale", "F8_E4M3", &[1, 4], &s4),
                    ("t.weight_scale_2", "F32", shape2, bytes),
                ],
                "t.weight_scale_2",
            );
        }
        // A width whose k = 2 · (k/2) overflows, zero rows so the file holds no bytes.
        shape(
            &[
                ("t.weight", "U8", &[0, usize::MAX / 2 + 33], &[]),
                ("t.weight_scale", "F8_E4M3", &[0, 4], &[]),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight",
        );
        // A lookup's tensors are checked as the file's are: bytes short of or beyond the
        // shape, and the overflowing width.
        let lookup = |list: &[(&str, Dtype, &[usize], &[u8])], name: &str| {
            let e = lin(
                |n| {
                    list.iter()
                        .find(|t| t.0 == n)
                        .map(|&(_, dtype, shape, bytes)| Raw {
                            dtype,
                            shape,
                            bytes,
                        })
                        .ok_or_else(|| safetensors::Error::Missing(n.into()))
                },
                "t",
            )
            .unwrap_err();
            assert!(
                matches!(&e, Error::Shape { name: n, .. } if n == name),
                "{e:?}"
            );
        };
        let s8 = [0x38u8; 8];
        for (wshape, w, sshape, s, name) in [
            (&[2, 32][..], &codes[..32], &[2, 4][..], &s8[..], "t.weight"),
            (&[1, 32], &codes[..], &[1, 4], &s4, "t.weight"),
            (&[1, 32], &codes[..32], &[1, 4], &s4[..3], "t.weight_scale"),
            (&[1, 32], &codes[..32], &[1, 4], &s8, "t.weight_scale"),
            (&[0, usize::MAX / 2 + 33], &[], &[0, 4], &[], "t.weight"),
        ] {
            lookup(
                &[
                    ("t.weight", Dtype::U8, wshape, w),
                    ("t.weight_scale", Dtype::F8E4M3, sshape, s),
                    ("t.weight_scale_2", Dtype::F32, &[], &one),
                ],
                name,
            );
        }
        // A weight of another dtype, and a missing scale.
        let st =
            Safetensors::parse(write(&[("t.weight", "BF16", &[1, 16], &codes[..32])])).unwrap();
        let e = lin(|n| st.raw(n), "t").unwrap_err();
        assert!(
            matches!(
                &e,
                Error::Safetensors(safetensors::Error::Kind {
                    have: "BF16",
                    want: "U8",
                    ..
                })
            ),
            "{e:?}"
        );
        let st = Safetensors::parse(write(&[("t.weight", "U8", &[1, 32], &codes[..32])])).unwrap();
        let e = lin(|n| st.raw(n), "t").unwrap_err();
        assert!(
            matches!(&e, Error::Safetensors(safetensors::Error::Missing(n)) if n == "t.weight_scale"),
            "{e:?}"
        );
        // A stack of no expert, and one whose second expert has other rows.
        let st = fixture();
        let none: [&str; 0] = [];
        assert!(matches!(
            stack(|n| st.raw(n), &none),
            Err(Error::Shape { .. })
        ));
        let e = stack(|n| st.raw(n), &[STEMS[0], "synthetic"]).unwrap_err();
        assert!(
            matches!(&e, Error::Shape { name, .. } if name == "synthetic.weight"),
            "{e:?}"
        );
    }
}
