//! A ModelOpt NVFP4 checkpoint's weights in GGUF's NVFP4 block.
//!
//! ModelOpt stores an NVFP4 linear `<stem>` as a triple: `<stem>.weight`, U8 `[rows, k/2]`, two
//! E2M1 codes a byte (value `2i` in the low nibble, `2i + 1` in the high); `<stem>.weight_scale`,
//! F8_E4M3 `[rows, k/16]`, one scale a 16-value group; and `<stem>.weight_scale_2`, one F32, the
//! tensor's scale. A value is `e2m1 · weight_scale · weight_scale_2`.
//!
//! GGUF's NVFP4 block (type 40, [`gguf::quant::dequant_row`]) holds 64 values in 36 bytes: the four
//! groups' scale bytes, then 32 code bytes, group `s`'s value `j` in the low nibble of byte `8s + j`
//! and value `j + 8` in its high nibble. The reader moves the nibbles into that order and copies
//! the scale bytes as they are; the tensor scale becomes the f32 the GGUF file keeps beside the
//! weight as `<stem>.scale`. A routed stack ([`experts`]) has one tensor scale per expert, so its
//! `.scale` holds one value per expert, in expert order.
//!
//! A scale byte UE4M3 does not define as a scale (bit 7 set, or `0x7F`) is refused by name, as the
//! block's decode refuses it, rather than stripped of bit 7. `<stem>.input_scale`, the activation
//! scale ModelOpt calibrates for 4-bit activations, has no place in the block; the reader leaves it.

use gguf::quant::{QuantError, ue4m3_to_f32_half};

use crate::safetensors::{self, Dtype, Raw};

/// Why a ModelOpt tensor was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Safetensors(#[from] safetensors::Error),
    #[error("modelopt nvfp4: {name}: {what}")]
    Shape { name: String, what: String },
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
pub struct Nvfp4 {
    /// The NVFP4 blocks, row-major, a stack expert by expert: `rows · k / 64` blocks of 36 bytes
    /// an expert.
    pub blocks: Vec<u8>,
    /// ggml's dims: `[k, rows]`, or `[k, rows, n_expert]` for a stack.
    pub dims: Vec<u64>,
    /// The `<stem>.scale` values: the tensor scale, or one per expert for a stack.
    pub scale: Vec<f32>,
}

/// The NVFP4 linear `stem`, its tensors looked up by name through `get`.
///
/// # Errors
///
/// A tensor `get` refuses or lacks, or whose bytes do not fill its shape; a weight that is not U8
/// `[rows, k/2]` with `k` a multiple of 64, a scale that is not F8_E4M3 `[rows, k/16]`, a tensor
/// scale that is not one finite F32, and a scale byte that is no UE4M3 scale, each by name.
pub fn linear<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    stem: &str,
) -> Result<Nvfp4, Error> {
    let mut out = Nvfp4::default();
    let [rows, k] = append(&get, stem, &mut out)?;
    out.dims = vec![k as u64, rows as u64];
    Ok(out)
}

/// A routed stack: the NVFP4 linears `stems`, one an expert in expert order, as one tensor of
/// `[k, rows, n_expert]` with one tensor scale an expert.
///
/// # Errors
///
/// As [`linear`] for each expert; an empty stack, and an expert whose shape differs from the
/// first's, by name.
pub fn experts<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    stems: &[impl AsRef<str>],
) -> Result<Nvfp4, Error> {
    let first = stems.first().ok_or_else(|| Error::Shape {
        name: "routed stack".into(),
        what: "no expert".into(),
    })?;
    let mut out = Nvfp4::default();
    let shape = append(&get, first.as_ref(), &mut out)?;
    for stem in &stems[1..] {
        let stem = stem.as_ref();
        let have = append(&get, stem, &mut out)?;
        if have != shape {
            return Err(Error::Shape {
                name: stem.to_string(),
                what: format!(
                    "rows and k {have:?}; the stack's first expert {} has {shape:?}",
                    first.as_ref()
                ),
            });
        }
    }
    let [rows, k] = shape;
    out.dims = vec![k as u64, rows as u64, stems.len() as u64];
    Ok(out)
}

/// Reads the triple `stem` and appends its blocks and tensor scale to `out`. Returns `[rows, k]`.
fn append<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    stem: &str,
    out: &mut Nvfp4,
) -> Result<[usize; 2], Error> {
    let typed = |suffix: &str, want: Dtype| -> Result<(String, Raw<'a>), Error> {
        let name = format!("{stem}.{suffix}");
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
    let (wname, w) = typed("weight", Dtype::U8)?;
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
    let (sname, s) = typed("weight_scale", Dtype::F8E4M3)?;
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
    let (tname, t) = typed("weight_scale_2", Dtype::F32)?;
    let scale = <[u8; 4]>::try_from(t.bytes)
        .map(f32::from_le_bytes)
        .map_err(|_| shape(&tname, format!("shape {:?}, not one value", t.shape)))?;
    if !scale.is_finite() {
        return Err(shape(&tname, format!("{scale} is not a finite scale")));
    }

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
                ue4m3_to_f32_half(byte).map_err(|source| Error::Scale {
                    name: sname.clone(),
                    row,
                    group: 4 * block + sub,
                    source,
                })?;
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
            let t = linear(|n| st.raw(n), stem).unwrap();
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
        let t = experts(|n| st.raw(n), &STEMS).unwrap();
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
            linear(|n| st.raw(n), "t").unwrap()
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
            let e = linear(|n| st.raw(n), "t").unwrap_err();
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
        let two = [0u8; 8];
        let codes = [0u8; 48];
        let s3 = [0x38u8; 6];
        let s4 = [0x38u8; 4];
        let shape = |list: &[(&str, &str, &[usize], &[u8])], name: &str| {
            let st = Safetensors::parse(write(list)).unwrap();
            let e = linear(|n| st.raw(n), "t").unwrap_err();
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
        for (shape2, bytes) in [(&[2][..], &two[..]), (&[][..], &nan[..])] {
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
            let e = linear(
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
        let e = linear(|n| st.raw(n), "t").unwrap_err();
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
        let e = linear(|n| st.raw(n), "t").unwrap_err();
        assert!(
            matches!(&e, Error::Safetensors(safetensors::Error::Missing(n)) if n == "t.weight_scale"),
            "{e:?}"
        );
        // A stack of no expert, and one whose second expert has other rows.
        let st = fixture();
        let none: [&str; 0] = [];
        assert!(matches!(
            experts(|n| st.raw(n), &none),
            Err(Error::Shape { .. })
        ));
        let e = experts(|n| st.raw(n), &[STEMS[0], "synthetic"]).unwrap_err();
        assert!(
            matches!(&e, Error::Shape { name, .. } if name == "synthetic"),
            "{e:?}"
        );
    }
}
