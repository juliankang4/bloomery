//! One rank of N ([`RankOf`]) and its part of each tensor of a source: the indices the family's
//! rule (`model::arch::glm5next::tp::rank_span`) gives the rank ([`RankOf::take`]), which the
//! source (`model::part::Source`) reads from a GGUF file or a checkpoint folder alike.

use bloomery_gpu::GpuError;
use gguf::GgmlType;
use model::arch::glm5next::tp::rank_span;
use model::part::Take;
use model::placement::tp::Cut;

/// What the errors name.
const WHAT: &str = "glm5next tp source";

/// One rank of a tensor-parallel load: its index among the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankOf {
    rank: usize,
    ranks: usize,
}

impl RankOf {
    /// Rank `rank` of `ranks`; no rank, or a rank past them, is refused by name.
    pub fn new(rank: usize, ranks: usize) -> Result<RankOf, GpuError> {
        model::placement::tp::check_rank(WHAT, rank, ranks)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        Ok(RankOf { rank, ranks })
    }

    /// This rank's part of `v`, `what`'s count over the ranks, by the plan's rule for a cut of
    /// one-value units (`Cut::part`): refused by name when the ranks do not divide it or
    /// outnumber it.
    pub fn part(self, what: &str, v: usize) -> Result<usize, GpuError> {
        Cut { dim: 0, unit: 1 }
            .part(what, &[v as u64], [1, 1], self.ranks)
            .map(|p| p as usize)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))
    }

    /// What the rank reads of the glm5next tensor `name` of card type `ty` and `dims` over
    /// `heads` heads: the family's rule (`model::arch::glm5next::tp::rank_span`), refused by name.
    pub(crate) fn take(
        self,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        heads: u64,
    ) -> Result<Take, GpuError> {
        Ok(
            match rank_span(name, ty, dims, heads, self.rank, self.ranks)
                .map_err(|e| GpuError::shape(WHAT, e.to_string()))?
            {
                None => Take::Whole,
                Some((dim, at)) => Take::Part(dim, at),
            },
        )
    }

    #[must_use]
    pub fn rank(self) -> usize {
        self.rank
    }

    #[must_use]
    pub fn ranks(self) -> usize {
        self.ranks
    }
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, RankOf, Take, WHAT};

    /// A rank takes the family's span of a split tensor (`rank_span`, whose test holds the
    /// rule) and the whole of a kept one; no rank and a rank past them are refused, and a count
    /// the ranks do not divide or outnumber is refused by name.
    #[test]
    fn a_rank_takes_the_family_s_span() {
        let at = |r, n| RankOf::new(r, n).unwrap();
        let up = [4096u64, 512, 288];
        let take = |r, n, name| at(r, n).take(name, GgmlType::NVFP4, &up, 64).unwrap();
        assert_eq!(
            take(1, 4, "blk.3.ffn_up_exps.weight"),
            Take::Part(1, 128..256)
        );
        assert_eq!(take(0, 1, "blk.3.ffn_up_exps.weight"), Take::Whole);
        assert!(
            at(0, 2)
                .take("blk.3.attn_sinks.weight", GgmlType::F32, &[64], 64)
                .is_err()
        );
        for (rank, ranks) in [(0, 0), (2, 2), (usize::MAX, 4)] {
            let e = RankOf::new(rank, ranks).unwrap_err().to_string();
            assert!(
                e.contains(&format!(
                    "tp: {WHAT}: rank {rank} is not one of {ranks} ranks"
                )),
                "{e}"
            );
        }
        assert_eq!(at(1, 4).part("heads", 64).unwrap(), 16);
        assert!(at(0, 4).part("heads", 6).is_err() && at(0, 4).part("heads", 2).is_err());
    }

    /// The card's bytes of each buffer of `w`, in allocation order.
    #[cfg(feature = "hf")]
    fn device(gpu: &bloomery_gpu::Gpu, w: &bloomery_gpu::weights::DevWeight) -> Vec<Vec<u8>> {
        use bloomery_gpu::weights::DevWeight;
        let s = gpu.stream();
        let words = |b: &cuda_core::DeviceBuffer<u32>| -> Vec<u8> {
            b.to_host_vec(s)
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
        let floats = |b: &cuda_core::DeviceBuffer<f32>| -> Vec<u8> {
            b.to_host_vec(s)
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
        match w {
            DevWeight::Nvfp4Stack { qs, e, scales, .. } => {
                vec![words(qs.buf()), words(e.buf()), floats(scales.buf())]
            }
            DevWeight::Mxfp8 { qs, e, .. } | DevWeight::Fp8B128 { qs, e, .. } => {
                vec![words(qs.buf()), words(e.buf())]
            }
            DevWeight::F32 { w, .. } => vec![floats(w.buf())],
            _ => panic!("no reader for this weight's buffers"),
        }
    }

    /// A checkpoint folder at `dir` of one file holding `tensors` (name, dtype, shape, bytes).
    #[cfg(feature = "hf")]
    fn folder(dir: &std::path::Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
        use safetensors::{Folder, write};
        std::fs::create_dir_all(dir).unwrap();
        let entries: Vec<_> = tensors
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        std::fs::write(dir.join("model.safetensors"), write(&entries)).unwrap();
        let index: Vec<String> = tensors
            .iter()
            .map(|(n, ..)| format!("\"{n}\": \"model.safetensors\""))
            .collect();
        std::fs::write(
            dir.join(Folder::INDEX),
            format!("{{\"weight_map\": {{{}}}}}", index.join(", ")),
        )
        .unwrap();
    }

    /// A rank's parts of a checkpoint folder go up on the card and join there into the whole
    /// tensor the one-rank load holds: an NVFP4 routed stack cut along K with one tensor scale an
    /// expert on every rank, MXFP8 cut by row, FP8 with 128 by 128 block scales cut along K, a
    /// BF16 head cut by vocabulary row, and an F32 norm whole on every rank. The token embedding
    /// and a tensor of a layer past the trunk's are not loaded. A part no upload takes is refused
    /// by name before any part is read.
    #[test]
    #[cfg(feature = "hf")]
    #[ignore = "needs a CUDA device; `just gate-gpu-glm5next-tp` runs it on the box"]
    fn hw_a_folder_s_rank_parts_join_into_its_whole_on_the_card() {
        use std::collections::BTreeMap;

        use bloomery_gpu::weights::{TensorPart, Weights};
        use hf::modelopt::{Plain, Use};
        use hf::quant_config::{Format, Placed, Plan};
        use model_hf::part::{Checkpoint, Made};
        use safetensors::Folder;

        let dir = std::env::temp_dir().join(format!("tp-folder-{}", std::process::id()));
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let n = |len: usize, f: &dyn Fn(usize) -> u8| (0..len).map(f).collect::<Vec<u8>>();
        let e4m3 = |i: usize| (i * 7 % 0x7e) as u8;
        let mut t: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
            ("up.w".into(), "F8_E4M3", vec![512, 256], n(131072, &e4m3)),
            (
                "up.s".into(),
                "U8",
                vec![512, 8],
                n(4096, &|i| 120 + (i % 9) as u8),
            ),
            (
                "down.w".into(),
                "F8_E4M3",
                vec![256, 1024],
                n(262144, &|i| e4m3(i + 3)),
            ),
            (
                "down.s".into(),
                "F32",
                vec![2, 8],
                f32s(&[
                    0.5, 0.25, 2.0, 1.0, 0.125, 4.0, 1.5, 0.75, 0.25, 2.0, 0.5, 4.0, 1.0, 0.125,
                    0.75, 1.5,
                ]),
            ),
            (
                "head.w".into(),
                "BF16",
                vec![64, 256],
                n(32768, &|i| {
                    if i % 2 == 0 {
                        (i / 2 % 251) as u8
                    } else {
                        0x3f
                    }
                }),
            ),
            (
                "norm.w".into(),
                "F32",
                vec![256],
                f32s(&(0..256).map(|i| i as f32).collect::<Vec<_>>()),
            ),
        ];
        for x in 0..2 {
            t.push((
                format!("x{x}.w"),
                "U8",
                vec![256, 256],
                n(65536, &|i| (i * 37 + x) as u8),
            ));
            t.push((
                format!("x{x}.s"),
                "F8_E4M3",
                vec![256, 32],
                n(8192, &|i| 0x30 + (i % 13) as u8),
            ));
            t.push((
                format!("x{x}.t"),
                "F32",
                vec![],
                f32s(&[0.25 * (x + 1) as f32]),
            ));
        }
        folder(&dir, &t);
        let place = |format, companions: &[&str]| Placed {
            format,
            companions: companions.iter().map(|c| c.to_string()).collect(),
        };
        let placed = BTreeMap::from([
            ("up.w".to_string(), place(Format::Mxfp8, &["up.s"])),
            (
                "down.w".into(),
                place(
                    Format::Fp8Block {
                        rows: 128,
                        cols: 128,
                    },
                    &["down.s"],
                ),
            ),
            ("head.w".into(), place(Format::Bf16, &[])),
            ("norm.w".into(), place(Format::F32, &[])),
            ("x0.w".into(), place(Format::Nvfp4, &["x0.s", "x0.t"])),
            ("x1.w".into(), place(Format::Nvfp4, &["x1.s", "x1.t"])),
        ]);
        let made = |sources: &[&str], plain, dims: &[u64]| Made {
            sources: sources.iter().map(|s| s.to_string()).collect(),
            role: Use::Matrix(plain),
            dims: dims.to_vec(),
            step: None,
        };
        let tensors: [(&str, Made, Option<usize>); 5] = [
            (
                "blk.0.ffn_down_exps.weight",
                made(&["x0.w", "x1.w"], Plain::F32, &[512, 256, 2]),
                Some(0),
            ),
            (
                "blk.0.ffn_up.weight",
                made(&["up.w"], Plain::F32, &[256, 512]),
                Some(1),
            ),
            (
                "blk.0.ffn_down.weight",
                made(&["down.w"], Plain::F32, &[1024, 256]),
                Some(0),
            ),
            (
                "output.weight",
                made(&["head.w"], Plain::Bf16, &[256, 64]),
                Some(1),
            ),
            (
                "blk.0.attn_norm.weight",
                made(&["norm.w"], Plain::F32, &[256]),
                None,
            ),
        ];
        let cuts: Vec<(String, Vec<u64>, Option<usize>)> = tensors
            .iter()
            .map(|(n, m, c)| (n.to_string(), m.dims.clone(), *c))
            .collect();
        let plan = Plan {
            placed,
            ..Plan::default()
        };
        let table = tensors
            .into_iter()
            .map(|(n, m, _)| (n.to_string(), m))
            .chain([
                (
                    "token_embd.weight".into(),
                    made(&["head.w"], Plain::Bf16, &[256, 64]),
                ),
                (
                    "blk.1.nextn.enorm.weight".into(),
                    made(&["norm.w"], Plain::F32, &[256]),
                ),
            ])
            .collect();
        let c = Checkpoint::new(Folder::open(&dir).unwrap(), plan, table).unwrap();
        let gpu = bloomery_gpu::Gpu::new().unwrap();
        let load = |rank, ranks| {
            super::super::load(&gpu, &c, RankOf::new(rank, ranks).unwrap(), 1, 64).unwrap()
        };
        let off_card = ["token_embd.weight", "blk.1.nextn.enorm.weight"];
        let whole = load(0, 1);
        assert!(off_card.iter().all(|n| whole.get(n).is_none()));
        for ranks in [1, 2, 4] {
            let budget = model::placement::file_card_format("output.weight", GgmlType::BF16, 256)
                .unwrap()
                .resident_bytes(GgmlType::BF16, 256, 64 / ranks as u64)
                .unwrap();
            let w = load(0, ranks);
            assert!(off_card.iter().all(|n| w.get(n).is_none()));
            let allocated: usize = device(&gpu, w.get("output.weight").unwrap())
                .iter()
                .map(Vec::len)
                .sum();
            assert_eq!(allocated as u64, budget, "BF16 rank 0 of {ranks}");
            // PIN(2026-10-22): K=256, 64 BF16 head rows occupy MXFP8 codes and scales.
            assert_eq!(budget, 16_896 / ranks as u64);
        }
        for ranks in [2, 4] {
            let parts: Vec<_> = (0..ranks).map(|rank| load(rank, ranks)).collect();
            for (name, dims, cut) in &cuts {
                let w = device(&gpu, whole.get(name).unwrap());
                let p: Vec<_> = parts
                    .iter()
                    .map(|r| device(&gpu, r.get(name).unwrap()))
                    .collect();
                for (b, whole) in w.iter().enumerate() {
                    let stack_scales = dims.len() == 3 && b == 2;
                    let outer = match cut {
                        _ if stack_scales => 0,
                        None => 0,
                        Some(0) => (dims[1] * dims.get(2).copied().unwrap_or(1)) as usize,
                        _ => dims.get(2).copied().unwrap_or(1) as usize,
                    };
                    if outer == 0 {
                        assert!(
                            p.iter().all(|r| r[b] == *whole),
                            "{name} buffer {b} whole on each rank"
                        );
                        continue;
                    }
                    let chunks = |v: &[u8]| {
                        v.chunks(v.len() / outer)
                            .map(<[u8]>::to_vec)
                            .collect::<Vec<_>>()
                    };
                    let rank_chunks: Vec<_> = p.iter().map(|r| chunks(&r[b])).collect();
                    let joined: Vec<Vec<u8>> = (0..outer)
                        .map(|row| {
                            rank_chunks
                                .iter()
                                .flat_map(|r| r[row].iter().copied())
                                .collect()
                        })
                        .collect();
                    assert_eq!(joined, chunks(whole), "{ranks} ranks: {name} buffer {b}");
                    assert!(
                        p.windows(2).all(|pair| pair[0][b] != pair[1][b]),
                        "{name} buffer {b}: the ranks hold distinct parts"
                    );
                }
            }
            let scales = &device(&gpu, parts[1].get("blk.0.ffn_down_exps.weight").unwrap())[2];
            assert_eq!(scales, &f32s(&[0.25, 0.5]));
        }
        let bf16 = |nan: bool| -> Vec<u8> {
            (0..64)
                .flat_map(|i| u16::to_le_bytes(if nan && i == 3 { 0x7fc0 } else { 0x3f80 }))
                .collect()
        };
        let bad = dir.join("bad");
        folder(
            &bad,
            &[
                ("a.w".into(), "BF16", vec![64], bf16(true)),
                ("z.w".into(), "BF16", vec![64], bf16(false)),
            ],
        );
        let plan = Plan {
            placed: BTreeMap::from([
                ("a.w".to_string(), place(Format::Bf16, &[])),
                ("z.w".into(), place(Format::Bf16, &[])),
            ]),
            ..Plan::default()
        };
        let table = BTreeMap::from([
            ("a.weight".to_string(), made(&["a.w"], Plain::F32, &[64])),
            ("derived.z".into(), made(&["z.w"], Plain::F32, &[64])),
        ]);
        let c = Checkpoint::new(Folder::open(&bad).unwrap(), plan, table).unwrap();
        let e = super::super::load(&gpu, &c, RankOf::new(0, 1).unwrap(), 1, 64)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("\"derived.z\" has a"), "{e}");

        let ok = vec![0u8; 128];
        let mut bad = vec![0u8; 132];
        bad[0] = 0xff;
        let info = |name: &str, ty, k, nbytes| gguf::TensorInfo {
            name: name.into(),
            dims: vec![k, 1],
            ty,
            offset: 0,
            nbytes,
        };
        let mut w = Weights::default();
        let e = w
            .add_parts(
                gpu.stream(),
                &[
                    TensorPart {
                        info: info("a.weight", GgmlType::F32, 32, 128),
                        spans: vec![&ok],
                        scale: None,
                    },
                    TensorPart {
                        info: info("b.weight", GgmlType::MXFP8, 128, 132),
                        spans: vec![&bad],
                        scale: None,
                    },
                ],
            )
            .unwrap_err();
        let e = e.to_string();
        assert!(e.contains("b.weight") && e.contains("row 0"), "{e}");
        assert_eq!(w.names().count(), 0, "a failed group publishes no parts");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
