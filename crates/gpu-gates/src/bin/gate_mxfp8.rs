//! GPU gate for bloomery's E4M3 weights: MXFP8 (`GgmlType::MXFP8`: OCP MX
//! blocks of 32 F8_E4M3 values and one E8M0 scale) and FP8_B128
//! (`GgmlType::FP8_B128`: 128 F8_E4M3 values and one f32 scale, the
//! checkpoints' 128 by 128 block scale on each of its rows). For each type,
//! the dense gemv (`Q8F32Kernels::enqueue_e4m3_gemv`, through `site::gemv`)
//! and the wide tile (`Gemm32Kernels::enqueue_e4m3_tile`, through
//! `site::gemm`) at three
//! projection shapes of a 4,096-wide model, `attn_q_b` (16,384 rows × 1,536),
//! `attn_output` (4,096 × 16,384) and `ssm_f_b` (8,192 × 128), on synthetic
//! tensors written to a GGUF and loaded through the engine's own path: a split
//! sizing the type by its row, `Weights::load_where` (the
//! `CardFormat::Mxfp8Planes` or `Fp8B128Planes` upload and its repack),
//! `site::file_site` and `site::site`. Every clause line names its type
//! (`mxfp8` or `fp8_b128`).
//!
//! The projection tensors: every MXFP8 scale byte drawn from 112..=136
//! (`2^-15` to `2^9`, so every nonzero product of a value and its scale is a
//! normal f32), every FP8_B128 scale an f32 of the same binades with all 23
//! fraction bits drawn (so the product of a value and its scale rounds, once,
//! as the host dequant rounds it), every value byte from the 254 bytes that
//! hold an F8_E4M3 value. Activations uniform on ±2.
//! For each shape at m = 1 and at m > 1 columns, every output is checked:
//! - `written`: finite, over a NaN sentinel the launch must overwrite;
//! - `band`: within the derived band of the reference `Σ_j w_j x_j` in f64,
//!   `w` the host dequant (`gguf::quant::dequant_row`);
//! - `bits`: equal bit for bit to the host rule `runtime::hc_gated::card::q8_dot`
//!   over the host dequant: lane `L` visits four-value words `32i + L` in
//!   increasing `i`, four fused multiply-adds in byte order per word, then the
//!   warp butterfly.
//!
//! At one column each shape also runs through `site::resident_gemv`
//! (`decode`), the launch by resident form that glm5next's decode step runs
//! for every projection (`gpu-glm5next`'s `body::gemv`: the latent layer's
//! `attn_q_b` and `attn_output`, here at their shapes): the same checks.
//!
//! The wide tile, at 64 columns on every shape: `written`, `band` as above,
//! and `bits` equal to the F32 tile's rule `card::f32_dot` over the host
//! dequant (lane `L` the values `32i + L` in increasing `i`, one fused
//! multiply-add each, the butterfly), since its values are decoded exactly as
//! they are staged.
//!
//! Three 8-row, 128-value tensors of each type also exercise the scale's
//! ends through the gemv at one and three columns and the tile at 64 columns:
//! MXFP8's scale bytes E=0, E=1 and E=254, whose values include the smallest
//! E4M3 subnormal and, at E=254, the largest finite accepted weight,
//! `1.875 * 2^127` (byte 0x3F); FP8_B128's smallest f32 subnormal scale
//! `2^-149`, smallest normal scale `2^-126` and a scale of `2^118`, whose
//! values include the smallest E4M3 subnormal and the largest, ±448, whose
//! product `1.75 * 2^126` stays finite. Activations scaled by `2^64` at the
//! small scales and `2^-100` at the large keep sums finite. They use the
//! same `written`, `band` and `bits` checks.
//!
//! The shared paths the E4M3 entries reuse keep their contracts, bit for
//! bit and within the same band, on tensors of the same file: the Q8_0 site
//! gemv at one and three columns against the same byte-order rule (`q8_0`, at
//! `attn_q_b`'s shape), and the F32 site at 64 columns, which runs the F32
//! tile, against the F32 tile's rule (`f32_tile`, 256 rows of 1,536).
//!
//! Parts of each type joined into one weight at load (`Weights::join_rows`,
//! as a model joins its projections) run through `site::resident_gemv`, bit
//! for bit the host walk over the parts' rows (`join`).
//!
//! The expert select (`sel`, through `site::resident_sel_f32`, the MTP
//! draft's routed launch): a stack of 4 experts of 64 rows of 256 values in
//! FP8_B128, and the same stack's shape in Q8_0 (the shared entry), at six
//! slots over three columns, two slots a column: every slot's rows bit for
//! bit the byte-order rule over its expert's host rows, and within the band.
//! An id past the stack writes its slot NaN and raises `ExpertId`, the other
//! slot keeping its bits; an MXFP8 stack is refused by name.
//!
//! And the load's refusals, each naming the tensor: an MXFP8 scale byte 0xFF
//! (E8M0's NaN), a value byte 0x7F (E4M3's NaN) of either type, an MXFP8 row
//! of 96 values (no whole scale word), an FP8_B128 scale that is not finite,
//! and an FP8_B128 scale a value times leaves f32's range. Both launchers
//! refuse malformed planes of either type with one diagnostic (`plane-shape`:
//! K misaligned, a wrong scale-row count, a wrong scale width), and the tile
//! a column count whose sizes overflow.
//!
//! The band. Every term `w_j · x_j` enters its lane by one fused multiply-add
//! of `w_j = v_j · d` rounded once to f32, as the host dequant `w` the band is
//! taken over rounds it (exact for MXFP8, a 4-bit significand times a power
//! of two), so a lane's sum has `k / 32` roundings and the butterfly five more:
//! `|ours − ref| ≤ γ(k/32 + 5) · Σ_j |w_j x_j|`, plus the f64 reference's own
//! `k · u64` of the same sum. The bound is the worst case, every error
//! aligned.
//!
//! `--bench-kernels` runs no case: it prices the gemv's launch at each shape
//! (`bench op=mxfp8_<rows>`, one column) beside the Q8_0 gemv on the same
//! rows (`bench op=q8_0_<rows>`, 34 bytes a 32 values against MXFP8's 33),
//! through `shared/bench.rs`'s burst and replay, cycling weight copies toward
//! 512 MiB with at most 64 copies. The capped footprint alone does not establish
//! DRAM traffic. `--bench-arm <op>` keeps arms whose name starts with `<op>`.
//! Nothing there asserts on a result.

#[path = "shared/bench.rs"]
mod bench_arm;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mxfp8", gate::run())
}

mod gate {
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
    use bloomery_gpu::q8f32::{E4m3GemvArgs, E4m3Scales, E4m3Weight, SelArgs};
    use bloomery_gpu::site::{self, SiteTy, SiteWeight, WideIn, WideKernels};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_gates::rounding::{U64, gamma};
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, par_map, verdict, write_gguf};
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use runtime::hc_gated::card::{f32_dot as tile_rule, q8_dot as byte_rule};
    use std::path::Path;

    /// A projection: its tensor stem, rows, row width and the
    /// column counts its gemv runs at.
    pub(crate) struct Shape {
        pub(crate) stem: &'static str,
        pub(crate) rows: usize,
        pub(crate) k: usize,
        pub(crate) ms: [usize; 2],
        pub(crate) seed: u64,
    }

    /// A latent attention's `attn_q_b` and `attn_output` and a KDA layer's
    /// `ssm_f_b` (`dims` 1,536 × 16,384, 16,384 × 4,096 and 128 × 8,192,
    /// `dims[0]` the row width): a short row, the row a lane walks in whole
    /// trips, and the one-word-a-lane row.
    pub(crate) const SHAPES: [Shape; 3] = [
        Shape {
            stem: "blk.3.attn_q_b",
            rows: 16_384,
            k: 1536,
            ms: [1, 3],
            seed: 0x5eed_0801,
        },
        Shape {
            stem: "blk.3.attn_output",
            rows: 4096,
            k: 16_384,
            ms: [1, 8],
            seed: 0x5eed_0802,
        },
        Shape {
            stem: "blk.4.ssm_f_b",
            rows: 8192,
            k: 128,
            ms: [1, 2],
            seed: 0x5eed_0803,
        },
    ];

    /// Columns of the wide tile.
    const WIDE: usize = 64;

    /// `rows` MXFP8 rows of `k` values drawn from `seed`: each block's scale
    /// byte uniform on 112..=136, its 32 value bytes uniform over the 254
    /// that hold a value.
    pub(crate) fn mxfp8_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let values = e4m3_values();
        let blocks = rows * k / 32;
        let mut out = Vec::with_capacity(blocks * 33);
        for _ in 0..blocks {
            out.push(112 + rng.below(25) as u8);
            for _ in 0..32 {
                out.push(values[rng.below(values.len())]);
            }
        }
        out
    }

    /// The 254 bytes that hold an F8_E4M3 value.
    fn e4m3_values() -> Vec<u8> {
        (0..=u8::MAX).filter(|b| b & 0x7F != 0x7F).collect()
    }

    /// `rows` FP8_B128 rows of `k` values drawn from `seed`: each block's
    /// f32 scale `2^e · (1 + f / 2^23)`, `e` uniform on -15..=9 and the 23
    /// fraction bits `f` uniform, its 128 value bytes uniform over the 254
    /// that hold a value.
    fn fp8_b128_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let values = e4m3_values();
        let blocks = rows * k / 128;
        let mut out = Vec::with_capacity(blocks * 132);
        for _ in 0..blocks {
            let bits = (112 + rng.below(25) as u32) << 23 | rng.below(1 << 23) as u32;
            out.extend(bits.to_le_bytes());
            for _ in 0..128 {
                out.push(values[rng.below(values.len())]);
            }
        }
        out
    }

    /// `rows` rows of `k` values of the E4M3 type `ty` drawn from `seed`.
    fn e4m3_bytes(ty: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        if ty == GgmlType::MXFP8 {
            mxfp8_bytes(rows, k, seed)
        } else {
            fp8_b128_bytes(rows, k, seed)
        }
    }

    /// One tensor of a synthetic file: name, dims, type, bytes.
    type Decl = (String, Vec<u64>, GgmlType, Vec<u8>);

    /// A weight of type `ty` of `rows` rows of `k` as a tensor of a file.
    fn decl(name: &str, ty: GgmlType, (rows, k): (usize, usize), bytes: Vec<u8>) -> Decl {
        (name.to_string(), vec![k as u64, rows as u64], ty, bytes)
    }

    /// The two E4M3 types and the tag their clause lines carry.
    const TYPES: [(GgmlType, &str); 2] =
        [(GgmlType::MXFP8, "mxfp8"), (GgmlType::FP8_B128, "fp8_b128")];

    /// The scale kind of the E4M3 type `ty`.
    fn scales_of(ty: GgmlType) -> E4m3Scales {
        if ty == GgmlType::MXFP8 {
            E4m3Scales::Mxfp8
        } else {
            E4m3Scales::Fp8B128
        }
    }

    /// The name of shape `s`'s tensor of the type tagged `tag`.
    fn shape_name(s: &Shape, tag: &str) -> String {
        format!("{}.{tag}.weight", s.stem)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let bench = std::env::args().any(|a| a == "--bench-kernels");
        let only = if bench {
            bloomery_gpu_gates::bench_arm(
                std::env::args(),
                SHAPES
                    .iter()
                    .flat_map(|s| [format!("mxfp8_{}", s.rows), format!("q8_0_{}", s.rows)]),
            )?
        } else {
            None
        };
        let gpu = Gpu::new()?;
        if bench {
            return bench::run(&gpu, only.as_deref());
        }
        println!("mxfp8 device={}", gpu.device_name()?);
        let dir = std::env::temp_dir().join(format!("bloomery-mxfp8-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome = cases(&gpu, &dir);
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        let ok = outcome?;
        cleanup?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The Q8_0 and F32 tensors beside the MXFP8 ones: `attn_q_b`'s shape in
    /// Q8_0, and an F32 weight of 256 rows of 1,536.
    const Q8_0_NAME: &str = "blk.3.indexer.attn_q_b.weight";
    const F32_NAME: &str = "blk.3.f32.weight";
    const F32_ROWS: usize = 256;

    /// `rows` Q8_0 rows of `k` values drawn from `seed`: f16 scales of
    /// `2^-8` to `2^1`, codes uniform over all 256 bytes.
    fn q8_0_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let mut out = Vec::with_capacity(rows * k / 32 * 34);
        for _ in 0..rows * k / 32 {
            let bits = ((7 + rng.below(9)) << 10 | rng.below(1024)) as u16;
            out.extend(bits.to_le_bytes());
            out.extend((0..32).map(|_| rng.below(256) as u8));
        }
        out
    }

    /// The expert-select stacks' experts, rows an expert, row width and
    /// slots a column, and the slots' ids.
    const SEL: (usize, usize, usize, usize) = (4, 64, 256, 2);
    const SEL_IDS: [u32; 6] = [2, 0, 3, 3, 1, 0];

    /// The expert-select stacks: FP8_B128 and Q8_0, `[k, rows, experts]`.
    const SEL_FP8: &str = "blk.0.ffn_gate_exps.fp8_b128.weight";
    const SEL_Q8_0: &str = "blk.0.ffn_gate_exps.q8_0.weight";

    /// The 8-row, 128-value tensors at the ends of a type's scale: name
    /// stem, block scale bytes, activation scale (module doc).
    fn extremes(ty: GgmlType) -> Vec<(&'static str, Vec<u8>, f32)> {
        if ty == GgmlType::MXFP8 {
            return [
                ("scale_e0", 0u8, 2.0f32.powi(64)),
                ("scale_e1", 1, 2.0f32.powi(64)),
                ("scale_e254", 254, 2.0f32.powi(-100)),
            ]
            .map(|(stem, e, x)| (stem, vec![e], x))
            .to_vec();
        }
        [
            ("scale_sub", f32::from_bits(1), 2.0f32.powi(64)),
            ("scale_min", f32::MIN_POSITIVE, 2.0f32.powi(64)),
            ("scale_2p118", 2.0f32.powi(118), 2.0f32.powi(-100)),
        ]
        .map(|(stem, d, x)| (stem, d.to_le_bytes().to_vec(), x))
        .to_vec()
    }

    fn cases(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
        let path = dir.join("mxfp8.gguf");
        let mut decls = Vec::new();
        for (ty, tag) in TYPES {
            for s in &SHAPES {
                let bytes = e4m3_bytes(ty, s.rows, s.k, s.seed);
                decls.push(decl(&shape_name(s, tag), ty, (s.rows, s.k), bytes));
            }
        }
        let (q_rows, k) = (SHAPES[0].rows, SHAPES[0].k);
        let q8 = q8_0_bytes(q_rows, k, 0x5eed_0810);
        decls.push(decl(Q8_0_NAME, GgmlType::Q8_0, (q_rows, k), q8));
        let mut rng = GateRng(0x5eed_0820);
        let f32s: Vec<u8> = (0..F32_ROWS * k)
            .flat_map(|_| rng.uni(-1.0, 1.0).to_le_bytes())
            .collect();
        decls.push(decl(F32_NAME, GgmlType::F32, (F32_ROWS, k), f32s));
        // The refusals' tensors: 8 rows of 256 unless named.
        let mx = GgmlType::MXFP8;
        let mut nan_scale = mxfp8_bytes(8, 256, 7);
        nan_scale[9 * 33] = 0xFF;
        decls.push(decl("nan_scale.weight", mx, (8, 256), nan_scale));
        let mut nan_value = mxfp8_bytes(8, 256, 8);
        nan_value[9 * 33 + 5] = 0x7F;
        decls.push(decl("nan_value.weight", mx, (8, 256), nan_value));
        decls.push(decl("k96.weight", mx, (8, 96), mxfp8_bytes(8, 96, 9)));
        let b128 = GgmlType::FP8_B128;
        let mut inf_scale = fp8_b128_bytes(8, 256, 10);
        inf_scale[5 * 132..][..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        decls.push(decl("b128_inf_scale.weight", b128, (8, 256), inf_scale));
        let mut nan_value = fp8_b128_bytes(8, 256, 11);
        nan_value[5 * 132 + 4 + 9] = 0xFF;
        decls.push(decl("b128_nan_value.weight", b128, (8, 256), nan_value));
        let mut past_f32 = fp8_b128_bytes(8, 256, 12);
        past_f32[5 * 132..][..4].copy_from_slice(&2.0f32.powi(127).to_le_bytes());
        past_f32[5 * 132 + 4 + 3] = 0x7E;
        decls.push(decl("b128_past_f32.weight", b128, (8, 256), past_f32));
        let mut joined = Vec::new();
        for (ty, tag) in TYPES {
            let parts = [(96, 0x5eed_0830), (32, 0x5eed_0831)].map(|(rows, seed)| {
                let name = format!("join_{tag}_{rows}.weight");
                let bytes = e4m3_bytes(ty, rows, 256, seed);
                decls.push(decl(&name, ty, (rows, 256), bytes.clone()));
                (name, bytes)
            });
            joined.push((ty, tag, parts));
        }
        let mut extreme = Vec::new();
        for (ty, tag) in TYPES {
            let block = if ty == GgmlType::MXFP8 { 32 } else { 128 };
            let values: &[u8] = if ty == GgmlType::MXFP8 {
                &[
                    0x00, 0x01, 0x07, 0x08, 0x37, 0x38, 0x3F, 0x80, 0x81, 0x87, 0x88, 0xB7, 0xB8,
                    0xBF,
                ]
            } else {
                &[
                    0x00, 0x01, 0x07, 0x08, 0x37, 0x38, 0x7E, 0x80, 0x81, 0x87, 0x88, 0xB7, 0xB8,
                    0xFE,
                ]
            };
            for (stem, scale, x_scale) in extremes(ty) {
                let s = Shape {
                    stem,
                    rows: 8,
                    k: 128,
                    ms: [1, 3],
                    seed: 0x5eed_0840,
                };
                let mut bytes = Vec::new();
                for b in 0..s.rows * s.k / block {
                    bytes.extend_from_slice(&scale);
                    bytes.extend((0..block).map(|i| values[(b + i) % values.len()]));
                }
                decls.push(decl(&shape_name(&s, tag), ty, (s.rows, s.k), bytes.clone()));
                extreme.push((ty, tag, s, bytes, x_scale));
            }
        }
        let (n_exp, rpe, sel_k, _) = SEL;
        let stack = |name: &str, ty, bytes| {
            (
                name.to_string(),
                vec![sel_k as u64, rpe as u64, n_exp as u64],
                ty,
                bytes,
            )
        };
        let sel_fp8 = fp8_b128_bytes(n_exp * rpe, sel_k, 0x5eed_0850);
        let sel_q8 = q8_0_bytes(n_exp * rpe, sel_k, 0x5eed_0851);
        decls.push(stack(SEL_FP8, b128, sel_fp8.clone()));
        decls.push(stack(SEL_Q8_0, GgmlType::Q8_0, sel_q8.clone()));
        write_gguf(&path, &[], &decls)?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        let gemm = GemmKernels::load(gpu.context())?;
        let mut route = GemmRoute::new(gpu.stream(), WIDE, 1)?;
        gemm.enqueue_route_dense(gpu.stream(), WIDE, &mut route, gpu.unlabelled_sink())?;
        let wide = (&g32, &gemm, &route);
        let mut ok = true;
        for (i, (ty, tag)) in TYPES.into_iter().enumerate() {
            for (j, s) in SHAPES.iter().enumerate() {
                let bytes = &decls[i * SHAPES.len() + j].3;
                ok &= e4m3_case(gpu, wide, &split, (ty, tag, s), bytes, 1.0)?;
            }
        }
        for (ty, tag, s, bytes, x_scale) in &extreme {
            ok &= e4m3_case(gpu, wide, &split, (*ty, tag, s), bytes, *x_scale)?;
        }
        let at = TYPES.len() * SHAPES.len();
        ok &= q8_0_case(gpu, &g32, &split, &decls[at].3)?;
        ok &= f32_tile_case(gpu, &g32, &split, &decls[at + 1].3)?;
        for (tag, name, want) in [
            ("mxfp8", "nan_scale.weight", "E8M0's NaN"),
            ("mxfp8", "nan_value.weight", "byte 0x7f"),
            ("mxfp8", "k96.weight", "have no Mxfp8Planes layout"),
            ("fp8_b128", "b128_inf_scale.weight", "not finite"),
            ("fp8_b128", "b128_nan_value.weight", "byte 0xff"),
            ("fp8_b128", "b128_past_f32.weight", "leaves f32's range"),
        ] {
            let got = match Weights::load_where(gpu.stream(), &split, |n| n == name) {
                Ok(_) => "loaded".to_string(),
                Err(e) => e.to_string(),
            };
            let pass = got != "loaded" && got.contains(want) && got.contains(name);
            println!(
                "{tag} clause=refusal tensor={name} want={want:?} got={got:?} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        for (ty, tag, parts) in &joined {
            ok &= join_case(gpu, &split, (*ty, tag), parts)?;
        }
        ok &= sel_case(gpu, &split, (b128, SEL_FP8), &sel_fp8)?;
        ok &= sel_case(gpu, &split, (GgmlType::Q8_0, SEL_Q8_0), &sel_q8)?;
        ok &= sel_refusal(gpu, &split)?;
        for (ty, tag) in TYPES {
            ok &= plane_shapes(gpu, &g32, (scales_of(ty), tag))?;
        }
        Ok(ok)
    }

    /// The expert select over the stack `name` of type `ty` (`bytes` its
    /// file bytes) through `site::resident_sel_f32`: at [`SEL_IDS`] every
    /// slot's rows bit for bit the byte-order rule over its expert's host
    /// rows and within the band; at ids `[1, n_experts]` the second slot NaN
    /// with an `ExpertId` fault and the first slot's bits kept.
    fn sel_case(
        gpu: &Gpu,
        split: &Split,
        (ty, name): (GgmlType, &str),
        bytes: &[u8],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (n_exp, rpe, k, spc) = SEL;
        let w = Weights::load_where(stream, split, |n| n == name)?;
        let wf = host_rows(ty, bytes, n_exp * rpe, k);
        let tag = if ty == GgmlType::Q8_0 {
            "q8_0"
        } else {
            "fp8_b128"
        };
        let mut ok = true;
        for (ids, fault) in [
            (&SEL_IDS[..], None),
            (&[1, n_exp as u32][..], Some(FaultSite::ExpertId)),
        ] {
            let n_slots = ids.len();
            let m = n_slots.div_ceil(spc);
            let (x, x_dev, mut y_dev) = columns(gpu, (m, k, n_slots * rpe), 0x5eed_0852, 1.0)?;
            let sel = DeviceBuffer::from_host(stream, ids)?;
            let a = SelArgs {
                x: &x_dev,
                sel: &sel,
                n_slots,
                rows_per_expert: rpe,
                slots_per_col: spc,
            };
            gpu.clear_fault()?;
            site::resident_sel_f32(gpu, &w, name, &a, gpu.unlabelled_sink(), &mut y_dev)?;
            let got = gpu.take_fault()?;
            let y = y_dev.to_host_vec(stream)?;
            let want = fault.map(|f| Fault::at(LAYER_NONE, f));
            let shown = format!("ids={ids:?} rows_per_expert={rpe} k={k} slots_per_col={spc}");
            let pass = got == want;
            println!(
                "{tag} clause=sel.fault {shown} fault={got:?} want={want:?} {}",
                verdict(pass)
            );
            ok &= pass;
            for (slot, &id) in ids.iter().enumerate() {
                let y = &y[slot * rpe..][..rpe];
                let x = &x[slot / spc * k..][..k];
                let shown = format!("{shown} slot={slot} id={id}");
                let at = id as usize * rpe;
                let Some(w) = wf.get(at..at + rpe) else {
                    let nan = y.iter().all(|v| v.is_nan());
                    println!("{tag} clause=sel.bad_id {shown} nan={nan} {}", verdict(nan));
                    ok &= nan;
                    continue;
                };
                ok &= check((tag, "sel"), &shown, (y, x, k), w, &|r, _| {
                    byte_rule(&w[r], x)
                });
            }
        }
        Ok(ok)
    }

    /// `site::resident_sel_f32` refuses an MXFP8 stack by name.
    fn sel_refusal(gpu: &Gpu, split: &Split) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let name = shape_name(&SHAPES[2], "mxfp8");
        let w = Weights::load_where(stream, split, |n| n == name)?;
        let x = DeviceBuffer::<f32>::zeroed(stream, SHAPES[2].k)?;
        let sel = DeviceBuffer::from_host(stream, &[0u32])?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, 64)?;
        let a = SelArgs {
            x: &x,
            sel: &sel,
            n_slots: 1,
            rows_per_expert: 64,
            slots_per_col: 1,
        };
        let got = match site::resident_sel_f32(gpu, &w, &name, &a, gpu.unlabelled_sink(), &mut y) {
            Ok(()) => "launched".to_string(),
            Err(e) => e.to_string(),
        };
        let pass = got.contains(&name) && got.contains("a q8_0 or fp8_b128 stack");
        println!("mxfp8 clause=sel.refusal got={got:?} {}", verdict(pass));
        Ok(pass)
    }

    /// Both launchers refuse malformed planes of `scales` with one
    /// diagnostic, and the tile a column count whose sizes overflow.
    fn plane_shapes(
        gpu: &Gpu,
        g32: &Gemm32Kernels,
        (scales, tag): (E4m3Scales, &str),
    ) -> Result<bool, GateError> {
        let x = DeviceBuffer::<f32>::zeroed(gpu.stream(), 256)?;
        let mut y = DeviceBuffer::<f32>::zeroed(gpu.stream(), 8)?;
        let mut ok = true;
        for (case, words, scale_rows, scale_cols) in [
            ("k-alignment", 40, 8, 1),
            ("scale-rows", 32, 9, 1),
            ("scale-width", 32, 8, 2),
        ] {
            let qs = DeviceTensor::<u32>::zeroed(gpu.stream(), 8, words)?;
            let e = DeviceTensor::<u32>::zeroed(gpu.stream(), scale_rows, scale_cols)?;
            let w = E4m3Weight {
                qs: &qs,
                e: &e,
                scales,
            };
            let gemv = gpu.q8f32().enqueue_e4m3_gemv(
                gpu.stream(),
                E4m3GemvArgs {
                    w,
                    x: &x,
                    m: 1,
                    y: &mut y,
                },
            );
            let tile = g32.enqueue_e4m3_tile(gpu.stream(), w, &x, 1, &mut y);
            let errors = [gemv, tile].map(|r| match r {
                Ok(()) => "launched".to_owned(),
                Err(e) => e.to_string(),
            });
            let detail = |s: &str| s.split_once(": E4M3 planes: ").map(|(_, d)| d.to_owned());
            let pass = detail(&errors[0]).is_some() && detail(&errors[0]) == detail(&errors[1]);
            println!(
                "{tag} clause=plane-shape case={case} gemv={:?} tile={:?} {}",
                errors[0],
                errors[1],
                verdict(pass)
            );
            ok &= pass;
        }
        // 2^61 + 1 columns of 128 values over 8 rows: n·k and rows·n wrap to
        // 128 and 8 in 64 bits, which these buffers hold.
        let qs = DeviceTensor::<u32>::zeroed(gpu.stream(), 8, 32)?;
        let e = DeviceTensor::<u32>::zeroed(gpu.stream(), 8, 1)?;
        let w = E4m3Weight {
            qs: &qs,
            e: &e,
            scales,
        };
        let tile = g32.enqueue_e4m3_tile(gpu.stream(), w, &x, (1 << 61) + 1, &mut y);
        let pass = tile
            .as_ref()
            .is_err_and(|e| e.to_string().contains("overflows"));
        println!(
            "{tag} clause=plane-shape case=column-overflow tile={tile:?} {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// Parts of the E4M3 type `ty` joined into one weight, as a model joins
    /// its projections at load (`Weights::join_rows`), run through
    /// `site::resident_gemv`: every output bit for bit the host walk over the
    /// parts' rows one after another, and within the band.
    fn join_case(
        gpu: &Gpu,
        split: &Split,
        (ty, tag): (GgmlType, &str),
        parts: &[(String, Vec<u8>)],
    ) -> Result<bool, GateError> {
        const JOINT: &str = "derived.e4m3_joint";
        let stream = gpu.stream();
        let k = 256;
        let mut w = Weights::load_where(stream, split, |n| parts.iter().any(|p| p.0 == n))?;
        let names: Vec<&str> = parts.iter().map(|p| p.0.as_str()).collect();
        w.join_rows(stream, &names, JOINT.to_string())?;
        let bytes: Vec<u8> = parts.iter().flat_map(|p| p.1.iter().copied()).collect();
        let rows = bytes.len() / (k / 128 * 132);
        let wf = host_rows(ty, &bytes, rows, k);
        let (x, x_dev, mut y_dev) = columns(gpu, (1, k, rows), 0x5eed_0832, 1.0)?;
        site::resident_gemv(gpu, &w, JOINT, &x_dev, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let walk = |r: usize, c: usize| byte_rule(&wf[r], &x[c * k..][..k]);
        let shown = format!("parts={} rows={rows} k={k} m=1", parts.len());
        Ok(check((tag, "join"), &shown, (&y, &x, k), &wf, &walk))
    }

    /// The host rows of a file tensor's `bytes`: `rows` rows of `k` values
    /// of type `ty`.
    fn host_rows(ty: GgmlType, bytes: &[u8], rows: usize, k: usize) -> Vec<Vec<f32>> {
        let row_bytes = bytes.len() / rows;
        par_map(rows, |r| {
            let mut v = vec![0.0f32; k];
            dequant_row(ty, &bytes[r * row_bytes..][..row_bytes], &mut v)
                .unwrap_or_else(|e| panic!("row {r}: {e}"));
            v
        })
    }

    /// One launch's outputs (token-major, `rows` a column) against the host:
    /// `rule(r, c)` the bits the kernel must give for row `r` of column
    /// `c`, `w` the host rows for the band. Prints the clauses `written`,
    /// `band` and `bits` and returns whether all hold.
    fn check(
        (ty, clause): (&str, &str),
        tag: &str,
        (y, x, k): (&[f32], &[f32], usize),
        w: &[Vec<f32>],
        rule: &(dyn Fn(usize, usize) -> f32 + Sync),
    ) -> bool {
        let rows = w.len();
        let per: Vec<(bool, f64, f64)> = par_map(y.len(), |at| {
            let (c, r) = (at / rows, at % rows);
            let (sum, mag) =
                w[r].iter()
                    .zip(&x[c * k..][..k])
                    .fold((0.0f64, 0.0f64), |(s, m), (&a, &b)| {
                        let t = f64::from(a) * f64::from(b);
                        (s + t, m + t.abs())
                    });
            (
                y[at].to_bits() == rule(r, c).to_bits(),
                (f64::from(y[at]) - sum).abs(),
                (gamma(k / 32 + 5) + k as f64 * U64) * mag,
            )
        });
        let written = y.iter().all(|v| v.is_finite());
        let bits_n = per.iter().filter(|p| p.0).count();
        let band_ok = per.iter().all(|p| p.1 <= p.2);
        let worst = per
            .iter()
            .map(|p| if p.2 > 0.0 { p.1 / p.2 } else { p.1 })
            .fold(0.0f64, f64::max);
        println!("{ty} clause={clause}.written {tag} {}", verdict(written));
        println!(
            "{ty} clause={clause}.band {tag} worst_err_over_band={worst:.3e} {}",
            verdict(band_ok)
        );
        println!(
            "{ty} clause={clause}.bits {tag} equal={bits_n}/{} {}",
            per.len(),
            verdict(bits_n == per.len())
        );
        written && band_ok && bits_n == per.len()
    }

    /// A launch's activations on the host and on the card, and its output.
    type Columns = (Vec<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>);

    /// `m` columns of `k` activations uniform on ±2 from `seed`, multiplied
    /// by `scale`, on the card, and an output of `m * rows` NaN sentinels.
    fn columns(
        gpu: &Gpu,
        (m, k, rows): (usize, usize, usize),
        seed: u64,
        scale: f32,
    ) -> Result<Columns, GateError> {
        let mut rng = GateRng(seed);
        let x: Vec<f32> = (0..m * k).map(|_| rng.uni(-2.0, 2.0) * scale).collect();
        let x_dev = DeviceBuffer::from_host(gpu.stream(), &x)?;
        let y_dev = DeviceBuffer::from_host(gpu.stream(), &vec![f32::NAN; m * rows])?;
        Ok((x, x_dev, y_dev))
    }

    /// One shape of the E4M3 type `ty` at each of its column counts through `site::gemv`
    /// and at [`WIDE`] columns through `site::gemm` on `route`'s dense table
    /// (the tile), with activations scaled by `x_scale`: the clauses of the
    /// module doc.
    fn e4m3_case(
        gpu: &Gpu,
        (g32, gemm, route): (&Gemm32Kernels, &GemmKernels, &GemmRoute),
        split: &Split,
        (file_ty, tag, s): (GgmlType, &str, &Shape),
        bytes: &[u8],
        x_scale: f32,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let name = shape_name(s, tag);
        let (rows, k) = (s.rows, s.k);
        let want = SiteTy::of_ggml(file_ty).ok_or_else(|| format!("{file_ty}: no site type"))?;
        let ty = site::file_site(split, "gate_mxfp8", &name, (rows, k), &[want])?;
        let w = Weights::load_where(stream, split, |n| n == name)?;
        site::site(&w, "gate_mxfp8", &name, (rows, k), ty)?;
        let wf = host_rows(file_ty, bytes, rows, k);
        let mut ok = true;
        for (i, &m) in s.ms.iter().chain(&[WIDE]).enumerate() {
            let (x, x_dev, mut y_dev) =
                columns(gpu, (m, k, rows), s.seed ^ (0x1000 + i as u64), x_scale)?;
            let wide = m == WIDE;
            if wide {
                let x_in = WideIn {
                    q128: None,
                    q32: None,
                    f32: Some(&x_dev),
                };
                let k32 = WideKernels {
                    gemm,
                    g32: Some(g32),
                };
                let src = (ty, &w, SiteWeight::Resident(&name));
                let table = (route, GemmInput::PerSlot);
                site::gemm(gpu, &k32, src, rows, &x_in, table, m, &mut y_dev)?;
            } else {
                site::gemv(gpu, g32, (ty, &w, &name), &x_dev, m, &mut y_dev)?;
            }
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            let walk = |r: usize, c: usize| byte_rule(&wf[r], &x[c * k..][..k]);
            let tile = |r: usize, c: usize| tile_rule(&wf[r], &x[c * k..][..k]);
            let shown = format!("shape={} rows={rows} k={k} m={m}", s.stem);
            ok &= if wide {
                check((tag, "tile"), &shown, (&y, &x, k), &wf, &tile)
            } else {
                check((tag, "gemv"), &shown, (&y, &x, k), &wf, &walk)
            };
            if m == 1 {
                let mut y_dev = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows])?;
                site::resident_gemv(gpu, &w, &name, &x_dev, &mut y_dev)?;
                stream.synchronize()?;
                let y = y_dev.to_host_vec(stream)?;
                ok &= check((tag, "decode"), &shown, (&y, &x, k), &wf, &walk);
            }
        }
        Ok(ok)
    }

    /// The Q8_0 site gemv keeps its walk: at one and at three columns, every
    /// output bit for bit the byte-order rule and within the band.
    fn q8_0_case(
        gpu: &Gpu,
        g32: &Gemm32Kernels,
        split: &Split,
        bytes: &[u8],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (rows, k) = (SHAPES[0].rows, SHAPES[0].k);
        let ty = site::file_site(split, "gate_mxfp8", Q8_0_NAME, (rows, k), &[SiteTy::Q8_0])?;
        let w = Weights::load_where(stream, split, |n| n == Q8_0_NAME)?;
        let wf = host_rows(GgmlType::Q8_0, bytes, rows, k);
        let mut ok = true;
        for m in [1, 3] {
            let (x, x_dev, mut y_dev) = columns(gpu, (m, k, rows), 0x5eed_0811 + m as u64, 1.0)?;
            site::gemv(gpu, g32, (ty, &w, Q8_0_NAME), &x_dev, m, &mut y_dev)?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            let walk = |r: usize, c: usize| byte_rule(&wf[r], &x[c * k..][..k]);
            let tag = format!("rows={rows} k={k} m={m}");
            ok &= check(("q8_0", "q8_0"), &tag, (&y, &x, k), &wf, &walk);
        }
        Ok(ok)
    }

    /// The F32 tile keeps its rule: the F32 site at [`WIDE`] columns, every
    /// output bit for bit the host F32 rule and within the band.
    fn f32_tile_case(
        gpu: &Gpu,
        g32: &Gemm32Kernels,
        split: &Split,
        bytes: &[u8],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (rows, k) = (F32_ROWS, SHAPES[0].k);
        let ty = site::file_site(split, "gate_mxfp8", F32_NAME, (rows, k), &[SiteTy::F32])?;
        let w = Weights::load_where(stream, split, |n| n == F32_NAME)?;
        let wf = host_rows(GgmlType::F32, bytes, rows, k);
        let (x, x_dev, mut y_dev) = columns(gpu, (WIDE, k, rows), 0x5eed_0821, 1.0)?;
        site::gemv(gpu, g32, (ty, &w, F32_NAME), &x_dev, WIDE, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let tile = |r: usize, c: usize| tile_rule(&wf[r], &x[c * k..][..k]);
        let tag = format!("rows={rows} k={k} m={WIDE}");
        Ok(check(("f32", "f32_tile"), &tag, (&y, &x, k), &wf, &tile))
    }

    /// `--bench-kernels` (module doc): one column at each shape, the MXFP8
    /// planes of [`mxfp8_bytes`] and Q8_0 planes of the same rows, the
    /// activation from `fill_pattern_f32`.
    mod bench {
        use super::{SHAPES, mxfp8_bytes};
        use crate::bench_arm::{GREPS, N, ROUNDS, burst, fill_pattern, fill_pattern_f32};
        use crate::bench_arm::{print_arm, replay};
        use bloomery_gpu::q8f32::{E4m3GemvArgs, E4m3Scales, E4m3Weight, e4m3_planes};
        use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
        use bloomery_gpu_gates::GateError;
        use cuda_core::{CudaStream, DeviceBuffer};

        /// Target bytes of weight copies, subject to the per-burst copy cap.
        const SPAN: u64 = 512 << 20;

        /// Copies needed to reach [`SPAN`], capped at [`N`] launches per burst.
        fn copies(bytes: u64) -> usize {
            (SPAN.div_ceil(bytes) as usize).clamp(1, N)
        }

        pub fn run(gpu: &Gpu, only: Option<&str>) -> Result<(), GateError> {
            println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS}");
            let stream = gpu.stream();
            let q8 = gpu.q8f32();
            for s in &SHAPES {
                let (rows, k) = (s.rows, s.k);
                let x = DeviceBuffer::from_host(stream, &fill_pattern_f32(k))?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
                // The weight's bytes, the column's f32 values, the outputs.
                let io = (4 * k + 4 * rows) as u64;
                let op = format!("mxfp8_{rows}");
                if only.is_none_or(|o| op.starts_with(o)) {
                    let bytes = (rows * k / 32 * 33) as u64;
                    let bytes_in = mxfp8_bytes(rows, k, s.seed);
                    let (qs, e) = e4m3_planes(E4m3Scales::Mxfp8, &bytes_in, rows, k)?;
                    let ws = (0..copies(bytes))
                        .map(|_| {
                            Ok((
                                DeviceTensor::upload(stream, &qs, rows, k / 4)?,
                                DeviceTensor::upload(stream, &e, rows, k / 128)?,
                            ))
                        })
                        .collect::<Result<Vec<_>, GpuError>>()?;
                    let mut at = 0;
                    let mut enq = |st: &CudaStream| {
                        let (qs, e) = &ws[at % ws.len()];
                        at += 1;
                        q8.enqueue_e4m3_gemv(
                            st,
                            E4m3GemvArgs {
                                w: E4m3Weight {
                                    qs,
                                    e,
                                    scales: E4m3Scales::Mxfp8,
                                },
                                x: &x,
                                m: 1,
                                y: &mut y,
                            },
                        )
                    };
                    let eager = burst(stream, &mut enq)?;
                    let g = gpu.capture(|st| (0..N).try_for_each(|_| enq(st)))?;
                    let graph = replay(stream, &g)?;
                    print_arm(&op, rows, bytes + io, g.node_count(), eager, graph);
                }
                let op = format!("q8_0_{rows}");
                if only.is_none_or(|o| op.starts_with(o)) {
                    let words = fill_pattern(rows * k / 4);
                    // f16 scales of 2^-8 to 2^1: finite, like a file's.
                    let d: Vec<u16> = fill_pattern(rows * k / 32)
                        .into_iter()
                        .map(|p| (((7 + (p >> 16) % 9) << 10) | (p & 0x3ff)) as u16)
                        .collect();
                    let bytes = (rows * k / 32 * 34) as u64;
                    let ws = (0..copies(bytes))
                        .map(|_| {
                            Ok((
                                DeviceTensor::upload(stream, &words, rows, k / 4)?,
                                DeviceTensor::upload(stream, &d, rows, k / 32)?,
                            ))
                        })
                        .collect::<Result<Vec<_>, GpuError>>()?;
                    let mut at = 0;
                    let mut enq = |st: &CudaStream| {
                        let (qs, d) = &ws[at % ws.len()];
                        at += 1;
                        q8.enqueue_q8_0_gemv(st, qs, d, &x, 1, &mut y)
                    };
                    let eager = burst(stream, &mut enq)?;
                    let g = gpu.capture(|st| (0..N).try_for_each(|_| enq(st)))?;
                    let graph = replay(stream, &g)?;
                    print_arm(&op, rows, bytes + io, g.node_count(), eager, graph);
                }
            }
            Ok(())
        }
    }
}
