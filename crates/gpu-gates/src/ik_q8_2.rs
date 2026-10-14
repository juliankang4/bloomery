//! ik's rule for a Q8_0 weight times f32 activations on its AVX2 build: the
//! activations quantized to q8_2 blocks (`quantize_row_q8_2_x4`,
//! `iqk_quantize.cpp`), then the Q8_0 × q8_2 dot
//! (`mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, _, block_q8_2>`,
//! `iqk_gemm_legacy_quants.cpp`). A gate that simulates ik on such a weight
//! reads the rule here; it is transcribed from ik and calls no engine or
//! device code. [`project`] computes a weight's rows by this rule and by ours
//! (f32 activations), both exactly, and [`Proj::gap_bound`] bounds the two
//! apart, for the gates that check them.

use crate::rounding::{U, U64};
use gguf::quant::{Q8Block, half_to_f32};

/// Values per Q8_0 and q8_2 block.
pub const QK: usize = 32;

/// ik's q8_2 activation blocks of `x`: per 32 values `d = amax/127` rounded
/// to bf16 (nearest even, `ggml_compute_fp32_to_bf16`), `id = 1/d` (0 when
/// `d` is 0), each code `x·id` rounded half to even and saturated to i8 by
/// the two packs. Returns the codes and the scales. `x` is whole blocks.
/// Panics on a partial block, and on a non-finite value, which this form has
/// no code for.
pub fn quantize(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert!(x.len().is_multiple_of(QK), "q8_2 blocks are {QK} values");
    crate::act_rule::refuse_non_finite("ik_q8_2::quantize", x);
    let mut q = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / QK);
    for blk in x.as_chunks::<QK>().0 {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let bits = (amax / 127.0).to_bits();
        let db = f32::from_bits(((bits + (0x7fff + ((bits >> 16) & 1))) >> 16) << 16);
        let id = if db > 0.0 { 1.0 / db } else { 0.0 };
        q.extend(
            blk.iter()
                .map(|&v| (v * id).round_ties_even().clamp(-128.0, 127.0) as i8),
        );
        d.push(db);
    }
    (q, d)
}

/// The values ik's q8_2 blocks of `x` ([`quantize`]) stand for, `code · d`
/// each — what a gate bounds ik's dot through. Exact in f32: a code of at
/// most eight bits times a bf16 scale. A zero code gives `+0.0`, never
/// `-0.0`, since the codes are integers. `x` is whole blocks.
pub fn reconstruct(x: &[f32]) -> Vec<f32> {
    let (q, d) = quantize(x);
    q.as_chunks::<QK>()
        .0
        .iter()
        .zip(d)
        .flat_map(|(blk, s)| blk.iter().map(move |&c| f32::from(c) * s))
        .collect()
}

/// The activation code `SignedDot` pairs with weight code `w`:
/// `_mm256_sign_epi8(a, w)` — `a` under a positive weight, `−a` under a
/// negative one, wrapping, so −128 stays −128, and 0 under a zero weight.
pub fn folded(w: i8, a: i8) -> i8 {
    match w.signum() {
        -1 => a.wrapping_neg(),
        0 => 0,
        _ => a,
    }
}

/// ik's integer partial of one lane: values `16h .. 16h + 16` of weight
/// block `w` against activation codes `a` (the block's 32).
/// `maddubs(|w|, folded)` sums byte pairs into i16 with saturation — never
/// reached: a pair lies in [−2·128·128, 2·128·127] — and `madd` and the
/// lane combination of `Sum4TypeQ82S` add the rest exactly.
pub fn half_sum(w: &Q8Block, a: &[i8], h: usize) -> i32 {
    (8 * h..8 * h + 8)
        .map(|m| {
            let term =
                |v: usize| i32::from(w.q[v].unsigned_abs()) * i32::from(folded(w.q[v], a[v]));
            (term(2 * m) + term(2 * m + 1)).clamp(i32::from(i16::MIN), i32::from(i16::MAX))
        })
        .sum()
}

/// ik's integer dot of weight block `w` with activation codes `a`: its two
/// [`half_sum`]s.
fn block_sum(w: &Q8Block, a: &[i8]) -> i32 {
    half_sum(w, a, 0) + half_sum(w, a, 1)
}

/// ik's dot of one row: lane L of eight takes block `4i + L%4` of each group
/// of four, values `16·(L/4) ..` of it, and fuses one multiply-add of that
/// integer partial by the scales' product `d_w·d_x` (exact in f32) into its
/// accumulator (`AccumT<MinusType0, _, true>` with `ScaleHelperQ8_2S`);
/// `hsum_float_8` adds the halves lane-wise, then pairs. Panics on a block
/// count that is not a multiple of four.
pub fn dot(blocks: &[Q8Block], xq: &[i8], xd: &[f32]) -> f32 {
    assert!(
        blocks.len().is_multiple_of(4),
        "ik's q8_2 dot takes blocks in groups of 4"
    );
    let mut acc = [0.0f32; 8];
    for i in 0..blocks.len() / 4 {
        for (l, a) in acc.iter_mut().enumerate() {
            let b = 4 * i + l % 4;
            let dd = half_to_f32(blocks[b].d) * xd[b];
            let p = half_sum(&blocks[b], &xq[b * QK..(b + 1) * QK], l / 4);
            *a = dd.mul_add(p as f32, *a);
        }
    }
    let s = [
        acc[0] + acc[4],
        acc[1] + acc[5],
        acc[2] + acc[6],
        acc[3] + acc[7],
    ];
    (s[0] + s[2]) + (s[1] + s[3])
}

/// The projection of `x` by a Q8_0 weight, per output row: both rules'
/// exact values and the magnitudes the accumulation bounds scale.
pub struct Proj {
    /// Our rule's exact value: the f64 dot with the f32 activations.
    pub ours: Vec<f64>,
    /// ik's exact value: the integer block dots with its q8_2 activations
    /// times both scales.
    pub ik: Vec<f64>,
    /// `Σ|w·x|`.
    abs_ours: Vec<f64>,
    /// `Σ_b|d_w·d_x·isum_b|`.
    pub abs_ik: Vec<f64>,
}

impl Proj {
    /// Per row, how far our value may sit from ik's, `k` values a row: the
    /// two exact values' distance plus both sides' accumulation bounds and
    /// the f64 sums' own error.
    pub fn gap_bound(&self, k: usize) -> Vec<f64> {
        let f64_sum = k as f64 * U64;
        (0..self.ours.len())
            .map(|r| {
                (self.ours[r] - self.ik[r]).abs()
                    + (ik_gemv_rel(k) + f64_sum) * self.abs_ik[r]
                    + (our_gemv_rel(k) + f64_sum) * self.abs_ours[r]
            })
            .collect()
    }
}

/// How far ik's f32 projection sits from its exact value, relative to
/// `Σ_b|d_w·d_x·isum_b|`: each block term rounds at most twice (the
/// scales' product, then the integer sum's), and a sum of `k/32` terms in
/// any order rounds at most `k/32 − 1` times; two more for the margin.
pub fn ik_gemv_rel(k: usize) -> f64 {
    (k / QK + 4) as f64 * U
}

/// How far our `q8_0_gemv` at m = 1 sits from its exact value, relative
/// to `Σ|w·x|`: each lane runs one fused multiply-add per value along
/// its `k/32` values (the weight `q·d` is exact in f32), then five
/// butterfly levels.
fn our_gemv_rel(k: usize) -> f64 {
    (k / 32 + 5) as f64 * U
}

/// [`Proj`] of `x` through the Q8_0 weight `w` (file bytes) of `rows` rows,
/// the rows split across the host's threads. `x` is whole blocks.
pub fn project(w: &[u8], rows: usize, x: &[f32]) -> Proj {
    let rb = x.len() / QK * 34;
    let (qx, dx) = quantize(x);
    let mut p = Proj {
        ours: vec![0.0; rows],
        ik: vec![0.0; rows],
        abs_ours: vec![0.0; rows],
        abs_ik: vec![0.0; rows],
    };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|sc| {
        let parts = p
            .ours
            .chunks_mut(chunk)
            .zip(p.ik.chunks_mut(chunk))
            .zip(p.abs_ours.chunks_mut(chunk))
            .zip(p.abs_ik.chunks_mut(chunk))
            .enumerate();
        for (ci, (((ours, ik), ao), ai)) in parts {
            let (qx, dx) = (&qx, &dx);
            sc.spawn(move || {
                let outs = ours.iter_mut().zip(ik).zip(ao).zip(ai);
                for (j, (((o, i), a), b)) in outs.enumerate() {
                    let row = &w[(ci * chunk + j) * rb..(ci * chunk + j + 1) * rb];
                    let (mut so, mut si, mut sao, mut sai) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    let blocks = row.as_chunks::<34>().0.iter().zip(x.as_chunks::<QK>().0);
                    for (bi, (blk, xb)) in blocks.enumerate() {
                        let qb = Q8Block::from_bytes(blk);
                        let dw = half_to_f32(qb.d);
                        for (&qw, &xv) in qb.q.iter().zip(xb) {
                            let t = f64::from(f32::from(qw) * dw) * f64::from(xv);
                            so += t;
                            sao += t.abs();
                        }
                        let isum = block_sum(&qb, &qx[bi * QK..(bi + 1) * QK]);
                        let t = f64::from(dw) * f64::from(dx[bi]) * f64::from(isum);
                        si += t;
                        sai += t.abs();
                    }
                    (*o, *i, *a, *b) = (so, si, sao, sai);
                }
            });
        }
    });
    p
}

#[cfg(test)]
mod tests {
    use super::{QK, dot, quantize};
    use gguf::quant::Q8Block;

    /// Whether `f` panicked. A panic without `msg` in its message fails the
    /// test.
    fn refused(f: impl FnOnce() + std::panic::UnwindSafe, msg: &str) -> bool {
        match std::panic::catch_unwind(f) {
            Ok(()) => false,
            Err(e) => {
                let text = e
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| e.downcast_ref::<&str>().copied());
                assert!(
                    text.is_some_and(|s| s.contains(msg)),
                    "panicked with {text:?}, not {msg:?}"
                );
                true
            }
        }
    }

    /// A partial block stops `quantize`, and a block count off a multiple
    /// of four stops `dot`, by name instead of dropping the tail.
    #[test]
    fn partial_blocks_stop_by_name() {
        for n in 0..=2 * QK {
            let x = crate::activations(n, 1, 7);
            let stopped = refused(|| _ = quantize(&x), "q8_2 blocks are 32 values");
            assert_eq!(stopped, !n.is_multiple_of(QK), "quantize of {n} values");
        }
        let blocks = [Q8Block {
            d: 0x3c00,
            q: [1; QK],
        }; 8];
        for n in 0..=8 {
            let (xq, xd) = (vec![1i8; n * QK], vec![1.0f32; n]);
            let stopped = refused(
                || _ = dot(&blocks[..n], &xq, &xd),
                "ik's q8_2 dot takes blocks in groups of 4",
            );
            assert_eq!(stopped, !n.is_multiple_of(4), "dot of {n} blocks");
        }
    }
}
