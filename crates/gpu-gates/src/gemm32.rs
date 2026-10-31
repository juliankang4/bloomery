//! The host transcription of the 32-value GEMM family (`bloomery_gpu::gemm`'s
//! `Gemm32Kernels`): its quantizer and the per-output numeric contract that
//! `gemm32.rs`'s module doc states. One owner, so every gate that holds a
//! `gemm_q8_0p`, `gemm_q8_0f`, `gemm_q5_1`, `gemm_q5_0p` or `gemm_nvfp4p`
//! output bit for bit holds it to the same transcription (`gate_gemm`'s
//! `g32_*` cases, `gate_glm5next_gemm`).

use crate::rounding::scaled_sum;
use gguf::quant::{KVALUES_MXFP4, QuantError, ue4m3_to_f32_half};

/// A `GemmAct32`'s three planes on the host, in the device layout: codes,
/// scales, sums.
pub type Planes32 = (Vec<u32>, Vec<f32>, Vec<i32>);

/// The host quantizer of one column into the device layout: per 32-value
/// block `d = amax/127` (1 for an all-zero block), codes `round(x/d)`
/// clamped to ±127 and their sum; a block holding a non-finite value
/// refused (NaN `d`, zero codes and sum). Returns the column's `16·steps`
/// code words, `2·steps` scales and sums, padding zero.
#[must_use]
pub fn quant_col(x: &[f32], steps: usize) -> Planes32 {
    let (mut q, mut d, mut s) = (
        vec![0u32; 16 * steps],
        vec![0f32; 2 * steps],
        vec![0i32; 2 * steps],
    );
    for (b, blk) in x.as_chunks::<32>().0.iter().enumerate() {
        if blk.iter().any(|v| !v.is_finite()) {
            d[b] = f32::NAN;
            continue;
        }
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let db = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        d[b] = db;
        let mut sum = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let c = (v / db).round().clamp(-127.0, 127.0) as i8;
            sum += i32::from(c);
            q[8 * b + i / 4] |= u32::from(c as u8) << (8 * (i % 4));
        }
        s[b] = sum;
    }
    (q, d, s)
}

/// The activations as the contract reads them: per column its codes (value
/// order), block scales and sums.
pub struct HostAct {
    pub k: usize,
    pub codes: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<i32>,
}

/// Quantize `x` (`n` columns of `k`) on the host; also returns the three
/// planes in the device layout for a bitwise comparison.
#[must_use]
pub fn host_act(x: &[f32], k: usize, n: usize) -> (HostAct, Planes32) {
    let steps = k.div_ceil(64);
    let kb = k / 32;
    let mut planes = (Vec::new(), Vec::new(), Vec::new());
    let mut ha = HostAct {
        k,
        codes: Vec::with_capacity(n * k),
        d: Vec::with_capacity(n * kb),
        s: Vec::with_capacity(n * kb),
    };
    for c in 0..n {
        let (q, d, s) = quant_col(&x[c * k..(c + 1) * k], steps);
        for w in &q[..k / 4] {
            ha.codes.extend(w.to_le_bytes().iter().map(|&b| b as i8));
        }
        ha.d.extend_from_slice(&d[..kb]);
        ha.s.extend_from_slice(&s[..kb]);
        planes.0.extend(q);
        planes.1.extend(d);
        planes.2.extend(s);
    }
    (ha, planes)
}

/// One output of weight row `(wq, wd, wm)` — its `k` codes, its `k/32` block
/// scales and (Q5_1, `mins`) block mins, read only when `mins` — against
/// column `col` of `a`, three ways: the contract's transcription (per block
/// the exact i32 dot, then `acc = fma(d_a, d_w·f32(isum), acc)`, the min
/// term `fma(m_w, f32(s_a), ·)` inside for Q5_1, from 0, blocks in
/// increasing k, each step a [`scaled_sum`] into the F32 accumulator), the
/// f64 value and its band's magnitude `Σ_b (|D_b| + |M_b|)`.
#[must_use]
pub fn dot32(
    wq: &[i8],
    wd: &[f32],
    wm: &[f32],
    a: &HostAct,
    col: usize,
    mins: bool,
) -> (f32, f64, f64) {
    let kb = a.k / 32;
    let (mut acc, mut y, mut mag) = (0.0f32, 0.0f64, 0.0f64);
    for b in 0..kb {
        let wb = &wq[32 * b..32 * b + 32];
        let ab = &a.codes[col * a.k + 32 * b..col * a.k + 32 * b + 32];
        let isum: i32 = wb
            .iter()
            .zip(ab)
            .map(|(&p, &q)| i32::from(p) * i32::from(q))
            .sum();
        let dw = wd[b];
        let (da, sa) = (a.d[col * kb + b], a.s[col * kb + b]);
        let (m, s) = if mins { (wm[b], sa as f32) } else { (0.0, 0.0) };
        acc = scaled_sum(
            [f64::from(dw), f64::from(m)],
            [isum as f32, s],
            da,
            f64::from(acc),
        ) as f32;
        let dd = f64::from(da) * f64::from(dw) * f64::from(isum);
        let mm = if mins {
            f64::from(da) * f64::from(wm[b]) * f64::from(sa)
        } else {
            0.0
        };
        y += dd + mm;
        mag += dd.abs() + mm.abs();
    }
    (acc, y, mag)
}

/// One output of an NVFP4 weight row — its `k` decoded E2M1 int8 codes in
/// value order, its `k/16` sub-block scales (`d0`, `d1` of each 32-value
/// unit in turn) and the tensor's `scale` — against column `col` of `a`,
/// three ways: the contract's transcription (per unit the two exact i32
/// sub-block dots, `acc = fma(d_a, fma(d1, f32(i1), d0·f32(i0)), acc)` from
/// 0, units in increasing k, each step a [`scaled_sum`] into an F64
/// accumulator, the stored value `scale·acc` one more), the f64 value
/// `scale · Σ_b d_a·(d0·i0 + d1·i1)` and its band's magnitude.
#[must_use]
pub fn dot32_nv(wq: &[i8], wsc: &[f32], scale: f32, a: &HostAct, col: usize) -> (f32, f64, f64) {
    let kb = a.k / 32;
    let (mut acc, mut y, mut mag) = (0.0f64, 0.0f64, 0.0f64);
    for b in 0..kb {
        let (wb, ab) = (
            &wq[32 * b..32 * b + 32],
            &a.codes[col * a.k + 32 * b..col * a.k + 32 * b + 32],
        );
        let da = a.d[col * kb + b];
        let (d0, d1) = (wsc[2 * b], wsc[2 * b + 1]);
        let dot = |w: &[i8], x: &[i8]| -> i32 {
            w.iter()
                .zip(x)
                .map(|(&p, &q)| i32::from(p) * i32::from(q))
                .sum()
        };
        let (i0, i1) = (dot(&wb[..16], &ab[..16]), dot(&wb[16..], &ab[16..]));
        acc = scaled_sum(
            [f64::from(d0), f64::from(d1)],
            [i0 as f32, i1 as f32],
            da,
            acc,
        );
        let t = f64::from(da) * (f64::from(d0) * f64::from(i0) + f64::from(d1) * f64::from(i1));
        y += t;
        mag += t.abs();
    }
    (
        scaled_sum([acc, 0.0], [scale, 0.0], 1.0, 0.0) as f32,
        f64::from(scale) * y,
        f64::from(scale) * mag,
    )
}

/// The index of the ascending `table`'s entry nearest `x`, ties to the even
/// index.
fn nearest(table: &[f64], x: f64) -> usize {
    (1..table.len()).fold(0, |best, i| {
        let (di, db) = ((x - table[i]).abs(), (x - table[best]).abs());
        if di < db || (di == db && i % 2 == 0) {
            i
        } else {
            best
        }
    })
}

/// One output of the native NVFP4 product (`gemm_nvfp4_mma`, W4A4): weight
/// row `(wq, wsc, scale)` as [`dot32_nv`] takes it, against column `col` of
/// `a` packed as that product's activation producer packs it. The column
/// scale `gs` is `max_b d_b · (127/6)` over the blocks holding a nonzero
/// code, 1 when none does. Each 16-value sub-block of values `code·d_b`
/// takes the UE4M3 scale nearest `amax / (6·gs)`, and each value divided by
/// that scale times `gs` takes the nearest E2M1 level, ties to even both
/// times. A sub-block whose scale does not decode to a normal f32 is packed
/// again from `code · (d_b/gs)` against `amax / 6` and the scale alone.
/// Returns the f64 value `scale · gs · Σ_s w_s·a_s·Σ codes` and its band's
/// magnitude; both are NaN when a block of the column was refused.
///
/// # Errors
///
/// [`QuantError::NotUe4m3`] from the scale table, whose bytes are all
/// UE4M3 scales.
pub fn dot32_nv_fp4(
    wq: &[i8],
    wsc: &[f32],
    scale: f32,
    a: &HostAct,
    col: usize,
) -> Result<(f64, f64), QuantError> {
    let (k, kb) = (a.k, a.k / 32);
    let codes = &a.codes[col * k..(col + 1) * k];
    let d = &a.d[col * kb..(col + 1) * kb];
    if d.iter().any(|v| !v.is_finite()) {
        return Ok((f64::NAN, f64::NAN));
    }
    let max = (0..kb)
        .filter(|&b| codes[32 * b..32 * b + 32].iter().any(|&c| c != 0))
        .fold(0.0f32, |m, b| m.max(d[b]));
    let gs = if max > 0.0 { max * (127.0 / 6.0) } else { 1.0 };
    let halves = (0..0x7f)
        .map(ue4m3_to_f32_half)
        .collect::<Result<Vec<_>, _>>()?;
    let scales: Vec<f64> = halves.iter().map(|&h| 2.0 * f64::from(h)).collect();
    let levels: Vec<f64> = KVALUES_MXFP4[..8]
        .iter()
        .map(|&v| f64::from(v) / 2.0)
        .collect();
    let (mut y, mut mag) = (0.0f64, 0.0f64);
    for (s, q) in codes.as_chunks::<16>().0.iter().enumerate() {
        let pack = |unit: f32, g: f32| {
            let v = q.map(|c| f32::from(c) * unit);
            let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let byte = nearest(&scales, f64::from(amax / (6.0 * g)));
            (v, amax, byte, halves[byte] * 2.0 * g)
        };
        let (mut v, amax, mut byte, mut decoded) = pack(d[s / 2], gs);
        if !amax.is_finite()
            || !(6.0 * gs).is_finite()
            || decoded < f32::MIN_POSITIVE
            || !(1.0 / decoded).is_finite()
        {
            let nonzero = q.iter().any(|&c| c != 0);
            (v, _, byte, decoded) = pack(if nonzero { d[s / 2] / gs } else { 0.0 }, 1.0);
        }
        let inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
        let isum: i32 = v
            .iter()
            .zip(&wq[16 * s..16 * s + 16])
            .map(|(&x, &w)| {
                let t = x * inv;
                let code = KVALUES_MXFP4[nearest(&levels, f64::from(t.abs()))];
                i32::from(w) * i32::from(if t < 0.0 { -code } else { code })
            })
            .sum();
        let t = f64::from(wsc[s]) * f64::from(halves[byte]) * f64::from(isum);
        y += t;
        mag += t.abs();
    }
    let g = f64::from(scale) * f64::from(gs);
    Ok((g * y, g.abs() * mag))
}

/// One column's native FP4 form as the fused NVFP4 gate and up launch
/// (`Gemm32Kernels::enqueue_gate_up_sel32`) stores it: each value's E2M1
/// level (signed), the UE4M3 scale words (four sub-blocks a word) and the
/// column scale.
pub struct Fp4Column {
    pub levels: Vec<f32>,
    pub scale_words: Vec<u32>,
    pub scale: f32,
}

/// The host transcription of the fused launch's FP4 of one column from its
/// 32-value Q8 codes `codes` (value order) and block scales `d` (NaN for a
/// refused block). Each 16-value sub-block takes the scale `s = (8 + m) ·
/// 2^(e - 3)`, `amax / 6` (`big · d / 6` in f32, in f64 rounded once to
/// f32 where `d` is above `f32::MAX / 128`) rounded to three mantissa bits
/// (ties to even), and each value the E2M1 level nearest `code · (d / s)`
/// (ties to even, saturating at 6). A sub-block whose codes are all zero or whose `e`
/// is below -142 is zero, scale byte 0. The column scale is `2^(E - 7)` for
/// the largest `e` of the column, each scale byte the UE4M3 field
/// `e - E + 14` with `m`, 0 below field 1; a refused column's bytes are 0
/// and its scale NaN, a zero column's scale 1.
#[must_use]
pub fn gate_up_fp4(codes: &[i8], d: &[f32]) -> Fp4Column {
    let levels_of: [f64; 8] = std::array::from_fn(|j| f64::from(KVALUES_MXFP4[j]) / 2.0);
    let mut levels = vec![0.0f32; codes.len()];
    let mut subs = Vec::with_capacity(codes.len() / 16);
    for (s, q) in codes.as_chunks::<16>().0.iter().enumerate() {
        let db = d[s / 2];
        let big = q.iter().map(|c| c.unsigned_abs()).max().unwrap_or(0);
        let t = if db > f32::MAX / 128.0 {
            (f64::from(big) * f64::from(db) / 6.0) as f32
        } else {
            f32::from(big) * db / 6.0
        };
        if db.is_nan() || t == 0.0 {
            subs.push(None);
            continue;
        }
        let (bits, off) = if t < f32::MIN_POSITIVE {
            ((t * 2f32.powi(64)).to_bits(), -64)
        } else {
            (t.to_bits(), 0)
        };
        let rounded = bits + 0x7_ffff + ((bits >> 20) & 1);
        let (e, m) = ((rounded >> 23) as i32 - 127 + off, (rounded >> 20) & 7);
        if e < -142 {
            subs.push(None);
            continue;
        }
        let ratio = db / (f64::from(8 + m) * 2f64.powi(e - 3)) as f32;
        for (i, &c) in q.iter().enumerate() {
            let x = f64::from(f32::from(c) * ratio);
            levels[16 * s + i] =
                (levels_of[nearest(&levels_of, x.abs())] as f32).copysign(x as f32);
        }
        subs.push(Some((e, m)));
    }
    let refused = d.iter().any(|v| v.is_nan());
    let top = subs.iter().flatten().map(|&(e, _)| e).max();
    let mut scale_words = vec![0u32; codes.len() / 64];
    if let (false, Some(top)) = (refused, top) {
        for (s, &(e, m)) in subs
            .iter()
            .enumerate()
            .filter_map(|(s, v)| v.as_ref().map(|v| (s, v)))
        {
            let field = e - top + 14;
            if field >= 1 {
                scale_words[s / 4] |= (((field as u32) << 3) | m) << (8 * (s % 4));
            }
        }
    }
    let scale = match (refused, top) {
        (true, _) => f32::NAN,
        (false, None) => 1.0,
        (false, Some(top)) => 2f64.powi(top - 7) as f32,
    };
    Fp4Column {
        levels,
        scale_words,
        scale,
    }
}

/// The f32 roundings that separate a `gemm_q5_0p` output from the Q5_0
/// gemv's (`q5.rs`, `q5_0_gemv`) for the same weight row and the same
/// activation codes, at `kb` blocks: the band of the gate's
/// `q5_0p_is_the_gemv` is `γ(this) · Σ_b |d_a·d_w·isum_b|`.
///
/// Both kernels form each block's exact i32 `isum_b` (the same int8 codes:
/// `q − 16` and the column's q8, the quantizers' bits equal) and the same
/// `t_b = d_w·f32(isum_b)` (`|isum_b| <= 32·16·127 < 2^24`, so the
/// conversion is exact; one product, the same rounding on both sides). They
/// differ in how the terms `d_a·t_b` reach the sum:
/// - the GEMM: `acc = fma(d_a, t_b, acc)` from 0, blocks in increasing k —
///   each term through at most `kb` roundings;
/// - the gemv: lane `b % 32` adds `(t_b·d_a)` (one rounding, or none if the
///   backend fuses it into the add) to its partial, then a five-level xor
///   butterfly — each term through at most `1 + (ceil(kb/32) − 1) + 5`.
///
/// So both lie within `γ(kb)` and `γ(ceil(kb/32) + 5)` of `Σ_b d_a·t_b`,
/// whose magnitude is at most `(1 + u)` times the exact `Σ_b |d_a d_w
/// isum_b|`; `γ(a) + γ(b) <= γ(a + b)` and `(1 + u)·γ(n) <= γ(n + 1)` give
/// `|y_gemm − y_gemv| <= γ(kb + ceil(kb/32) + 6) · Σ_b |d_a d_w isum_b|`.
/// Not bit-equal: the two sum orders differ for every `kb > 1`.
#[must_use]
pub fn q5_0p_gemv_roundings(kb: usize) -> usize {
    kb + kb.div_ceil(32) + 6
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NVFP4 outputs whose F32 walk overflows before the expert scale, at
    /// K 64 with activation codes 127 and tensor scale 2^-120. Codes 2 under
    /// sub-block scales 0.5 with `d_a` 2^120: the kernel keeps the sum in
    /// F64 and stores 8128. Codes `(12 ×16, 1, 0 ×15)` and their negation
    /// under sub-block scales `(224, 2^-10)` with `d_a` `(1 + 2^-23)·2^120`:
    /// sub-block dots `(24384, 127)` then `(-24384, -127)`, and the second
    /// step's fused F64 multiply-add keeps the first step's rounding error,
    /// stored as 2^-33 (a separate product and sum would cancel to 0).
    #[test]
    fn nvfp4_contract_keeps_range_through_the_expert_scale() {
        let unit: Vec<i8> = [[12; 16].as_slice(), &[1], &[0; 15]].concat();
        let cancel = [unit.clone(), unit.iter().map(|c| -c).collect()].concat();
        let d_cancel = f32::from_bits(0x7b80_0001);
        for (codes, wsc, d_a, want, want_y) in [
            (vec![2; 64], [0.5; 4], 2f32.powi(120), 8128.0, 8128.0),
            (
                cancel,
                [224.0, 2f32.powi(-10), 224.0, 2f32.powi(-10)],
                d_cancel,
                2f32.powi(-33),
                0.0,
            ),
        ] {
            let (a, _) = host_act(&[127.0 * d_a; 64], 64, 1);
            assert_eq!((a.codes[0], a.d[0]), (127, d_a));
            let (got, y, _) = dot32_nv(&codes, &wsc, 2f32.powi(-120), &a, 0);
            assert_eq!(
                (got.to_bits(), y),
                (want.to_bits(), want_y),
                "{got:e} against {want:e}"
            );
        }
    }

    /// A Q5_1 step whose product leaves the F32 normal range: `d_a` 2^-149,
    /// inner sum `1.5 - 2^-24`, whose F32 rounding 1.5 would tie the
    /// subnormal product up to 2^-148; the exact inner sum gives 2^-149.
    #[test]
    fn plain_contract_keeps_a_subnormal_step_exact() {
        let mut codes = vec![0i8; 32];
        codes[0] = 1;
        let a = HostAct {
            k: 32,
            codes: codes.clone(),
            d: vec![f32::from_bits(1)],
            s: vec![1],
        };
        let (got, _, _) = dot32(&codes, &[-(2f32.powi(-24))], &[1.5], &a, 0, true);
        assert_eq!(got.to_bits(), 1);
    }
}
