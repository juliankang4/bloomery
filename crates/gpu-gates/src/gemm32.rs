//! The host transcription of the 32-value GEMM family (`bloomery_gpu::gemm`'s
//! `Gemm32Kernels`): its quantizer and the per-output numeric contract that
//! `gemm32.rs`'s module doc states. One owner, so every gate that holds a
//! `gemm_q8_0p`, `gemm_q8_0f`, `gemm_q5_1`, `gemm_q5_0p` or `gemm_nvfp4p`
//! output bit for bit holds it to the same transcription (`gate_gemm`'s
//! `g32_*` cases, `gate_glm5next_gemm`).

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
/// increasing k), the f64 value and its band's magnitude
/// `Σ_b (|D_b| + |M_b|)`.
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
        let t = dw * isum as f32;
        let t = if mins { wm[b].mul_add(sa as f32, t) } else { t };
        acc = da.mul_add(t, acc);
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
/// 0, units in increasing k, the stored value `scale·acc`), the f64 value
/// `scale · Σ_b d_a·(d0·i0 + d1·i1)` and its band's magnitude.
#[must_use]
pub fn dot32_nv(wq: &[i8], wsc: &[f32], scale: f32, a: &HostAct, col: usize) -> (f32, f64, f64) {
    let kb = a.k / 32;
    let (mut acc, mut y, mut mag) = (0.0f32, 0.0f64, 0.0f64);
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
        let tv = d0 * i0 as f32;
        let tv = d1.mul_add(i1 as f32, tv);
        acc = da.mul_add(tv, acc);
        let t = f64::from(da) * (f64::from(d0) * f64::from(i0) + f64::from(d1) * f64::from(i1));
        y += t;
        mag += t.abs();
    }
    (scale * acc, f64::from(scale) * y, f64::from(scale) * mag)
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
    // The index of the ascending table's entry nearest x, ties to the even index.
    let nearest = |table: &[f64], x: f64| {
        (1..table.len()).fold(0, |best, i| {
            let (di, db) = ((x - table[i]).abs(), (x - table[best]).abs());
            if di < db || (di == db && i % 2 == 0) {
                i
            } else {
                best
            }
        })
    };
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
