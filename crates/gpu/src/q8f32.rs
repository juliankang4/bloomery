//! Q8_0-weight and F32-weight gemv with f32 activations (package P3): the
//! two matmul sites whose weights are not K-quants. The attention `q_nope2`
//! site consumes the load-time Q8_0 requant of `wk_b` (32-value blocks), and
//! the MoE router `ffn_gate_inp` is F32. Both are precision-sensitive — a
//! flipped near-tie in the router's top-k changes the token — so activations
//! stay f32 here: the kernel dequantizes inline and accumulates in f32, and
//! the gate band is accordingly tighter than the q8_1-activation kernels'.
//!
//! Q8_0 device layout, fixed at load time (decision 4: format conversion is
//! load-time work):
//! - `qs`: row-major `u32` words, 8 words per 32-value block, code `j` in
//!   word `j/4`, byte `j%4` (little-endian), `k/4` words per row;
//! - `d`: row-major block scales, `k/32` per row, each the Q8_0 block's f16
//!   bits as the file stores them. A lane widens the scale it loads with the
//!   hardware convert (`flash::half_bits_to_f32`); widening f16 to f32 is
//!   exact, so the kernels multiply by the value the reference dequantizes
//!   with and do no f16 arithmetic.
//!
//! MXFP8 (`GgmlType::MXFP8`: OCP MX blocks of 32 F8_E4M3 values and one E8M0
//! scale) is read by the same walk ([`ByteBlocks`]) from two planes fixed at
//! load ([`mxfp8_planes`]): `qs` the value bytes laid out as Q8_0's codes,
//! and `e` the blocks' scale bytes in row order, four to a u32 word, `k/128`
//! words per row (so `k` is a multiple of 128). A byte's value is its E4M3
//! value and a block's scale `2^(E − 127)`, each exact in f32, so each term
//! is the reference's product. `mxfp8_gemv` and `mxfp8_gemv_mcol` are
//! `q8_0_gemv` and `q8_0_gemv_mcol` over those planes, under the same
//! numeric contract.
//!
//! Numeric contract, every kernel here: one warp owns one output row, each lane
//! accumulates its share of the row sequentially, one f32 multiply-add per
//! term, and the 32 lane sums are then combined by the fixed five-step
//! butterfly (xor 16, 8, 4, 2, 1). A lane's share:
//! - F32: the row's 32-value chunks — value index `32·it + L` for lane L,
//!   chunks in increasing `it`, columns in increasing order;
//! - Q8_0 (`q8_0_gemv`, the single-column decode shape, and every column of
//!   the m-column entries): whole code words — words `L, L + 32, L + 64, …` of the row in
//!   increasing order, each word's four values in byte order.
//!
//! The combination tree is a function of (k, m) only — never of the data,
//! the row index, or the grid geometry. Output layout matches the K-quant
//! gemvs, `y[r·m + c]`; `q8_0_gemv_mcol` can write token-major instead
//! ([`GemvOut`]).
//!
//! [`f32_tile_chunk`] is the F32 lane share for a warp that owns a
//! [`TILE`]-row × [`TILE`]-column tile instead of one row (the qwen3moe
//! ubatch router logits): every (row, column) pair's lane sum takes the row
//! walk's terms in the row walk's order, and each column's sums go through
//! the same butterfly, so a tile writes the row walk's bits.
//!
//! `q8_0_gemv` takes one column only; the launcher sends a Q8_0 gemv of
//! `m > 1` columns to `q8_0_gemv_mcol`, where every lane owns the whole code
//! words it owns at `m = 1` and keeps one sum per column, so column c of an `m`-column
//! launch is bit for bit the `m = 1` launch of that column, and each weight
//! word is read once for all `m` columns. `q8_0_gemv_heads_mcol` is the same
//! body over per-head activation windows.
//!
//! `q8_0_gemv_sel_f32` is the expert-select form over a stack of experts in
//! the two planes: slot `s` dots the rows of expert `sel[s]` with activation
//! column `s / slots_per_col`, each row `q8_0_gemv`'s lane walk and
//! butterfly, so a slot's rows are bit for bit `q8_0_gemv` of that expert's
//! rows on that column. An id at or past the stack's experts raises
//! [`FaultSite::ExpertId`] and writes its slot's rows NaN; no slot is the
//! host's here.
//!
//! `q8_0_gemv_ids` and `q8_0_gemv_ids_mcol` are `q8_0_gemv` and
//! `q8_0_gemv_mcol` over a row list of one matrix, read in place: launch row
//! `i` is the matrix's row `ids[i]`, its sums that row's in the plain entry,
//! bit for bit, written row-major at row `i`. An id at or past the matrix's
//! rows raises [`FaultSite::TokenId`] and writes its row NaN.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::half_bits_to_f32;
use crate::launch_u32;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::{F32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::quant::check_mxfp8;
use std::sync::Arc;

// --------------------------------------------------------------- cores
//
// Same standing as `crate::cores`: ordinary `#[inline(always)]` functions a
// per-op wrapper and a later fused kernel can both call. Slice arguments are
// the whole device buffer plus the row index — the verified core shape
// (`cores::q4k_a_chain` takes a buffer and a base), not a `DisjointSlice`.

/// Lane `lane`'s partial sums for one F32 row: `Σ_it fma(w[row·k + 32·it +
/// lane], x[c·k + 32·it + lane])`, accumulated sequentially in `it`.
/// Columns past `m_cols` stay 0.0.
///
/// # Safety
///
/// `w.len() >= (row + 1) * k`, `x.len() >= m_cols * k`, `k` a positive
/// multiple of 32, `m_cols` in 1..=8, `lane < 32`. The body reads `w` and
/// `x` unchecked within those bounds.
#[inline(always)]
pub(crate) unsafe fn f32_lane_partials(
    w: &[f32],
    x: &[f32],
    k: u32,
    row: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    // One column is the decode shape, and it gets a body of its own. See
    // `f32_lane_partial_1col`.
    if m_cols == 1 {
        // SAFETY: with m_cols 1 this function's safety contract is the
        // callee's.
        let f0 = unsafe { f32_lane_partial_1col(w, x, k, row, lane) };
        return [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let k = k as usize;
    let w_row = row * k;
    let iters = k >> 5;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    let mut it = 0usize;
    // The column guards are runtime tests, so a multiply-add sits behind
    // its column's branch, and one chunk a trip would leave one load in
    // flight. A trip takes [`LANE_UNROLL`] chunks instead: the row's words
    // once for every column, then each live column's chunks loaded before
    // its first multiply-add. Each column still adds its chunks in
    // increasing `it`; the guarded walk below is the tail.
    while it + LANE_UNROLL <= iters {
        let kk = it * 32 + lane;
        // SAFETY: w_row + kk + 224 < w_row + k <= w.len(): kk + 224 < k by
        // the loop guard (LANE_UNROLL = 8) and the safety contract.
        let wv = unsafe { f32_load_chunks(w, w_row + kk) };
        // SAFETY: kk + 224 < k <= x.len(), since m_cols >= 1.
        f0 = unsafe { f32_fold_chunks(f0, &wv, x, kk) };
        if m_cols > 1 {
            // SAFETY: k + kk + 224 < 2*k <= x.len(), since m_cols > 1.
            f1 = unsafe { f32_fold_chunks(f1, &wv, x, k + kk) };
        }
        if m_cols > 2 {
            // SAFETY: 2*k + kk + 224 < 3*k <= x.len(), since m_cols > 2.
            f2 = unsafe { f32_fold_chunks(f2, &wv, x, 2 * k + kk) };
        }
        if m_cols > 3 {
            // SAFETY: 3*k + kk + 224 < 4*k <= x.len(), since m_cols > 3.
            f3 = unsafe { f32_fold_chunks(f3, &wv, x, 3 * k + kk) };
        }
        if m_cols > 4 {
            // SAFETY: 4*k + kk + 224 < 5*k <= x.len(), since m_cols > 4.
            f4 = unsafe { f32_fold_chunks(f4, &wv, x, 4 * k + kk) };
        }
        if m_cols > 5 {
            // SAFETY: 5*k + kk + 224 < 6*k <= x.len(), since m_cols > 5.
            f5 = unsafe { f32_fold_chunks(f5, &wv, x, 5 * k + kk) };
        }
        if m_cols > 6 {
            // SAFETY: 6*k + kk + 224 < 7*k <= x.len(), since m_cols > 6.
            f6 = unsafe { f32_fold_chunks(f6, &wv, x, 6 * k + kk) };
        }
        if m_cols > 7 {
            // SAFETY: 7*k + kk + 224 < 8*k <= x.len(), since m_cols > 7.
            f7 = unsafe { f32_fold_chunks(f7, &wv, x, 7 * k + kk) };
        }
        it += LANE_UNROLL;
    }
    while it < iters {
        let kk = it * 32 + lane;
        // SAFETY: kk < k <= w.len() - w_row by the safety contract.
        let wv = unsafe { *w.get_unchecked(w_row + kk) };
        // Column 0 (always active).
        // SAFETY: kk < k <= x.len() by the safety contract (m_cols >= 1).
        f0 = f32::mul_add(wv, unsafe { *x.get_unchecked(kk) }, f0);
        // Columns 1..7: one launch-uniform guard per column so the work
        // scales with m_cols; each guard makes the column's span live.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => x.len() >= 2*k > k + kk.
            f1 = f32::mul_add(wv, unsafe { *x.get_unchecked(k + kk) }, f1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => x.len() >= 3*k > 2*k + kk.
            f2 = f32::mul_add(wv, unsafe { *x.get_unchecked(2 * k + kk) }, f2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => x.len() >= 4*k > 3*k + kk.
            f3 = f32::mul_add(wv, unsafe { *x.get_unchecked(3 * k + kk) }, f3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => x.len() >= 5*k > 4*k + kk.
            f4 = f32::mul_add(wv, unsafe { *x.get_unchecked(4 * k + kk) }, f4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => x.len() >= 6*k > 5*k + kk.
            f5 = f32::mul_add(wv, unsafe { *x.get_unchecked(5 * k + kk) }, f5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => x.len() >= 7*k > 6*k + kk.
            f6 = f32::mul_add(wv, unsafe { *x.get_unchecked(6 * k + kk) }, f6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => x.len() >= 8*k > 7*k + kk.
            f7 = f32::mul_add(wv, unsafe { *x.get_unchecked(7 * k + kk) }, f7);
        }
        it += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// The [`LANE_UNROLL`] chunk values a lane reads at `at`:
/// `v[at + 32·i]` for i = 0..8.
///
/// # Safety
///
/// `at + 224 < v.len()`.
#[inline(always)]
unsafe fn f32_load_chunks(v: &[f32], at: usize) -> [f32; LANE_UNROLL] {
    // SAFETY: at + 32·i <= at + 224 < v.len() by this fn's contract.
    unsafe {
        [
            *v.get_unchecked(at),
            *v.get_unchecked(at + 32),
            *v.get_unchecked(at + 64),
            *v.get_unchecked(at + 96),
            *v.get_unchecked(at + 128),
            *v.get_unchecked(at + 160),
            *v.get_unchecked(at + 192),
            *v.get_unchecked(at + 224),
        ]
    }
}

/// `f` with the [`LANE_UNROLL`] chunks of one column added in order:
/// `fma(w[i], x[at + 32·i], ·)` for i = 0..8, the column's loads issued
/// before the first multiply-add.
///
/// # Safety
///
/// `at + 224 < x.len()`.
#[inline(always)]
unsafe fn f32_fold_chunks(f: f32, w: &[f32; LANE_UNROLL], x: &[f32], at: usize) -> f32 {
    // SAFETY: at + 224 < x.len() by this fn's contract.
    let [x0, x1, x2, x3, x4, x5, x6, x7] = unsafe { f32_load_chunks(x, at) };
    let f = f32::mul_add(w[0], x0, f);
    let f = f32::mul_add(w[1], x1, f);
    let f = f32::mul_add(w[2], x2, f);
    let f = f32::mul_add(w[3], x3, f);
    let f = f32::mul_add(w[4], x4, f);
    let f = f32::mul_add(w[5], x5, f);
    let f = f32::mul_add(w[6], x6, f);
    f32::mul_add(w[7], x7, f)
}

/// Chunks whose loads the F32 single-column body issues before its first
/// multiply-add. The row walk is a dependent chain — each chunk's address is
/// known in advance but its multiply-add feeds the next — so with one chunk
/// in flight a lane pays one memory round trip per chunk and the row's cost
/// is the chunk count times that latency, whatever the grid geometry is.
/// Issuing this many loads first is what turns the walk into a bandwidth
/// problem; it does not touch the accumulation order. The body spells the
/// chunks out one by one, so the value is pinned to the width it is
/// written for. The Q8_0 single-column body walks whole words instead and
/// carries its own width, [`Q8_STEP_UNROLL`].
pub const LANE_UNROLL: usize = 8;
const _: () = assert!(LANE_UNROLL == 8);

/// Lane `lane`'s partial sum for one F32 row against a single activation
/// column: `Σ_it fma(w[row·k + 32·it + lane], x[32·it + lane])`, accumulated
/// sequentially in `it` — [`f32_lane_partials`]'s column 0, in the same order,
/// with [`LANE_UNROLL`] chunks' loads hoisted above the multiply-adds that
/// consume them.
///
/// # Safety
///
/// The safety contract of [`f32_lane_partials`] with `m_cols` 1: `w.len() >=
/// (row + 1) * k`, `x.len() >= k`, `k` a positive multiple of 32, `lane <
/// 32`. The body reads `w` and `x` unchecked within those bounds.
#[inline(always)]
pub unsafe fn f32_lane_partial_1col(w: &[f32], x: &[f32], k: u32, row: usize, lane: usize) -> f32 {
    let k = k as usize;
    let w_row = row * k;
    let iters = k >> 5;
    let mut f = 0.0f32;
    let mut it = 0usize;
    // Every chunk is picked by a constant index, never by a loop: an array
    // whose element a loop selects is served from a local depot whatever the
    // loop unrolls to, and the round trip that costs is the thing this body
    // exists to avoid. [`f32_fold_chunks`] owns the chunk order.
    while it + LANE_UNROLL <= iters {
        let kk = it * 32 + lane;
        // SAFETY: w_row + kk + 224 < w_row + k <= w.len(): kk + 224 < k by
        // the loop guard (LANE_UNROLL = 8) and the caller contract.
        let wv = unsafe { f32_load_chunks(w, w_row + kk) };
        // SAFETY: kk + 224 < k <= x.len() by the loop guard and the caller
        // contract.
        f = unsafe { f32_fold_chunks(f, &wv, x, kk) };
        it += LANE_UNROLL;
    }
    while it < iters {
        let kk = it * 32 + lane;
        // SAFETY: kk < k by the loop guard, so both reads are inside
        // w[w_row .. w_row + k] and x[0 .. k] by the caller contract.
        let (wv, xv) = unsafe { (*w.get_unchecked(w_row + kk), *x.get_unchecked(kk)) };
        f = f32::mul_add(wv, xv, f);
        it += 1;
    }
    f
}

/// Chunks whose loads [`f32_lane_partial_1col_w32`] issues before its first
/// multiply-add: [`LANE_UNROLL`]'s reasoning at four times the width, for a
/// kernel whose grid holds too few warps to cover a memory round trip with
/// eight chunks each. It costs a register per loaded value, so it belongs
/// to entries that call the walk once per lane; the body spells the chunks
/// out as named scalars, so the value is pinned to the width it is written
/// for.
pub const LANE_UNROLL_W32: usize = 32;
const _: () = assert!(LANE_UNROLL_W32 == 32 && LANE_UNROLL_W32.is_multiple_of(LANE_UNROLL));

/// [`f32_lane_partial_1col`] with [`LANE_UNROLL_W32`] chunks' loads hoisted
/// above their multiply-adds instead of [`LANE_UNROLL`]: the same
/// `Σ_it fma(w[row·k + 32·it + lane], x[32·it + lane])`, accumulated
/// sequentially in `it`, so the result is that function's bit for bit. The
/// chunks past the last whole 32-chunk trip run through the 8-wide trip and
/// then one at a time, continuing the same sum in the same order.
///
/// # Safety
///
/// [`f32_lane_partial_1col`]'s contract: `w.len() >= (row + 1) * k`,
/// `x.len() >= k`, `k` a positive multiple of 32, `lane < 32`. The body
/// reads `w` and `x` unchecked within those bounds.
#[inline(always)]
pub unsafe fn f32_lane_partial_1col_w32(
    w: &[f32],
    x: &[f32],
    k: u32,
    row: usize,
    lane: usize,
) -> f32 {
    let k = k as usize;
    let w_row = row * k;
    let iters = k >> 5;
    let mut f = 0.0f32;
    let mut it = 0usize;
    // Named scalars, not an array, for [`f32_lane_partial_1col`]'s reason.
    while it + LANE_UNROLL_W32 <= iters {
        let kk = it * 32 + lane;
        // SAFETY: kk + 992 < k by the loop guard (LANE_UNROLL_W32 = 32), so
        // every read is inside w[w_row .. w_row + k] and x[0 .. k], both
        // covered by the caller contract (w.len() >= (row+1)*k, x.len() >= k).
        let (
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
            w16,
            w17,
            w18,
            w19,
            w20,
            w21,
            w22,
            w23,
            w24,
            w25,
            w26,
            w27,
            w28,
            w29,
            w30,
            w31,
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15,
            x16,
            x17,
            x18,
            x19,
            x20,
            x21,
            x22,
            x23,
            x24,
            x25,
            x26,
            x27,
            x28,
            x29,
            x30,
            x31,
        ) = unsafe {
            (
                *w.get_unchecked(w_row + kk),
                *w.get_unchecked(w_row + kk + 32),
                *w.get_unchecked(w_row + kk + 64),
                *w.get_unchecked(w_row + kk + 96),
                *w.get_unchecked(w_row + kk + 128),
                *w.get_unchecked(w_row + kk + 160),
                *w.get_unchecked(w_row + kk + 192),
                *w.get_unchecked(w_row + kk + 224),
                *w.get_unchecked(w_row + kk + 256),
                *w.get_unchecked(w_row + kk + 288),
                *w.get_unchecked(w_row + kk + 320),
                *w.get_unchecked(w_row + kk + 352),
                *w.get_unchecked(w_row + kk + 384),
                *w.get_unchecked(w_row + kk + 416),
                *w.get_unchecked(w_row + kk + 448),
                *w.get_unchecked(w_row + kk + 480),
                *w.get_unchecked(w_row + kk + 512),
                *w.get_unchecked(w_row + kk + 544),
                *w.get_unchecked(w_row + kk + 576),
                *w.get_unchecked(w_row + kk + 608),
                *w.get_unchecked(w_row + kk + 640),
                *w.get_unchecked(w_row + kk + 672),
                *w.get_unchecked(w_row + kk + 704),
                *w.get_unchecked(w_row + kk + 736),
                *w.get_unchecked(w_row + kk + 768),
                *w.get_unchecked(w_row + kk + 800),
                *w.get_unchecked(w_row + kk + 832),
                *w.get_unchecked(w_row + kk + 864),
                *w.get_unchecked(w_row + kk + 896),
                *w.get_unchecked(w_row + kk + 928),
                *w.get_unchecked(w_row + kk + 960),
                *w.get_unchecked(w_row + kk + 992),
                *x.get_unchecked(kk),
                *x.get_unchecked(kk + 32),
                *x.get_unchecked(kk + 64),
                *x.get_unchecked(kk + 96),
                *x.get_unchecked(kk + 128),
                *x.get_unchecked(kk + 160),
                *x.get_unchecked(kk + 192),
                *x.get_unchecked(kk + 224),
                *x.get_unchecked(kk + 256),
                *x.get_unchecked(kk + 288),
                *x.get_unchecked(kk + 320),
                *x.get_unchecked(kk + 352),
                *x.get_unchecked(kk + 384),
                *x.get_unchecked(kk + 416),
                *x.get_unchecked(kk + 448),
                *x.get_unchecked(kk + 480),
                *x.get_unchecked(kk + 512),
                *x.get_unchecked(kk + 544),
                *x.get_unchecked(kk + 576),
                *x.get_unchecked(kk + 608),
                *x.get_unchecked(kk + 640),
                *x.get_unchecked(kk + 672),
                *x.get_unchecked(kk + 704),
                *x.get_unchecked(kk + 736),
                *x.get_unchecked(kk + 768),
                *x.get_unchecked(kk + 800),
                *x.get_unchecked(kk + 832),
                *x.get_unchecked(kk + 864),
                *x.get_unchecked(kk + 896),
                *x.get_unchecked(kk + 928),
                *x.get_unchecked(kk + 960),
                *x.get_unchecked(kk + 992),
            )
        };
        f = f32::mul_add(w0, x0, f);
        f = f32::mul_add(w1, x1, f);
        f = f32::mul_add(w2, x2, f);
        f = f32::mul_add(w3, x3, f);
        f = f32::mul_add(w4, x4, f);
        f = f32::mul_add(w5, x5, f);
        f = f32::mul_add(w6, x6, f);
        f = f32::mul_add(w7, x7, f);
        f = f32::mul_add(w8, x8, f);
        f = f32::mul_add(w9, x9, f);
        f = f32::mul_add(w10, x10, f);
        f = f32::mul_add(w11, x11, f);
        f = f32::mul_add(w12, x12, f);
        f = f32::mul_add(w13, x13, f);
        f = f32::mul_add(w14, x14, f);
        f = f32::mul_add(w15, x15, f);
        f = f32::mul_add(w16, x16, f);
        f = f32::mul_add(w17, x17, f);
        f = f32::mul_add(w18, x18, f);
        f = f32::mul_add(w19, x19, f);
        f = f32::mul_add(w20, x20, f);
        f = f32::mul_add(w21, x21, f);
        f = f32::mul_add(w22, x22, f);
        f = f32::mul_add(w23, x23, f);
        f = f32::mul_add(w24, x24, f);
        f = f32::mul_add(w25, x25, f);
        f = f32::mul_add(w26, x26, f);
        f = f32::mul_add(w27, x27, f);
        f = f32::mul_add(w28, x28, f);
        f = f32::mul_add(w29, x29, f);
        f = f32::mul_add(w30, x30, f);
        f = f32::mul_add(w31, x31, f);
        it += LANE_UNROLL_W32;
    }
    while it + LANE_UNROLL <= iters {
        let kk = it * 32 + lane;
        // SAFETY: w_row + kk + 224 < w_row + k <= w.len(): kk + 224 < k by
        // the loop guard (LANE_UNROLL = 8) and the caller contract.
        let wv = unsafe { f32_load_chunks(w, w_row + kk) };
        // SAFETY: kk + 224 < k <= x.len() by the loop guard and the caller
        // contract.
        f = unsafe { f32_fold_chunks(f, &wv, x, kk) };
        it += LANE_UNROLL;
    }
    while it < iters {
        let kk = it * 32 + lane;
        // SAFETY: kk < k by the loop guard, so both reads are inside
        // w[w_row .. w_row + k] and x[0 .. k] by the caller contract.
        let (wv, xv) = unsafe { (*w.get_unchecked(w_row + kk), *x.get_unchecked(kk)) };
        f = f32::mul_add(wv, xv, f);
        it += 1;
    }
    f
}

/// Rows and columns of the register tile [`f32_tile_chunk`] advances: a
/// warp owns `TILE` rows of an F32 weight against `TILE` activation columns,
/// each lane holding one partial sum per (column, row) pair.
pub const TILE: usize = 8;
const _: () = assert!(TILE == 8);

/// `acc[i] + w[i]·x` for each of the tile's rows, one f32 multiply-add per
/// row: the term [`f32_lane_partials`] adds to row `i`'s sum of column `x`.
#[inline(always)]
fn fma_rows(w: [f32; TILE], x: f32, acc: [f32; TILE]) -> [f32; TILE] {
    [
        f32::mul_add(w[0], x, acc[0]),
        f32::mul_add(w[1], x, acc[1]),
        f32::mul_add(w[2], x, acc[2]),
        f32::mul_add(w[3], x, acc[3]),
        f32::mul_add(w[4], x, acc[4]),
        f32::mul_add(w[5], x, acc[5]),
        f32::mul_add(w[6], x, acc[6]),
        f32::mul_add(w[7], x, acc[7]),
    ]
}

/// One 32-value chunk of lane `lane`'s [`TILE`]-row × [`TILE`]-column tile
/// of F32 partial sums, read from values the block has staged: `w` points at
/// the lane's value of the tile's first row in this chunk and `x` at its
/// value of the first column, each next row or column `stride` floats on.
/// Returns `acc` with the chunk's terms added, `acc[c][i] + w_i·x_c` — for
/// row `i` and column `c` the term [`f32_lane_partials`] adds to that row
/// and column for this chunk. So a tile advanced over a row's chunks in
/// increasing order holds, lane for lane, the row walk's partial sums, and
/// [`gemv_lane_sums`] over column `c`'s [`TILE`] sums gives its logits bit
/// for bit. Every array slot is a constant index: no loop selects one, so
/// the 64 sums and 16 values stay in registers.
///
/// # Safety
///
/// `w.add(i · stride)` and `x.add(c · stride)` for every `i, c < TILE` are
/// readable f32 values that nothing writes during the call.
#[inline(always)]
pub(crate) unsafe fn f32_tile_chunk(
    acc: [[f32; TILE]; TILE],
    w: *const f32,
    x: *const f32,
    stride: usize,
) -> [[f32; TILE]; TILE] {
    // SAFETY: i·stride for i < TILE, readable by this fn's contract.
    let wv = unsafe {
        [
            *w,
            *w.add(stride),
            *w.add(2 * stride),
            *w.add(3 * stride),
            *w.add(4 * stride),
            *w.add(5 * stride),
            *w.add(6 * stride),
            *w.add(7 * stride),
        ]
    };
    // SAFETY: c·stride for c < TILE, readable by this fn's contract.
    let xv = unsafe {
        [
            *x,
            *x.add(stride),
            *x.add(2 * stride),
            *x.add(3 * stride),
            *x.add(4 * stride),
            *x.add(5 * stride),
            *x.add(6 * stride),
            *x.add(7 * stride),
        ]
    };
    [
        fma_rows(wv, xv[0], acc[0]),
        fma_rows(wv, xv[1], acc[1]),
        fma_rows(wv, xv[2], acc[2]),
        fma_rows(wv, xv[3], acc[3]),
        fma_rows(wv, xv[4], acc[4]),
        fma_rows(wv, xv[5], acc[5]),
        fma_rows(wv, xv[6], acc[6]),
        fma_rows(wv, xv[7], acc[7]),
    ]
}

/// Values one step of the single-column Q8_0 body covers: each of the 32
/// lanes takes one whole code word — four consecutive values of one block —
/// so a step is 32 consecutive words, 128 values, four blocks, and the warp's
/// code load in a step is one contiguous 128-byte line.
const Q8_STEP: usize = 128;

/// Steps whose code, scale and activation loads the single-column Q8_0 body
/// issues before its first multiply-add: the Q8_0 counterpart of
/// [`LANE_UNROLL`], counted in 128-value steps. A step is six registers of
/// loaded data per lane (a code word, a scale, four activations), so four
/// steps are 24, and the entry keeps four 256-thread blocks per SM. The body
/// spells the steps out as named scalars, so the value is pinned to the
/// width it is written for.
pub const Q8_STEP_UNROLL: usize = 4;
const _: () = assert!(Q8_STEP_UNROLL == 4);

/// A weight of 32-value blocks, one byte a value in the `qs` words and one
/// scale a block in a plane of its own, as the f32-activation walk reads it
/// ([`blocks_lane_partial_1col`], [`blocks_lane_partials_mcol`]): Q8_0
/// ([`Q8_0Bytes`]) and MXFP8 ([`Mxfp8Bytes`]).
pub(crate) trait ByteBlocks {
    /// One element of the scale plane.
    type Scale: Copy;

    /// The value of the byte in the low eight bits of `q`, exact in f32.
    fn value(q: u32) -> f32;

    /// The element of the scale plane `d` that holds block `b`'s scale, its
    /// blocks counted from the plane's first, the rows back to back: a plain
    /// load, widened by [`ByteBlocks::scale`] once the step's loads are out.
    ///
    /// # Safety
    ///
    /// Block `b`'s scale lies inside `d`.
    unsafe fn scale_at(d: &[Self::Scale], b: usize) -> Self::Scale;

    /// Block `b`'s scale from the element [`ByteBlocks::scale_at`] loaded.
    fn scale(at: Self::Scale, b: usize) -> f32;
}

/// Q8_0 in the q8f32 planes (module doc): a byte is a signed code, a block's
/// scale its f16 bits, widened by the hardware convert.
pub(crate) struct Q8_0Bytes;

impl ByteBlocks for Q8_0Bytes {
    type Scale = u16;

    #[inline(always)]
    fn value(q: u32) -> f32 {
        q as u8 as i8 as f32
    }

    #[inline(always)]
    unsafe fn scale_at(d: &[u16], b: usize) -> u16 {
        // SAFETY: block b's scale is inside d by this fn's contract.
        unsafe { *d.get_unchecked(b) }
    }

    #[inline(always)]
    fn scale(bits: u16, _: usize) -> f32 {
        half_bits_to_f32(bits)
    }
}

/// MXFP8 in its two planes (module doc, [`mxfp8_planes`]): a byte is an
/// F8_E4M3 value, a block's scale its E8M0 byte, four to a word of the `e`
/// plane.
pub(crate) struct Mxfp8Bytes;

impl ByteBlocks for Mxfp8Bytes {
    type Scale = u32;

    /// F8_E4M3 (OCP E4M3FN): its magnitude twice the UE4M3 half
    /// ([`crate::mxfp4::ue4m3_half`], exact), its sign bit 7. The repack
    /// takes no NaN byte (0x7F, 0xFF).
    #[inline(always)]
    fn value(q: u32) -> f32 {
        let v = 2.0 * crate::mxfp4::ue4m3_half(q & 0x7F);
        f32::from_bits(v.to_bits() | ((q & 0x80) << 24))
    }

    /// Word `b / 4`, which holds block `b`'s scale byte.
    #[inline(always)]
    unsafe fn scale_at(d: &[u32], b: usize) -> u32 {
        // SAFETY: word b / 4 holds block b's scale, inside d by this fn's
        // contract.
        unsafe { *d.get_unchecked(b >> 2) }
    }

    /// `2^(E − 127)` for the scale byte `E` of block `b`, byte `b % 4` of
    /// its word: the f32 whose exponent field is `E`, and at `E = 0` the
    /// subnormal `2^-127`. The repack takes no `E = 255` (E8M0's NaN).
    #[inline(always)]
    fn scale(word: u32, b: usize) -> f32 {
        let e = (word >> (8 * (b & 3))) & 0xFF;
        f32::from_bits(if e == 0 { 0x0040_0000 } else { e << 23 })
    }
}

/// [`blocks_lane_partial_1col`] for a Q8_0 row in the q8f32 planes.
///
/// # Safety
///
/// As [`blocks_lane_partial_1col`] under [`Q8_0Bytes`]: `d.len() >= (row +
/// 1) * k/32`.
#[inline(always)]
pub unsafe fn q8_0_lane_partial_1col(
    qs: &[u32],
    d: &[u16],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    lane: usize,
) -> f32 {
    // SAFETY: this fn's contract is the body's under Q8_0Bytes.
    unsafe { blocks_lane_partial_1col::<Q8_0Bytes>(qs, d, x, k, row, x0, lane) }
}

/// [`blocks_lane_partials_mcol`] for a Q8_0 row in the q8f32 planes.
///
/// # Safety
///
/// As [`blocks_lane_partials_mcol`] under [`Q8_0Bytes`]: `d.len() >= (row +
/// 1) * k/32`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
pub unsafe fn q8_0_lane_partials_mcol<const M: usize>(
    qs: &[u32],
    d: &[u16],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    x_col: usize,
    lane: usize,
) -> [f32; 8] {
    // SAFETY: this fn's contract is the body's under Q8_0Bytes.
    unsafe { blocks_lane_partials_mcol::<Q8_0Bytes, M>(qs, d, x, k, row, x0, x_col, lane) }
}

/// Step `s` of lane `lane`'s walk: the lane's code word `qs[wq + 32s]`, the
/// scale of block `wd + 4s`, the block that word sits in
/// ([`ByteBlocks::scale`]), and the word's four activations, quad `lane +
/// 32s` of `xq`.
///
/// # Safety
///
/// Word `w = 32s + lane` of the row must exist, `w < k/4`, with `wq`, `wd`
/// and `xq` built as [`blocks_lane_partial_1col`] builds them under its
/// caller contract.
#[inline(always)]
unsafe fn block_step<F: ByteBlocks>(
    qs: &[u32],
    d: &[F::Scale],
    xq: &[F32x4],
    wq: usize,
    wd: usize,
    lane: usize,
    s: usize,
) -> (u32, f32, F32x4) {
    let b = wd + 4 * s;
    // SAFETY: w = 32s + lane < k/4 by this function's contract, so the code
    // word wq + 32s is below row*k/4 + k/4 <= qs.len(), its block b = wd + 4s
    // = row*k/32 + w/8 below row*k/32 + k/32, whose scales d holds, and quad
    // w below k/4 = xq.len().
    let (q, at, v) = unsafe {
        (
            *qs.get_unchecked(wq + 32 * s),
            F::scale_at(d, b),
            *xq.get_unchecked(lane + 32 * s),
        )
    };
    (q, F::scale(at, b), v)
}

/// `f` plus one word's four terms, in byte order: `fma(v_j·d, x_j, f)` with
/// `v_j` the value of byte `j` of `q` ([`ByteBlocks::value`]) and `x_j` lane
/// `j` of `x`. The product `v_j·d` is exact in f32 for a Q8_0 block, and for
/// an MXFP8 one whose product is a normal f32, so each term rounds once.
#[inline(always)]
fn word_dot<F: ByteBlocks>(f: f32, q: u32, d: f32, x: F32x4) -> f32 {
    let [x0, x1, x2, x3] = x.to_array();
    let f = f32::mul_add(d * F::value(q), x0, f);
    let f = f32::mul_add(d * F::value(q >> 8), x1, f);
    let f = f32::mul_add(d * F::value(q >> 16), x2, f);
    f32::mul_add(d * F::value(q >> 24), x3, f)
}

/// Lane `lane`'s partial sum for one row of 32-value byte blocks of format
/// `F` against a single activation column. The weight at value `kk` of `row`
/// is `v·d`, with `v` the value of the byte in word `qs[row·k/4 + kk/4]`,
/// byte `kk%4` ([`ByteBlocks::value`]), and `d` the scale of block
/// `row·k/32 + kk/32` ([`ByteBlocks::scale`]): the values the reference's
/// dequantizer produces. `x` is read from base `x0`, so a caller can dot
/// against a slice of a wider buffer without subslicing it. Lane L owns
/// whole code words — words `L, L + 32, L + 64, …` below `k/4`, word `w`
/// holding values `4w .. 4w + 3` — and accumulates
/// them in increasing order, each word's four values in byte order, one f32
/// multiply-add per term. [`Q8_STEP_UNROLL`] steps' loads are hoisted above
/// the multiply-adds that consume them; up to three leftover steps run as a
/// hoisted pair and a single, and a `k` that is not a multiple of 128 ends
/// with one word on each of the first `(k % 128) / 4` lanes. A row of at
/// most 128 values — one word per lane at most — skips that ladder: it is
/// the same single word, reached by one test instead of four. The
/// activations are read as one 16-byte quad per word when `x0` leaves them
/// 16-byte aligned, and as four scalars otherwise — the same values in the
/// same order, so the sum does not depend on which.
///
/// # Safety
///
/// `qs.len() >= (row + 1) * k/4`, `d` holds the scales of blocks below
/// `(row + 1) * k/32`, `x.len() >= x0 + k`, `k` a positive multiple of 32,
/// `lane < 32`. The body reads `qs`, `d` and `x` unchecked within those
/// bounds.
#[inline(always)]
pub(crate) unsafe fn blocks_lane_partial_1col<F: ByteBlocks>(
    qs: &[u32],
    d: &[F::Scale],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    lane: usize,
) -> f32 {
    let k = k as usize;
    let words = k >> 2;
    let steps = k / Q8_STEP;
    // Step 0 of the lane's walk: its word and that word's scale; a step
    // moves both on by 32 words and 4 scales.
    let wq = row * words + lane;
    let wd = row * (k >> 5) + (lane >> 3);
    // SAFETY: x0 + k <= x.len() by the caller contract.
    let xr = unsafe { x.get_unchecked(x0..x0 + k) };
    let mut f = 0.0f32;
    let Some(xq) = as_vectors::<F32x4>(xr) else {
        // Activations not 16-byte aligned at x0: the same walk, one word at a
        // time, through scalar reads.
        let mut w = lane;
        while w < words {
            let b = wd - (lane >> 3) + (w >> 3);
            // SAFETY: w < k/4 by the loop guard, so word row*k/4 + w, its
            // block row*k/32 + w/8 and values 4w .. 4w + 3 of xr are inside
            // qs, d and xr by the caller contract.
            let (q, at, xv) = unsafe {
                (
                    *qs.get_unchecked(wq - lane + w),
                    F::scale_at(d, b),
                    F32x4::new([
                        *xr.get_unchecked(4 * w),
                        *xr.get_unchecked(4 * w + 1),
                        *xr.get_unchecked(4 * w + 2),
                        *xr.get_unchecked(4 * w + 3),
                    ]),
                )
            };
            f = word_dot::<F>(f, q, F::scale(at, b), xv);
            w += 32;
        }
        return f;
    };
    // At most 128 values: at most one word per lane, step 0's — the word the
    // ladder below reaches through four tests, each its own branch region,
    // reached here through one.
    if words <= 32 {
        if lane < words {
            // SAFETY: lane < k/4 is the step helper's bound for step 0.
            let (q0, d0, v0) = unsafe { block_step::<F>(qs, d, xq, wq, wd, lane, 0) };
            f = word_dot::<F>(f, q0, d0, v0);
        }
        return f;
    }
    let mut s = 0usize;
    while s + Q8_STEP_UNROLL <= steps {
        // SAFETY: s + 4 <= steps by the loop guard, so steps s .. s + 3 are
        // whole: each lane's word in them is below 32 * steps <= k/4.
        let ((q0, d0, v0), (q1, d1, v1), (q2, d2, v2), (q3, d3, v3)) = unsafe {
            (
                block_step::<F>(qs, d, xq, wq, wd, lane, s),
                block_step::<F>(qs, d, xq, wq, wd, lane, s + 1),
                block_step::<F>(qs, d, xq, wq, wd, lane, s + 2),
                block_step::<F>(qs, d, xq, wq, wd, lane, s + 3),
            )
        };
        f = word_dot::<F>(f, q0, d0, v0);
        f = word_dot::<F>(f, q1, d1, v1);
        f = word_dot::<F>(f, q2, d2, v2);
        f = word_dot::<F>(f, q3, d3, v3);
        s += Q8_STEP_UNROLL;
    }
    if s + 2 <= steps {
        // SAFETY: s + 2 <= steps by the guard: both steps are whole.
        let ((q0, d0, v0), (q1, d1, v1)) = unsafe {
            (
                block_step::<F>(qs, d, xq, wq, wd, lane, s),
                block_step::<F>(qs, d, xq, wq, wd, lane, s + 1),
            )
        };
        f = word_dot::<F>(f, q0, d0, v0);
        f = word_dot::<F>(f, q1, d1, v1);
        s += 2;
    }
    if s < steps {
        // SAFETY: s < steps by the guard: the step is whole.
        let (q0, d0, v0) = unsafe { block_step::<F>(qs, d, xq, wq, wd, lane, s) };
        f = word_dot::<F>(f, q0, d0, v0);
    }
    if 32 * steps + lane < words {
        // SAFETY: the guard is the step helper's bound for step `steps`.
        let (q0, d0, v0) = unsafe { block_step::<F>(qs, d, xq, wq, wd, lane, steps) };
        f = word_dot::<F>(f, q0, d0, v0);
    }
    f
}

/// Step `s`'s code word and scale of lane `lane`'s walk — [`block_step`]
/// without the activations, for a body that reads several columns against
/// one word.
///
/// # Safety
///
/// As [`block_step`]: word `32s + lane` of the row exists, with `wq` and `wd`
/// built as [`blocks_lane_partials_mcol`] builds them.
#[inline(always)]
unsafe fn block_code<F: ByteBlocks>(
    qs: &[u32],
    d: &[F::Scale],
    wq: usize,
    wd: usize,
    s: usize,
) -> (u32, f32) {
    let b = wd + 4 * s;
    // SAFETY: word 32s + lane < k/4 by this function's contract, so the code
    // word wq + 32s is inside the row's k/4 words and its block wd + 4s
    // inside the row's k/32 scales, both inside qs and d by the caller
    // contract.
    let (q, at) = unsafe { (*qs.get_unchecked(wq + 32 * s), F::scale_at(d, b)) };
    (q, F::scale(at, b))
}

/// Lane `lane`'s partial sums for one row of format `F` against `M`
/// activation columns, column c read from base `x0 + c·x_col`. The lane's
/// share of the row is [`blocks_lane_partial_1col`]'s — whole code words `L, L + 32, …` in
/// increasing order, each word's four values in byte order — and each word is
/// loaded once and dotted against every column, one sum per column. Column
/// c's multiply-adds are therefore the single-column body's on column c's
/// activations, term for term and in the same order, and the partial it
/// returns is that body's bit for bit. Columns `M..8` stay 0.0.
///
/// [`Q8_STEP_UNROLL`] steps' code words and scales are loaded before the
/// first multiply-add that uses them; a column's four activation quads are
/// loaded next to its own multiply-adds, one column after another, so the
/// live set is the hoisted words plus one column's quads, not every column's.
/// Leftover whole steps run one at a time and a `k` that is not a multiple
/// of 128 ends with one word on each of the first `(k % 128) / 4` lanes. The
/// activations are read as quads when `x0` and `x_col` leave every column
/// 16-byte aligned, and as scalars otherwise — the same values in the same
/// order.
///
/// # Safety
///
/// `qs.len() >= (row + 1) * k/4`, `d` holds the scales of blocks below
/// `(row + 1) * k/32`, `x.len() >= x0 + (M - 1) * x_col + k`, `k` a positive
/// multiple of 32, `M` in 1..=8, `lane < 32`. The body reads `qs`, `d` and
/// `x` unchecked within those bounds.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
pub(crate) unsafe fn blocks_lane_partials_mcol<F: ByteBlocks, const M: usize>(
    qs: &[u32],
    d: &[F::Scale],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    x_col: usize,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let k = k as usize;
    let words = k >> 2;
    let steps = k / Q8_STEP;
    let wq = row * words + lane;
    let wd = row * (k >> 5) + (lane >> 3);
    let span = (M - 1) * x_col + k;
    // SAFETY: x0 + (M-1)*x_col + k <= x.len() by the caller contract.
    let xr = unsafe { x.get_unchecked(x0..x0 + span) };
    let (mut f0, mut f1, mut f2, mut f3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut f4, mut f5, mut f6, mut f7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let Some(xq) = as_vectors::<F32x4>(xr) else {
        // Some column not 16-byte aligned: the same walk, one word at a
        // time, through scalar reads.
        let mut w = lane;
        while w < words {
            // SAFETY: w < k/4 by the loop guard: the code word row*k/4 + w
            // and its block row*k/32 + w/8 are inside qs and d.
            let (q, dv) =
                unsafe { block_code::<F>(qs, d, wq - lane + w, wd - (lane >> 3) + (w >> 3), 0) };
            macro_rules! col_scalar {
                ($f:ident, $c:literal) => {
                    if M > $c {
                        let b = $c * x_col + 4 * w;
                        // SAFETY: b + 3 < (M-1)*x_col + k = xr.len(), since
                        // $c <= M-1 and 4w + 3 < k.
                        let v = unsafe {
                            F32x4::new([
                                *xr.get_unchecked(b),
                                *xr.get_unchecked(b + 1),
                                *xr.get_unchecked(b + 2),
                                *xr.get_unchecked(b + 3),
                            ])
                        };
                        $f = word_dot::<F>($f, q, dv, v);
                    }
                };
            }
            col_scalar!(f0, 0);
            col_scalar!(f1, 1);
            col_scalar!(f2, 2);
            col_scalar!(f3, 3);
            col_scalar!(f4, 4);
            col_scalar!(f5, 5);
            col_scalar!(f6, 6);
            col_scalar!(f7, 7);
            w += 32;
        }
        return [f0, f1, f2, f3, f4, f5, f6, f7];
    };
    // Quads between one column's base and the next.
    let cq = x_col >> 2;
    let mut s = 0usize;
    while s + Q8_STEP_UNROLL <= steps {
        // SAFETY: s + 4 <= steps by the loop guard, so steps s .. s + 3 are
        // whole: each lane's word in them is below 32 * steps <= k/4.
        let ((q0, d0), (q1, d1), (q2, d2), (q3, d3)) = unsafe {
            (
                block_code::<F>(qs, d, wq, wd, s),
                block_code::<F>(qs, d, wq, wd, s + 1),
                block_code::<F>(qs, d, wq, wd, s + 2),
                block_code::<F>(qs, d, wq, wd, s + 3),
            )
        };
        let a = lane + 32 * s;
        macro_rules! col_trip {
            ($f:ident, $c:literal) => {
                if M > $c {
                    let b = $c * cq + a;
                    // SAFETY: quad a + 96 = word 32(s+3) + lane < k/4 of
                    // column $c <= M-1, whose quads start at $c*cq, all
                    // inside xq (xr.len()/4 = (M-1)*cq + k/4).
                    let (v0, v1, v2, v3) = unsafe {
                        (
                            *xq.get_unchecked(b),
                            *xq.get_unchecked(b + 32),
                            *xq.get_unchecked(b + 64),
                            *xq.get_unchecked(b + 96),
                        )
                    };
                    $f = word_dot::<F>($f, q0, d0, v0);
                    $f = word_dot::<F>($f, q1, d1, v1);
                    $f = word_dot::<F>($f, q2, d2, v2);
                    $f = word_dot::<F>($f, q3, d3, v3);
                }
            };
        }
        col_trip!(f0, 0);
        col_trip!(f1, 1);
        col_trip!(f2, 2);
        col_trip!(f3, 3);
        col_trip!(f4, 4);
        col_trip!(f5, 5);
        col_trip!(f6, 6);
        col_trip!(f7, 7);
        s += Q8_STEP_UNROLL;
    }
    // Whole steps left over, then the partial word: every lane's word of step
    // `steps` exists only when 32 * steps + lane < k/4.
    while s <= steps {
        if s == steps && 32 * steps + lane >= words {
            break;
        }
        // SAFETY: either s < steps (a whole step) or the guard above put
        // this lane's word of step `steps` below k/4.
        let (q0, d0) = unsafe { block_code::<F>(qs, d, wq, wd, s) };
        let a = lane + 32 * s;
        macro_rules! col_one {
            ($f:ident, $c:literal) => {
                if M > $c {
                    // SAFETY: quad a = word 32s + lane < k/4 of column $c <=
                    // M-1, inside xq as in the trip above.
                    let v0 = unsafe { *xq.get_unchecked($c * cq + a) };
                    $f = word_dot::<F>($f, q0, d0, v0);
                }
            };
        }
        col_one!(f0, 0);
        col_one!(f1, 1);
        col_one!(f2, 2);
        col_one!(f3, 3);
        col_one!(f4, 4);
        col_one!(f5, 5);
        col_one!(f6, 6);
        col_one!(f7, 7);
        s += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// [`blocks_lane_partials_mcol`] for a launch-uniform `m_cols`: the body
/// instantiated for that column count. `None` for an `m_cols` outside 1..=8,
/// which the launchers never send.
///
/// # Safety
///
/// As [`blocks_lane_partials_mcol`] with `M = m_cols`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
unsafe fn blocks_lane_partials_m<F: ByteBlocks>(
    qs: &[u32],
    d: &[F::Scale],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    x_col: usize,
    m_cols: u32,
    lane: usize,
) -> Option<[f32; 8]> {
    Some(match m_cols {
        1 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 1>(qs, d, x, k, row, x0, x_col, lane) }
        }
        2 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 2>(qs, d, x, k, row, x0, x_col, lane) }
        }
        3 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 3>(qs, d, x, k, row, x0, x_col, lane) }
        }
        4 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 4>(qs, d, x, k, row, x0, x_col, lane) }
        }
        5 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 5>(qs, d, x, k, row, x0, x_col, lane) }
        }
        6 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 6>(qs, d, x, k, row, x0, x_col, lane) }
        }
        7 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 7>(qs, d, x, k, row, x0, x_col, lane) }
        }
        8 => {
            // SAFETY: M = m_cols, so this fn's contract is the body's.
            unsafe { blocks_lane_partials_mcol::<F, 8>(qs, d, x, k, row, x0, x_col, lane) }
        }
        _ => return None,
    })
}

/// The row's m sums from the per-lane partials: the fixed five-step
/// butterfly per column (`warp::reduce_sum_f32`), columns past `m_cols`
/// left at 0.0. `m_cols` must be warp-uniform — a launch-wide constant in
/// every caller.
#[inline(always)]
pub(crate) fn gemv_lane_sums(f: [f32; 8], m_cols: u32) -> [f32; 8] {
    let s0 = warp::reduce_sum_f32(f[0]);
    if m_cols == 1 {
        return [s0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let s1 = warp::reduce_sum_f32(f[1]);
    let s2 = if m_cols > 2 {
        warp::reduce_sum_f32(f[2])
    } else {
        0.0
    };
    let s3 = if m_cols > 3 {
        warp::reduce_sum_f32(f[3])
    } else {
        0.0
    };
    let s4 = if m_cols > 4 {
        warp::reduce_sum_f32(f[4])
    } else {
        0.0
    };
    let s5 = if m_cols > 5 {
        warp::reduce_sum_f32(f[5])
    } else {
        0.0
    };
    let s6 = if m_cols > 6 {
        warp::reduce_sum_f32(f[6])
    } else {
        0.0
    };
    let s7 = if m_cols > 7 {
        warp::reduce_sum_f32(f[7])
    } else {
        0.0
    };
    [s0, s1, s2, s3, s4, s5, s6, s7]
}

/// Lane 0's store of a row's `m` sums: `y[base + c·stride] = sums[c]` for
/// `c < m`.
///
/// # Safety
///
/// `1 <= m <= 8`, every slot `base + c·stride` for `c < m` inside `y`, and no other
/// thread of the launch writes any of them.
#[inline(always)]
unsafe fn store_sums(
    y: &mut DisjointSlice<f32>,
    base: usize,
    stride: usize,
    m: usize,
    sums: &[f32; 8],
) {
    // One guarded store per column with a constant index: a loop over `c`
    // would index `sums` at run time and put it in a local depot.
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store below is guarded by c < m, so its slot base +
    // c*stride is one this fn's contract puts inside y and gives to this
    // thread alone.
    unsafe {
        *y.get_unchecked_mut(base) = s0;
        if m > 1 {
            *y.get_unchecked_mut(base + stride) = s1;
        }
        if m > 2 {
            *y.get_unchecked_mut(base + 2 * stride) = s2;
        }
        if m > 3 {
            *y.get_unchecked_mut(base + 3 * stride) = s3;
        }
        if m > 4 {
            *y.get_unchecked_mut(base + 4 * stride) = s4;
        }
        if m > 5 {
            *y.get_unchecked_mut(base + 5 * stride) = s5;
        }
        if m > 6 {
            *y.get_unchecked_mut(base + 6 * stride) = s6;
        }
        if m > 7 {
            *y.get_unchecked_mut(base + 7 * stride) = s7;
        }
    }
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod q8f32_kernels {
    use super::*;

    /// F32 gemv, `y[r·m + c] = Σ_k w[r·k + k'] · x[c·k + k']`, M <= 8: one
    /// warp per row, 8 rows per 256-thread block, the K-quant gemvs'
    /// skeleton. The row guard is warp-uniform, so the cores' butterfly
    /// always sees a full warp. Summation order: the module doc's contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn f32_gemv(
        w: &[f32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows, so w.len() >= n_rows·k >= (row + 1)·k, and
        // x.len() >= m_cols·k, by the launch contract; the launcher passes k
        // a positive multiple of 32 and m_cols in 1..=8
        // (`check_gemv_geometry`); lane < 32.
        let partials = unsafe { f32_lane_partials(w, x, k, row, m_cols, lane) };
        let sums = gemv_lane_sums(partials, m_cols);
        if lane == 0 {
            // SAFETY: the slots row*m_cols + c, c < m_cols <= 8, lie in
            // row*m_cols .. (row+1)*m_cols <= n_rows*m_cols <= y.len(), the
            // launch contract's bound; only lane 0 of the row's warp writes
            // them.
            unsafe { store_sums(&mut y, row * m_cols as usize, 1, m_cols as usize, &sums) };
        }
    }

    /// Q8_0 gemv against one f32 activation column, `y[r] = Σ_k' w[r, k'] ·
    /// x[k']`, the decode shape: `f32_gemv`'s skeleton, weights decoded per
    /// the module doc's device layout, each lane's share of the row walked by
    /// [`q8_0_lane_partial_1col`]. `k` a multiple of 32 (host-validated; the
    /// contract binds the word and scale buffers through `k`: 4·qs.len() and
    /// 32·d.len() cover `n_rows * k` values exactly when 32 | k).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= k,
            y.len() >= n_rows
        )
    )]
    pub fn q8_0_gemv(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words and scales inside qs and
        // d (the contract's 4·qs.len() and 32·d.len() bounds), and x.len() >= k
        // is the one column the body reads from x0 = 0.
        let f = unsafe { q8_0_lane_partial_1col(qs, d, x, k, row, 0, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len() by the launch contract, and only
            // lane 0 of the row's warp writes slot row.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// Q8_0 gemv against f32 activations, `m_cols` in 1..=8, in the
    /// single-column lane order ([`q8_0_lane_partials_mcol`]): column c of
    /// `y` is bit for bit `q8_0_gemv`'s `m = 1` output on column c, and each
    /// weight word is read once for every column. Skeleton as `q8_0_gemv`:
    /// one warp per row, 8 rows per 256-thread block. Row r's column c lands
    /// at `y[r·y_row + c·y_col]` — `q8_0_gemv`'s `y[r·m + c]` with `(m, 1)`,
    /// token-major `y[c·n_rows + r]` with `(1, n_rows)`; the host picks one
    /// of the two, so every slot has one writer.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= (n_rows - 1) * y_row + (m_cols - 1) * y_col + 1
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn q8_0_gemv_mcol(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        y_row: u32,
        y_col: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words and scales inside qs and
        // d (the contract's 4·qs.len() and 32·d.len() bounds), and column c
        // at c*k with x.len() >= m_cols*k = (m_cols-1)*k + k.
        let Some(f) = (unsafe {
            blocks_lane_partials_m::<Q8_0Bytes>(qs, d, x, k, row, 0, k as usize, m_cols, lane)
        }) else {
            return;
        };
        let sums = gemv_lane_sums(f, m_cols);
        if lane == 0 {
            // SAFETY: the slots row*y_row + c*y_col, c < m_cols <= 8, lie at
            // or below (n_rows-1)*y_row + (m_cols-1)*y_col < y.len(), and
            // the host's layout gives each to this row's lane 0 alone.
            unsafe {
                store_sums(
                    &mut y,
                    row * y_row as usize,
                    y_col as usize,
                    m_cols as usize,
                    &sums,
                );
            }
        }
    }

    /// `q8_0_gemv` for an MXFP8 weight in its two planes (module doc): the
    /// same skeleton and lane walk under [`Mxfp8Bytes`], one f32 activation
    /// column. `k` a multiple of 128 (host-validated, so a row's scales are
    /// whole `e` words).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            128 * e.len() >= n_rows * k,
            x.len() >= k,
            y.len() >= n_rows
        )
    )]
    pub fn mxfp8_gemv(
        qs: &[u32],
        e: &[u32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words and its blocks' scales
        // inside qs and e (the contract's 4·qs.len() and 128·e.len() bounds:
        // block b's scale is in word b/4), and x.len() >= k is the one column
        // the body reads from x0 = 0.
        let f = unsafe { blocks_lane_partial_1col::<Mxfp8Bytes>(qs, e, x, k, row, 0, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len() by the launch contract, and only
            // lane 0 of the row's warp writes slot row.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// `q8_0_gemv_mcol` for an MXFP8 weight in its two planes: column c of
    /// `y` is bit for bit `mxfp8_gemv`'s `m = 1` output on column c, row r's
    /// column c at `y[c*n_rows+r]`. `k` a multiple of 128.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            128 * e.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn mxfp8_gemv_mcol(
        qs: &[u32],
        e: &[u32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts this row's words and scales inside
        // qs and e; x.len() >= m_cols*k covers every input column.
        let Some(f) = (unsafe {
            blocks_lane_partials_m::<Mxfp8Bytes>(qs, e, x, k, row, 0, k as usize, m_cols, lane)
        }) else {
            return;
        };
        let sums = gemv_lane_sums(f, m_cols);
        if lane == 0 {
            // SAFETY: row + c*n_rows < n_rows*m_cols <= y.len();
            // this row's lane 0 alone writes these slots.
            unsafe {
                store_sums(&mut y, row, n_rows as usize, m_cols as usize, &sums);
            }
        }
    }

    /// [`q8_0_gemv_mcol`] over per-head activation windows: launch row `r`
    /// belongs to head `h = r / rows_per_head` and dots weight row `r`
    /// against column c's window `x[c·x_col_stride + h·x_head_stride .. +k]`,
    /// lane 0 storing column c's sum at `y[c·y_col_stride + h·y_head_stride +
    /// y_off + r % rows_per_head]`. At `m_cols = 1` it is
    /// `q8_0_gemv_heads` bit for bit, and column c of an `m_cols` launch is
    /// the `m_cols = 1` launch of that column. `n_rows` must be `n_heads *
    /// rows_per_head`; stores are disjoint when heads and columns do not
    /// overlap in `y` (host-validated).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= (m_cols - 1) * x_col_stride + (n_heads - 1) * x_head_stride + k,
            y.len() >= (m_cols - 1) * y_col_stride + (n_heads - 1) * y_head_stride + y_off + rows_per_head
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn q8_0_gemv_heads_mcol(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        n_heads: u32,
        rows_per_head: u32,
        x_head_stride: u32,
        y_head_stride: u32,
        y_off: u32,
        m_cols: u32,
        x_col_stride: u32,
        y_col_stride: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let h = row / rows_per_head as usize;
        // As `q8_0_gemv_heads`: the contract cannot bind n_rows to
        // n_heads*rows_per_head, so a violated divisibility stops here.
        if h >= n_heads as usize {
            return;
        }
        let j = row % rows_per_head as usize;
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows bounds the row's words and scales by the
        // contract; h < n_heads and c < m_cols bound every column window,
        // x0 + c*x_col_stride + k <= (m_cols-1)*x_col_stride +
        // (n_heads-1)*x_head_stride + k <= x.len().
        let Some(f) = (unsafe {
            blocks_lane_partials_m::<Q8_0Bytes>(
                qs,
                d,
                x,
                k,
                row,
                h * x_head_stride as usize,
                x_col_stride as usize,
                m_cols,
                lane,
            )
        }) else {
            return;
        };
        let sums = gemv_lane_sums(f, m_cols);
        if lane == 0 {
            let base = h * y_head_stride as usize + y_off as usize + j;
            // SAFETY: every slot base + c*y_col_stride, c < m_cols, is inside
            // y by the launch contract, and the host-validated strides give
            // each (column, head, slot) its own address, owned by this row.
            unsafe { store_sums(&mut y, base, y_col_stride as usize, m_cols as usize, &sums) };
        }
    }

    /// Q8_0 expert-select gemv against f32 activations (module doc): thread
    /// row `n = slot · rows_per_expert + r` dots weight row `sel[slot] ·
    /// rows_per_expert + r` of the stack with activation column `slot /
    /// slots_per_col` through [`q8_0_lane_partial_1col`] and the butterfly,
    /// and lane 0 stores `y[n]`. One warp a row, eight rows a 256-thread
    /// block, as `q8_0_gemv`. The id load and its refusal are warp-uniform
    /// (a warp's lanes share `row`): an id at or past `n_experts` raises
    /// [`FaultSite::ExpertId`], reads no weight word and stores NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_experts * rows_per_expert * k,
            32 * d.len() >= n_experts * rows_per_expert * k,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            x.len() >= m_cols * k,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn q8_0_gemv_sel_f32(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        k: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if lane == 0 {
                fault.raise(FaultSite::ExpertId);
                // SAFETY: row < n_slots · rows_per_expert <= y.len(); lane 0
                // of the row's warp alone writes y[row].
                unsafe { *y.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        let row_abs = id as usize * rpe + row % rpe;
        let col = slot / slots_per_col as usize;
        // SAFETY: row_abs < n_experts · rows_per_expert puts the row's words
        // and scales inside qs and d (the contract's 4·qs.len() and 32·d.len()
        // bounds), and col < m_cols (slot < n_slots <= m_cols · slots_per_col)
        // puts the column's k values at col·k inside x.len() >= m_cols·k; the
        // host passes k a positive multiple of 32.
        let f = unsafe { q8_0_lane_partial_1col(qs, d, x, k, row_abs, col * k as usize, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_slots · rows_per_expert <= y.len() by the launch
            // contract; lane 0 of the row's warp alone writes y[row].
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// Q8_0 gemv over a row list of one matrix against one f32 activation
    /// column (module doc), the decode shape: launch row `i` dots the
    /// matrix's row `ids[i]` with `x` through [`q8_0_lane_partial_1col`] and
    /// the butterfly, and lane 0 stores `y[i]` — `q8_0_gemv`'s output row
    /// `ids[i]`, bit for bit. One warp a row, eight rows a 256-thread block,
    /// as `q8_0_gemv`. The id load and its refusal are warp-uniform: an id at
    /// or past `n_matrix` raises [`FaultSite::TokenId`], reads no weight word
    /// and stores NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_matrix * k,
            32 * d.len() >= n_matrix * k,
            ids.len() >= n_rows,
            x.len() >= k,
            y.len() >= n_rows
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn q8_0_gemv_ids(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        ids: &[u32],
        n_rows: u32,
        n_matrix: u32,
        k: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        // SAFETY: row < n_rows <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(row) };
        let lane = warp::lane_id() as usize;
        if id >= n_matrix {
            if lane == 0 {
                fault.raise(FaultSite::TokenId);
                // SAFETY: row < n_rows <= y.len() by the launch contract; lane
                // 0 of the row's warp alone writes y[row].
                unsafe { *y.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        // SAFETY: id < n_matrix puts the matrix row's words and scales inside
        // qs and d (the contract's 4·qs.len() and 32·d.len() bounds), and
        // x.len() >= k is the one column the body reads from x0 = 0; the host
        // passes k a positive multiple of 32.
        let f = unsafe { q8_0_lane_partial_1col(qs, d, x, k, id as usize, 0, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len() by the launch contract, and only
            // lane 0 of the row's warp writes slot row.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// Q8_0 gemv over a row list of one matrix (module doc): launch row `i`
    /// dots the matrix's row `ids[i]` against `m_cols` f32 activation
    /// columns in [`q8_0_lane_partials_mcol`]'s order, and lane 0 stores
    /// column c at `y[i·m_cols + c]` — `q8_0_gemv_mcol`'s row-major output of
    /// a matrix whose row `i` is that row, bit for bit. One warp a row, eight
    /// rows a 256-thread block, as `q8_0_gemv`. The id load and its refusal
    /// are warp-uniform (a warp's lanes share `row`): an id at or past
    /// `n_matrix` raises [`FaultSite::TokenId`], reads no weight word and
    /// stores NaN in the row's columns.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_matrix * k,
            32 * d.len() >= n_matrix * k,
            ids.len() >= n_rows,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn q8_0_gemv_ids_mcol(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        ids: &[u32],
        n_rows: u32,
        n_matrix: u32,
        k: u32,
        m_cols: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        // SAFETY: row < n_rows <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(row) };
        let lane = warp::lane_id() as usize;
        if id >= n_matrix {
            if lane == 0 {
                fault.raise(FaultSite::TokenId);
                // SAFETY: the slots row*m_cols + c, c < m_cols <= 8, lie in
                // row*m_cols .. (row+1)*m_cols <= n_rows*m_cols <= y.len(), the
                // launch contract's bound; only lane 0 of the row's warp
                // writes them.
                unsafe {
                    store_sums(
                        &mut y,
                        row * m_cols as usize,
                        1,
                        m_cols as usize,
                        &[f32::NAN; 8],
                    );
                }
            }
            return;
        }
        // SAFETY: id < n_matrix puts the matrix row's words and scales inside
        // qs and d (the contract's 4·qs.len() and 32·d.len() bounds), and
        // column c at c*k with x.len() >= m_cols*k = (m_cols-1)*k + k; the
        // host passes k a positive multiple of 32.
        let Some(f) = (unsafe {
            blocks_lane_partials_m::<Q8_0Bytes>(
                qs,
                d,
                x,
                k,
                id as usize,
                0,
                k as usize,
                m_cols,
                lane,
            )
        }) else {
            return;
        };
        let sums = gemv_lane_sums(f, m_cols);
        if lane == 0 {
            // SAFETY: the slots row*m_cols + c, c < m_cols <= 8, lie in
            // row*m_cols .. (row+1)*m_cols <= n_rows*m_cols <= y.len(), the
            // launch contract's bound; only lane 0 of the row's warp writes
            // them.
            unsafe { store_sums(&mut y, row * m_cols as usize, 1, m_cols as usize, &sums) };
        }
    }
}

/// The loaded P3 device module: `f32_gemv`, `q8_0_gemv` and the two
/// m-column Q8_0 entries. Owns no context and no stream — the caller passes
/// the engine stream (`Gpu::stream()`) per enqueue, so launches order with
/// the rest of the step and are capturable.
pub struct Q8F32Kernels {
    module: q8f32_kernels::LoadedModule,
}

impl Q8F32Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q8F32Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(q8f32_kernels, ctx)? };
        Ok(Q8F32Kernels { module })
    }

    /// Enqueue `y = W · x` for an F32 weight of `w.rows()` rows × `w.cols()`
    /// (= k) values, against `m` f32 activation columns of k values each.
    /// `y` holds `rows * m` f32, row-major with m outputs per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_f32_gemv(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, k) = (w.rows(), w.cols());
        check_gemv_geometry("enqueue_f32_gemv", n_rows, k, x.len(), m, y.len())?;
        let what = "enqueue_f32_gemv";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_f32_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module
            .f32_gemv(stream, &prep, w.buf(), x, n_rows, k, m, y)?;
        Ok(())
    }

    /// Enqueue `y = W · x` for a Q8_0 weight in the module doc's device
    /// layout: `qs` `rows × k/4` u32 words and `d` `rows × k/32` f16 scale
    /// bits — against `m` f32 activation columns of `k = d.cols() * 32` values
    /// each (the scale buffer's width fixes k; `qs.cols()` must equal
    /// `d.cols() * 8`). Output layout as `enqueue_f32_gemv`. `m = 1` runs
    /// `q8_0_gemv`, the decode shape; `m > 1` runs `q8_0_gemv_mcol`, so column
    /// c of the output is the `m = 1` launch of column c bit for bit.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_q8_0_gemv(
        &self,
        stream: &CudaStream,
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<u16>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if m > 1 {
            return self.enqueue_q8_0_gemv_mcol(
                stream,
                Q8_0GemvMcolArgs {
                    qs,
                    d,
                    x,
                    m,
                    out: GemvOut::RowMajor,
                    y,
                },
            );
        }
        let (n_rows, k) = q8_0_launch_dims("enqueue_q8_0_gemv", qs, d, x.len(), m, y.len())?;
        let prep =
            self.module
                .prepare_q8_0_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module
            .q8_0_gemv(stream, &prep, qs.buf(), d.buf(), x, n_rows, k, y)?;
        Ok(())
    }

    /// Enqueue [`Self::enqueue_q8_0_gemv`]'s product through
    /// `q8_0_gemv_mcol` at any `m` in 1..=8 — at `m = 1` too, where the
    /// engine launches `q8_0_gemv` instead and the two agree bit for bit —
    /// writing `y` in `out`'s layout. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_q8_0_gemv_mcol(
        &self,
        stream: &CudaStream,
        a: Q8_0GemvMcolArgs<'_>,
    ) -> Result<(), GpuError> {
        let Q8_0GemvMcolArgs {
            qs,
            d,
            x,
            m,
            out,
            y,
        } = a;
        let what = "enqueue_q8_0_gemv_mcol";
        let (n_rows, k) = q8_0_launch_dims(what, qs, d, x.len(), m, y.len())?;
        let m = launch_u32(what, "m", m)?;
        let (y_row, y_col) = match out {
            GemvOut::RowMajor => (m, 1),
            GemvOut::TokenMajor => (1, n_rows),
        };
        let prep =
            self.module
                .prepare_q8_0_gemv_mcol(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q8_0_gemv_mcol(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            n_rows,
            k,
            m,
            y_row,
            y_col,
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = W · x` for an MXFP8 weight in its two planes (module doc,
    /// [`Mxfp8Weight`]) against `m` (1..=8) f32 activation columns of `k =
    /// 4 · qs.cols()` values each, token-major (row r's column c at
    /// `y[c · n_rows + r]`): `m = 1` runs `mxfp8_gemv`, `m > 1`
    /// `mxfp8_gemv_mcol`, whose column c is the `m = 1` launch of column c
    /// bit for bit. `k` must be a multiple of 128
    /// and `e` `k / 128` words a row over the same rows; another geometry is
    /// a named error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_mxfp8_gemv(
        &self,
        stream: &CudaStream,
        a: Mxfp8GemvArgs<'_>,
    ) -> Result<(), GpuError> {
        let Mxfp8GemvArgs { w, x, m, y } = a;
        let what = "enqueue_mxfp8_gemv";
        let (n_rows, k) = w.dimensions(what)?;
        check_gemv_geometry(what, n_rows, k, x.len(), m, y.len())?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let cfg = LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0);
        if m == 1 {
            let prep = self.module.prepare_mxfp8_gemv(cfg)?;
            self.module
                .mxfp8_gemv(stream, &prep, w.qs.buf(), w.e.buf(), x, n_rows, k, y)?;
            return Ok(());
        }
        let m = launch_u32(what, "m", m)?;
        let prep = self.module.prepare_mxfp8_gemv_mcol(cfg)?;
        self.module
            .mxfp8_gemv_mcol(stream, &prep, w.qs.buf(), w.e.buf(), x, n_rows, k, m, y)?;
        Ok(())
    }

    /// Enqueue the per-head Q8_0 gemv over `m` activation columns — the
    /// `m`-column form of `StepKernels::enqueue_q8_0_gemv_heads`, the same
    /// weight planes, head layout and checks, with column c's head `h`
    /// reading `x[c*x_col_stride + h*x_head_stride .. +k]` and writing
    /// `y[c*y_col_stride + h*y_head_stride + y_off + j]` for weight row
    /// `h*rows_per_head + j`. Column c is the `m = 1` launch of that column
    /// bit for bit, and each weight word is read once for all columns. Both
    /// column strides must be multiples of 4 (every window starts 16-byte
    /// aligned), and `y_col_stride` at least one column's span of heads,
    /// `(n_heads-1)*y_head_stride + rows_per_head`, so no two columns share a
    /// slot. Asynchronous, allocation-free, capturable.
    pub fn enqueue_q8_0_gemv_heads_mcol(
        &self,
        stream: &CudaStream,
        a: Q8_0GemvHeadsMcolArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_q8_0_gemv_heads_mcol";
        let Q8_0GemvHeadsMcolArgs {
            qs,
            d,
            x,
            rows_per_head,
            x_head_stride,
            y_head_stride,
            y_off,
            m,
            x_col_stride,
            y_col_stride,
            y,
        } = a;
        let n_rows = d.rows();
        let k = d.cols() * 32;
        let refuse = |detail: String| Err(GpuError::shape(what, detail));
        if n_rows == 0 || k == 0 {
            return refuse(format!(
                "need a non-empty weight, d is {}x{}",
                d.rows(),
                d.cols()
            ));
        }
        if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return refuse(format!(
                "qs is {}x{}, want {}x{} (k/4 words per row, k = d.cols()*32 = {k})",
                qs.rows(),
                qs.cols(),
                n_rows,
                d.cols() * 8
            ));
        }
        if !(1..=8).contains(&m) {
            return refuse(format!("need 1 <= m <= 8, got m={m}"));
        }
        if rows_per_head == 0 || !n_rows.is_multiple_of(rows_per_head) {
            return refuse(format!(
                "n_rows={n_rows} is not a positive multiple of rows_per_head={rows_per_head}"
            ));
        }
        // Off the quad grid the row body takes its scalar walk: the same sums
        // at a quarter of the load width, which the launcher refuses.
        if !x_head_stride.is_multiple_of(4) || !x_col_stride.is_multiple_of(4) {
            return refuse(format!(
                "x_head_stride={x_head_stride} and x_col_stride={x_col_stride} must be multiples \
                 of 4: every window starts 16-byte aligned"
            ));
        }
        let n_heads = n_rows / rows_per_head;
        let heads_span = (n_heads - 1) * y_head_stride + rows_per_head;
        if m > 1 && y_col_stride < heads_span {
            return refuse(format!(
                "y_col_stride={y_col_stride} < (n_heads-1)*y_head_stride + rows_per_head = \
                 {heads_span}: two columns would share a slot"
            ));
        }
        let x_need = (m - 1) * x_col_stride + (n_heads - 1) * x_head_stride + k;
        if x.len() < x_need {
            return refuse(format!(
                "x.len() {} < (m-1)*x_col_stride + (n_heads-1)*x_head_stride + k = {x_need}",
                x.len()
            ));
        }
        let y_need = (m - 1) * y_col_stride + y_off + heads_span;
        if y.len() < y_need {
            return refuse(format!(
                "y.len() {} < (m-1)*y_col_stride + y_off + (n_heads-1)*y_head_stride + \
                 rows_per_head = {y_need}",
                y.len()
            ));
        }
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let prep = self
            .module
            .prepare_q8_0_gemv_heads_mcol(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q8_0_gemv_heads_mcol(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            n_rows,
            launch_u32(what, "k", k)?,
            launch_u32(what, "n_heads", n_heads)?,
            launch_u32(what, "rows_per_head", rows_per_head)?,
            launch_u32(what, "x_head_stride", x_head_stride)?,
            launch_u32(what, "y_head_stride", y_head_stride)?,
            launch_u32(what, "y_off", y_off)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "x_col_stride", x_col_stride)?,
            launch_u32(what, "y_col_stride", y_col_stride)?,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the Q8_0 expert-select gemv ([`Q8_0SelArgs`], the module
    /// doc's `q8_0_gemv_sel_f32`): slot `s` writes `y[s · rows_per_expert
    /// ..][..rows_per_expert]` as the rows of expert `sel[s]` of the stack
    /// dotted with activation column `s / slots_per_col`, `k = d.cols() · 32`
    /// values a column. An id past the stack raises on `fault` and writes its
    /// slot NaN. Asynchronous, allocation-free, capturable.
    pub fn enqueue_q8_0_gemv_sel_f32(
        &self,
        stream: &CudaStream,
        a: &Q8_0SelArgs<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_q8_0_gemv_sel_f32";
        let (qs, d) = (a.qs, a.d);
        let (rows, k) = (d.rows(), d.cols() * 32);
        let (n_slots, rpe, spc) = (a.n_slots, a.rows_per_expert, a.slots_per_col);
        if k == 0 || qs.rows() != rows || qs.cols() != d.cols() * 8 {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q8_0 planes qs {}x{} and d {}x{}: equal rows, 8 words a scale, k >= 32",
                    qs.rows(),
                    qs.cols(),
                    d.rows(),
                    d.cols()
                ),
            ));
        }
        if rpe == 0 || rows == 0 || !rows.is_multiple_of(rpe) {
            return Err(GpuError::shape(
                what,
                format!("{rows} stack rows are not a positive multiple of {rpe} rows an expert"),
            ));
        }
        let m_cols = n_slots.div_ceil(spc.max(1));
        if n_slots == 0
            || spc == 0
            || a.sel.len() < n_slots
            || a.x.len() < m_cols * k
            || y.len() < n_slots * rpe
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots (at least one) at {spc} a column (at least one): sel.len() {} \
                     >= {n_slots}, x.len() {} >= {m_cols} columns of {k}, y.len() {} >= {}",
                    a.sel.len(),
                    a.x.len(),
                    y.len(),
                    n_slots * rpe
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(8))?;
        let prep = self
            .module
            .prepare_q8_0_gemv_sel_f32(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.q8_0_gemv_sel_f32(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            a.x,
            a.sel,
            launch_u32(what, "n_experts", rows / rpe)?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "m_cols", m_cols)?,
            launch_u32(what, "slots_per_col", spc)?,
            launch_u32(what, "k", k)?,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = W[ids] · x`: rows `ids[..rows]` of a Q8_0 matrix in the
    /// module doc's device layout, read in place, against `m` f32 activation
    /// columns of `k = d.cols() * 32` values each — `m = 1` through
    /// `q8_0_gemv_ids`, the decode shape, `m > 1` through
    /// `q8_0_gemv_ids_mcol`, as [`Self::enqueue_q8_0_gemv`] splits them.
    /// `y` holds `rows * m` f32, row-major with m outputs per row, row `i`
    /// bit for bit [`Self::enqueue_q8_0_gemv`]'s output row `ids[i]`. An id
    /// at or past the matrix's rows raises [`FaultSite::TokenId`] on the
    /// device and writes its row NaN. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_q8_0_gemv_ids(
        &self,
        stream: &CudaStream,
        a: Q8_0GemvIdsArgs<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_q8_0_gemv_ids";
        let Q8_0GemvIdsArgs {
            qs,
            d,
            ids,
            rows,
            x,
            m,
            y,
        } = a;
        let (n_matrix, k) = (d.rows(), d.cols() * 32);
        if qs.rows() != n_matrix || qs.cols() != d.cols() * 8 {
            return Err(GpuError::shape(
                what,
                format!(
                    "qs is {}x{}, want {}x{} (k/4 words per row, k = d.cols()*32 = {k})",
                    qs.rows(),
                    qs.cols(),
                    n_matrix,
                    d.cols() * 8
                ),
            ));
        }
        if rows > ids.len() || rows > n_matrix {
            return Err(GpuError::shape(
                what,
                format!(
                    "{rows} listed rows: at most the map's {} ids and the matrix's {n_matrix} rows",
                    ids.len()
                ),
            ));
        }
        check_gemv_geometry(what, rows, k, x.len(), m, y.len())?;
        let n_rows = launch_u32(what, "rows", rows)?;
        let n_matrix = launch_u32(what, "n_matrix", n_matrix)?;
        let k = launch_u32(what, "k", k)?;
        let grid = LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0);
        if m == 1 {
            let prep = self.module.prepare_q8_0_gemv_ids(grid)?;
            self.module.q8_0_gemv_ids(
                stream,
                &prep,
                qs.buf(),
                d.buf(),
                x,
                ids,
                n_rows,
                n_matrix,
                k,
                fault,
                y,
            )?;
            return Ok(());
        }
        let prep = self.module.prepare_q8_0_gemv_ids_mcol(grid)?;
        self.module.q8_0_gemv_ids_mcol(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            ids,
            n_rows,
            n_matrix,
            k,
            launch_u32(what, "m", m)?,
            fault,
            y,
        )?;
        Ok(())
    }
}

/// [`Q8F32Kernels::enqueue_q8_0_gemv_ids`]'s arguments: the Q8_0 planes of
/// the whole matrix as [`Q8F32Kernels::enqueue_q8_0_gemv`] takes them, the
/// row → matrix row map and how many of its rows the launch reads, `m`
/// activation columns (1..=8), and the output.
pub struct Q8_0GemvIdsArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub ids: &'a DeviceBuffer<u32>,
    pub rows: usize,
    pub x: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q8F32Kernels::enqueue_q8_0_gemv_sel_f32`]'s arguments: a stack of
/// experts in the Q8_0 planes (`rows_per_expert` rows an expert), the f32
/// activation columns, an id a slot (read on the device at each launch),
/// the slots, and how many consecutive slots read one column (a token's
/// routed slots for a gate or an up; 1 for a down, which reads its slot's
/// own column).
pub struct Q8_0SelArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub x: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    pub slots_per_col: usize,
}

/// Where an m-column gemv writes row r's column c.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvOut {
    /// `y[r·m + c]`: every gemv's layout, `m` outputs per row.
    RowMajor,
    /// `y[c·rows + r]`: column c's `rows` outputs together — the token-major
    /// order the step's element ops read.
    TokenMajor,
}

/// [`Q8F32Kernels::enqueue_q8_0_gemv_mcol`]'s arguments: the Q8_0 planes as
/// [`Q8F32Kernels::enqueue_q8_0_gemv`] takes them, `m` activation columns
/// (1..=8), and the output layout.
pub struct Q8_0GemvMcolArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub x: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub out: GemvOut,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// An MXFP8 weight resident in its two planes ([`mxfp8_planes`]): `qs` rows
/// × `k / 4` words of value bytes, `e` rows × `k / 128` words of scale
/// bytes.
#[derive(Clone, Copy)]
pub struct Mxfp8Weight<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub e: &'a DeviceTensor<u32>,
}

impl Mxfp8Weight<'_> {
    /// Check plane alignment and matching shapes; return rows and k.
    pub(crate) fn dimensions(self, what: &'static str) -> Result<(usize, usize), GpuError> {
        let (rows, k) = (self.qs.rows(), 4 * self.qs.cols());
        if !k.is_multiple_of(128) || self.e.rows() != rows || self.e.cols() != k / 128 {
            return Err(GpuError::shape(
                what,
                format!(
                    "MXFP8 planes: k={k} must be a multiple of 128 and e {rows}x{}; got {}x{}",
                    k / 128,
                    self.e.rows(),
                    self.e.cols()
                ),
            ));
        }
        Ok((rows, k))
    }
}

/// [`Q8F32Kernels::enqueue_mxfp8_gemv`]'s arguments: the weight, `m`
/// activation columns (1..=8) of `x`, and the token-major output.
pub struct Mxfp8GemvArgs<'a> {
    pub w: Mxfp8Weight<'a>,
    pub x: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The two planes of `rows` MXFP8 rows of `k` values from their file blocks
/// (`GgmlType::MXFP8`, `bytes` exactly the rows'): `qs` the value bytes in row
/// order as little-endian words, `k / 4` a row, and `e` the blocks' scale
/// bytes in the same order four to a word, `k / 128` a row. `k` must be a
/// positive multiple of 128; a byte that holds no value
/// (`gguf::quant::check_mxfp8`) is refused by name with its row.
pub fn mxfp8_planes(bytes: &[u8], rows: usize, k: usize) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    let what = "q8f32::mxfp8_planes";
    if k == 0 || !k.is_multiple_of(128) {
        return Err(GpuError::shape(
            what,
            format!("k {k} is not a positive multiple of 128"),
        ));
    }
    let row_bytes = (k / 32)
        .checked_mul(33)
        .ok_or_else(|| GpuError::shape(what, "MXFP8 row byte count overflows"))?;
    let Some(want) = rows.checked_mul(row_bytes) else {
        return Err(GpuError::shape(
            what,
            format!("{rows} rows of {k} MXFP8 values: the byte count overflows"),
        ));
    };
    if bytes.len() != want {
        return Err(GpuError::shape(
            what,
            format!(
                "{} bytes for {rows} rows of {k} MXFP8 values, want {want}",
                bytes.len()
            ),
        ));
    }
    let mut qs = Vec::with_capacity(rows * k / 4);
    let mut e = Vec::with_capacity(rows * k / 128);
    for (r, row) in bytes.chunks_exact(row_bytes).enumerate() {
        check_mxfp8(row).map_err(|why| GpuError::shape(what, format!("row {r}: {why}")))?;
        for four in row.as_chunks::<132>().0 {
            let mut word = 0u32;
            for (i, blk) in four.as_chunks::<33>().0.iter().enumerate() {
                word |= u32::from(blk[0]) << (8 * i);
                for c in blk[1..].as_chunks::<4>().0 {
                    qs.push(u32::from_le_bytes(*c));
                }
            }
            e.push(word);
        }
    }
    Ok((qs, e))
}

/// [`Q8F32Kernels::enqueue_q8_0_gemv_heads_mcol`]'s arguments. The strides
/// and the offset count f32 elements, `rows_per_head` weight rows; `m` is
/// the column count, 1..=8.
pub struct Q8_0GemvHeadsMcolArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub x: &'a DeviceBuffer<f32>,
    pub rows_per_head: usize,
    pub x_head_stride: usize,
    pub y_head_stride: usize,
    pub y_off: usize,
    pub m: usize,
    pub x_col_stride: usize,
    pub y_col_stride: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The two Q8_0 gemv launchers' shared checks: geometry as
/// [`check_gemv_geometry`] with `k = d.cols() * 32`, and `qs` the matching
/// word plane. Returns `(n_rows, k)` as the launch's `u32`s; `m` is checked
/// here but converted only by the launcher that passes it on.
fn q8_0_launch_dims(
    what: &'static str,
    qs: &DeviceTensor<u32>,
    d: &DeviceTensor<u16>,
    x_len: usize,
    m: usize,
    y_len: usize,
) -> Result<(u32, u32), GpuError> {
    let n_rows = d.rows();
    let k = d.cols() * 32;
    check_gemv_geometry(what, n_rows, k, x_len, m, y_len)?;
    if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
        return Err(GpuError::shape(
            what,
            format!(
                "qs is {}x{}, want {}x{} (k/4 words per row, k = d.cols()*32 = {k})",
                qs.rows(),
                qs.cols(),
                n_rows,
                d.cols() * 8
            ),
        ));
    }
    Ok((
        launch_u32(what, "n_rows", n_rows)?,
        launch_u32(what, "k", k)?,
    ))
}

/// Reject geometry the gemvs' launch contracts do not cover: every body
/// walks the row in 32-value blocks, so k must be a positive multiple of 32,
/// rows >= 1, and all support 1..=8 columns.
fn check_gemv_geometry(
    what: &'static str,
    n_rows: usize,
    k: usize,
    x_len: usize,
    m: usize,
    y_len: usize,
) -> Result<(), GpuError> {
    if n_rows == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(GpuError::shape(
            what,
            format!("need n_rows >= 1 and k a positive multiple of 32, got n_rows={n_rows} k={k}"),
        ));
    }
    if !(1..=8).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("need 1 <= m <= 8, got m={m}"),
        ));
    }
    if x_len < m * k {
        return Err(GpuError::shape(
            what,
            format!("x.len() {x_len} < m*k = {}", m * k),
        ));
    }
    if y_len < n_rows * m {
        return Err(GpuError::shape(
            what,
            format!("y.len() {y_len} < n_rows*m = {}", n_rows * m),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ByteBlocks, Mxfp8Bytes, f32_lane_partial_1col_w32, f32_lane_partials, mxfp8_planes,
        q8_0_lane_partial_1col, q8_0_lane_partials_mcol,
    };
    use gguf::quant::{GgmlType, dequant_row};

    /// The card's MXFP8 decode is the host's: blocks of every scale byte
    /// below 0xFF and every value byte that holds a value, through
    /// [`mxfp8_planes`] and the walk's two reads (`Mxfp8Bytes::value` on a
    /// `qs` word, `Mxfp8Bytes::scale` on the `e` plane), give
    /// `gguf::quant::dequant_row`'s values bit for bit, subnormal products
    /// and the sign of zero included. The planes refuse a NaN byte, and a
    /// scale a value of its block leaves f32's range at, naming the row.
    #[test]
    fn mxfp8_card_decode_is_the_hosts() {
        let ty = GgmlType::MXFP8;
        let values: Vec<u8> = (0..=u8::MAX).filter(|b| b & 0x7F != 0x7F).collect();
        let (rows, k) = (16, 512);
        let mut bytes = Vec::new();
        for b in 0..rows * k / 32 {
            let scale = u8::try_from(b).unwrap_or(127).min(254);
            // The values whose product with the scale is a finite f32.
            let held: Vec<u8> = values
                .iter()
                .copied()
                .filter(|&v| {
                    (Mxfp8Bytes::value(u32::from(v)) * Mxfp8Bytes::scale(u32::from(scale), 0))
                        .is_finite()
                })
                .collect();
            bytes.push(scale);
            for j in 0..32 {
                bytes.push(held[(32 * b + j) % held.len()]);
            }
        }
        let (qs, e) = mxfp8_planes(&bytes, rows, k).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((qs.len(), e.len()), (rows * k / 4, rows * k / 128));
        let mut want = vec![0.0f32; rows * k];
        dequant_row(ty, &bytes, &mut want).unwrap_or_else(|e| panic!("{e}"));
        for (v, w) in want.iter().enumerate() {
            // SAFETY: block v / 32 < rows * k / 32, whose scales e holds.
            let d = Mxfp8Bytes::scale(unsafe { Mxfp8Bytes::scale_at(&e, v / 32) }, v / 32);
            let got = Mxfp8Bytes::value(qs[v / 4] >> (8 * (v % 4))) * d;
            assert_eq!(got.to_bits(), w.to_bits(), "value {v}");
        }
        bytes[528 + 33 + 3] = 0xFF;
        match mxfp8_planes(&bytes, rows, k) {
            Err(err) => {
                let err = err.to_string();
                assert!(err.contains("row 1") && err.contains("value 34"), "{err}");
            }
            Ok(_) => panic!("a NaN value byte repacked"),
        }
        bytes[528 + 33 + 3] = 0x01;
        bytes[528 + 33] = 254;
        bytes[528 + 33 + 4] = 0x40;
        match mxfp8_planes(&bytes, rows, k) {
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains("row 1") && err.contains("f32's range"),
                    "{err}"
                );
            }
            Ok(_) => panic!("2 · 2^127 repacked"),
        }
        // 2^62 rows of 132 bytes are 2^64 · 33 bytes: a product that wraps
        // to the empty input's length.
        match mxfp8_planes(&[], 1 << 62, 128) {
            Err(err) => assert!(err.to_string().contains("overflows"), "{err}"),
            Ok(_) => panic!("2^62 rows sized from an empty input"),
        }
        let err = mxfp8_planes(&[0; 116], 1, 17_887_751_829_051_686_528)
            .expect_err("an overflowing MXFP8 row width");
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    /// The single-column Q8_0 body sums every value of the row exactly once,
    /// for every k the launchers accept (a positive multiple of 32) up to two
    /// hoisted trips — every combination of trips, pair, single and partial
    /// word, and the one-word rows of at most 128 values — on the quad path
    /// and on the unaligned scalar path. The codes, scales (f16 bits of 1, 2
    /// and 3) and activations are small integers, so every lane sum and the
    /// total are exact in f32 whatever the order: a skipped or doubled value
    /// shows as a different integer.
    #[test]
    fn q8_0_1col_sums_each_value_once() {
        let code = |r: usize, v: usize| ((r * 131 + v * 37) % 255) as i32 - 127;
        // Block b's scale, and its f16 bits: 1.0, 2.0 and 3.0.
        let scale = |b: usize| (b % 3 + 1) as i64;
        const F16_1_2_3: [u16; 3] = [0x3c00, 0x4000, 0x4200];
        for k in (32..=1024).step_by(32) {
            let rows = 3;
            let words = k / 4;
            let mut qs = vec![0u32; rows * words];
            for r in 0..rows {
                for v in 0..k {
                    qs[r * words + v / 4] |= u32::from(code(r, v) as u8) << (8 * (v % 4));
                }
            }
            let d: Vec<u16> = (0..rows * k / 32).map(|b| F16_1_2_3[b % 3]).collect();
            let x: Vec<f32> = (0..k + 8).map(|i| ((i * 7) % 17) as f32 - 8.0).collect();
            // The first element offset at which x is 16-byte aligned takes the
            // quad path; one past it, the scalar path.
            let aligned = (16 - x.as_ptr() as usize % 16) % 16 / 4;
            for x0 in [aligned, aligned + 1] {
                for r in 0..rows {
                    let want: i64 = (0..k)
                        .map(|v| {
                            i64::from(code(r, v)) * scale(r * k / 32 + v / 32) * x[x0 + v] as i64
                        })
                        .sum();
                    let got: f32 = (0..32)
                        .map(|lane| {
                            // SAFETY: qs and d hold `rows` whole rows of k
                            // values, x holds k + 8 >= x0 + k (x0 <= 4), k is
                            // a multiple of 32 and lane < 32: the body's
                            // contract for r < rows.
                            unsafe { q8_0_lane_partial_1col(&qs, &d, &x, k as u32, r, x0, lane) }
                        })
                        .sum();
                    assert_eq!(got, want as f32, "k={k} row={r} x0={x0}");
                }
            }
        }
    }

    /// Column c of the m-column body is the single-column body on column c,
    /// bit for bit, lane by lane — for every column count, every k the
    /// launchers accept up to two hoisted trips (each tail the walk can end
    /// in), a column stride equal to k and one past it, on the quad path and
    /// on the unaligned scalar path. The values are not small integers: the
    /// sums round, so a term out of order shows as different bits.
    #[test]
    fn q8_0_mcol_column_is_1col() {
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for k in (32..=1024).step_by(32) {
            let rows = 2;
            let words = k / 4;
            let qs: Vec<u32> = (0..rows * words).map(|_| next()).collect();
            // f16 bits of scales in [2^-8, 2^1): exponent 7..=15, any mantissa.
            let d: Vec<u16> = (0..rows * k / 32)
                .map(|_| {
                    let r = next();
                    (((7 + (r >> 16) % 9) << 10) | (r & 0x3ff)) as u16
                })
                .collect();
            for x_col in [k, k + 4] {
                let len = 7 * x_col + k + 8;
                let x: Vec<f32> = (0..len)
                    .map(|_| (next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
                    .collect();
                let aligned = (16 - x.as_ptr() as usize % 16) % 16 / 4;
                for x0 in [aligned, aligned + 1] {
                    for r in 0..rows {
                        for lane in 0..32 {
                            // SAFETY: qs and d hold `rows` rows of k values,
                            // x holds x0 + 7*x_col + k (x0 <= 4 < 8), k is a
                            // multiple of 32 and lane < 32: both bodies'
                            // contracts for every column count up to 8.
                            let got: [[f32; 8]; 8] = unsafe {
                                [
                                    q8_0_lane_partials_mcol::<1>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<2>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<3>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<4>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<5>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<6>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<7>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                    q8_0_lane_partials_mcol::<8>(
                                        &qs, &d, &x, k as u32, r, x0, x_col, lane,
                                    ),
                                ]
                            };
                            for (mi, cols) in got.iter().enumerate() {
                                for (c, &v) in cols.iter().enumerate() {
                                    let want = if c <= mi {
                                        // SAFETY: as above, column c's base
                                        // x0 + c*x_col leaves k values in x.
                                        unsafe {
                                            q8_0_lane_partial_1col(
                                                &qs,
                                                &d,
                                                &x,
                                                k as u32,
                                                r,
                                                x0 + c * x_col,
                                                lane,
                                            )
                                        }
                                    } else {
                                        0.0
                                    };
                                    assert_eq!(
                                        v.to_bits(),
                                        want.to_bits(),
                                        "k={k} x_col={x_col} x0={x0} row={r} lane={lane} m={} col={c}",
                                        mi + 1
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Column c of the F32 body is a scalar walk of `fma` over column c's
    /// chunks in increasing order, bit for bit, lane by lane: for every
    /// column count 1..=8 (1 is the single-column body, which the 32-chunk
    /// single-column body matches too) and every k up to 1536 (each width's
    /// unrolled trips and each tail the walks can end in). The values round,
    /// so a chunk added out of order shows as other bits.
    #[test]
    fn f32_walks_are_the_scalar_walk() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        for k in (32..=1536).step_by(32) {
            let rows = 2;
            let w: Vec<f32> = (0..rows * k).map(|_| next()).collect();
            let x: Vec<f32> = (0..8 * k).map(|_| next()).collect();
            for m in 1..=8u32 {
                for r in 0..rows {
                    for lane in 0..32 {
                        // SAFETY: w holds `rows` rows of k values, x holds
                        // 8·k >= m·k, k is a multiple of 32, m <= 8 and
                        // lane < 32.
                        let got = unsafe { f32_lane_partials(&w, &x, k as u32, r, m, lane) };
                        for (c, &v) in got.iter().enumerate() {
                            let want = if c < m as usize {
                                (0..k / 32).fold(0.0f32, |acc, it| {
                                    f32::mul_add(
                                        w[r * k + 32 * it + lane],
                                        x[c * k + 32 * it + lane],
                                        acc,
                                    )
                                })
                            } else {
                                0.0
                            };
                            assert_eq!(
                                v.to_bits(),
                                want.to_bits(),
                                "k={k} m={m} row={r} lane={lane} col={c}"
                            );
                        }
                        if m == 1 {
                            // SAFETY: w covers row r and x covers k values;
                            // k is a positive multiple of 32 and lane < 32.
                            let w32 =
                                unsafe { f32_lane_partial_1col_w32(&w, &x, k as u32, r, lane) };
                            assert_eq!(
                                w32.to_bits(),
                                got[0].to_bits(),
                                "w32 k={k} row={r} lane={lane}"
                            );
                        }
                    }
                }
            }
        }
    }
}
