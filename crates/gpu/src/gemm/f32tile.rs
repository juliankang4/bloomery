//! The wide F32 product: an F32 weight of any row count against up to a
//! ubatch of f32 columns, as register tiles — `f32_tile_gemm` (declared in
//! `kernels32.rs`), for the F32 projections a wide prompt call runs (a
//! linear-attention layer's β·α rows, a selecting attention's indexer
//! projections) where the decode path runs `q8f32::f32_gemv`.
//!
//! Contract: every output is bit for bit `f32_gemv`'s for its (row,
//! column). The body is `qwen3moe_router_logits`' with the row count a
//! launch argument: a block of sixteen warps covers [`TILE_COLS`] columns
//! and [`TILE_ROWS`] rows, each warp eight rows by eight columns, the rows'
//! and columns' next 64 values staged in shared memory with `cp.async`,
//! three stages deep; each lane advances its 64 sums over the staged 32-value
//! chunks in increasing order (`q8f32::f32_tile_chunk`), which takes each
//! (row, column) pair's terms in `f32_gemv`'s row walk order (`q8f32.rs`'s
//! module doc says why that keeps its bits), and each column's eight sums go
//! through the same butterfly (`q8f32::gemv_lane_sums`). A row past the last
//! is staged as the last row and a column past the last as the last column;
//! their sums are not stored. The output is column-major, `y[c · rows + r]`
//! — the GEMM families' `y[s][n]`, not `f32_gemv`'s `y[r · m + c]`.
//!
//! The rows come from a [`TileRows`]: an F32 weight's values are copied into
//! the stage as they are ([`F32Rows`], `cp.async`), and an E4M3 weight's
//! planes (`q8f32::e4m3_planes`) are decoded into it ([`ByteRows`]: each
//! piece's value word and its block's scale loaded, the four values formed
//! as the planes' `ByteBlocks` gives them, exactly the host's dequant, and
//! stored): `mxfp8_tile_gemm` and `fp8_b128_tile_gemm`. Either way the warps
//! read f32 values from the stage, so an E4M3 tile is bit for bit `f32_gemv`
//! over the host's dequantized rows. BF16 words are widened into the same stage
//! ([`Bf16Rows`], `q8f32`'s `bf16_tile_gemm`), bit for bit the F32 tile over those values.
//!
//! GB10 prompt sites select [`bf16_mma_body`] for BF16 and E4M3 weights.
//! It rounds activations and decoded weights to BF16 and accumulates in F32.
//! Exceptional operands replay the tile through the exact F32 owner, preserving
//! its exponent range and reduction order. The exact tile entries stay available.

use super::gemm32::{GEMM32_GROUP, raster32};
use super::kernels32::Gemm32Kernels;
use crate::q8f32::{
    Bf16Words, ByteBlocks, E4m3Scales, E4m3Weight, TILE, f32_tile_chunk, gemv_lane_sums,
};
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use core::marker::PhantomData;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::wmma::mma_m16n8k16_f32_bf16;
use cuda_device::{DisjointSlice, thread, warp};

/// Columns and rows one block covers.
pub(super) const TILE_COLS: usize = 32;
pub(super) const TILE_ROWS: usize = 32;
/// Threads of a block: sixteen warps, [`TILE_ROWS`]` / 8` row groups by
/// [`TILE_COLS`]` / 8` column groups.
pub(super) const F32T_THREADS: usize = 512;
pub(crate) const F32T_THREADS_U32: u32 = F32T_THREADS as u32;
const ROW_GROUPS: usize = TILE_ROWS / TILE;
const COL_GROUPS: usize = TILE_COLS / TILE;
/// 32-value chunks one stage holds of every row and column, and the values
/// of one such line: K must be a multiple of it.
const STAGE_CHUNKS: usize = 2;
/// Values in a staged F32 tile line; K must be a positive multiple.
pub const LINE: usize = 32 * STAGE_CHUNKS;
/// Stages in shared memory: the one the warps read and two being copied in.
pub(crate) const STAGES: usize = 3;
/// One stage: every row's line, then every column's.
pub(crate) const STAGE_FLOATS: usize = (TILE_ROWS + TILE_COLS) * LINE;
/// 16-byte copies per line.
const LINE_COPIES: usize = LINE / 4;

const _: () = assert!(ROW_GROUPS * TILE == TILE_ROWS && COL_GROUPS * TILE == TILE_COLS);
const _: () = assert!(ROW_GROUPS * COL_GROUPS * 32 == F32T_THREADS && F32T_THREADS == 512);
// Every thread copies one piece of one row line and of one column line.
const _: () = assert!(TILE_ROWS * LINE_COPIES == F32T_THREADS && TILE_COLS == TILE_ROWS);
const _: () = assert!(4 * STAGES * STAGE_FLOATS <= 48 * 1024);

/// A tile's weight rows: how one thread's piece of a row line reaches the
/// stage.
pub(crate) trait TileRows: Copy {
    /// Values `at .. at + 4` of the weight (its rows back to back, `at` a
    /// multiple of 4) into the stage's 16 bytes at `dst`, readable by every
    /// warp of the block after the next `cp_async_wait_group` that completes
    /// this thread's last committed group and the barrier after it.
    ///
    /// # Safety
    ///
    /// The four values lie inside the weight's rows; `dst` is 16 bytes of
    /// shared memory, 16-byte aligned, that this thread alone writes and no
    /// warp reads before that barrier.
    unsafe fn stage(self, dst: *mut f32, at: usize);
}

/// An F32 weight's values, copied as they are.
#[derive(Clone, Copy)]
pub(super) struct F32Rows<'a>(pub &'a [f32]);

impl TileRows for F32Rows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: the 16 source bytes are values at .. at + 4 of the weight,
        // 16-byte aligned (at a multiple of 4, the base 16-byte aligned: the
        // body's contract), and dst is this thread's own 16 bytes, by this
        // fn's contract; the copy completes at the wait that the contract
        // names.
        unsafe { cp_async_cg_16(dst.cast::<u32>(), self.0.as_ptr().add(at).cast::<u32>()) };
    }
}

/// An E4M3 weight's two planes, decoded as they are staged by `F`, the
/// planes' [`ByteBlocks`].
pub(super) struct ByteRows<'a, F> {
    qs: &'a [u32],
    e: &'a [u32],
    blocks: PhantomData<F>,
}

impl<'a, F> ByteRows<'a, F> {
    #[inline(always)]
    pub(super) fn new(qs: &'a [u32], e: &'a [u32]) -> Self {
        ByteRows {
            qs,
            e,
            blocks: PhantomData,
        }
    }
}

impl<F> Clone for ByteRows<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for ByteRows<'_, F> {}

impl<F: ByteBlocks<Scale = u32>> TileRows for ByteRows<'_, F> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: values at .. at + 4 are the bytes of word at / 4, and their
        // block at / 32 has its scale in the `e` plane: both inside the
        // planes because the values lie inside the weight's rows (this fn's
        // contract).
        let (q, e) = unsafe {
            (
                *self.qs.get_unchecked(at >> 2),
                F::scale_at(self.e, at >> 5),
            )
        };
        let d = F::scale(e, at >> 5);
        // SAFETY: dst is this thread's own 16 bytes of the stage (this fn's
        // contract); the stores are plain, published by the barrier.
        unsafe {
            *dst = F::value(q) * d;
            *dst.add(1) = F::value(q >> 8) * d;
            *dst.add(2) = F::value(q >> 16) * d;
            *dst.add(3) = F::value(q >> 24) * d;
        }
    }
}

/// A BF16 weight's words (`DevWeight::Bf16`), widened as they are staged.
#[derive(Clone, Copy)]
pub(crate) struct Bf16Rows<'a>(pub &'a [u32]);

impl TileRows for Bf16Rows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: values at .. at + 4 are the halves of words at / 2 and
        // at / 2 + 1 (at a multiple of 4), inside the words because the
        // values lie inside the weight's rows (this fn's contract).
        let (lo, hi) = unsafe {
            (
                *self.0.get_unchecked(at >> 1),
                *self.0.get_unchecked((at >> 1) + 1),
            )
        };
        // SAFETY: dst is this thread's own 16 bytes of the stage (this fn's
        // contract); the stores are plain, published by the barrier.
        unsafe {
            *dst = Bf16Words::widen(lo, false);
            *dst.add(1) = Bf16Words::widen(lo, true);
            *dst.add(2) = Bf16Words::widen(hi, false);
            *dst.add(3) = Bf16Words::widen(hi, true);
        }
    }
}

/// Lane 0's stores of the first `live` of one column's [`TILE`] row sums,
/// `y[base + i] = sums[i]`, each store guarded by a constant index.
///
/// # Safety
///
/// `live <= TILE`, `base + live <= y.len()`, and no other thread writes
/// those slots.
#[inline(always)]
unsafe fn store_live(y: &mut DisjointSlice<f32>, base: usize, sums: &[f32; TILE], live: usize) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store is guarded by i < live, so its slot base + i lies
    // inside y and belongs to this thread alone by this fn's contract.
    unsafe {
        if live > 0 {
            *y.get_unchecked_mut(base) = s0;
        }
        if live > 1 {
            *y.get_unchecked_mut(base + 1) = s1;
        }
        if live > 2 {
            *y.get_unchecked_mut(base + 2) = s2;
        }
        if live > 3 {
            *y.get_unchecked_mut(base + 3) = s3;
        }
        if live > 4 {
            *y.get_unchecked_mut(base + 4) = s4;
        }
        if live > 5 {
            *y.get_unchecked_mut(base + 5) = s5;
        }
        if live > 6 {
            *y.get_unchecked_mut(base + 6) = s6;
        }
        if live > 7 {
            *y.get_unchecked_mut(base + 7) = s7;
        }
    }
}

/// The product of `n` columns over `rows` weight rows (module doc): block
/// `b` covers columns `32·(b / B) ..` and rows `32·(b % B) ..`, `B =
/// ⌈rows / 32⌉`; writes `y[c · rows + r]`.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the block is
/// [`F32T_THREADS`] wide and the grid `⌈n / 32⌉ · B` blocks; `sh` is the
/// block's own `STAGES · STAGE_FLOATS` shared values, 16-byte aligned;
/// `rows >= 1`, `n >= 1`, `k` a positive multiple of [`LINE`], `w` holding
/// `rows` rows of `k` values ([`TileRows::stage`]'s bound; an F32 weight
/// 16-byte aligned), `x.len() >= n·k`, 16-byte aligned, `y.len() >= rows·n`.
#[inline(always)]
pub(crate) unsafe fn f32_tile_body<W: TileRows>(
    w: W,
    x: &[f32],
    rows: u32,
    k: u32,
    n: u32,
    y: &mut DisjointSlice<f32>,
    sh: *mut f32,
) {
    let rows = rows as usize;
    let n = n as usize;
    let row_blocks = rows.div_ceil(TILE_ROWS);
    let b = thread::blockIdx_x() as usize;
    let c0 = (b / row_blocks) * TILE_COLS;
    let row0 = (b % row_blocks) * TILE_ROWS;
    if c0 >= n {
        return; // block-uniform, before the first barrier
    }
    // SAFETY: the wrapper supplies this block's checked tile and all 512 threads.
    unsafe { f32_tile_at::<W, F32T_THREADS>(w, x, (rows, k as usize, n), y, sh, (row0, c0, 0)) };
}

/// Exact tile owner, with 256 threads walking one row half or 512 walking both.
///
/// # Safety
///
/// All THREADS threads call converged. THREADS is 256 or 512; row_half is
/// 0 for 512 and 0 or 1 for 256. Rows and n are positive; kk is a positive
/// multiple of LINE. The weight holds rows*kk values, x holds n*kk aligned
/// values, and y holds rows*n outputs. The aligned shared allocation holds
/// STAGES*STAGE_FLOATS values. No other block writes this output tile.
#[inline(always)]
unsafe fn f32_tile_at<W: TileRows, const THREADS: usize>(
    w: W,
    x: &[f32],
    (rows, kk, n): (usize, usize, usize),
    y: &mut DisjointSlice<f32>,
    sh: *mut f32,
    (row0, c0, row_half): (usize, usize, usize),
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let (rg, cg) = (
        (tid / 32) / COL_GROUPS + row_half * 2,
        (tid / 32) % COL_GROUPS,
    );
    let stages = kk / LINE;

    // This thread's copies of every stage: piece `piece` of row line `line`
    // and of column line `line`, a row past the last staged as the last and
    // a column past the last as the last.
    macro_rules! stage_copy {
        ($s:expr, $buf:expr) => {{
            let (s, buf): (usize, usize) = ($s, $buf);
            for i in 0..F32T_THREADS / THREADS {
                thread::__unroll_config::<0>();
                let copy = tid + i * THREADS;
                let line = copy / LINE_COPIES;
                let piece = 4 * (copy % LINE_COPIES);
                let w_off = (row0 + line).min(rows - 1) * kk + piece;
                let x_off = (c0 + line).min(n - 1) * kk + piece;
                let w_dst = line * LINE + piece;
                let x_dst = (TILE_ROWS + line) * LINE + piece;
                // SAFETY: buf < STAGES selects this block's allocated stage.
                let dst = unsafe { sh.add(buf * STAGE_FLOATS) };
                // SAFETY: the copies partition the checked tile's aligned
                // weight pieces; no warp reads them until the wait/barrier.
                unsafe { w.stage(dst.add(w_dst), w_off + s * LINE) };
                // SAFETY: x_off selects a live aligned column piece and x_dst
                // is this thread's disjoint aligned piece of the stage.
                unsafe {
                    cp_async_cg_16(
                        dst.add(x_dst).cast::<u32>(),
                        x.as_ptr().add(x_off + s * LINE).cast::<u32>(),
                    )
                };
            }
        }};
    }

    stage_copy!(0, 0);
    // SAFETY: commits this thread's copies above as one group.
    unsafe { cp_async_commit_group() };
    if stages > 1 {
        stage_copy!(1, 1);
    }
    // SAFETY: commits this thread's copies of stage 1 as the second group,
    // empty when `stages` is 1; the loop's wait counts it as stage 1's either
    // way.
    unsafe { cp_async_commit_group() };

    let mut acc = [[0.0f32; TILE]; TILE];
    let mut s = 0usize;
    let mut buf = 0usize;
    while s < stages {
        // SAFETY: one group per stage was committed before this wait (two
        // ahead of the loop, one per earlier pass), so leaving the newest
        // pending completes stage s's. The barrier then publishes every
        // thread's copies of it, and tells that every warp is done reading
        // stage s − 1, whose buffer the copy below refills.
        unsafe { cp_async_wait_group(1) };
        thread::sync_threads();
        if s + 2 < stages {
            stage_copy!(s + 2, if buf == 0 { STAGES - 1 } else { buf - 1 });
        }
        // SAFETY: one group per pass, empty near the end.
        unsafe { cp_async_commit_group() };
        // SAFETY: stage buf's row lines rg·TILE + i and column lines
        // TILE_ROWS + cg·TILE + c (i, c < TILE) hold values 32·j + lane of
        // chunks j < STAGE_CHUNKS, published by the barrier above and not
        // refilled before the next pass's barrier.
        unsafe {
            let base = sh.add(buf * STAGE_FLOATS).cast_const();
            let wp = base.add(rg * TILE * LINE + lane);
            let xp = base.add((TILE_ROWS + cg * TILE) * LINE + lane);
            acc = f32_tile_chunk(acc, wp, xp, LINE);
            acc = f32_tile_chunk(acc, wp.add(32), xp.add(32), LINE);
        }
        s += 1;
        buf = if buf == STAGES - 1 { 0 } else { buf + 1 };
    }

    // The warp's first column, how many of its columns exist, its first row
    // and how many of its rows exist.
    let cw = c0 + cg * TILE;
    let live = if cw < n { (n - cw).min(TILE) } else { 0 };
    let r0 = row0 + rg * TILE;
    let live_rows = if r0 < rows { (rows - r0).min(TILE) } else { 0 };
    // Every lane runs each column's butterfly; lane 0 stores a live one's
    // live rows.
    macro_rules! column {
        ($c:literal) => {{
            let sums = gemv_lane_sums(acc[$c], TILE as u32);
            if lane == 0 && $c < live {
                // SAFETY: column cw + $c < n, so its slots (cw + $c)·rows +
                // r0 + i, i < live_rows, lie inside y (this fn's contract);
                // lane 0 of the warp owning rows r0 .. of that column is their
                // only writer.
                unsafe { store_live(y, (cw + $c) * rows + r0, &sums, live_rows) };
            }
        }};
    }
    column!(0);
    column!(1);
    column!(2);
    column!(3);
    column!(4);
    column!(5);
    column!(6);
    column!(7);
}

/// Threads in a dense BF16 MMA tile: eight warps, each covering 32 by 16.
pub(crate) const MMA_THREADS: u32 = 256;
const MMA_ROWS: usize = 64;
const MMA_COLS: usize = 64;
const MMA_LINE: usize = 32;
const _: () = assert!(MMA_THREADS == 256 && MMA_ROWS == 64 && MMA_COLS == 64 && MMA_LINE == 32);
const _: () = assert!(MMA_THREADS as usize * 2 == MMA_ROWS * (MMA_LINE / 4));
const _: () = assert!((MMA_ROWS + MMA_COLS) * MMA_LINE == STAGE_FLOATS && STAGES == 3);

/// BF16 tensor-core product over the stored weight's decoded rows.
/// Activations and decoded weights round once to BF16; accumulation is F32.
///
/// # Safety
///
/// All 256 threads call converged. Rows and columns are positive, K is a
/// positive multiple of 64, x holds n aligned K-value columns, y holds
/// rows*n outputs, and w holds rows*K values. The grid covers 64 by 64
/// output tiles. sh holds STAGES*STAGE_FLOATS aligned shared values.
#[inline(always)]
pub(crate) unsafe fn bf16_mma_body<W: TileRows>(
    w: W,
    x: &[f32],
    rows: u32,
    k: u32,
    n: u32,
    y: &mut DisjointSlice<f32>,
    sh: *mut f32,
) {
    let (rows, kk, n) = (rows as usize, k as usize, n as usize);
    let row_blocks = rows.div_ceil(MMA_ROWS);
    let block = thread::blockIdx_x() as usize;
    let group_size = GEMM32_GROUP * TILE_COLS / MMA_COLS;
    let Some((col_tile, row_tile)) = raster32(block, row_blocks, n.div_ceil(MMA_COLS), group_size)
    else {
        return;
    };
    let row0 = row_tile * MMA_ROWS;
    let col0 = col_tile * MMA_COLS;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let warp = tid / 32;
    let wr = (warp / 4) * 32;
    let wc = (warp % 4) * 16;
    let group = lane / 4;
    let pair = 2 * (lane % 4);
    let mut acc = [[[0.0f32; 4]; 2]; 2];
    let mut retry = false;
    let stages = kk / MMA_LINE;
    let staged_at = |at: usize| at ^ (((at / MMA_LINE) & 3) * 8);
    macro_rules! copy {
        ($s:expr, $buf:expr) => {{
            let (s, buf): (usize, usize) = ($s, $buf);
            for i in 0..2 {
                thread::__unroll_config::<0>();
                let piece = tid + i * MMA_THREADS as usize;
                let line = piece / (MMA_LINE / 4);
                let offset = 4 * (piece % (MMA_LINE / 4));
                let wa = (row0 + line).min(rows - 1) * kk + s * MMA_LINE + offset;
                let xa = (col0 + line).min(n - 1) * kk + s * MMA_LINE + offset;
                // SAFETY: each thread owns two aligned pieces of each plane;
                // source rows are clamped inside the checked weight.
                unsafe {
                    w.stage(
                        sh.add(buf * STAGE_FLOATS + staged_at(line * MMA_LINE + offset)),
                        wa,
                    )
                };
                // SAFETY: xa is an aligned live four-value piece of x. Its
                // stage plane is disjoint from the weight plane.
                unsafe {
                    cp_async_cg_16(
                        sh.add(
                            buf * STAGE_FLOATS + staged_at((MMA_ROWS + line) * MMA_LINE + offset),
                        )
                        .cast(),
                        x.as_ptr().add(xa).cast(),
                    )
                };
            }
        }};
    }
    copy!(0, 0);
    // SAFETY: commits this thread's copies of the first stage.
    unsafe { cp_async_commit_group() };
    copy!(1, 1);
    // SAFETY: K is a positive multiple of 64, so a second 32-value stage exists.
    unsafe { cp_async_commit_group() };
    for s in 0..stages {
        let buf = s % STAGES;
        // SAFETY: two groups precede this wait; the oldest is this stage.
        unsafe { cp_async_wait_group(1) };
        thread::sync_threads();
        if s + 2 < stages {
            copy!(s + 2, (s + 2) % STAGES);
        }
        // SAFETY: advances the pipeline even when the final group is empty.
        unsafe { cp_async_commit_group() };
        // SAFETY: this stage is published and no thread refills it before
        // the next iteration's barrier. Its extent is the allocation's.
        let stage =
            unsafe { core::slice::from_raw_parts(sh.add(buf * STAGE_FLOATS), STAGE_FLOATS) };
        let read = |at| {
            // SAFETY: each fragment coordinate has row < 128 and K < 32.
            // The permutation changes only K bits, so its slot is below 4096.
            unsafe { *stage.get_unchecked(staged_at(at)) }
        };
        let packed = |at| {
            let (a, b) = (read(at), read(at + 1));
            let fits = |v: f32| {
                let bits = v.to_bits() & 0x7fff_ffff;
                // |a|, |b| <= 2^47 and K < 2^32 bound every partial
                // absolute sum below 2^126, including the exact butterfly.
                bits == 0 || (bits >= 0x0080_0000 && bits <= 0x5700_0000)
            };
            (cvt_bf16x2_f32(a, b), fits(a) && fits(b))
        };
        for step in 0..2 {
            thread::__unroll_config::<0>();
            let off = step * 16 + pair;
            for mi in 0..2 {
                thread::__unroll_config::<0>();
                let at = (wr + mi * 16 + group) * MMA_LINE + off;
                let a0 = packed(at);
                let a1 = packed(at + 8 * MMA_LINE);
                let a2 = packed(at + 8);
                let a3 = packed(at + 8 * MMA_LINE + 8);
                let a = [a0.0, a1.0, a2.0, a3.0];
                for j in 0..2 {
                    thread::__unroll_config::<0>();
                    let at = (MMA_ROWS + wc + j * 8 + group) * MMA_LINE + off;
                    let b0 = packed(at);
                    let b1 = packed(at + 8);
                    let safe = a0.1 && a1.1 && a2.1 && a3.1 && b0.1 && b1.1;
                    // SAFETY: the whole warp is converged; A is row-major
                    // and B column-major in the m16n8k16 fragment layout.
                    let next = unsafe { mma_m16n8k16_f32_bf16(acc[mi][j], a, [b0.0, b1.0]) };
                    let finite = next[0].is_finite()
                        && next[1].is_finite()
                        && next[2].is_finite()
                        && next[3].is_finite();
                    retry |= !(safe && finite);
                    acc[mi][j] = next;
                }
            }
        }
    }
    for mi in 0..2 {
        thread::__unroll_config::<0>();
        for j in 0..2 {
            thread::__unroll_config::<0>();
            for i in 0..4 {
                thread::__unroll_config::<0>();
                let r = row0 + wr + mi * 16 + group + (i / 2) * 8;
                let c = col0 + wc + j * 8 + pair + i % 2;
                if r < rows && c < n {
                    // SAFETY: this lane owns the fragment output; its checked
                    // live row and column lie inside rows*n.
                    unsafe { *y.get_unchecked_mut(c * rows + r) = acc[mi][j][i] };
                }
            }
        }
    }
    // The next stage's buffer is no longer read or written by the pipeline.
    // SAFETY: the buffer is inside sh; the last stage reads a different buffer.
    let votes = unsafe { sh.add((stages % STAGES) * STAGE_FLOATS) };
    let retry = warp::any(retry);
    if lane == 0 {
        // SAFETY: each warp owns one slot of that unused buffer.
        unsafe { *votes.add(warp) = if retry { 1.0 } else { 0.0 } };
    }
    thread::sync_threads();
    let mut any = false;
    for i in 0..MMA_THREADS as usize / 32 {
        // SAFETY: all warp votes were published by the block barrier.
        any |= unsafe { *votes.add(i) != 0.0 };
    }
    if any {
        // All readers finish the votes before the exact owner reuses shared memory.
        thread::sync_threads();
        for dr in 0..MMA_ROWS / TILE_ROWS {
            for dc in 0..MMA_COLS / TILE_COLS {
                for half in 0..2 {
                    // SAFETY: all 256 threads replay disjoint halves of the
                    // checked 64-by-64 tile with the exact owner's staging bound.
                    unsafe {
                        f32_tile_at::<W, { MMA_THREADS as usize }>(
                            w,
                            x,
                            (rows, kk, n),
                            y,
                            sh,
                            (row0 + dr * TILE_ROWS, col0 + dc * TILE_COLS, half),
                        )
                    };
                    // The next pass must not refill a stage another warp still reads.
                    thread::sync_threads();
                }
            }
        }
    }
}

impl Gemm32Kernels {
    /// Enqueue `y[c · rows + r] = Σ_k w[r·k + k'] · x[c·k + k']` for the
    /// first `n` columns of `x`, `w` an F32 weight of `rows × k` (module
    /// doc): every output bit for bit `Q8F32Kernels::enqueue_f32_gemv`'s for
    /// its row and column. `k` must be a positive multiple of 64 (the
    /// staged line) and `w` and `x` 16-byte aligned (the staged copies are
    /// 16-byte pieces); another is a named `Shape` error. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_f32_tile(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_f32_tile";
        let (rows, k) = (w.rows(), w.cols());
        if w.buf().len() < rows * k {
            return Err(GpuError::shape(
                what,
                format!("w.len() {} >= rows*k = {rows}*{k}", w.buf().len()),
            ));
        }
        let w_at = w.buf().cu_deviceptr();
        if !w_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("w at {w_at:#x} must be 16-byte aligned"),
            ));
        }
        let t = tile_launch(what, rows, k, x, n, y)?;
        let prep =
            self.module
                .prepare_f32_tile_gemm(LaunchConfig1D::new(t.grid, F32T_THREADS_U32, 0))?;
        self.module
            .f32_tile_gemm(stream, &prep, w.buf(), x, t.rows, t.k, t.n, y)?;
        Ok(())
    }

    /// Enqueue [`Self::enqueue_f32_tile`]'s product for an E4M3 weight in
    /// its two planes (`rows × k` values, `k = 4 · w.qs.cols()`), its values
    /// decoded as they are staged (module doc) by `mxfp8_tile_gemm` or
    /// `fp8_b128_tile_gemm` as its scales say: every output bit for bit
    /// `Q8F32Kernels::enqueue_f32_gemv`'s over the host's dequantized rows.
    /// `k` must be a positive multiple of 128 with `e` `k / 128` words a row
    /// over the same rows, and `x` 16-byte aligned; another is a named
    /// `Shape` error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_e4m3_tile(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_e4m3_tile";
        let (rows, k) = w.dimensions(what)?;
        let t = tile_launch(what, rows, k, x, n, y)?;
        let cfg = LaunchConfig1D::new(t.grid, F32T_THREADS_U32, 0);
        let (qs, e, module) = (w.qs.buf(), w.e.buf(), &self.module);
        match w.scales {
            E4m3Scales::Mxfp8 => {
                let prep = module.prepare_mxfp8_tile_gemm(cfg)?;
                module.mxfp8_tile_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
            E4m3Scales::Fp8B128 => {
                let prep = module.prepare_fp8_b128_tile_gemm(cfg)?;
                module.fp8_b128_tile_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
        }
        Ok(())
    }
}

impl Gemm32Kernels {
    /// Dense E4M3 prompt product, with BF16 operands and F32 accumulation.
    /// The stored planes are decoded in shared memory, never rewritten.
    pub fn enqueue_e4m3_mma(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_e4m3_mma";
        let (rows, k) = w.dimensions(what)?;
        let t = mma_launch(what, rows, k, x, n, y)?;
        let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
        let (qs, e, module) = (w.qs.buf(), w.e.buf(), &self.module);
        match w.scales {
            E4m3Scales::Mxfp8 => {
                let prep = module.prepare_mxfp8_mma_gemm(cfg)?;
                module.mxfp8_mma_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
            E4m3Scales::Fp8B128 => {
                let prep = module.prepare_fp8_b128_mma_gemm(cfg)?;
                module.fp8_b128_mma_gemm(stream, &prep, qs, e, x, t.rows, t.k, t.n, y)?;
            }
        }
        Ok(())
    }
}

/// Checked spans shared with the exact tile, with the MMA tile's grid.
pub(crate) fn mma_launch(
    what: &'static str,
    rows: usize,
    k: usize,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<TileLaunch, GpuError> {
    let mut launch = tile_launch(what, rows, k, x, n, y)?;
    launch.grid = launch_u32(what, "grid", n.div_ceil(MMA_COLS) * rows.div_ceil(MMA_ROWS))?;
    Ok(launch)
}

/// A tile launch's grid and its `u32` arguments.
pub(crate) struct TileLaunch {
    pub(crate) grid: u32,
    pub(crate) rows: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
}

/// The checks every tile launcher shares, beyond the weight's own: `k` a
/// positive multiple of [`LINE`], `rows` and `n` positive, `x` holding `n`
/// columns of `k` values 16-byte aligned and `y` `rows · n` outputs; each
/// failure a named `Shape` error as `what`'s.
pub(crate) fn tile_launch(
    what: &'static str,
    rows: usize,
    k: usize,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &DeviceBuffer<f32>,
) -> Result<TileLaunch, GpuError> {
    if k == 0 || !k.is_multiple_of(LINE) {
        return Err(GpuError::shape(
            what,
            format!("k must be a positive multiple of {LINE} (the staged line), got {k}"),
        ));
    }
    if rows == 0 || n == 0 {
        return Err(GpuError::shape(
            what,
            format!("rows and n must be positive, got {rows} and {n}"),
        ));
    }
    let (Some(x_len), Some(y_len)) = (n.checked_mul(k), rows.checked_mul(n)) else {
        return Err(GpuError::shape(
            what,
            format!("n*k = {n}*{k} or rows*n = {rows}*{n} overflows"),
        ));
    };
    if x.len() < x_len || y.len() < y_len {
        return Err(GpuError::shape(
            what,
            format!(
                "x.len() {} >= n*k = {n}*{k}, y.len() {} >= rows*n = {rows}*{n}",
                x.len(),
                y.len()
            ),
        ));
    }
    let x_at = x.cu_deviceptr();
    if !x_at.is_multiple_of(16) {
        return Err(GpuError::shape(
            what,
            format!("x at {x_at:#x} must be 16-byte aligned"),
        ));
    }
    let grid = n.div_ceil(TILE_COLS) * rows.div_ceil(TILE_ROWS);
    Ok(TileLaunch {
        grid: launch_u32(what, "grid", grid)?,
        rows: launch_u32(what, "rows", rows)?,
        k: launch_u32(what, "k", k)?,
        n: launch_u32(what, "n", n)?,
    })
}
