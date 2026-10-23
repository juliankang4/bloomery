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
//! A BF16 site whose input holds its columns already rounded
//! ([`bf16_cols_body`], `GemmAct32::with_bf16`) copies those words instead,
//! with the same bits.
//! A cell whose row or column stages an operand outside the tensor cores'
//! range is recomputed by the exact F32 owner, in its exponent range and
//! reduction order; every other cell keeps its MMA value, so a cell's bits
//! depend on its own row and column alone. The exact tile entries stay
//! available.

use super::GemmAct32;
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
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::vector::{F32x4, U32x4};
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_bf16};
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
pub(crate) struct F32Rows<'a>(pub &'a [f32]);

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
    unsafe {
        f32_tile_at::<W, F32T_THREADS, false>(w, x, (rows, k as usize, n), y, sh, (row0, c0, 0))
    };
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
unsafe fn f32_tile_at<W: TileRows, const THREADS: usize, const MARKED: bool>(
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
                if MARKED {
                    for i in 0..TILE {
                        thread::__unroll_config::<0>();
                        if i < live_rows {
                            // SAFETY: this lane owns the checked live cell, published
                            // before replay; marked stores preserve healthy MMA cells.
                            let slot = unsafe { y.get_unchecked_mut((cw + $c) * rows + r0 + i) };
                            if slot.is_nan() {
                                *slot = sums[i];
                            }
                        }
                    }
                } else {
                    // SAFETY: this warp owns the checked column's live rows.
                    unsafe { store_live(y, (cw + $c) * rows + r0, &sums, live_rows) };
                }
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

/// Threads in a dense BF16 MMA tile: eight warps, two row halves by four
/// column quarters, each covering 64 rows by 32 columns.
pub(crate) const MMA_THREADS: u32 = 256;
/// Lanes that hold one fragment row's cells.
const MMA_ROW_LANES: usize = 4;
/// Rows and columns of a block's outputs.
pub(crate) const MMA_ROWS: usize = 128;
pub(crate) const MMA_COLS: usize = 128;
/// K values one stage holds of every row and column: two 16-value MMA steps.
const MMA_LINE: usize = 32;
/// 16-byte pieces of BF16 values in one staged line.
const MMA_PIECES: usize = MMA_LINE / 8;
/// 32-bit words in one staged line, two BF16 values a word.
const MMA_LINE_WORDS: usize = MMA_LINE / 2;
/// Staged words before the column lines.
const MMA_COLS_AT: usize = MMA_ROWS * MMA_LINE_WORDS;
/// Lines whose pieces one pass of the block stages.
const MMA_PASS_LINES: usize = MMA_THREADS as usize / MMA_PIECES;
const _: () = assert!(MMA_THREADS == 256 && MMA_ROWS == 128 && MMA_COLS == 128 && MMA_LINE == 32);
// Every thread stages the same piece of two row lines and of two column lines.
const _: () = assert!(2 * MMA_PASS_LINES == MMA_ROWS && MMA_ROWS == MMA_COLS);
// The BF16 rows and columns of a stage fill the exact owner's stage, which the replay reuses.
const _: () = assert!((MMA_ROWS + MMA_COLS) * MMA_LINE_WORDS == STAGE_FLOATS && STAGES == 3);

/// Word offset of piece `slot` of staged line `line`. The slot is permuted by
/// the line's pair, so the eight lines one `ldmatrix` phase reads at one slot
/// cover all 32 banks.
#[inline(always)]
fn mma_piece_at(line: usize, slot: usize) -> usize {
    line * MMA_LINE_WORDS + 4 * (slot ^ ((line >> 1) & (MMA_PIECES - 1)))
}

/// Whether an operand stays on the tensor cores: zero or normal with
/// `|v| <= 2^47`. With K below `2^32` that bounds every partial absolute sum
/// below `2^126`, so no product, sum or exact butterfly leaves the finite range.
#[inline(always)]
#[allow(
    clippy::manual_range_contains,
    reason = "the kernel's two compares on the bits; a RangeInclusive would move its PTX"
)]
fn mma_fits(v: f32) -> bool {
    let bits = v.to_bits() & 0x7fff_ffff;
    bits == 0 || (bits >= 0x0080_0000 && bits <= 0x5700_0000)
}

/// [`mma_fits`] for both BF16 values of a word, each its f32's high half.
#[inline(always)]
fn mma_fits_bf16x2(word: u32) -> bool {
    // Every 15-bit magnitude lane stays below 2^16 after each add, so no
    // carry crosses into the other value's lane.
    let big = (word & 0x7fff_7fff) + 0x28ff_28ff; // bit 15 set above 0x5700
    let exponent = (word & 0x7f80_7f80) + 0x7f80_7f80; // bit 15 set for a nonzero exponent
    let mantissa = (word & 0x007f_007f) + 0x7fff_7fff; // bit 15 set for a nonzero mantissa
    (big | (mantissa & !exponent)) & 0x8000_8000 == 0
}

/// A weight that the MMA tile stages as BF16 pieces of eight values.
pub(crate) trait MmaRows: TileRows {
    /// What a thread holds of one piece between [`Self::fetch`] and
    /// [`Self::land`].
    type Piece: Copy + Default;

    /// Whether [`Self::fetch`] is an asynchronous copy, so a thread holds no
    /// piece in registers across the products. The tile then runs the stages
    /// that fetch as their own loop, with no branch around the fetches; a
    /// weight that holds its pieces keeps one loop, where the compiler's
    /// hoisting of the next pieces' loads would spill.
    const COPIED: bool = true;

    /// Column tiles of one raster group ([`raster32`]). A wave of resident
    /// blocks reads the weight slabs of its rows and the F32 activations of
    /// its columns; for a BF16 weight, half an activation's bytes a value,
    /// eight tiles balance the two on GB10's 96 resident blocks.
    const GROUP: usize = 8;

    /// Starts moving values `at .. at + 8` (`at` a multiple of 8) to the 16
    /// bytes at `dst`: an asynchronous copy completed by this thread's
    /// `cp_async_wait_group` for its committed group, a store it makes
    /// itself, or loads [`Self::land`] stores.
    ///
    /// # Safety
    ///
    /// The eight values lie inside the weight's rows; `dst` is 16 bytes of
    /// shared memory, 16-byte aligned, that this thread alone writes and no
    /// warp reads before the barrier after that wait or that store.
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> Self::Piece;

    /// Stores a fetched piece of values `at ..` at `dst` as BF16 rounded to
    /// nearest even; false when a value fails [`mma_fits`]. The default, for
    /// a piece [`Self::fetch`] copies as BF16 words, stores nothing.
    ///
    /// # Safety
    ///
    /// `piece` and `at` are one [`Self::fetch`]'s, and `dst` is its 16 bytes.
    #[inline(always)]
    unsafe fn land(_piece: Self::Piece, _dst: *mut u32, _at: usize) -> bool {
        true
    }

    /// After this thread's wait completed its copy to `dst`, whether the
    /// piece's values pass [`mma_fits`]. The default reads the copied BF16
    /// words; a weight whose [`Self::land`] checks its values returns true.
    ///
    /// # Safety
    ///
    /// `dst` is a completed, 16-byte aligned piece of this thread's.
    #[inline(always)]
    unsafe fn landed_fits(dst: *const u32) -> bool {
        // SAFETY: this fn's contract.
        let [a, b, c, d] = unsafe { (*dst.cast::<U32x4>()).0 };
        mma_fits_bf16x2(a) & mma_fits_bf16x2(b) & mma_fits_bf16x2(c) & mma_fits_bf16x2(d)
    }
}

impl MmaRows for Bf16Rows<'_> {
    type Piece = ();

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: values at .. at + 8 are words at / 2 .. at / 2 + 4, inside
        // the weight by this fn's contract and 16-byte aligned (the launcher
        // checks the words' alignment, at is a multiple of 8); dst is this
        // thread's own 16 bytes until the wait that completes the copy.
        unsafe { cp_async_cg_16(dst, self.0.as_ptr().add(at >> 1)) };
    }
}

impl<F: ByteBlocks<Scale = u32>> MmaRows for ByteRows<'_, F> {
    /// The piece's two value words and its block's scale element.
    type Piece = (u32, u32, u32);
    const COPIED: bool = false;
    /// The F32 tiles' column span: a byte weight is a quarter of an
    /// activation's bytes a value.
    const GROUP: usize = GEMM32_GROUP * TILE_COLS / MMA_COLS;

    #[inline(always)]
    unsafe fn fetch(self, _: *mut u32, at: usize) -> (u32, u32, u32) {
        // SAFETY: values at .. at + 8 are the bytes of words at / 4 and
        // at / 4 + 1, inside `qs` by this fn's contract.
        let (lo, hi) = unsafe {
            (
                *self.qs.get_unchecked(at >> 2),
                *self.qs.get_unchecked((at >> 2) + 1),
            )
        };
        // SAFETY: their 32-value block at / 32 has its scale in the `e` plane.
        let e = unsafe { F::scale_at(self.e, at >> 5) };
        (lo, hi, e)
    }

    #[inline(always)]
    unsafe fn land((lo, hi, e): (u32, u32, u32), dst: *mut u32, at: usize) -> bool {
        let d = F::scale(e, at >> 5);
        let v = [
            F::value(lo) * d,
            F::value(lo >> 8) * d,
            F::value(lo >> 16) * d,
            F::value(lo >> 24) * d,
            F::value(hi) * d,
            F::value(hi >> 8) * d,
            F::value(hi >> 16) * d,
            F::value(hi >> 24) * d,
        ];
        // SAFETY: dst is this fetch's 16 bytes (this fn's contract).
        unsafe { land_values(v, dst) }
    }

    #[inline(always)]
    unsafe fn landed_fits(_: *const u32) -> bool {
        true
    }
}

/// Eight f32 values at `dst` as BF16 rounded to nearest even, two a word in
/// increasing order; false when one fails [`mma_fits`].
///
/// # Safety
///
/// `dst` is 16 bytes, 16-byte aligned, that this thread alone writes and no
/// thread reads before the barrier after this store.
#[inline(always)]
pub(crate) unsafe fn land_values(v: [f32; 8], dst: *mut u32) -> bool {
    let fits = mma_fits(v[0])
        & mma_fits(v[1])
        & mma_fits(v[2])
        & mma_fits(v[3])
        & mma_fits(v[4])
        & mma_fits(v[5])
        & mma_fits(v[6])
        & mma_fits(v[7]);
    let words = U32x4([
        cvt_bf16x2_f32(v[0], v[1]),
        cvt_bf16x2_f32(v[2], v[3]),
        cvt_bf16x2_f32(v[4], v[5]),
        cvt_bf16x2_f32(v[6], v[7]),
    ]);
    // SAFETY: dst is this thread's own aligned 16 bytes (this fn's contract).
    unsafe { *dst.cast::<U32x4>() = words };
    fits
}

/// Row and column of the first output of this block's tile among `rows` by
/// `n` outputs, in the raster used by `bf16_mma_tile`; `None` for a block
/// past the last tile.
#[inline(always)]
#[allow(
    clippy::question_mark,
    reason = "the let-else of bf16_mma_tile's raster, where a `?` moves the PTX"
)]
pub(crate) fn mma_tile_origin<W: MmaRows>(rows: usize, n: usize) -> Option<(usize, usize)> {
    let row_blocks = rows.div_ceil(MMA_ROWS);
    let block = thread::blockIdx_x() as usize;
    let Some((col_tile, row_tile)) = raster32(block, row_blocks, n.div_ceil(MMA_COLS), W::GROUP)
    else {
        return None;
    };
    let row0 = row_tile * MMA_ROWS;
    let col0 = col_tile * MMA_COLS;
    Some((row0, col0))
}

/// Activations that the MMA tile stages as BF16 pieces of eight values,
/// column `c`'s value `i` at `c·K + i`.
pub(crate) trait MmaCols: Copy {
    /// What a thread holds of one piece between [`Self::fetch`] and
    /// [`Self::land`].
    type Piece: Copy + Default;

    /// The f32 columns the exact owner replays.
    fn exact(&self) -> &[f32];

    /// Whether column `c` holds a value [`mma_fits`] refuses, known before
    /// any piece is staged; false when [`Self::land`] checks each piece.
    ///
    /// # Safety
    ///
    /// `c` is a live column.
    unsafe fn refused(self, c: usize) -> bool;

    /// Starts moving values `at .. at + 8` to the 16 bytes at `dst`, as
    /// [`MmaRows::fetch`] does.
    ///
    /// # Safety
    ///
    /// [`MmaRows::fetch`]'s, over the columns.
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> Self::Piece;

    /// Stores a fetched piece at `dst` as BF16 rounded to nearest even; false
    /// when a value fails [`mma_fits`].
    ///
    /// # Safety
    ///
    /// `piece` is one [`Self::fetch`]'s, and `dst` is its 16 bytes.
    unsafe fn land(piece: Self::Piece, dst: *mut u32) -> bool;
}

/// F32 activations, loaded a piece at a time and rounded as they are staged.
#[derive(Clone, Copy)]
pub(crate) struct F32Cols<'a>(pub &'a [f32]);

impl MmaCols for F32Cols<'_> {
    type Piece = [f32; 8];

    #[inline(always)]
    fn exact(&self) -> &[f32] {
        self.0
    }

    #[inline(always)]
    unsafe fn refused(self, _: usize) -> bool {
        false
    }

    #[inline(always)]
    unsafe fn fetch(self, _: *mut u32, at: usize) -> [f32; 8] {
        let p = self.0.as_ptr().wrapping_add(at).cast::<F32x4>();
        // SAFETY: p starts eight values of a live column inside x, 16-byte
        // aligned (x aligned, at a multiple of 8).
        let [F32x4([a0, a1, a2, a3]), F32x4([b0, b1, b2, b3])] = unsafe { [*p, *p.add(1)] };
        [a0, a1, a2, a3, b0, b1, b2, b3]
    }

    #[inline(always)]
    unsafe fn land(v: [f32; 8], dst: *mut u32) -> bool {
        // SAFETY: dst is this fetch's 16 bytes (this fn's contract).
        unsafe { land_values(v, dst) }
    }
}

/// The same activations already rounded ([`bf16_cols_body`]): BF16 words,
/// two values a word in increasing order, copied asynchronously as a BF16
/// weight's are, and a word a column that is nonzero when one of the
/// column's f32 values fails [`mma_fits`]. The f32 columns stay for the
/// replay.
#[derive(Clone, Copy)]
pub(crate) struct Bf16Cols<'a> {
    pub words: &'a [u32],
    pub refused: &'a [u32],
    pub exact: &'a [f32],
}

impl MmaCols for Bf16Cols<'_> {
    type Piece = ();

    #[inline(always)]
    fn exact(&self) -> &[f32] {
        self.exact
    }

    #[inline(always)]
    unsafe fn refused(self, c: usize) -> bool {
        // SAFETY: a live column has its word (this fn's contract).
        unsafe { *self.refused.get_unchecked(c) != 0 }
    }

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: the columns' words are laid out as a BF16 weight's rows, and
        // this fn's contract is Bf16Rows::fetch's over them.
        unsafe { Bf16Rows(self.words).fetch(dst, at) };
    }

    #[inline(always)]
    unsafe fn land(_: (), _: *mut u32) -> bool {
        true
    }
}

/// Threads of a [`bf16_cols_body`] block: a warp a column.
pub(crate) const BF16_COLS_THREADS: u32 = 256;

/// The first `n` columns of `k` f32 values (`k` a multiple of 8) as
/// [`Bf16Cols`]' words, each value rounded exactly as [`land_values`] rounds
/// it, and each column's refusal word. Block `b`'s warp `w` writes column
/// `8b + w`.
///
/// # Safety
///
/// x holds n aligned K-value columns, words n·K/2 and refused n.
#[inline(always)]
pub(crate) unsafe fn bf16_cols_body(
    x: &[f32],
    k: u32,
    n: u32,
    words: &mut DisjointSlice<u32>,
    refused: &mut DisjointSlice<u32>,
) {
    let (k, n) = (k as usize, n as usize);
    let c = thread::blockIdx_x() as usize * (BF16_COLS_THREADS as usize / 32)
        + thread::threadIdx_x() as usize / 32;
    let lane = warp::lane_id() as usize;
    let mut fits = true;
    if c < n {
        let mut piece = lane;
        while piece < k / 8 {
            let at = c * k + 8 * piece;
            // SAFETY: values at .. at + 8 lie in column c of x (this fn's contract).
            let v = unsafe { F32Cols(x).fetch(core::ptr::null_mut(), at) };
            // SAFETY: words at/2 .. at/2 + 4 are this lane's own, 16-byte aligned.
            fits &= unsafe { land_values(v, words.as_mut_ptr().add(at >> 1)) };
            piece += 32;
        }
    }
    let refuse = warp::any(!fits);
    if c < n && lane == 0 {
        // SAFETY: lane 0 of column c's warp is its word's only writer.
        unsafe { *refused.get_unchecked_mut(c) = u32::from(refuse) };
    }
}

/// Cells of a thread's MMA fragments: four row by four column tiles of four.
pub(crate) const MMA_CELLS: usize = 64;
/// Words of the stage left after the pipeline before the line flags
/// [`MmaTile::needs_replay`] marks cells with: one vote a warp.
const MMA_FLAGS_AT: usize = MMA_THREADS as usize / 32;
const _: () = assert!(MMA_FLAGS_AT.is_multiple_of(4) && MMA_CELLS == 4 * 4 * 4);
const _: () = assert!(MMA_FLAGS_AT + (MMA_ROWS + MMA_COLS) * MMA_PIECES <= STAGE_FLOATS);
/// Rows and columns of one warp's outputs.
pub(crate) const MMA_WARP_ROWS: usize = 64;
pub(crate) const MMA_WARP_COLS: usize = 32;
/// Shared floats [`MmaTile::warp_scratch`] gives each warp: the two stages
/// that hold no votes, divided between the eight warps.
pub(crate) const MMA_WARP_SCRATCH: usize = 2 * STAGE_FLOATS / (MMA_THREADS as usize / 32);
const _: () = assert!(MMA_ROWS == 2 * MMA_WARP_ROWS && MMA_COLS == 4 * MMA_WARP_COLS);

/// Row and column of warp `warp`'s outputs inside the tile.
#[inline(always)]
fn mma_warp_at(warp: usize) -> (usize, usize) {
    ((warp / 4) * MMA_WARP_ROWS, (warp % 4) * MMA_WARP_COLS)
}

/// A thread's completed MMA fragment and logical output coordinates.
pub(crate) struct MmaTile {
    values: [[[f32; 4]; 4]; 4],
    row0: usize,
    col0: usize,
    warp: usize,
    lane: usize,
    /// Bit i (2 + i): a value this thread staged of its weight (activation)
    /// line i fails [`mma_fits`].
    lines: u32,
    /// The cells [`Self::needs_replay`] marks for the exact owner.
    bad: u64,
}

impl MmaTile {
    #[inline(always)]
    pub(crate) fn coordinates(&self, cell: usize) -> (usize, usize) {
        let (mi, j, i) = (cell / 16, (cell / 4) % 4, cell % 4);
        let (r0, c0) = self.warp_origin();
        (
            r0 + mi * 16 + self.lane / MMA_ROW_LANES + (i / 2) * 8,
            c0 + j * 8 + 2 * (self.lane % MMA_ROW_LANES) + i % 2,
        )
    }

    /// The accumulated value of `cell` (< [`MMA_CELLS`]). Cells
    /// `n·MMA_CELLS/4 ..` hold the warp's output rows `16n .. 16n + 16`.
    #[inline(always)]
    pub(crate) fn value(&self, cell: usize) -> f32 {
        self.values[cell / 16][(cell / 4) % 4][cell % 4]
    }

    /// Row and column of this warp's first output.
    #[inline(always)]
    pub(crate) fn warp_origin(&self) -> (usize, usize) {
        let (wr, wc) = mma_warp_at(self.warp);
        (self.row0 + wr, self.col0 + wc)
    }

    /// This warp's [`MMA_WARP_SCRATCH`] floats of `sh`, the tile's allocation
    /// for K = `k`. No other warp uses them, and once [`Self::needs_replay`]
    /// returns no read of the pipeline or the votes is left in them; a replay
    /// reuses them.
    #[inline(always)]
    pub(crate) fn warp_scratch(&self, sh: *mut f32, k: usize) -> *mut f32 {
        let warps_a_stage = STAGE_FLOATS / MMA_WARP_SCRATCH;
        let stage = (k / MMA_LINE + 1 + self.warp / warps_a_stage) % STAGES;
        sh.wrapping_add(stage * STAGE_FLOATS + (self.warp % warps_a_stage) * MMA_WARP_SCRATCH)
    }
}

impl MmaTile {
    /// Whether any thread of the block staged an operand that fails
    /// [`mma_fits`]; if so, also marks this thread's cells whose row or column
    /// staged one, the cells [`Self::store`] leaves to [`Self::replay`].
    ///
    /// # Safety
    /// All 256 threads call with the tile's original shared allocation and K.
    /// After a true return, a barrier precedes any write to the stages.
    #[inline(always)]
    pub(crate) unsafe fn needs_replay(&mut self, sh: *mut f32, k: usize) -> bool {
        // The pipeline left this stage unread since its last pass's barrier;
        // the votes and then the line flags reuse it.
        let stage = sh.wrapping_add((k / MMA_LINE % STAGES) * STAGE_FLOATS);
        let retry = warp::any(self.lines != 0);
        if self.lane == 0 {
            // SAFETY: each warp owns one vote slot of the unused stage.
            unsafe { *stage.add(self.warp) = if retry { 1.0 } else { 0.0 } };
        }
        thread::sync_threads();
        let mut any = false;
        for i in 0..MMA_THREADS as usize / 32 {
            // SAFETY: the block barrier published all eight warp votes.
            any |= unsafe { *stage.add(i) != 0.0 };
        }
        if !any {
            return false;
        }
        // Each thread flags its four lines' pieces after the votes; a cell is
        // marked when a piece of its row or column is flagged.
        let flags = stage.cast::<u32>().wrapping_add(MMA_FLAGS_AT);
        let tid = self.warp * 32 + self.lane;
        let (line, slot) = (tid / MMA_PIECES, tid % MMA_PIECES);
        for i in 0..4 {
            thread::__unroll_config::<0>();
            let at = (MMA_ROWS * (i / 2) + line + MMA_PASS_LINES * (i % 2)) * MMA_PIECES + slot;
            // SAFETY: this thread alone writes its piece's flag, inside the stage
            // and past the votes.
            unsafe { *flags.add(at) = (self.lines >> i) & 1 };
        }
        thread::sync_threads();
        let line_bad = |line: usize| {
            // SAFETY: the barrier published the line's four aligned flags.
            let [a, b, c, d] = unsafe { (*flags.add(line * MMA_PIECES).cast::<U32x4>()).0 };
            a | b | c | d != 0
        };
        for cell in 0..MMA_CELLS {
            thread::__unroll_config::<0>();
            let (r, c) = self.coordinates(cell);
            if line_bad(r - self.row0) || line_bad(MMA_ROWS + c - self.col0) {
                self.bad |= 1 << cell;
            }
        }
        true
    }

    /// Stores this thread's live cells; with MARKED, a marked or non-finite
    /// cell as NaN, which [`Self::replay`] recomputes.
    ///
    /// # Safety
    /// y holds rows*n cells. This tile is disjoint from every other block's.
    #[inline(always)]
    pub(crate) unsafe fn store<const MARKED: bool>(
        &self,
        y: &mut DisjointSlice<f32>,
        rows: usize,
        n: usize,
    ) {
        for cell in 0..MMA_CELLS {
            thread::__unroll_config::<0>();
            let (r, c) = self.coordinates(cell);
            if r < rows && c < n {
                let value = self.value(cell);
                let value = if MARKED && (self.bad & (1 << cell) != 0 || !value.is_finite()) {
                    f32::NAN
                } else {
                    value
                };
                // SAFETY: this thread owns the checked live output cell.
                unsafe { *y.get_unchecked_mut(c * rows + r) = value };
            }
        }
    }

    /// Recomputes the cells [`Self::needs_replay`] marked, which
    /// `store::<true>` left NaN, with the exact owner; every other cell keeps
    /// its stored value.
    ///
    /// # Safety
    /// All 256 threads call converged with this tile's checked spans and shared
    /// allocation. Prior output stores and shared readers have completed.
    #[inline(always)]
    pub(crate) unsafe fn replay<W: TileRows>(
        &self,
        w: W,
        x: &[f32],
        dims: (usize, usize, usize),
        y: &mut DisjointSlice<f32>,
        sh: *mut f32,
    ) {
        for dr in 0..MMA_ROWS / TILE_ROWS {
            for dc in 0..MMA_COLS / TILE_COLS {
                let (r, c) = (self.row0 + dr * TILE_ROWS, self.col0 + dc * TILE_COLS);
                if r >= dims.0 || c >= dims.2 {
                    continue; // block-uniform: the sub-tile holds no output
                }
                for half in 0..2 {
                    // SAFETY: the passes cover disjoint checked halves and reuse
                    // the exact owner's shared allocation with 256 threads.
                    unsafe {
                        f32_tile_at::<W, { MMA_THREADS as usize }, true>(
                            w,
                            x,
                            dims,
                            y,
                            sh,
                            (r, c, half),
                        )
                    };
                    thread::sync_threads();
                }
            }
        }
    }
}

/// BF16 tensor-core fragments of the stored weight's decoded rows, and
/// whether the tile replays. Activations and decoded weights round once to
/// BF16; accumulation is F32, each output's 16-value MMA steps in increasing K.
///
/// Both operands are staged as BF16 lines, three stages deep, two stages
/// ahead: the weight's pieces through [`MmaRows`] (a BF16 weight by
/// asynchronous copy) and the activations' through [`MmaCols`] (F32 values
/// loaded and converted after the current stage's products, or rounded BF16
/// words copied). Each warp reads its fragments with `ldmatrix`. The tile
/// keeps, per thread, which of its staged lines held an operand [`mma_fits`]
/// refuses; [`MmaTile::needs_replay`] marks the cells of those rows and
/// columns for the exact owner.
///
/// # Safety
///
/// All 256 threads call converged. Rows and n are positive, K is a positive
/// multiple of 64, x holds n aligned K-value columns and w holds rows*K values
/// ([`MmaRows::fetch`]'s bound). The grid covers 128 by 128 output tiles. sh
/// holds STAGES*STAGE_FLOATS aligned shared values.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
#[allow(
    clippy::question_mark,
    reason = "the kernel's early return on the raster; a `?` would move its PTX"
)]
pub(crate) unsafe fn bf16_mma_tile<W: MmaRows, X: MmaCols>(
    w: W,
    x: X,
    rows: u32,
    k: u32,
    n: u32,
    sh: *mut f32,
) -> Option<MmaTile> {
    let (rows, kk, n) = (rows as usize, k as usize, n as usize);
    let Some((row0, col0)) = mma_tile_origin::<W>(rows, n) else {
        return None;
    };
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let warp = tid / 32;
    let (wr, wc) = mma_warp_at(warp);
    let stages = kk / MMA_LINE;
    let words = sh.cast::<u32>();
    // SAFETY: sh is this block's shared allocation (this fn's contract).
    let shared = unsafe { cvta_generic_to_shared_u32(words.cast_const().cast::<u8>()) };

    // This thread's piece `slot` of lines `line` and `line + MMA_PASS_LINES`
    // of both operands; a row past the last is staged as the last row and a
    // column past the last as the last column.
    let slot = tid % MMA_PIECES;
    let line = tid / MMA_PIECES;
    let w_src = [
        (row0 + line).min(rows - 1) * kk + 8 * slot,
        (row0 + line + MMA_PASS_LINES).min(rows - 1) * kk + 8 * slot,
    ];
    let cols = [
        (col0 + line).min(n - 1),
        (col0 + line + MMA_PASS_LINES).min(n - 1),
    ];
    let x_src = [cols[0] * kk + 8 * slot, cols[1] * kk + 8 * slot];
    let w_dst = [
        mma_piece_at(line, slot),
        mma_piece_at(line + MMA_PASS_LINES, slot),
    ];
    let x_dst = [MMA_COLS_AT + w_dst[0], MMA_COLS_AT + w_dst[1]];

    // Bit i (2 + i) is set once this thread stages a value of its weight
    // (activation) line i that fails mma_fits.
    let mut bad = 0u32;
    for i in 0..2 {
        thread::__unroll_config::<0>();
        // SAFETY: cols[i] < n is a live column.
        bad |= u32::from(unsafe { x.refused(cols[i]) }) << (2 + i);
    }
    for s in 0..2 {
        thread::__unroll_config::<0>();
        let stage = words.wrapping_add(s * STAGE_FLOATS);
        for i in 0..2 {
            thread::__unroll_config::<0>();
            let at = w_src[i] + s * MMA_LINE;
            let dst = stage.wrapping_add(w_dst[i]);
            // SAFETY: K is a positive multiple of 64, so stage s < 2 lies
            // inside it; dst is this thread's own aligned 16 bytes of the stage.
            let piece = unsafe { w.fetch(dst, at) };
            // SAFETY: that fetch's piece and bytes.
            bad |= u32::from(!unsafe { W::land(piece, dst, at) }) << i;
            let dst = stage.wrapping_add(x_dst[i]);
            // SAFETY: as the weight's, in this thread's own activation piece.
            let piece = unsafe { x.fetch(dst, x_src[i] + s * MMA_LINE) };
            // SAFETY: that fetch's piece and bytes.
            bad |= u32::from(!unsafe { X::land(piece, dst) }) << (2 + i);
        }
        // SAFETY: commits this thread's copies of stage s as one group.
        unsafe { cp_async_commit_group() };
    }

    // [m tile][n tile] accumulators of the warp's 64 by 32 outputs.
    let mut acc = [[[0.0f32; 4]; 4]; 4];
    // ldmatrix rows: A row `lane % 16` at K half `lane / 16`; B column
    // `lane % 8 + 8·(lane / 16)` at K half `(lane / 8) % 2`. Bits 1..2 of the
    // line pick the permutation, and the warp's line offsets keep them.
    let a_lane = mma_piece_at(lane % 16, lane / 16);
    let b_lane = mma_piece_at(lane % 8 + 8 * (lane / 16), (lane / 8) % 2);
    // Stage s's products; with `ahead`, stage s + 2 goes to the buffer every
    // warp finished reading before this pass's barrier: piece `step` of each
    // operand is fetched before MMA step `step` and landed after it, so a
    // thread holds one piece of each in flight. A copied weight's stages that
    // fetch run as their own loop ([`MmaRows::COPIED`]).
    macro_rules! stage_pass {
        ($s:expr, $ahead:expr) => {{
            let (s, ahead) = ($s, $ahead);
            let buf = s % STAGES;
            // SAFETY: two groups precede this wait; the oldest is stage s.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
            let next = words.wrapping_add(((s + 2) % STAGES) * STAGE_FLOATS);
            let stage = shared + (4 * buf * STAGE_FLOATS) as u32;
            for step in 0..2 {
                thread::__unroll_config::<0>();
                let at = w_src[step] + (s + 2) * MMA_LINE;
                let mut piece = W::Piece::default();
                let mut xs = X::Piece::default();
                if ahead {
                    // SAFETY: stage s + 2 lies inside K; the piece is this thread's.
                    piece = unsafe { w.fetch(next.wrapping_add(w_dst[step]), at) };
                    let x_at = x_src[step] + (s + 2) * MMA_LINE;
                    // SAFETY: as the weight's, in the activation piece.
                    xs = unsafe { x.fetch(next.wrapping_add(x_dst[step]), x_at) };
                }
                if step == 1 {
                    // SAFETY: commits this pass's copies as one group, empty near the end.
                    unsafe { cp_async_commit_group() };
                }
                let half = 8 * step;
                // Both column fragments first; then one row fragment at a time,
                // so the warp holds twelve fragment words, not twenty.
                let mut b = [[0u32; 4]; 2];
                for nj in 0..2 {
                    thread::__unroll_config::<0>();
                    let at = MMA_COLS_AT + (wc + 16 * nj) * MMA_LINE_WORDS + (b_lane ^ half);
                    // SAFETY: the warp is converged; each lane's 16-byte row lies
                    // in this published stage's column lines.
                    b[nj] = unsafe { ldmatrix_x4_shared_u32(stage + (4 * at) as u32) };
                }
                for mi in 0..4 {
                    thread::__unroll_config::<0>();
                    let at = (wr + 16 * mi) * MMA_LINE_WORDS + (a_lane ^ half);
                    // SAFETY: as above, in the stage's row lines.
                    let a = unsafe { ldmatrix_x4_shared_u32(stage + (4 * at) as u32) };
                    for j in 0..4 {
                        thread::__unroll_config::<0>();
                        let b = [b[j / 2][2 * (j % 2)], b[j / 2][2 * (j % 2) + 1]];
                        // SAFETY: the whole warp is converged; A is row-major and
                        // B column-major in the m16n8k16 fragment layout.
                        acc[mi][j] = unsafe { mma_m16n8k16_f32_bf16(acc[mi][j], a, b) };
                    }
                }
                if step == 1 {
                    for i in 0..2 {
                        thread::__unroll_config::<0>();
                        let fits =
                            // SAFETY: the wait completed this thread's copies of stage
                            // s, which no warp overwrites before the next pass's barrier.
                            unsafe { W::landed_fits(words.add(buf * STAGE_FLOATS + w_dst[i])) };
                        bad |= u32::from(!fits) << i;
                    }
                }
                if ahead {
                    // SAFETY: this thread's own aligned activation piece of stage
                    // s + 2, which no warp reads before the next pass's barrier.
                    let fits = unsafe { X::land(xs, next.wrapping_add(x_dst[step])) };
                    bad |= u32::from(!fits) << (2 + step);
                    // SAFETY: this step's fetch of stage s + 2, into its piece, as above.
                    let fits = unsafe { W::land(piece, next.wrapping_add(w_dst[step]), at) };
                    bad |= u32::from(!fits) << step;
                }
            }
        }};
    }
    if W::COPIED {
        for s in 0..stages - 2 {
            stage_pass!(s, true);
        }
        for s in stages - 2..stages {
            stage_pass!(s, false);
        }
    } else {
        for s in 0..stages {
            stage_pass!(s, s + 2 < stages);
        }
    }
    Some(MmaTile {
        values: acc,
        row0,
        col0,
        warp,
        lane,
        lines: bad,
        bad: 0,
    })
}

/// Dense prompt product whose cells of an exceptional row or column replay on
/// the exact owner.
///
/// # Safety
/// The checked launch supplies 256 converged threads, aligned stages and x,
/// rows*K weight values, n*K activations and rows*n disjoint output cells.
#[inline(always)]
pub(crate) unsafe fn bf16_mma_body<W: MmaRows, X: MmaCols>(
    w: W,
    x: X,
    rows: u32,
    k: u32,
    n: u32,
    y: &mut DisjointSlice<f32>,
    sh: *mut f32,
) {
    // SAFETY: the wrapper's spans and launch satisfy the tile's contract.
    let Some(mut tile) = (unsafe { bf16_mma_tile(w, x, rows, k, n, sh) }) else {
        return;
    };
    let (rows, k, n) = (rows as usize, k as usize, n as usize);
    // SAFETY: all threads share the original staging allocation and K.
    if unsafe { tile.needs_replay(sh, k) } {
        // SAFETY: the checked output belongs to this tile.
        unsafe { tile.store::<true>(y, rows, n) };
        // Publishes the marked stores and retires the flags' readers.
        thread::sync_threads();
        // SAFETY: spans and allocation are unchanged.
        unsafe { tile.replay(w, x.exact(), (rows, k, n), y, sh) };
    } else {
        // SAFETY: the checked output belongs to this tile.
        unsafe { tile.store::<false>(y, rows, n) };
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

    /// `Q8F32Kernels::enqueue_bf16_mma`'s product, bit for bit, over the BF16
    /// form `act` holds of the first `n` columns of `x`
    /// ([`Self::enqueue_bf16_cols`]): the tile copies the rounded words, and
    /// only a replay reads `x`. A form of other rows or of fewer columns is a
    /// named `State` error. Asynchronous, allocation-free, capturable.
    pub fn enqueue_bf16_mma_cols(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        x: &DeviceBuffer<f32>,
        act: &GemmAct32,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_bf16_mma_cols";
        let t = bf16_mma_launch(what, w, x, n, y)?;
        if act.k() != t.k as usize {
            return Err(GpuError::shape(
                what,
                format!("act holds {} values a column, the weight {}", act.k(), t.k),
            ));
        }
        let form = act.bf16_of(x, n).ok_or(GpuError::state(
            what,
            "act's BF16 form of x's first n columns (Gemm32Kernels::enqueue_bf16_cols)",
        ))?;
        let prep =
            self.module
                .prepare_bf16_mma_cols_gemm(LaunchConfig1D::new(t.grid, MMA_THREADS, 0))?;
        let (words, refused) = (&form.words, &form.refused);
        self.module.bf16_mma_cols_gemm(
            stream,
            &prep,
            w.buf(),
            x,
            words,
            refused,
            t.rows,
            t.k,
            t.n,
            y,
        )?;
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

/// [`mma_launch`] for a BF16 weight's `rows × 2·cols` values. The tile copies
/// its words in 16-byte pieces, so they must be 16-byte aligned; another is a
/// named `Shape` error.
pub(crate) fn bf16_mma_launch(
    what: &'static str,
    w: &DeviceTensor<u32>,
    x: &DeviceBuffer<f32>,
    n: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<TileLaunch, GpuError> {
    let k = w
        .cols()
        .checked_mul(2)
        .ok_or_else(|| GpuError::shape(what, "BF16 words to values overflows"))?;
    let launch = mma_launch(what, w.rows(), k, x, n, y)?;
    crate::tensor::check_alignment::<U32x4, _>(what, "w", w.buf())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_mma_layout() {
        let mut seen = std::collections::HashSet::new();
        for warp in 0..8 {
            for lane in 0..32 {
                let tile = MmaTile {
                    values: [[[0.0; 4]; 4]; 4],
                    row0: 128,
                    col0: 256,
                    warp,
                    lane,
                    lines: 0,
                    bad: 0,
                };
                let (r0, c0) = tile.warp_origin();
                for cell in 0..MMA_CELLS {
                    let (r, c) = tile.coordinates(cell);
                    assert!((128..256).contains(&r) && (256..384).contains(&c));
                    assert!(seen.insert((r, c)));
                    assert!((r - r0) / 16 == cell / 16 && c - c0 < MMA_WARP_COLS);
                }
            }
        }
        assert_eq!(seen.len(), MMA_ROWS * MMA_COLS);
    }
}
