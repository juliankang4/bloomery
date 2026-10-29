//! The prefill flash of `flash_gqa_prefill` over heads of [`HEAD_512`]
//! values (EmbeddingGemma 2's global layers): `T` query rows, each attending
//! over its own live key count `n_keys[t]`, `PACK` query heads of a block
//! sharing one key/value head over the cache planes `[n_kv][ctx][512]` f16
//! that `rope_neox::head_norm_neox_append_512` writes. An encoder layer gives
//! every row the whole input (`n_keys[t] = T`) on `gqa_prefill_flash_512_p4`.
//! `gqa_prefill_flash_512_p4_lower` attends `[k_first[t], n_keys[t])`. The walk
//! keeps the per-row counts of the 256 entries.
//!
//! The 256 entries' block does not fit at this width: their query fragments
//! live in registers and their 64-key tiles take 96 KB of shared memory at
//! 256 values, so 512 values would double both past the block's limits. This
//! entry keeps the query in shared memory, walks [`KEY_TILE_512`]-key tiles,
//! and splits each row's 512 output values over two blocks of the grid (the
//! value half `vh`, outputs `256·vh ..`), each block recomputing the scores.
//!
//! Geometry: block `(query tile, key head's pack, value half)` owns
//! [`Q_ROWS`] staged query rows, `Q_ROWS / PACK` consecutive positions ×
//! `PACK` heads, position-major. Warp `w` owns m16 row slice `w % SLICES`
//! and dim half `w / SLICES`: for its scores the half's 256 key dims, for
//! its outputs 128 of the block's 256 value dims. A slice is `16 / PACK`
//! positions × `PACK` heads (`flash_gqa_prefill::row_pos`, `row_head`).
//!
//! Per tile and warp:
//! 1. `S_w = Q·Kᵀ` over the warp's 256 dims on `mma.m16n8k16`, the query rows
//!    rounded to f16 once per block (`f32x2_to_f16x2_bits`) and read from
//!    shared memory a k16 step at a time, the sixteen steps ascending; the
//!    warp publishes `S_w` and adds its partner's (`w ^ SLICES`): `S = S_lo +
//!    S_hi`, one rounded add, commutative, so both warps of a pair hold the
//!    same bits; then `s = S · scale` rounded. A key at or past the row's
//!    count is `−inf`, and on the lower-bound entry so is a key below its first key.
//! 2. The online softmax of every prefill body (`flash_gqa_prefill`'s
//!    `prefill_softmax!`): the tile max over the row's four lanes,
//!    a rescale `f = dev_exp(m − m')` on a bump (0 for the first), the
//!    weights `exp_weight(s − m')` rounded to f16, the lane's running sum of
//!    the rounded weights, the accumulators times `f` unless every lane of
//!    the warp has `f = 1`.
//! 3. `O += P̂·V` over the warp's 128 value dims, the tile's two k16 steps
//!    ascending, f32 accumulation.
//!
//! After the last tile: the row sum over its four lanes `(l + l^1) + l^2` and
//! `o · (1/l)` (`prefill_pack_store!`, as the 256 PACK body; the row counts
//! are its `prefill_pack_counts!`). Tile `i` holds keys `[32·i, 32·i + 32)`; the key tile, the
//! value tile and the exchange are staged as in the 256 body (`cp.async`, the
//! value tile landing under the scores, the next key tile under the value
//! product), each row a cache row's bytes unpadded, its 16-byte chunks
//! permuted within each 128-byte line by `chunk ^ (row & 7)`.
//!
//! Reduction structure, the fixed contract (reruns are bit-identical): a
//! row's arithmetic depends only on its own query row, the key and value rows
//! below its count and the fixed tile partition, and not on `T`, the rows
//! that share its block, the grid or the value half. Keys at or past the block's
//! largest count are never loaded (zeros are staged); on the lower-bound entry,
//! keys below the block's smallest first key are not loaded either. Keys between
//! a row's count and the block's largest are loaded and masked, so a non-finite
//! value there reaches the masking row through its zero weight. A count of zero or
//! past `ctx` on a row below `T` raises [`FaultSite::KeyCount`], and that
//! row's output is NaN.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::f32x2_to_f16x2_bits;
use crate::flash_gqa::PACK_4;
use crate::flash_gqa_prefill::{
    GqaPrefillArgs, layout_holds, pack_rows_hold, prefill_pack_counts, prefill_pack_store,
    prefill_softmax, staged,
};
use crate::rope_neox::HEAD_512;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::vector::U32x4;
use cuda_device::{
    DisjointSlice, DynamicSharedArray, kernel, launch_bounds, launch_contract, shared, thread,
    warp, wmma,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Keys one tile stages. Tile `i` holds keys `[32·i, 32·i + 32)`.
pub const KEY_TILE_512: usize = 32;
const KEY_TILE_U32: u32 = KEY_TILE_512 as u32;
/// Row slices of a block, its warps (two dim halves a slice) and threads.
const SLICES: usize = 2;
const WARPS: usize = 2 * SLICES;
const THREADS: usize = WARPS * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS == 128);
/// Rows of one m16 row slice, and the staged query rows of a block.
const SLICE_ROWS: usize = 16;
const Q_ROWS: usize = SLICES * SLICE_ROWS;
/// u32 words of a staged query or key row, and of a staged value half-row.
const ROW_WORDS: usize = HEAD_512 / 2;
const V_HALF: usize = HEAD_512 / 2;
const V_ROW_WORDS: usize = V_HALF / 2;
/// u32 words of the query rows, a key tile and a value tile.
const Q_TILE_WORDS: usize = Q_ROWS * ROW_WORDS;
const K_TILE_WORDS: usize = KEY_TILE_512 * ROW_WORDS;
const V_TILE_WORDS: usize = KEY_TILE_512 * V_ROW_WORDS;
/// u32 words of a 128-byte line (the permutation's reach) and the `ldmatrix`
/// chunk pairs one line holds.
const LINE: usize = 32;
const LINE_PAIRS: usize = LINE / 8;
/// A warp's k16 steps over its 256 score dims, the n8 tiles of a key tile,
/// the value product's k16 steps over a tile, and the n16 pairs of the
/// warp's 128 value dims.
const SCORE_HALF: usize = HEAD_512 / 2;
const QK_STEPS: usize = SCORE_HALF / 16;
const KEY_NT: usize = KEY_TILE_512 / 8;
const PV_STEPS: usize = KEY_TILE_512 / 16;
const WARP_DIMS: usize = V_HALF / 2;
const DIM_PAIRS: usize = WARP_DIMS / 16;
/// f32 of one warp's published half-dim scores: 32 lanes × `KEY_NT · 4`.
const EXCH_WARP: usize = 32 * KEY_NT * 4;
/// 16-byte chunks a thread copies of a key tile and of a value tile, and the
/// chunks of one staged key row and value half-row.
const K_ROW_CHUNKS: usize = ROW_WORDS / 4;
const V_ROW_CHUNKS: usize = V_ROW_WORDS / 4;
const K_CHUNKS: usize = KEY_TILE_512 * K_ROW_CHUNKS / THREADS;
const V_CHUNKS: usize = KEY_TILE_512 * V_ROW_CHUNKS / THREADS;
/// f16 pairs a thread stages of the query rows.
const Q_WORDS: usize = Q_TILE_WORDS / THREADS;
/// Byte offsets of the block's dynamic shared memory: the query rows, the
/// key tile, the value tile and the score exchange, in that order.
const KT_AT: usize = Q_TILE_WORDS * 4;
const VT_AT: usize = KT_AT + K_TILE_WORDS * 4;
const EX_AT: usize = VT_AT + V_TILE_WORDS * 4;
const DYN_BYTES: usize = EX_AT + WARPS * EXCH_WARP * 4;
const DYN_BYTES_U32: u32 = DYN_BYTES as u32;
const _: () = assert!(DYN_BYTES_U32 as usize == DYN_BYTES);
// `#[launch_contract(dynamic_shared = ...)]` takes a literal: the kernel's
// declaration spells the byte count out, and this is the two agreeing. It
// is inside the opt-in window of a cc 8.6 card (101,376 bytes), the
// smallest the engine serves.
const _: () = assert!(DYN_BYTES == 90_112 && DYN_BYTES <= 101_376);
// Every tile starts on a 128-byte line, so the permutation's lines are the
// layout's.
const _: () = assert!(KT_AT.is_multiple_of(128) && VT_AT.is_multiple_of(128));
const _: () = assert!(EX_AT.is_multiple_of(128));
// Whole lines a row, whole copies a thread, whole k16 steps.
const _: () = assert!((V_ROW_WORDS * 4).is_multiple_of(128) && (ROW_WORDS * 4).is_multiple_of(128));
const _: () = assert!(K_CHUNKS * THREADS == KEY_TILE_512 * K_ROW_CHUNKS);
const _: () = assert!(V_CHUNKS * THREADS == KEY_TILE_512 * V_ROW_CHUNKS);
const _: () = assert!(Q_WORDS * THREADS == Q_TILE_WORDS && THREADS.is_multiple_of(K_ROW_CHUNKS));
const _: () =
    assert!(THREADS.is_multiple_of(V_ROW_CHUNKS) && (2 * ROW_WORDS).is_multiple_of(THREADS));
const _: () = assert!(KEY_TILE_512 == 2 * 16 && QK_STEPS.is_multiple_of(LINE_PAIRS));
const _: () = assert!(DIM_PAIRS.is_multiple_of(LINE_PAIRS) && WARP_DIMS * 2 == V_HALF);
// The launch contract spells HEAD_512 out as 512 and PACK_4 as 4.
const _: () = assert!(HEAD_512 == 512 && PACK_4 == 4);

const _: () = assert!(layout_holds(Q_ROWS, ROW_WORDS) && layout_holds(KEY_TILE_512, ROW_WORDS));
const _: () = assert!(layout_holds(KEY_TILE_512, V_ROW_WORDS));

const _: () = assert!(pack_rows_hold::<PACK_4>(SLICES));

/// The attention of the module doc in blocks of `PACK` query heads, `packs`
/// of them per key head, over an f16 cache: the body of
/// `gqa_prefill_flash_512_p4` (`PACK = 4`). Block `b` is value half `b % 2`
/// and unit `r = (b / 2) % (n_kv · packs)` (key head `r / packs`, query
/// heads `head0 = r·PACK ..`) of query tile `n_tiles − 1 − (b / 2) / (n_kv ·
/// packs)`, rows `(Q_ROWS / PACK)·tile ..` of `t_rows`. Row `t`'s query head
/// `head0 + j` is read at `q[(t · n_head + head0 + j)·512 ..]` and its value
/// half's outputs written at `y[(t · n_head + head0 + j)·512 + 256·vh ..]`.
/// Every guard around a shuffle or a barrier is block- or warp-pair-uniform:
/// `live` and the unmasked path read the slice's largest and smallest
/// counts; a lane's two rows mask by their own. With `LOWER`, row `t` attends
/// over keys `[k_first_buf[t], n_keys[t])` and `k_first_buf` holds `t_rows`
/// first keys; without it, `k_first_buf` is never read and every row starts at
/// key 0.
///
/// # Safety
///
/// The launch holds the entry's contract at `n_kv · packs · PACK` query heads:
/// `n_keys_buf` holds `t_rows` counts, and with `LOWER` so does `k_first_buf`,
/// `q` and `y` hold `t_rows` rows of
/// `n_kv · packs · PACK` heads of 512 values, `kc` and `vc` hold `n_kv · ctx`
/// rows of 512 f16 each, `q` starts 8-byte and `kc`, `vc` 16-byte aligned.
/// The block is [`THREADS`] threads with [`DYN_BYTES`] bytes of dynamic shared
/// memory, whose window starts at the block's shared base (no static shared
/// memory in the entry), and every thread of the block runs the body.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn prefill_512_p<const LOWER: bool>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    k_first_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    t_rows: u32,
    packs: usize,
    fault: FaultSink,
    mut y: DisjointSlice<f32>,
) {
    const PACK: usize = PACK_4;
    const SLICE_POS: usize = SLICE_ROWS / PACK;
    const { assert!(SLICE_POS * PACK == SLICE_ROWS) };
    let positions = Q_ROWS / PACK;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let rows = t_rows as usize;
    let n_tiles = rows.div_ceil(positions);
    let units = nkv * packs;
    if b >= 2 * n_tiles * units {
        return; // block-uniform
    }
    let vh = b % 2;
    let u = b / 2;
    let qt = n_tiles - 1 - u / units;
    let r = u % units;
    let kh = r / packs;
    let head0 = r * PACK;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wid = tid / 32;
    // The warp's row slice and dim half.
    let rs = wid % SLICES;
    let dh = wid / SLICES;
    let g = lane / 4;
    let t4 = lane % 4;
    let n_head = units * PACK;
    let t0 = qt * positions;

    // The block's dynamic shared memory, DYN_BYTES bytes (the launch
    // contract): the query rows, the key tile, the value tile and the
    // exchange at byte offsets 0, KT_AT, VT_AT and EX_AT, every write ordered
    // before its reads by a block barrier (the cp.async copies by the wait
    // before it). The base is declared 16-byte aligned, what the 16-byte
    // copies and `ldmatrix` rows need; with no static shared memory here it
    // is the window's start, so the tiles' 128-byte lines are the layout's.
    let (qs, kt, vt, ex) = (
        DynamicSharedArray::<u32, 16>::get(),
        DynamicSharedArray::<u32, 16>::offset(KT_AT),
        DynamicSharedArray::<u32, 16>::offset(VT_AT),
        DynamicSharedArray::<f32, 16>::offset(EX_AT),
    );

    // Every row's count as the walk uses it (the 256 body's rule).
    prefill_pack_counts! {
        out: (hi, warp_hi, warp_lo, any_bad, cnt, bad),
        shape: (SLICES, SLICE_POS, PACK),
        rows: (t0, rows, rs, g),
        counts: (n_keys_buf, ctx),
        lower: (LOWER, k_first_buf),
        first: (lo, warp_first_lo, warp_first_hi, first),
    }
    // The false monomorph never reads the first-key bounds.
    if !LOWER {
        let _ = (lo, warp_first_lo, warp_first_hi, first);
    }
    if lane == 0 && dh == 0 && vh == 0 && any_bad {
        fault.raise(FaultSite::KeyCount);
    }

    // The block's query rows, rounded to f16 pairs: staged row `lp·PACK + j`
    // is position `t0 + lp`'s head `head0 + j`, so row slice `rs` is rows
    // `16·rs ..`. A row past `t_rows` is zero. Pass `e` of thread `tid`
    // stages word `wd = (tid + THREADS·e) % ROW_WORDS` of row `(tid +
    // THREADS·e) / ROW_WORDS`, consecutive threads reading consecutive u64.
    let q64 = q.as_ptr().cast::<u64>();
    for e in 0..Q_WORDS {
        cuda_device::thread::__unroll_config::<8>();
        let i = tid + THREADS * e;
        let row = i / ROW_WORDS;
        let wd = i % ROW_WORDS;
        let t = t0 + row / PACK;
        let raw = if t < rows {
            // SAFETY: t < t_rows and the head head0 + row % PACK < head0 +
            // PACK <= n_head, so the row lies inside q (launch contract); a
            // row is 512 f32 = 256 u64 and q starts 8-byte aligned
            // (host-checked), so the u64 read is aligned.
            unsafe { *q64.add((t * n_head + head0 + row % PACK) * ROW_WORDS + wd) }
        } else {
            0
        };
        let pair = f32x2_to_f16x2_bits(
            f32::from_bits(raw as u32),
            f32::from_bits((raw >> 32) as u32),
        );
        // SAFETY: row < Q_ROWS and wd < ROW_WORDS: inside the query rows; one
        // owner per word.
        unsafe { *qs.add(staged(row, wd, ROW_WORDS)) = pair };
    }

    // The f16 copies: thread `tid`'s copies of a key tile are row `tid /
    // K_ROW_CHUNKS + (THREADS / K_ROW_CHUNKS)·i`, chunk `tid % K_ROW_CHUNKS`;
    // of a value tile row `tid / V_ROW_CHUNKS + (THREADS / V_ROW_CHUNKS)·i`,
    // chunk `tid % V_ROW_CHUNKS` of the block's value half.
    let k_rows = kc.as_ptr().cast::<u32>();
    let v_rows = vc.as_ptr().cast::<u32>();
    let plane0 = kh * ctx as usize * ROW_WORDS;
    let k_src = k_rows.wrapping_add(plane0);
    let v_src = v_rows.wrapping_add(plane0 + V_ROW_WORDS * vh);
    let tiles = hi.div_ceil(KEY_TILE_U32);
    // The walk's first tile: the one holding the block's smallest first key.
    let kb_first = if LOWER {
        (lo / KEY_TILE_U32).min(tiles)
    } else {
        0
    };
    // Stage tile `kb`; `full` means every staged key is below `hi` and, with
    // `LOWER`, at or above `lo`. A key below `lo` is staged as a key past `hi` is.
    macro_rules! stage {
        ($dst:ident, $src:ident, $words:ident, $chunks:ident, $copies:ident, $kb:expr, $full:literal) => {{
            let kb: u32 = $kb;
            for i in 0..$copies {
                cuda_device::thread::__unroll_config::<0>();
                let row = tid / $chunks + (THREADS / $chunks) * i;
                let col = 4 * (tid % $chunks);
                // SAFETY: row < KEY_TILE_512; this thread owns one aligned
                // four-word chunk inside the staged row.
                let d = unsafe { $dst.add(staged(row, col, $words)) };
                let key = kb * KEY_TILE_U32 + (row as u32);
                let inside = if LOWER {
                    key < hi && key >= lo
                } else {
                    key < hi
                };
                if $full || inside {
                    // SAFETY: the key is below hi <= ctx; rows are 1,024 bytes,
                    // value halves start 512 bytes in, and kc, vc are 16-byte
                    // aligned (host-checked), as is this thread's chunk.
                    unsafe {
                        cp_async_cg_16(
                            d,
                            $src.wrapping_add((kb as usize * KEY_TILE_512 + row) * ROW_WORDS + col),
                        )
                    };
                } else {
                    // SAFETY: this thread owns the aligned destination chunk;
                    // readers wait for the following block barrier.
                    unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                }
            }
            // SAFETY: commits this thread's preceding copies.
            unsafe { cp_async_commit_group() };
        }};
    }
    if kb_first < tiles {
        let full = if LOWER {
            KEY_TILE_U32 * (kb_first + 1) <= hi && KEY_TILE_U32 * kb_first >= lo
        } else {
            KEY_TILE_U32 * (kb_first + 1) <= hi
        };
        if full {
            stage!(kt, k_src, ROW_WORDS, K_ROW_CHUNKS, K_CHUNKS, kb_first, true);
        } else {
            stage!(
                kt,
                k_src,
                ROW_WORDS,
                K_ROW_CHUNKS,
                K_CHUNKS,
                kb_first,
                false
            );
        }
    }

    // This lane's `ldmatrix` addresses in the first line of the warp's
    // half, one per chunk pair `j`: the query rows `16·rs + lane % 16`, the
    // key and value rows of the tile's first sixteen; every other address of
    // the walk is one of these plus a constant (`layout_holds`).
    let q_row = SLICE_ROWS * rs + lane % 16;
    let k_row = lane % 8 + 8 * (lane / 16);
    let v_row = lane % 8 + 8 * ((lane / 8) % 2);
    let score_bytes = (4 * (SCORE_HALF / 2) * dh) as u32;
    let value_bytes = (4 * (WARP_DIMS / 2) * dh) as u32;
    let mut q_at = [0u32; LINE_PAIRS];
    let mut k_at = [0u32; LINE_PAIRS];
    let mut v_at = [0u32; LINE_PAIRS];
    for j in 0..LINE_PAIRS {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: q_row < Q_ROWS and words 8·j + 4 .. + 4 <= LINE < ROW_WORDS:
        // inside the query rows; only the address is formed here (the half's
        // offset stays inside the row).
        let q_ptr = unsafe { qs.add(staged(q_row, 8 * j + 4 * (lane / 16), ROW_WORDS)) };
        // SAFETY: q_ptr points inside the block's live shared query tile.
        q_at[j] = unsafe { shared::cvta_generic_to_shared_u32(q_ptr.cast_const().cast::<u8>()) }
            + score_bytes;
        // SAFETY: k_row < 16 <= KEY_TILE_512 and words 8·j + 4 .. + 4 <= LINE
        // < ROW_WORDS: inside the key tile; only the address is formed here.
        let k_ptr = unsafe { kt.add(staged(k_row, 8 * j + 4 * ((lane / 8) % 2), ROW_WORDS)) };
        // SAFETY: k_ptr points inside the block's live shared key tile.
        k_at[j] = unsafe { shared::cvta_generic_to_shared_u32(k_ptr.cast_const().cast::<u8>()) }
            + score_bytes;
        // SAFETY: v_row < 16 <= KEY_TILE_512 and words 8·j + 4 .. + 4 <= LINE
        // <= V_ROW_WORDS: inside the value tile; only the address is formed
        // here.
        let v_ptr = unsafe { vt.add(staged(v_row, 8 * j + 4 * (lane / 16), V_ROW_WORDS)) };
        // SAFETY: v_ptr points inside the block's live shared value tile.
        v_at[j] = unsafe { shared::cvta_generic_to_shared_u32(v_ptr.cast_const().cast::<u8>()) }
            + value_bytes;
    }
    // This lane's slots of its own and its partner's published scores.
    let ex_own = ex.wrapping_add(wid * EXCH_WARP + lane);
    let ex_mate = ex.wrapping_add((wid ^ SLICES) * EXCH_WARP + lane);

    let t4k = 2 * t4 as u32;
    let mut m = [f32::NEG_INFINITY; 2];
    let mut l = [0.0f32; 2];
    let mut o = [[0.0f32; 4]; 2 * DIM_PAIRS];
    let mut kb = kb_first;
    while kb < tiles {
        let end = KEY_TILE_U32 * (kb + 1);
        // Key tile kb has landed (and, on the first, the query rows are
        // written), and every warp is done reading the value tile of kb − 1
        // and the exchange of kb − 1.
        // SAFETY: waits for this thread's outstanding groups; the barrier
        // then publishes every thread's copies.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        let full_v = if LOWER {
            end <= hi && KEY_TILE_U32 * kb >= lo
        } else {
            end <= hi
        };
        if full_v {
            stage!(vt, v_src, V_ROW_WORDS, V_ROW_CHUNKS, V_CHUNKS, kb, true);
        } else {
            stage!(vt, v_src, V_ROW_WORDS, V_ROW_CHUNKS, V_CHUNKS, kb, false);
        }
        // Both warps of a pair hold the same rows, so the same `live`.
        let live = if LOWER {
            KEY_TILE_U32 * kb < warp_hi && end > warp_first_lo
        } else {
            KEY_TILE_U32 * kb < warp_hi
        };

        // ---- S_w = Q·Kᵀ over the warp's half, published.
        let mut sc = [[0.0f32; 4]; KEY_NT];
        if live {
            for kk in 0..QK_STEPS {
                cuda_device::thread::__unroll_config::<0>();
                let line = (4 * LINE * (kk / LINE_PAIRS)) as u32;
                // SAFETY: query row q_row's chunk pair 2·kk of the warp's
                // half, whole inside the query rows (`layout_holds`),
                // published by the barrier above; every lane issues the load.
                let qa = unsafe { wmma::ldmatrix_x4_shared_u32(q_at[kk % LINE_PAIRS] + line) };
                for nj in 0..KEY_NT / 2 {
                    cuda_device::thread::__unroll_config::<0>();
                    // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2 of the
                    // warp's half.
                    let at = k_at[kk % LINE_PAIRS] + line + (4 * 16 * nj * ROW_WORDS) as u32;
                    // SAFETY: the row is below KEY_TILE_512 and the chunk is
                    // whole inside KT (`layout_holds`), published by the
                    // barrier above; every lane issues the load.
                    let bf = unsafe { wmma::ldmatrix_x4_shared_u32(at) };
                    // SAFETY: the whole warp issues these mma.sync with the
                    // fragments it loaded (`live` is warp-uniform).
                    sc[2 * nj] =
                        unsafe { wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qa, [bf[0], bf[1]]) };
                    // SAFETY: every lane issues this MMA on qa and bf; live is warp-uniform.
                    sc[2 * nj + 1] =
                        unsafe { wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qa, [bf[2], bf[3]]) };
                }
            }
            for nt in 0..KEY_NT {
                cuda_device::thread::__unroll_config::<0>();
                for j in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: (4·nt + j)·32 + lane < EXCH_WARP: this lane's
                    // own slot of this warp's exchange.
                    unsafe { *ex_own.add((4 * nt + j) * 32) = sc[nt][j] };
                }
            }
        }
        // Both halves are published.
        thread::sync_threads();

        let mut pa = [[0u32; 4]; PV_STEPS];
        let mut f = [1.0f32; 2];
        if live {
            // S = S_w + S_mate = S_lo + S_hi: a rounded add is commutative,
            // so both warps of the pair hold the same bits.
            for nt in 0..KEY_NT {
                cuda_device::thread::__unroll_config::<0>();
                for j in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: the partner's slot for the same fragment value,
                    // written before the barrier above.
                    let mate = unsafe { *ex_mate.add((4 * nt + j) * 32) };
                    sc[nt][j] = add_rn_f32(sc[nt][j], mate);
                }
            }

            if LOWER {
                prefill_softmax! {
                    unmasked: end <= warp_lo && KEY_TILE_U32 * kb >= warp_first_hi,
                    key0: KEY_TILE_U32 * kb + t4k,
                    tiles: (KEY_NT, PV_STEPS),
                    scores: sc,
                    counts: cnt,
                    lower: (LOWER, first),
                    scale: scale,
                    state: (m, l, f),
                    weights: pa,
                }
            } else {
                prefill_softmax! {
                    unmasked: end <= warp_lo,
                    key0: KEY_TILE_U32 * kb + t4k,
                    tiles: (KEY_NT, PV_STEPS),
                    scores: sc,
                    counts: cnt,
                    scale: scale,
                    state: (m, l, f),
                    weights: pa,
                }
            }

            if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                for nd in 0..2 * DIM_PAIRS {
                    cuda_device::thread::__unroll_config::<0>();
                    for j in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                    }
                }
            }
        }

        // The value tile has landed, and every warp is done reading the key
        // tile.
        // SAFETY: waits until none of this thread's groups is pending (its
        // copies of value tile kb); the barrier then publishes every thread's
        // copies before the value product reads VT.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        if kb + 1 < tiles {
            let full_next = if LOWER {
                end + KEY_TILE_U32 <= hi && KEY_TILE_U32 * (kb + 1) >= lo
            } else {
                end + KEY_TILE_U32 <= hi
            };
            if full_next {
                stage!(kt, k_src, ROW_WORDS, K_ROW_CHUNKS, K_CHUNKS, kb + 1, true);
            } else {
                stage!(kt, k_src, ROW_WORDS, K_ROW_CHUNKS, K_CHUNKS, kb + 1, false);
            }
        }

        // ---- O += P̂·V over the warp's 128 value dims.
        if live {
            for kk2 in 0..PV_STEPS {
                cuda_device::thread::__unroll_config::<0>();
                for nd in 0..DIM_PAIRS {
                    cuda_device::thread::__unroll_config::<0>();
                    // Value row 16·kk2 + v_row, chunk 2·nd + lane/16 of the
                    // warp's dims.
                    let at = v_at[nd % LINE_PAIRS]
                        + (4 * (16 * kk2 * V_ROW_WORDS + LINE * (nd / LINE_PAIRS))) as u32;
                    // SAFETY: the row is below KEY_TILE_512 and the chunk is
                    // whole inside VT (`layout_holds`), published by the
                    // barrier above; every lane issues the load.
                    let bf = unsafe { wmma::ldmatrix_x4_trans_shared_u32(at) };
                    // SAFETY: the whole warp issues these mma.sync with the
                    // fragments it holds (`live` is warp-uniform).
                    o[2 * nd] =
                        unsafe { wmma::mma_m16n8k16_f32_f16(o[2 * nd], pa[kk2], [bf[0], bf[1]]) };
                    // SAFETY: every lane issues this MMA on pa and bf; live is warp-uniform.
                    o[2 * nd + 1] = unsafe {
                        wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pa[kk2], [bf[2], bf[3]])
                    };
                }
            }
        }
        kb += 1;
    }

    // ---- the row sums over the four lanes of a row, then o · (1/l): one
    // owner lane per value over the two value halves.
    prefill_pack_store! {
        acc: (o, l, bad, 2 * DIM_PAIRS),
        rows: (t0 + SLICE_POS * rs, rows, PACK, g, t4),
        cols: (V_HALF * vh + WARP_DIMS * dh, HEAD_512),
        out: (y, n_head, head0),
    }
}

#[cuda_module]
mod flash_gqa_prefill_512_kernels {
    use super::*;

    /// The module doc's attention in blocks of [`PACK_4`] query heads,
    /// `packs` of them per key head ([`prefill_512_p`]): a block
    /// is 8 positions × four heads and one value half, a row slice four
    /// positions × four heads.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 90112,
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 4 * 512,
            kc.len() >= n_kv * ctx * 512,
            vc.len() >= n_kv * ctx * 512,
            y.len() >= t_rows * n_kv * packs * 4 * 512
        )
    )]
    pub fn gqa_prefill_flash_512_p4(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_512_p`'s at `packs · PACK_4`
        // heads per key head with `LOWER` false, so the first-key buffer is
        // not read (the counts stand in). The block is THREADS threads with
        // DYN_BYTES bytes of dynamic shared memory.
        unsafe {
            prefill_512_p::<false>(
                q,
                kc,
                vc,
                n_keys_buf,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }

    /// [`gqa_prefill_flash_512_p4`] with a first key a row ([`prefill_512_p`]
    /// with `LOWER`): row `t` attends over keys `[k_first[t], n_keys[t])`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128, 1)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        dynamic_shared = 90112,
        requires = (
            n_keys_buf.len() >= t_rows,
            k_first_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 4 * 512,
            kc.len() >= n_kv * ctx * 512,
            vc.len() >= n_kv * ctx * 512,
            y.len() >= t_rows * n_kv * packs * 4 * 512
        )
    )]
    pub fn gqa_prefill_flash_512_p4_lower(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        k_first_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_512_p`'s at `packs · PACK_4`
        // heads per key head with `k_first_buf` bound at `t_rows`, and the
        // block is THREADS threads with DYN_BYTES bytes of dynamic shared
        // memory.
        unsafe {
            prefill_512_p::<true>(
                q,
                kc,
                vc,
                n_keys_buf,
                k_first_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct FlashGqaPrefill512 {
    module: flash_gqa_prefill_512_kernels::LoadedModule,
}

impl FlashGqaPrefill512 {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashGqaPrefill512, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(flash_gqa_prefill_512_kernels, ctx)? };
        Ok(FlashGqaPrefill512 { module })
    }

    /// Enqueue `t` rows' attention over heads of [`HEAD_512`] values:
    /// `args.q` and `args.y` hold `t · n_head` rows of 512 f32, the planes
    /// `n_kv · ctx` rows of 512 f16, and `n_head` is a nonzero multiple of
    /// `4 · n_kv`. One launch of `2 · ⌈t / 8⌉ · n_head / 4` blocks of 128
    /// threads with 90,112 bytes of dynamic shared memory. A group that is
    /// not a multiple of four, an empty launch, a short buffer, or a `q` not
    /// 8-byte aligned or `kc`/`vc` not 16-byte aligned is refused by name
    /// before any launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_p4(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill_512::enqueue_p4";
        let packs = crate::flash_gqa_prefill::checked_packed(
            what,
            args.q,
            crate::flash_gqa::CachePlanes::F16 {
                kc: args.kc,
                vc: args.vc,
            },
            args.n_keys,
            &*args.y,
            (args.n_head, args.n_kv, args.ctx, args.t),
            (HEAD_512, KEY_TILE_512, PACK_4),
        )?;
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head: _,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        let grid = launch_u32(what, "grid", 2 * t.div_ceil(Q_ROWS / PACK_4) * n_kv * packs)?;
        let packs = launch_u32(what, "packs", packs)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep = self
            .module
            .prepare_gqa_prefill_flash_512_p4(LaunchConfig1D::new(
                grid,
                THREADS_U32,
                DYN_BYTES_U32,
            ))?;
        self.module.gqa_prefill_flash_512_p4(
            stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, t, packs, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaPrefill512::enqueue_p4`] with a first key a row: row `t`
    /// attends over keys `[k_first[t], n_keys[t])`, `k_first` on the device,
    /// `t` of them (refused by name when shorter). A first key at or past its
    /// row's count raises [`FaultSite::KeyCount`] and leaves that row NaN.
    /// The grid, the block and the refusals are [`FlashGqaPrefill512::enqueue_p4`]'s.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_p4_lower(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
        k_first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill_512::enqueue_p4_lower";
        if k_first.len() < args.t {
            return Err(GpuError::shape(
                what,
                format!("k_first.len() {} < {}", k_first.len(), args.t),
            ));
        }
        let packs = crate::flash_gqa_prefill::checked_packed(
            what,
            args.q,
            crate::flash_gqa::CachePlanes::F16 {
                kc: args.kc,
                vc: args.vc,
            },
            args.n_keys,
            &*args.y,
            (args.n_head, args.n_kv, args.ctx, args.t),
            (HEAD_512, KEY_TILE_512, PACK_4),
        )?;
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head: _,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        let grid = launch_u32(what, "grid", 2 * t.div_ceil(Q_ROWS / PACK_4) * n_kv * packs)?;
        let packs = launch_u32(what, "packs", packs)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep = self
            .module
            .prepare_gqa_prefill_flash_512_p4_lower(LaunchConfig1D::new(
                grid,
                THREADS_U32,
                DYN_BYTES_U32,
            ))?;
        self.module.gqa_prefill_flash_512_p4_lower(
            stream, &prep, q, kc, vc, n_keys, k_first, scale, n_kv, ctx, t, packs, fault, y,
        )?;
        Ok(())
    }
}
