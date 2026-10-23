//! FP4 tensor-core products over routed NVFP4 planes.

use super::{GemmAct32, GemmRoute};
use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, ptx_asm, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

const TILE_ROWS: usize = 128;
const TILE_THREADS: u32 = 256;

pub(super) struct Act {
    q: DeviceBuffer<u32>,
    e: DeviceBuffer<u32>,
    scale: DeviceBuffer<f32>,
}

impl Act {
    pub(super) fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        Ok(Self {
            q: DeviceBuffer::zeroed(stream, cols * k / 8)?,
            e: DeviceBuffer::zeroed(stream, cols * k / 64)?,
            scale: DeviceBuffer::zeroed(stream, cols)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.q.num_bytes() + self.e.num_bytes() + self.scale.num_bytes()
    }
}

#[cuda_device::device]
fn nvfp4_pack_q8(words: [u32; 4], d: f32, gs: f32) -> ([u32; 2], u32) {
    let mut values = [0.0f32; 16];
    #[unroll]
    for i in 0..16 {
        values[i] = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * d;
    }
    let mut largest = 0.0f32;
    #[unroll]
    for i in 0..16 {
        largest = largest.max(values[i].abs());
    }
    let mut scale = if gs.is_finite() {
        let x = largest / (6.0 * gs);
        u32::from(cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(x, x)) & 255
    } else {
        0
    };
    let decoded = crate::mxfp4::ue4m3_half(scale) * 2.0 * gs;
    let mut inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
    if !largest.is_finite()
        || !(6.0 * gs).is_finite()
        || decoded < f32::MIN_POSITIVE
        || !inv.is_finite()
    {
        let nonzero = words[0] | words[1] | words[2] | words[3];
        let ratio = if gs > 0.0 && nonzero != 0 {
            d / gs
        } else {
            0.0
        };
        largest = 0.0;
        #[unroll]
        for i in 0..16 {
            values[i] = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * ratio;
            largest = largest.max(values[i].abs());
        }
        let x = largest / 6.0;
        scale = if gs.is_finite() {
            u32::from(cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(x, x)) & 255
        } else {
            0
        };
        let decoded = crate::mxfp4::ue4m3_half(scale) * 2.0;
        inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
    }
    let mut packed = [0u32; 2];
    #[unroll]
    for i in 0..8 {
        let byte: u32;
        // SAFETY: this conversion only reads and writes registers.
        unsafe {
            ptx_asm!(
                "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; cvt.u32.u8 %0, b; }",
                out("=r") byte, in("f") values[2 * i] * inv, in("f") values[2 * i + 1] * inv,
            );
        }
        packed[i / 4] |= byte << (8 * (i & 3));
    }
    (packed, scale)
}

// Bound each block's cache to 32 KiB. Wider rows recompute the uncached quads.
const ACT_CACHE_VALUES: usize = 8192;
const _: () = assert!((ACT_CACHE_VALUES + 2) * 4 == 32776);

/// # Safety
/// Both inputs contain the aligned four-value span at `base`.
#[inline(always)]
unsafe fn act_quad(g: &[f32], u: &[f32], base: usize, code: u32, limit: f32) -> [f32; 4] {
    // SAFETY: the caller supplies an aligned, in-bounds gate quad.
    let g = unsafe { (*g.as_ptr().add(base).cast::<cuda_device::vector::F32x4>()).0 };
    // SAFETY: the caller supplies an aligned, in-bounds up quad.
    let u = unsafe { (*u.as_ptr().add(base).cast::<cuda_device::vector::F32x4>()).0 };
    if crate::fault::quad_finite(g) & crate::fault::quad_finite(u) {
        [
            crate::kquant::act::apply(code, limit, g[0], u[0]),
            crate::kquant::act::apply(code, limit, g[1], u[1]),
            crate::kquant::act::apply(code, limit, g[2], u[2]),
            crate::kquant::act::apply(code, limit, g[3], u[3]),
        ]
    } else {
        [f32::NAN; 4]
    }
}

#[cuda_device::device]
fn fused_swiglu(
    g: &[f32],
    u: &[f32],
    code: u32,
    limit: f32,
    k: u32,
    mut q: DisjointSlice<u32>,
    mut d: DisjointSlice<f32>,
    mut sums: DisjointSlice<i32>,
    mut out: DisjointSlice<u32>,
    mut scales: DisjointSlice<u32>,
    mut global: DisjointSlice<f32>,
    fault: crate::fault::FaultSink,
) {
    let col = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let k = k as usize;
    let cached = k.min(ACT_CACHE_VALUES);
    let cache = cuda_device::DynamicSharedArray::<f32>::get();
    let mut largest = 0.0f32;
    let mut at = 4 * tid;
    while at < k {
        // SAFETY: K divides 64; the loop's aligned quad is inside both checked input columns.
        let v = unsafe { act_quad(g, u, col * k + at, code, limit) };
        if at < cached {
            // SAFETY: this thread owns the aligned quad inside the allocated shared prefix.
            unsafe {
                *cache.add(at).cast::<cuda_device::vector::F32x4>() =
                    cuda_device::vector::F32x4::new(v);
            }
        }
        largest = if crate::fault::quad_finite(v) {
            largest
                .max(v[0].abs())
                .max(v[1].abs())
                .max(v[2].abs())
                .max(v[3].abs())
        } else {
            f32::INFINITY
        };
        at += 256;
    }
    let mut delta = 16;
    while delta > 0 {
        largest = largest.max(warp::shuffle_xor_f32(largest, delta));
        delta /= 2;
    }
    if lane == 0 {
        // SAFETY: each warp owns one of the two shared reduction words after the cached prefix.
        unsafe {
            *cache.add(cached + tid / 32) = largest;
        }
    }
    thread::sync_threads();
    // SAFETY: the first warp published its maximum in the first reduction word.
    let first = unsafe { *cache.add(cached) };
    // SAFETY: the second warp published its maximum in the second reduction word.
    let second = unsafe { *cache.add(cached + 1) };
    let largest = first.max(second);
    let gs = if !largest.is_finite() {
        f32::NAN
    } else if largest > 0.0 {
        (largest / 127.0) * (127.0 / 6.0)
    } else {
        1.0
    };
    if tid == 0 {
        // SAFETY: this block owns the live column's global scale.
        unsafe {
            *global.get_unchecked_mut(col) = gs;
        }
    }
    let blocks = k / 32;
    let steps = k / 64;
    let mut group = tid / 32;
    while group < blocks.div_ceil(4) {
        let block = (4 * group + (lane >> 3)).min(blocks - 1);
        let at = 32 * block + 4 * (lane & 7);
        let v = if at < cached {
            // SAFETY: this aligned quad is inside the published shared prefix.
            unsafe { (*cache.add(at).cast::<cuda_device::vector::F32x4>()).0 }
        } else {
            // SAFETY: this aligned quad is inside both checked input columns.
            unsafe { act_quad(g, u, col * k + at, code, limit) }
        };
        // SAFETY: each warp owns one group of four 32-value blocks in this checked column.
        let refused = unsafe {
            super::act32::quant32_group(
                v, col, group, blocks, steps, lane, &mut q, &mut d, &mut sums,
            )
        };
        if refused && lane & 7 == 0 {
            fault.raise(crate::FaultSite::QuantColumn);
        }
        group += 2;
    }
    thread::sync_threads();
    let mut block = tid;
    while block - lane < k / 16 {
        let source = block.min(k / 16 - 1);
        let at = col * steps * 16 + source * 4;
        // SAFETY: this aligned four-word block is in the column's published Q8 plane.
        let words = unsafe { (*q.as_mut_ptr().add(at).cast::<cuda_device::vector::U32x4>()).0 };
        // SAFETY: the block barrier published this checked column's Q8 scales.
        let dx = unsafe { *d.as_mut_ptr().add(col * 2 * steps + source / 2) };
        let (packed, scale) = nvfp4_pack_q8(words, dx, gs);
        if block < k / 16 {
            // SAFETY: this lane owns the first word of the live 16-value sub-block.
            unsafe {
                *out.get_unchecked_mut(col * k / 8 + 2 * block) = packed[0];
            }
            // SAFETY: this lane owns the second word of the live 16-value sub-block.
            unsafe {
                *out.get_unchecked_mut(col * k / 8 + 2 * block + 1) = packed[1];
            }
        }
        let mut word = scale << (8 * (lane & 3));
        word |= warp::shuffle_xor(word, 1);
        word |= warp::shuffle_xor(word, 2);
        if block < k / 16 && lane & 3 == 0 {
            // SAFETY: one lane per quad owns these four packed scales.
            unsafe {
                *scales.get_unchecked_mut(col * steps + block / 4) = word;
            }
        }
        block += 64;
    }
}

#[inline(always)]
fn packed_pair(a: u32, b: u32) -> [u32; 2] {
    let (lo, hi): (u32, u32);
    // SAFETY: shifts, nibble selection and byte permutation use registers only.
    unsafe {
        ptx_asm!(
            "{ .reg .b32 a4, a8, b4, b8, al, ah, bl, bh;\n\
             shr.u32 a4, %2, 4; shr.u32 a8, %2, 8;\n\
             shr.u32 b4, %3, 4; shr.u32 b8, %3, 8;\n\
             lop3.b32 al, %2, a4, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 ah, a4, a8, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 bl, %3, b4, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 bh, b4, b8, 0x0f0f0f0f, 0xe4;\n\
             prmt.b32 %0, al, bl, 0x6420;\n\
             prmt.b32 %1, ah, bh, 0x6420; }",
            out("=r") lo, out("=r") hi, in("r") a, in("r") b,
        );
    }
    [lo, hi]
}

#[inline(always)]
fn mma(c: [f32; 4], a: [u32; 4], b: [u32; 2], sa: u32, sb: u32) -> [f32; 4] {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    // SAFETY: the caller executes this register-only instruction in every lane of its warp.
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,0}, %15, {0,0};",
            out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sa, in("r") sb,
        );
    }
    [d0, d1, d2, d3]
}

#[cuda_module]
mod native {
    use super::*;
    use cuda_device::shared::cvta_generic_to_shared_u32;
    use cuda_device::wmma::{ldmatrix_x2_shared_u32, ldmatrix_x4_shared_u32};

    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared_range = (264, 32776))]
    pub fn swiglu_nvfp4_32(
        g: &[f32],
        u: &[f32],
        code: u32,
        limit: f32,
        k: u32,
        q: DisjointSlice<u32>,
        d: DisjointSlice<f32>,
        sums: DisjointSlice<i32>,
        out: DisjointSlice<u32>,
        scales: DisjointSlice<u32>,
        global: DisjointSlice<f32>,
        fault: crate::fault::FaultSink,
    ) {
        fused_swiglu(g, u, code, limit, k, q, d, sums, out, scales, global, fault);
    }

    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared_range = (264, 32776))]
    pub fn swiglu_nvfp4_32_sel(
        g: &[f32],
        u: &[f32],
        sel: &[u32],
        n_card: u32,
        code: u32,
        limit: f32,
        k: u32,
        q: DisjointSlice<u32>,
        d: DisjointSlice<f32>,
        sums: DisjointSlice<i32>,
        out: DisjointSlice<u32>,
        scales: DisjointSlice<u32>,
        global: DisjointSlice<f32>,
        fault: crate::fault::FaultSink,
    ) {
        let place = sel[thread::blockIdx_x() as usize];
        if place == crate::hybrid::HOST {
            return;
        }
        if place >= n_card {
            if thread::threadIdx_x() == 0 {
                fault.raise(crate::FaultSite::ExpertId);
            }
            return;
        }
        fused_swiglu(g, u, code, limit, k, q, d, sums, out, scales, global, fault);
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn quantize_nvfp4_32(
        q: &[u32],
        d: &[f32],
        steps: u32,
        mut out: DisjointSlice<u32>,
        mut scales: DisjointSlice<u32>,
        mut global: DisjointSlice<f32>,
    ) {
        let col = thread::blockIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let steps = steps as usize;
        let mut max = 0.0f32;
        let mut block = lane;
        while block < 2 * steps {
            let at = col * 16 * steps + 8 * block;
            let nonzero = q[at]
                | q[at + 1]
                | q[at + 2]
                | q[at + 3]
                | q[at + 4]
                | q[at + 5]
                | q[at + 6]
                | q[at + 7];
            // A refused block's NaN scale makes the column's global scale NaN,
            // so every output of the column reads NaN.
            let dx = d[col * 2 * steps + block];
            if !dx.is_finite() {
                max = f32::INFINITY;
            } else if nonzero != 0 {
                max = max.max(dx);
            }
            block += 32;
        }
        let mut delta = 16;
        while delta > 0 {
            max = max.max(warp::shuffle_xor_f32(max, delta));
            delta /= 2;
        }
        let gs = if !max.is_finite() {
            f32::NAN
        } else if max > 0.0 {
            max * (127.0 / 6.0)
        } else {
            1.0
        };
        if lane == 0 {
            // SAFETY: one block owns this column, below the allocated column count.
            unsafe {
                *global.get_unchecked_mut(col) = gs;
            }
        }
        block = lane;
        while block - lane < 4 * steps {
            let source = block.min(4 * steps - 1);
            let at = col * steps * 16 + source * 4;
            let dx = d[col * 2 * steps + source / 2];
            let words = [q[at], q[at + 1], q[at + 2], q[at + 3]];
            let (packed, scale) = nvfp4_pack_q8(words, dx, gs);
            if block < 4 * steps {
                // SAFETY: this lane owns the live sub-block's first word.
                unsafe {
                    *out.get_unchecked_mut(col * steps * 8 + block * 2) = packed[0];
                }
                // SAFETY: this lane owns the live sub-block's second word.
                unsafe {
                    *out.get_unchecked_mut(col * steps * 8 + block * 2 + 1) = packed[1];
                }
            }
            let mut word = scale << (8 * (lane & 3));
            word |= warp::shuffle_xor(word, 1);
            word |= warp::shuffle_xor(word, 2);
            if block < 4 * steps && lane & 3 == 0 {
                // SAFETY: one lane in each live quad owns its four packed scales.
                unsafe {
                    *scales.get_unchecked_mut(col * steps + block / 4) = word;
                }
            }
            block += 32;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn gemm_nvfp4_mma(
        wq: &[u32],
        we: &[u32],
        ws: &[f32],
        aq: &[u32],
        ae: &[u32],
        ag: &[f32],
        cols: &[u32],
        tiles: &[u32],
        ntiles: &[u32],
        rows: u32,
        steps: u32,
        slots: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block / row_tiles as usize;
        let rt = block % row_tiles as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let rows = rows as usize;
        let steps = steps as usize;
        if tile == 0 {
            let refused = ntiles[1] as usize;
            let first = slots as usize - refused;
            let mut i = tid;
            while i < refused * TILE_ROWS {
                let row = rt * TILE_ROWS + i % TILE_ROWS;
                if row < rows {
                    // SAFETY: the route's refused slots are disjoint from live tiles; one thread owns this row.
                    unsafe {
                        *y.get_unchecked_mut(cols[first + i / TILE_ROWS] as usize * rows + row) =
                            f32::NAN;
                    }
                }
                i += TILE_THREADS as usize;
            }
        }
        if tile >= ntiles[0] as usize {
            return;
        }
        let expert = tiles[2 * tile] as usize;
        let (start, len) = super::super::route::tile_parts(tiles[2 * tile + 1]);
        let (start, len) = (start as usize, len as usize);
        static mut W: cuda_device::SharedArray<u32, 3072, 16> = cuda_device::SharedArray::UNINIT;
        static mut B: cuda_device::SharedArray<u32, 1536, 16> = cuda_device::SharedArray::UNINIT;
        static mut S: cuda_device::SharedArray<u32, 384, 16> = cuda_device::SharedArray::UNINIT;
        // SAFETY: W is this block's shared weight allocation.
        let wp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut W) };
        // SAFETY: B is this block's shared activation allocation.
        let bp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut B) };
        // SAFETY: S is this block's shared scale allocation.
        let sp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut S) };
        let row = rt * TILE_ROWS + 16 * (tid / 32);
        let g = lane / 4;
        let t = lane & 3;
        let nt = len.div_ceil(8);
        let sr = tid / 2;
        let half = tid & 1;
        let wr = expert * rows + (rt * TILE_ROWS + sr).min(rows - 1);
        let source = cols[start + sr.min(len - 1)] as usize;
        let col = if slot_div == 1 {
            source
        } else {
            source / slot_div as usize
        };
        let wi = wr * steps * 8 + 4 * half;
        // SAFETY: K is positive and divides 64; wr and both aligned halves are in the checked plane.
        let mut w = unsafe { (*wq.as_ptr().add(wi).cast::<cuda_device::vector::U32x4>()).0 };
        let mut av = cuda_device::vector::U32x4::new([0; 4]);
        if sr < 64 {
            // SAFETY: col is a live column and each half is an aligned four-word vector.
            av = unsafe {
                *aq.as_ptr()
                    .add(col * steps * 8 + 4 * half)
                    .cast::<cuda_device::vector::U32x4>()
            };
        }
        let mut sw = if half == 0 { we[wr * steps] } else { 0 };
        let mut sb = if half == 0 && sr < 64 {
            ae[col * steps]
        } else {
            0
        };
        let mut acc = [0.0f32; 32];
        let mut step = 0;
        while step < steps {
            // SAFETY: this selects one of W's two disjoint stages.
            let wp = unsafe { wp.add((step & 1) * 1536) };
            // SAFETY: this selects one of B's two disjoint stages.
            let bp = unsafe { bp.add((step & 1) * 768) };
            // SAFETY: this selects one of S's two disjoint stages.
            let sp = unsafe { sp.add((step & 1) * 192) };
            let p0 = packed_pair(w[0], w[1]);
            let p1 = packed_pair(w[2], w[3]);
            let packed = cuda_device::vector::U32x4::new([p0[0], p0[1], p1[0], p1[1]]);
            // SAFETY: each thread owns one aligned four-word half of its shared row.
            unsafe {
                *wp.add(sr * 12 + 4 * half)
                    .cast::<cuda_device::vector::U32x4>() = packed;
            }
            if sr < 64 {
                // SAFETY: each thread owns one aligned four-word half of its shared activation row.
                unsafe {
                    *bp.add(sr * 12 + 4 * half)
                        .cast::<cuda_device::vector::U32x4>() = av;
                }
            }
            if half == 0 {
                // SAFETY: the first thread of each row owns its weight scale word.
                unsafe {
                    *sp.add(sr) = sw;
                }
                if sr < 64 {
                    // SAFETY: the first thread of each live activation row owns its scale word.
                    unsafe {
                        *sp.add(TILE_ROWS + sr) = sb;
                    }
                }
            }
            thread::sync_threads();
            if step + 1 < steps {
                let wi = wr * steps * 8 + (step + 1) * 8 + 4 * half;
                // SAFETY: the next step is in the same aligned live weight row.
                w = unsafe { (*wq.as_ptr().add(wi).cast::<cuda_device::vector::U32x4>()).0 };
                if sr < 64 {
                    let ai = col * steps * 8 + (step + 1) * 8 + 4 * half;
                    // SAFETY: the next step is in this aligned live activation column.
                    av = unsafe { *aq.as_ptr().add(ai).cast::<cuda_device::vector::U32x4>() };
                }
                if half == 0 {
                    sw = we[wr * steps + step + 1];
                    if sr < 64 {
                        sb = ae[col * steps + step + 1];
                    }
                }
            }
            if row < rows {
                let r = 16 * (tid / 32) + g;
                let a_off = (16 * (tid / 32) + (lane & 15)) * 12 + 4 * (lane >> 4);
                // SAFETY: all lanes issue the load; its four sets of eight aligned rows lie in the live shared stage.
                let a = unsafe {
                    ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                        wp.add(a_off).cast_const().cast::<u8>(),
                    ))
                };
                // SAFETY: each pair of lanes reads the scale row selected by the MMA layout.
                let sa = unsafe { *sp.add(r + 8 * (t & 1)) };
                #[unroll]
                for n in 0..8 {
                    if n >= nt {
                        continue;
                    }
                    let br = n * 8 + g;
                    let b_off = (n * 8 + (lane & 7)) * 12 + 4 * ((lane >> 3) & 1);
                    // SAFETY: the whole warp loads two sets of eight aligned rows within this activation stage.
                    let b = unsafe {
                        ldmatrix_x2_shared_u32(cvta_generic_to_shared_u32(
                            bp.add(b_off).cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: the activation scale follows the weight scales in this 192-word stage.
                    let sb = unsafe { *sp.add(TILE_ROWS + br) };
                    let c = mma(
                        [acc[n * 4], acc[n * 4 + 1], acc[n * 4 + 2], acc[n * 4 + 3]],
                        a,
                        b,
                        sa,
                        sb,
                    );
                    acc[n * 4] = c[0];
                    acc[n * 4 + 1] = c[1];
                    acc[n * 4 + 2] = c[2];
                    acc[n * 4 + 3] = c[3];
                }
            }
            step += 1;
        }
        if row >= rows {
            return;
        }
        let weight_scale = ws[expert];
        #[unroll]
        for n in 0..8 {
            #[unroll]
            for j in 0..2 {
                let c = n * 8 + 2 * t + j;
                if c < len {
                    let slot = cols[start + c] as usize;
                    let column = if slot_div == 1 {
                        slot
                    } else {
                        slot / slot_div as usize
                    };
                    let input_scale = ag[column];
                    // SAFETY: this lane owns the first row of this live column; row counts divide 16.
                    unsafe {
                        *y.get_unchecked_mut(slot * rows + row + g) =
                            crate::gemm::grouped::scaled_sum(
                                [f64::from(acc[n * 4 + j]), 0.0],
                                [weight_scale, 0.0],
                                input_scale,
                                0.0,
                            ) as f32;
                    }
                    // SAFETY: this lane owns the row eight positions later in the same live column.
                    unsafe {
                        *y.get_unchecked_mut(slot * rows + row + g + 8) =
                            crate::gemm::grouped::scaled_sum(
                                [f64::from(acc[n * 4 + j + 2]), 0.0],
                                [weight_scale, 0.0],
                                input_scale,
                                0.0,
                            ) as f32;
                    }
                }
            }
        }
    }
}

pub(super) struct Kernels {
    module: native::LoadedModule,
}

impl Kernels {
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        // SAFETY: this package owns the embedded device module.
        Ok(Self {
            module: unsafe { crate::shared_module!(native, ctx)? },
        })
    }

    pub(super) fn quantize(
        &self,
        stream: &CudaStream,
        act: &mut GemmAct32,
    ) -> Result<(), GpuError> {
        let filled = act.filled() as u32;
        if let Some(out) = act.native.as_mut() {
            let prep = self
                .module
                .prepare_quantize_nvfp4_32(LaunchConfig1D::new(filled, 32, 0))?;
            self.module.quantize_nvfp4_32(
                stream,
                &prep,
                &act.q,
                &act.d,
                (act.k / 64) as u32,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
            )?;
        }
        Ok(())
    }

    pub(super) fn swiglu(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        selected: Option<(&DeviceBuffer<u32>, u32)>,
        rule: crate::kquant::Act,
        cols: u32,
        act: &mut GemmAct32,
        fault: crate::fault::FaultSink,
    ) -> Result<(), GpuError> {
        let (code, limit) = rule.code("NVFP4 activation")?;
        crate::tensor::check_alignment::<cuda_device::vector::F32x4, _>(
            "NVFP4 activation",
            "g",
            g,
        )?;
        crate::tensor::check_alignment::<cuda_device::vector::F32x4, _>(
            "NVFP4 activation",
            "u",
            u,
        )?;
        let k = act.k as u32;
        let shared = ((act.k.min(ACT_CACHE_VALUES) + 2) * 4) as u32;
        let out = act
            .native
            .as_mut()
            .ok_or_else(|| GpuError::shape("NVFP4 activation", "K must divide 64"))?;
        let config = LaunchConfig1D::new(cols, 64, shared);
        if let Some((sel, n_card)) = selected {
            let prep = self.module.prepare_swiglu_nvfp4_32_sel(config)?;
            self.module.swiglu_nvfp4_32_sel(
                stream,
                &prep,
                g,
                u,
                sel,
                n_card,
                code,
                limit,
                k,
                &mut act.q,
                &mut act.d,
                &mut act.s,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
                fault,
            )?;
        } else {
            let prep = self.module.prepare_swiglu_nvfp4_32(config)?;
            self.module.swiglu_nvfp4_32(
                stream,
                &prep,
                g,
                u,
                code,
                limit,
                k,
                &mut act.q,
                &mut act.d,
                &mut act.s,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
                fault,
            )?;
        }
        Ok(())
    }

    pub(super) fn product(
        &self,
        stream: &CudaStream,
        w: crate::mxfp4_sel::Nvfp4Stack<'_>,
        act: &GemmAct32,
        route: &GemmRoute,
        rows: u32,
        slots: u32,
        max_tiles: u32,
        slot_div: u32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let a = act
            .native
            .as_ref()
            .ok_or_else(|| GpuError::shape("NVFP4 MMA", "activation K must divide 64"))?;
        let rt = rows.div_ceil(TILE_ROWS as u32);
        let grid = max_tiles
            .checked_mul(rt)
            .ok_or_else(|| GpuError::shape("NVFP4 MMA", "grid overflow"))?;
        let prep =
            self.module
                .prepare_gemm_nvfp4_mma(LaunchConfig1D::new(grid, TILE_THREADS, 0))?;
        self.module.gemm_nvfp4_mma(
            stream,
            &prep,
            w.qs.buf(),
            w.e.buf(),
            w.scales.buf(),
            &a.q,
            &a.e,
            &a.scale,
            &route.cols,
            &route.tiles,
            &route.n_tiles,
            rows,
            (act.k / 64) as u32,
            slots,
            slot_div,
            rt,
            y,
        )?;
        Ok(())
    }
}
