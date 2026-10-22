//! FP4 tensor-core products over routed NVFP4 planes.

use super::{GemmAct32, GemmRoute};
use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, ptx_asm, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

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
        self.q.len() * 4 + self.e.len() * 4 + self.scale.len() * 4
    }
}

#[inline(always)]
fn e2m1(x: f32) -> u32 {
    let a = x.abs();
    let q = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    q | ((x.to_bits() >> 28) & 8)
}

#[inline(always)]
fn e4m3(x: f32) -> u32 {
    let out: u16;
    // SAFETY: conversion only reads and writes registers.
    unsafe {
        ptx_asm!("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;", out("=h") out, in("f") x);
    }
    u32::from(out) & 255
}

#[inline(always)]
fn compact(x: u32) -> u32 {
    (x & 15) | ((x >> 4) & 240) | ((x >> 8) & 3840) | ((x >> 12) & 61440)
}

#[inline(always)]
fn weight_word(q: &[u32], row: usize, steps: usize, step: usize, word: usize) -> u32 {
    let at = row * steps * 8 + step * 8 + (word / 2) * 2;
    let shift = 4 * (word & 1);
    compact(q[at] >> shift) | (compact(q[at + 1] >> shift) << 16)
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

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn quantize(
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
            if nonzero != 0 {
                max = max.max(d[col * 2 * steps + block]);
            }
            block += 32;
        }
        let mut delta = 16;
        while delta > 0 {
            max = max.max(warp::shuffle_xor_f32(max, delta));
            delta /= 2;
        }
        let gs = if max > 0.0 { max * (127.0 / 6.0) } else { 1.0 };
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
            let mut largest = 0.0f32;
            let mut i = 0;
            while i < 16 {
                let v = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * dx;
                largest = largest.max(v.abs());
                i += 1;
            }
            let scale = e4m3(largest / (6.0 * gs));
            let decoded = crate::mxfp4::ue4m3_half(scale) * 2.0 * gs;
            let inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
            let mut packed = [0u32; 2];
            i = 0;
            while i < 16 {
                let v = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * dx;
                packed[i / 8] |= e2m1(v * inv) << (4 * (i % 8));
                i += 1;
            }
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
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn product(
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
            while i < refused * 64 {
                let row = rt * 64 + i % 64;
                if row < rows {
                    // SAFETY: the route's refused slots are disjoint from live tiles; one thread owns this row.
                    unsafe {
                        *y.get_unchecked_mut(cols[first + i / 64] as usize * rows + row) = f32::NAN;
                    }
                }
                i += 128;
            }
        }
        if tile >= ntiles[0] as usize {
            return;
        }
        let expert = tiles[2 * tile] as usize;
        let (start, len) = super::super::route::tile_parts(tiles[2 * tile + 1]);
        let (start, len) = (start as usize, len as usize);
        let row = rt * 64 + 16 * (tid / 32);
        if row >= rows {
            return;
        }
        let g = lane / 4;
        let t = lane & 3;
        let nt = len.div_ceil(8);
        let mut acc = [0.0f32; 32];
        let mut step = 0;
        while step < steps {
            let r = expert * rows + row + g;
            let a = [
                weight_word(wq, r, steps, step, t),
                weight_word(wq, r + 8, steps, step, t),
                weight_word(wq, r, steps, step, t + 4),
                weight_word(wq, r + 8, steps, step, t + 4),
            ];
            let sa = we[(r + 8 * (t & 1)) * steps + step];
            #[unroll]
            for n in 0..8 {
                if n >= nt {
                    continue;
                }
                let j = if n * 8 + g < len { n * 8 + g } else { 0 };
                let col = cols[start + j] as usize / slot_div as usize;
                let b = [
                    aq[col * steps * 8 + step * 8 + t],
                    aq[col * steps * 8 + step * 8 + t + 4],
                ];
                let c = mma(
                    [acc[n * 4], acc[n * 4 + 1], acc[n * 4 + 2], acc[n * 4 + 3]],
                    a,
                    b,
                    sa,
                    ae[col * steps + step],
                );
                acc[n * 4] = c[0];
                acc[n * 4 + 1] = c[1];
                acc[n * 4 + 2] = c[2];
                acc[n * 4 + 3] = c[3];
            }
            step += 1;
        }
        #[unroll]
        for n in 0..8 {
            #[unroll]
            for j in 0..4 {
                let c = n * 8 + 2 * t + (j & 1);
                if c < len {
                    let slot = cols[start + c] as usize;
                    // SAFETY: each lane owns these two rows and two columns of a live route tile.
                    unsafe {
                        *y.get_unchecked_mut(slot * rows + row + g + 8 * (j / 2)) =
                            acc[n * 4 + j] * ws[expert] * ag[slot / slot_div as usize];
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
        if let Some(out) = act.native.as_mut() {
            let prep =
                self.module
                    .prepare_quantize(LaunchConfig1D::new(act.filled as u32, 32, 0))?;
            self.module.quantize(
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
        let rt = rows.div_ceil(64);
        let grid = max_tiles
            .checked_mul(rt)
            .ok_or_else(|| GpuError::shape("NVFP4 MMA", "grid overflow"))?;
        let prep = self
            .module
            .prepare_product(LaunchConfig1D::new(grid, 128, 0))?;
        self.module.product(
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
