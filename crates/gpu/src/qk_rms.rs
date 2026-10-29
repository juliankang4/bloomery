//! RMS norm of the query and key heads, with no rope, and the f16 append of
//! the unrotated key and of the value.
//!
//! The same norm as `head_norm_neox_append` at [`HEAD`]: each value squared
//! in f32, the two squares added in f64, the warp by the xor butterfly, the
//! two warps added with warp 0 first, the mean `(sum / HEAD) as f32`, the
//! scale from `rms_scale_of_mean_checked`. The normalized value is written
//! back. Nothing reads a rope table and nothing calls `neox_pair`.
//!
//! The planes are head-major, `[n_kv][ctx][HEAD]` u16: key head `h` of the
//! token at position `p` is row `h·ctx + p`. A position at or past `ctx`
//! raises [`FaultSite::CachePos`], writes NaN over the head and appends
//! nothing.
//! A non-finite cache source or f16 overflow raises [`FaultSite::CacheValue`]
//! and marks that cache value NaN.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::cache_f16_pair;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values per head. Thread `t` owns values `t` and `t + HEAD/2`.
pub const HEAD: usize = 128;
const THREADS: usize = HEAD / 2;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS == 64);

#[cuda_module]
mod qk_rms_kernels {
    use super::*;

    /// Norm and append of `m` tokens. Block `b = t·(n_head + n_kv) + h` is
    /// token `t`, query head `h` when `h < n_head` and otherwise key head
    /// `h - n_head`. No rope table is an argument.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn qk_rms_append(
        gq: &[f32],
        gk: &[f32],
        pos: &[u32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let t = b / heads;
        let h = b - t * heads;
        let tid = thread::threadIdx_x() as usize;
        let is_q = h < n_head as usize;
        let kh = h.wrapping_sub(n_head as usize);
        let base = if is_q {
            (t * n_head as usize + h) * HEAD
        } else {
            (t * n_kv as usize + kh) * HEAD
        };
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;

        // SAFETY: base + tid + 64 is inside the head's buffer by the launch
        // contract. One thread per value pair.
        let (x0, x1) = unsafe {
            if is_q {
                (
                    *q.get_unchecked_mut(base + tid),
                    *q.get_unchecked_mut(base + tid + THREADS),
                )
            } else {
                (
                    *k.get_unchecked_mut(base + tid),
                    *k.get_unchecked_mut(base + tid + THREADS),
                )
            }
        };
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        // SAFETY: WSUM is this block's own shared allocation. Two slots, one
        // per warp, written before the barrier that publishes them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 2. One lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: both slots were written before the barrier above.
        let sum = unsafe { *ws.add(0) + *ws.add(1) };
        let mean = (sum / HEAD as f64) as f32;
        let scale = rms_scale_of_mean_checked(mean, eps, fault, tid == 0);

        if p >= ctx as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            // SAFETY: the positions read above. This thread owns them.
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(base + tid) = f32::NAN;
                    *q.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                } else {
                    *k.get_unchecked_mut(base + tid) = f32::NAN;
                    *k.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                }
            }
            return; // block-uniform: p is the token's
        }
        // SAFETY: tid + 64 < 128 <= the gain's length.
        let (g0, g1) = unsafe {
            let g = if is_q { gq } else { gk };
            (*g.get_unchecked(tid), *g.get_unchecked(tid + THREADS))
        };
        let n0 = (scale * g0) * x0;
        let n1 = (scale * g1) * x1;

        if is_q {
            // SAFETY: the positions read above. This thread owns them.
            unsafe {
                *q.get_unchecked_mut(base + tid) = n0;
                *q.get_unchecked_mut(base + tid + THREADS) = n1;
            }
            return;
        }
        // SAFETY: the positions read above. This thread owns them.
        unsafe {
            *k.get_unchecked_mut(base + tid) = n0;
            *k.get_unchecked_mut(base + tid + THREADS) = n1;
        }
        let row = (kh * ctx as usize + p) * HEAD;
        // SAFETY: kh < n_kv and p < ctx, so the row is inside both planes.
        // base + tid + 64 is inside v. One thread per pair. The tokens of
        // one launch hold distinct positions, so no two blocks write one row.
        unsafe {
            let (k0, k1) = cache_f16_pair(n0, n1, fault);
            let (v0, v1) = cache_f16_pair(
                *v.get_unchecked(base + tid),
                *v.get_unchecked(base + tid + THREADS),
                fault,
            );
            *cache_k.get_unchecked_mut(row + tid) = k0;
            *cache_k.get_unchecked_mut(row + tid + THREADS) = k1;
            *cache_v.get_unchecked_mut(row + tid) = v0;
            *cache_v.get_unchecked_mut(row + tid + THREADS) = v1;
        }
    }
}

/// Arguments of one no-rope norm and append. `eps` is the layer's RMS epsilon.
pub struct QkRmsArgs<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub gq: &'a DeviceBuffer<f32>,
    pub gk: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct QkRmsKernels {
    module: qk_rms_kernels::LoadedModule,
}

impl QkRmsKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<QkRmsKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(qk_rms_kernels, ctx)? };
        Ok(QkRmsKernels { module })
    }

    /// Enqueue the norm and append of `args.m` tokens: one block per
    /// (token, head). A position `>= ctx` raises [`FaultSite::CachePos`],
    /// leaves that token's heads NaN and is not appended. A zero count, a
    /// short buffer or `m > ctx` is refused by name before the launch.
    pub fn enqueue(&self, stream: &CudaStream, args: QkRmsArgs<'_>) -> Result<(), GpuError> {
        let what = "qk_rms::enqueue";
        let QkRmsArgs {
            q,
            k,
            v,
            gq,
            gk,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let plane = n_kv * ctx * HEAD;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), plane),
            ("cache_v", cache_v.len(), plane),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_qk_rms_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.qk_rms_append(
            stream, &prep, gq, gk, pos, v, eps, n_head, n_kv, ctx, m, fault, q, k, cache_k, cache_v,
        )?;
        Ok(())
    }
}
