//! Per-slot protection of quantized expert down inputs and restoration before routing.

use crate::fault::{FaultSink, FaultSite};
use crate::kquant::{Act, act::apply_checked};
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

/// Slot selection: resident `_sel` places, or the route's unique slot list and refused count.
pub(crate) struct Slots<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub counts: &'a DeviceBuffer<u32>,
    pub routed: bool,
    pub n_card: usize,
    pub len: usize,
}

#[cuda_module]
mod expert_input_kernels {
    use super::*;

    #[inline(always)]
    unsafe fn slot(
        ids: &[u32],
        counts: &[u32],
        routed: u32,
        n_card: u32,
        len: u32,
        i: usize,
    ) -> Option<usize> {
        if i >= len as usize {
            return None;
        }
        if routed != 0 {
            // SAFETY: the route owns two counters; its slot list has len entries.
            if i >= len as usize - unsafe { *counts.get_unchecked(1) } as usize {
                return None;
            }
            // SAFETY: i < len <= ids.len(). Route slots are unique and below len.
            Some(unsafe { *ids.get_unchecked(i) } as usize)
        } else {
            // SAFETY: i < len <= ids.len().
            (unsafe { *ids.get_unchecked(i) } < n_card).then_some(i)
        }
    }

    #[allow(clippy::too_many_arguments, reason = "flat device ABI")]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (
        ids.len() >= slots, counts.len() >= 2 * routed, h.len() >= slots * width,
        scales.len() >= slots, u.len() >= activate * slots * width
    ))]
    pub fn normalize(
        mut h: DisjointSlice<f32>,
        u: &[f32],
        mut scales: DisjointSlice<f32>,
        ids: &[u32],
        counts: &[u32],
        routed: u32,
        n_card: u32,
        slots: u32,
        width: u32,
        activate: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
    ) {
        let i = thread::index_1d().get() / 32;
        // SAFETY: the launch contract bounds ids and counts; route invariants bound slot ids.
        let Some(s) = (unsafe { slot(ids, counts, routed, n_card, slots, i) }) else {
            return;
        };
        let lane = warp::lane_id() as usize;
        let base = s * width as usize;
        let mut max = 0.0f32;
        let mut bad = false;
        let mut d = lane;
        while d < width as usize {
            // SAFETY: s < slots, d < width; each lane owns its strided cells.
            let mut v = unsafe { *h.get_unchecked_mut(base + d) };
            if activate != 0 {
                // SAFETY: u holds slots whole rows when activate is selected.
                v = apply_checked(act, limit, v, unsafe { *u.get_unchecked(base + d) }, fault);
                // SAFETY: this lane alone owns this cell.
                unsafe { *h.get_unchecked_mut(base + d) = v };
            }
            bad |= !v.is_finite();
            max = max.max(v.abs());
            d += 32;
        }
        let max = warp::reduce_max_f32(max);
        let bad = warp::ballot(bad) != 0;
        // Keep the pinned graph's log/multiply/ceil/clamp/multiply/exp order.
        let e = if bad {
            f32::NAN
        } else {
            (((max * (1.0 / 1024.0)).ln() * std::f32::consts::LOG2_E)
                .ceil()
                .clamp(0.0, 127.0)
                * std::f32::consts::LN_2)
                .exp()
        };
        if lane == 0 {
            // SAFETY: this warp uniquely owns slot s.
            unsafe { *scales.get_unchecked_mut(s) = e };
            if bad {
                fault.raise(FaultSite::QuantColumn);
            }
        }
        d = lane;
        while d < width as usize {
            // SAFETY: the same lane owns this cell, and s < slots.
            unsafe {
                let v = *h.get_unchecked_mut(base + d);
                *h.get_unchecked_mut(base + d) = v / e;
            }
            d += 32;
        }
    }

    #[allow(clippy::too_many_arguments, reason = "flat device ABI")]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (
        ids.len() >= slots, counts.len() >= 2 * routed, y.len() >= slots * width, scales.len() >= slots
    ))]
    pub fn restore(
        mut y: DisjointSlice<f32>,
        scales: &[f32],
        ids: &[u32],
        counts: &[u32],
        routed: u32,
        n_card: u32,
        slots: u32,
        width: u32,
    ) {
        let i = thread::index_1d().get() / 32;
        // SAFETY: launch bounds and route invariants are as in normalize.
        let Some(s) = (unsafe { slot(ids, counts, routed, n_card, slots, i) }) else {
            return;
        };
        // SAFETY: s < slots <= scales.len().
        let e = unsafe { *scales.get_unchecked(s) };
        let mut d = warp::lane_id() as usize;
        while d < width as usize {
            // SAFETY: each lane owns its cells of the unique selected slot.
            unsafe {
                let v = *y.get_unchecked_mut(s * width as usize + d);
                *y.get_unchecked_mut(s * width as usize + d) = v * e;
            }
            d += 32;
        }
    }
}

pub(crate) struct ExpertInputKernels {
    module: expert_input_kernels::LoadedModule,
}

impl ExpertInputKernels {
    pub fn load(gpu: &Gpu) -> Result<Self, GpuError> {
        // SAFETY: this crate owns the embedded module and enforces its launch contracts.
        let module = unsafe { crate::shared_module!(expert_input_kernels, gpu.context())? };
        Ok(Self { module })
    }

    pub fn normalize(
        &self,
        gpu: &Gpu,
        (h, u, scales): (
            &mut DeviceBuffer<f32>,
            &DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
        ),
        slots: &Slots<'_>,
        width: usize,
        rule: Option<Act>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "ExpertInput::normalize";
        let (act, limit) = rule.unwrap_or(Act::SiluMul).code(what)?;
        let len = launch_u32(what, "slots", slots.len)?;
        let width = launch_u32(what, "width", width)?;
        let n_card = launch_u32(what, "n_card", slots.n_card)?;
        let prep = self
            .module
            .prepare_normalize(LaunchConfig1D::new(len, 32, 0))?;
        self.module.normalize(
            gpu.stream(),
            &prep,
            h,
            u,
            scales,
            slots.ids,
            slots.counts,
            u32::from(slots.routed),
            n_card,
            len,
            width,
            u32::from(rule.is_some()),
            act,
            limit,
            fault,
        )?;
        Ok(())
    }

    pub fn restore(
        &self,
        gpu: &Gpu,
        y: &mut DeviceBuffer<f32>,
        scales: &DeviceBuffer<f32>,
        slots: &Slots<'_>,
        width: usize,
    ) -> Result<(), GpuError> {
        let what = "ExpertInput::restore";
        let len = launch_u32(what, "slots", slots.len)?;
        let width = launch_u32(what, "width", width)?;
        let n_card = launch_u32(what, "n_card", slots.n_card)?;
        let prep = self
            .module
            .prepare_restore(LaunchConfig1D::new(len, 32, 0))?;
        self.module.restore(
            gpu.stream(),
            &prep,
            y,
            scales,
            slots.ids,
            slots.counts,
            u32::from(slots.routed),
            n_card,
            len,
            width,
        )?;
        Ok(())
    }
}

pub(crate) struct Protection {
    pub kernels: ExpertInputKernels,
    pub scales: DeviceBuffer<f32>,
}

impl Protection {
    pub fn new(gpu: &Gpu, slots: usize) -> Result<Self, GpuError> {
        Ok(Self {
            kernels: ExpertInputKernels::load(gpu)?,
            scales: DeviceBuffer::zeroed(gpu.stream(), slots)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.scales.num_bytes()
    }
}
