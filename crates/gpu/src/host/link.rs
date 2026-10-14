//! The link the tier page crosses: how the stage card, each tier card and
//! the host reach the page's bytes ([`super::tier::TierPage`]). The tier
//! protocol takes every device address it enqueues from one side's end
//! ([`LinkEnd`]) and every word the host touches from the [`Link`], so where
//! the bytes sit is this module's alone.
//!
//! The link is the shared page: one host-mapped allocation ([`MappedHost`])
//! that the stage card's context allocates and every tier card's context
//! reaches at the same device address, which a tier's end checks when it is
//! made ([`Link::tier_end`]). Each word is then one location that both cards
//! and the host read and write.
//!
//! The host touches the page's flag words through the link alone: it reads a
//! word ([`Link::load`]), stores one ([`Link::store`]), and releases, checks
//! at rest and resets a counter ([`Link::release`], [`Link::at_rest`],
//! [`Link::reset`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cuda_core::{CudaContext, sys};

use crate::GpuError;
use crate::graph::{MappedHost, cu};

/// The tier page's bytes as every side reaches them: the shared page.
pub(super) struct Link {
    page: MappedHost,
}

/// One side's view of the page: the device address of each byte as that
/// side's launches, copies and stream memory operations take it.
#[derive(Clone, Copy, Debug)]
pub(super) struct LinkEnd {
    dev: sys::CUdeviceptr,
}

impl LinkEnd {
    /// The device address of byte `off` of the page, on this side.
    pub(super) fn dev(self, off: usize) -> sys::CUdeviceptr {
        self.dev + off as u64
    }
}

impl Link {
    /// `bytes` zeroed bytes of shared page, allocated in the stage card's
    /// context `ctx`. Load-time only.
    pub(super) fn shared(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Link, GpuError> {
        Ok(Link {
            page: MappedHost::new(ctx, bytes, "cuMemHostAlloc (tier page)")?,
        })
    }

    /// The stage card's end.
    pub(super) fn stage_end(&self) -> LinkEnd {
        LinkEnd {
            dev: self.page.dev_at(0),
        }
    }

    /// The end of the tier card named `name` in its context `ctx`, which is
    /// current on return: the page must sit at the stage card's address in
    /// that context too (unified addressing), else the end is refused by name
    /// as `what`.
    pub(super) fn tier_end(
        &self,
        ctx: &CudaContext,
        what: &'static str,
        name: &str,
    ) -> Result<LinkEnd, GpuError> {
        ctx.bind_to_thread()?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: the tier's context is current on this thread (bound above),
        // `dev` is a live local the call writes, and the page's first byte is
        // the start of its live mapped allocation; the flags must be 0.
        let rc =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, self.page.host_at(0).cast(), 0) };
        cu(rc, "cuMemHostGetDevicePointer_v2 (tier page, tier context)")?;
        if dev != self.page.dev_at(0) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the tier page is at {:#x} in the stage card's context and at {dev:#x} in the \
                     tier {name}'s: the tier's copies and the stage's waits need one address",
                    self.page.dev_at(0),
                ),
            ));
        }
        Ok(LinkEnd { dev })
    }

    /// The u32 at byte `off`, which the caller's layout keeps 4-aligned and
    /// inside the page.
    fn word(&self, off: usize) -> &AtomicU32 {
        self.page
            .atomic_u32(off)
            .expect("the tier page's layout keeps every flag word 4-aligned inside the page")
    }

    /// The word at byte `off` as the stage card's side holds it.
    pub(super) fn load(&self, off: usize) -> u32 {
        self.word(off).load(Ordering::Acquire)
    }

    /// Store `v` into the word at byte `off`.
    pub(super) fn store(&self, off: usize, v: u32) {
        self.word(off).store(v, Ordering::Release);
    }

    /// Release every wait on the counter at byte `off` for good: it goes to
    /// `v`, far past anything a step subtracts.
    pub(super) fn release(&self, off: usize, v: u32) {
        self.store(off, v);
    }

    /// Whether the counter at byte `off` holds what a drained step leaves:
    /// no go or signal pending.
    pub(super) fn at_rest(&self, off: usize) -> bool {
        self.load(off) == 0
    }

    /// The counter at byte `off` back to a fresh page's, once every stream
    /// that reaches it has drained.
    pub(super) fn reset(&self, off: usize) {
        self.store(off, 0);
    }
}
