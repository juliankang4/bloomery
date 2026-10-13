//! The link the tier page crosses: how the stage card, each tier card and
//! the host reach the page's bytes ([`super::tier::TierPage`]). The tier
//! protocol takes every device address it enqueues from one side's end
//! ([`LinkEnd`]) and every word the host touches from the [`Link`], so where
//! the bytes sit is this module's alone.
//!
//! A link is one of two kinds ([`Link::new`]), and the tier protocol's code
//! is the same over both.
//!
//! - The shared page ([`Link::shared`]): one host-mapped allocation
//!   ([`MappedHost`]) that the stage card's context allocates and every tier
//!   card's context reaches at the same device address, which a tier's end
//!   checks when it is made ([`Link::tier_end`]). Each word is one location
//!   that both cards and the host read and write.
//! - The loopback ([`Link::loopback`]): two pages, the stage card's and the
//!   tier cards', and a host pump thread between them ([`pump`]) that carries
//!   each counter as a generation, with the bytes it publishes, from the
//!   writer's page to the reader's. Only the writing card adds to a counter
//!   on the writer's page; the cards' operations are the shared page's.
//!
//! The host touches the page's flag words through the link alone: it reads a
//! word ([`Link::load`]), stores one ([`Link::store`]),
//! checks at rest and resets a counter ([`Link::at_rest`], [`Link::reset`]),
//! and releases a tier's counters, all of them in one call
//! ([`Link::release`]). On the shared page each word access is one load or
//! store, and a release one store a counter; on the loopback each acts on
//! the side of the counter it names, and a release on all of its counters
//! under one hold of the pump's lock.

mod pump;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use bloomery_levers::TierLink;
use cuda_core::{CudaContext, sys};
use threads::helper::{Placement, spawn_helper};

use crate::GpuError;
use crate::graph::{MappedHost, cu};
pub(super) use pump::{Crossing, Reader};

/// The tier page's bytes as every side reaches them: the stage card's page,
/// which every tier card reaches too on the shared page, and on the loopback
/// the tier cards' own page and the pump.
///
/// Field order is drop order: the pump stops before the pages go.
pub(super) struct Link {
    mirror: Option<Mirror>,
    page: Arc<MappedHost>,
}

/// The loopback's tier side: the tier cards' page and the pump thread that
/// carries the counters between it and the stage card's page.
struct Mirror {
    pump: Arc<Pump>,
    thread: Option<JoinHandle<()>>,
    /// A test's tier card that runs on once released ([`Link::run_on`]).
    #[cfg(test)]
    run_on: Option<(usize, u32)>,
}

/// What the pump thread and the host share.
struct Pump {
    stage: Arc<MappedHost>,
    tier: Arc<MappedHost>,
    quit: AtomicBool,
    relays: Mutex<pump::Relays>,
}

impl pump::Page for MappedHost {
    fn word(&self, off: usize) -> &AtomicU32 {
        self.atomic_u32(off)
            .expect("the tier page's layout keeps every word the link touches 4-aligned inside it")
    }
}

impl Pump {
    fn lock(&self) -> MutexGuard<'_, pump::Relays> {
        self.relays.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The pump thread's loop: a pass over every counter until the link
    /// drops, spinning while nothing moves.
    fn run(&self) {
        while !self.quit.load(Ordering::Acquire) {
            if !self.lock().pass(&*self.stage, &*self.tier) {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        self.pump.quit.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            // A pump that panicked has stopped; there is nothing to carry.
            let _ = t.join();
        }
    }
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
    /// The link of kind `kind` over `bytes` zeroed bytes of page: the shared
    /// page, allocated in `stage`, or the loopback, whose pump carries
    /// `crossings` ([`Link::loopback`]). A crossing the loopback refuses is
    /// named as `what`. Load-time only; `stage` is current on return.
    pub(super) fn new(
        kind: TierLink,
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        what: &'static str,
    ) -> Result<Link, GpuError> {
        match kind {
            TierLink::Shared => Link::shared(stage, bytes),
            TierLink::Loopback => Link::loopback(stage, tier, bytes, crossings, what),
        }
    }

    /// `bytes` zeroed bytes of shared page, allocated in the stage card's
    /// context `ctx`. Load-time only.
    fn shared(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Link, GpuError> {
        Ok(Link {
            mirror: None,
            page: Arc::new(MappedHost::new(ctx, bytes, "cuMemHostAlloc (tier page)")?),
        })
    }

    /// The loopback: `bytes` zeroed bytes of page for the stage card,
    /// allocated in `stage`, as many for the tier cards, allocated in `tier`,
    /// and the pump thread that carries `crossings` between them. A span or
    /// a word of a crossing not 4-aligned inside the page is refused by name
    /// as `what`. Load-time only; `stage` is current on return.
    fn loopback(
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        what: &'static str,
    ) -> Result<Link, GpuError> {
        let fits = |r: &std::ops::Range<usize>| {
            r.start.is_multiple_of(4)
                && r.end.is_multiple_of(4)
                && r.start <= r.end
                && r.end <= bytes
        };
        if let Some(c) = crossings.iter().find(|c| {
            !fits(&(c.counter..c.counter + 4))
                || c.prog.is_some_and(|p| !fits(&(p..p + 4)))
                || !c.spans.iter().all(fits)
        }) {
            return Err(GpuError::shape(
                what,
                format!("the crossing {c:?} is not 4-aligned inside a page of {bytes} bytes"),
            ));
        }
        let tier = Arc::new(MappedHost::new(
            tier,
            bytes,
            "cuMemHostAlloc (tier page, tier side)",
        )?);
        let page = Arc::new(MappedHost::new(stage, bytes, "cuMemHostAlloc (tier page)")?);
        let relays = pump::Relays::new(crossings, &*page, &*tier);
        let pump = Arc::new(Pump {
            stage: Arc::clone(&page),
            tier,
            quit: AtomicBool::new(false),
            relays: Mutex::new(relays),
        });
        // The load thread may be pinned to one cpu, which the pump's spin
        // would share; the helper floats it.
        let (thread, _) = {
            let pump = Arc::clone(&pump);
            spawn_helper("tier-link-pump", Placement::Float, move || pump.run())
                .map_err(|e| GpuError::plan(what, e))?
        };
        Ok(Link {
            mirror: Some(Mirror {
                pump,
                thread: Some(thread),
                #[cfg(test)]
                run_on: None,
            }),
            page,
        })
    }

    /// The page the tier cards reach: the shared page, or the loopback's
    /// tier side.
    fn tier_page(&self) -> &MappedHost {
        match &self.mirror {
            None => &self.page,
            Some(m) => &m.pump.tier,
        }
    }

    /// The stage card's end.
    pub(super) fn stage_end(&self) -> LinkEnd {
        LinkEnd {
            dev: self.page.dev_at(0),
        }
    }

    /// The end of the tier card named `name` in its context `ctx`, which is
    /// current on return: the tier cards' page must sit at one address in
    /// every context that reaches it (unified addressing), the stage card's
    /// included on the shared page, else the end is refused by name as
    /// `what`.
    pub(super) fn tier_end(
        &self,
        ctx: &CudaContext,
        what: &'static str,
        name: &str,
    ) -> Result<LinkEnd, GpuError> {
        ctx.bind_to_thread()?;
        let mut dev: sys::CUdeviceptr = 0;
        let page = self.tier_page();
        // SAFETY: the tier's context is current on this thread (bound above),
        // `dev` is a live local the call writes, and the page's first byte is
        // the start of its live mapped allocation; the flags must be 0.
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, page.host_at(0).cast(), 0) };
        cu(rc, "cuMemHostGetDevicePointer_v2 (tier page, tier context)")?;
        if dev != page.dev_at(0) {
            let made = if self.mirror.is_none() {
                "the stage card's context"
            } else {
                "the context that made it"
            };
            return Err(GpuError::shape(
                what,
                format!(
                    "the tier page is at {:#x} in {made} and at {dev:#x} in the tier {name}'s: the \
                     tier's copies and the stage's waits need one address",
                    page.dev_at(0),
                ),
            ));
        }
        Ok(LinkEnd { dev })
    }

    /// The u32 at byte `off` of the stage card's page, which the caller's
    /// layout keeps 4-aligned and inside the page.
    fn word(&self, off: usize) -> &AtomicU32 {
        pump::Page::word(&*self.page, off)
    }

    /// The word at byte `off` as the stage card's side holds it.
    pub(super) fn load(&self, off: usize) -> u32 {
        self.word(off).load(Ordering::Acquire)
    }

    /// Store `v` into the word at byte `off` on every page, the tier cards'
    /// first: a word the pump carries goes from there to the stage card's.
    pub(super) fn store(&self, off: usize, v: u32) {
        if let Some(m) = &self.mirror {
            pump::Page::word(&*m.pump.tier, off).store(v, Ordering::Release);
        }
        self.word(off).store(v, Ordering::Release);
    }

    /// Release every wait on the counters at bytes `offs`, a tier's every
    /// counter, until their resets: the counter each reader waits on goes to
    /// `v`, far past anything a step subtracts. A tier's counters go in one
    /// call: on the loopback it holds the pump's lock across all of them, so
    /// no pass falls between two of them while the released card runs on and
    /// steps the counters it writes.
    pub(super) fn release(&self, offs: &[usize], v: u32) {
        match &self.mirror {
            None => offs.iter().for_each(|&off| self.store(off, v)),
            Some(m) => {
                m.pump
                    .lock()
                    .release(offs, v, &*m.pump.stage, &*m.pump.tier);
                #[cfg(test)]
                m.after_release();
            }
        }
    }

    /// Whether the counter at byte `off` holds what a drained step leaves:
    /// no go or signal pending, and on the loopback every generation
    /// carried and no relay broken.
    pub(super) fn at_rest(&self, off: usize) -> bool {
        match &self.mirror {
            None => self.load(off) == 0,
            Some(m) => m.pump.lock().at_rest(off, &*m.pump.stage, &*m.pump.tier),
        }
    }

    /// The counter at byte `off` back to rest once every stream that reaches
    /// it has drained: 0 on the shared page; on the loopback the reader's
    /// counter at 0, the writer's generation taken as carried, the progress
    /// word taken from the tier cards' page, and that counter carried again.
    pub(super) fn reset(&self, off: usize) {
        match &self.mirror {
            None => self.store(off, 0),
            Some(m) => m.pump.lock().reset(off, &*m.pump.stage, &*m.pump.tier),
        }
    }

    /// Why the loopback stopped carrying for good: a relay that stepped by
    /// neither 0 nor 1, or a pump thread that ended. `None` on the shared
    /// page.
    pub(super) fn broken(&self) -> Option<String> {
        let m = self.mirror.as_ref()?;
        if let Some(b) = m.pump.lock().broken() {
            return Some(b.to_string());
        }
        m.thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
            .then(|| "the link's pump thread has ended".to_string())
    }
}

#[cfg(test)]
impl Link {
    /// A tier card that runs on once released, for a test over the
    /// loopback: after each release the counter at byte `off` steps by `by`
    /// on the tier cards' page, and the pump makes one pass before the host
    /// takes its lock again.
    pub(super) fn run_on(&mut self, off: usize, by: u32) {
        self.mirror.as_mut().expect("a loopback link").run_on = Some((off, by));
    }
}

#[cfg(test)]
impl Mirror {
    /// [`Link::run_on`]'s step, after a release has dropped the pump's lock.
    fn after_release(&self) {
        if let Some((off, by)) = self.run_on {
            pump::Page::word(&*self.pump.tier, off).fetch_add(by, Ordering::Release);
            self.pump.lock().pass(&*self.pump.stage, &*self.pump.tier);
        }
    }
}
