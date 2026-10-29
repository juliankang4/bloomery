//! Device buffers whose address range is reserved whole and whose pages are
//! backed only as far as the rows in use reach, through the driver's virtual
//! memory management (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`). A
//! buffer keeps its length, its layout and its address, so every launch and
//! every capture over it stays as it is; the card holds only the pages the
//! rows a sequence reached lie on.
//!
//! A [`Room`] backs a page with its own memory, zeroed on its stream, the
//! first time a fit reaches it ([`Room::fit`]), and at a sequence's end gives
//! back every page no row of that sequence reached ([`Room::settle`]). A page
//! with no memory of its own maps the device's one spare page ([`spare`]):
//! the driver checks a copy's whole extent, the bytes between its rows too
//! (a 2-D copy of a store's planes), so no page of the range is left
//! unmapped. A load of `ctx` positions a slot so holds the rows its requests
//! use, rounded up to the device's allocation granularity, and a slot that
//! serves requests of one length takes no memory after its first. The plan
//! reserves every buffer's page rounding on every resident slot and the
//! device's spare page before choosing experts or context.
//!
//! Every writer calls `fit` first, outside capture, for all positions it
//! can write. An unbacked page aliases the spare: a missed fit can corrupt
//! another room without a device fault.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use cuda_core::sys::CUdevice;
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::GpuError;

const WHAT: &str = "room";

/// Rows of one buffer in a room: `rows` rows of `row` bytes from byte `at`
/// of the buffer, one row for every `per` positions (a pooled key's row
/// covers `per` positions, every other row one).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub(crate) at: usize,
    pub(crate) row: usize,
    pub(crate) per: usize,
    pub(crate) rows: usize,
}

impl Run {
    /// The bytes `[at, end)` the rows of the first `positions` positions take.
    fn span(&self, positions: usize) -> (usize, usize) {
        let rows = positions.div_ceil(self.per).min(self.rows);
        (self.at, self.at + rows * self.row)
    }
}

/// A page of a room's range: its mapping, and the memory of its own a fit
/// gave it (none while it maps the spare). The mapping drops first
/// (unmapped), then the memory is released.
struct Page {
    _map: Mapping,
    own: Option<PhysicalAllocation>,
}

/// The page every page of `device`'s rooms with no memory of its own maps,
/// and whether it was made just now (its bytes not set yet). One a device
/// while any room holds it. Its bytes are no row's: nothing reads or
/// writes them but an access past the rows in use.
fn spare(device: CUdevice, page: usize) -> Result<(Arc<PhysicalAllocation>, bool), GpuError> {
    static SPARES: Mutex<Vec<(CUdevice, Weak<PhysicalAllocation>)>> = Mutex::new(Vec::new());
    let mut spares = SPARES.lock().unwrap_or_else(PoisonError::into_inner);
    spares.retain(|(_, s)| s.strong_count() > 0);
    if let Some(s) = spares
        .iter()
        .find(|(d, _)| *d == device)
        .and_then(|(_, s)| s.upgrade())
    {
        return Ok((s, false));
    }
    let s = Arc::new(PhysicalAllocation::new(device, page).map_err(driver("cuMemCreate"))?);
    spares.push((device, Arc::downgrade(&s)));
    Ok((s, true))
}

fn driver(op: &'static str) -> impl Fn(DriverError) -> GpuError {
    move |source| GpuError::Driver {
        op: Some(op),
        source,
    }
}

/// The reserved range of `N` buffers of one length, each at a page boundary,
/// and its pages. The pages drop before the reservation (field order), as the
/// driver wants every mapping gone before its range is freed.
pub(crate) struct Room {
    /// Every page of the range; none only after a remap failed.
    pages: Vec<Option<Page>>,
    /// Every buffer's runs, at their offsets in the range.
    runs: Vec<Run>,
    /// The positions whose rows lie on pages with memory of their own: a
    /// fit up to it backs nothing new.
    fitted: usize,
    /// The most positions a fit asked since the last settle.
    reach: usize,
    /// The pages with memory of their own.
    mapped: usize,
    page: usize,
    spare: Arc<PhysicalAllocation>,
    stream: Arc<CudaStream>,
    reserve: VirtualReservation,
}

impl Room {
    /// The pages any run's rows of the first `positions` positions lie on.
    fn wanted(&self, positions: usize) -> Vec<bool> {
        let mut want = vec![false; self.pages.len()];
        for r in &self.runs {
            let (at, end) = r.span(positions);
            if end > at {
                want[at / self.page..=(end - 1) / self.page].fill(true);
            }
        }
        want
    }

    /// Page `i` mapped read-write for the room's device to `own`, or to the
    /// spare without it, whatever mapped it before unmapped first.
    fn back(&mut self, i: usize, own: Option<PhysicalAllocation>) -> Result<(), GpuError> {
        if self.pages[i].take().is_some_and(|p| p.own.is_some()) {
            self.mapped -= 1;
        }
        let va = self.reserve.base() + (i * self.page) as u64;
        let map = Mapping::new(va, self.page, own.as_ref().unwrap_or(&self.spare), 0)
            .map_err(driver("cuMemMap"))?;
        vmm::set_access(va, self.page, &[self.stream.context().cu_device()])
            .map_err(driver("cuMemSetAccess"))?;
        self.mapped += usize::from(own.is_some());
        self.pages[i] = Some(Page { _map: map, own });
        Ok(())
    }

    /// Page `i`'s bytes zeroed on the room's stream, in order before any
    /// launch enqueued after.
    fn zero(&self, i: usize) -> Result<(), GpuError> {
        let va = self.reserve.base() + (i * self.page) as u64;
        // SAFETY: `va` is the start of a page of this room's reservation,
        // `page` bytes long, mapped read-write for this device; the room
        // keeps a mapping there until it drops, after the device finished.
        unsafe {
            cuda_core::simt::memory::memset_d8_async(va, 0, self.page, self.stream.cu_stream())
        }
        .map_err(driver("cuMemsetD8Async"))
    }

    /// The most positions whose rows lie on the pages the rows of the first
    /// `positions` positions take, in every run.
    fn covered(&self, positions: usize) -> usize {
        self.runs
            .iter()
            .map(|r| {
                let rows = positions.div_ceil(r.per).min(r.rows);
                if rows == r.rows {
                    return usize::MAX;
                }
                if rows == 0 {
                    return 0;
                }
                let end = (r.at + rows * r.row).div_ceil(self.page) * self.page;
                ((end - r.at) / r.row).saturating_mul(r.per)
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    /// Every page the rows of the first `positions` positions lie on backed
    /// by its own memory: those not yet are given it, zeroed. A fit within
    /// the backed pages returns at once (a decode step's, but on a page's
    /// first row). Never inside a capture.
    pub(crate) fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.reach = self.reach.max(positions);
        if positions <= self.fitted {
            return Ok(());
        }
        let mut bound = false;
        for r in 0..self.runs.len() {
            let (_, from) = self.runs[r].span(self.fitted);
            let (_, end) = self.runs[r].span(positions);
            if end == from {
                continue;
            }
            for i in from / self.page..end.div_ceil(self.page) {
                if self.pages[i].as_ref().is_some_and(|p| p.own.is_some()) {
                    continue;
                }
                let ctx = self.stream.context();
                if !bound {
                    ctx.bind_to_thread()?;
                    bound = true;
                }
                let own = PhysicalAllocation::new(ctx.cu_device(), self.page)
                    .map_err(driver("cuMemCreate"))?;
                self.back(i, Some(own))?;
                self.zero(i)?;
            }
        }
        self.fitted = self.covered(positions);
        Ok(())
    }

    /// The end of a sequence: every page none of the rows of the positions
    /// it reached lies on given back to the spare, after the device has
    /// finished every call that could read one, and the next sequence's
    /// reach counted from none.
    pub(crate) fn settle(&mut self) -> Result<(), GpuError> {
        let give: Vec<usize> = (self.wanted(self.reach).into_iter().enumerate())
            .filter(|&(i, keep)| !keep && self.pages[i].as_ref().is_some_and(|p| p.own.is_some()))
            .map(|(i, _)| i)
            .collect();
        if !give.is_empty() {
            crate::capsync::ctx_sync(self.stream.context(), "room settle")?;
            self.stream.context().bind_to_thread()?;
            for i in give {
                self.back(i, None)?;
            }
        }
        self.fitted = self.fitted.min(self.covered(self.reach));
        self.reach = 0;
        Ok(())
    }

    /// Device bytes the pages with memory of their own hold.
    pub(crate) fn mapped(&self) -> usize {
        self.mapped * self.page
    }

    /// Device bytes every page of the range holds once backed: what a fill
    /// takes.
    pub(crate) fn whole(&self) -> usize {
        self.pages.len() * self.page
    }
}

impl Drop for Room {
    fn drop(&mut self) {
        // Pages unmap after this body; keep a failed drain on the context.
        self.stream
            .context()
            .record_err(crate::capsync::ctx_sync_in_drop(
                self.stream.context(),
                "room drop",
            ));
    }
}

/// What a room's buffers are held as: every buffer taken apart without a
/// free when the holder drops, since its memory is the room's.
pub(crate) trait Views {
    fn dissolve(self);
}

impl<T> Views for DeviceBuffer<T> {
    fn dissolve(self) {
        drop(self.into_raw_parts());
    }
}

impl<T, const N: usize> Views for [DeviceBuffer<T>; N] {
    fn dissolve(self) {
        self.into_iter().for_each(Views::dissolve);
    }
}

/// Buffers over one or more [`Room`]s and the rooms: the buffers read and
/// write as any other (`Deref`), the rooms back their pages
/// ([`Roomed::fit`]).
pub(crate) struct Roomed<V: Views> {
    views: ManuallyDrop<V>,
    rooms: Vec<Room>,
}

impl<V: Views> Roomed<V> {
    /// `N` buffers of `len` elements in one reserve on `stream`'s device,
    /// each at a page boundary, built into `V` by `build`; `runs` are the
    /// rows of one buffer, which every buffer has at its own offset. Every
    /// page maps the spare: the rows a fit reaches get memory of their own,
    /// zeroed ([`Room::fit`]). Refused by name for a run past a buffer.
    pub(crate) fn new<T, const N: usize>(
        stream: Arc<CudaStream>,
        len: usize,
        runs: &[Run],
        build: impl FnOnce([DeviceBuffer<T>; N]) -> V,
    ) -> Result<Roomed<V>, GpuError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| GpuError::shape(WHAT, format!("{len} elements pass usize")))?;
        if let Some(r) = runs.iter().find(|r| r.at + r.rows * r.row > bytes) {
            return Err(GpuError::shape(
                WHAT,
                format!("{r:?} past a buffer of {bytes} bytes"),
            ));
        }
        let ctx = Arc::clone(stream.context());
        ctx.bind_to_thread()?;
        let page = crate::allocation_granularity(ctx.ordinal())?;
        let stride = vmm::align_size(bytes.max(1), page);
        let reserve =
            VirtualReservation::new(stride * N, page).map_err(driver("cuMemAddressReserve"))?;
        let (spare, fresh) = spare(ctx.cu_device(), page)?;
        let base = reserve.base();
        let mut room = Room {
            pages: (0..N * stride / page).map(|_| None).collect(),
            runs: (0..N)
                .flat_map(|i| {
                    runs.iter().map(move |r| Run {
                        at: r.at + i * stride,
                        ..*r
                    })
                })
                .collect(),
            fitted: 0,
            reach: 0,
            mapped: 0,
            page,
            spare,
            stream,
            reserve,
        };
        for i in 0..room.pages.len() {
            room.back(i, None)?;
        }
        if fresh {
            room.zero(0)?;
        }
        // The views last: an error before them returns with no buffer over
        // the range, so none frees an address in it.
        let views = std::array::from_fn(|i| {
            // SAFETY: the buffer lies in this room's reservation, which the
            // `Roomed` below owns and outlives it: `Roomed::drop` takes every
            // buffer apart without a free (`Views::dissolve`) before the room
            // drops, so no `cuMemFree` ever reaches these addresses.
            unsafe {
                DeviceBuffer::from_raw_parts(base + (i * stride) as u64, len, Arc::clone(&ctx))
            }
        });
        Ok(Roomed {
            views: ManuallyDrop::new(build(views)),
            rooms: vec![room],
        })
    }

    /// The same buffers and rooms held as `W`, built by `build`.
    pub(crate) fn map<W: Views>(self, build: impl FnOnce(V) -> W) -> Roomed<W> {
        let (views, rooms) = self.into_parts();
        Roomed {
            views: ManuallyDrop::new(build(views)),
            rooms,
        }
    }

    /// The buffers of `a` and `b` built into one `V` by `build`, over both
    /// rooms: buffers of two element types or lengths in one holder.
    pub(crate) fn join<A: Views, B: Views>(
        a: Roomed<A>,
        b: Roomed<B>,
        build: impl FnOnce(A, B) -> V,
    ) -> Roomed<V> {
        let (a, mut rooms) = a.into_parts();
        let (b, more) = b.into_parts();
        rooms.extend(more);
        Roomed {
            views: ManuallyDrop::new(build(a, b)),
            rooms,
        }
    }

    /// The views and the rooms, the holder gone without dissolving them.
    fn into_parts(mut self) -> (V, Vec<Room>) {
        let rooms = std::mem::take(&mut self.rooms);
        // SAFETY: the views are taken once, here; the holder is forgotten
        // below, so its `Drop` never takes them again.
        let views = unsafe { ManuallyDrop::take(&mut self.views) };
        std::mem::forget(self);
        (views, rooms)
    }

    /// The rooms the buffers lie in.
    pub(crate) fn rooms_mut(&mut self) -> &mut [Room] {
        &mut self.rooms
    }

    /// Every page the rows of the first `positions` positions lie on backed
    /// by its own memory, zeroed ([`Room::fit`]). Never inside a capture.
    pub(crate) fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.rooms.iter_mut().try_for_each(|r| r.fit(positions))
    }

    /// Every row backed: what a read or write of the whole buffers takes.
    pub(crate) fn fill(&mut self) -> Result<(), GpuError> {
        self.fit(usize::MAX)
    }

    /// A sequence's end: the memory of the pages past the rows it reached
    /// given back ([`Room::settle`]).
    pub(crate) fn settle(&mut self) -> Result<(), GpuError> {
        self.rooms.iter_mut().try_for_each(Room::settle)
    }

    /// Device bytes the pages with memory of their own hold.
    pub(crate) fn mapped(&self) -> usize {
        self.rooms.iter().map(Room::mapped).sum()
    }

    /// Device bytes every page holds once backed ([`Room::whole`]).
    pub(crate) fn whole(&self) -> usize {
        self.rooms.iter().map(Room::whole).sum()
    }
}

impl<V: Views> Deref for Roomed<V> {
    type Target = V;
    fn deref(&self) -> &V {
        &self.views
    }
}

impl<V: Views> DerefMut for Roomed<V> {
    fn deref_mut(&mut self) -> &mut V {
        &mut self.views
    }
}

impl<V: Views> Drop for Roomed<V> {
    fn drop(&mut self) {
        // SAFETY: the views are taken once, here, and never touched again;
        // the rooms drop after this body, unmapping what they addressed.
        unsafe { ManuallyDrop::take(&mut self.views) }.dissolve();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Pitched, Window, WindowMut, dtoh_2d, htod_2d};

    /// A room backs only the pages a fit reaches, zeroed, at the addresses
    /// reserved for the whole buffers: two buffers of two heads' rows, four
    /// rows a page and eighteen rows a head, put the second head half a
    /// page past a boundary: fits of one, four and five reach different
    /// page boundaries in the two heads;
    /// values written to a backed row read back and a page just backed
    /// reads zero; a settle keeps every page the reach needs and gives the
    /// rest back at the next one; a 2-D copy of each head's first row,
    /// whose extent crosses the unbacked pages between, writes and reads;
    /// a fill backs every page; a run past its buffer is refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_room_backs_the_rows_a_fit_reaches() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let page = vmm::allocation_granularity(ctx.cu_device()).expect("the granularity");
        let (row, ctx_rows) = (page / 4, 18);
        let half = row / size_of::<u16>();
        let runs: Vec<Run> = (0..2)
            .map(|h| Run {
                at: h * ctx_rows * row,
                row,
                per: 1,
                rows: ctx_rows,
            })
            .collect();
        let len = 2 * ctx_rows * half;
        let mut r: Roomed<[DeviceBuffer<u16>; 2]> =
            Roomed::new(Arc::clone(&stream), len, &runs, |b| b).expect("a room");
        let at = (r[0].cu_deviceptr(), r[1].cu_deviceptr());
        assert_eq!(r.mapped(), 0, "no page before a fit");
        assert_eq!(r.whole(), 18 * page, "every page of both buffers");
        for (positions, pages) in [(1, 4), (4, 6), (5, 8)] {
            r.fit(positions).expect("a fit");
            assert_eq!(r.mapped(), pages * page, "a fit of {positions} positions");
        }
        let head1_row4 = (ctx_rows + 4) * row;
        WindowMut::<u16>::of_mut(&mut r[0], head1_row4, half)
            .expect("a backed row")
            .copy_from_host(&stream, &vec![7u16; half])
            .expect("a write");
        let read = |b: &DeviceBuffer<u16>, byte: usize| {
            Window::<u16>::of(b, byte, half)
                .expect("a backed row")
                .to_host_vec(&stream)
                .expect("a read")
        };
        assert_eq!(read(&r[0], head1_row4), vec![7u16; half]);
        assert_eq!(
            read(&r[1], 4 * row),
            vec![0u16; half],
            "a page just mapped reads zero"
        );
        r.settle().expect("a settle");
        assert_eq!(
            r.mapped(),
            8 * page,
            "the reach of five keeps both pages of each head"
        );
        r.fit(1).expect("a fit");
        r.settle().expect("a settle");
        assert_eq!(r.mapped(), 4 * page, "a reach of one keeps the first pages");
        let heads = Pitched {
            width: row,
            rows: 2,
            dev_pitch: ctx_rows * row,
            host_pitch: row,
        };
        let put = vec![3u16; 2 * half];
        let mut got = vec![0u16; 2 * half];
        // SAFETY: `put` holds the copy's `2 * row` bytes and is neither
        // written nor freed before the stream's synchronize below.
        unsafe { htod_2d(&stream, &mut r[1], 0, put.as_ptr().cast(), 2 * row, heads) }
            .expect("a 2-D write of each head's first row over the unbacked pages between");
        // SAFETY: `got` holds the copy's `2 * row` bytes and is neither read
        // nor freed before the stream's synchronize below.
        unsafe { dtoh_2d(&stream, &r[1], 0, got.as_mut_ptr().cast(), 2 * row, heads) }
            .expect("a 2-D read of each head's first row over the unbacked pages between");
        stream.synchronize().expect("the copies");
        assert_eq!(got, put);
        r.fill().expect("a fill");
        assert_eq!(r.mapped(), r.whole(), "a fill backs every page");
        assert_eq!(
            read(&r[0], head1_row4),
            vec![0u16; half],
            "a page given back and backed again reads zero"
        );
        assert_eq!(
            (r[0].cu_deviceptr(), r[1].cu_deviceptr()),
            at,
            "the addresses stay"
        );
        let past = [Run {
            rows: ctx_rows + 1,
            ..runs[1]
        }];
        let Err(e) = Roomed::<[DeviceBuffer<u16>; 2]>::new(Arc::clone(&stream), len, &past, |b| b)
        else {
            panic!("a run past its buffer is refused");
        };
        assert!(e.to_string().contains("past a buffer"), "{e}");
        let pooled = [Run {
            at: 0,
            row: page,
            per: 4,
            rows: 4,
        }];
        let mut p: Roomed<[DeviceBuffer<u16>; 1]> =
            Roomed::new(Arc::clone(&stream), 4 * page / 2, &pooled, |b| b).expect("a room");
        for (positions, pages) in [(1, 1), (4, 1), (5, 2)] {
            p.fit(positions).expect("a fit");
            assert_eq!(
                p.mapped(),
                pages * page,
                "a pooled fit of {positions} positions"
            );
        }
        stream.synchronize().expect("the room's work finished");
        let drain_error = DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_UNKNOWN);
        ctx.record_err::<()>(Err(drain_error));
        drop(p);
        assert_eq!(
            ctx.check_err(),
            Err(drain_error),
            "a room's failed drain stays on its retained context"
        );
    }
}
