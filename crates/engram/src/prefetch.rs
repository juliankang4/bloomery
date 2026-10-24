//! A helper thread that turns one token's engram read into a memcpy — of an
//! [`crate::Engram`]'s sites, or of any other row-gathered tables
//! ([`crate::RowTables`], [`Prefetcher::over`]).
//!
//! The engine's step is its calling thread's serial time, so the question this
//! module answers is not "how long does the read take" but "how many
//! microseconds does the read steal from the step thread". Everything the read
//! costs — the `WILLNEED` syscall per distinct merged page run, the optional
//! `MADV_POPULATE_READ` pass, and the minor fault per row that consuming
//! through the mapping takes — is moved onto a helper. What crosses back is a buffer of plain bytes.
//!
//! The handoff is by ownership, not by sharing: two buffers circulate over a
//! pair of channels, so there is no lock on the step thread's path and no
//! `unsafe`. [`Prefetcher::submit`] gives the helper the next token's ids and
//! one free buffer; [`Prefetcher::wait`] takes the filled one back.
//!
//! A token may ask for fewer rows of a site than the prefetcher was sized for —
//! a cache in front of it hands over only its misses — and a token that asks
//! for none never reaches the helper: its `wait` returns at once.
//!
//! Steady state allocates nothing: the id vectors travel with the buffers and
//! are refilled in place, and the buffers are written once at construction so
//! that the first token's faults are the table's and not the allocator's.
//!
//! Where the helper runs is the owner's call ([`HelperOptions::cpu`]): a
//! helper that shares a core with a pool worker slows the worker's chunk by
//! more than the read it saves. [`caller_sibling`] names the one slot that is
//! free by construction for an owner that blocks on [`Prefetcher::wait`]: the
//! SMT sibling of the owner's own pinned core, idle while the owner waits.
//!
//! A floating helper may split a large token over more threads
//! ([`HelperOptions::workers`]): each takes one part of every site's rows and
//! advises, fills and copies them into its own part of the buffer, so the
//! advice syscalls and the faults of a prompt's thousands of rows run side by
//! side. A token too small to split, and every token of a pinned helper, is
//! read on the helper alone.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::{Engram, EngramError, Faults, RowTable, RowTables, faults_thread};

/// How [`Prefetcher::with_options`] starts its helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HelperOptions {
    pub mode: FillMode,
    /// The logical cpu the helper pins itself to; `None` leaves it floating.
    pub cpu: Option<usize>,
    /// Count per token the rows whose pages were all in the page cache before
    /// the advise ([`crate::Site::resident_rows`], one `mincore` per row).
    /// Off, the helper makes no such call.
    pub classify: bool,
    /// Threads a floating helper (`cpu` `None`) splits a token over, itself
    /// included, as many as give every part at least `PART_ROWS` rows of the
    /// token's largest site; 1, or a `cpu` to pin to, reads every token on
    /// the helper alone.
    pub workers: usize,
}

/// Rows of a token's largest site one part of a split read takes at least
/// ([`HelperOptions::workers`]): a decode step's few rows stay on the helper,
/// where a thread's start would cost more than its share of the read.
const PART_ROWS: usize = 1024;

impl HelperOptions {
    /// A floating helper that classifies nothing.
    #[must_use]
    pub fn of(mode: FillMode) -> HelperOptions {
        HelperOptions {
            mode,
            cpu: None,
            classify: false,
            workers: 1,
        }
    }
}

/// What the helper does between advising the rows and copying them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMode {
    /// Nothing: the copy itself faults the pages in. One minor fault per row,
    /// on the helper.
    Touch,
    /// [`crate::Site::populate`] over the same rows first, so the copy finds
    /// every page already mapped. Only the user-mode trap per fault goes away —
    /// the kernel's fault work is the same work, done in a batch.
    Populate,
}

/// One token's work and the storage it is done into. Both travel with the job
/// so that neither side allocates per token.
struct Job {
    /// Row ids per site, in site order. Refilled in place by `submit`.
    ids: Vec<Vec<u32>>,
    /// Sites' rows concatenated in site order.
    buf: Box<[u8]>,
    /// The bytes of `buf` this token's rows fill; zero once a fill failed.
    len: usize,
}

struct Filled {
    job: Job,
    err: Option<EngramError>,
    /// Travels with the buffer rather than in a shared cell. A shared cell
    /// would be read after `recv` returns, by which time the helper has already
    /// taken the next queued token and begun overwriting it — the pipelined
    /// caller keeps one submitted at all times, so that window is always open.
    times: HelperTimes,
}

/// What the helper spent on the token [`Prefetcher::wait`] just returned, and
/// the faults it has taken since it started. A token read in parts
/// ([`HelperOptions::workers`]) takes each time from its slowest part, and
/// counts every part's faults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HelperTimes {
    pub submit_ns: u64,
    pub fill_ns: u64,
    pub copy_ns: u64,
    pub major_faults: u64,
    pub minor_faults: u64,
    /// Rows the token asked for.
    pub rows: u64,
    /// Of those, the rows already in the page cache before the advise; zero
    /// unless the helper classifies ([`HelperOptions::classify`]).
    pub resident_rows: u64,
    /// The classification's own time, before `submit_ns`; zero unless the
    /// helper classifies.
    pub classify_ns: u64,
}

/// What the next [`Prefetcher::wait`] takes back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InFlight {
    /// A job on the helper.
    Helper,
    /// A token with no rows: nothing was sent.
    Empty,
}

/// A helper thread reading one token ahead for its owner.
pub struct Prefetcher {
    /// `None` only inside `drop`, which closes the channel to end the helper.
    jobs: Option<Sender<Job>>,
    done: Receiver<Filled>,
    /// Buffers not in flight. Two of them: one being filled, one being read.
    spare: Vec<Job>,
    /// The token `wait` last took back; what `filled` lends out.
    current: Option<Job>,
    /// The token submitted and not yet waited for, if any.
    in_flight: Option<InFlight>,
    /// Per site: the most rows one token may ask, and one row's bytes.
    max_rows: Vec<usize>,
    row_bytes: Vec<usize>,
    last: HelperTimes,
    /// The cpu the helper pinned itself to, if it was asked to and could.
    cpu: Option<usize>,
    handle: Option<JoinHandle<()>>,
}

impl Prefetcher {
    /// Start a helper for `engram`, sized for at most `max_rows_per_site[s]`
    /// rows of site `s` per token.
    ///
    /// `max_rows_per_site` must name every site. The helper holds an [`Arc`] of
    /// the table, so the mappings outlive it.
    pub fn new(
        engram: Arc<Engram>,
        max_rows_per_site: &[usize],
        mode: FillMode,
    ) -> Result<Prefetcher, EngramError> {
        Prefetcher::with_options(engram, max_rows_per_site, HelperOptions::of(mode))
    }

    /// [`Prefetcher::new`] with the helper's cpu and classification named.
    /// Returns once the helper has placed itself: pinned to the cpu asked
    /// for, or floating when none is asked or the kernel refuses the pin (a
    /// container may) — [`Prefetcher::pinned_cpu`] says which. A floating
    /// helper spawned from a caller pinned to one cpu widens its mask, so it
    /// does not share that cpu; a mask the kernel will not widen is an
    /// error.
    pub fn with_options(
        engram: Arc<Engram>,
        max_rows_per_site: &[usize],
        options: HelperOptions,
    ) -> Result<Prefetcher, EngramError> {
        Prefetcher::over(engram, max_rows_per_site, options)
    }

    /// [`Prefetcher::with_options`] over any row-gathered tables
    /// ([`RowTables`]: an [`Engram`]'s sites, or a table in its load's split,
    /// [`crate::rows::SplitTable`]), sized for at most
    /// `max_rows_per_site[t]` rows of table `t` per token.
    pub fn over(
        tables: Arc<dyn RowTables>,
        max_rows_per_site: &[usize],
        options: HelperOptions,
    ) -> Result<Prefetcher, EngramError> {
        if max_rows_per_site.len() != tables.count() {
            return Err(EngramError::Prefetch(
                "max_rows_per_site must have one entry per site",
            ));
        }
        if options.workers == 0 {
            return Err(EngramError::Prefetch("a helper of no workers"));
        }
        let row_bytes: Vec<usize> = (0..tables.count())
            .map(|i| table_of(tables.as_ref(), i).map(|t| t.row_bytes() as usize))
            .collect::<Result<_, _>>()?;
        let buf_len: usize = max_rows_per_site
            .iter()
            .zip(&row_bytes)
            .map(|(n, rb)| n * rb)
            .sum();

        // Written, not just allocated: `vec![0u8; n]` is lazily zero-mapped, and
        // the first write to it would fault on whichever thread got there first
        // and land in the token counts this module exists to separate.
        let spare: Vec<Job> = (0..2)
            .map(|_| Job {
                ids: max_rows_per_site
                    .iter()
                    .map(|&n| Vec::with_capacity(n))
                    .collect(),
                buf: vec![0xA5u8; buf_len].into_boxed_slice(),
                len: 0,
            })
            .collect();

        let (job_tx, job_rx) = channel::<Job>();
        let (done_tx, done_rx) = channel::<Filled>();
        let (handle, cpu) = threads::helper::spawn_helper(
            "engram-prefetch",
            options.cpu.map_or(
                threads::helper::Placement::Float,
                threads::helper::Placement::Pin,
            ),
            move || {
                helper(tables.as_ref(), options, &job_rx, &done_tx);
            },
        )
        .map_err(|e| match e {
            threads::helper::HelperError::Spawn(e) => EngramError::Io(e),
            e => EngramError::Prefetch(e.what()),
        })?;

        Ok(Prefetcher {
            jobs: Some(job_tx),
            done: done_rx,
            spare,
            current: None,
            in_flight: None,
            max_rows: max_rows_per_site.to_vec(),
            row_bytes,
            last: HelperTimes::default(),
            cpu,
            handle: Some(handle),
        })
    }

    /// The cpu the helper runs pinned to; `None` when it floats.
    #[must_use]
    pub fn pinned_cpu(&self) -> Option<usize> {
        self.cpu
    }

    /// Hand the helper one token's row ids and let it run.
    ///
    /// Returns as soon as the ids are copied into a free buffer — this is the
    /// step thread's whole share of the read. Each site may ask for any number
    /// of rows up to the prefetcher's size for it, and a token that asks for
    /// none is not sent at all. Two submits in a row without a
    /// [`Prefetcher::wait`] between them is a caller bug, not a runtime
    /// condition. `token_ids` yields one id list per site, in site order.
    pub fn submit<S: AsRef<[u32]>>(
        &mut self,
        token_ids: impl IntoIterator<Item = S>,
    ) -> Result<(), EngramError> {
        if self.in_flight.is_some() {
            return Err(EngramError::Prefetch(
                "a token is already in flight; wait() first",
            ));
        }
        let Some(mut job) = self.spare.pop() else {
            return Err(EngramError::Prefetch("no free buffer; wait() first"));
        };
        job.len = 0;
        let mut sites = 0usize;
        let mut refused = None;
        for (i, src) in token_ids.into_iter().enumerate() {
            let src = src.as_ref();
            sites = i + 1;
            // Checked before the copy: past its capacity an id vector would
            // reallocate, and the buffer would be too short for the rows.
            match (job.ids.get_mut(i), self.max_rows.get(i)) {
                (Some(dst), Some(&max)) if src.len() <= max => {
                    dst.clear();
                    dst.extend_from_slice(src);
                    job.len += src.len() * self.row_bytes[i];
                }
                (Some(_), Some(_)) => {
                    refused =
                        Some("token asks more rows of a site than the prefetcher was sized for");
                    break;
                }
                _ => {
                    refused = Some("token has a different site count");
                    break;
                }
            }
        }
        if refused.is_none() && sites != self.max_rows.len() {
            refused = Some("token has a different site count");
        }
        if let Some(why) = refused {
            self.spare.push(job);
            return Err(EngramError::Prefetch(why));
        }
        if job.len == 0 {
            self.spare.push(job);
            self.in_flight = Some(InFlight::Empty);
            return Ok(());
        }
        let Some(tx) = self.jobs.as_ref() else {
            self.spare.push(job);
            return Err(EngramError::Prefetch("helper is shutting down"));
        };
        tx.send(job)
            .map_err(|_| EngramError::Prefetch("helper thread exited"))?;
        self.in_flight = Some(InFlight::Helper);
        Ok(())
    }

    /// Block until the helper has finished the token submitted before this one,
    /// and take its buffer back.
    ///
    /// The bytes are then at [`Prefetcher::filled`] and the helper's own split
    /// at [`Prefetcher::last`]. Split from `filled` on purpose: the caller
    /// submits the next token while still holding these bytes, which a `wait`
    /// that returned the slice would forbid.
    ///
    /// A token that asked for no rows returns at once with nothing filled; the
    /// helper's times for it are zero and its fault counts carry over. A
    /// `wait` with nothing submitted is an error rather than a wait forever.
    pub fn wait(&mut self) -> Result<(), EngramError> {
        let Some(kind) = self.in_flight.take() else {
            return Err(EngramError::Prefetch("nothing in flight; submit() first"));
        };
        if kind == InFlight::Empty {
            if let Some(done) = self.current.take() {
                self.spare.push(done);
            }
            self.last = HelperTimes {
                submit_ns: 0,
                fill_ns: 0,
                copy_ns: 0,
                rows: 0,
                resident_rows: 0,
                classify_ns: 0,
                ..self.last
            };
            return Ok(());
        }
        let filled = self
            .done
            .recv()
            .map_err(|_| EngramError::Prefetch("helper thread exited"))?;
        self.last = filled.times;
        if let Some(done) = self.current.take() {
            self.spare.push(done);
        }
        let mut job = filled.job;
        if filled.err.is_some() {
            job.len = 0;
        }
        self.current = Some(job);
        match filled.err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The rows of the token [`Prefetcher::wait`] last took back: every site's
    /// rows concatenated in site order, `ids.len() * row_bytes()` per site.
    /// Empty before the first `wait`, after a token that asked for no rows, and
    /// after a fill that failed.
    pub fn filled(&self) -> &[u8] {
        self.current.as_ref().map_or(&[], |j| &j.buf[..j.len])
    }

    /// What the helper spent on that token.
    pub fn last(&self) -> HelperTimes {
        self.last
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Closing the channel is what ends the helper's receive loop; joining
        // after it is what keeps a dead round's helper from racing the next
        // one's on the same drive.
        self.jobs = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Advise, fill and copy one token at a time until the job channel closes, a
/// token of enough rows in parts ([`HelperOptions::workers`]).
///
/// Errors travel back with the buffer rather than ending the helper: the owner
/// is blocked in `wait` and has to be told, and the next token may well be
/// fine.
fn helper(
    tables: &dyn RowTables,
    options: HelperOptions,
    jobs: &Receiver<Job>,
    done: &Sender<Filled>,
) {
    let HelperOptions { mode, classify, .. } = options;
    // A pinned helper's part threads would inherit its one cpu.
    let workers = if options.cpu.is_some() {
        1
    } else {
        options.workers
    };
    let base = faults_thread();
    // Faults the parts' own threads took: the helper's own count none of them.
    let mut parts_faults = Faults::default();
    // The sizing checked every table at `over`, so each index here names one.
    let sites: Vec<&dyn RowTable> = (0..tables.count())
        .filter_map(|i| tables.table(i))
        .collect();
    while let Ok(mut job) = jobs.recv() {
        let mut err = None;
        let rows: u64 = job.ids.iter().map(|ids| ids.len() as u64).sum();

        // Before the advise: afterwards every row's folio is in the cache,
        // its read in flight, and would count as resident.
        let (mut resident_rows, mut classify_ns) = (0u64, 0u64);
        if classify {
            let tc = Instant::now();
            for (site, ids) in sites.iter().zip(&job.ids) {
                match site.resident_rows(ids) {
                    Ok(n) => resident_rows += n,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            classify_ns = tc.elapsed().as_nanos() as u64;
        }

        let widest = job.ids.iter().map(Vec::len).max().unwrap_or(0);
        let parts = workers.min(widest / PART_ROWS).max(1);
        let read = match err {
            Some(e) => Read {
                err: Some(e),
                ..Read::default()
            },
            None if parts == 1 => {
                let mut rest = &mut job.buf[..];
                let slots = sites
                    .iter()
                    .zip(&job.ids)
                    .map(|(site, ids)| take_slot(&mut rest, ids.len() * site.row_bytes() as usize));
                read_part(&sites, mode, &job.ids, slots)
            }
            None => {
                let read = read_parts(&sites, mode, &job.ids, &mut job.buf, parts);
                parts_faults.major += read.faults.major;
                parts_faults.minor += read.faults.minor;
                read
            }
        };
        let now = faults_thread();
        let times = HelperTimes {
            submit_ns: read.submit_ns,
            fill_ns: read.fill_ns,
            copy_ns: read.copy_ns,
            major_faults: now.major.saturating_sub(base.major) + parts_faults.major,
            minor_faults: now.minor.saturating_sub(base.minor) + parts_faults.minor,
            rows,
            resident_rows,
            classify_ns,
        };
        let err = read.err;
        if done.send(Filled { job, err, times }).is_err() {
            break;
        }
    }
}

/// A token's read, or one part of it ([`read_part`]): its times, the faults
/// its thread took, and its first refusal.
#[derive(Default)]
struct Read {
    submit_ns: u64,
    fill_ns: u64,
    copy_ns: u64,
    faults: Faults,
    err: Option<EngramError>,
}

/// `ids[s]`, rows of site `s`, advised (`WILLNEED`, one batch at the drive's
/// queue depth), filled by `mode` and copied into the slot `slots` yields for
/// that site, in site order; a slot the buffer cannot hold is refused by name.
fn read_part<'b, I: AsRef<[u32]>>(
    sites: &[&dyn RowTable],
    mode: FillMode,
    ids: &[I],
    slots: impl Iterator<Item = Option<&'b mut [u8]>>,
) -> Read {
    let mut err = None;
    let t0 = Instant::now();
    for (site, ids) in sites.iter().zip(ids) {
        if let Err(e) = site.prefetch(ids.as_ref()) {
            err = Some(e);
            break;
        }
    }
    let submit_ns = t0.elapsed().as_nanos() as u64;

    let t1 = Instant::now();
    if err.is_none() && mode == FillMode::Populate {
        for (site, ids) in sites.iter().zip(ids) {
            if let Err(e) = site.populate(ids.as_ref()) {
                err = Some(e);
                break;
            }
        }
    }
    let fill_ns = t1.elapsed().as_nanos() as u64;

    let t2 = Instant::now();
    if err.is_none() {
        for ((site, ids), slot) in sites.iter().zip(ids).zip(slots) {
            let Some(slot) = slot else {
                err = Some(EngramError::Prefetch(
                    "token wants more bytes than the buffer",
                ));
                break;
            };
            if let Err(e) = site.copy_rows(ids.as_ref(), slot) {
                err = Some(e);
                break;
            }
        }
    }
    let copy_ns = t2.elapsed().as_nanos() as u64;
    Read {
        submit_ns,
        fill_ns,
        copy_ns,
        faults: Faults::default(),
        err,
    }
}

/// The next `n` bytes of `rest`, taken off its front; `None`, leaving it
/// empty, when it holds fewer.
fn take_slot<'b>(rest: &mut &'b mut [u8], n: usize) -> Option<&'b mut [u8]> {
    let (slot, tail) = std::mem::take(rest).split_at_mut_checked(n)?;
    *rest = tail;
    Some(slot)
}

/// A token read in `parts` parts: each site's ids cut into `parts` runs in
/// order ([`threads::chunk_bounds`]), part `p` taking run `p` of every site
/// into its own slots of `buf` (the sites' rows in site order, as one read
/// lays them out). The helper reads part 0 and a thread of its own each
/// other part; the result takes each time from the slowest part, the faults
/// of the parts' own threads and the first refusal in part order. A thread
/// the system will not start is refused by name.
fn read_parts(
    sites: &[&dyn RowTable],
    mode: FillMode,
    ids: &[Vec<u32>],
    buf: &mut [u8],
    parts: usize,
) -> Read {
    let mut cut: Vec<_> = (0..parts)
        .map(|_| {
            (
                Vec::with_capacity(sites.len()),
                Vec::with_capacity(sites.len()),
            )
        })
        .collect();
    let mut rest = buf;
    for (site, ids) in sites.iter().zip(ids) {
        let rb = site.row_bytes() as usize;
        for (p, (part_ids, slots)) in cut.iter_mut().enumerate() {
            let (start, end) = threads::chunk_bounds(ids.len(), parts, p);
            let run = &ids[start..end];
            part_ids.push(run);
            slots.push(take_slot(&mut rest, run.len() * rb));
        }
    }
    std::thread::scope(|s| {
        let mut cut = cut.into_iter();
        let first = cut.next();
        let started: Vec<_> = cut
            .map(|(ids, slots)| {
                std::thread::Builder::new()
                    .name("engram-prefetch-part".into())
                    .spawn_scoped(s, move || {
                        let base = faults_thread();
                        let mut read = read_part(sites, mode, &ids, slots.into_iter());
                        let now = faults_thread();
                        read.faults = Faults {
                            major: now.major.saturating_sub(base.major),
                            minor: now.minor.saturating_sub(base.minor),
                        };
                        read
                    })
            })
            .collect();
        // The helper's own part: its faults are on the helper's count.
        let mut out = match first {
            Some((ids, slots)) => read_part(sites, mode, &ids, slots.into_iter()),
            None => Read::default(),
        };
        for h in started {
            let r = match h {
                Ok(h) => h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)),
                Err(e) => Read {
                    err: Some(EngramError::Io(e)),
                    ..Read::default()
                },
            };
            out.submit_ns = out.submit_ns.max(r.submit_ns);
            out.fill_ns = out.fill_ns.max(r.fill_ns);
            out.copy_ns = out.copy_ns.max(r.copy_ns);
            out.faults.major += r.faults.major;
            out.faults.minor += r.faults.minor;
            if out.err.is_none() {
                out.err = r.err;
            }
        }
        out
    })
}

/// The SMT sibling of the core the calling thread is pinned to; `None` for a
/// floating caller or a core without SMT: the helper then floats too, off the
/// caller's cpu ([`Prefetcher::with_options`]).
pub use threads::helper::caller_sibling;

/// Table `i` of `tables`, refused by name when the source counts it and
/// holds none.
fn table_of(tables: &dyn RowTables, i: usize) -> Result<&dyn RowTable, EngramError> {
    tables.table(i).ok_or(EngramError::Prefetch(
        "the row source counts a table it does not hold",
    ))
}

#[cfg(test)]
mod tests {
    use super::{FillMode, HelperOptions, PART_ROWS, Prefetcher};
    use crate::EngramError;
    use crate::rows::{PageAdvice, RunTable};
    use std::sync::Arc;

    /// A token of enough rows read in parts fills the buffer as one read
    /// does: every site's rows in the token's order, in site order, its rows
    /// counted once; a token too small to split fills the same way. Each
    /// part's own run of ids and its own slot are what make the bytes land
    /// where one read puts them. A helper of no workers is refused by name.
    #[test]
    fn a_token_read_in_parts_fills_as_one_read() {
        let dir = std::env::temp_dir().join(format!("engram-parts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Two tables of 4,096 rows, 24 and 8 bytes a row.
        let (mut tables, mut bytes) = (Vec::new(), Vec::new());
        for (t, rb) in [(0u32, 24u64), (1, 8)] {
            let path = dir.join(format!("t{t}"));
            let b: Vec<u8> = (0..4096 * rb as u32)
                .map(|i| (i.wrapping_mul(31) ^ (t * 101)) as u8)
                .collect();
            std::fs::write(&path, &b).unwrap();
            let runs = [(path, 0..b.len() as u64)];
            tables.push(RunTable::open("t", rb, &runs, PageAdvice::Random).unwrap());
            bytes.push((b, rb as usize));
        }
        let want = |ids: &[Vec<u32>]| -> Vec<u8> {
            bytes
                .iter()
                .zip(ids)
                .flat_map(|((b, rb), ids)| {
                    ids.iter()
                        .flat_map(move |&i| &b[i as usize * rb..][..*rb])
                        .copied()
                })
                .collect()
        };
        let options = HelperOptions {
            workers: 4,
            ..HelperOptions::of(FillMode::Touch)
        };
        let max = 3 * PART_ROWS + 5;
        let tables: Arc<Vec<RunTable>> = Arc::new(tables);
        let none = HelperOptions {
            workers: 0,
            ..options
        };
        assert!(matches!(
            Prefetcher::over(tables.clone(), &[max, max], none),
            Err(EngramError::Prefetch("a helper of no workers"))
        ));
        let mut pf = Prefetcher::over(tables, &[max, max], options).unwrap();
        // 3,077 and 2,500 rows: three parts, each site cut into three runs
        // of its own; then a token of one part.
        let big = vec![
            (0..max as u32).map(|i| i * 7 % 4096).collect::<Vec<_>>(),
            (0..2500u32).map(|i| 4095 - i * 13 % 4096).collect(),
        ];
        let small = vec![vec![9u32, 4000, 9], vec![3u32]];
        for ids in [&big, &small] {
            pf.submit(ids.iter()).unwrap();
            pf.wait().unwrap();
            assert!(pf.filled() == want(ids), "rows {}", ids[0].len());
            let rows: usize = ids.iter().map(Vec::len).sum();
            assert_eq!(pf.last().rows, rows as u64);
        }
        drop(pf);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
