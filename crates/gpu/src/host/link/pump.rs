//! The loopback link's pump: the host thread that carries the tier page's
//! counters and the bytes they publish from one page to the other
//! ([`super::Link::loopback`]).
//!
//! On the shared page a counter of the tier protocol has two writers, at
//! different times: the stage card adds one to a row's go and the tier takes
//! it back, and the tier adds one to a row's done and the stage card takes it
//! back. Across two pages a copy of the writer's absolute value would erase
//! the reader's take back. So the counter crosses as a generation: on the
//! writer's page only the writing card adds to it, and for each new
//! generation the pump copies the bytes it publishes to the reader's page and
//! then adds one to the counter on the reader's page ([`Relay`]). The reading
//! card waits for at least one and takes it back as on the shared page, so a
//! captured wait keeps its one fixed target on every replay.
//!
//! The progress word the host reads on the stage card's page is the count of
//! dones the pump has carried: each signal adds one to a row's done and to
//! the progress word, so that count is the progress. The pump writes it after
//! the bytes of the done it counts and before the done itself.

use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

/// A page of 4-byte words the pump reads and writes by byte offset.
pub(super) trait Page: Send + Sync {
    /// The u32 at byte `off`, which the caller keeps 4-aligned and inside the
    /// page.
    fn word(&self, off: usize) -> &AtomicU32;
}

/// The side a counter's reader sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::host) enum Reader {
    /// The tier card reads it: a row's go.
    Tier,
    /// The stage card reads it: a row's done.
    Stage,
}

/// One counter that crosses the link, as the tier page lays it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::host) struct Crossing {
    /// The counter's byte offset: on the writer's page the generation the
    /// writing card adds one to, on the reader's the counter the pump adds
    /// one to and the reading card's wait takes back.
    pub counter: usize,
    pub reader: Reader,
    /// Byte spans the writing card fills before it adds to the counter,
    /// carried to the reader's page before each generation.
    pub spans: Vec<Range<usize>>,
    /// The progress word on the reader's page that counts this counter's
    /// generations with every other done of the same tier; `None` for a go.
    pub prog: Option<usize>,
}

/// One counter's generations as the pump has carried them: the last one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Relay {
    seen: u32,
}

impl Relay {
    /// A relay that has carried every generation up to `g`.
    pub(super) fn at(g: u32) -> Relay {
        Relay { seen: g }
    }

    /// What the writer's generation `now` carries: `Ok(false)` nothing new,
    /// `Ok(true)` one generation more, and the step as the error for
    /// anything else, since a row has at most one go or done in flight.
    /// Wrapping.
    pub(super) fn take(&mut self, now: u32) -> Result<bool, u32> {
        match now.wrapping_sub(self.seen) {
            0 => Ok(false),
            1 => {
                self.seen = now;
                Ok(true)
            }
            step => Err(step),
        }
    }

    /// Whether every generation up to `g` was carried.
    pub(super) fn holds(self, g: u32) -> bool {
        self.seen == g
    }
}

/// One counter's pump state.
struct Line {
    c: Crossing,
    /// The index of the counter's progress word in [`Relays::progs`].
    prog: Option<usize>,
    relay: Relay,
    /// Set by a release: the line carries nothing until its reset.
    stopped: bool,
}

/// A tier's progress as the reader's page shows it: the writer's progress
/// when the count started, and the dones carried since.
struct Prog {
    off: usize,
    base: u32,
    dones: u32,
}

/// The pump's state over every counter of the page.
pub(super) struct Relays {
    lines: Vec<Line>,
    progs: Vec<Prog>,
    /// The first step that was neither 0 nor 1, by name. From then on no
    /// line carries, across any reset.
    broken: Option<String>,
}

impl Relays {
    /// The pump over `crossings`, each counter's generation and progress
    /// taken as `stage` and `tier` hold them now, so nothing already there is
    /// carried again.
    pub(super) fn new(crossings: Vec<Crossing>, stage: &dyn Page, tier: &dyn Page) -> Relays {
        let mut progs: Vec<Prog> = Vec::new();
        let lines = crossings
            .into_iter()
            .map(|c| {
                let prog = c.prog.map(|off| {
                    progs.iter().position(|p| p.off == off).unwrap_or_else(|| {
                        progs.push(Prog {
                            off,
                            base: 0,
                            dones: 0,
                        });
                        progs.len() - 1
                    })
                });
                Line {
                    c,
                    prog,
                    relay: Relay::at(0),
                    stopped: false,
                }
            })
            .collect();
        let mut r = Relays {
            lines,
            progs,
            broken: None,
        };
        for i in 0..r.lines.len() {
            r.resync(i, stage, tier);
        }
        r
    }

    /// The pages a line's counter is written on and read on.
    fn sides<'a>(c: &Crossing, stage: &'a dyn Page, tier: &'a dyn Page) -> [&'a dyn Page; 2] {
        match c.reader {
            Reader::Tier => [stage, tier],
            Reader::Stage => [tier, stage],
        }
    }

    /// One pass over every counter not stopped: each new generation is
    /// carried. `true` when anything moved.
    pub(super) fn pass(&mut self, stage: &dyn Page, tier: &dyn Page) -> bool {
        let mut moved = false;
        for i in 0..self.lines.len() {
            if self.broken.is_some() {
                break;
            }
            if !self.lines[i].stopped {
                moved |= self.carry(i, stage, tier);
            }
        }
        moved
    }

    /// Line `i`'s next generation, if its writer added one: the step is
    /// checked first, then the bytes it publishes go to the reader's page,
    /// then the progress that counts it for a done, then one more on the
    /// reader's counter. A step other than 0 or 1 publishes nothing and stops
    /// the pump by name.
    fn carry(&mut self, i: usize, stage: &dyn Page, tier: &dyn Page) -> bool {
        let line = &mut self.lines[i];
        let [w, r] = Relays::sides(&line.c, stage, tier);
        let g = w.word(line.c.counter).load(Ordering::Acquire);
        match line.relay.take(g) {
            Ok(false) => return false,
            Ok(true) => {}
            Err(step) => {
                self.broken = Some(format!(
                    "the tier page's counter at byte {} stepped by {step} between two relays: a \
                     row has at most one go or done in flight",
                    line.c.counter
                ));
                return false;
            }
        }
        for span in &line.c.spans {
            for off in span.clone().step_by(4) {
                let v = w.word(off).load(Ordering::Relaxed);
                r.word(off).store(v, Ordering::Relaxed);
            }
        }
        if let Some(p) = line.prog {
            let p = &mut self.progs[p];
            p.dones = p.dones.wrapping_add(1);
            r.word(p.off)
                .store(p.base.wrapping_add(p.dones), Ordering::Release);
        }
        r.word(line.c.counter).fetch_add(1, Ordering::Release);
        true
    }

    /// The line of the counter at byte `off`; every counter the tier
    /// protocol names crosses.
    fn line(&self, off: usize) -> usize {
        self.lines
            .iter()
            .position(|l| l.c.counter == off)
            .expect("every counter of the tier page crosses the link")
    }

    /// Stop carrying the counter at byte `off` and set its reader's counter
    /// to `v`. The other counters carry on.
    pub(super) fn release(&mut self, off: usize, v: u32, stage: &dyn Page, tier: &dyn Page) {
        let i = self.line(off);
        let line = &mut self.lines[i];
        line.stopped = true;
        let [_, r] = Relays::sides(&line.c, stage, tier);
        r.word(off).store(v, Ordering::Release);
    }

    /// Whether the counter at byte `off` is at rest after one more pass:
    /// the reader's counter at 0, every generation carried, and no relay
    /// broken.
    pub(super) fn at_rest(&mut self, off: usize, stage: &dyn Page, tier: &dyn Page) -> bool {
        self.pass(stage, tier);
        let line = &self.lines[self.line(off)];
        let [w, r] = Relays::sides(&line.c, stage, tier);
        self.broken.is_none()
            && r.word(off).load(Ordering::Acquire) == 0
            && line.relay.holds(w.word(off).load(Ordering::Acquire))
    }

    /// The counter at byte `off` back to rest once both cards' streams have
    /// drained: every generation taken as carried, its tier's progress taken
    /// from the writer's page, the reader's counter at 0, and the counter
    /// carried again unless a relay broke. Only this counter resumes; one
    /// still released keeps what its writer adds until its own reset. A
    /// tier's dones share its progress word, so they are reset once that
    /// tier's stream has drained.
    pub(super) fn reset(&mut self, off: usize, stage: &dyn Page, tier: &dyn Page) {
        let i = self.line(off);
        self.resync(i, stage, tier);
        let line = &mut self.lines[i];
        let [_, r] = Relays::sides(&line.c, stage, tier);
        r.word(off).store(0, Ordering::Release);
        line.stopped = false;
    }

    /// Line `i` taken as carried up to the writer's generation, and its
    /// progress word restarted at the writer's.
    fn resync(&mut self, i: usize, stage: &dyn Page, tier: &dyn Page) {
        let line = &mut self.lines[i];
        let [w, r] = Relays::sides(&line.c, stage, tier);
        line.relay = Relay::at(w.word(line.c.counter).load(Ordering::Acquire));
        if let Some(p) = line.prog {
            let p = &mut self.progs[p];
            p.base = w.word(p.off).load(Ordering::Acquire);
            p.dones = 0;
            r.word(p.off).store(p.base, Ordering::Release);
        }
    }

    /// The broken relay's description, if one broke.
    pub(super) fn broken(&self) -> Option<&str> {
        self.broken.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use super::{Crossing, Page, Reader, Relay, Relays};
    use crate::host::step::RELEASE;

    struct Mem(Vec<AtomicU32>);

    impl Page for Mem {
        fn word(&self, off: usize) -> &AtomicU32 {
            &self.0[off / 4]
        }
    }

    #[test]
    fn a_relay_delivers_each_generation_once_across_the_wrap() {
        let mut r = Relay::at(u32::MAX - 1);
        assert_eq!(r.take(u32::MAX - 1), Ok(false));
        assert_eq!(r.take(u32::MAX), Ok(true));
        assert_eq!(r.take(u32::MAX), Ok(false));
        assert_eq!(r.take(0), Ok(true));
        assert!(r.holds(0));
        assert_eq!(r.take(2), Err(2));
        assert_eq!(r.take(u32::MAX), Err(u32::MAX));
    }

    // Byte offsets of the model page: one row of one tier.
    const GO: usize = 0;
    const CNT: usize = 64;
    const PROG: usize = 128;
    const IMAGE: usize = 192;
    const ROWS: usize = 256;
    const WORDS: usize = 80;

    fn page(seed: &[(usize, u32)]) -> Arc<Mem> {
        let m = Mem((0..WORDS).map(|_| AtomicU32::new(0)).collect());
        for &(off, v) in seed {
            m.word(off).store(v, Ordering::Relaxed);
        }
        Arc::new(m)
    }

    // The row's go, which publishes its image, and its done, which publishes
    // its rows and counts in the progress word.
    fn row() -> Vec<Crossing> {
        let (image, rows) = (IMAGE..IMAGE + 64, ROWS..ROWS + 64);
        vec![
            Crossing {
                counter: GO,
                reader: Reader::Tier,
                spans: vec![image],
                prog: None,
            },
            Crossing {
                counter: CNT,
                reader: Reader::Stage,
                spans: vec![rows],
                prog: Some(PROG),
            },
        ]
    }

    fn wait_take(p: &Mem, off: usize, until: Instant) -> bool {
        while p.word(off).load(Ordering::Acquire) < 1 {
            if Instant::now() > until {
                return false;
            }
            std::hint::spin_loop();
        }
        p.word(off).fetch_sub(1, Ordering::AcqRel);
        true
    }

    // The model cards run the tier protocol's ops: add one to the counter
    // each writes, wait for at least one on the counter each reads and take
    // one back. Goes `ks` carry image k and dones rows k + ROWS_OF; the bytes
    // are checked for every go up to `checked`, the ones before a release.
    const ROWS_OF: u32 = 1 << 20;

    fn tier_card(
        tier: Arc<Mem>,
        ks: std::ops::RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            for k in ks {
                if !wait_take(&tier, GO, until) {
                    return Err(format!("the tier's go {k} never came"));
                }
                let img = tier.word(IMAGE).load(Ordering::Relaxed);
                if k <= checked && img != k {
                    return Err(format!("the tier's go {k} carried the image of go {img}"));
                }
                tier.word(ROWS).store(k + ROWS_OF, Ordering::Relaxed);
                tier.word(CNT).fetch_add(1, Ordering::Release);
                tier.word(PROG).fetch_add(1, Ordering::Release);
            }
            Ok(())
        })
    }

    fn stage_card(
        stage: &Mem,
        ks: std::ops::RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> Result<(), String> {
        for k in ks {
            stage.word(IMAGE).store(k, Ordering::Relaxed);
            stage.word(GO).fetch_add(1, Ordering::Release);
            if !wait_take(stage, CNT, until) {
                return Err(format!("the stage's done {k} never came"));
            }
            let rows = stage.word(ROWS).load(Ordering::Relaxed);
            if k <= checked && rows != k + ROWS_OF {
                return Err(format!("the stage's done {k} carried rows {rows:#x}"));
            }
        }
        Ok(())
    }

    // Two model cards over the pump thread, from generations near the wrap:
    // every go carries its image and every done its rows. Then a release, and
    // the cancelled graph runs K more layers on both cards with nothing
    // carried; once both drain, the go's reset and the done's are taken
    // apart, the pump running between them, and the next N goes and dones
    // are carried as before. Both counters end at rest and the progress counts
    // every done.
    #[test]
    fn the_pump_carries_every_go_and_done_once_with_its_bytes_across_a_recovery() {
        const N: u32 = 2000;
        const K: u32 = 3;
        let seed = u32::MAX - 700;
        let stage = page(&[(GO, seed), (PROG, seed)]);
        let tier = page(&[(CNT, seed), (PROG, seed)]);
        let relays = Arc::new(Mutex::new(Relays::new(row(), &*stage, &*tier)));
        let quit = Arc::new(AtomicBool::new(false));
        let until = Instant::now() + Duration::from_secs(20);
        let pump = {
            let (relays, quit, stage, tier) =
                (relays.clone(), quit.clone(), stage.clone(), tier.clone());
            std::thread::spawn(move || {
                while !quit.load(Ordering::Acquire) {
                    relays.lock().unwrap().pass(&*stage, &*tier);
                }
            })
        };
        let run = || -> Result<(), String> {
            let card = tier_card(tier.clone(), 1..=N + K, N, until);
            stage_card(&stage, 1..=N, N, until)?;
            for off in [GO, CNT] {
                relays
                    .lock()
                    .unwrap()
                    .release(off, RELEASE, &*stage, &*tier);
            }
            stage_card(&stage, N + 1..=N + K, N, until)?;
            card.join().unwrap()?;
            relays.lock().unwrap().reset(GO, &*stage, &*tier);
            // The pump runs between the two resets, as it does while the
            // host drains the next tier's stream.
            let window = Instant::now() + Duration::from_millis(100);
            while Instant::now() < window && relays.lock().unwrap().broken().is_none() {
                std::thread::yield_now();
            }
            relays.lock().unwrap().reset(CNT, &*stage, &*tier);
            let card = tier_card(tier.clone(), N + K + 1..=2 * N + K, 2 * N + K, until);
            stage_card(&stage, N + K + 1..=2 * N + K, 2 * N + K, until)?;
            card.join().unwrap()
        };
        let ran = run();
        quit.store(true, Ordering::Release);
        pump.join().unwrap();
        assert_eq!(ran, Ok(()));
        let mut r = relays.lock().unwrap();
        assert_eq!(r.broken(), None);
        assert!(r.at_rest(GO, &*stage, &*tier) && r.at_rest(CNT, &*stage, &*tier));
        assert_eq!(
            stage.word(PROG).load(Ordering::Acquire),
            seed.wrapping_add(2 * N + K)
        );
    }

    // A release stops its own counter: what its writer adds meanwhile is not
    // carried. A reset of another counter leaves it stopped, and its own
    // reset takes what was added as carried, so the next generation is
    // delivered once and counted once.
    #[test]
    fn a_reset_reopens_its_own_counter_alone() {
        let stage = page(&[]);
        let tier = page(&[]);
        let mut r = Relays::new(row(), &*stage, &*tier);
        r.release(GO, RELEASE, &*stage, &*tier);
        r.release(CNT, RELEASE, &*stage, &*tier);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        tier.word(CNT).store(2, Ordering::Release);
        tier.word(PROG).store(2, Ordering::Release);
        assert!(!r.pass(&*stage, &*tier));
        r.reset(GO, &*stage, &*tier);
        r.pass(&*stage, &*tier);
        assert_eq!(r.broken(), None);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        assert!(!r.at_rest(CNT, &*stage, &*tier));
        r.reset(CNT, &*stage, &*tier);
        assert!(r.at_rest(GO, &*stage, &*tier) && r.at_rest(CNT, &*stage, &*tier));
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 2);
        tier.word(CNT).store(3, Ordering::Release);
        tier.word(PROG).store(3, Ordering::Release);
        r.pass(&*stage, &*tier);
        r.pass(&*stage, &*tier);
        assert_eq!(r.broken(), None);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 3);
    }

    // A done whose generation steps by two publishes nothing, not the rows it
    // carries (a raised fault copy among them), the progress or the counter:
    // the host sees the tier behind, not a done the stage card never gets. The
    // pump stops by name, and the stop holds across a reset.
    #[test]
    fn a_doubled_done_publishes_nothing() {
        let stage = page(&[]);
        let tier = page(&[]);
        let mut r = Relays::new(row(), &*stage, &*tier);
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(2, Ordering::Release);
        tier.word(PROG).store(2, Ordering::Release);
        r.pass(&*stage, &*tier);
        for off in [ROWS, PROG, CNT] {
            assert_eq!(stage.word(off).load(Ordering::Acquire), 0, "byte {off}");
        }
        assert!(r.broken().is_some_and(|b| b.contains("stepped by 2")));
        assert!(!r.at_rest(CNT, &*stage, &*tier));
        r.reset(CNT, &*stage, &*tier);
        assert!(r.broken().is_some());
        stage.word(GO).store(1, Ordering::Release);
        assert!(!r.pass(&*stage, &*tier));
        assert_eq!(tier.word(GO).load(Ordering::Acquire), 0);
    }
}
