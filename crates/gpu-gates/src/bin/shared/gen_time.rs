//! Pass walls and delivered-token timing for a drafted CLI run.

use bloomery_gpu_gates::record::{self, Record};

/// What the drafted generation's windows kept: every kept token at its
/// position with its window's wall over the window's kept rows, and every
/// window's proposal, kept rows and wall in ms.
pub(super) struct Windows<const ROWS: usize> {
    pub(super) kept: Vec<(u32, u32, f64)>,
    pub(super) passes: Vec<(bool, usize, usize, f64)>,
}

pub(super) struct Times {
    pub(super) steps: Vec<f64>,
    pub(super) counted_ms: f64,
    pub(super) positions: usize,
    pub(super) ms: f64,
}

impl<const ROWS: usize> Windows<ROWS> {
    pub(super) fn times(&self, steps: usize, warm: usize) -> Times {
        let per: Vec<f64> = self.kept[..steps].iter().map(|r| r.2).collect();
        let counted_ms = per[warm..].iter().sum();
        let positions = self.passes.iter().map(|p| p.1).sum();
        Times {
            steps: per,
            counted_ms,
            positions,
            ms: counted_ms * positions as f64 / (steps - warm) as f64,
        }
    }

    pub(super) fn summary(&self, t: &Times) -> Record {
        let mut kept = [0u64; ROWS];
        let mut widths = [0u64; ROWS];
        for &(_, k, rows, _) in &self.passes {
            kept[k - 1] += 1;
            widths[rows - 1] += 1;
        }
        Record::new(&record::MTP_SUMMARY)
            .u("proposals", self.passes.iter().filter(|p| p.0).count())
            .list("kept", &kept)
            .list("widths", &widths)
            .u("positions", t.positions)
            .u("passes", self.passes.len())
            .f("tok/s(positions)", t.positions as f64 * 1e3 / t.ms)
    }

    pub(super) fn pass_times(&self, warm: usize) -> impl Iterator<Item = Record> + '_ {
        self.passes
            .iter()
            .enumerate()
            .scan(1, move |at, (k, &(proposed, kept, _, ms))| {
                let line = Record::new(&record::TIME_PASS)
                    .u("i", k + 1)
                    .flag("warm", *at <= warm)
                    .f("ms", ms)
                    .u("positions", kept)
                    .w("kind", if proposed { "mtp" } else { "plain" });
                *at += kept;
                Some(line)
            })
    }
}

impl Times {
    pub(super) fn smoke(
        &self,
        mode: &str,
        place: &str,
        prompt: usize,
        generated: usize,
        warm: usize,
    ) -> Record {
        let counted = &self.steps[warm..];
        let mut sorted = counted.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p50 = sorted[sorted.len() / 2];
        Record::new(&record::SMOKE)
            .w("mode", mode)
            .w("place", place)
            .u("prompt_tokens", prompt)
            .u("depth", prompt)
            .u("generated", generated)
            .u("warm", warm)
            .u("steps", counted.len())
            .f("p50_ms", p50)
            .f("mean_ms", self.counted_ms / counted.len() as f64)
            .f("tok/s(p50)", 1e3 / p50)
            .u("positions", counted.len())
            .f(
                "tok/s(positions)",
                counted.len() as f64 * 1e3 / self.counted_ms,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::{Windows, record};

    fn windows() -> Windows<2> {
        Windows {
            kept: vec![(4, 10, 10.0), (5, 11, 20.0), (6, 12, 20.0)],
            passes: vec![(false, 1, 1, 10.0), (true, 2, 2, 40.0)],
        }
    }

    #[test]
    fn drafted_times_keep_the_last_pass_wall() {
        let w = windows();
        let t = w.times(2, 0);
        assert_eq!(t.steps, [10.0, 40.0]);
        assert_eq!(t.counted_ms, 50.0);
        assert_eq!((t.positions, t.ms), (3, 50.0));
        let exact = w.times(3, 0);
        assert_eq!(exact.steps, [10.0, 20.0, 20.0]);
        assert_eq!(exact.counted_ms, 50.0);
    }

    #[test]
    fn drafted_warm_counts_tokens_and_keeps_an_overlapping_pass() {
        let w = windows();
        for (steps, warm) in [(2, 1), (3, 1), (3, 2)] {
            let t = w.times(steps, warm);
            assert_eq!(t.counted_ms, 40.0, "steps={steps}, warm={warm}");
            assert_eq!((t.positions, t.ms), (3, 50.0));
        }
    }

    #[test]
    fn drafted_records_distinguish_executed_positions_from_delivered_tokens() {
        let w = windows();
        for (n_gen, warm, delivered, mean, rate) in [
            (3, 0, 2, 25.0, 40.0),
            (3, 1, 1, 40.0, 25.0),
            (4, 2, 1, 40.0, 25.0),
        ] {
            let t = w.times(n_gen - 1, warm);
            let mut lines = String::new();
            for line in w
                .pass_times(warm)
                .chain([w.summary(&t), t.smoke("eager", "a", 4, n_gen, warm)])
            {
                lines.push_str(&line.line());
                lines.push('\n');
            }
            println!("{lines}");
            let log = record::Log::of(&lines, record::GENERATE_GLM5NEXT);
            let summary = log.all(&record::MTP_SUMMARY).unwrap();
            assert_eq!(summary[0].u64("positions").unwrap(), 3);
            assert_eq!(summary[0].u64("passes").unwrap(), 2);
            assert_eq!(summary[0].f64("tok/s(positions)").unwrap(), 60.0);
            let smoke = log.all(&record::SMOKE).unwrap();
            assert_eq!(smoke[0].u64("generated").unwrap(), n_gen as u64);
            assert_eq!(smoke[0].u64("steps").unwrap(), delivered);
            assert_eq!(smoke[0].u64("positions").unwrap(), delivered);
            assert_eq!(smoke[0].f64("mean_ms").unwrap(), mean);
            assert_eq!(smoke[0].f64("tok/s(positions)").unwrap(), rate);
            let passes = log.all(&record::TIME_PASS).unwrap();
            assert_eq!(passes[0].flag("warm").unwrap(), warm >= 1);
            assert!(!passes[1].flag("warm").unwrap());
            assert_eq!(
                passes.iter().map(|p| p.f64("ms").unwrap()).sum::<f64>(),
                50.0
            );
        }
    }
}
