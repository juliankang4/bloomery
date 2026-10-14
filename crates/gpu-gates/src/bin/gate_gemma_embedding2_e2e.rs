//! GPU gate for EmbeddingGemma 2's encoder (`bloomery-gpu-gemma-embedding2`)
//! end to end on the Q8_0 file (`--model FILE`, else [`MODEL`]: the path the
//! sets' `# model` line must name), against llama.cpp mainline's text set dumped
//! on the card ([`TEXT_SET`]) and its CPU twin ([`TEXT_CPU_SET`]): each case
//! the set names, its ids read from the ids set and held to the set's count
//! and digest, through one pass at the load's context with its taps read
//! back.
//!
//! A row's band is [`TWIN`] times the CPU twin's own distance from the card
//! set, and at least [`FLOOR`]: the twin is the floor any two realizations of
//! the graph lie apart (the Qwen seats' rule, `shared/glm5next_media.rs`).
//!
//! 1. (t) Per case: `inp_scaled` bit for bit (the host's token rows times
//!    `√hidden`, the one f32 multiply `ggml_scale` makes); `inp_per_layer`,
//!    every layer's `l_out`, `result_norm`, the token rows against `embd`
//!    and their mean ([`Pass::pooled`]) against `pooled`, each within its
//!    band. A case over more positions than a sliding layer reaches on both
//!    sides holds the window; every case holds the global layers.
//! 2. (d) The longest case run again with no taps is bit for bit the first
//!    run's rows.
//! 3. (r) Refused by name, the load untouched after each: a pass of no ids,
//!    of one id past the load's context and of an id past the vocabulary,
//!    and a load at one position past [`CARD_CTX`].
//!
//! The sets' family is the refset branch's `text-gemma-embedding2` row, its
//! names and pins mirrored here ([`text_family`]) until the gate reads it from
//! `refset::arch`.

#[cfg(feature = "gemma-embedding2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_e2e", gate::run())
}

#[cfg(feature = "gemma-embedding2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gemma-embedding2")]
mod gate {
    use std::path::Path;
    use std::process::Command;
    use std::sync::OnceLock;
    use std::time::Instant;

    use crate::e2e::{clause, elapsed, refused_by, rel, same_bits, tap};
    use bloomery_gpu::Gpu;
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, verdict};
    use bloomery_gpu_gemma_embedding2::{Encoder, Pass, card_bytes};
    use gguf::Split;
    use model::arch::gemma_embedding2::hparams::{CARD_CTX, Hparams};
    use refset::RefError;
    use refset::family::{Build, Family, Identity};

    /// The file every set is dumped from, unless `--model` names another.
    const MODEL: &str = "/models/embeddinggemma-2/embeddinggemma-2-Q8_0.gguf";
    /// The mainline commit every set names in its `# build` line.
    const LCPP_BUILD: &str = "6868a52cb";
    const ARCH: &str = "gemma-embedding2";
    /// The text set dumped on the card, and its CPU twin.
    const TEXT_SET: &str = "ref_gemma_embedding2_text";
    const TEXT_CPU_SET: &str = "ref_gemma_embedding2_text.cpu";
    /// The ids of every case, `<id>.ids`, one id a line.
    const IDS_SET: &str = "ref_gemma_embedding2_ids";
    const RECIPE: &str = "just dump-ref-gemma-embedding2";
    const CONSUMERS: &[&str] = &["gate-gpu-gemma-embedding2-e2e"];

    /// A row's band: this many times the CPU twin's distance from the card
    /// set (`shared/glm5next_media.rs`'s `TWIN`, the Qwen seats' rule).
    const TWIN: f64 = 2.5;

    /// The least band, derived: a q8_0 GEMM input is rounded to a step of
    /// `amax / 127` a block of 32, an rms error of `step / √12`; a block
    /// whose largest value is 2.5 of its rms rounds at
    /// `2.5 / (127 · √12) ≈ 5.7e-3` of it, and the card set rounds its own
    /// input apart from ours, `√2` that, `8e-3`, rounded up.
    const FLOOR: f64 = 1.0e-2;

    /// `--model`'s file, set once at the start of [`run`].
    static RUNS: OnceLock<String> = OnceLock::new();

    /// The text file the gate runs: `--model`'s, else [`MODEL`].
    fn model() -> Result<String, RefError> {
        Ok(RUNS.get().map_or_else(|| MODEL.to_string(), Clone::clone))
    }

    /// The family of the text sets, as the refset branch's row names it.
    const fn text_family(name: &'static str, set: &'static [&'static str]) -> Family {
        Family {
            name,
            sets: set,
            resolve: None,
            recipe: RECIPE,
            identity: Identity::Manifest,
            arch: Some(ARCH),
            build: Some(Build::Is(LCPP_BUILD)),
            runs: Some(model),
            draft_runs: None,
            consumers: CONSUMERS,
        }
    }

    static TEXT: Family = text_family("text-gemma-embedding2", &[TEXT_SET]);
    static TEXT_CPU: Family = text_family("text-gemma-embedding2-cpu", &[TEXT_CPU_SET]);

    /// One case of the set: its name and its ids.
    struct Case {
        id: String,
        ids: Vec<u32>,
    }

    /// The cases `man`'s `# case <id> <n> <path> <sha256>` lines name, each
    /// one's ids read from [`IDS_SET`] and refused by name when their count
    /// or their file's digest is not the line's.
    fn cases(man: &RefManifest) -> Result<Vec<Case>, GateError> {
        let dir = data_dir().join(IDS_SET);
        let mut out = Vec::new();
        for line in &man.header.other {
            let Some(rest) = line.strip_prefix("# case\t") else {
                continue;
            };
            let fields: Vec<&str> = rest.split('\t').collect();
            let [id, n, _, sha] = fields[..] else {
                return Err(format!("{TEXT_SET}: a case line of another shape: {line:?}").into());
            };
            let path = dir.join(format!("{id}.ids"));
            let ids = std::fs::read_to_string(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .lines()
                .map(|t| {
                    t.trim()
                        .parse::<u32>()
                        .map_err(|e| format!("{}: id {t:?}: {e}", path.display()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let digest = sha256(&path)?;
            if ids.len().to_string() != n || digest != sha {
                return Err(format!(
                    "{}: {} ids of digest {digest}; the set's case line says {n} of {sha}",
                    path.display(),
                    ids.len()
                )
                .into());
            }
            out.push(Case {
                id: id.to_string(),
                ids,
            });
        }
        if out.is_empty() {
            return Err(format!("{TEXT_SET}: no case line").into());
        }
        Ok(out)
    }

    /// The sha256 of the file at `path`, as `sha256sum` prints it.
    fn sha256(path: &Path) -> Result<String, GateError> {
        let out = Command::new("sha256sum").arg(path).output()?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        match text.split_whitespace().next() {
            Some(h) if out.status.success() => Ok(h.to_owned()),
            _ => Err(format!("sha256sum {}: {:?}", path.display(), text.trim()).into()),
        }
    }

    /// One row held to the card set: ours, the card set's and the twin's.
    /// Its line and whether it is within its band.
    fn held(what: &str, ours: &[f32], card: &[f32], twin: &[f32]) -> (String, bool) {
        let (r, floor) = (rel(ours, card), rel(twin, card));
        let band = (TWIN * floor).max(FLOOR);
        (
            format!("{what} {r:.3e} (twin {floor:.3e}, band {band:.3e})"),
            r <= band,
        )
    }

    /// Clause (t) for one case: its pass with taps, held tap by tap. The
    /// pass's rows, for (d).
    fn case(
        enc: &mut Encoder,
        c: &Case,
        man: &RefManifest,
        cpu: &RefManifest,
    ) -> Result<(bool, Pass), GateError> {
        let t0 = Instant::now();
        let pass = enc.pass(&c.ids, true)?;
        let taps = pass.taps.as_ref().ok_or("a pass with taps returned none")?;
        let row = |m: &RefManifest, name: &str| tap(m, &format!("{}/{name}", c.id));
        let pair = |name: &str| -> Result<(Vec<f32>, Vec<f32>), GateError> {
            Ok((row(man, name)?, row(cpu, name)?))
        };
        let mut ok = true;
        let mut lines = Vec::new();
        let exact = same_bits(&taps.inp_scaled, &row(man, "inp_scaled")?);
        lines.push(format!("inp_scaled bits {}", verdict(exact)));
        ok &= exact;
        let (card, twin) = pair("inp_per_layer")?;
        let (line, pass_ok) = held("inp_per_layer", &taps.inp_per_layer, &card, &twin);
        lines.push(line);
        ok &= pass_ok;
        // The worst layer by its distance over its band.
        let mut worst = (0.0f64, 0usize, String::new());
        for (l, ours) in taps.l_out.iter().enumerate() {
            let (card, twin) = pair(&format!("l_out-{l}"))?;
            let (r, floor) = (rel(ours, &card), rel(&twin, &card));
            let band = (TWIN * floor).max(FLOOR);
            if r > band {
                println!(
                    "(t) {}: layer {l} l_out {r:.3e} past its band {band:.3e} {}",
                    c.id,
                    verdict(false)
                );
                ok = false;
            }
            if r / band >= worst.0 {
                worst = (
                    r / band,
                    l,
                    format!("l_out worst layer {l} {r:.3e} (band {band:.3e})"),
                );
            }
        }
        lines.push(worst.2);
        for (what, ours, name) in [
            ("result_norm", &taps.result_norm, "result_norm"),
            ("embd", &pass.rows, "embd"),
        ] {
            let (card, twin) = pair(name)?;
            let (line, pass_ok) = held(what, ours, &card, &twin);
            lines.push(line);
            ok &= pass_ok;
        }
        let (card, twin) = pair("pooled")?;
        let (line, pass_ok) = held("pooled", &pass.pooled, &card, &twin);
        lines.push(line);
        ok &= pass_ok;
        println!(
            "(t) {}: {} ids, {}; {:.1} ms (runtime value) {}",
            c.id,
            c.ids.len(),
            lines.join("; "),
            t0.elapsed().as_secs_f64() * 1e3,
            verdict(ok)
        );
        Ok((ok, pass))
    }

    /// Clause (r): each refusal by its words, and a pass after them the
    /// first one's bits.
    fn refusals(
        enc: &mut Encoder,
        c: &Case,
        rows: &[f32],
        vocab: usize,
    ) -> Result<bool, GateError> {
        const PASS: &str = "Encoder::pass";
        let past = vec![c.ids[0]; enc.ctx() + 1];
        let mut bad = c.ids.clone();
        bad[0] = u32::try_from(vocab)?;
        let checks = [
            (
                "no ids",
                refused_by(&enc.pass(&[], false), PASS, "0 positions"),
            ),
            (
                "one past the context",
                refused_by(&enc.pass(&past, false), PASS, "a pass takes 1 to"),
            ),
            (
                "an id past the vocabulary",
                refused_by(&enc.pass(&bad, false), PASS, "past the vocabulary"),
            ),
        ];
        let mut ok = true;
        for (what, refused) in checks {
            println!("(r) {what}: refused by name {}", verdict(refused));
            ok &= refused;
        }
        enc.pass(&vec![c.ids[0]; enc.ctx()], false)?;
        let again = enc.pass(&c.ids, false)?;
        let same = same_bits(&again.rows, rows);
        println!(
            "(r) {}: after the refusals the rows bit for bit {}",
            c.id,
            verdict(same)
        );
        Ok(ok && same)
    }

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match &args[..] {
            [] => {}
            [flag, file] if flag == "--model" => {
                RUNS.set(file.clone()).map_err(|_| "--model is set once")?;
            }
            _ => return Err(format!("usage: [--model FILE]; got {args:?}").into()),
        }
        let path = model()?;
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split).map_err(|e| format!("{path}: {e}"))?;
        let ctx = CARD_CTX;
        let bytes = card_bytes(&split, ctx)?;
        let man = RefManifest::open(&TEXT.path_in(&data_dir(), TEXT_SET), &TEXT)?;
        let cpu = RefManifest::open(&TEXT_CPU.path_in(&data_dir(), TEXT_CPU_SET), &TEXT_CPU)?;
        let cases = cases(&man)?;
        let gpu = Gpu::new()?;
        println!(
            "gate_gemma_embedding2_e2e: device {}: {path} at {ctx} positions, card figure {bytes} B; \
             {} cases; band {TWIN} x the CPU twin, at least {FLOOR:e}",
            gpu.device_name()?,
            cases.len()
        );
        let t = Instant::now();
        let mut enc = Encoder::open(split, ctx, gpu)?;
        elapsed("(open)", &t);
        let mut ok = true;
        let mut longest: Option<(usize, Pass)> = None;
        for (i, c) in cases.iter().enumerate() {
            let r = case(&mut enc, c, &man, &cpu).map(|(pass_ok, pass)| {
                if longest
                    .as_ref()
                    .is_none_or(|(j, _)| c.ids.len() > cases[*j].ids.len())
                {
                    longest = Some((i, pass));
                }
                pass_ok
            });
            ok &= clause("(t)", &c.id, r);
        }
        if let Some((i, first)) = longest {
            let c = &cases[i];
            ok &= clause(
                "(d)",
                &c.id,
                enc.pass(&c.ids, false)
                    .map_err(GateError::from)
                    .map(|again| {
                        let same = same_bits(&again.rows, &first.rows);
                        println!(
                            "(d) {}: {} ids, a rerun's rows bit for bit {}",
                            c.id,
                            c.ids.len(),
                            verdict(same)
                        );
                        same
                    }),
            );
            ok &= clause(
                "(r)",
                "refusals",
                refusals(&mut enc, c, &first.rows, hp.n_vocab),
            );
        }
        let past = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let refused = refused_by(
            &Encoder::open(past, ctx + 1, Gpu::new()?),
            "Encoder::open",
            "a context of",
        );
        println!(
            "(r) a load at {} positions: refused by name {}",
            ctx + 1,
            verdict(refused)
        );
        ok &= refused;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
