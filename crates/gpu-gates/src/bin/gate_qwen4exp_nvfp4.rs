//! The Qwen3.8-Flash-Next NVFP4 gate: the split `prepare` writes from `nvidia/Qwen3.8-Flash-Next-NVFP4`
//! (routed experts NVFP4 with a tensor scale an expert, dense projections Q8_0, gains F32, the PLE
//! table in its F8 rows) against the checkpoint's own forward with the weights the split holds
//! (refset `hf-qwen4exp-nvfp4`, `tools/ref/qwen4exp-nvfp4-oracle.py`), by the e2e gate's bands
//! (`shared/q38_oracle.rs`). The plan is `--place a`'s on the one visible card, a DGX Spark's GB10,
//! whose budget holds every routed expert: no CPU kernel reads an NVFP4 stack.
//!
//! What is asserted:
//! - (l) the load: the plan puts every routed expert of every layer on the card and none on the
//!   host, so each layer's step runs without the host leg, and the load opens.
//! - (c) free: the batch set's five ids by eager steps with the taps armed. Every route tap is the
//!   top ten of its own logits; each layer output is within `FREE_BAND` of the oracle's off every
//!   flip's path; every flip's pair error is within `FLIP_ERR_CAP` router-logit units; the last
//!   argmax is the oracle's.
//! - (p) the batch set as one pass (`Prompt38::Pass`: the batch port, no layer served on the host)
//!   and (g) as one ubatch walk (`Prompt38::Gemm`: the routed experts through the NVFP4 GEMM): each
//!   last argmax is the oracle's, the logits' distance printed.
//! - (t) D3K's step at position 3,000 after its prefill by ubatches: the argmax is the oracle's or a
//!   named tie (ours the oracle's runner-up, the oracle's margin at most twice our distance at the
//!   two ids, the logits within `FREE_BAND`). The layer outputs are printed: a flip in any of the
//!   3,000 positions before it may lie on their path.
//! - Recorded, not judged: the batch run and the D3K step against the sets that keep the
//!   checkpoint's BF16 projections, the Q8_0 rounding's share of the distance.
//!
//! Real-only: the fixture holds no NVFP4 stack. Loads the whole file on the card.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_nvfp4", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the compare, layer-table, flip-accounting and step-set helpers; the rest serve the other gates"
)]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/q38_oracle.rs"]
mod q38_oracle;

#[cfg(feature = "gpu")]
#[path = "shared/qwen38_open.rs"]
#[allow(
    dead_code,
    reason = "the e2e gate prints and reads the plan's whole facts; this gate reads the counts"
)]
mod qwen38_open;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use crate::e2e::{
        argmax, flips_report, ik_last, layer_rels, layer_table, print_layers, rel, tie_numbers,
        worst_off_path,
    };
    use crate::q38_oracle::{FLIP_ERR_CAP, FREE_BAND, IkRoute, route_flips};
    use crate::qwen38_open::{Open38, open_split};

    use bloomery_gpu::arch::qwen3moe::{Prompt38, Qwen38Model, RouteTap};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, verdict};
    use gguf::GgmlType;
    use model::arch::qwen35moe::names;
    use model::arch::qwen35moe::place::{Experts, PlanInputs};
    use model::placement::PlanLevers;
    use refset::arch::qwen4exp::nvfp4::{BATCH, BATCH_BF16, D3K, D3K_BF16, HF};

    /// The split `prepare` writes from the family's folder, first of its shards.
    const MODEL: &str =
        "/models/Qwen3.8-Flash-Next-NVFP4/Qwen3.8-Flash-Next-NVFP4-00001-of-00004.gguf";

    /// Cache rows: D3K's step at position 3,000, with room.
    const CTX: usize = 3072;

    /// What the clauses read of the file's header: the trunk's layers, a layer output's values
    /// (every stream of the hidden width), and the routed experts and the slots a token takes.
    struct Dims {
        n_layer: usize,
        row: usize,
        n_expert: usize,
        n_used: usize,
    }

    /// Steps' last logits, and with the taps armed each step's layer outputs and route taps.
    struct Run {
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
        routes: Vec<Vec<RouteTap>>,
    }

    /// The set `name` of the family, checked against it.
    fn set(name: &str) -> Result<RefManifest, GateError> {
        let man = RefManifest::open(&HF.path_in(&data_dir(), name), &HF)?;
        refset::arch::qwen4exp::check_inputs(&man, name)?;
        Ok(man)
    }

    /// (l): the plan on the one visible card, every routed expert on it, and the load.
    fn open(d: &Dims) -> Result<(Qwen38Model, bool), GateError> {
        let file = open_split(Path::new(MODEL))?;
        let levers = bloomery_levers::at_main(&[])?;
        let specs = Place::A.on_host()?.card_specs()?;
        let [card] = specs[..] else {
            return Err(format!("--place a resolved to {} cards, not one", specs.len()).into());
        };
        let t = Instant::now();
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let plan_levers = PlanLevers::default();
        let planned = Open38 {
            card,
            ctx: CTX,
            ub,
            experts: Experts::Card,
            slots: 1,
            plan_levers: &plan_levers,
            host: levers.host(),
            room: None,
            residency: Residency::Off,
        }
        .plan(file)?;
        let p = planned.facts();
        let short: Vec<usize> = (0..d.n_layer)
            .filter(|&l| p.n_l.get(l) != Some(&(d.n_expert as u64)))
            .collect();
        let ok = p.host_experts == 0 && short.is_empty();
        println!(
            "(l) plan card={} ctx_max={} ubatch={ub} host_experts={} card_experts={} layers short of \
             {} card experts: {short:?} {}",
            card.name,
            p.ctx_max,
            p.host_experts,
            p.card_experts,
            d.n_expert,
            verdict(ok)
        );
        let o = planned.open()?;
        println!(
            "(l) load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            o.model.resident_bytes(),
            o.model.layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok((o.model, ok))
    }

    /// `toks` by eager steps from where the model stands, the taps armed.
    fn run_steps(m: &mut Qwen38Model, toks: &[u32]) -> Result<Run, GateError> {
        m.set_mode(StepMode::Eager);
        m.set_layer_taps(true)?;
        let mut r = Run {
            logits: Vec::new(),
            taps: Vec::new(),
            routes: Vec::new(),
        };
        for &t in toks {
            m.step(&[t])?;
            r.logits.push(m.logits()?);
            let (gpu, _, b) = m.body_parts("run_steps")?;
            r.taps.push(b.taps(gpu)?);
            r.routes.push(b.route_taps(gpu)?);
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        Ok(r)
    }

    /// (c) on `man`: the route taps' self-consistency, the layer outputs and flips against the
    /// oracle's, and the last argmax; the verdict, or `true` with the lines printed only when
    /// `judged` is false.
    fn free(
        man: &RefManifest,
        r: &Run,
        d: &Dims,
        vocab: usize,
        judged: bool,
    ) -> Result<bool, GateError> {
        let arm = man.dir.display().to_string();
        let inconsistent = r
            .routes
            .iter()
            .flatten()
            .filter(|t| {
                t.logits.len() != d.n_expert || !flip::picks_its_top(&t.ids, &t.logits, d.n_used)
            })
            .count();
        let table = layer_table(man, &r.taps, d.row, d.n_layer)?;
        let routes = (0..d.n_layer)
            .map(|l| IkRoute::read(man, l, d.n_expert, d.n_used))
            .collect::<Result<Vec<_>, _>>()?;
        let flips = route_flips(&r.routes, &routes)?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!(
                "free {arm} layer={l} l_out_rel by position {}",
                cells.join(" ")
            );
        }
        let flips_ok = flips_report(&flips, "free", FLIP_ERR_CAP);
        let (held, hl, ht, exempt) = worst_off_path(&table, &flips, false);
        let oracle = ik_last(man, vocab)?;
        let ours = r.logits.last().ok_or("no logits")?;
        let (top, oracle_top) = (argmax(ours), argmax(&oracle));
        let ok = held <= FREE_BAND && flips_ok && top == oracle_top && inconsistent == 0;
        println!(
            "free {arm}: {} tokens, {inconsistent} route taps not the top {} of their own logits, {} \
             flips ({} allowed), {exempt} outputs on a flip's path; worst l_out_rel off every flip's \
             path {held:.3e} at layer {hl} position {ht} (band {FREE_BAND:.2}); last argmax \
             ours={top} oracle={oracle_top} logits_rel={:.3e} (printed) {}",
            r.logits.len(),
            d.n_used,
            flips.len(),
            flips.iter().filter(|f| f.allowed(FLIP_ERR_CAP)).count(),
            rel(ours, &oracle),
            if judged {
                verdict(ok)
            } else {
                "(recorded, not judged)"
            }
        );
        Ok(ok || !judged)
    }

    /// (p) or (g), `tag`: the batch set by `path` (one pass or one ubatch walk, `how`) from a
    /// reset; its last argmax against the oracle's.
    fn batch_by(
        m: &mut Qwen38Model,
        toks: &[u32],
        vocab: usize,
        (tag, path, how): (&str, Prompt38, &str),
        man: &RefManifest,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let top = m.prompt38(toks, path)?;
        let ours = m.logits()?;
        let oracle = ik_last(man, vocab)?;
        let ok = top == argmax(&oracle);
        println!(
            "({tag}) the batch set by {how}: last argmax ours={top} oracle={} logits_rel={:.3e} \
             (printed) {}",
            argmax(&oracle),
            rel(&ours, &oracle),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (t): D3K's prefill by ubatches from a reset, then its step by an eager step with the taps
    /// armed, against `D3K` and, recorded, `D3K_BF16`.
    fn d3k_step(
        m: &mut Qwen38Model,
        d: &Dims,
        vocab: usize,
        man: &RefManifest,
        bf16: &RefManifest,
    ) -> Result<bool, GateError> {
        let (pos, step, prefill) = man.step()?;
        let tok = step[0];
        m.reset()?;
        let t = Instant::now();
        m.prompt38(prefill, Prompt38::Gemm)?;
        let fed = t.elapsed().as_secs_f64();
        let r = run_steps(m, &[tok])?;
        let ours = r.logits.last().ok_or("no logits")?;
        let mut ok = true;
        for (name, man, judged) in [(D3K, man, true), (D3K_BF16, bf16, false)] {
            let oracle = ik_last(man, vocab)?;
            let (top, oracle_top, oracle_2, margin, dist, logits_rel) = tie_numbers(ours, &oracle);
            let tie = top != oracle_top
                && top == oracle_2
                && margin <= 2.0 * dist
                && logits_rel <= FREE_BAND;
            let rels = layer_rels(man, &r.taps, d.row, d.n_layer)?;
            print_layers(name, &rels, 0, FREE_BAND);
            let step_ok = top == oracle_top || tie;
            println!(
                "(t) {name}: position {pos} after {} ids by ubatches ({fed:.1} s, runtime value); \
                 argmax ours={top} oracle={oracle_top} (its runner-up {oracle_2}, margin {margin:.4}, \
                 our distance at the two {dist:.4}{}); logits_rel {logits_rel:.3e}; worst \
                 l_out_rel {:.3e} (printed) {}",
                prefill.len(),
                if tie { ", a named tie" } else { "" },
                rels.iter().map(|r| r.0).fold(0.0, f64::max),
                if judged {
                    verdict(step_ok)
                } else {
                    "(recorded, not judged)"
                }
            );
            ok &= step_ok || !judged;
        }
        Ok(ok)
    }

    /// The header's dims, and the file's routed gate type, which must be NVFP4.
    fn dims() -> Result<Dims, GateError> {
        let file = open_split(Path::new(MODEL))?;
        let hp = PlanInputs::describe(&file)?.hp;
        let gate = file
            .find(&names::ffn_gate_exps(0))
            .map(|(_, t)| t.ty)
            .ok_or("the file has no routed gate stack on layer 0")?;
        if gate != GgmlType::NVFP4 {
            return Err(format!("{MODEL}: layer 0's routed gate is {gate:?}, not NVFP4").into());
        }
        let streams = hp.exp.as_ref().map_or(1, |e| e.hc_streams);
        Ok(Dims {
            n_layer: hp.n_trunk,
            row: streams * hp.n_embd,
            n_expert: hp.n_expert,
            n_used: hp.n_used,
        })
    }

    pub fn run() -> Result<(), GateError> {
        let batch = set(BATCH)?;
        let batch_bf16 = set(BATCH_BF16)?;
        let d3k = set(D3K)?;
        let d3k_bf16 = set(D3K_BF16)?;
        for (q8, bf16) in [(&batch, &batch_bf16), (&d3k, &d3k_bf16)] {
            if q8.header.tokens != bf16.header.tokens {
                return Err(format!(
                    "{}: oracle token ids differ from {}",
                    bf16.dir.display(),
                    q8.dir.display()
                )
                .into());
            }
        }
        let (_, toks, _) = batch.step()?;
        let d = dims()?;
        let (mut m, mut ok) = open(&d)?;
        let vocab = m.body("run")?.vocab();
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        ok &= free(&batch, &r, &d, vocab, true)?;
        ok &= free(&batch_bf16, &r, &d, vocab, false)?;
        ok &= batch_by(
            &mut m,
            toks,
            vocab,
            ("p", Prompt38::Pass, "one pass"),
            &batch,
        )?;
        ok &= batch_by(
            &mut m,
            toks,
            vocab,
            ("g", Prompt38::Gemm, "one ubatch"),
            &batch,
        )?;
        ok &= d3k_step(&mut m, &d, vocab, &d3k, &d3k_bf16)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
