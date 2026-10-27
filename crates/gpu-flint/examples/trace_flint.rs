//! Forced-ID Flint trajectory capture. Diagnostic artifacts, not an accuracy gate or refset.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_flint::{Body, FlintModel, prefill, set_prefill_group, set_taps};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::flint::place::{self, PlanInputs};
use model::placement::{Plan, PlanLevers, workstation};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

const WHAT: &str = "trace_flint";
const WIDTH: usize = 2048;
const LAYERS: usize = 49;
const VOCAB: usize = 150_144;
const PROMPT: usize = 22;
const TOTAL: usize = 96;
const CALLS: usize = TOTAL - PROMPT + 1;
const CTX: usize = 1024;
const SLOTS: usize = 2;
const USAGE: &str = "Usage: trace_flint -m MODEL --ids FILE --prompt 22 --out NEWDIR --ctx 1024 --slots 2 --mode eager|graph [--taps]\n\
Captures flint-trajectory-v2: 21-ID prefix prefill, then the last prompt ID and 74 forced one-ID steps.\n\
FILE must contain exactly 96 decimal token IDs; no tokenization, sampling or EOG stopping.\n\
--ctx is per-slot (total context 2048); exactly one CUDA device must be visible.\n\
--taps captures each call's last-position layer outputs and result_norm.\n\
Without --taps those tensors are explicitly absent; full logits are always captured.\n\
The output directory must not exist. MANIFEST.tsv is published only after all files finish.\n\
-h, --help print this usage without opening CUDA. --levers prints the existing lever table.";

struct Args {
    model: PathBuf,
    ids: PathBuf,
    out: PathBuf,
    mode: StepMode,
    taps: bool,
    plan_levers: PlanLevers,
}

fn assign<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(format!("{name}: duplicate option").into());
    }
    Ok(())
}

fn parse() -> Result<Option<Args>> {
    let mut args = std::env::args_os().skip(1);
    let (mut model, mut ids, mut out, mut mode) = (None, None, None, None);
    let (mut prompt, mut ctx, mut slots) = (None, None, None);
    let mut taps = false;
    while let Some(arg) = args.next() {
        let option = arg.to_str().ok_or("option is not UTF-8")?;
        if matches!(option, "-h" | "--help") {
            println!("{USAGE}");
            return Ok(None);
        }
        if option == "--levers" {
            continue;
        }
        if option == "--taps" {
            if taps {
                return Err("--taps: duplicate option".into());
            }
            taps = true;
            continue;
        }
        if !matches!(
            option,
            "-m" | "--ids" | "--out" | "--mode" | "--prompt" | "--ctx" | "--slots"
        ) {
            return Err(format!("unknown option {option:?}\n{USAGE}").into());
        }
        let value: OsString = args
            .next()
            .ok_or_else(|| format!("{option}: missing value"))?;
        match option {
            "-m" => assign(&mut model, PathBuf::from(value), option)?,
            "--ids" => assign(&mut ids, PathBuf::from(value), option)?,
            "--out" => assign(&mut out, PathBuf::from(value), option)?,
            "--mode" => {
                let value = match value.to_str() {
                    Some("eager") => StepMode::Eager,
                    Some("graph") => StepMode::Graph,
                    _ => return Err("--mode: expected eager or graph".into()),
                };
                assign(&mut mode, value, option)?;
            }
            "--prompt" | "--ctx" | "--slots" => {
                let value = value.to_str().ok_or("numeric option is not UTF-8")?;
                if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(format!("{option}: expected an unsigned decimal integer").into());
                }
                let value: usize = value.parse()?;
                let slot = match option {
                    "--prompt" => &mut prompt,
                    "--ctx" => &mut ctx,
                    _ => &mut slots,
                };
                assign(slot, value, option)?;
            }
            _ => unreachable!(),
        }
    }
    // This exits through the common parser before any CUDA call.
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    if prompt != Some(PROMPT) || ctx != Some(CTX) || slots != Some(SLOTS) {
        return Err(
            format!("required: --prompt {PROMPT} --ctx {CTX} --slots {SLOTS}\n{USAGE}").into(),
        );
    }
    Ok(Some(Args {
        model: model.ok_or("required: -m MODEL")?,
        ids: ids.ok_or("required: --ids FILE")?,
        out: out.ok_or("required: --out NEWDIR")?,
        mode: mode.ok_or("required: --mode eager|graph")?,
        taps,
        plan_levers: PlanLevers::from_levers(&levers)?,
    }))
}

fn tsv_path(path: &Path) -> Result<&str> {
    let text = path.to_str().ok_or("artifact path is not UTF-8")?;
    if text.contains(['\t', '\n', '\r']) {
        return Err("artifact path contains a TSV delimiter".into());
    }
    Ok(text)
}

fn read_ids(path: &Path) -> Result<Vec<u32>> {
    let text = fs::read_to_string(path)?;
    let ids = text
        .split_whitespace()
        .map(|word| {
            if !word.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("invalid decimal token ID {word:?}").into());
            }
            let id: u32 = word.parse()?;
            if id as usize >= VOCAB {
                return Err(format!("token ID {id} is outside vocabulary {VOCAB}").into());
            }
            Ok(id)
        })
        .collect::<Result<Vec<_>>>()?;
    if ids.len() != TOTAL {
        return Err(format!("expected exactly {TOTAL} IDs, found {}", ids.len()).into());
    }
    Ok(ids)
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn finish(mut file: BufWriter<File>) -> Result<()> {
    file.flush()?;
    file.get_ref().sync_all()?;
    Ok(())
}

fn floats(file: &mut BufWriter<File>, name: &str, values: &[f32], len: usize) -> Result<()> {
    if values.len() != len {
        return Err(format!("{name}: expected {len} floats, found {}", values.len()).into());
    }
    if let Some(index) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("{name}: non-finite float at {index}").into());
    }
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn placement(out: &Path, plan: &Plan<'_>, m: &FlintModel) -> Result<()> {
    let mut rows = create(&out.join("placement.tsv"))?;
    writeln!(
        rows,
        "tensor\tshard\tlayer\trole\tfile_type\tdims\tstage\tdevice\tformat\texpert_ids\tresident_bytes\tread_bytes"
    )?;
    for row in &plan.rows {
        let tensor = &plan.model.tensors[row.tensor];
        for segment in &row.segments {
            writeln!(
                rows,
                "{}\t{}\t{:?}\t{:?}\t{:?}\t{:?}\t{}\t{:?}\t{:?}\t{}\t{}\t{}",
                tensor.name,
                tensor.shard,
                tensor.layer,
                tensor.role,
                tensor.ty,
                tensor.dims,
                row.stage,
                segment.device,
                segment.format,
                segment
                    .experts
                    .as_ref()
                    .map_or_else(|| "absent".into(), ToString::to_string),
                segment.resident_bytes,
                row.read_bytes
            )?;
        }
    }
    finish(rows)?;
    let body = m.body(WHAT)?;
    let map = body.hybrid().slots();
    let mut slots = create(&out.join("expert-slots.tsv"))?;
    writeln!(slots, "layer\texpert_id\tactual_slot")?;
    for layer in map.layers() {
        for id in 0..map.n_expert() {
            let id = u32::try_from(id)?;
            writeln!(
                slots,
                "{layer}\t{id}\t{:?}",
                map.slot(layer, id).ok_or("missing loaded expert slot")?
            )?;
        }
    }
    finish(slots)?;
    let mut details = create(&out.join("placement.txt"))?;
    writeln!(
        details,
        "machine={:#?}\ncard_totals={:#?}\nhost_totals={:#?}\ncard_budget={:?}\nbudget_from_pool={}\nn_l={:?}\ntier_n_l={:?}\nnvme_bytes={}\nloaded_ctx={}\nloaded_store_bytes_per_slot={}\nloaded_scratch_bytes={}\nloaded_prompt_bytes={}",
        plan.machine,
        plan.cards,
        plan.host,
        plan.card_budget,
        plan.budget_from_pool,
        plan.n_l,
        plan.tier_n_l,
        plan.nvme_bytes,
        body.ctx(),
        body.store_bytes(),
        body.scratch_bytes(),
        body.prompt_bytes()
    )?;
    finish(details)
}

struct Capture {
    steps: BufWriter<File>,
    graph: BufWriter<File>,
    counters: BufWriter<File>,
    logits: BufWriter<File>,
    taps: Option<(BufWriter<File>, BufWriter<File>)>,
    row: Vec<f32>,
    graph_nodes: Option<usize>,
}

impl Capture {
    fn new(out: &Path, taps: bool) -> Result<Self> {
        let mut steps = create(&out.join("steps.tsv"))?;
        writeln!(steps, "call\tprocessed_pos\tpredict_pos\tinput_id\targmax")?;
        let mut graph = create(&out.join("graph.tsv"))?;
        writeln!(graph, "call\tmode\thas_capture\tdecode_nodes")?;
        let mut counters = create(&out.join("counters.tsv"))?;
        writeln!(counters, "call\thost_stats\ttier_stats")?;
        Ok(Self {
            steps,
            graph,
            counters,
            logits: create(&out.join("logits.f32"))?,
            taps: if taps {
                Some((
                    create(&out.join("layers.f32"))?,
                    create(&out.join("norm.f32"))?,
                ))
            } else {
                None
            },
            row: vec![0.0; VOCAB],
            graph_nodes: None,
        })
    }

    fn call(&mut self, m: &FlintModel, call: usize, input: u32, argmax: u32) -> Result<()> {
        let processed = PROMPT - 1 + call;
        if m.pos() as usize != processed + 1 || m.n_vocab()? != VOCAB || argmax as usize >= VOCAB {
            return Err(format!("call {call}: invalid position, vocabulary or argmax").into());
        }
        // Read the last call's taps before another prompt/step can overwrite them.
        if let Some((layers, norm)) = &mut self.taps {
            let values = m.body(WHAT)?.taps(m.gpu())?;
            floats(layers, "layers", &values, LAYERS * WIDTH)?;
            floats(norm, "result_norm", &m.head_normed()?, WIDTH)?;
        }
        m.logits_into(&mut self.row)?;
        floats(&mut self.logits, "logits", &self.row, VOCAB)?;
        let best = self.row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if self.row[argmax as usize] != best {
            return Err(format!(
                "call {call}: device argmax {argmax} does not maximize saved logits"
            )
            .into());
        }
        let captured = m.has_capture();
        if m.mode() == StepMode::Graph && !captured {
            return Err(format!("call {call}: graph mode did not capture a decode chain").into());
        }
        if m.mode() == StepMode::Eager && captured {
            return Err(format!("call {call}: eager mode unexpectedly holds a capture").into());
        }
        let nodes = if captured {
            if let Some(nodes) = self.graph_nodes {
                nodes
            } else {
                let nodes = m.step_graph_nodes()?.len();
                self.graph_nodes = Some(nodes);
                nodes
            }
        } else {
            0
        };
        if captured && nodes == 0 {
            return Err("captured decode graph is empty".into());
        }
        writeln!(
            self.steps,
            "{call}\t{processed}\t{}\t{input}\t{argmax}",
            processed + 1
        )?;
        writeln!(self.graph, "{call}\t{:?}\t{captured}\t{nodes}", m.mode())?;
        let hybrid = m.body(WHAT)?.hybrid();
        writeln!(
            self.counters,
            "{call}\t{:?}\t{:?}",
            hybrid.stats(),
            hybrid.tier_stats()
        )?;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        finish(self.steps)?;
        finish(self.graph)?;
        finish(self.counters)?;
        finish(self.logits)?;
        if let Some((layers, norm)) = self.taps {
            finish(layers)?;
            finish(norm)?;
        }
        Ok(())
    }
}

fn run(args: Args) -> Result<()> {
    let model = fs::canonicalize(&args.model)?;
    let ids_path = fs::canonicalize(&args.ids)?;
    tsv_path(&model)?;
    tsv_path(&ids_path)?;
    let ids = read_ids(&ids_path)?;
    let file = Split::open(&model)?;
    let mut inputs = PlanInputs::read(&file)?;
    if inputs.spec.layers.len() != LAYERS
        || inputs.spec.hidden as usize != WIDTH
        || inputs.spec.vocab as usize != VOCAB
    {
        return Err("expected the full 49-layer, width-2048, vocab-150144 Flint model".into());
    }
    place::configure_group(&mut inputs, 1)?;
    // Atomic directory creation rejects existing directories, files and symlinks before CUDA is touched.
    fs::create_dir(&args.out)?;
    let out = fs::canonicalize(&args.out)?;
    let mut tokens = create(&out.join("tokens.ids"))?;
    for id in &ids {
        writeln!(tokens, "{id}")?;
    }
    finish(tokens)?;
    let census = bloomery_gpu::census()?;
    let [device] = census.as_slice() else {
        return Err(format!(
            "exactly one CUDA device must be visible, found {}",
            census.len()
        )
        .into());
    };
    let mut hardware = create(&out.join("device.txt"))?;
    writeln!(hardware, "{device:#?}")?;
    finish(hardware)?;
    let machine = workstation::plan_on(workstation::spec_of_device(device), LAYERS);
    let slots = NonZeroUsize::new(SLOTS).ok_or("zero resident slots")?;
    let plan = inputs.plan_with_slots(&machine, CTX as u64, &args.plan_levers, slots)?;
    let host = HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };
    let mut m = Body::open_placed_slots(file, &plan, &inputs, 0, host, Residency::Off, slots)?;
    m.add_slots(SLOTS)?;
    if m.body(WHAT)?.ctx() != CTX {
        return Err("loaded per-slot context is not 1024".into());
    }
    set_prefill_group(&mut m, 1)?;
    if args.taps {
        set_taps(&mut m, true)?;
    }
    // Arming taps selects eager and invalidates captures; the requested arm is selected afterwards.
    m.set_mode(args.mode);
    placement(&out, &plan, &m)?;
    let mut capture = Capture::new(&out, args.taps)?;
    // The HTTP Slot::close_call prefills all but the last prompt ID, then calls next.
    prompt_pass(&mut m, |m| {
        m.run_rows(PROMPT - 1, WHAT, |gpu, weights, body, head, at| {
            prefill(gpu, weights, body, head, &ids[..PROMPT - 1], at)
        })
    })?
    .ok_or("batched prompt prefix produced no final argmax")?;
    for (call, &id) in ids[PROMPT - 1..].iter().enumerate() {
        let argmax = m.step(&[id])?;
        capture.call(&m, call, id, argmax)?;
    }
    let mut nodes = create(&out.join("graph-nodes.tsv"))?;
    writeln!(nodes, "node\tdescriptor")?;
    if args.mode == StepMode::Graph {
        for (index, node) in m.step_graph_nodes()?.iter().enumerate() {
            writeln!(nodes, "{index}\t{node:?}")?;
        }
    }
    finish(nodes)?;
    capture.finish()?;
    let mut manifest = create(&out.join("MANIFEST.tsv.partial"))?;
    let mode = match args.mode {
        StepMode::Eager => "eager",
        StepMode::Graph => "graph",
    };
    let instrumentation = if args.taps {
        "layer-head-taps"
    } else {
        "no-taps"
    };
    let layers_file = if args.taps {
        "layers.f32"
    } else {
        "absent:not-requested"
    };
    let norm_file = if args.taps {
        "norm.f32"
    } else {
        "absent:not-requested"
    };
    writeln!(
        manifest,
        "protocol\tflint-trajectory-v2\nmodel\t{}\nids_source\t{}\nengine\tbloomery-gpu-flint\nmode\t{mode}\ninstrumentation\t{instrumentation}\nctx_per_slot\t{CTX}\nctx_total\t{}\nslots\t{SLOTS}\nactive_slot\t0\nwidth\t{WIDTH}\nlayers\t{LAYERS}\nvocab\t{VOCAB}\nprompt\t{PROMPT}\ntotal_ids\t{TOTAL}\ncalls\t{CALLS}\nlogits\tlogits.f32\nlayers_file\t{layers_file}\nnorm_file\t{norm_file}\nlogits_layout\tf32le[call][150144]\nlayers_layout\tf32le[call][49][2048]\nnorm_layout\tf32le[call][2048]\nresidency\toff\nhost_populate\tfalse\nhost_lock\tfalse\ncard_dontneed\tfalse\nr8\tfalse\nroute_trace\tfalse\nprefill_group\t1\nprompt_submission\tprefix-prefill-then-next\nprefix_prefill_ids\t21\nfirst_observed_call\tone-id-next\nprompt_mode\teager-batched-prefix\ndecode_graph_captured\t{}\nplacement\tplacement.tsv\nloaded_expert_slots\texpert-slots.tsv\nplacement_counters\tplacement.txt\ndevice\tdevice.txt\ngraph_state\tgraph.tsv\ngraph_nodes\tgraph-nodes.tsv\nruntime_counters\tcounters.tsv\nprovenance\texternal-receipt-required\ncomplete\t1",
        tsv_path(&model)?,
        tsv_path(&ids_path)?,
        CTX * SLOTS,
        m.has_capture()
    )?;
    finish(manifest)?;
    fs::rename(out.join("MANIFEST.tsv.partial"), out.join("MANIFEST.tsv"))?;
    File::open(&out)?.sync_all()?;
    Ok(())
}

fn main() {
    let result = parse().and_then(|args| args.map_or(Ok(()), run));
    if let Err(error) = result {
        eprintln!("{WHAT}: {error}");
        std::process::exit(1);
    }
}
