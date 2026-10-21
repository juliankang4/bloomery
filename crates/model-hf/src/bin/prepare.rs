//! prepare — a Hugging Face checkpoint folder as a GGUF split the server opens as it opens any
//! (`gguf::Split`). The folder's quantization config places each tensor in a format
//! (`hf::quant_config`), the model's map names it, says how its role reads it and moves it into
//! the program's layout (`model_hf::arch::qwen35moe::Checkpoint`, qwen4exp's reader), and one table writes each
//! format as its GGUF type (`hf::modelopt::normalize`).
//!
//!     prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]
//!
//! writes `<out prefix>-00001-of-0000N.gguf` and on, each shard at most `n` GiB (48 by default)
//! unless one tensor with its scale and file layout is larger, the metadata in the first.
//! A quantized tensor keeps its stored format (NVFP4, MXFP8, FP8), and a BF16 matrix stays BF16,
//! the head and the token embedding among them; the server reads each through its BF16 site.
//! BF16 and F32 expert stacks are refused by name. An F32 matrix the map reads as Q8_0 is
//! refused unless `BLOOMERY_PREPARE_Q8=1` writes each single BF16 or F32 matrix read as Q8_0 as
//! ggml's reference Q8_0 instead, a lossy rewrite whose cost its lever row states; `--keep-bf16`
//! asks for the stored form and is refused beside it. Qwen3.8's decode reads about 9.27 GiB
//! of weights a token from the stored split where the Q8_0 one reads 5.73, the head alone
//! 0.55 GiB more
//! (1,271,398,400 bytes of BF16 against 34 bytes a 32 values). Each shard is written as
//! `<name>.part` and renamed when whole. The disk must hold the split and a tenth
//! more. The written split is opened and its hyperparameters read as the server reads them.
//! One line per tensor, a summary line, and exit status 1 with the error on any failure. Before
//! any shard is written, the tokenizer serving builds from the header is checked to split text
//! and stop generation where Hugging Face's does.

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use bloomery_levers::PREPARE_Q8;
use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Value};
use hf::modelopt::{self, Normal, Plain, Use};
use hf::quant_config::{self, Config, Format, Plan};
use model_hf::arch::qwen35moe::{Checkpoint, Mapped};
use safetensors::{Dtype, Folder};

const USAGE: &str = "usage: prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]";

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let ran = bloomery_levers::at_main(&[PREPARE_Q8])
        .map_err(Into::into)
        .and_then(|levers| {
            let args: Vec<String> = std::env::args().skip(1).collect();
            run(&args, levers.prepare_q8())
        });
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("prepare: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// One tensor of the split: its declaration and how its bytes are made.
struct Planned {
    decl: TensorDecl,
    make: Make,
}

enum Make {
    /// The mapped tensor, from its sources as the plan places them.
    Tensor(Mapped),
    /// The `.scale` beside the tensor `of`: its tensor scales, one a source for a stack.
    Scale { of: String },
}

/// The split of the folder `args` name; `q8` (`BLOOMERY_PREPARE_Q8`) writes its BF16 matrices the
/// map reads as Q8_0, and its F32 matrices, in Q8_0.
fn run(args: &[String], q8: bool) -> Res<()> {
    let (mut shard_gib, mut keep_bf16, mut paths) = (48u64, false, vec![]);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shard-gib" => shard_gib = it.next().ok_or(USAGE)?.parse()?,
            "--keep-bf16" => keep_bf16 = true,
            _ => paths.push(a),
        }
    }
    let [dir, prefix] = paths[..] else {
        return Err(USAGE.into());
    };
    if keep_bf16 && q8 {
        return Err(format!(
            "--keep-bf16 keeps the BF16 matrices {PREPARE_Q8}=1 asks to write as Q8_0: set one"
        )
        .into());
    }
    let limit = u64::checked_mul(shard_gib, 1 << 30)
        .filter(|&l| l > 0)
        .ok_or_else(|| {
            format!("--shard-gib {shard_gib}: not a shard size of 1 GiB or more in u64 bytes")
        })?;
    let dir = Path::new(dir);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let checkpoint = Checkpoint::parse(&config)?;
    let folder = Folder::open(dir)?;
    let entries: Vec<(String, Dtype, Vec<usize>)> = folder
        .names()
        .map(|n| {
            folder
                .raw(n)
                .map(|r| (n.to_string(), r.dtype, r.shape.to_vec()))
        })
        .collect::<Result<_, _>>()?;
    let (plan, planned) = plan_tensors(&checkpoint, &Config::read(dir)?, &entries, q8)?;

    let mut kvs = checkpoint.header(
        planned
            .iter()
            .map(|p| (p.decl.name.as_str(), p.decl.dims.as_slice())),
        &folder,
    )?;
    let vocab = hf::vocab::keys(dir, &checkpoint.vocab)?;
    kvs.extend(vocab.keys.iter().cloned());
    served_as_hf(
        &kvs,
        &vocab,
        Path::new(&format!("{prefix}.tokenizer.gguf.part")),
    )?;

    let tensors = planned.len();
    let shards = plan_shards(planned, &kvs, limit)?;
    let count = shards.len();
    let total = shards.iter().try_fold(0u64, |sum, (_, layout)| {
        sum.checked_add(layout.file_len())
            .ok_or("the split's file sizes overflow u64")
    })?;
    let out_dir = Path::new(prefix)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let free = model::fileio::free_bytes(out_dir)
        .map_err(|e| format!("statvfs {}: {e}", out_dir.display()))?;
    let needed = total
        .checked_add(total / 10)
        .ok_or("the split's file sizes and free-space margin overflow u64")?;
    if free < needed {
        return Err(format!(
            "{}: {free} bytes free; the split needs {total} and a tenth more",
            out_dir.display()
        )
        .into());
    }
    println!(
        "prepare: {} tensors, {total} bytes, {count} shards; plan refused {} and left {} input scales",
        tensors,
        plan.refused.len(),
        plan.input_scales.len()
    );

    let get = |n: &str| folder.raw(n);
    let mut tensor_scales: std::collections::HashMap<String, Vec<f32>> = Default::default();
    for (no, (shard, layout)) in shards.into_iter().enumerate() {
        let path = PathBuf::from(format!("{prefix}-{:05}-of-{count:05}.gguf", no + 1));
        let part = path.with_extension("gguf.part");
        let mut w = Writer::new(BufWriter::new(File::create(&part)?), layout)?;
        for p in shard {
            let t = Instant::now();
            let name = &p.decl.name;
            match p.make {
                Make::Scale { of } => {
                    let s = tensor_scales
                        .remove(&of)
                        .ok_or_else(|| format!("{name}: {of} gave no scale"))?;
                    w.tensor(
                        name,
                        &s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                    )?;
                }
                Make::Tensor(m) if m.role == Use::Rows => {
                    // A row table's parts, one at a time, all with the table's one scale.
                    let (mut failed, mut scale) = (None, vec![]);
                    let parts = m.parts(&plan, get).map_while(|n| match n {
                        Ok(n) => {
                            scale = n.scale;
                            Some(n.bytes)
                        }
                        Err(e) => {
                            failed = Some(e);
                            None
                        }
                    });
                    let written = w.tensor_parts(name, parts);
                    if let Some(e) = failed {
                        return Err(e.into());
                    }
                    written?;
                    tensor_scales.insert(name.clone(), scale);
                }
                Make::Tensor(m) => {
                    let mut n: Normal = modelopt::normalize(get, &m.placed(&plan)?, m.role)?;
                    for step in &m.steps {
                        step.apply(&checkpoint.geometry, name, &mut n)?;
                    }
                    if n.type_id != p.decl.type_id {
                        return Err(format!(
                            "{name}: type {}, declared {}",
                            n.type_id, p.decl.type_id
                        )
                        .into());
                    }
                    w.tensor(name, &n.bytes)?;
                    if !n.scale.is_empty() {
                        tensor_scales.insert(name.clone(), n.scale);
                    }
                }
            }
            println!(
                "prepare: {name} type={} bytes={} secs={:.3}",
                p.decl.type_id,
                p.decl.nbytes,
                t.elapsed().as_secs_f64()
            );
        }
        w.finish()?;
        std::fs::rename(&part, &path)?;
        println!("prepare: wrote {}", path.display());
    }
    // The split opens as the server opens it, its hyperparameters and layers whole.
    let split = gguf::Split::open(format!("{prefix}-00001-of-{count:05}.gguf"))?;
    let read = model::arch::spec(&split)?;
    println!(
        "prepare: the split opens: {} layers, {} tensors",
        read.spec.layers.len(),
        split.tensor_count()
    );
    Ok(())
}

/// The folder's `entries` (name, dtype, shape) placed by its `config`, and each tensor the map
/// makes of them declared in the type its placed format and role give, its `.scale` beside it
/// where the type reads one. An unquantized matrix the map reads as Q8_0 stays in its stored
/// format unless `q8` asks for the rewrite; F32 matrices need that rewrite. Unquantized expert
/// stacks are unsupported.
fn plan_tensors(
    checkpoint: &Checkpoint,
    config: &Config,
    entries: &[(String, Dtype, Vec<usize>)],
    q8: bool,
) -> Res<(Plan, Vec<Planned>)> {
    let plan = quant_config::plan(
        config,
        entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
    );
    let logical: Vec<(&str, Vec<usize>)> = entries
        .iter()
        .filter_map(
            |(n, _, s)| match (plan.placed.get(n), plan.refused.contains_key(n)) {
                (Some(p), _) => Some((n.as_str(), p.format.logical(s))),
                (None, true) => Some((n.as_str(), s.clone())),
                (None, false) => None,
            },
        )
        .collect();
    let mapped = checkpoint.map(logical.iter().map(|(n, s)| (*n, s.as_slice())))?;
    let mut planned = Vec::new();
    for (name, mut m) in mapped {
        let source = &m.sources[0];
        let format = modelopt::format_of(&m.placed(&plan)?)?;
        if m.role == Use::Matrix(Plain::Q8_0) && matches!(format, Format::Bf16 | Format::F32) {
            if m.dims.len() == 3 {
                return Err(format!(
                    "{source}: {name}: {format} expert stacks are unsupported by the converter"
                )
                .into());
            }
            if !q8 {
                if format == Format::F32 {
                    return Err(format!(
                        "{source}: {name} is stored {format}, a form its card site does not read; {PREPARE_Q8}=1 writes Q8_0"
                    )
                    .into());
                }
                m.role = Use::Matrix(Plain::Bf16);
            }
        }
        let type_id = modelopt::type_of(format, m.role).ok_or_else(|| {
            format!(
                "{source}: {format} read as {:?} has no GGUF type here",
                m.role
            )
        })?;
        let nbytes = size_of(type_id, &m.dims).ok_or_else(|| {
            format!(
                "{name}: dims {:?} are not whole blocks of type {type_id}",
                m.dims
            )
        })?;
        // One tensor scale an expert of a stack; one for a row table, whatever its parts.
        let scales = GgmlType::from_u32(type_id)
            .reads_scale()
            .then_some(match m.role {
                Use::Rows => 1,
                Use::Matrix(_) => m.sources.len(),
            });
        let decl = TensorDecl {
            name: name.clone(),
            dims: m.dims.clone(),
            type_id,
            nbytes,
        };
        planned.push(Planned {
            decl,
            make: Make::Tensor(m),
        });
        if let Some(n) = scales {
            let n = u64::try_from(n).map_err(|_| format!("{name}: scale count exceeds u64"))?;
            let nbytes = size_of(0, &[n])
                .ok_or_else(|| format!("{name}: scale byte count overflows u64"))?;
            let stem = name.strip_suffix(".weight").unwrap_or(&name);
            let decl = TensorDecl {
                name: format!("{stem}.scale"),
                dims: vec![n],
                type_id: 0,
                nbytes,
            };
            planned.push(Planned {
                decl,
                make: Make::Scale { of: name },
            });
        }
    }
    Ok((plan, planned))
}

fn plan_shards(
    planned: Vec<Planned>,
    kvs: &[(String, Value)],
    limit: u64,
) -> Res<Vec<(Vec<Planned>, Layout)>> {
    let tensors32 =
        i32::try_from(planned.len()).map_err(|_| "too many tensors for a GGUF split")?;
    let shard_keys = |no: u16, count: u16| {
        let mut keys = if no == 0 { kvs.to_vec() } else { vec![] };
        keys.extend([
            ("split.no".to_string(), Value::U16(no)),
            ("split.count".to_string(), Value::U16(count)),
            ("split.tensors.count".to_string(), Value::I32(tensors32)),
        ]);
        keys
    };
    let mut shards: Vec<Vec<Planned>> = vec![vec![]];
    let mut pending = planned.into_iter().peekable();
    while let Some(p) = pending.next() {
        let mut group = vec![p];
        if pending
            .peek()
            .is_some_and(|p| matches!(p.make, Make::Scale { .. }))
        {
            group.push(pending.next().expect("the scale was just inspected"));
        }
        let no = shards.len() - 1;
        let no16 = u16::try_from(no).map_err(|_| "too many shards for a GGUF split")?;
        let last = &mut shards[no];
        let declarations = last.iter().chain(&group).map(|p| p.decl.clone()).collect();
        // split.count has the same U16 width for every final count.
        let candidate = Layout::new(&shard_keys(no16, 1), declarations)?;
        if !last.is_empty() && candidate.file_len() > limit {
            shards.push(group);
        } else {
            last.extend(group);
        }
    }
    let count16 = u16::try_from(shards.len()).map_err(|_| "too many shards for a GGUF split")?;
    shards
        .into_iter()
        .enumerate()
        .map(|(no, shard)| {
            let keys = shard_keys(u16::try_from(no)?, count16);
            let layout = Layout::new(&keys, shard.iter().map(|p| p.decl.clone()).collect())?;
            Ok((shard, layout))
        })
        .collect()
}

/// Serving's tokenizer, which must preserve HF's split, rendering, special ids, automatic BOS/EOS
/// and generation stop sets. The header-only `probe` is removed before any shard is written.
fn served_as_hf(
    kvs: &[(String, Value)],
    vocab: &hf::vocab::Keys,
    probe: &Path,
) -> Res<tokenizer::Tokenizer> {
    let open = || -> Res<tokenizer::Tokenizer> {
        let layout = Layout::new(kvs, Vec::new())?;
        Writer::new(File::create(probe)?, layout)?.finish()?;
        Ok(tokenizer::Tokenizer::from_gguf(probe)?)
    };
    let served = open();
    if probe.exists() {
        std::fs::remove_file(probe)?;
    }
    let served = served?;
    let mut split = served.special_tokens().to_vec();
    split.sort_unstable();
    if split != vocab.added {
        return Err(format!(
            "serving splits text on tokens {split:?}, HF on its added tokens {:?}",
            vocab.added
        )
        .into());
    }
    if let Some(id) = vocab.stop.iter().find(|id| !served.eog().contains(id)) {
        return Err(format!(
            "HF's generate stops at token {id}, where serving does not end generation"
        )
        .into());
    }
    if let Some(id) = served.eog().iter().find(|id| !vocab.stop.contains(id)) {
        return Err(format!("serving stops at token {id}, where HF's generate continues").into());
    }
    let expected = |key: &str| vocab.keys.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let Some(Value::Array(types)) = expected("tokenizer.ggml.token_type") else {
        return Err("HF tokenizer has no tokenizer.ggml.token_type".into());
    };
    if types.len() != served.n_vocab() {
        return Err("serving vocabulary size differs from HF's".into());
    }
    for (id, ty) in types.iter().enumerate() {
        let id = u32::try_from(id)?;
        let control = served
            .attr(id)
            .is_some_and(|a| a & (tokenizer::attr::CONTROL | tokenizer::attr::UNKNOWN) != 0);
        if control != (*ty == Value::I32(3)) {
            return Err(
                format!("serving control/rendering at token {id} differs from HF's").into(),
            );
        }
    }
    for (key, actual) in [
        ("tokenizer.ggml.add_bos_token", served.add_bos()),
        ("tokenizer.ggml.add_eos_token", served.add_eos()),
    ] {
        if actual != expected(key).and_then(Value::as_bool).unwrap_or(false) {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    let specials = served.specials();
    for (key, actual) in [
        ("tokenizer.ggml.bos_token_id", specials.bos),
        ("tokenizer.ggml.eos_token_id", specials.eos),
        ("tokenizer.ggml.padding_token_id", specials.pad),
        ("tokenizer.ggml.unknown_token_id", specials.unk),
        ("tokenizer.ggml.seperator_token_id", specials.sep),
    ] {
        if actual.is_none()
            && matches!(
                key,
                "tokenizer.ggml.bos_token_id" | "tokenizer.ggml.eos_token_id"
            )
        {
            return Err(format!("serving requires {key}").into());
        }
        if actual.map(u64::from) != expected(key).and_then(Value::as_u64) {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    Ok(served)
}

/// The bytes of a tensor of `type_id` and `dims`, its blocks as the type's row sizes them.
fn size_of(type_id: u32, dims: &[u64]) -> Option<u64> {
    let ty = GgmlType::from_u32(type_id);
    let (blck, size) = ty.blck_size().zip(ty.type_size())?;
    if dims.is_empty() || !dims[0].is_multiple_of(blck) {
        return None;
    }
    gguf::tensor_bytes(blck, size, dims)
}

#[cfg(test)]
mod tests {
    use gguf::GENERAL_ARCHITECTURE;
    #[test]
    fn shard_budget_counts_scales_metadata_and_padding() {
        use super::*;

        let config: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
                "/../hf/tests/fixtures/quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz"))[..],
        )).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let shape = [
            config["text_config"]["vocab_size"].as_u64().unwrap() as usize,
            config["text_config"]["hidden_size"].as_u64().unwrap() as usize,
        ];
        let kvs = vec![
            (
                GENERAL_ARCHITECTURE.into(),
                Value::String("qwen4exp".into()),
            ),
            ("general.description".into(), Value::String("x".repeat(33))),
        ];
        for (limit, lengths) in [
            (640, vec![480, 352]),
            (704, vec![704]),
            (479, vec![480, 352]),
        ] {
            let mapped = checkpoint
                .map([
                    ("lm_head.weight", shape.as_slice()),
                    ("model.language_model.embed_tokens.weight", shape.as_slice()),
                ])
                .unwrap();
            assert_eq!(mapped.len(), 2);
            let mut planned = vec![];
            for (stem, m) in ["a", "b"].into_iter().zip(mapped.into_values()) {
                let name = format!("{stem}.weight");
                planned.push(Planned {
                    decl: TensorDecl {
                        name: name.clone(),
                        dims: vec![32],
                        type_id: 0,
                        nbytes: 128,
                    },
                    make: Make::Tensor(m),
                });
                planned.push(Planned {
                    decl: TensorDecl {
                        name: format!("{stem}.scale"),
                        dims: vec![1],
                        type_id: 0,
                        nbytes: 4,
                    },
                    make: Make::Scale { of: name },
                });
            }
            let shards = plan_shards(planned, &kvs, limit).unwrap();
            let mut written = vec![];
            for (shard, layout) in shards {
                let file_len = layout.file_len();
                let mut writer = Writer::new(vec![], layout).unwrap();
                for p in &shard {
                    writer
                        .tensor(&p.decl.name, &vec![0; p.decl.nbytes as usize])
                        .unwrap();
                }
                let bytes = writer.finish().unwrap();
                assert_eq!(bytes.len() as u64, file_len);
                eprintln!("budget={limit} file_len={file_len} tensors={}", shard.len());
                assert!(
                    file_len <= limit || shard.len() == 2,
                    "{file_len} > {limit}: {} tensors",
                    shard.len()
                );
                written.push(file_len);
            }
            assert_eq!(written, lengths, "budget={limit}");
        }
    }

    /// The HF contract is exact, including when the legacy vocabulary guesses extra roles.
    #[test]
    fn the_served_tokenizer_splits_and_stops_as_hf() {
        use super::*;

        let probe =
            std::env::temp_dir().join(format!("prepare-served-{}.gguf", std::process::id()));
        let stop_key = "tokenizer.bloomery.eog_token_ids";
        for (second, added, stop, explicit, refused) in [
            ("b", vec![2], vec![2], false, None),
            (
                "</s>",
                vec![2],
                vec![2],
                false,
                Some("splits text on tokens [1, 2]"),
            ),
            ("b", vec![2], vec![2, 0], false, Some("stops at token 0")),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![2],
                false,
                Some("serving stops at token 1"),
            ),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![1, 2],
                false,
                Some("control/rendering at token 1"),
            ),
            (
                "<|channel|>",
                vec![1, 2],
                vec![2],
                false,
                Some("control/rendering at token 1"),
            ),
            ("", vec![2], vec![2], true, Some("tokens[1] \"\"")),
            ("▁", vec![2], vec![2], true, Some("tokens[1] \"▁\"")),
            ("Ġx", vec![1, 2], vec![2], true, Some("tokens[1] \"Ġx\"")),
            ("▁x", vec![1, 2], vec![2], true, None),
            ("<|endoftext|>", vec![1, 2], vec![2], true, None),
            ("<|fim_pad|>", vec![1, 2], vec![2], true, None),
            ("<|repo_name|>", vec![1, 2], vec![2], true, None),
            ("<|file_sep|>", vec![1, 2], vec![2], true, None),
            ("<|channel|>", vec![1, 2], vec![2], true, None),
        ] {
            let second_type =
                if second == "<|channel|>" || (second == "<|endoftext|>" && stop == [2]) {
                    3
                } else if added.contains(&1) {
                    4
                } else {
                    1
                };
            let mut keys = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        ["a", second, "<|im_end|>", "aa"]
                            .map(|t| Value::String(t.into()))
                            .to_vec(),
                    ),
                ),
                (
                    "tokenizer.ggml.token_type".into(),
                    Value::Array([1, second_type, 3, 1].map(Value::I32).to_vec()),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    Value::Array(vec![Value::String("a a".into())]),
                ),
                ("tokenizer.ggml.eos_token_id".into(), Value::U32(2)),
                ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
            ];
            if explicit {
                keys.push((
                    stop_key.into(),
                    Value::Array(stop.iter().copied().map(Value::U32).collect()),
                ));
            }
            let vocab = hf::vocab::Keys {
                keys: keys.clone(),
                added,
                stop,
            };
            match (served_as_hf(&keys, &vocab, &probe).map(|_| ()), refused) {
                (Ok(()), None) => {}
                (Err(e), Some(what)) if e.to_string().contains(what) => {}
                (other, _) => panic!("{second} {:?}: {other:?}", vocab.stop),
            }
            assert!(!probe.exists());
            if explicit && refused.is_none() {
                for (key, value, error) in [
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0), Value::U32(2)]),
                        "serving stops at token 0",
                    ),
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0)]),
                        "stops at token 2",
                    ),
                    (
                        "tokenizer.ggml.add_bos_token",
                        Value::Bool(true),
                        "add_bos_token",
                    ),
                    (
                        "tokenizer.ggml.add_eos_token",
                        Value::Bool(true),
                        "add_eos_token",
                    ),
                    ("tokenizer.ggml.bos_token_id", Value::U32(1), "bos_token_id"),
                    ("tokenizer.ggml.eos_token_id", Value::U32(0), "eos_token_id"),
                    (
                        "tokenizer.ggml.padding_token_id",
                        Value::U32(1),
                        "padding_token_id",
                    ),
                    (
                        "tokenizer.ggml.unknown_token_id",
                        Value::U32(1),
                        "unknown_token_id",
                    ),
                    (
                        "tokenizer.ggml.seperator_token_id",
                        Value::U32(1),
                        "seperator_token_id",
                    ),
                ] {
                    let mut changed = keys.clone();
                    changed.retain(|(k, _)| k != key);
                    changed.push((key.into(), value));
                    let err = served_as_hf(&changed, &vocab, &probe).err().unwrap();
                    assert!(err.to_string().contains(error), "{second} {key}: {err}");
                    assert!(!probe.exists());
                }
                for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
                    let mut absent = hf::vocab::Keys {
                        keys: keys.clone(),
                        added: vocab.added.clone(),
                        stop: vocab.stop.clone(),
                    };
                    absent.keys.retain(|(k, _)| k != key);
                    let err = served_as_hf(&absent.keys, &absent, &probe).err().unwrap();
                    assert!(
                        err.to_string().contains(&format!("serving requires {key}")),
                        "{err}"
                    );
                    assert!(!probe.exists());
                }
                let mut both = keys.clone();
                let mut expected = hf::vocab::Keys {
                    keys: both.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                both.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                assert!(
                    served_as_hf(&both, &expected, &probe)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token")
                );
                expected.keys = both.clone();
                assert!(served_as_hf(&both, &expected, &probe).is_ok());
            }
        }
    }

    /// Serving's tokenizer, built from a Qwen3.8 folder's files as `prepare` builds it, encodes
    /// every text of the corpus to the ids Hugging Face's `tokenizers` gives
    /// (`hf/tests/fixtures/hf_encode.py`, which pins the `tokenizer.json` it ran on).
    #[test]
    #[ignore = "needs a Qwen3.8 folder's config.json and tokenizer files at $BLOOMERY_QWEN38_HF"]
    fn hw_the_served_tokenizer_encodes_the_corpus_as_hf() {
        use super::*;
        use sha2::{Digest, Sha256};

        let dir = PathBuf::from(
            std::env::var_os("BLOOMERY_QWEN38_HF").expect("$BLOOMERY_QWEN38_HF names the folder"),
        );
        let path = format!(
            "{}/../hf/tests/fixtures/qwen3.8-hf-encode.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let corpus: serde_json::Value =
            serde_json::from_reader(flate2::read::GzDecoder::new(File::open(&path).unwrap()))
                .unwrap();
        let file = std::fs::read(dir.join("tokenizer.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&file)),
            corpus["sha256"].as_str().unwrap(),
            "{}: not the tokenizer.json of {}",
            dir.display(),
            corpus["source"]
        );
        let config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap());
        let checkpoint = Checkpoint::parse(&config.unwrap()).unwrap();
        let vocab = hf::vocab::keys(&dir, &checkpoint.vocab).unwrap();
        let probe =
            std::env::temp_dir().join(format!("prepare-corpus-{}.gguf", std::process::id()));
        let served = served_as_hf(&vocab.keys, &vocab, &probe).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() >= 2000, "{} cases in {path}", cases.len());
        let (mut ids, mut failures) = (0, Vec::new());
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let ours = served.encode(text, false, true);
            ids += want.len();
            if ours != want {
                failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
            }
        }
        println!(
            "HF-ENCODE cases={} ids={ids} differences={} tokenizers={} source={}",
            cases.len(),
            failures.len(),
            corpus["tokenizers"],
            corpus["source"]
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A shard size of no bytes, or of more bytes than a u64 holds, is refused before the folder
    /// is read.
    #[test]
    fn a_shard_size_out_of_range_is_refused() {
        for n in ["0", "17179869184", "17179869185"] {
            let args = ["no-folder", "out", "--shard-gib", n].map(String::from);
            let e = super::run(&args, false).unwrap_err().to_string();
            assert!(e.contains("--shard-gib"), "{n}: {e}");
        }
    }

    /// Every tensor of layers 0, 1 and 3 and the unlayered text tensors of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4 is declared in its stored format: each one unsloth's
    /// UD-Q4_K_XL holds in Q8_0, a BF16 matrix in the checkpoint (the head and the token
    /// embedding among them), is BF16, and Q8_0 only when `BLOOMERY_PREPARE_Q8` asks; every other
    /// tensor (the NVFP4 stacks, the PLE table's F8 rows, the gains, the scales) is the same
    /// either way. Synthetic F32 matrices require the Q8_0 rewrite; BF16 and F32 routed stacks
    /// are refused with either lever value. Dropping either refusal fails these assertions.
    /// `--keep-bf16` asks for the stored form: taken alone, refused beside the lever.
    #[test]
    fn a_bf16_matrix_is_written_as_stored_unless_q8_is_asked() {
        use super::*;
        use model::arch::qwen35moe::names;
        use std::collections::BTreeMap;

        let gz = |name: &str| -> serde_json::Value {
            let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
            let file = File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            serde_json::from_reader(flate2::read::GzDecoder::new(file)).unwrap()
        };
        let config = gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz");
        let hf_quant =
            gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let quant = Config::parse(&config, Some(&hf_quant)).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let header = gz("checkpoint-headers/qwen3.8-flash-next-nvfp4-layers-0-1-3.tensors.json.gz");
        let entries: Vec<(String, Dtype, Vec<usize>)> = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let shape = e["shape"].as_array().unwrap();
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape.iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                )
            })
            .collect();
        let unsloth: Vec<(String, Vec<u64>, u32)> = serde_json::from_value(gz(
            "checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-layers-0-1-3.tensors.json.gz",
        ))
        .unwrap();
        let q8_0: Vec<&str> = unsloth
            .iter()
            .filter(|(_, _, ty)| *ty == GgmlType::Q8_0.as_u32())
            .map(|(n, _, _)| n.as_str())
            .collect();
        for name in [names::output(), names::token_embd(), names::attn_output(3)] {
            assert!(q8_0.contains(&name.as_str()), "{name}");
        }
        let types = |q8| -> BTreeMap<String, GgmlType> {
            let (_, planned) = plan_tensors(&checkpoint, &quant, &entries, q8).unwrap();
            planned
                .into_iter()
                .map(|p| (p.decl.name, GgmlType::from_u32(p.decl.type_id)))
                .collect()
        };
        let (stored, rewritten) = (types(false), types(true));
        assert_eq!(
            stored.keys().collect::<Vec<_>>(),
            rewritten.keys().collect::<Vec<_>>()
        );
        for (name, ty) in &stored {
            let want = if q8_0.contains(&name.as_str()) {
                (GgmlType::BF16, GgmlType::Q8_0)
            } else {
                (*ty, *ty)
            };
            assert_eq!((*ty, rewritten[name]), want, "{name}");
        }
        for (target, source) in [
            (
                names::attn_output(3),
                "model.language_model.layers.3.self_attn.o_proj.weight",
            ),
            (names::output(), "lm_head.weight"),
            (
                names::token_embd(),
                "model.language_model.embed_tokens.weight",
            ),
        ] {
            let mut f32 = entries.clone();
            f32.iter_mut().find(|(n, _, _)| n == source).unwrap().1 = Dtype::F32;
            for q8 in [false, true] {
                let result = plan_tensors(&checkpoint, &quant, &f32, q8);
                if !q8 {
                    let Err(e) = result else {
                        panic!("F32 matrix {target} must be refused");
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source) && e.contains(&target) && e.contains(PREPARE_Q8),
                        "{e}"
                    );
                } else {
                    let (_, planned) = result.unwrap();
                    let p = planned.iter().find(|p| p.decl.name == target).unwrap();
                    assert_eq!(p.decl.type_id, 8, "{target}");
                }
            }
        }
        let bf16: Vec<_> = entries
            .iter()
            .filter_map(|(n, d, s)| {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    n.ends_with(".weight")
                        .then(|| (n.clone(), Dtype::Bf16, Format::Nvfp4.logical(s)))
                } else {
                    Some((n.clone(), *d, s.clone()))
                }
            })
            .collect();
        let mut single = config.clone();
        single["text_config"]["num_experts"] = 1.into();
        single["text_config"]["num_experts_per_tok"] = 1.into();
        let single = Checkpoint::parse(&single).unwrap();
        for dtype in [Dtype::Bf16, Dtype::F32] {
            let mut stack = bf16.clone();
            for (n, d, _) in &mut stack {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    *d = dtype;
                }
            }
            let source = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
            let one: Vec<_> = stack
                .iter()
                .filter(|(n, _, _)| {
                    n == source
                        || n == "lm_head.weight"
                        || n == "model.language_model.embed_tokens.weight"
                })
                .cloned()
                .collect();
            for (checkpoint, stack) in [(&checkpoint, &stack), (&single, &one)] {
                for q8 in [false, true] {
                    let Err(e) = plan_tensors(checkpoint, &quant, stack, q8) else {
                        panic!(
                            "{dtype:?} expert stacks must be refused for q8={q8}, entries={}",
                            stack.len()
                        );
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source)
                            && e.contains(&names::ffn_gate_exps(0))
                            && e.contains("expert stacks are unsupported"),
                        "{e}"
                    );
                }
            }
        }
        for q8 in [false, true] {
            let args = ["no-folder", "out", "--keep-bf16"].map(String::from);
            let e = run(&args, q8).unwrap_err().to_string();
            assert_eq!(e.contains("--keep-bf16"), q8, "q8={q8}: {e}");
        }
    }
}
