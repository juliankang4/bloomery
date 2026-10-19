//! A checkpoint's quantization config read into one format a tensor.
//!
//! ModelOpt writes `hf_quant_config.json`: `quant_algo`, the per-module `quantized_layers` of a
//! MIXED_PRECISION export and `exclude_modules` globs. transformers' `fp8` (`config.json`
//! `quantization_config`) gives `weight_block_size` and `modules_to_not_convert`;
//! compressed-tensors gives `config_groups` and `ignore`. [`Config`] reads them into the formats
//! each module is declared, and [`plan`] gives every tensor of a checkpoint one [`Format`]: each
//! weight's stored tensors (dtype, shape and companion scales) are matched against the formats
//! its module is declared.
//!
//! A BF16 or F32 tensor is placed as stored: configs quantize linears, and a tensor table cannot
//! tell a linear from a norm. A quantized weight whose module the config leaves unquantized, or
//! whose tensors fit none of its module's formats, is refused by name, as is a companion (a scale,
//! an input scale, a packed shape) no placed weight claims, and any other dtype. An unknown
//! `quant_method`, algorithm or scheme is refused when the config is read. `input_scale` (an
//! activation scale) is reported, not used.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use gguf::GgmlType;
use regex::Regex;
use serde_json::Value;

use crate::safetensors::Dtype;
use crate::schema::{self, Fields, Rule::Read};

/// Why a config or a tensor was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("{0}: not JSON: {1}")]
    Json(String, #[source] serde_json::Error),
    #[error("quant config: {0}")]
    Config(String),
}

fn bad(what: impl Into<String>) -> Error {
    Error::Config(what.into())
}

/// How a weight is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// U8 E2M1 pairs, F8_E4M3 `weight_scale` a 16 values, F32 `weight_scale_2` a tensor.
    Nvfp4,
    /// [`Format::Nvfp4`] with its `weight_scale` stored compressed as `csf`: a `nvfp4_csf_fixed`
    /// U8 slab a 16 rows and `nvfp4_csf_exceptions` U32.
    Nvfp4Csf,
    /// [`Format::Nvfp4`] with its `weight_scale` stored compressed as `csf_32x2_v1`: the
    /// `csf32_bases`, `_codes`, `_page_bases`, `_offsets` and `_exceptions` parts.
    Nvfp4Csf32,
    /// OCP MXFP8: F8_E4M3 values, one E8M0 scale (U8 or F8_E8M0) a 32 values.
    Mxfp8,
    /// F8_E4M3 with one scale a `rows` by `cols` block.
    Fp8Block {
        rows: usize,
        cols: usize,
    },
    /// F8_E4M3 with one scale an output row.
    Fp8Channel,
    /// F8_E4M3 with one scale a tensor.
    Fp8Tensor,
    /// OCP MXFP4: U8 or I8 E2M1 pairs, one E8M0 scale a 32 values.
    Mxfp4,
    /// compressed-tensors pack-quantized: eight symmetric int4 an I32, one scale a `group` values.
    Int4Pack {
        group: usize,
    },
    Bf16,
    F32,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::Nvfp4 => f.write_str("NVFP4-ModelOpt"),
            Format::Nvfp4Csf => f.write_str("NVFP4-csf"),
            Format::Nvfp4Csf32 => f.write_str("NVFP4-csf32"),
            Format::Mxfp8 => f.write_str("MXFP8"),
            Format::Fp8Block { rows, cols } => write!(f, "FP8-block {rows}x{cols}"),
            Format::Fp8Channel => f.write_str("FP8-channel"),
            Format::Fp8Tensor => f.write_str("FP8-tensor"),
            Format::Mxfp4 => f.write_str("MXFP4-OCP"),
            Format::Int4Pack { group } => write!(f, "INT4-pack group {group}"),
            Format::Bf16 => f.write_str("BF16"),
            Format::F32 => f.write_str("F32"),
        }
    }
}

/// A tensor of one format as a card holds it, its block scales decoded to plain ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    /// The values and their block scales: the bytes a cut splits.
    pub bytes: u64,
    /// One tensor scale a matrix, which every rank holds whole.
    pub whole_bytes: u64,
    /// The values a cut keeps whole along dims 0 and 1 (ggml's order).
    pub block: [u64; 2],
}

impl Format {
    /// The card type the format is held in, whose row ([`GgmlType::row`]) gives its bytes and
    /// cut units: NVFP4 in GGUF's NVFP4 block whichever encoding its scales were stored in.
    fn card_type(self) -> Option<GgmlType> {
        match self {
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 => Some(GgmlType::NVFP4),
            Format::Mxfp4 => Some(GgmlType::MXFP4),
            Format::Bf16 => Some(GgmlType::BF16),
            Format::F32 => Some(GgmlType::F32),
            _ => None,
        }
    }

    /// A format with no card type row: the values a block spans along dims 0 and 1 and the
    /// bytes of its one scale, beside one byte a value. MXFP8's scale is one E8M0 byte a 32
    /// values; an FP8 block format's one F32 a block.
    fn scaled_bytes(self) -> Option<([u64; 2], u64)> {
        match self {
            Format::Mxfp8 => Some(([32, 1], 1)),
            Format::Fp8Block { rows, cols } => Some(([cols as u64, rows as u64], 4)),
            _ => None,
        }
    }

    /// The held form of a tensor of the format and logical `dims` (ggml's order: dim 0 the values
    /// a row holds, dim 1 the rows, the dims past them whole matrices); `None` for a format no
    /// card holds, a zero dimension, a byte count past `u64`, or a row that is not whole blocks of
    /// its card type. A format with a card type reads its row; NVFP4 adds its one F32 tensor scale
    /// a matrix.
    #[must_use]
    pub fn held(self, dims: &[u64]) -> Option<Held> {
        let k = *dims.first().filter(|_| !dims.contains(&0))?;
        let rows = dims.get(1).copied().unwrap_or(1);
        let matrices = dims
            .iter()
            .skip(2)
            .try_fold(1u64, |a, &d| a.checked_mul(d))?;
        let (bytes, block) = match self.card_type() {
            Some(ty) => {
                let row = ty.row()?;
                let block = [row.blck, 1];
                if k % block[0] != 0 {
                    return None;
                }
                (gguf::tensor_bytes(row.blck, row.bytes, dims)?, block)
            }
            None => {
                let (block, scale) = self.scaled_bytes()?;
                let values = k.checked_mul(rows)?.checked_mul(matrices)?;
                let scales = k
                    .div_ceil(block[0])
                    .checked_mul(rows.div_ceil(block[1]))?
                    .checked_mul(matrices)?;
                (values.checked_add(scale.checked_mul(scales)?)?, block)
            }
        };
        let whole_bytes = match self {
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 => matrices.checked_mul(4)?,
            _ => 0,
        };
        Some(Held {
            bytes,
            whole_bytes,
            block,
        })
    }

    /// The logical shape of a tensor of the format with `stored` shape (row-major): two E2M1
    /// values a stored byte for NVFP4 and MXFP4, eight int4 an I32 for the packed format. `stored`
    /// is a weight [`plan`] placed in the format, whose unpacked count it checked fits.
    #[must_use]
    pub fn logical(self, stored: &[usize]) -> Vec<usize> {
        let mut s = stored.to_vec();
        let per = match self {
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 | Format::Mxfp4 => 2,
            Format::Int4Pack { .. } => 8,
            _ => 1,
        };
        if let Some(last) = s.last_mut() {
            *last *= per;
        }
        s
    }
}

/// A checkpoint's quantization config: the formats each module is declared.
#[derive(Debug, Default)]
pub struct Config {
    /// ModelOpt's config, which takes a module first.
    modelopt: Option<Modelopt>,
    /// `config.json`'s own config: the modules ModelOpt leaves, or every module.
    other: Option<Other>,
}

#[derive(Debug)]
struct Modelopt {
    /// The top-level algorithm's format; `None` for MIXED_PRECISION.
    all: Option<Format>,
    /// `quantized_layers`: a module and its format.
    layers: Vec<(String, Format)>,
    exclude: Vec<String>,
}

#[derive(Debug)]
enum Other {
    Fp8 {
        block: Format,
        mxfp4: bool,
        not_convert: Vec<String>,
        convert: Vec<String>,
    },
    CompressedTensors {
        groups: Vec<(Vec<Target>, Format)>,
        ignore: Vec<Target>,
    },
}

/// A compressed-tensors target: every linear, a module by name, or `re:` a regex from the start.
#[derive(Debug)]
enum Target {
    Linear,
    Name(String),
    Re(Regex),
}

impl Target {
    fn parse(t: &str) -> Result<Target, Error> {
        Ok(match t.strip_prefix("re:") {
            Some(re) => Target::Re(
                Regex::new(&format!("^(?:{re})"))
                    .map_err(|e| bad(format!("compressed-tensors pattern {t}: {e}")))?,
            ),
            None if t == "Linear" => Target::Linear,
            None => Target::Name(t.to_string()),
        })
    }

    fn matches(&self, module: &str) -> bool {
        match self {
            Target::Linear => true,
            Target::Name(n) => n == module,
            Target::Re(re) => re.is_match(module),
        }
    }
}

impl Config {
    /// The config of the model folder `dir`: `config.json`, and `hf_quant_config.json` when the
    /// folder has one.
    pub fn read(dir: &Path) -> Result<Config, Error> {
        let json = |name: &str| crate::read_json(dir, name, Error::Io, Error::Json);
        let config = json("config.json")?.ok_or_else(|| bad("the folder has no config.json"))?;
        Config::parse(&config, json("hf_quant_config.json")?.as_ref())
    }

    /// The config from `config.json` and ModelOpt's `hf_quant_config.json`.
    pub fn parse(config: &Value, hf_quant: Option<&Value>) -> Result<Config, Error> {
        let qc = config
            .get("quantization_config")
            .or_else(|| config.get("text_config")?.get("quantization_config"));
        let method = qc.map(|q| str_of(q, "quant_method")).transpose()?;
        if method == Some("quark") {
            return Err(bad("Quark checkpoints are not read yet"));
        }
        let mut out = Config::default();
        // ModelOpt's own file wins over the copy it writes into config.json.
        let modelopt = match (hf_quant, method) {
            (Some(h), _) => Some(h.get("quantization").unwrap_or(h)),
            (None, Some("modelopt")) => qc,
            _ => None,
        };
        if let Some(m) = modelopt {
            out.modelopt = Some(Modelopt::parse(m)?);
        }
        out.other = match (qc, method) {
            (_, None | Some("modelopt")) => None,
            (Some(q), Some("fp8")) => Some(Other::fp8(q)?),
            (Some(q), Some("compressed-tensors")) => Some(Other::compressed_tensors(q)?),
            (_, Some(m)) => {
                return Err(bad(format!(
                    "quant_method {m} is not one this reader knows"
                )));
            }
        };
        Ok(out)
    }

    /// The formats `module` is declared, and the module whose scale a tensor-scale format shares;
    /// empty when the config leaves it unquantized.
    fn declared(&self, module: &str) -> (Vec<Format>, Option<&str>) {
        if let Some(m) = &self.modelopt {
            let layer = m
                .layers
                .iter()
                .filter(|(k, _)| within(module, k))
                .max_by_key(|(k, _)| k.len());
            if let Some((k, f)) = layer {
                return (vec![*f], Some(k));
            }
            if let Some(f) = m.all
                && !m.exclude.iter().any(|p| glob(p, module))
            {
                return (vec![f], None);
            }
        }
        let formats = match &self.other {
            None => vec![],
            Some(Other::Fp8 {
                block,
                mxfp4,
                not_convert,
                convert,
            }) => {
                let skip = |ps: &[String]| {
                    ps.iter()
                        .any(|p| module.starts_with(p.as_str()) || module.ends_with(p.as_str()))
                };
                if skip(not_convert) && !skip(convert) {
                    vec![]
                } else if *mxfp4 {
                    vec![*block, Format::Mxfp4]
                } else {
                    vec![*block]
                }
            }
            Some(Other::CompressedTensors { groups, ignore }) => {
                if ignore.iter().any(|t| t.matches(module)) {
                    vec![]
                } else {
                    groups
                        .iter()
                        .filter(|(ts, _)| ts.iter().any(|t| t.matches(module)))
                        .map(|(_, f)| *f)
                        .collect()
                }
            }
        };
        (formats, None)
    }
}

/// A `quantized_layers` entry's keys, each read by [`modelopt_format`].
const LAYER: &Fields = &[
    ("quant_algo", Read),
    ("group_size", Read),
    ("weight_block_size", Read),
    ("weight_scale_encoding", Read),
];

impl Modelopt {
    fn parse(q: &Value) -> Result<Modelopt, Error> {
        let algo = str_of(q, "quant_algo")?;
        let all = match algo {
            "MIXED_PRECISION" => None,
            _ => Some(modelopt_format("the checkpoint", algo, q)?),
        };
        let mut layers = Vec::new();
        if let Some(l) = q.get("quantized_layers") {
            let l = l
                .as_object()
                .ok_or_else(|| bad("modelopt quantized_layers is no object"))?;
            for (module, entry) in l {
                let entry = entry.as_object().ok_or_else(|| {
                    bad(format!("modelopt quantized_layers {module} is no object"))
                })?;
                let value = Value::Object(entry.clone());
                let format = modelopt_format(module, str_of(&value, "quant_algo")?, &value)?;
                schema::check(
                    &value,
                    &format!("modelopt quantized_layers {module}"),
                    &[LAYER],
                )
                .map_err(bad)?;
                layers.push((module.clone(), format));
            }
        }
        let exclude = q
            .get("exclude_modules")
            .or_else(|| q.get("ignore"))
            .map_or(Ok(vec![]), |v| strings(v, "modelopt exclude_modules"))?;
        if let Some(p) = exclude.iter().find(|p| p.contains('[')) {
            return Err(bad(format!(
                "modelopt exclude pattern {p}: a [ class is not matched by this reader"
            )));
        }
        Ok(Modelopt {
            all,
            layers,
            exclude,
        })
    }
}

/// The format of ModelOpt's `algo` for `module`, with `entry`'s group or block size and NVFP4's
/// `weight_scale_encoding`. W4A16_NVFP4 stores its weights as NVFP4 does; only its activations
/// stay 16-bit, which no stored tensor shows.
fn modelopt_format(module: &str, algo: &str, entry: &Value) -> Result<Format, Error> {
    let group = match entry.get("group_size") {
        None | Some(Value::Null) => None,
        Some(g) => Some(g.as_u64().filter(|&g| g > 0).ok_or_else(|| {
            bad(format!(
                "modelopt group_size {g} for {module} is not a count"
            ))
        })?),
    };
    let refuse = || {
        bad(format!(
            "modelopt quant_algo {algo} for {module} is not a format this reader places"
        ))
    };
    let encoding = entry.get("weight_scale_encoding");
    if encoding.is_some() && algo != "NVFP4" {
        return Err(bad(format!(
            "modelopt quant_algo {algo} for {module}: weight_scale_encoding is NVFP4's alone"
        )));
    }
    match algo {
        "NVFP4" if group.is_none_or(|g| g == 16) => match encoding {
            None => Ok(Format::Nvfp4),
            Some(e) => match e.as_str() {
                Some("csf") => Ok(Format::Nvfp4Csf),
                Some("csf_32x2_v1") => Ok(Format::Nvfp4Csf32),
                _ => Err(bad(format!(
                    "modelopt weight_scale_encoding {e} for {module} is not one this reader knows"
                ))),
            },
        },
        "W4A16_NVFP4" if group.is_none_or(|g| g == 16) => Ok(Format::Nvfp4),
        "MXFP8"
            if group == Some(32)
                || block_of(entry.get("weight_block_size"))
                    == Some(Format::Fp8Block { rows: 1, cols: 32 }) =>
        {
            Ok(Format::Mxfp8)
        }
        "FP8" => Ok(Format::Fp8Tensor),
        "FP8_BLOCK_SCALES" if group.is_none_or(|g| g == 128) => Ok(Format::Fp8Block {
            rows: 128,
            cols: 128,
        }),
        "FP8_PB_WO" => block_of(entry.get("weight_block_size")).ok_or_else(refuse),
        _ => Err(refuse()),
    }
}

impl Other {
    fn fp8(q: &Value) -> Result<Other, Error> {
        let refuse = |what: &str| bad(format!("fp8 {what} is not one this reader places"));
        for (key, kind) in [
            ("fmt", Value::is_string as fn(&Value) -> bool),
            ("weight_per_tensor", Value::is_boolean),
            ("expert_dtype", Value::is_string),
            ("store_dtype", Value::is_string),
            ("mxfp4_block_size", Value::is_u64),
        ] {
            if q.get(key).is_some_and(|v| !v.is_null() && !kind(v)) {
                return Err(refuse(key));
            }
        }
        if let Some(f) = q
            .get("fmt")
            .and_then(Value::as_str)
            .filter(|f| *f != "e4m3")
        {
            return Err(refuse(&format!("fmt {f}")));
        }
        if q.get("weight_per_tensor").and_then(Value::as_bool) == Some(true) {
            return Err(refuse("weight_per_tensor"));
        }
        let block =
            block_of(q.get("weight_block_size")).ok_or_else(|| refuse("weight_block_size"))?;
        let mxfp4 = match (
            q.get("expert_dtype").and_then(Value::as_str),
            q.get("store_dtype").and_then(Value::as_str),
        ) {
            (None, None) => false,
            (Some("fp4"), None) | (None, Some("mxfp4")) => true,
            (e, s) => return Err(refuse(&format!("expert_dtype {e:?} store_dtype {s:?}"))),
        };
        if q.get("mxfp4_block_size")
            .and_then(Value::as_u64)
            .is_some_and(|b| b != 32)
        {
            return Err(refuse("mxfp4_block_size"));
        }
        let list = |key: &str| q.get(key).map_or(Ok(vec![]), |v| strings(v, key));
        let mut not_convert = list("modules_to_not_convert")?;
        not_convert.extend(list("ignored_layers")?);
        let convert = list("modules_to_convert")?;
        // transformers matches a pattern as a regex from the start or a suffix; these are taken
        // literally, so a pattern with another regex character is refused.
        if let Some(p) = not_convert
            .iter()
            .chain(&convert)
            .find(|p| p.contains(|c: char| "*?[]()|+^$\\{}".contains(c)))
        {
            return Err(refuse(&format!("module pattern {p}")));
        }
        Ok(Other::Fp8 {
            block,
            mxfp4,
            not_convert,
            convert,
        })
    }

    fn compressed_tensors(q: &Value) -> Result<Other, Error> {
        let groups = q
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("compressed-tensors has no config_groups object"))?;
        let mut out = Vec::new();
        for (name, g) in groups {
            let format = g
                .get("format")
                .or_else(|| q.get("format"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let w = g
                .get("weights")
                .ok_or_else(|| bad(format!("compressed-tensors {name} has no weights")))?;
            let field = |k: &str| w.get(k).cloned().unwrap_or(Value::Null);
            let f = match (
                format,
                field("num_bits").as_u64(),
                field("type").as_str(),
                field("strategy").as_str(),
            ) {
                ("float-quantized", Some(8), Some("float"), Some("channel")) => {
                    Some(Format::Fp8Channel)
                }
                ("float-quantized", Some(8), Some("float"), Some("tensor")) => {
                    Some(Format::Fp8Tensor)
                }
                ("float-quantized", Some(8), Some("float"), Some("block")) => {
                    block_of(w.get("block_structure"))
                }
                ("pack-quantized", Some(4), Some("int"), Some("group"))
                    if field("symmetric") == Value::Bool(true) =>
                {
                    field("group_size")
                        .as_u64()
                        .filter(|&g| g > 0)
                        .map(|g| Format::Int4Pack { group: g as usize })
                }
                _ => None,
            };
            let f = f.ok_or_else(|| bad(format!("compressed-tensors {name}: {format} weights {w} are not a format this reader places")))?;
            let targets = strings(
                g.get("targets").unwrap_or(&Value::Null),
                "compressed-tensors targets",
            )?;
            out.push((
                targets
                    .iter()
                    .map(|t| Target::parse(t))
                    .collect::<Result<_, _>>()?,
                f,
            ));
        }
        let ignore = q
            .get("ignore")
            .map_or(Ok(vec![]), |v| strings(v, "compressed-tensors ignore"))?;
        Ok(Other::CompressedTensors {
            groups: out,
            ignore: ignore
                .iter()
                .map(|t| Target::parse(t))
                .collect::<Result<_, _>>()?,
        })
    }
}

/// A weight's place in the checkpoint: its format and the companion tensors it reads.
#[derive(Debug, PartialEq)]
pub struct Placed {
    pub format: Format,
    /// Scales and the packed shape, by name; `input_scale` is listed in [`Plan::input_scales`].
    pub companions: Vec<String>,
}

/// Every tensor of a checkpoint placed or refused.
#[derive(Debug, Default)]
pub struct Plan {
    /// Each weight and each other placed tensor by name; companions are under their weight.
    pub placed: BTreeMap<String, Placed>,
    /// Each refused tensor by name, and why.
    pub refused: BTreeMap<String, String>,
    /// The `input_scale` tensors of placed weights, not read.
    pub input_scales: Vec<String>,
}

/// The companion suffixes a weight's format reads.
const COMPANIONS: [&str; 7] = [
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "scale",
    "input_scale",
    "weight_shape",
    "weight_zero_point",
];

/// Every tensor of `tensors` (name, dtype, shape) placed by `config`.
pub fn plan<'a>(
    config: &Config,
    tensors: impl IntoIterator<Item = (&'a str, Dtype, &'a [usize])>,
) -> Plan {
    let table: BTreeMap<&str, (Dtype, &[usize])> =
        tensors.into_iter().map(|(n, d, s)| (n, (d, s))).collect();
    let mut out = Plan::default();
    let mut claimed = BTreeSet::new();
    for (&name, &(dtype, shape)) in &table {
        let Some(module) = name
            .strip_suffix(".weight")
            .or_else(|| name.strip_suffix(".weight_packed"))
        else {
            continue;
        };
        if plain(dtype).is_some() {
            continue;
        }
        let (formats, owner) = config.declared(module);
        let found = formats
            .iter()
            .find_map(|&f| layout(f, name, module, owner, dtype, shape, &table).map(|c| (f, c)));
        match found {
            Some((format, companions)) => {
                claimed.extend(companions.iter().cloned());
                let (inputs, companions): (Vec<_>, Vec<_>) = companions
                    .into_iter()
                    .partition(|c| c.ends_with(".input_scale"));
                out.input_scales.extend(inputs);
                out.placed
                    .insert(name.to_string(), Placed { format, companions });
            }
            None if formats.is_empty() => {
                out.refused.insert(
                    name.to_string(),
                    format!(
                        "{} {shape:?}, but the config leaves {module} unquantized",
                        dtype.name()
                    ),
                );
            }
            None => {
                let have: Vec<String> = COMPANIONS
                    .iter()
                    .filter_map(|s| table.get_key_value(format!("{module}.{s}").as_str()))
                    .map(|(n, (d, s))| format!("{n} {} {s:?}", d.name()))
                    .collect();
                let want: Vec<String> = formats.iter().map(Format::to_string).collect();
                out.refused.insert(
                    name.to_string(),
                    format!("{} {shape:?} with {have:?} fits none of {want:?}, the config's formats for {module}", dtype.name()),
                );
            }
        }
    }
    for (&name, &(dtype, _)) in &table {
        if out.placed.contains_key(name) || out.refused.contains_key(name) || claimed.contains(name)
        {
            continue;
        }
        let companion = name
            .rsplit_once('.')
            .is_some_and(|(_, last)| COMPANIONS.contains(&last))
            || name.contains(".weight_scale.");
        let why = if companion {
            "a companion no placed weight reads".to_string()
        } else if let Some(f) = plain(dtype) {
            out.placed.insert(
                name.to_string(),
                Placed {
                    format: f,
                    companions: vec![],
                },
            );
            continue;
        } else {
            format!("{} outside a quantized weight", dtype.name())
        };
        out.refused.insert(name.to_string(), why);
    }
    out.input_scales.sort();
    out.input_scales.dedup();
    out
}

fn plain(dtype: Dtype) -> Option<Format> {
    match dtype {
        Dtype::Bf16 => Some(Format::Bf16),
        Dtype::F32 => Some(Format::F32),
        _ => None,
    }
}

/// The companions the weight `name` of `module` reads when stored as `format`, or `None` when
/// its tensors do not fit. `owner` is the module a tensor scale may be shared from.
fn layout(
    format: Format,
    name: &str,
    module: &str,
    owner: Option<&str>,
    dtype: Dtype,
    shape: &[usize],
    table: &BTreeMap<&str, (Dtype, &[usize])>,
) -> Option<Vec<String>> {
    let &[rows, cols] = shape else { return None };
    // The first of `suffixes` beside the weight (or, for a shared scale, beside `owner`) that is
    // one of `dtypes` and of `want`'s shape.
    let find = |at: &str, suffixes: &[&str], dtypes: &[Dtype], want: &dyn Fn(&[usize]) -> bool| {
        suffixes.iter().find_map(|s| {
            let n = format!("{at}.{s}");
            let &(d, sh) = table.get(n.as_str())?;
            (dtypes.contains(&d) && want(sh)).then_some(n)
        })
    };
    let scalar = |s: &[usize]| s.is_empty() || s == [1];
    let input = |out: &mut Vec<String>| {
        out.extend(find(
            module,
            &["input_scale"],
            &[Dtype::F32, Dtype::Bf16],
            &scalar,
        ))
    };
    let mut out = Vec::new();
    match format {
        Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 => {
            let k = cols.checked_mul(2)?;
            if dtype != Dtype::U8 || k % 16 != 0 {
                return None;
            }
            let c = k / 16;
            let line = |s: &[usize]| s.len() == 1;
            match format {
                Format::Nvfp4 => {
                    out.push(find(module, &["weight_scale"], &[Dtype::F8E4M3], &|s| {
                        s == [rows, c]
                    })?)
                }
                Format::Nvfp4Csf => {
                    if rows % 16 != 0 || c % 2 != 0 {
                        return None;
                    }
                    let fixed = [rows / 16, 16 * (1 + c / 2)];
                    out.push(find(
                        module,
                        &["weight_scale.nvfp4_csf_fixed"],
                        &[Dtype::U8],
                        &|s| s == fixed,
                    )?);
                    out.push(find(
                        module,
                        &["weight_scale.nvfp4_csf_exceptions"],
                        &[Dtype::U32],
                        &line,
                    )?);
                }
                _ => {
                    let codes = rows.checked_mul(c)?.div_ceil(2);
                    for (part, dtype, want) in [
                        ("bases", Dtype::U8, Some(rows)),
                        ("codes", Dtype::U8, Some(codes)),
                        ("page_bases", Dtype::U32, None),
                        ("offsets", Dtype::U16, None),
                        ("exceptions", Dtype::U16, None),
                    ] {
                        let suffix = format!("weight_scale.csf32_{part}");
                        out.push(find(module, &[suffix.as_str()], &[dtype], &|s| {
                            line(s) && want.is_none_or(|n| s == [n])
                        })?);
                    }
                }
            }
            let own = find(module, &["weight_scale_2"], &[Dtype::F32], &scalar);
            out.push(own.or_else(|| find(owner?, &["weight_scale_2"], &[Dtype::F32], &scalar))?);
            input(&mut out);
        }
        Format::Mxfp8 => {
            if dtype != Dtype::F8E4M3 || cols % 32 != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale"],
                &[Dtype::U8, Dtype::F8E8M0],
                &|s| s == [rows, cols / 32],
            )?);
            input(&mut out);
        }
        Format::Mxfp4 => {
            let k = cols.checked_mul(2)?;
            if !matches!(dtype, Dtype::U8 | Dtype::I8) || k % 32 != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale", "scale"],
                &[Dtype::U8, Dtype::F8E8M0],
                &|s| s == [rows, k / 32],
            )?);
        }
        Format::Fp8Block { rows: a, cols: b } if dtype == Dtype::F8E4M3 => {
            let (r, c) = (rows.div_ceil(a), cols.div_ceil(b));
            let dtypes = [Dtype::F32, Dtype::Bf16, Dtype::F8E8M0];
            // ModelOpt's FP8_PB_WO keeps the block grid's unit axes: [r, 1, c, 1].
            out.push(find(
                module,
                &["weight_scale_inv", "weight_scale", "scale"],
                &dtypes,
                &|s| s == [r, c] || s == [r, 1, c, 1],
            )?);
            input(&mut out);
        }
        Format::Fp8Channel if dtype == Dtype::F8E4M3 => {
            let dtypes = [Dtype::F32, Dtype::Bf16];
            out.push(find(module, &["weight_scale"], &dtypes, &|s| {
                s == [rows] || s == [rows, 1]
            })?);
            input(&mut out);
        }
        Format::Fp8Tensor if dtype == Dtype::F8E4M3 => {
            let dtypes = [Dtype::F32, Dtype::Bf16];
            let own = find(module, &["weight_scale"], &dtypes, &scalar);
            out.push(own.or_else(|| find(owner?, &["weight_scale"], &dtypes, &scalar))?);
            input(&mut out);
        }
        Format::Int4Pack { group } => {
            let k = cols.checked_mul(8)?;
            if dtype != Dtype::I32 || !name.ends_with(".weight_packed") || k % group != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale"],
                &[Dtype::Bf16, Dtype::F16, Dtype::F32],
                &|s| s == [rows, k / group],
            )?);
            out.push(find(
                module,
                &["weight_shape"],
                &[Dtype::I64, Dtype::I32],
                &|s| s == [2],
            )?);
            if table.contains_key(format!("{module}.weight_zero_point").as_str()) {
                return None;
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Whether `module` is `key` or inside it.
fn within(module: &str, key: &str) -> bool {
    module
        .strip_prefix(key)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
}

/// fnmatch's `*` (any run) and `?` (one character) over the whole of `s`.
fn glob(p: &str, s: &str) -> bool {
    let (p, s): (Vec<char>, Vec<char>) = (p.chars().collect(), s.chars().collect());
    let (mut i, mut j, mut star, mut mark) = (0, 0, None, 0);
    while j < s.len() {
        if i < p.len() && (p[i] == '?' || p[i] == s[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(at) = star {
            i = at + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    p[i..].iter().all(|&c| c == '*')
}

fn str_of<'v>(v: &'v Value, key: &str) -> Result<&'v str, Error> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("{key} is not a string")))
}

fn strings(v: &Value, what: &str) -> Result<Vec<String>, Error> {
    v.as_array()
        .and_then(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect())
        .ok_or_else(|| bad(format!("{what} is not a list of strings")))
}

/// A `[rows, cols]` block size.
fn block_of(v: Option<&Value>) -> Option<Format> {
    let b = v?.as_array()?;
    match b.as_slice() {
        [r, c] => {
            let (rows, cols) = (
                r.as_u64().filter(|&r| r > 0)?,
                c.as_u64().filter(|&c| c > 0)?,
            );
            Some(Format::Fp8Block {
                rows: rows as usize,
                cols: cols as usize,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    /// A config fixture, gunzipped.
    fn fixture(name: &str) -> Option<Value> {
        let path = format!(
            "{}/tests/fixtures/quant-configs/{name}.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let gz = match std::fs::read(&path) {
            Ok(gz) => gz,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => panic!("{path}: {e}"),
        };
        let mut text = Vec::new();
        flate2::read::GzDecoder::new(gz.as_slice())
            .read_to_end(&mut text)
            .unwrap();
        Some(serde_json::from_slice(&text).unwrap())
    }

    /// The config of the checkpoint whose fixtures start with `tag`.
    fn config(tag: &str) -> Result<Config, Error> {
        let config = fixture(&format!("{tag}__config.json")).unwrap();
        Config::parse(
            &config,
            fixture(&format!("{tag}__hf_quant_config.json")).as_ref(),
        )
    }

    const B128: Format = Format::Fp8Block {
        rows: 128,
        cols: 128,
    };

    /// Each checkpoint config declares its modules' formats or is refused by producer or algorithm.
    #[test]
    fn checkpoint_configs_declare_formats_or_refuse_by_name() {
        let lm = "model.language_model.layers";
        type Modules<'a> = &'a [(&'a str, &'a [Format])];
        let cases: &[(&str, Modules)] = &[
            (
                "nvidia__GLM-5.3-Flash-NVFP4",
                &[
                    (&format!("{lm}.3.mlp.experts.0.gate_proj"), &[Format::Nvfp4]),
                    (&format!("{lm}.0.mlp.down_proj"), &[Format::Nvfp4]),
                    (&format!("{lm}.3.self_attn.q_a_proj"), &[]),
                    (&format!("{lm}.3.mlp.shared_experts.up_proj"), &[]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "nvidia__Qwen3.8-Flash-Next-NVFP4",
                &[
                    (&format!("{lm}.0.mlp.experts.7.up_proj"), &[Format::Nvfp4]),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding.shard_0"),
                        &[Format::Fp8Tensor],
                    ),
                    ("mtp.layers.0.mlp.experts.0.gate_proj", &[B128]),
                    (&format!("{lm}.0.linear_attn.in_proj_qkvz"), &[]),
                ],
            ),
            (
                "nvidia__GLM-5.3-Flash-FP4-Local",
                &[
                    (
                        &format!("{lm}.3.mlp.experts.0.gate_proj"),
                        &[Format::Nvfp4Csf32],
                    ),
                    (
                        &format!("{lm}.45.mlp.experts.0.gate_proj"),
                        &[Format::Nvfp4],
                    ),
                    (&format!("{lm}.0.self_attn.q_proj"), &[Format::Mxfp8]),
                    (&format!("{lm}.3.self_attn.q_a_proj"), &[B128]),
                    (&format!("{lm}.3.mlp.shared_experts.up_proj"), &[B128]),
                    (&format!("{lm}.3.self_attn.kv_b_proj"), &[]),
                    ("model.visual.blocks.0.attn.qkv", &[Format::Mxfp8]),
                    ("model.visual.blocks.0.mlp.down_proj", &[Format::Nvfp4]),
                    ("lm_head", &[Format::Mxfp8]),
                ],
            ),
            (
                "nvidia__Qwen3.8-Flash-Next-FP4-Local",
                &[
                    (
                        &format!("{lm}.0.mlp.experts.7.up_proj"),
                        &[Format::Nvfp4Csf],
                    ),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding.shard_0"),
                        &[Format::Nvfp4],
                    ),
                    (&format!("{lm}.0.linear_attn.in_proj_qkv"), &[Format::Mxfp8]),
                    ("mtp.layers.0.mlp.experts.0.gate_proj", &[Format::Nvfp4]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "nvidia__NVIDIA-Nemotron-3-Nano-4B-FP8",
                &[
                    ("backbone.layers.0.mixer.in_proj", &[Format::Fp8Tensor]),
                    ("backbone.layers.11.mixer.in_proj", &[]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "zai-org__GLM-5.3-Flash",
                &[
                    (&format!("{lm}.29.mlp.experts.78.down_proj"), &[B128]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "Qwen__Qwen3.8-Flash-Next-FP8",
                &[
                    (&format!("{lm}.3.mlp.experts.0.down_proj"), &[B128]),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding"),
                        &[B128],
                    ),
                    ("lm_head", &[]),
                ],
            ),
            (
                "deepseek-ai__DeepSeek-V4.1-Flash",
                &[(
                    "layers.21.ffn.experts.0.w1",
                    &[Format::Fp8Block { rows: 32, cols: 32 }, Format::Mxfp4],
                )],
            ),
            (
                "openbmb__MiniCPM-V-4.7-35B-A3B-FP8",
                &[
                    (
                        &format!("{lm}.20.mlp.experts.148.down_proj"),
                        &[Format::Fp8Channel],
                    ),
                    ("model.vision_tower.encoder.layers.0.self_attn.k_proj", &[]),
                ],
            ),
            (
                "openbmb__MiniCPM-V-4.7-35B-A3B-int4",
                &[
                    (
                        &format!("{lm}.36.mlp.experts.212.down_proj"),
                        &[Format::Int4Pack { group: 128 }],
                    ),
                    ("lm_head", &[]),
                ],
            ),
        ];
        for (tag, probes) in cases {
            let c = config(tag).unwrap_or_else(|e| panic!("{tag}: {e}"));
            for (module, want) in *probes {
                assert_eq!(c.declared(module).0, *want, "{tag}: {module}");
            }
        }
        let e = config("nvidia__NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB").unwrap_err();
        assert!(
            e.to_string().contains(
                "quant_algo IQ2_XXS for language_model.backbone.layers.1.mixer.experts.0.up_proj \
                 is not a format"
            ),
            "{e}"
        );
        for (entry, want) in [
            (
                serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_64" }),
                "weight_scale_encoding \"csf_64\" for m is not one this reader knows",
            ),
            (
                serde_json::json!({ "quant_algo": "MXFP8", "group_size": 32, "weight_scale_encoding": "csf" }),
                "quant_algo MXFP8 for m: weight_scale_encoding is NVFP4's alone",
            ),
            (
                serde_json::json!({ "quant_algo": "MXFP8", "group_size": 16 }),
                "quant_algo MXFP8 for m is not a format",
            ),
            (
                serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16, "zzz": 1 }),
                "modelopt quantized_layers m.zzz is not a key this converter reads",
            ),
        ] {
            let q = serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION", "quantized_layers": { "m": entry } } });
            let e = Config::parse(&serde_json::json!({}), Some(&q)).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        }
        let int4 = |group: Value| {
            serde_json::json!({ "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": { "g": { "format": "pack-quantized", "targets": ["Linear"],
                    "weights": { "num_bits": 4, "type": "int", "strategy": "group",
                        "group_size": group, "symmetric": true } } } } })
        };
        let fp8 = |key: &str, v: Value| {
            let mut q =
                serde_json::json!({ "quant_method": "fp8", "weight_block_size": [128, 128] });
            q[key] = v;
            serde_json::json!({ "quantization_config": q })
        };
        let modelopt = |group: Value| {
            serde_json::json!({ "quantization_config": { "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION", "quantized_layers": {
                    "m": { "quant_algo": "NVFP4", "group_size": group } } } })
        };
        for (config, want) in [
            (int4(0.into()), "are not a format this reader places"),
            (
                modelopt("16".into()),
                "group_size \"16\" for m is not a count",
            ),
            (
                fp8("fmt", 4.into()),
                "fp8 fmt is not one this reader places",
            ),
            (
                fp8("weight_per_tensor", "no".into()),
                "fp8 weight_per_tensor is not",
            ),
            (fp8("expert_dtype", 4.into()), "fp8 expert_dtype is not"),
            (
                fp8("mxfp4_block_size", "32".into()),
                "fp8 mxfp4_block_size is not",
            ),
            (
                serde_json::json!({ "quantization_config": { "quant_method": "modelopt",
                    "quant_algo": "FP8", "exclude_modules": ["lm_hea[d]"] } }),
                "modelopt exclude pattern lm_hea[d]: a [ class",
            ),
            (
                serde_json::json!({ "quantization_config": { "quant_method": "quark" } }),
                "quant config: Quark checkpoints are not read yet",
            ),
        ] {
            let e = Config::parse(&config, None).unwrap_err();
            assert!(e.to_string().contains(want), "{config}: {e}");
        }
        let unknown =
            serde_json::json!({ "quantization_config": { "quant_method": "gptq", "bits": 4 } });
        let e = Config::parse(&unknown, None).unwrap_err();
        assert!(
            e.to_string()
                .contains("quant_method gptq is not one this reader knows"),
            "{e}"
        );
    }

    /// Every tensor of a table gets one format or a refusal naming it: each format's layout is
    /// matched with its companions, a tensor scale is shared from the module the config names,
    /// input scales are reported, and a scale with no weight, a quantized weight the config leaves
    /// unquantized and a dtype outside every format are refused.
    #[test]
    fn every_tensor_gets_one_format_or_a_refusal_by_name() {
        use Dtype::*;
        let c = Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": null,
                "quantized_layers": {
                    "e": { "quant_algo": "NVFP4", "group_size": 16 },
                    "ple": { "quant_algo": "FP8" },
                    "mtp": { "quant_algo": "FP8_BLOCK_SCALES", "group_size": 128 },
                },
                "exclude_modules": ["attn*"],
            }})),
        )
        .unwrap();
        let t: &[(&str, Dtype, &[usize])] = &[
            ("e.0.weight", U8, &[256, 64]),
            ("e.0.weight_scale", F8E4M3, &[256, 8]),
            ("e.0.weight_scale_2", F32, &[]),
            ("e.0.input_scale", F32, &[]),
            ("ple.shard_0.weight", F8E4M3, &[10, 4]),
            ("ple.shard_1.weight", F8E4M3, &[10, 4]),
            ("ple.weight_scale", Bf16, &[1]),
            ("mtp.0.weight", F8E4M3, &[640, 2560]),
            ("mtp.0.weight_scale_inv", Bf16, &[5, 20]),
            ("attn.q.weight", F8E4M3, &[8, 8]),
            ("attn.q.weight_scale", F32, &[]),
            ("norm.weight", Bf16, &[8]),
            ("A_log", F32, &[8]),
            ("lone.weight_scale_2", F32, &[]),
            ("e.1.weight", U8, &[256, 64]),
            ("e.1.weight_scale", F8E4M3, &[256, 4]),
            ("ple.ids", I64, &[3]),
            ("e.2.weight", U8, &[0, 9_223_372_036_854_775_840]),
        ];
        let p = plan(&c, t.iter().copied());
        let got = |n: &str| p.placed.get(n).map(|pl| (pl.format, pl.companions.clone()));
        assert_eq!(
            got("e.0.weight").unwrap(),
            (
                Format::Nvfp4,
                vec!["e.0.weight_scale".into(), "e.0.weight_scale_2".into()]
            )
        );
        assert_eq!(
            got("ple.shard_1.weight").unwrap(),
            (Format::Fp8Tensor, vec!["ple.weight_scale".into()])
        );
        assert_eq!(
            got("mtp.0.weight").unwrap(),
            (B128, vec!["mtp.0.weight_scale_inv".into()])
        );
        assert_eq!(got("norm.weight").unwrap().0, Format::Bf16);
        assert_eq!(got("A_log").unwrap().0, Format::F32);
        assert_eq!(p.input_scales, ["e.0.input_scale"]);
        let refused = |n: &str, why: &str| assert!(p.refused[n].contains(why), "{n}: {p:?}");
        refused("attn.q.weight", "the config leaves attn.q unquantized");
        refused("attn.q.weight_scale", "a companion no placed weight reads");
        refused("lone.weight_scale_2", "a companion no placed weight reads");
        refused("e.1.weight", "fits none of [\"NVFP4-ModelOpt\"]");
        refused("ple.ids", "I64 outside a quantized weight");
        refused("e.2.weight", "fits none of [\"NVFP4-ModelOpt\"]");
        assert_eq!(p.placed.len() + p.refused.len() + 5, t.len());

        let ct = Config::parse(
            &serde_json::json!({ "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": { "g": { "format": "pack-quantized", "targets": ["Linear"],
                    "weights": { "num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "symmetric": true } } },
                "ignore": ["lm_head", "re:.*\\.mlp\\.gate$"],
            }}),
            None,
        )
        .unwrap();
        let t: &[(&str, Dtype, &[usize])] = &[
            ("x.weight_packed", I32, &[2048, 64]),
            ("x.weight_scale", Bf16, &[2048, 4]),
            ("x.weight_shape", I64, &[2]),
            ("l.mlp.gate.weight_packed", I32, &[8, 64]),
            ("y.weight_packed", I32, &[0, usize::MAX / 4]),
        ];
        let p = plan(&ct, t.iter().copied());
        assert_eq!(
            p.placed["x.weight_packed"].format,
            Format::Int4Pack { group: 128 }
        );
        assert!(p.refused["l.mlp.gate.weight_packed"].contains("leaves l.mlp.gate unquantized"));
        assert!(p.refused["y.weight_packed"].contains("fits none of"));
        let wide: &[usize] = &[0, usize::MAX / 2 + 1];
        for (format, shape) in [
            (Format::Mxfp4, wide),
            (Format::Nvfp4Csf32, &[usize::MAX / 2 + 1, 16]),
        ] {
            assert_eq!(
                layout(format, "m.weight", "m", None, U8, shape, &BTreeMap::new()),
                None
            );
        }
    }

    /// Real headers: every tensor of nvidia/GLM-5.3-Flash-NVFP4's layers 0 (dense) and 3 (288
    /// experts) and its unlayered tensors under its config, and a deepseek-ai/DeepSeek-V4.1-Flash
    /// shard under its own, each tensor placed and none refused.
    #[test]
    fn real_checkpoint_headers_place_under_their_configs() {
        let header = |name: &str| -> serde_json::Map<String, Value> {
            let gz = std::fs::read(format!(
                "{}/tests/fixtures/checkpoint-headers/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let mut text = Vec::new();
            flate2::read::GzDecoder::new(gz.as_slice())
                .read_to_end(&mut text)
                .unwrap();
            let mut h: serde_json::Map<String, Value> = serde_json::from_slice(&text).unwrap();
            h.remove("__metadata__");
            h
        };
        let place = |tag: &str, files: &[&str]| {
            let heads: Vec<_> = files.iter().map(|f| header(f)).collect();
            let entries: Vec<(String, Dtype, Vec<usize>)> = heads
                .iter()
                .flatten()
                .map(|(n, e)| {
                    let shape = e["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize)
                        .collect();
                    (
                        n.clone(),
                        Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                        shape,
                    )
                })
                .collect();
            let p = plan(
                &config(tag).unwrap(),
                entries
                    .iter()
                    .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            );
            let mut count = BTreeMap::new();
            for pl in p.placed.values() {
                *count.entry(pl.format.to_string()).or_insert(0) += 1;
            }
            (count.into_iter().collect::<Vec<_>>(), p)
        };
        let (count, p) = place(
            "nvidia__GLM-5.3-Flash-NVFP4",
            &["glm-5.3-flash-nvfp4-layers-0-3.tensors.json.gz"],
        );
        let want = [("BF16", 47), ("F32", 6), ("NVFP4-ModelOpt", 867)];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(p.input_scales.len(), 867);
        // Layers 0 (linear attention, dense MLP), 3 (latent attention, experts) and 4 (linear
        // attention, experts): each expert's five csf32 parts and its two tensor scales are its
        // companions.
        let (count, p) = place(
            "nvidia__GLM-5.3-Flash-FP4-Local",
            &["glm-5.3-flash-fp4-local-layers-0-3-4.tensors.json.gz"],
        );
        let want = [
            ("BF16", 40),
            ("F32", 22),
            ("FP8-block 128x128", 10),
            ("MXFP8", 12),
            ("NVFP4-csf32", 1728),
        ];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(p.input_scales.len(), 1728);
        let expert = &p.placed["model.language_model.layers.3.mlp.experts.0.down_proj.weight"];
        assert_eq!(expert.companions.len(), 6, "{:?}", expert.companions);
        // One of Qwen3.8's 128 PLE parts: its tensor scale is the one the declared table holds.
        let (count, p) = place(
            "nvidia__Qwen3.8-Flash-Next-FP4-Local",
            &["qwen3.8-flash-next-fp4-local-ple-part-0.tensors.json.gz"],
        );
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(count, [("NVFP4-ModelOpt".to_string(), 1)]);
        let ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        assert_eq!(
            p.placed[&format!("{ple}.shard_0.weight")].companions,
            [
                format!("{ple}.shard_0.weight_scale"),
                format!("{ple}.weight_scale_2")
            ]
        );
        let (count, p) = place(
            "deepseek-ai__DeepSeek-V4.1-Flash",
            &["deepseek-v4.1-flash-model-00024-of-00048.header.json.gz"],
        );
        let want = [
            ("BF16", 5),
            ("F32", 9),
            ("FP8-block 32x32", 8),
            ("MXFP4-OCP", 1152),
        ];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
    }

    /// Each format's held bytes and blocks, the stored values plus plain scales: NVFP4 in every
    /// encoding is GGUF's 36 bytes a 64 values and one tensor scale a matrix.
    #[test]
    fn each_format_holds_its_values_and_plain_scales() {
        let expert = [2048, 4096, 288];
        for f in [Format::Nvfp4, Format::Nvfp4Csf, Format::Nvfp4Csf32] {
            assert_eq!(
                f.held(&expert),
                Some(Held {
                    bytes: 2048 * 4096 * 288 / 2 + 2048 * 4096 * 288 / 16,
                    whole_bytes: 4 * 288,
                    block: [64, 1],
                })
            );
        }
        assert_eq!(Format::Nvfp4.held(&[48, 16]), None);
        assert_eq!(
            Format::Mxfp8.held(&[4096, 8192]),
            Some(Held {
                bytes: 4096 * 8192 + 4096 * 8192 / 32,
                whole_bytes: 0,
                block: [32, 1],
            })
        );
        let fp8 = Format::Fp8Block {
            rows: 128,
            cols: 128,
        };
        assert_eq!(
            fp8.held(&[16384, 4096]),
            Some(Held {
                bytes: 16384 * 4096 + 4 * 32 * 128,
                whole_bytes: 0,
                block: [128, 128],
            })
        );
        assert_eq!(
            Format::Bf16.held(&[512, 256, 64]).unwrap().bytes,
            2 * 512 * 256 * 64
        );
        assert_eq!(Format::F32.held(&[64]).unwrap().bytes, 256);
        assert_eq!(Format::Fp8Tensor.held(&[64, 64]), None);
        assert_eq!(Format::Bf16.held(&[0, 8192]), None);
        assert_eq!(Format::Bf16.held(&[64, 0]), None);
        assert_eq!(Format::Bf16.held(&[u64::MAX / 2, 4]), None);
        assert_eq!(Format::Mxfp8.held(&[1 << 40, 1 << 40]), None);
        assert_eq!(Format::Nvfp4Csf32.logical(&[2048, 1024]), [2048, 2048]);
        assert_eq!(Format::Mxfp8.logical(&[2048, 1024]), [2048, 1024]);
    }
}
