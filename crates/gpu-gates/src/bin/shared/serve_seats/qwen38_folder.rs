//! A Qwen3.8 checkpoint folder as a server or a gate opens it: a model path that is a directory is
//! one, read as stored (`model_hf::folder::open`, no prepare step). A build without the
//! `checkpoint` feature refuses it by name.

use std::path::Path;

use bloomery_gpu::arch::qwen3moe::Folder38;
use bloomery_gpu_gates::GateError;
use gguf::Split;
use tokenizer::Tokenizer;

/// A checkpoint folder opened for a load: its header, which plans as a split's first shard does,
/// its tokenizer, its MTP layer's header (`None` for a folder without one, or why the folder's
/// preparation refuses it), and the load of its parts, which takes the header back once it has
/// planned.
pub struct Opened {
    pub header: Split,
    pub tokenizer: Tokenizer,
    pub draft: Result<Option<Split>, String>,
    pub parts: Box<dyn FnOnce(Split) -> Folder38>,
}

/// The checkpoint folder `path` opened for a load: a Qwen3.8 checkpoint; a folder of any other
/// architecture is refused by name.
#[cfg(feature = "checkpoint")]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    let mut f = bloomery_gpu_gates::model_file::open_folder(path)?;
    if f.header.architecture() != Some("qwen4exp") {
        return Err(format!(
            "checkpoint folder {}: a {} model, where this seat serves qwen4exp",
            path.display(),
            f.header.architecture().unwrap_or("<no architecture>")
        )
        .into());
    }
    let ple = f
        .rows
        .remove(&model::arch::qwen35moe::names::per_layer_token_embd());
    if let Some(name) = f.rows.keys().next() {
        return Err(format!(
            "checkpoint folder {}: the row table {name}, which this seat does not read",
            path.display()
        )
        .into());
    }
    let source = f.source;
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        draft: f.draft.map_err(|e| e.to_string()),
        parts: Box::new(move |header| Folder38::with(header, source, ple)),
    })
}

/// The checkpoint folder's MTP header, borrowing its target's embedding and head; a folder
/// without an MTP layer, and one whose layer the preparation refuses, are refused by name.
#[cfg(feature = "checkpoint")]
pub fn draft(path: &Path) -> Result<Split, GateError> {
    open(path)?
        .draft
        .map_err(|e| format!("checkpoint folder {}: its MTP layer: {e}", path.display()))?
        .ok_or_else(|| format!("checkpoint folder {} holds no MTP layer", path.display()).into())
}

/// A checkpoint folder's draft in a build without checkpoint support, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn draft(path: &Path) -> Result<Split, GateError> {
    open(path).map(|f| f.header)
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    Err(format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        path.display()
    )
    .into())
}
