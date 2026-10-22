//! A Qwen3.8 checkpoint folder as a server or a gate opens it: a model path that is a directory is
//! one, read as stored (`model_hf::arch::qwen35moe::folder::open`, no prepare step). A build
//! without the `checkpoint` feature refuses it by name.

use std::path::Path;

use bloomery_gpu::arch::qwen3moe::Folder38;
use bloomery_gpu_gates::GateError;
use gguf::Split;
use tokenizer::Tokenizer;

/// A checkpoint folder opened for a load: its header, which plans as a split's first shard does,
/// its tokenizer, and the load of its parts, which takes the header back once it has planned.
pub struct Opened {
    pub header: Split,
    pub tokenizer: Tokenizer,
    pub has_draft: bool,
    pub parts: Box<dyn FnOnce(Split) -> Folder38>,
}

/// The checkpoint folder `path` opened for a load: a Qwen3.8 checkpoint; any other folder is
/// refused by name.
#[cfg(feature = "checkpoint")]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    let f = model_hf::arch::qwen35moe::folder::open(path)
        .map_err(|e| format!("checkpoint folder {}: {e}", path.display()))?;
    let (source, ple) = (f.source, f.ple);
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        has_draft: f.draft.is_some(),
        parts: Box::new(move |header| Folder38::new(header, source, ple)),
    })
}

/// The checkpoint folder's MTP header, borrowing its target's embedding and head.
#[cfg(feature = "checkpoint")]
pub fn draft(path: &Path) -> Result<Split, GateError> {
    model_hf::arch::qwen35moe::folder::open(path)?
        .draft
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
