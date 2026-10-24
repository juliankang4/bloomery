//! A MiniMax-M3 model at a path as the body loads it: a GGUF split, or a checkpoint folder read as
//! stored (`model_hf::folder::open`, no prepare step), whose card tensors the body reads through the
//! folder's part source (`bloomery_gpu_minimaxm3::body::Folder`). This module is the one owner of that open: the serve seat opens a
//! folder once on its main thread ([`open`]) and loads it on the engine thread ([`model`]), and a gate
//! takes the model at a path in one call ([`model_at`]). A build without the `checkpoint` feature
//! refuses a folder by name.

use std::path::Path;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_minimaxm3::body::Model;
use gguf::Split;
use tokenizer::Tokenizer;

/// What a folder's load reads card tensors through (`model_hf::folder::Opened::source`). A build
/// without the `checkpoint` feature opens no folder, so it has none.
#[cfg(feature = "checkpoint")]
pub type Source = model_hf::part::Checkpoint;
#[cfg(not(feature = "checkpoint"))]
pub type Source = ();

/// A MiniMax checkpoint folder opened for a load: its located header, which plans as a split's first
/// shard does, its tokenizer and its part source.
pub struct Opened {
    pub header: Split,
    pub tokenizer: Tokenizer,
    pub source: Source,
}

/// The checkpoint folder `dir` read as stored: a MiniMax-M3 folder, whose card tensors the body reads
/// through its part source. A folder of another architecture, and one with a row table (which this
/// body does not read), is refused by name.
#[cfg(feature = "checkpoint")]
pub fn open(dir: &Path) -> Result<Opened, GateError> {
    let f = bloomery_gpu_gates::model_file::open_folder(dir)?;
    let arch = refset::arch::minimaxm3::ARCH;
    if f.header.architecture() != Some(arch) {
        return Err(format!(
            "checkpoint folder {}: a {} model, where this seat serves {arch}",
            dir.display(),
            f.header.architecture().unwrap_or("<no architecture>"),
        )
        .into());
    }
    if let Some(name) = f.rows.keys().next() {
        return Err(format!(
            "checkpoint folder {}: the row table {name}, which this seat does not read",
            dir.display()
        )
        .into());
    }
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        source: f.source,
    })
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn open(dir: &Path) -> Result<Opened, GateError> {
    Err(format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        dir.display()
    )
    .into())
}

/// The model of a folder whose located header is `header` and whose part source is `source`.
#[cfg(feature = "checkpoint")]
pub fn model(header: Split, source: Source) -> Result<Model, GateError> {
    Ok(bloomery_gpu_minimaxm3::body::Folder::new(header, source).into())
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn model(_header: Split, _source: Source) -> Result<Model, GateError> {
    Err("a checkpoint folder is opened by a build with the checkpoint feature".into())
}

/// The model at `path` as the body loads it: its GGUF split, or the checkpoint folder a directory is.
pub fn model_at(path: &Path) -> Result<Model, GateError> {
    if path.is_dir() {
        let f = open(path)?;
        return model(f.header, f.source);
    }
    Ok(Split::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .into())
}
