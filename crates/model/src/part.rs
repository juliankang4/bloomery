//! The tensors a load reads, each whole or the part a rank holds, from a source ([`Source`]): a
//! GGUF file here, a Hugging Face checkpoint folder in `model_hf::part`. The load gives one rule
//! ([`Source::plan`]): for a tensor's name, card type and dims, whether it is read and which part
//! ([`Take`]). A source knows no model and no rank count. Every part is planned from headers
//! alone before any value is read; [`Source::read`] then gives the part's bytes in its card type
//! and the tensor scales it reads.
//!
//! A GGUF file holds its tensors in their card types: a part is runs of its rows' bytes
//! ([`part_of`]), and a scaled tensor's scales are the `.scale` beside it, one an expert along a
//! cut of a stack's experts.

use std::ops::Range;
use std::path::PathBuf;

use gguf::{GgmlType, Split, TensorInfo};

use crate::placement::tp::{TpError, part_of};

/// Why a part was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{name}: {what}")]
    Tensor { name: String, what: String },
    #[error(transparent)]
    Cut(#[from] TpError),
    #[error(transparent)]
    Load(#[from] gguf::LoadError),
}

/// The refusal of tensor `name` for `what`.
pub fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// The indices of a tensor a part holds: a dim and the range along it; `None` the whole tensor.
pub type At = Option<(usize, Range<u64>)>;

/// What a load reads of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Take {
    Skip,
    Whole,
    /// The indices `.1` along dim `.0`.
    Part(usize, Range<u64>),
}

/// What `rule` takes of tensor `t`: `None` skipped, else the indices it reads; a refusal of
/// `rule` names the tensor.
pub fn take<E: std::fmt::Display>(
    rule: &impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    t: &TensorInfo,
) -> Result<Option<At>, Error> {
    Ok(
        match rule(&t.name, t.ty, &t.dims).map_err(|e| refuse(&t.name, e.to_string()))? {
            Take::Skip => None,
            Take::Whole => Some(None),
            Take::Part(dim, at) => Some(Some((dim, at))),
        },
    )
}

/// One part a load reads ([`Source::plan`]): the indices it holds and its header entry (name,
/// card type, dims and bytes).
#[derive(Clone, Debug)]
pub struct Planned {
    pub at: At,
    pub info: TensorInfo,
}

/// A part's bytes in its card type ([`Source::read`]), with its header entry and the tensor
/// scales its type reads.
pub struct Part<'s> {
    pub info: TensorInfo,
    bytes: Bytes<'s>,
    pub scale: Option<Vec<f32>>,
}

enum Bytes<'s> {
    /// Runs of a mapped file's rows.
    File(Vec<&'s [u8]>),
    /// Bytes the source made in memory of its own.
    Made(Vec<u8>),
}

impl Part<'_> {
    /// A part whose bytes `bytes` the source made in memory of its own.
    #[must_use]
    pub fn made(info: TensorInfo, bytes: Vec<u8>, scale: Option<Vec<f32>>) -> Part<'static> {
        Part {
            info,
            bytes: Bytes::Made(bytes),
            scale,
        }
    }

    /// The part's rows' bytes, runs in row order.
    #[must_use]
    pub fn spans(&self) -> Vec<&[u8]> {
        match &self.bytes {
            Bytes::File(runs) => runs.clone(),
            Bytes::Made(bytes) => vec![bytes],
        }
    }
}

/// One plane of a row-gathered table, in row order: each run holds whole
/// rows of `row_bytes` bytes where the source stores them.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredPlane {
    pub row_bytes: u64,
    pub runs: Vec<(PathBuf, Range<u64>)>,
}

/// A row-gathered table where its source stores it: F8 values, or NVFP4
/// codes with a separate plane of block scales, at one tensor `scale`.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRows {
    pub values: StoredPlane,
    pub block_scales: Option<StoredPlane>,
    pub scale: f32,
}

/// Where a load reads its tensors.
pub trait Source {
    /// Why the source refused a part.
    type Error: std::error::Error;

    /// Every tensor of the source `rule` takes, each at the part it takes ([`Take`]) given the
    /// tensor's name, card type and dims, planned from headers alone: a refusal of `rule` and a
    /// part off the tensor's blocks or past its dims are refused by name before any value is
    /// read.
    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Self::Error>;

    /// The bytes of the part `p` planned ([`Source::plan`]) and the tensor scales its type reads;
    /// a tensor no longer in the source is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Self::Error>;

    /// Drop the file pages the source read part `p` from, once a load has copied its bytes and no
    /// later reader takes them from the file. A source whose loads release their own pages (a
    /// split's placed load, `PageDrop`) drops none here.
    fn release(&self, _p: &Planned) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A source whose plan considers only the names `keep` selects. Reads and page release use
/// the underlying source, so several model components can share its stored tensors.
pub struct Filtered<'s, S, F> {
    pub source: &'s S,
    pub keep: F,
}

impl<S: Source, F: Fn(&str) -> bool> Source for Filtered<'_, S, F> {
    type Error = S::Error;

    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Self::Error> {
        self.source.plan(|name, ty, dims| {
            if (self.keep)(name) {
                rule(name, ty, dims)
            } else {
                Ok(Take::Skip)
            }
        })
    }

    fn read(&self, p: &Planned) -> Result<Part<'_>, Self::Error> {
        self.source.read(p)
    }

    fn release(&self, p: &Planned) -> Result<(), Self::Error> {
        self.source.release(p)
    }
}

impl Source for Split {
    type Error = Error;

    /// A `.scale` beside a scaled tensor is read with it, not on its own.
    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Error> {
        let mut out = Vec::new();
        for (_, t) in self.iter_tensors() {
            if gguf::scaled_weight(self, &t.name).is_some() {
                continue;
            }
            if let Some(at) = take(&rule, t)? {
                let (info, _) = part_of(t, at.clone())?;
                out.push(Planned { at, info });
            }
        }
        Ok(out)
    }

    /// A scale the file lacks or does not hold as its type reads it is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Error> {
        let name = p.info.name.as_str();
        let (shard, t) = self
            .find(name)
            .ok_or_else(|| refuse(name, "is not in the file"))?;
        let data = self
            .shard(shard)
            .ok_or_else(|| refuse(name, format!("names shard {shard}")))?
            .data(t)?;
        let (info, runs) = part_of(t, p.at.clone())?;
        let find = |n: &str| {
            let (s, t) = self.find(n)?;
            Some((self.shard(s)?, t))
        };
        let scale = gguf::tensor_scales(t, find)?.map(|s| match &p.at {
            Some((2, r)) if t.dims.len() == 3 => s[r.start as usize..r.end as usize].to_vec(),
            _ => s,
        });
        Ok(Part {
            info,
            bytes: Bytes::File(runs.into_iter().map(|r| &data[r]).collect()),
            scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::write::{Layout, TensorDecl};

    #[test]
    fn shared_parts_are_planned_only_for_the_selected_component() {
        let names = ["target.weight", "draft.weight"];
        let layout = Layout::new(
            &[],
            names
                .iter()
                .map(|name| TensorDecl {
                    name: (*name).to_string(),
                    dims: vec![1],
                    type_id: GgmlType::F32.as_u32(),
                    nbytes: 4,
                })
                .collect(),
        )
        .unwrap();
        let source = Split::header_only(PathBuf::from("/synthetic/shared-parts"), &layout).unwrap();
        for wanted in names {
            let selected = Filtered {
                source: &source,
                keep: |name: &str| name == wanted,
            };
            let parts = selected
                .plan(|_, _, _| Ok::<_, std::convert::Infallible>(Take::Whole))
                .unwrap();
            assert_eq!(
                parts
                    .iter()
                    .map(|p| p.info.name.as_str())
                    .collect::<Vec<_>>(),
                vec![wanted]
            );
        }
    }
}
