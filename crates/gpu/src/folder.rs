//! What a model loaded from a checkpoint folder is to a body's open: the one owner of the choice
//! between a GGUF split and a folder, of the folder as a load reads it, and of what a folder's
//! load does not run.
//!
//! A body's open takes a [`Model`] (`impl Into<Model>`): a [`Split`], which the load reads as it
//! reads any GGUF, or a [`Folder`]. [`Model::into_origin`] makes the choice once: the header the
//! load plans by and an [`Origin`], whose [`Origin::bytes`] is what
//! [`GpuModel::load_placed_from`](crate::GpuModel::load_placed_from) takes for the card's
//! tensors. A body that is a folder's load adds only what its architecture forces: the part of
//! the folder it reads beyond the parts (`X`, [`Folder::with`]). Under a residency machine the
//! machine's source reads the experts where the folder stores them, through the located header
//! (`crate::host::swap_source::FileSwap`).

use std::sync::Mutex;

use gguf::Split;
use model::part::Source;
use model::placement::Plan;

use crate::GpuError;
use crate::host::swap::Residency;
use crate::model::CardBytes;
use crate::parts::{InPlace, PlacedSource};

/// What the errors name.
const WHAT: &str = "folder::Model::into_origin";

type CardParts = Box<dyn crate::model::PartsLoad>;

/// The model a load reads: a GGUF split, or a checkpoint folder ([`Folder`]). `X` is what the
/// folder's architecture carries beside the folder's parts.
pub enum Model<X = ()> {
    Split(Split),
    Folder(Folder<X>),
}

impl<X> Model<X> {
    /// The header the load plans by: the split's first shard's, or the folder's located header.
    #[must_use]
    pub fn header(&self) -> &Split {
        match self {
            Model::Split(file) => file,
            Model::Folder(f) => &f.header,
        }
    }

    /// The header the load plans by and where the card's tensors come from, the choice a body's
    /// open takes once. A folder's load under `residency` stages its experts from where its
    /// located header says the folder stores them; a header that locates no file
    /// (`gguf::Split::header_only`) is refused by name under a residency machine, its host holding
    /// no expert to stage.
    pub fn into_origin(self, residency: Residency) -> Result<(Split, Origin<X>), GpuError> {
        match self {
            Model::Split(file) => Ok((file, Origin::File)),
            Model::Folder(f) => {
                if residency != Residency::Off && f.header.is_header_only() {
                    return Err(GpuError::shape(
                        WHAT,
                        "a checkpoint folder whose header locates no file runs under \
                         BLOOMERY_RESIDENCY=off: the residency machine's source reads each \
                         expert where the folder stores it",
                    ));
                }
                Ok((
                    f.header,
                    Origin::Folder {
                        parts: f.parts,
                        extra: f.extra,
                    },
                ))
            }
        }
    }
}

impl<X> From<Split> for Model<X> {
    fn from(file: Split) -> Model<X> {
        Model::Split(file)
    }
}

impl<X> From<Folder<X>> for Model<X> {
    fn from(folder: Folder<X>) -> Model<X> {
        Model::Folder(folder)
    }
}

/// A checkpoint folder as a load reads it: its header (`gguf::Split::located`, or
/// `gguf::Split::header_only` for a folder whose host reads nothing where the folder stores it),
/// the header its plan is made from, whose in-place tensors the host reads where the folder
/// stores them ([`InPlace`]), and its card tensors through its part source
/// (`crate::parts::PlacedSource`). `X` is the architecture's own addition to the load, `()` for
/// none.
pub struct Folder<X = ()> {
    header: Split,
    parts: CardParts,
    extra: X,
}

impl<X> Folder<X> {
    /// The folder of header `header`, its card tensors read from `source`, with the
    /// architecture's `extra`. A load reads only its plan's tensors of the source, so the source
    /// may also hold another plan's (a draft's, which loads through [`Origin::bytes`] too).
    pub fn with<S: Source + Send + Sync + 'static>(
        header: Split,
        source: S,
        extra: X,
    ) -> Folder<X> {
        let in_place = InPlace::of(&header);
        Folder {
            header,
            parts: Box::new(PlacedSource {
                source,
                in_place,
                last: Mutex::default(),
            }),
            extra,
        }
    }
}

impl<X> Folder<X> {
    /// The least host bytes a load of card `card` of `plan` reads beside the plan's own terms: its
    /// parts one at a time, with the staging ring. The load's own check counts its read-ahead
    /// (`HostNeed::read_ahead`) from the same parts at the room it reads, never under this.
    pub fn least_reads(&self, plan: &Plan<'_>, card: usize) -> Result<u64, GpuError> {
        self.parts.host_bytes(plan, card, 0)
    }
}

impl Folder {
    /// The folder of header `header`, its card tensors read from `source`.
    pub fn new<S: Source + Send + Sync + 'static>(header: Split, source: S) -> Folder {
        Folder::with(header, source, ())
    }
}

/// Where a load's card tensors come from ([`Model::into_origin`]).
pub enum Origin<X = ()> {
    /// The split's own mapped tensors.
    File,
    /// A folder's part source, and the architecture's `extra`.
    Folder { parts: CardParts, extra: X },
}

impl<X> Origin<X> {
    /// The card's bytes as [`GpuModel::load_placed_from`](crate::GpuModel::load_placed_from)
    /// reads them.
    #[must_use]
    pub fn bytes(&self) -> CardBytes<'_> {
        match self {
            Origin::File => CardBytes::File,
            Origin::Folder { parts, .. } => CardBytes::Parts(parts.as_ref()),
        }
    }

    /// The folder's architecture addition; `None` for a split.
    #[must_use]
    pub fn extra(&self) -> Option<&X> {
        match self {
            Origin::File => None,
            Origin::Folder { extra, .. } => Some(extra),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::Value;
    use gguf::write::{Layout, TensorDecl};
    use model::part::{Part, Planned, Take};

    /// A part source that holds nothing: a load that read it would be refused by name.
    struct Empty;

    impl Source for Empty {
        type Error = std::fmt::Error;

        fn plan<E: std::fmt::Display>(
            &self,
            _: impl Fn(&str, gguf::GgmlType, &[u64]) -> Result<Take, E>,
        ) -> Result<Vec<Planned>, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read_bytes(&self, _: &Planned) -> Result<u64, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read(&self, _: &Planned) -> Result<Part<'_>, Self::Error> {
            Err(std::fmt::Error)
        }
    }

    fn header() -> Split {
        let kvs = [("toy.layers".to_string(), Value::U32(1))];
        let tensors = vec![TensorDecl {
            name: "w.weight".into(),
            dims: vec![32],
            type_id: gguf::GgmlType::F32.as_u32(),
            nbytes: 128,
        }];
        let layout = Layout::new(&kvs, tensors).unwrap();
        Split::header_only(std::env::temp_dir().join("folder-toy.gguf"), &layout).unwrap()
    }

    /// A load of a split reads the split's own tensors under any residency and carries no
    /// addition; a folder's reads its part source and carries its addition.
    #[test]
    fn a_load_reads_a_splits_own_tensors_or_a_folders_parts() {
        let mid = Residency::Mid {
            pinned: 1,
            spares: 1,
        };
        let (_, split) = Model::<u8>::from(header()).into_origin(mid).unwrap();
        assert!(matches!(split.bytes(), CardBytes::File));
        assert!(split.extra().is_none());
        let (_, folder) = Model::from(Folder::with(header(), Empty, 7u8))
            .into_origin(Residency::Off)
            .unwrap();
        assert!(matches!(folder.bytes(), CardBytes::Parts(_)));
        assert_eq!(folder.extra(), Some(&7));
        let plain: Model = Folder::new(header(), Empty).into();
        assert!(plain.into_origin(Residency::Off).is_ok());
    }

    // PIN(2026-10-22): folder drafts use the placed MTP load; only residency is refused here.
    /// A folder whose header locates no file runs no residency machine: it is refused by name,
    /// and a split's load under one passes.
    #[test]
    fn a_folder_load_refuses_residency() {
        let mid = Residency::Mid {
            pinned: 1,
            spares: 1,
        };
        let folder = || Model::<()>::from(Folder::new(header(), Empty));
        let e = folder().into_origin(mid).err().unwrap();
        assert!(e.to_string().contains("BLOOMERY_RESIDENCY=off"), "{e}");
        assert!(folder().into_origin(Residency::Off).is_ok());
        assert!(Model::<()>::from(header()).into_origin(mid).is_ok());
    }
}
