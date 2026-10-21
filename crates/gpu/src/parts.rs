//! The one load of a part source onto a card (`model::part::Source`: a GGUF file or a
//! checkpoint folder). The caller's rule says which tensors the card holds, whole or the part
//! of a rank, as the source plans them; every planned part's card format is checked before any
//! is read; then each part is read and staged in turn through one staging ring, which one
//! synchronize ends ([`load`]). A rank of a tensor-parallel load of N ranks passes its family's
//! cut; a one-card load passes its placement plan's card ([`OnCard`]).

use std::collections::{BTreeMap, HashMap, HashSet};

use cuda_core::CudaStream;
use gguf::{GgmlType, TensorInfo};
use model::part::{Source, Take};
use model::placement::{CardFormat, Device, Format, ModelTensor, Plan, Segment};

use crate::GpuError;
use crate::upload::UploadRing;
use crate::weights::{TensorPart, Weights, part_format, stage_parts};

/// What the errors name.
const WHAT: &str = "parts::load";

/// The parts of `source` that `rule` takes, each read and uploaded to the card of `stream` in
/// the card format `card` names for it, else its type's ([`part_format`]). Every part is planned
/// and its format checked, and refused by name, before any is read; then one part at a time is
/// read and staged, so the load holds one part's bytes of its own beside the card (a file's
/// parts are its mapping's and hold none). The weights are the caller's only once every copy
/// has ended: a refusal at any part drops what was staged. Under `release` each part's file
/// pages are dropped once it is staged ([`Source::release`]), as a split's placed load drops its
/// card segments' (`BLOOMERY_CARD_DONTNEED`).
pub fn load<S: Source, E: std::fmt::Display>(
    stream: &CudaStream,
    source: &S,
    rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
    release: bool,
) -> Result<Weights, GpuError> {
    let planned = source.plan(rule).map_err(|e| refused(e.to_string()))?;
    let mut budget = 0usize;
    for p in &planned {
        budget = budget
            .checked_add(part_format(&p.info)?.1)
            .ok_or_else(|| refused("the parts' staging bytes pass usize".into()))?;
    }
    stream.context().bind_to_thread()?;
    let mut staged = BTreeMap::new();
    // Made after `staged` and dropped before it: its drop drains the copies, so no buffer of a
    // refused load is freed while a copy still writes it.
    let mut ring = UploadRing::new(stream, budget)?;
    for p in &planned {
        let part = source.read(p).map_err(|e| refused(e.to_string()))?;
        let one = TensorPart {
            info: part.info.clone(),
            spans: part.spans(),
            scale: part.scale.clone(),
        };
        let held = |name: &str| staged.contains_key(name);
        let added = stage_parts(stream, &mut ring, &[one], held, &card)?;
        staged.extend(added);
        drop(part);
        if release {
            source.release(p).map_err(|e| refused(e.to_string()))?;
        }
    }
    ring.finish(stream)?;
    Ok(Weights::of_staged(staged))
}

/// Card `card` of `plan` as a load from a part source reads it ([`load`]): each tensor the plan
/// holds whole on that card, in the plan's card format and resident bytes; a tensor the plan
/// uses nowhere, or reads by rows from the NVMe tier (a row-gathered table, which its own
/// reader maps where the source stores it), is skipped. A source fills no host set and no other
/// card, so a tensor the plan splits or holds on the host or another card is refused by name.
pub struct OnCard {
    held: HashMap<String, (CardFormat, u64)>,
    skipped: HashSet<String>,
}

impl OnCard {
    /// Card `card` of `plan`, each tensor's segments checked as the type says.
    pub fn of(plan: &Plan<'_>, card: usize) -> Result<OnCard, GpuError> {
        let model = plan.model;
        let rows = plan.rows.iter().map(|row| {
            model
                .tensors
                .get(row.tensor)
                .map(|t| (t, row.segments.as_slice(), model.experts_of(t)))
                .ok_or_else(|| refused(format!("a placement row names tensor {}", row.tensor)))
        });
        OnCard::of_rows(rows, card)
    }

    /// The tensors of `rows` (each with its segments and its stack's experts) on card `card`.
    fn of_rows<'a>(
        rows: impl Iterator<Item = Result<(&'a ModelTensor, &'a [Segment], u64), GpuError>>,
        card: usize,
    ) -> Result<OnCard, GpuError> {
        let mut held = HashMap::new();
        let mut skipped = HashSet::new();
        for row in rows {
            let (t, segments, experts) = row?;
            let elsewhere = |placed: String| {
                refused(format!(
                    "{} is placed {placed}: a load from a part source holds every tensor whole \
                     on card {card}, and reads a row-gathered table from the NVMe tier",
                    t.name
                ))
            };
            match segments {
                [] => {
                    skipped.insert(t.name.clone());
                }
                [s] if s.device == Device::Card(card) => {
                    let Format::Card(format) = s.format else {
                        return Err(refused(format!(
                            "{} is on the card as {}",
                            t.name, s.format
                        )));
                    };
                    let whole = s
                        .experts
                        .as_ref()
                        .is_none_or(|e| e.ids().len() as u64 == experts);
                    if !whole {
                        return Err(elsewhere(format!(
                            "on card {card} in part, the rest of its experts elsewhere"
                        )));
                    }
                    held.insert(t.name.clone(), (format, s.resident_bytes));
                }
                [s] if s.device == Device::Unused
                    || (s.device == Device::Nvme && t.gathered_rows.is_some()) =>
                {
                    skipped.insert(t.name.clone());
                }
                segments => {
                    let on: Vec<String> =
                        segments.iter().map(|s| format!("{:?}", s.device)).collect();
                    return Err(elsewhere(format!("on {}", on.join(" and "))));
                }
            }
        }
        Ok(OnCard { held, skipped })
    }

    /// The rule [`load`] plans by: a tensor the card holds whole, or none; a name the plan does
    /// not hold is refused by name.
    pub fn take(&self, name: &str) -> Result<Take, String> {
        if self.held.contains_key(name) {
            Ok(Take::Whole)
        } else if self.skipped.contains(name) {
            Ok(Take::Skip)
        } else {
            Err(format!("{name} is not a tensor of the plan"))
        }
    }

    /// The plan's card format of `t`.
    pub fn format(&self, t: &TensorInfo) -> Option<CardFormat> {
        self.held.get(&t.name).map(|&(f, _)| f)
    }

    /// Refused by name unless `w` holds every tensor the card holds, at the plan's resident
    /// bytes: a tensor the source does not make, or makes otherwise than the plan's file did.
    pub fn check(&self, w: &Weights) -> Result<(), GpuError> {
        for (name, &(_, bytes)) in &self.held {
            let got = w.get(name).map(|dw| dw.resident_bytes() as u64);
            if got != Some(bytes) {
                return Err(refused(format!(
                    "{name}: the source uploaded {got:?} device bytes, the plan holds {bytes}"
                )));
            }
        }
        Ok(())
    }
}

/// Card `card` of `plan` from `source` ([`OnCard`], [`load`], each part's pages released under
/// `release`), checked against the plan.
pub fn load_placed<S: Source>(
    stream: &CudaStream,
    source: &S,
    plan: &Plan<'_>,
    card: usize,
    release: bool,
) -> Result<Weights, GpuError> {
    let on = OnCard::of(plan, card)?;
    let w = load(
        stream,
        source,
        |name, _, _| on.take(name),
        |t| on.format(t),
        release,
    )?;
    on.check(&w)?;
    Ok(w)
}

fn refused(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::placement::{ExpertList, Role};

    /// The rule a load from a part source plans by: a tensor whole on the card is taken in the
    /// plan's card format, an unused tensor and a row-gathered table on the NVMe tier are
    /// skipped, and a tensor the plan places on the host, on another card, split, or in part
    /// on the card is refused by name, as is a name the plan does not hold; the check refuses
    /// weights that miss a held tensor.
    #[test]
    fn on_card_takes_the_card_s_tensors_and_refuses_the_rest() {
        let tensor = |name: &str, role, gathered_rows| ModelTensor {
            name: name.into(),
            shard: 0,
            layer: Some(0),
            role,
            ty: GgmlType::BF16,
            dims: vec![64, 4, 8],
            file_bytes: 4096,
            gathered_rows,
        };
        let seg = |device, format, experts: Option<Vec<u32>>| Segment {
            device,
            format,
            experts: experts.map(|ids| ExpertList::new(ids, 8).unwrap()),
            resident_bytes: 4096,
        };
        let bf16 = Format::Card(CardFormat::Bf16Raw);
        let held = tensor("a.weight", Role::Attention, None);
        let stack = tensor("x.weight", Role::RoutedExperts, None);
        let table = tensor("t.weight", Role::EngramTable, Some(1));
        let unused = tensor("u.weight", Role::Attention, None);
        let ok = [
            (&held, vec![seg(Device::Card(0), bf16, None)]),
            (
                &stack,
                vec![seg(Device::Card(0), bf16, Some((0..8).collect()))],
            ),
            (&table, vec![seg(Device::Nvme, Format::NvmeFile, None)]),
            (&unused, vec![seg(Device::Unused, Format::Unused, None)]),
        ];
        let of = |r: &[(&ModelTensor, Vec<Segment>)]| {
            OnCard::of_rows(r.iter().map(|(t, s)| Ok((*t, s.as_slice(), 8))), 0)
        };
        let on = of(&ok).unwrap();
        assert!(matches!(on.take("a.weight"), Ok(Take::Whole)));
        assert!(matches!(on.take("x.weight"), Ok(Take::Whole)));
        assert!(matches!(on.take("t.weight"), Ok(Take::Skip)));
        assert!(matches!(on.take("u.weight"), Ok(Take::Skip)));
        assert!(on.take("z.weight").unwrap_err().contains("z.weight"));
        let info = |name: &str| TensorInfo {
            name: name.into(),
            dims: vec![64],
            ty: GgmlType::BF16,
            offset: 0,
            nbytes: 128,
        };
        assert_eq!(on.format(&info("a.weight")), Some(CardFormat::Bf16Raw));
        assert_eq!(on.format(&info("t.weight")), None);
        let e = on.check(&Weights::default()).unwrap_err().to_string();
        assert!(e.contains("uploaded None device bytes"), "{e}");
        let host_table = tensor("t.weight", Role::EngramTable, Some(1));
        for (t, segments, named) in [
            (
                &held,
                vec![seg(Device::Host, Format::HostFile, None)],
                "on Host",
            ),
            (&held, vec![seg(Device::Card(1), bf16, None)], "on Card(1)"),
            (
                &host_table,
                vec![seg(Device::Host, Format::HostFile, None)],
                "on Host",
            ),
            (
                &held,
                vec![
                    seg(Device::Card(0), bf16, None),
                    seg(Device::Host, Format::HostFile, None),
                ],
                "on Card(0) and Host",
            ),
            (
                &stack,
                vec![seg(Device::Card(0), bf16, Some((0..6).collect()))],
                "in part",
            ),
        ] {
            let e = of(&[(t, segments)]).err().expect("refused").to_string();
            assert!(e.contains(&t.name) && e.contains(named), "{e}");
        }
    }
}
