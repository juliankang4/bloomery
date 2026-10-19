//! Tensor-parallel placement: N ranks, one a node, each holding a slice of every split tensor and
//! all of every whole one in its own node's pool. No host or NVMe tier: a rank holds its bytes
//! resident, and a plan that does not fit a rank's pool is refused by name.
//!
//! A tensor is split along one of its dims (ggml's order: dim 0 is the inner width a row's values
//! run along, dim 1 the rows, the dims past them whole matrices) into N equal parts. A part must
//! hold whole units of the tensor's role along that dim (a head's rows, say) and whole blocks of
//! its format (a 64-value NVFP4 block along dim 0, a 128-row FP8 block along dim 1): a cut off
//! either is refused by name. What the dims, units and blocks are comes from the caller: the
//! family says which dim and unit ([`TpTensor::cut`]), the tensor's format its bytes and blocks.

use super::devices::DeviceId;
use super::workstation::{CardSpec, unified_of};

/// A tensor as the plan takes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpTensor {
    pub name: String,
    /// Logical element counts, ggml's order.
    pub dims: Vec<u64>,
    /// The bytes that split with the values (the values and their block scales).
    pub bytes: u64,
    /// The bytes every rank holds whole beside its part (a tensor scale a matrix).
    pub whole_bytes: u64,
    /// The values a format block spans along dims 0 and 1.
    pub block: [u64; 2],
    /// `None` keeps the tensor whole on every rank.
    pub cut: Option<Cut>,
}

/// Where a tensor splits: along `dim`, in parts of whole `unit`s (the role's granule there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    pub dim: usize,
    pub unit: u64,
}

/// One rank's node, by the name its lines print, and the bytes its pool holds for the rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPool {
    pub node: String,
    pub pool_bytes: u64,
    /// The rank's device, when its census named one: two ranks on one device are refused.
    pub device: Option<DeviceId>,
}

impl RankPool {
    /// The pool of the rank whose one device is `spec`, from its node's census: on a device that
    /// allocates from host memory, the node's pool ([`unified_of`]); else the card's usable bytes,
    /// capped by its free reading.
    #[must_use]
    pub fn of(node: String, spec: &CardSpec) -> RankPool {
        let pool_bytes = match unified_of(spec) {
            Some(pool) => pool.bytes,
            None => spec
                .free_bytes
                .map_or(spec.usable_bytes(), |free| free.min(spec.usable_bytes())),
        };
        RankPool {
            node,
            pool_bytes,
            device: spec.device,
        }
    }
}

/// One rank of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rank {
    pub rank: usize,
    pub node: String,
    pub pool_bytes: u64,
    /// The parts of the split tensors the rank holds.
    pub split_bytes: u64,
    /// The whole tensors and the whole bytes of the split ones.
    pub whole_bytes: u64,
    pub split_tensors: usize,
    pub whole_tensors: usize,
}

impl Rank {
    /// Every byte the rank holds.
    ///
    /// # Panics
    ///
    /// If the rank's byte sum exceeds u64.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.split_bytes
            .checked_add(self.whole_bytes)
            .expect("tp: rank weight byte sum exceeds u64")
    }
}

/// Why a plan was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TpError {
    #[error("tensor parallel: no rank to plan")]
    NoRank,
    #[error("tp: rank weight byte sum exceeds u64")]
    Overflow,
    #[error("tp: {name}: dim {dim} is past its dims {dims:?}")]
    NoDim {
        name: String,
        dim: usize,
        dims: Vec<u64>,
    },
    #[error(
        "tp: {name}: dim {dim} of {extent} into {ranks} ranks is not a whole number of {unit}-value units a rank"
    )]
    Unit {
        name: String,
        dim: usize,
        extent: u64,
        ranks: usize,
        unit: u64,
    },
    #[error(
        "tp: {name}: dim {dim} of {extent} into {ranks} ranks cuts off its format's {block}-value block"
    )]
    Block {
        name: String,
        dim: usize,
        extent: u64,
        ranks: usize,
        block: u64,
    },
    #[error("tp: rank {rank} on {node} is the device of rank {other}")]
    SameDevice {
        rank: usize,
        node: String,
        other: usize,
    },
    #[error("tp: rank {rank} on {node} holds {bytes} B, past its pool of {pool} B")]
    Over {
        rank: usize,
        node: String,
        bytes: u64,
        pool: u64,
    },
}

/// The plan of `tensors` over one rank a pool of `pools`, in rank order.
///
/// # Errors
///
/// A cut along a dim the tensor lacks, into parts that are not whole units or whole format blocks,
/// a rank past its pool and two ranks on one device, each by name; no rank or a byte sum past u64.
pub fn plan_tp(pools: &[RankPool], tensors: &[TpTensor]) -> Result<Vec<Rank>, TpError> {
    let n = pools.len();
    if n == 0 {
        return Err(TpError::NoRank);
    }
    for (rank, p) in pools.iter().enumerate() {
        let same = |q: &RankPool| {
            p.device
                .is_some_and(|d| q.device.is_some_and(|e| e.uuid == d.uuid))
        };
        if let Some(other) = pools[..rank].iter().position(same) {
            return Err(TpError::SameDevice {
                rank,
                node: p.node.clone(),
                other,
            });
        }
    }
    let (mut split, mut whole, mut split_n, mut whole_n) = (0u64, 0u64, 0, 0);
    for t in tensors {
        let Some(cut) = t.cut else {
            whole = whole
                .checked_add(t.bytes)
                .and_then(|sum| sum.checked_add(t.whole_bytes))
                .ok_or(TpError::Overflow)?;
            whole_n += 1;
            continue;
        };
        let Some(&extent) = t.dims.get(cut.dim) else {
            return Err(TpError::NoDim {
                name: t.name.clone(),
                dim: cut.dim,
                dims: t.dims.clone(),
            });
        };
        let ranks = n as u64;
        let part = extent / ranks;
        if !extent.is_multiple_of(ranks) || !part.is_multiple_of(cut.unit) {
            return Err(TpError::Unit {
                name: t.name.clone(),
                dim: cut.dim,
                extent,
                ranks: n,
                unit: cut.unit,
            });
        }
        let block = t.block.get(cut.dim).copied().unwrap_or(1);
        if !part.is_multiple_of(block) {
            return Err(TpError::Block {
                name: t.name.clone(),
                dim: cut.dim,
                extent,
                ranks: n,
                block,
            });
        }
        // Whole blocks a part: the bytes split evenly with the values.
        split = split
            .checked_add(t.bytes / ranks)
            .ok_or(TpError::Overflow)?;
        whole = whole.checked_add(t.whole_bytes).ok_or(TpError::Overflow)?;
        split_n += 1;
    }
    split.checked_add(whole).ok_or(TpError::Overflow)?;
    pools
        .iter()
        .enumerate()
        .map(|(rank, p)| {
            let r = Rank {
                rank,
                node: p.node.clone(),
                pool_bytes: p.pool_bytes,
                split_bytes: split,
                whole_bytes: whole,
                split_tensors: split_n,
                whole_tensors: whole_n,
            };
            if r.bytes() > p.pool_bytes {
                return Err(TpError::Over {
                    rank,
                    node: p.node.clone(),
                    bytes: r.bytes(),
                    pool: p.pool_bytes,
                });
            }
            Ok(r)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Cut, RankPool, TpError, TpTensor, plan_tp};

    fn pools(n: usize, bytes: u64) -> Vec<RankPool> {
        (0..n)
            .map(|i| RankPool {
                node: format!("node{i}"),
                pool_bytes: bytes,
                device: None,
            })
            .collect()
    }

    /// A 2-D tensor of `rows` rows of `k` values, `bytes` a value, cut along `cut`.
    fn t(name: &str, k: u64, rows: u64, block: [u64; 2], cut: Option<Cut>) -> TpTensor {
        TpTensor {
            name: name.into(),
            dims: vec![k, rows],
            bytes: k * rows,
            whole_bytes: 4,
            block,
            cut,
        }
    }

    /// Every rank holds a 1/N part of each split tensor and all of each whole one; N = 1, 2 and 4
    /// give one sum by one rule, and a rank past its pool is refused naming it.
    #[test]
    fn ranks_hold_a_part_of_each_split_tensor_and_every_whole_one() {
        let rows = Some(Cut { dim: 1, unit: 128 });
        let cols = Some(Cut { dim: 0, unit: 1 });
        let ts = [
            t("q", 4096, 8192, [1, 1], rows),
            t("o", 8192, 4096, [64, 1], cols),
            t("norm", 4096, 1, [1, 1], None),
        ];
        for n in [1, 2, 4] {
            let plan = plan_tp(&pools(n, 1 << 40), &ts).unwrap();
            assert_eq!(plan.len(), n);
            for (i, r) in plan.iter().enumerate() {
                assert_eq!(r.rank, i);
                assert_eq!(r.split_bytes, 2 * 4096 * 8192 / n as u64);
                assert_eq!(r.whole_bytes, 4096 + 4 * 3);
                assert_eq!((r.split_tensors, r.whole_tensors), (2, 1));
            }
        }
        let need = 4096 * 8192 + 4096 + 12;
        assert_eq!(plan_tp(&pools(2, need), &ts).unwrap()[1].bytes(), need);
        let mut short = pools(2, need);
        short[1].pool_bytes = need - 1;
        assert_eq!(
            plan_tp(&short, &ts).unwrap_err(),
            TpError::Over {
                rank: 1,
                node: "node1".into(),
                bytes: need,
                pool: need - 1
            }
        );
    }

    /// A unified node's rank plans against its node's pool, a card's against its usable bytes
    /// capped by its free reading.
    #[test]
    fn a_rank_plans_against_its_own_nodes_pool() {
        use crate::placement::devices::DeviceId;
        use crate::placement::workstation::{CardSpec, gb10};
        let unified = gb10(125_303_951_360);
        assert_eq!(
            RankPool::of("a".into(), &unified).pool_bytes,
            125_303_951_360
        );
        let card = CardSpec {
            integrated: false,
            ..gb10(1 << 40)
        };
        assert_eq!(
            RankPool::of("b".into(), &card).pool_bytes,
            card.usable_bytes()
        );
        let busy = CardSpec {
            free_bytes: Some(1 << 30),
            ..card
        };
        assert_eq!(RankPool::of("c".into(), &busy).pool_bytes, 1 << 30);
        // A node answering for this process's own device is not a rank of its own.
        let id = |uuid| Some(DeviceId { ordinal: 0, uuid });
        let on = |node: &str, uuid| RankPool {
            device: id(uuid),
            ..RankPool::of(node.into(), &unified)
        };
        let pools = [
            on("local", [1; 16]),
            on("other", [2; 16]),
            on("loop", [1; 16]),
        ];
        assert_eq!(
            plan_tp(&pools, &[]).unwrap_err(),
            TpError::SameDevice {
                rank: 2,
                node: "loop".into(),
                other: 0
            }
        );
        assert!(plan_tp(&pools[..2], &[]).is_ok());
    }

    #[test]
    fn rank_byte_sums_must_fit_u64() {
        let tensor = |bytes, whole_bytes, cut| TpTensor {
            name: "large".into(),
            dims: vec![1 << 62],
            bytes,
            whole_bytes,
            block: [1, 1],
            cut,
        };
        let big = 1 << 63;
        let cut = Some(Cut { dim: 0, unit: 1 });
        let whole = tensor(big, 0, None);
        let split = tensor(big, 0, cut);
        let companion = tensor(0, big, cut);
        let results = [
            vec![whole.clone(), whole.clone()],
            vec![tensor(big, big, None)],
            vec![split.clone(), split.clone()],
            vec![companion.clone(), companion],
            vec![split, whole],
        ]
        .map(|ts| plan_tp(&pools(1, 1), &ts));
        assert!(
            results.iter().all(|r| matches!(r, Err(TpError::Overflow))),
            "{results:?}"
        );
        for cut in [None, cut] {
            let plan = plan_tp(&pools(1, u64::MAX), &[tensor(big, big - 1, cut)]).unwrap();
            assert_eq!(plan[0].bytes(), u64::MAX);
        }
    }

    /// A part that is not whole units of the role, or that cuts a format block, is refused by name;
    /// so are a dim the tensor lacks and no rank.
    #[test]
    fn cuts_off_a_unit_or_a_block_are_refused_by_name() {
        let heads = Some(Cut { dim: 1, unit: 128 });
        // 64 heads of 128 rows: 3 ranks part no whole head, 128 ranks take half a head each.
        for n in [3, 128] {
            let e = plan_tp(&pools(n, u64::MAX), &[t("q", 4096, 8192, [1, 1], heads)]).unwrap_err();
            assert!(
                matches!(e, TpError::Unit { ranks, unit: 128, .. } if ranks == n),
                "{e}"
            );
        }
        // A down of K 1024 at N = 4 is 256 a rank, whole 64-value NVFP4 blocks; K 1088 is 272.
        let down = |k| t("down", k, 4096, [64, 1], Some(Cut { dim: 0, unit: 1 }));
        assert!(plan_tp(&pools(4, u64::MAX), &[down(1024)]).is_ok());
        let e = plan_tp(&pools(4, u64::MAX), &[down(1088)]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "tp: down: dim 0 of 1088 into 4 ranks cuts off its format's 64-value block"
        );
        // A 128-row FP8 block: 2048 rows into 4 is 512, into 32 is 64.
        let gate = t(
            "gate",
            4096,
            2048,
            [128, 128],
            Some(Cut { dim: 1, unit: 1 }),
        );
        assert!(plan_tp(&pools(4, u64::MAX), std::slice::from_ref(&gate)).is_ok());
        assert!(matches!(
            plan_tp(&pools(32, u64::MAX), &[gate]).unwrap_err(),
            TpError::Block { block: 128, .. }
        ));
        let e = plan_tp(
            &pools(2, u64::MAX),
            &[t("k_b", 4096, 64, [1, 1], Some(Cut { dim: 2, unit: 1 }))],
        )
        .unwrap_err();
        assert!(matches!(e, TpError::NoDim { dim: 2, .. }), "{e}");
        assert_eq!(plan_tp(&[], &[]).unwrap_err(), TpError::NoRank);
    }
}
