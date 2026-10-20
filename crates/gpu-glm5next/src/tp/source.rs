//! Where a rank's tensors come from ([`GgufSource`]): each tensor's part for one rank of N, cut
//! by the family's rule (`model::arch::glm5next::tp::cut`) from the GGUF file's mapped bytes, as
//! the header entry of the part and its rows' bytes, which
//! [`Weights::load_parts`](bloomery_gpu::weights::Weights::load_parts) uploads.
//!
//! A part of a tensor in ggml's layout ([`part_of`]), dims `[d0, d1, …]` with `d0` the values a
//! row runs along: cut along dim 0, every row keeps its own run of `d0 / N` values, whole format
//! blocks, so the part is one slice a row; cut along a dim past 0, each index of the dims past
//! the cut keeps a run of `d / N` whole indices of the cut dim, so the part is one slice of
//! contiguous rows for each. A tensor kept whole is its one slice on every rank.

use std::ops::Range;

use bloomery_gpu::GpuError;
use bloomery_gpu::weights::TensorPart;
use gguf::{Split, TensorInfo};
use model::arch::glm5next::tp;
use model::placement::tp::Cut;

/// What the errors name.
const WHAT: &str = "glm5next tp source";

/// One rank of a tensor-parallel load: its index among the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankOf {
    rank: usize,
    ranks: usize,
}

impl RankOf {
    /// Rank `rank` of `ranks`; no rank, or a rank past them, is refused by name.
    pub fn new(rank: usize, ranks: usize) -> Result<RankOf, GpuError> {
        if rank >= ranks {
            return Err(GpuError::shape(
                WHAT,
                format!("rank {rank} of {ranks}: want at least one rank and an index below them"),
            ));
        }
        Ok(RankOf { rank, ranks })
    }

    /// This rank's part of `v`, `what`'s count over the ranks: refused by name when the ranks
    /// do not divide it or outnumber it.
    pub fn part(self, what: &str, v: usize) -> Result<usize, GpuError> {
        let n = self.ranks;
        (v.is_multiple_of(n) && v >= n)
            .then_some(v / n)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{what} {v} over {n} ranks")))
    }

    #[must_use]
    pub fn rank(self) -> usize {
        self.rank
    }

    #[must_use]
    pub fn ranks(self) -> usize {
        self.ranks
    }
}

/// The GGUF file's parts: each cut from the tensor's mapped bytes, every slice borrowed.
pub struct GgufSource<'f> {
    file: &'f Split,
    /// The heads the attention splits by.
    heads: u64,
}

impl<'f> GgufSource<'f> {
    /// The parts of `file`, whose attention has `heads` heads.
    #[must_use]
    pub fn new(file: &'f Split, heads: usize) -> GgufSource<'f> {
        GgufSource {
            file,
            heads: heads as u64,
        }
    }
}

impl GgufSource<'_> {
    /// Rank `at`'s part of tensor `name`, cut by the family's rule, or the whole tensor where the
    /// rule keeps it whole. A tensor the file lacks, a stem with no rule and a cut that does not
    /// fall on whole units and format blocks are refused by name.
    pub(super) fn part(&self, name: &str, at: RankOf) -> Result<TensorPart<'_>, GpuError> {
        let (shard, t) = self.file.find(name).ok_or_else(|| GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: "a tensor of the file",
        })?;
        let bytes = self
            .file
            .shard(shard)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{name} names shard {shard}")))?
            .data(t)?;
        let cut = tp::cut(name, &t.dims, self.heads)
            .map_err(|why| GpuError::shape(WHAT, format!("{name}: {why}")))?;
        let (info, spans) = part_of(t, cut, at)?;
        Ok(TensorPart {
            info,
            spans: spans.into_iter().map(|r| &bytes[r]).collect(),
        })
    }
}

/// Rank `at`'s part of tensor `t` cut by `cut` (module doc): the part's header entry (its dims,
/// its bytes) and the byte ranges of its rows in the tensor's bytes, in row order. An extent the
/// ranks do not divide into whole `cut.unit`s, a dim-0 part off the type's block, a dim past the
/// tensor's and byte counts past u64 or usize are refused by name.
fn part_of(
    t: &TensorInfo,
    cut: Option<Cut>,
    at: RankOf,
) -> Result<(TensorInfo, Vec<Range<usize>>), GpuError> {
    let refuse = |why: String| GpuError::shape(WHAT, format!("{}: {why}", t.name));
    let over = || refuse("its byte count passes the integer it is held in".into());
    let whole = usize::try_from(t.nbytes).map_err(|_| over())?;
    let Some(Cut { dim, unit }) = cut else {
        return Ok((t.clone(), std::iter::once(0..whole).collect()));
    };
    let n = at.ranks as u64;
    let extent = *t
        .dims
        .get(dim)
        .ok_or_else(|| refuse(format!("dim {dim} past its dims {:?}", t.dims)))?;
    let part = extent / n;
    if unit == 0 || part * n != extent || !part.is_multiple_of(unit) {
        return Err(refuse(format!(
            "dim {dim} of {extent} into {n} ranks is not whole {unit}-value units a rank"
        )));
    }
    let rows: u64 = t.dims[1..].iter().product();
    if rows == 0 || !t.nbytes.is_multiple_of(rows) {
        return Err(refuse(format!("{} bytes in {rows} rows", t.nbytes)));
    }
    let row = t.nbytes / rows;
    let r = at.rank as u64;
    let mut dims = t.dims.clone();
    dims[dim] = part;
    let spans: Vec<(u64, u64)> = if dim == 0 {
        let block =
            t.ty.blck_size()
                .filter(|&b| b > 0)
                .ok_or_else(|| refuse(format!("{} has no block size", t.ty)))?;
        if !part.is_multiple_of(block) || !extent.is_multiple_of(block) {
            return Err(refuse(format!(
                "dim 0 of {extent} into {n} ranks cuts off its {block}-value {} block",
                t.ty
            )));
        }
        let take = row / (extent / block) * (part / block);
        (0..rows).map(|i| (i * row + r * take, take)).collect()
    } else {
        let inner: u64 = t.dims[1..dim].iter().product();
        let outer: u64 = t.dims[dim + 1..].iter().product();
        let run = part
            .checked_mul(inner)
            .and_then(|v| v.checked_mul(row))
            .ok_or_else(over)?;
        let stride = run.checked_mul(n).ok_or_else(over)?;
        (0..outer).map(|o| (o * stride + r * run, run)).collect()
    };
    let nbytes = spans
        .iter()
        .try_fold(0u64, |sum, &(_, len)| sum.checked_add(len))
        .ok_or_else(over)?;
    let ranges = spans
        .into_iter()
        .map(|(start, len)| {
            let start = usize::try_from(start).map_err(|_| over())?;
            let end = usize::try_from(len)
                .ok()
                .and_then(|len| start.checked_add(len))
                .filter(|&end| end <= whole)
                .ok_or_else(over)?;
            Ok(start..end)
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    let info = TensorInfo {
        name: t.name.clone(),
        dims,
        ty: t.ty,
        offset: t.offset,
        nbytes,
    };
    Ok((info, ranges))
}

#[cfg(test)]
mod tests {
    use super::{Cut, RankOf, TensorInfo, part_of};
    use gguf::GgmlType;

    /// A tensor of `dims` of `ty`, its bytes numbered so a part's bytes name where they came from.
    fn tensor(ty: GgmlType, dims: &[u64]) -> (TensorInfo, Vec<u8>) {
        let rows: u64 = dims[1..].iter().product();
        let row = dims[0] / ty.blck_size().unwrap() * ty.type_size().unwrap();
        let nbytes = rows * row;
        let t = TensorInfo {
            name: "t".into(),
            dims: dims.to_vec(),
            ty,
            offset: 0,
            nbytes,
        };
        let bytes = (0..nbytes).map(|i| (i * 7 % 251) as u8).collect();
        (t, bytes)
    }

    /// Element `(i0, i1, i2)`'s bytes of a tensor of `dims`: `[row(i1, i2)][i0 block]`.
    fn block_bytes(t: &TensorInfo, bytes: &[u8], i: [u64; 3]) -> Vec<u8> {
        let (bs, ts) = (t.ty.blck_size().unwrap(), t.ty.type_size().unwrap());
        let d = |k: usize| t.dims.get(k).copied().unwrap_or(1);
        let row = d(0) / bs * ts;
        let at = (i[2] * d(1) + i[1]) * row + i[0] / bs * ts;
        bytes[at as usize..(at + ts) as usize].to_vec()
    }

    /// At N = 1, 2 and 4, along every dim of a 3-D stack and of a 2-D matrix, each rank's part
    /// is the dims with the cut dim over N, its bytes the tensor's blocks at those indices, and
    /// the parts' bytes add to the tensor's; whole tensors are their bytes on every rank.
    #[test]
    fn every_rank_holds_its_indices_of_the_cut_dim() {
        for (ty, dims) in [
            (GgmlType::Q5_K, vec![1024u64, 8, 4]),
            (GgmlType::Q8_0, vec![256, 16]),
            (GgmlType::F32, vec![8, 4, 16]),
        ] {
            let (t, bytes) = tensor(ty, &dims);
            let bs = ty.blck_size().unwrap();
            for n in [1usize, 2, 4] {
                for dim in 0..dims.len() {
                    let cut = Some(Cut { dim, unit: 1 });
                    let mut total = 0;
                    for r in 0..n {
                        let at = RankOf::new(r, n).unwrap();
                        let (p, spans) = part_of(&t, cut, at).unwrap();
                        let part = dims[dim] / n as u64;
                        let mut want_dims = dims.clone();
                        want_dims[dim] = part;
                        assert_eq!(p.dims, want_dims, "{ty} {dims:?} dim {dim} rank {r}/{n}");
                        let got: Vec<u8> = spans
                            .iter()
                            .flat_map(|s| bytes[s.clone()].to_vec())
                            .collect();
                        assert_eq!(got.len() as u64, p.nbytes);
                        let mut want = Vec::new();
                        let d = |k: usize| want_dims.get(k).copied().unwrap_or(1);
                        for i2 in 0..d(2) {
                            for i1 in 0..d(1) {
                                for i0 in (0..d(0)).step_by(bs as usize) {
                                    let mut i = [i0, i1, i2];
                                    i[dim] += r as u64 * part;
                                    want.extend(block_bytes(&t, &bytes, i));
                                }
                            }
                        }
                        assert_eq!(got, want, "{ty} {dims:?} dim {dim} rank {r}/{n}");
                        total += p.nbytes;
                    }
                    assert_eq!(total, t.nbytes);
                }
                let (p, spans) = part_of(&t, None, RankOf::new(n - 1, n).unwrap()).unwrap();
                assert_eq!((&p.dims, p.nbytes), (&t.dims, t.nbytes));
                assert_eq!(spans, vec![0..bytes.len()]);
            }
        }
    }

    /// An extent the ranks do not divide, a part of other than whole units, a dim-0 part off the
    /// type's block, a dim past the tensor's, no rank, and a width the ranks do not divide or
    /// outnumber are refused.
    #[test]
    fn a_cut_off_its_unit_or_block_is_refused() {
        let (t, _) = tensor(GgmlType::Q5_K, &[512, 6, 4]);
        let at = |r, n| RankOf::new(r, n).unwrap();
        for (cut, rank) in [
            (Cut { dim: 1, unit: 1 }, at(0, 4)),
            (Cut { dim: 1, unit: 2 }, at(0, 2)),
            (Cut { dim: 0, unit: 1 }, at(0, 4)),
            (Cut { dim: 3, unit: 1 }, at(0, 2)),
            (Cut { dim: 2, unit: 0 }, at(0, 2)),
        ] {
            assert!(part_of(&t, Some(cut), rank).is_err(), "{cut:?} {rank:?}");
        }
        assert!(RankOf::new(0, 0).is_err() && RankOf::new(2, 2).is_err());
        assert_eq!(at(1, 4).part("heads", 64).unwrap(), 16);
        assert!(at(0, 4).part("heads", 6).is_err() && at(0, 4).part("heads", 2).is_err());
    }
}
