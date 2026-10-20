//! The head's merge across ranks: each rank's head projects its part of the vocabulary and takes
//! that part's argmax, and the step's token is the merge of every rank's pick ([`merge`]).

use bloomery_gpu::GpuError;

/// What the errors name.
const WHAT: &str = "glm5next tp head merge";

/// The step's token from every rank's pick `(value, id)`, `id` in the whole vocabulary: the
/// largest value, and of equal values the lowest id, which is the one-card argmax's rule (the
/// first index of the largest value) over the ranks' parts in vocabulary order. No pick, and a
/// value that is not finite (a part's argmax refuses one), are refused by name.
pub(crate) fn merge(picks: &[(f32, u32)]) -> Result<u32, GpuError> {
    let mut best: Option<(f32, u32)> = None;
    for &(v, id) in picks {
        if !v.is_finite() {
            return Err(GpuError::shape(WHAT, format!("a pick of {v} at id {id}")));
        }
        best = match best {
            Some((bv, bid)) if bv > v || (bv == v && bid < id) => Some((bv, bid)),
            _ => Some((v, id)),
        };
    }
    best.map(|(_, id)| id)
        .ok_or_else(|| GpuError::shape(WHAT, "no rank's pick"))
}

#[cfg(test)]
mod tests {
    use super::merge;

    /// The merge of every split of a row into N = 1, 2 and 4 parts is the row's first index of
    /// its largest value, ties across parts and within one included, whatever order the picks
    /// come in; no pick and a pick that is not finite are refused.
    #[test]
    fn the_merge_is_the_whole_rows_argmax() {
        let rows: [&[f32]; 4] = [
            &[1.0, 3.0, 2.0, 3.0, 0.5, -1.0, 3.0, 2.5],
            &[-4.0, -2.0, -3.0, -2.0, -9.0, -2.5, -7.0, -8.0],
            &[0.0; 8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0],
        ];
        for row in rows {
            let want = row
                .iter()
                .enumerate()
                .fold(0, |b, (i, &x)| if x > row[b] { i } else { b }) as u32;
            for n in [1usize, 2, 4] {
                let part = row.len() / n;
                let mut picks: Vec<(f32, u32)> = row
                    .chunks(part)
                    .enumerate()
                    .map(|(r, c)| {
                        let i = c
                            .iter()
                            .enumerate()
                            .fold(0, |b, (i, &x)| if x > c[b] { i } else { b });
                        (c[i], (r * part + i) as u32)
                    })
                    .collect();
                assert_eq!(merge(&picks).unwrap(), want, "{row:?} in {n} parts");
                picks.reverse();
                assert_eq!(
                    merge(&picks).unwrap(),
                    want,
                    "{row:?} in {n} parts, reversed"
                );
            }
        }
        assert!(merge(&[]).is_err());
        assert!(merge(&[(1.0, 0), (f32::NAN, 1)]).is_err());
        assert!(merge(&[(f32::INFINITY, 0)]).is_err());
    }
}
