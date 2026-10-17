//! Embedding requests and answers without a model or a device.
//!
//! Encoders supply pooled f32 vectors; the request cuts them to a model-supported width,
//! normalizes the cut and writes llama-server's native or OpenAI response.

pub mod input;
pub mod request;

/// A normalization llama.cpp's `common_embd_normalize` names (`embd_normalize`): `-1` none, `0`
/// by the largest magnitude onto the int16 range, `2` euclidean (the default), any other `p > 0`
/// the p-norm (`1` taxicab).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Norm(i32);

impl Default for Norm {
    fn default() -> Norm {
        Norm(2)
    }
}

impl Norm {
    /// `p` as a normalization; below `-1`, which names none, is refused by name.
    pub fn of(p: i32) -> Result<Norm, String> {
        if p < -1 {
            return Err(
                "-1 (none), 0 (max absolute, int16), 1 (taxicab), 2 (euclidean) or a p-norm \
                 above 2"
                    .to_owned(),
            );
        }
        Ok(Norm(p))
    }
}

/// `v` normalized as `common_embd_normalize` does: the norm summed in f64 in order and its
/// reciprocal applied in f32; a vector of zeros comes back as zeros. A vector with a value that
/// is not zero whose norm comes to zero or past the f64 range, or whose reciprocal is past the
/// f32 range (a large `p` underflows or overflows the sum), is refused by name, where
/// `common_embd_normalize` gives zeros or infinities.
pub fn normalize(v: &[f32], norm: Norm) -> Result<Vec<f32>, String> {
    let sum = match norm.0 {
        -1 => 1.0,
        0 => {
            v.iter().fold(0.0f64, |m, &x| {
                let a = f64::from(x.abs());
                if m < a { a } else { m }
            }) / 32760.0
        }
        2 => v.iter().fold(0.0f64, |s, &x| s + f64::from(x * x)).sqrt(),
        p => {
            let p = f64::from(p);
            v.iter()
                .fold(0.0f64, |s, &x| s + f64::from(x.abs()).powf(p))
                .powf(1.0 / p)
        }
    };
    if v.iter().all(|&x| x == 0.0) {
        return Ok(vec![0.0; v.len()]);
    }
    let scale = (1.0 / sum) as f32;
    if !(sum > 0.0 && sum.is_finite() && scale > 0.0 && scale.is_finite()) {
        return Err(format!(
            "embd_normalize {}: this vector's norm comes to {sum:e}, which f32 cannot scale by",
            norm.0
        ));
    }
    Ok(v.iter().map(|&x| x * scale).collect())
}

/// The mean of `rows`, at least one whole row of `width` f32: each column
/// summed in f64 in position order and divided by the row count.
pub fn mean(rows: &[f32], width: usize) -> Vec<f32> {
    assert!(
        width > 0 && !rows.is_empty() && rows.len().is_multiple_of(width),
        "mean pooling needs nonempty whole rows of a nonzero width"
    );
    let n = rows.len() / width;
    let mut sum = vec![0.0f64; width];
    for row in rows.chunks_exact(width) {
        for (s, &x) in sum.iter_mut().zip(row) {
            *s += f64::from(x);
        }
    }
    sum.iter().map(|&s| (s / n as f64) as f32).collect()
}

/// RFC 4648 base64, with padding.
#[must_use]
pub fn encode_base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = c
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= c.len() {
                char::from(A[(n >> (18 - 6 * i) & 63) as usize])
            } else {
                '='
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each normalization against the arithmetic `common_embd_normalize` does, and the norms
    /// it cannot scale by refused.
    #[test]
    fn normalize_is_llama_cpps() {
        let n = |p| Norm::of(p).unwrap();
        let v = [3.0f32, -4.0];
        assert_eq!(normalize(&v, n(-1)).unwrap(), v);
        let int16 = (1.0 / (4.0f64 / 32760.0)) as f32;
        assert_eq!(normalize(&v, n(0)).unwrap(), [3.0 * int16, -4.0 * int16]);
        let l1 = (1.0 / 7.0f64) as f32;
        assert_eq!(normalize(&v, n(1)).unwrap(), [3.0 * l1, -4.0 * l1]);
        assert_eq!(normalize(&v, Norm::default()).unwrap(), [0.6, -0.8]);
        let p3 = (1.0 / (27.0f64 + 64.0).powf(1.0 / 3.0)) as f32;
        assert_eq!(normalize(&v, n(3)).unwrap(), [3.0 * p3, -4.0 * p3]);
        assert_eq!(normalize(&[0.0, 0.0], n(2)).unwrap(), [0.0, 0.0]);
        assert!(Norm::of(-2).is_err());
        for (v, p) in [
            ([0.5f32, 0.25], 2000),
            ([3.0f32, 4.0], 2000),
            ([1e-40f32, 0.0], 0),
        ] {
            let e = normalize(&v, n(p)).expect_err("refused");
            assert!(
                e.starts_with(&format!("embd_normalize {p}: ")),
                "{v:?} {p}: {e}"
            );
        }
    }
}
