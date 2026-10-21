//! One module per model architecture whose checkpoint the hub holds.

pub mod glm5next;
pub mod qwen35moe;

/// A checkpoint's `-exp(x)`, computed in f64 and rounded once to finite nonzero F32.
fn neg_exp(x: f32) -> Result<f32, String> {
    if !x.is_finite() {
        return Err(format!("non-finite input {x}"));
    }
    let y = (-f64::from(x).exp()) as f32;
    if y.is_finite() && y != 0.0 {
        Ok(y)
    } else {
        Err(format!(
            "-exp({x}) = {y} is outside its finite nonzero F32 range"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::neg_exp;

    #[test]
    fn neg_exp_checks_inputs_and_rounded_results() {
        assert_eq!(neg_exp(0.0).unwrap(), -1.0);
        assert_eq!(neg_exp(-0.0).unwrap(), -1.0);
        assert!(neg_exp(-100.0).unwrap().is_subnormal());
        for x in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let e = neg_exp(x).unwrap_err();
            assert!(e.contains("non-finite input"), "{x}: {e}");
        }
        for x in [100.0, -1000.0] {
            let e = neg_exp(x).unwrap_err();
            assert!(e.contains("finite nonzero F32 range"), "{x}: {e}");
        }
    }
}
