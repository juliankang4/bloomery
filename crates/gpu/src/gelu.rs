//! The tanh GELU (`ggml_gelu`): `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`, one rule on one
//! f32 for every kernel that applies it, and its host twin. The GELU-gated linear unit over f32
//! rows (`ggml_geglu_split`, a GEGLU feed-forward's activation), `y = gelu_tanh(gate) · up`, is
//! `crate::elem`'s `geglu_tanh` ([`crate::elem::ElemKernels::enqueue_geglu`]).
//!
//! Numeric rule: the expression of llama.cpp's CUDA `op_gelu` in f32 in its op order — the
//! polynomial `1 + (0.044715·x)·x` contracted to one fused multiply-add as nvcc contracts it,
//! every other op rounded on its own — with libdevice's `tanhf`. [`gelu_tanh_f32`] is the rule on
//! the host with the host's `tanh`, which differs from the device's by ulps.

use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};

/// `√(2/π)` as the f32 llama.cpp's `SQRT_2_OVER_PI` rounds to.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// llama.cpp's `GELU_COEF_A`.
const COEF_A: f32 = 0.044_715;

/// The rule on one f32 on the host.
#[must_use]
pub fn gelu_tanh_f32(x: f32) -> f32 {
    let poly = (COEF_A * x).mul_add(x, 1.0);
    let t = (SQRT_2_OVER_PI * x * poly).tanh();
    (0.5 * x) * (1.0 + t)
}

/// The rule on one f32 on the device: a core for kernels to call (docs/gpu-design.md decision 6).
#[inline(always)]
pub fn gelu_tanh_dev(v: f32) -> f32 {
    let poly = fma_rn_f32(mul_rn_f32(COEF_A, v), v, 1.0);
    let t = mul_rn_f32(mul_rn_f32(SQRT_2_OVER_PI, v), poly).tanh();
    mul_rn_f32(mul_rn_f32(0.5, v), add_rn_f32(1.0, t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use cuda_core::DeviceBuffer;

    /// f32's unit roundoff.
    const U: f64 = 1.0 / 16_777_216.0;

    fn gamma(n: u32) -> f64 {
        f64::from(n) * U / (1.0 - f64::from(n) * U)
    }

    /// The exact unit on one pair, in f64 at the rule's f32 constants, and the device rule's
    /// first-order distance to it. With `X` the exact tanh argument, `T = tanh X`, `S = 1 + T`:
    /// the polynomial's two roundings and the argument's two put `X` off by `γ(4)·|X|`, which
    /// `tanh` carries scaled by `1 − T²`; libdevice's `tanhf` adds 2 ulp, at most `4u·|T|`
    /// (CUDA Math API, `tanhf`); `1 + t` adds `u·|S|`; `0.5·g` is exact, and the two products add
    /// `γ(2)·|y|`. The first three move `y` scaled by `|0.5·g·up|`, not by `|y|`: where `S`
    /// cancels toward zero the distance does not shrink with it.
    fn exact(g: f32, u: f32) -> (f64, f64) {
        let (g, u) = (f64::from(g), f64::from(u));
        let x = f64::from(SQRT_2_OVER_PI) * g * (1.0 + f64::from(COEF_A) * g * g);
        let t = x.tanh();
        // 1 + tanh(x) without the cancellation at large negative x.
        let s = 2.0 / (1.0 + (-2.0 * x).exp());
        let y = 0.5 * g * s * u;
        let bound = (0.5 * g * u).abs()
            * (gamma(4) * x.abs() * (1.0 - t * t) + 4.0 * U * t.abs() + U * s)
            + gamma(2) * y.abs();
        (y, bound)
    }

    /// Gate values: every multiple of 1/256 in [−16, 16] and a few magnitudes past where `tanh`
    /// saturates; up values seeded in [−2, 2], every eighth one exactly 1.
    fn inputs() -> (Vec<f32>, Vec<f32>) {
        let mut gate: Vec<f32> = (-4096..=4096).map(|i| i as f32 / 256.0).collect();
        for m in [20.0f32, 37.5, 100.0, 1e3, 1e4] {
            gate.extend([m, -m]);
        }
        let mut s = 0x9e37_79b9u32;
        let up = (0..gate.len())
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if i.is_multiple_of(8) {
                    1.0
                } else {
                    (s >> 8) as f32 / 4_194_304.0 - 2.0
                }
            })
            .collect();
        (gate, up)
    }

    /// The gated unit on the card: every value within its bound of the exact unit, and an
    /// untouched output tail; a NaN gate or up value NaN
    /// at its own index and nowhere else; an empty launch and each short buffer refused by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_geglu_is_the_rule_within_its_bound() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (k, stream) = (gpu.elem(), gpu.stream());
        let run = |gate: &[f32], up: &[f32]| -> Vec<f32> {
            const GUARD: f32 = 12_345.0;
            let n = gate.len();
            let padded = (n + 1).next_multiple_of(256);
            let (mut g, mut u) = (gate.to_vec(), up.to_vec());
            g.resize(padded, 1.0);
            u.resize(padded, 1.0);
            let g = DeviceBuffer::from_host(stream, &g).expect("gate");
            let u = DeviceBuffer::from_host(stream, &u).expect("up");
            let mut y = DeviceBuffer::from_host(stream, &vec![GUARD; padded]).expect("y");
            k.enqueue_geglu(stream, &g, &u, n, &mut y).expect("geglu");
            stream.synchronize().expect("sync");
            let mut y = y.to_host_vec(stream).expect("y back");
            assert!(
                y[n..].iter().all(|v| v.to_bits() == GUARD.to_bits()),
                "geglu: output past n must stay untouched"
            );
            y.truncate(n);
            y
        };
        let (gate, up) = inputs();
        let y = run(&gate, &up);
        let (mut over, mut worst, mut host_equal) = (0usize, 0.0f64, 0usize);
        for (i, (&g, &u)) in gate.iter().zip(&up).enumerate() {
            assert!(y[i].is_finite(), "geglu: gate {g:e} up {u:e}: {}", y[i]);
            let (want, bound) = exact(g, u);
            let e = (f64::from(y[i]) - want).abs();
            if e > bound {
                over += 1;
                if over <= 8 {
                    eprintln!(
                        "geglu: gate {g:e} up {u:e}: {} off {want:e} by {e:e} > {bound:e}",
                        y[i]
                    );
                }
            }
            if bound > 0.0 {
                worst = worst.max(e / bound);
            }
            host_equal += usize::from(y[i].to_bits() == (gelu_tanh_f32(g) * u).to_bits());
        }
        println!(
            "geglu: {} values, {over} past their bound, worst error/bound {worst:.3}, {host_equal} \
             bit for bit the host rule",
            gate.len()
        );
        assert_eq!(over, 0, "geglu: values past their bound");

        // A NaN gate at −1.61, NaN up values at gates −16 (saturated) and 0.73.
        let (mut g2, mut u2) = (gate[3584..4608].to_vec(), up[3584..4608].to_vec());
        g2[0] = -16.0;
        u2[0] = f32::NAN;
        g2[100] = f32::NAN;
        u2[700] = f32::NAN;
        let y2 = run(&g2, &u2);
        let nan: Vec<usize> = (0..y2.len()).filter(|&i| y2[i].is_nan()).collect();
        assert_eq!(
            nan,
            [0, 100, 700],
            "geglu: a NaN input is NaN at its index alone"
        );

        for (gate_len, up_len, y_len, n, what) in [
            (9, 9, 9, 0, "an empty launch"),
            (8, 9, 9, 9, "a short gate buffer"),
            (9, 8, 9, 9, "a short up buffer"),
            (9, 9, 8, 9, "a short y buffer"),
        ] {
            let g = DeviceBuffer::from_host(stream, &gate[..gate_len]).expect("gate");
            let u = DeviceBuffer::from_host(stream, &up[..up_len]).expect("up");
            let mut y = DeviceBuffer::<f32>::zeroed(stream, y_len).expect("y");
            let e = k.enqueue_geglu(stream, &g, &u, n, &mut y).expect_err(what);
            assert!(e.to_string().contains("enqueue_geglu"), "{what}: {e}");
        }
    }
}
