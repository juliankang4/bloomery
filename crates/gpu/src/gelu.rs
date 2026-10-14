//! The tanh GELU (`ggml_gelu`): `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`, one rule on one
//! f32 for every kernel that applies it, and its host twin.
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
