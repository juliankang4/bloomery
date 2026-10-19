//! The activation rules a gate·up kernel of this family applies to a row's
//! two sums: `silu_mul` (`elem::silu_mul`, the reference's scalar SwiGLU
//! with the device `expf`) and ik's clamped SwiGLU [`swiglu_clamp`], whose
//! exponential is a transcription of ik's AVX2 `v_expf` and so rounds the
//! same on the device and the host. The two differ in their exponential, so
//! they are two rules, picked per launch by [`Act`].
//!
//! V4.1 kernels share these rules and the non-finite sum check. Gate-up
//! entries use [`apply_checked`] before writing their rows.

use crate::elem::silu_mul;
use crate::fault::{FaultSink, FaultSite};
pub use crate::linear::silu as silu_ik;

/// A gate·up kernel's activation rule, a launch argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// `silu(g) · u` with the device `expf` (`elem::silu_mul`).
    SiluMul,
    /// ik's clamped SwiGLU at `limit` ([`swiglu_clamp`]; `limit <= 1e-6`
    /// clamps nothing, as ik's own test).
    SwigluClamp { limit: f32 },
}

/// The device code of [`Act::SiluMul`].
pub const ACT_SILU_MUL: u32 = 0;
/// The device code of [`Act::SwigluClamp`].
pub const ACT_SWIGLU_CLAMP: u32 = 1;

impl Act {
    /// The kernel's `(act, limit)` arguments.
    #[must_use]
    pub fn code(self) -> (u32, f32) {
        match self {
            Act::SiluMul => (ACT_SILU_MUL, 0.0),
            Act::SwigluClamp { limit } => (ACT_SWIGLU_CLAMP, limit),
        }
    }
}

/// The rule of code `act` on a row's gate sum `g` and up sum `u`: the
/// launcher passes one of the two codes of [`Act::code`], and `limit` only
/// with [`ACT_SWIGLU_CLAMP`].
#[inline(always)]
#[must_use]
pub fn apply(act: u32, limit: f32, g: f32, u: f32) -> f32 {
    if act == ACT_SWIGLU_CLAMP {
        swiglu_clamp(g, u, limit)
    } else {
        silu_mul(g, u)
    }
}

/// Apply the rule only to finite gate and up sums. Otherwise raise
/// [`FaultSite::F32Product`] on `fault` and return NaN before a clamp can
/// hide the non-finite sum.
#[inline(always)]
#[must_use]
pub fn apply_checked(act: u32, limit: f32, g: f32, u: f32, fault: FaultSink) -> f32 {
    if g.is_finite() && u.is_finite() {
        apply(act, limit, g, u)
    } else {
        fault.raise(FaultSite::F32Product);
        f32::NAN
    }
}

/// ik's clamped SwiGLU for one row: `min(silu(g), limit) · max(-limit,
/// min(limit, u))`, or `silu(g) · u` when `limit <= 1e-6`, each `min`/`max`
/// spelled as the `std::min`/`std::max` comparison it compiles from (a NaN
/// `silu(g)` passes the first; a NaN `u` becomes `limit`).
#[inline(always)]
pub fn swiglu_clamp(g: f32, u: f32, limit: f32) -> f32 {
    let mut s = silu_ik(g);
    let mut uc = u;
    if limit > 1e-6 {
        s = if limit < s { limit } else { s };
        uc = if u < limit { u } else { limit };
        uc = if -limit < uc { uc } else { -limit };
    }
    uc * s
}
