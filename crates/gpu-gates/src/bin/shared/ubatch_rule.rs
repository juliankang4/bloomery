//! The prompt ubatch size a seat's user names ([`ubatch_rule`]): `-ub`/`--ubatch-size` over
//! `BLOOMERY_QWEN3_UBATCH`, read once for the seat's plans and its load — for the qwen3 seat's dense
//! loads (`qwen3moe_place`), the decide seat's backbone and the Qwen3.8 seat, each including this
//! file. It lives beside the bins, not in the lib, because it reads the qwen3moe body's lever.

use bloomery_gpu::arch::qwen3moe::ubatch::{UBATCH, UBATCH_ENV, ubatch_asked};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::ubatch_pick::Rule;

/// The rule a load whose prompt ubatch the user may name takes it by ([`Rule::of`] at the engine's
/// cap): `flag` (`-ub`, `--ubatch-size`) over `BLOOMERY_QWEN3_UBATCH`, else the load's own choice.
pub fn ubatch_rule(flag: Option<usize>) -> Result<Rule, GateError> {
    let env = ubatch_asked()?.map(|size| (size, UBATCH_ENV));
    Ok(Rule::of(flag, env, UBATCH)?)
}
