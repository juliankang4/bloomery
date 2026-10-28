//! The qwen3moe kernel gates, one process, each against ik's CPU dump of the
//! qwen3moe file: the per-head QK RMS norm and NEOX rope with the K/V cache
//! write (`rope`), the softmax router and the ubatch router (`router`), the
//! down `_sel` kernels (`down`), the Q6_K head with its argmax folded in
//! (`head`), the GQA flash decode passes (`flash`), and the chain's expert
//! kernels (`experts`). A gate is the module of its name
//! (`shared/qwen3moe_<name>_gate.rs`, whose head says what it pins) and runs
//! on a card handle and a reading of the model file of its own; it keeps its
//! name `gate_qwen3moe_<name>` in the lines it prints and in its errors.
//!
//! `gate_qwen3moe_kernels [GATE...]` runs the named gates, or all six in the
//! order below; a gate that fails does not stop the next one, and the process
//! ends red when any failed (`shared/gate_set.rs`).

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_down_gate.rs"]
mod down;
#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_experts_gate.rs"]
mod experts;
#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_flash_gate.rs"]
mod flash;
#[cfg(feature = "gpu")]
#[path = "shared/gate_set.rs"]
mod gate_set;
#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_head_gate.rs"]
mod head;
#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_rope_gate.rs"]
mod rope;
#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_router_gate.rs"]
mod router;

/// The gates in run order.
#[cfg(feature = "gpu")]
const GATES: [&str; 6] = ["rope", "router", "down", "head", "flash", "experts"];

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    // A lever set to a value it does not take, or a retired name that is set,
    // is refused by name before anything loads.
    if let Err(e) = bloomery_levers::at_main(&[]) {
        return bloomery_gpu_gates::exit_with("gate_qwen3moe_kernels", Err(e.into()));
    }
    gate_set::run_set(
        "gate_qwen3moe_kernels",
        "gate_qwen3moe_",
        &GATES,
        std::env::args().skip(1),
        |name| match name {
            "rope" => rope::run(),
            "router" => router::run(),
            "down" => down::run(),
            "head" => head::run(),
            "flash" => flash::run(),
            "experts" => experts::run(),
            other => unreachable!("run_set passes only the names of GATES, got {other:?}"),
        },
    )
}
