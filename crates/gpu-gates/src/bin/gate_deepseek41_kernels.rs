//! The V4.1 attention-side gates, one process: the hyper-connections (`hc`),
//! the rope family (`rope`), the compressor and index key (`comp`), the
//! indexer (`index`), the engram gate (`engram`), the attention output
//! projection (`woa`), the attention (`attn`), and the two chain pieces
//! (`chain_glue`, `chain_attn`), each against ik's CPU dump of the V4.1 file.
//! A gate is the module of its name (`shared/ds41_<name>_gate.rs`, whose
//! head says what it pins) and runs on a card handle and a reading of the
//! model file of its own; it keeps its name `gate_deepseek41_<name>` in the
//! lines it prints and in its errors.
//!
//! `gate_deepseek41_kernels [GATE...]` runs the named gates, or all nine in
//! the order below; a gate that fails does not stop the next one, and the
//! process ends red when any failed (`shared/gate_set.rs`).

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_attn_gate.rs"]
mod attn;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_chain_attn_gate.rs"]
mod chain_attn;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_chain_glue_gate.rs"]
mod chain_glue;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_comp_gate.rs"]
mod comp;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_cand.rs"]
mod ds41_cand;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_engram_gate.rs"]
mod engram;
#[cfg(feature = "deepseek41")]
#[path = "shared/gate_set.rs"]
mod gate_set;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_hc_gate.rs"]
mod hc;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_index_gate.rs"]
mod index;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_rope_gate.rs"]
mod rope;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_woa_gate.rs"]
mod woa;

/// The gates in run order: the op blocks first, then the pieces that chain them.
#[cfg(feature = "deepseek41")]
const GATES: [&str; 9] = [
    "hc",
    "rope",
    "comp",
    "index",
    "engram",
    "woa",
    "attn",
    "chain_glue",
    "chain_attn",
];

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    // A lever set to a value it does not take, or a retired name that is set,
    // is refused by name before anything loads.
    let levers = match bloomery_levers::at_main(&[bloomery_levers::ENGRAM_HELPER]) {
        Ok(levers) => levers,
        Err(e) => return bloomery_gpu_gates::exit_with("gate_deepseek41_kernels", Err(e.into())),
    };
    gate_set::run_set(
        "gate_deepseek41_kernels",
        "gate_deepseek41_",
        &GATES,
        std::env::args().skip(1),
        |name| match name {
            "hc" => hc::run(),
            "rope" => rope::run(),
            "comp" => comp::run(),
            "index" => index::run(),
            "engram" => engram::run(),
            "woa" => woa::run(),
            "attn" => attn::run(),
            "chain_glue" => chain_glue::run(&levers),
            "chain_attn" => chain_attn::run(),
            other => unreachable!("run_set passes only the names of GATES, got {other:?}"),
        },
    )
}
