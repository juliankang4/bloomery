//! Flint's composition of the common head128 attention, shared routed block and host tier.
//! Device-linked consumers build this crate with `cargo oxide`.

pub mod body;
pub mod prefill;
mod step;
pub mod tier;

pub use body::{Body, FlintModel, Seq, StepInput, set_taps};
pub use prefill::{prefill, set_prefill_group};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod two_card_tests;
