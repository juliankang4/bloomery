//! Flint's validated description, placement inputs and routed host adapter. Full and sliding
//! attention independently select NoPE or NeoX rotation; every layer has head-local q/k norms and
//! attention sinks. Routed blocks carry one ungated shared MLP whose width is explicit, not a
//! count of physical modules. Executable admission belongs to the common program/site registry.

pub mod host;
mod hparams;
pub mod place;
mod roles;
pub mod spec;

#[cfg(test)]
mod tests;
