//! A Raon-1 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over
//! this family's reading. What each layer holds is read off the description (an RDA layer's
//! state and conv ring, a latent layer's plane of cached rows), and a machine that hangs an
//! expert tier card is refused by name ([`PlaceError::Tier`]) until a body serves one. The down
//! arm admits the common downs, a Q6_K down included, beside a gate and up the leg runs, and a
//! packed Q5_0. The family's own is the reading, and the lanes a verifying load's delta layers
//! hold ([`VERIFY_LANES`]).

use std::num::NonZeroUsize;

use gguf::Split;

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::place_hybrid;
use crate::placement::PlacementError;

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Raon-1 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Raon-1";

/// The lanes each delta layer's state holds on a load that verifies a draft's rows: one a row of
/// the widest verify ([`PlanInputs::with_lanes`]). The card body binds its own constant to it.
pub const VERIFY_LANES: NonZeroUsize = NonZeroUsize::new(8).expect("eight lanes");

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]). The reader refuses a next-token block by its key.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that plans or describes it whatever the coverage list holds,
    /// and the host's room and the arena's given bytes.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let spec = spec::spec_of(
            &hp,
            chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::THINK))?,
        )?;
        PlanInputs::new(ENGINE, hp, model, spec)
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;
    use models::Mixer;
    use runtime::stores::{delta_lane_bytes, recurrent_bytes};

    use super::super::hparams::tests::{keys, tensors};
    use super::{KvLayout, PlaceError, PlanInputs, VERIFY_LANES};
    use crate::arch::synthetic::{V, header_shaped, host_run_experts};
    use crate::placement::{KvBytes, PlacementError, PlanLevers, workstation};

    /// The small header's file, opened; each test names its own file (the tests run in threads of
    /// one process).
    fn opened(name: &str) -> Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("raon")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(name, "raon1", &keys(), &global, &tensors());
        let split = Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The small header's file, described.
    fn described(name: &str) -> PlanInputs {
        let mut inputs =
            PlanInputs::describe(&opened(name)).expect("the synthetic header describes");
        host_run_experts(&mut inputs.model);
        inputs
    }

    /// A card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: a delta layer 4 heads of 16 x 16 f32 (4,096 B) + 11 ring rows of 3 x 4 x 16 f32
    /// (8,448 B) + a lane's u32 stamp = 12,548 B whatever the context; a latent layer a row of
    /// 32 + 8 f16 values (80 B) a position, 64 positions 5,120 B. At 64 positions the 4 delta and
    /// 2 latent layers hold 4 x 12,548 + 2 x 5,120 = 60,432 B, and a layer past the model none.
    /// Mutant: the unrotated values left out of the row, or a window on the latent layer.
    #[test]
    fn the_layers_stores_are_the_headers_widths() {
        let inputs = described("raon1-place-kv");
        let kv: &KvLayout = &inputs.kv;
        for l in [0, 1, 3, 4] {
            assert_eq!(kv.layer_bytes(l, 1), 12_548, "layer {l}");
            assert_eq!(kv.layer_bytes(l, 1 << 20), 12_548, "layer {l}");
        }
        for l in [2, 5] {
            assert_eq!(kv.layer_bytes(l, 1), 80, "layer {l}");
            assert_eq!(kv.layer_bytes(l, 64), 5_120, "layer {l}");
        }
        assert_eq!(kv.layer_bytes(6, 64), 0);
        assert_eq!(kv.bytes(0..6, 64), 60_432);
        assert_eq!(kv.stage_position_bytes(64), 0, "no layer rotates");
        let Mixer::DeltaRule(d) = &inputs.spec.layers[0].mixer else {
            panic!("layer 0 is a delta layer");
        };
        assert_eq!(
            recurrent_bytes(4, 4, 16, d.conv as usize) + delta_lane_bytes(4, 16, 1),
            12_548
        );
    }

    /// A verifying load's delta layer holds [`VERIFY_LANES`] stamped lanes [derived from the
    /// header's widths]: 12,548 B + 7 more lanes of 4 heads of 16 x 16 f32 (28,672 B) and their
    /// stamps (28 B) = 41,248 B whatever the context; a latent layer's plane does not move. Mutant:
    /// the layer's bytes not reading the lanes.
    #[test]
    fn a_verifying_load_holds_a_lane_a_row() {
        let inputs = described("raon1-place-lanes").with_lanes(VERIFY_LANES);
        let kv: &KvLayout = &inputs.kv;
        for l in [0, 1, 3, 4] {
            assert_eq!(kv.layer_bytes(l, 1), 41_248, "layer {l}");
            assert_eq!(kv.layer_bytes(l, 1 << 20), 41_248, "layer {l}");
        }
        assert_eq!(kv.layer_bytes(2, 64), 5_120);
        assert_eq!(kv.bytes(0..6, 64), 4 * 41_248 + 2 * 5_120);
    }

    /// The file is described whatever the coverage holds, and the body's program plans it; a
    /// context of no position is refused by its own name, and the read still refuses the small
    /// header through the coverage check (its widths are not Raon's).
    #[test]
    fn the_described_file_plans_on_the_body() {
        let inputs = described("raon1-place-program");
        let machine = workstation::plan_gate(6);
        let levers = PlanLevers::default();
        if let Err(e) = inputs.plan(&machine, 64, &levers) {
            panic!("{e:?}, not a plan on the body");
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
        match PlanInputs::read(&opened("raon1-place-read")) {
            Err(PlacementError::Unimplemented(items)) => assert!(!items.is_empty()),
            other => panic!("{other:?}, not the coverage refusal"),
        }
    }
}
