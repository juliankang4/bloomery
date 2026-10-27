//! Flint's checked description and host views for the common card-leg tier.

use std::ops::Range;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths};
use bloomery_gpu::host::node::NodeFamily;
use gguf::Split;
use model::ModelError;
use model::arch::flint::{host, place::PlanInputs};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::accepts::Program;
use model::r8file::R8Source;
use models::LayerSpec;

/// Maximum tokens per expert-tier launch; the placement reserves this capacity.
pub const CHUNK: usize = model::arch::flint::place::TIER_CHUNK as usize;

struct Flint;

impl LegFamily for Flint {
    const WHAT: &'static str = "flint tier";
    const WORD: &'static str = "flint";
    const ROWS: usize = 1;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

    fn describe(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(Self::WHAT, e))
    }

    fn into_model(inputs: PlanInputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &PlanInputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    fn widths(inputs: &PlanInputs) -> LegWidths {
        LegWidths {
            n_expert: 256,
            n_used: 6,
            n_embd: inputs.spec.hidden as usize,
            ff: 512,
            program: Program::FlintBody,
        }
    }

    fn expert_input(_inputs: &PlanInputs) -> bloomery_gpu::card_leg::ExpertInput {
        bloomery_gpu::card_leg::ExpertInput::AmaxF16
    }

    fn routed(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError> {
        run.map(|l| host::layer(src, &inputs.spec, l)).collect()
    }
}

/// The tier service uses the same Program and host arithmetic as the stage card.
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<Flint>();
