//! Flint's validated routed block as the common host and NVMe expert tiers serve it.

use models::{Act, Ffn, ModelSpec};

use crate::ModelError;
use crate::arch::names;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::ops::ExpertInput;
use crate::r8file::R8Source;

/// Build the requested layer's routed stacks with Flint's unclamped SwiGLU and activated-value
/// scaling. Dense layers, layers outside the description and non-SwiGLU routed blocks are refused
/// before a tensor is opened. Load-time only.
pub fn layer(src: R8Source<'_>, spec: &ModelSpec, l: usize) -> Result<HostLayer, ModelError> {
    let refuse = |detail: String| ModelError::Metadata {
        key: format!("layer {l} routed block"),
        detail,
    };
    let block = spec
        .layers
        .get(l)
        .ok_or_else(|| refuse(format!("outside the file's {} layers", spec.layers.len())))?;
    let Ffn::Moe(moe) = &block.ffn else {
        return Err(refuse("is dense, not a routed block".to_string()));
    };
    if moe.act != (Act::SwiGlu { limit: None }) {
        return Err(refuse(format!(
            "activation {:?}; Flint's host experts require unclamped SwiGLU",
            moe.act
        )));
    }
    let [gate, up, down] = names::routed_stacks(l);
    let mut layer = HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: moe.experts as usize,
            embd: spec.hidden as usize,
            ff: moe.expert_ff as usize,
            swiglu_limit: 0.0,
        },
    )?;
    layer.set_expert_input(ExpertInput::AmaxF16);
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{keys, tensors};
    use super::*;
    use crate::arch::synthetic::{V, header_shaped};

    #[test]
    fn refuses_dense_outside_and_non_swiglu_before_opening_experts() {
        let path = header_shaped(
            "flint-host-layer",
            "flint",
            &keys(),
            &[("tokenizer.ggml.pre", V::Str("flint"))],
            &tensors(),
        );
        let split = gguf::Split::open(&path).unwrap();
        let mut spec = super::super::spec::read(&split).unwrap().spec;
        let src = R8Source::rows(&split);
        for (l, fragment) in [(0, "is dense"), (4, "outside the file's 4 layers")] {
            let error = layer(src, &spec, l).err().expect("typed refusal");
            assert!(matches!(error, ModelError::Metadata { ref key, ref detail }
                if key == &format!("layer {l} routed block") && detail.contains(fragment)));
        }
        let Ffn::Moe(moe) = &mut spec.layers[1].ffn else {
            panic!("routed");
        };
        moe.act = Act::ReluSqr;
        let error = layer(src, &spec, 1)
            .err()
            .expect("non-gated activation refused");
        assert!(matches!(error, ModelError::Metadata { ref detail, .. }
            if detail.contains("require unclamped SwiGLU")));
        std::fs::remove_file(path).unwrap();
    }
}
