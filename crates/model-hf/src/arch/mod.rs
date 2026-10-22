//! One module per model architecture whose checkpoint the hub holds.

pub mod glm5next;
pub mod qwen35moe;

/// A checkpoint's `-exp(x)`, computed in f64 and rounded once to finite nonzero F32.
fn neg_exp(x: f32) -> Result<f32, String> {
    if !x.is_finite() {
        return Err(format!("non-finite input {x}"));
    }
    let y = (-f64::from(x).exp()) as f32;
    if y.is_finite() && y != 0.0 {
        Ok(y)
    } else {
        Err(format!(
            "-exp({x}) = {y} is outside its finite nonzero F32 range"
        ))
    }
}

/// The classes of weight modules the checkpoint maps read. Projection, embedding, norm and
/// convolution are distinct even when they share a placement role or a two-dimensional shape.
pub fn module_class(module: &str) -> Option<hf::quant_config::ModuleClass> {
    use hf::quant_config::ModuleClass::{Embedding, Linear};
    let short = module
        .strip_prefix("model.language_model.")
        .or_else(|| module.strip_prefix("model."))
        .or_else(|| module.strip_prefix("mtp."))
        .unwrap_or(module);
    match short {
        "embed_tokens" => return Some(Embedding),
        "lm_head" | "fc_embedding" | "fc_hidden" => return Some(Linear),
        "hyper_connection_mixer.input_mix_weight_down"
        | "hyper_connection_mixer.input_mix_weight_up" => return Some(Linear),
        _ => {}
    }
    let (layer, rest) = short.strip_prefix("layers.")?.split_once('.')?;
    layer.parse::<usize>().ok()?;
    if let Some(expert) = rest.strip_prefix("mlp.experts.") {
        let (id, projection) = expert.split_once('.')?;
        id.parse::<usize>().ok()?;
        return matches!(projection, "gate_proj" | "up_proj" | "down_proj").then_some(Linear);
    }
    if let Some(part) = rest.strip_prefix("ple.ple_embedding.ngram_embedding.shard_") {
        part.parse::<usize>().ok()?;
        return Some(Embedding);
    }
    if let Some(mixer) = rest
        .strip_prefix("attn_hyper_connection.")
        .or_else(|| rest.strip_prefix("mlp_hyper_connection."))
    {
        return matches!(mixer, "input_mix_weight_down" | "input_mix_weight_up").then_some(Linear);
    }
    matches!(
        rest,
        "eh_proj"
            | "self_attn.q_proj"
            | "self_attn.k_proj"
            | "self_attn.v_proj"
            | "self_attn.o_proj"
            | "self_attn.f_a_proj"
            | "self_attn.f_b_proj"
            | "self_attn.g_a_proj"
            | "self_attn.g_b_proj"
            | "self_attn.b_proj"
            | "self_attn.q_a_proj"
            | "self_attn.q_b_proj"
            | "self_attn.kv_a_proj_with_mqa"
            | "self_attn.kv_b_proj"
            | "self_attn.indexer.wq_b"
            | "self_attn.indexer.wk"
            | "self_attn.indexer.weights_proj"
            | "self_attn.indexer.index_qk_proj"
            | "linear_attn.in_proj_qkv"
            | "linear_attn.in_proj_z"
            | "linear_attn.in_proj_a"
            | "linear_attn.in_proj_b"
            | "linear_attn.out_proj"
            | "mlp.gate"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
            | "mlp.shared_experts.gate_proj"
            | "mlp.shared_experts.up_proj"
            | "mlp.shared_experts.down_proj"
            | "mlp.shared_expert.gate_proj"
            | "mlp.shared_expert.up_proj"
            | "mlp.shared_expert.down_proj"
            | "mlp.shared_expert_gate"
            | "ple.key_proj"
            | "ple.value_proj"
    )
    .then_some(Linear)
}

#[cfg(test)]
mod tests {
    use super::neg_exp;
    use hf::quant_config::{Config, ModuleClass, plan};
    use safetensors::Dtype;

    #[test]
    fn neg_exp_checks_inputs_and_rounded_results() {
        assert_eq!(neg_exp(0.0).unwrap(), -1.0);
        assert_eq!(neg_exp(-0.0).unwrap(), -1.0);
        assert!(neg_exp(-100.0).unwrap().is_subnormal());
        for x in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let e = neg_exp(x).unwrap_err();
            assert!(e.contains("non-finite input"), "{x}: {e}");
        }
        for x in [100.0, -1000.0] {
            let e = neg_exp(x).unwrap_err();
            assert!(e.contains("finite nonzero F32 range"), "{x}: {e}");
        }
    }

    #[test]
    fn class_targets_select_projection_and_embedding_roles_not_shapes() {
        use ModuleClass::{Embedding, Linear};
        let modules = [
            ("model.embed_tokens", Some(Embedding)),
            ("model.language_model.embed_tokens", Some(Embedding)),
            ("lm_head", Some(Linear)),
            ("model.layers.0.self_attn.q_proj", Some(Linear)),
            (
                "model.language_model.layers.0.linear_attn.in_proj_a",
                Some(Linear),
            ),
            ("model.layers.3.mlp.experts.0.down_proj", Some(Linear)),
            (
                "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0",
                Some(Embedding),
            ),
            ("model.layers.0.input_layernorm", None),
            ("model.language_model.layers.0.linear_attn.conv1d", None),
            (
                "model.language_model.layers.0.attn_hyper_connection.block_inject_weight",
                None,
            ),
            ("model.layers.0.self_attn.unknown_proj", None),
        ];
        let tensors: Vec<_> = modules
            .iter()
            .flat_map(|(module, _)| {
                [
                    (format!("{module}.weight_packed"), Dtype::I32, vec![8, 16]),
                    (format!("{module}.weight_scale"), Dtype::Bf16, vec![8, 1]),
                    (format!("{module}.weight_shape"), Dtype::I64, vec![2]),
                ]
            })
            .collect();
        for (ignore, ignored) in [
            ("", None),
            ("Linear", Some(Linear)),
            ("Embedding", Some(Embedding)),
        ] {
            let config = Config::parse(
                &serde_json::json!({"quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"g": {
                        "format": "pack-quantized", "targets": ["Linear", "Embedding"],
                        "weights": {"num_bits": 4, "type": "int", "strategy": "group",
                            "group_size": 128, "symmetric": true}
                    }},
                    "ignore": if ignore.is_empty() { vec![] } else { vec![ignore] }
                }}),
                None,
            )
            .unwrap();
            let placed = plan(
                &config,
                tensors
                    .iter()
                    .map(|(name, dtype, shape)| (name.as_str(), *dtype, shape.as_slice())),
                super::module_class,
            );
            for (module, class) in modules {
                let name = format!("{module}.weight_packed");
                if class.is_some() && class != ignored {
                    assert_eq!(
                        placed.placed.get(&name).map(|p| p.format),
                        Some(hf::quant_config::Format::Int4Pack { group: 128 }),
                        "{ignore}/{name}: {placed:?}"
                    );
                } else {
                    let why = &placed.refused[&name];
                    assert!(
                        why.contains(module) && why.contains("unquantized"),
                        "{ignore}/{why}"
                    );
                }
            }
        }
    }
}
