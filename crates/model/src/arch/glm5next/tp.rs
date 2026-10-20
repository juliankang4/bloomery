//! How tensor parallel splits each glm5next tensor ([`cut`]): along the dim and by the unit a
//! rank's part holds whole there, or whole on every rank. The dims are ggml's order, the file's
//! and the checkpoint map's alike (`model_hf::arch::glm5next`), so one rule serves the GGUF rank
//! load (`gpu-glm5next`'s `tp`) and the checkpoint's plan; the plan of the cuts is
//! [`plan_tp`](crate::placement::tp::plan_tp)'s.
//!
//! KDA and latent attention split by head: their q, k, v, gate and beta rows, the conv channels,
//! the decay and the latent heads' `q_b`, `k_b` and `v_b`, and the output projection along its
//! input. Dense blocks, the shared expert and every routed expert split by inner width: gate and
//! up by row, down along its input. The head splits by vocabulary row. The norms, the low-rank
//! inputs every head shares (`ssm_f_a`, `ssm_g_a`, `attn_q_a`, `attn_kv_a_mqa`), the indexer, the
//! router and its bias, the hyper-connection mixes, the next-token layer's own tensors and the
//! embedding stay whole on every rank.

use crate::placement::tp::Cut;

/// Where the glm5next tensor `name` of `dims` splits among ranks, `None` for whole on every rank.
/// A head's part along a dim is that dim over `heads`. A stem with no rule, a split tensor of
/// other dims than its stem has, and a head split of an extent that is not whole heads are
/// refused, the reason the error.
pub fn cut(name: &str, dims: &[u64], heads: u64) -> Result<Option<Cut>, String> {
    let stem = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, stem)| stem);
    // The dims the tensor has, the dim it splits along, and whether by head or by one value.
    let (rank, dim, by_head) = match stem {
        "token_embd.weight"
        | "output_norm.weight"
        | "attn_norm.weight"
        | "ffn_norm.weight"
        | "ssm_f_a.weight"
        | "ssm_g_a.weight"
        | "ssm_norm.weight"
        | "attn_q_a.weight"
        | "attn_q_a_norm.weight"
        | "attn_kv_a_mqa.weight"
        | "attn_kv_a_norm.weight"
        | "ffn_gate_inp.weight"
        | "exp_probs_b.bias" => return Ok(None),
        _ if stem.starts_with("indexer")
            || stem.starts_with("hc_")
            || stem.starts_with("nextn.") =>
        {
            return Ok(None);
        }
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "ssm_f_b.weight"
        | "ssm_g_b.weight" | "ssm_beta.weight" | "attn_q_b.weight" => (2, 1, true),
        "attn_output.weight" => (2, 0, true),
        "ssm_a" | "ssm_dt.bias" => (1, 0, true),
        "ssm_conv1d_q.weight"
        | "ssm_conv1d_k.weight"
        | "ssm_conv1d_v.weight"
        | "attn_k_b.weight"
        | "attn_v_b.weight" => (3, 2, true),
        "ffn_gate.weight"
        | "ffn_up.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "output.weight" => (2, 1, false),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => (3, 1, false),
        "ffn_down.weight" | "ffn_down_shexp.weight" => (2, 0, false),
        "ffn_down_exps.weight" => (3, 0, false),
        _ => return Err("no tensor-parallel split rule".into()),
    };
    if dims.len() != rank {
        return Err(format!("{dims:?} is not {rank}-D"));
    }
    let extent = dims[dim];
    if !by_head {
        return Ok(Some(Cut { dim, unit: 1 }));
    }
    if heads == 0 || !extent.is_multiple_of(heads) {
        return Err(format!("dim {dim} of {extent} is not {heads} heads"));
    }
    Ok(Some(Cut {
        dim,
        unit: extent / heads,
    }))
}

#[cfg(test)]
mod tests {
    use super::{Cut, cut};

    /// Each kind of split at the GLM fixture's dims (64 heads), the whole tensors, and the
    /// refusals: a stem with no rule, a split tensor of another dim count, and a head split of an
    /// extent that is not whole heads.
    #[test]
    fn every_stem_cuts_by_its_rule_and_the_rest_is_refused() {
        let at = |dim, unit| Ok(Some(Cut { dim, unit }));
        for (name, dims, want) in [
            ("blk.0.attn_q.weight", &[4096u64, 8192][..], at(1, 128)),
            ("blk.0.attn_output.weight", &[8192, 4096], at(0, 128)),
            ("blk.0.ssm_a", &[64], at(0, 1)),
            ("blk.0.ssm_conv1d_v.weight", &[4, 1, 8192], at(2, 128)),
            ("blk.3.attn_q_b.weight", &[1536, 16384], at(1, 256)),
            ("blk.3.attn_k_b.weight", &[256, 512, 64], at(2, 1)),
            ("blk.0.ffn_down.weight", &[12288, 4096], at(0, 1)),
            ("blk.3.ffn_up_exps.weight", &[4096, 512, 288], at(1, 1)),
            ("blk.3.ffn_down_exps.weight", &[512, 4096, 288], at(0, 1)),
            ("output.weight", &[4096, 154880], at(1, 1)),
        ] {
            assert_eq!(cut(name, dims, 64), want, "{name}");
        }
        for name in [
            "token_embd.weight",
            "blk.0.ssm_f_a.weight",
            "blk.3.attn_kv_a_mqa.weight",
            "blk.3.indexer.attn_k.weight",
            "blk.0.hc_attn_fn.weight",
            "blk.6.nextn.eh_proj.weight",
        ] {
            assert_eq!(cut(name, &[4096, 128], 64), Ok(None), "{name}");
        }
        assert_eq!(
            cut("blk.3.attn_sinks.weight", &[64], 64),
            Err("no tensor-parallel split rule".to_string())
        );
        for (name, dims) in [
            ("blk.3.ffn_up_exps.weight", &[4096u64, 288][..]),
            ("blk.0.ffn_down.weight", &[4096, 2048, 2]),
            ("blk.0.ssm_a", &[64, 1]),
        ] {
            assert!(
                cut(name, dims, 64).is_err_and(|e| e.ends_with("-D")),
                "{name}"
            );
        }
        assert!(cut("blk.0.attn_q.weight", &[4096, 8190], 64).is_err());
        assert!(cut("blk.0.attn_q.weight", &[4096, 8192], 0).is_err());
    }
}
