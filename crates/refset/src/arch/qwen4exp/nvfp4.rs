//! The Qwen3.8-Flash-Next NVFP4 oracle: the forward of `nvidia/Qwen3.8-Flash-Next-NVFP4` in
//! transformers' qwen4_exp, f32 on the CPU, with the weights the split `prepare` writes from that
//! folder holds — the routed experts by ModelOpt's NVFP4 rule, the PLE rows by their F8 rule, the
//! BF16 projections `prepare` writes as Q8_0 rounded by ggml's reference Q8_0 rule
//! (`tools/ref/qwen4exp-nvfp4-oracle.py`). Its sets are in ik's dump layout and carry ik's tap
//! names, so a gate reads them as it reads [`super::IK`]'s; the ids are the batch set's and
//! [`super::D3K`]'s. The two `_bf16` sets widen the checkpoint's BF16 projections instead: the
//! Q8_0 rounding's own share of the difference, recorded beside the oracle.

use super::ARCH;
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The checkpoint folder every set states in its `# model` line, as a node's model store mounts
/// it: the repository at the revision the family pins.
pub const MODEL: &str =
    "/models/nvidia__Qwen3.8-Flash-Next-NVFP4/fc694b54fb0174e0913e6adf86691ef85a4ead47";

/// The md5 of `tools/ref/qwen4exp-nvfp4-oracle.py`, the sets' writer, which states it in each
/// set's `# build` line.
// PIN(2026-10-20): the writer at transformers 4c414256 that dequantizes one decoder layer at a
// time and writes the route ids' logical copy and exact twins as ik's dumper does;
// `the_oracle_writer_pin_is_the_source_in_the_tree` holds it to the tree.
pub const HARNESS: &str = "892a07afd9ac0c7efbd63de165f655bf";

/// The 5-token batch, every position.
pub const BATCH: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4";
/// The 3,001 ids of [`super::D3K`] as a decode-step set: the step at position 3,000, every
/// position's taps kept.
pub const D3K: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k";
/// [`BATCH`] with the BF16 projections widened.
pub const BATCH_BF16: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_bf16";
/// [`D3K`] with the BF16 projections widened: the layer outputs and logits of the last position,
/// the routes of every position.
pub const D3K_BF16: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k_bf16";

/// [`MODEL`], as the family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The NVFP4 checkpoint's forward sets.
pub static HF: Family = Family {
    name: "hf-qwen4exp-nvfp4",
    sets: &[BATCH, D3K, BATCH_BF16, D3K_BF16],
    resolve: None,
    recipe: "just dump-ref-qwen4exp-nvfp4 IDS",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(HARNESS)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-nvfp4"],
};

#[cfg(test)]
mod tests {
    use super::{BATCH, HARNESS, HF, MODEL};
    use crate::RefError;
    use std::path::Path;

    /// The writer's pin is the source in the tree: an edit of the writer without new sets and a
    /// new pin is red.
    #[test]
    fn the_oracle_writer_pin_is_the_source_in_the_tree() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/qwen4exp-nvfp4-oracle.py");
        let source = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            crate::md5::hex_of(&source),
            HARNESS,
            "tools/ref/qwen4exp-nvfp4-oracle.py changed: write the sets again (just \
             dump-ref-qwen4exp-nvfp4) and pin the new md5"
        );
    }

    /// A complete set of the folder and the writer passes; one of the GGUF ik's sets are dumped
    /// from, or of another writer, is refused, naming what it states.
    #[test]
    fn the_family_takes_a_set_of_its_folder_and_writer() -> Result<(), RefError> {
        let dir =
            std::env::temp_dir().join(format!("bloomery-refset-nvfp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
        let check = |model: &str, build: &str| {
            let lines = [
                format!("# model\t{model}"),
                format!("# build\t{build}"),
                "# arch\tqwen4exp".to_string(),
                "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
                "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\thf".to_string(),
                "# complete\t1\t0".to_string(),
            ];
            let path = dir.join("MANIFEST.tsv");
            std::fs::write(&path, lines.join("\n") + "\n")
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            HF.check_set(&dir)
        };
        let p = check(MODEL, HARNESS)?;
        assert_eq!(
            (p.dumped_from.as_str(), p.build.as_deref()),
            (MODEL, Some(HARNESS))
        );
        let gguf = check(super::super::MODEL, HARNESS).expect_err("a set of the GGUF");
        assert!(gguf.to_string().contains(super::super::MODEL), "{gguf}");
        let other = check(MODEL, "0123").expect_err("a set of another writer");
        assert!(other.to_string().contains("0123"), "{other}");
        assert!(HF.sets.contains(&BATCH));
        std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))
    }
}
