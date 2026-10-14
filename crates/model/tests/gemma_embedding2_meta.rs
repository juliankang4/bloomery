//! gemma-embedding2 metadata gate: what the header reader resolves from the
//! EmbeddingGemma 2 Q8_0 file (ggml-org/embeddinggemma-2-GGUF). The header
//! test prints the summary, then fails with every pin that differs. The
//! override test refuses a path that is not UTF-8 before opening a file.
//!
//! `hw_`: needs the file on the box (`just gate-gemma-embedding2-meta`), at
//! the box's path or at `BLOOMERY_MODEL` inside the box
//! (`BLOOMERY_BOX_ENV='BLOOMERY_MODEL=<path>'`; on the calling side box.sh
//! reads `BLOOMERY_MODEL` as a profile). Headers only: the reader holds every
//! tensor to the dims the keys give it, so the tensor headers are read and
//! the bytes are not.

use std::fs;

use models::{Act, Ffn, Mixer, RopeMode};

/// The box's path of the file.
const FILE: &str = "/models/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";

/// Its length, so a different file at the same path fails before the pins.
const BYTES: u64 = 309_855_456;

/// The full layers (`sliding_window_pattern` false): head 512, one key head.
const FULL: &[usize] = &[5, 11, 17, 23];

#[test]
#[ignore = "needs the EmbeddingGemma 2 Q8_0 file on the box (just gate-gemma-embedding2-meta)"]
fn hw_gemma_embedding2_header() {
    let file = std::env::var_os("BLOOMERY_MODEL").map_or_else(
        || FILE.to_string(),
        |v| v.into_string().expect("BLOOMERY_MODEL must be UTF-8"),
    );
    let mut bad = Vec::new();
    let len = fs::metadata(&file).map(|m| m.len());
    let split = gguf::Split::open(&file).unwrap_or_else(|e| panic!("open {file}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {file}: {e}"));
    let hp = model::arch::gemma_embedding2::hparams::Hparams::read(&split)
        .unwrap_or_else(|e| panic!("hparams of {file}: {e}"));
    let s = &read.spec;
    let e = s.encoder.as_ref().expect("an encoder spec");
    println!(
        "gemma-embedding2 bytes={len:?} layers={} hidden={} vocab={} ctx={} window={} reach={:?} \
         per_layer={:?} proj={:?} tensors={} defaults={:?}",
        s.layers.len(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        hp.window,
        e.reach,
        e.per_layer,
        e.proj,
        read.tensors.tensors.len(),
        read.defaults
    );
    if len.as_ref().ok() != Some(&BYTES) {
        bad.push(format!("file bytes {len:?}, want {BYTES}"));
    }
    if s.arch != models::Arch::GemmaEmbedding2 {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (512, 262_144, 262_144) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps.to_bits() != 1e-6_f32.to_bits() {
        bad.push(format!("rms_eps bits {:#x}", s.rms_eps.to_bits()));
    }
    if !s.mtp.is_empty() || s.hc.is_some() || s.engram.is_some() {
        bad.push(format!(
            "mtp {} hc {:?} engram {:?}",
            s.mtp.len(),
            s.hc,
            s.engram
        ));
    }
    if s.chat.pre != "gemma4" || s.chat.tools.is_some() || s.chat.reasoning.is_some() {
        bad.push(format!("chat {:?}", s.chat));
    }
    if read.defaults
        != ["tokenizer.ggml.pre = gemma4 (tokenizer.ggml.model gemma4, as llama.cpp sets it)"]
    {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    if s.layers.len() != 24 {
        bad.push(format!("layers {}", s.layers.len()));
    }
    for (l, layer) in s.layers.iter().enumerate() {
        let full = FULL.contains(&l);
        let (head, kv, base) = if full { (512, 1, 1e6) } else { (256, 2, 1e4) };
        match &layer.mixer {
            Mixer::Gqa(g) => {
                if (g.heads, g.kv_heads, g.head_dim, g.value_dim) != (4, kv, head, head)
                    || (g.rope.mode, g.rope.dims, g.rope.base) != (RopeMode::Neox, head, base)
                    || !g.qk_norm
                    || g.window.is_some()
                {
                    bad.push(format!("layer {l} gqa {g:?}"));
                }
            }
            other => bad.push(format!("layer {l} mixer {other:?}, want Gqa")),
        }
        if !matches!(
            layer.ffn,
            Ffn::Dense {
                ff: 2048,
                act: Act::GeGlu
            }
        ) {
            bad.push(format!("layer {l} ffn {:?}", layer.ffn));
        }
        let reach = (!full).then_some(512);
        if e.reach.get(l) != Some(&reach) {
            bad.push(format!(
                "layer {l} reach {:?}, want {reach:?}",
                e.reach.get(l)
            ));
        }
    }
    if (e.per_layer, e.proj.as_slice()) != (Some(512), &[768][..]) {
        bad.push(format!("per_layer {:?} proj {:?}", e.per_layer, e.proj));
    }
    let items: Vec<String> = model::arch::coverage::check(s, &read.tensors)
        .into_iter()
        .map(|u| u.feature)
        .collect();
    for want in [
        "bidirectional GQA flash, head 256, group 2, 512 keys each side",
        "bidirectional GQA flash, head 512, group 4",
        "dense GELU GLU layer: ff 2048",
        "a layer program for gemma-embedding2",
    ] {
        if !items.iter().any(|i| i == want) {
            bad.push(format!("coverage has no {want:?}"));
        }
    }
    println!("gemma-embedding2 coverage: {items:?}");
    if read.tensors.tensors.len() != 413 {
        bad.push(format!("tensors {}", read.tensors.tensors.len()));
    }
    let ty = |name: &str| {
        read.tensors
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| (t.ty, t.dims.clone()))
    };
    if ty("per_layer_model_proj.weight") != Some((gguf::GgmlType::BF16, vec![512, 12_288])) {
        bad.push(format!(
            "per_layer_model_proj {:?}",
            ty("per_layer_model_proj.weight")
        ));
    }
    if ty("blk.5.attn_q.weight") != Some((gguf::GgmlType::Q8_0, vec![512, 2048])) {
        bad.push(format!("blk.5.attn_q {:?}", ty("blk.5.attn_q.weight")));
    }
    println!(
        "gemma-embedding2 layers=24 full={FULL:?} hidden=512 out=768 ple=512 ff=2048 \
         pins_differ={}",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// An invalid override is refused before any model file opens.
#[cfg(unix)]
#[test]
#[ignore = "runs the metadata gate in a child process (just gate-gemma-embedding2-meta)"]
fn hw_non_utf8_model_override_is_refused() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::process::Command;

    let output = Command::new(std::env::current_exe().expect("the test executable"))
        .args([
            "--exact",
            "hw_gemma_embedding2_header",
            "--ignored",
            "--nocapture",
        ])
        .env("BLOOMERY_MODEL", OsString::from_vec(vec![0xff]))
        .output()
        .expect("the metadata gate starts");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "an invalid override was accepted");
    assert!(stderr.contains("BLOOMERY_MODEL must be UTF-8"), "{stderr}");
}
