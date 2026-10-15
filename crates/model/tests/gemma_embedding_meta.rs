//! The EmbeddingGemma Q8_0 header contract. The file is opened on Linux by
//! `gate-gemma-embedding-meta`; tensor data is never read.

use models::{Act, Ffn, Mixer, RopeMode};

const FILE: &str = "/models/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const BYTES: u64 = 333_590_944;
const FULL: &[usize] = &[5, 11, 17, 23];

#[test]
#[ignore = "needs the EmbeddingGemma Q8_0 file (just gate-gemma-embedding-meta)"]
fn hw_gemma_embedding_header() {
    let file = std::env::var_os("BLOOMERY_MODEL").map_or_else(
        || FILE.to_string(),
        |v| v.into_string().expect("BLOOMERY_MODEL must be UTF-8"),
    );
    assert_eq!(
        std::fs::metadata(&file).expect("the model file").len(),
        BYTES
    );
    let split = gguf::Split::open(&file).unwrap_or_else(|e| panic!("open {file}: {e}"));
    for key in ["attention.causal", "attention.sliding_window_pattern"] {
        assert!(
            split.value(&split.arch_key(key)).is_none(),
            "unexpected {key}"
        );
    }
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {file}: {e}"));
    let hp = model::arch::gemma_embedding::hparams::Hparams::read(&split).expect("hparams");
    let s = &read.spec;
    let e = s.encoder.as_ref().expect("an encoder");
    assert_eq!(s.arch, models::Arch::GemmaEmbedding);
    assert_eq!((s.hidden, s.vocab, s.ctx_train), (768, 262_144, 2048));
    assert_eq!(s.layers.len(), 24);
    assert_eq!(s.rms_eps.to_bits(), 1e-6_f32.to_bits());
    assert_eq!(hp.window, 512);
    assert_eq!(e.per_layer, None);
    assert_eq!(e.proj, [3072, 768]);
    assert_eq!(s.chat.pre, "llama");
    assert!(s.chat.tools.is_none() && s.chat.reasoning.is_none());
    for (l, layer) in s.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("layer {l} is not GQA");
        };
        assert_eq!(
            (g.heads, g.kv_heads, g.head_dim, g.value_dim),
            (3, 1, 256, 256)
        );
        assert_eq!(
            (g.rope.mode, g.rope.dims, g.rope.base),
            (
                RopeMode::Neox,
                256,
                if FULL.contains(&l) { 1e6 } else { 1e4 }
            )
        );
        assert!(g.qk_norm && g.window.is_none());
        assert_eq!(
            layer.ffn,
            Ffn::Dense {
                ff: 1152,
                act: Act::GeGlu
            }
        );
        assert_eq!(e.reach[l], (!FULL.contains(&l)).then_some(256));
    }
    for (name, dims) in [
        ("dense_2.weight", vec![768, 3072]),
        ("dense_3.weight", vec![3072, 768]),
    ] {
        let tensor = read
            .tensors
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect(name);
        assert_eq!(tensor.dims, dims);
        assert_eq!(
            (tensor.role, tensor.layer),
            (model::placement::Role::Head, None)
        );
    }
    let items = model::arch::coverage::check(s, &read.tensors);
    for want in [
        "bidirectional GQA flash, head 256, group 3, 256 keys each side",
        "bidirectional GQA flash, head 256, group 3",
        "dense GELU GLU layer: ff 1152",
        "a layer program for gemma-embedding",
    ] {
        assert!(
            items.iter().any(|u| u.feature == want),
            "missing {want}: {items:?}"
        );
    }
    println!(
        "gemma-embedding bytes={BYTES} layers=24 full={FULL:?} hidden=768 out=768 dense=3072 ff=1152 window=512 head=256 query=3 kv=1 tensors={} defaults={:?}",
        read.tensors.tensors.len(),
        read.defaults
    );
}
