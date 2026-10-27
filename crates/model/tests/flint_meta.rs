//! Flint's pinned UD-Q4_K_XL header gate. Accepts a full GGUF or its bounded
//! header prefix via BLOOMERY_MODEL; never reads or synthesizes tensor payloads.
//! The raw header hash pins identity before rebuilding a strict header-only Split.

use gguf::write::{Layout, TensorDecl};
use models::{Ffn, Mixer, Need, RopeMode};
use std::fs::File;
use std::io::Read;

const HEADER_BYTES: u64 = 5_470_528;
const HEADER_SHA: &str = "fb0cc4b520f927f0b679193397ee0c36ca252426e7881283c5aa6de3e44a3a2b";
const FULL: &[usize] = &[0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48];

#[test]
#[ignore = "needs pinned Flint GGUF/header prefix in BLOOMERY_MODEL (just gate-flint-meta)"]
fn hw_flint_header() {
    let path = std::env::var_os("BLOOMERY_MODEL")
        .expect("BLOOMERY_MODEL must name the pinned Flint GGUF or header prefix");
    let path = std::path::PathBuf::from(path);
    let inv = gguf::inventory_of(&path).expect("the GGUF header parses");
    assert_eq!(inv.data_base, HEADER_BYTES);
    let mut prefix = File::open(&path).unwrap().take(HEADER_BYTES);
    assert_eq!(
        gguf::digest::sha256_reader(&path, &mut prefix).unwrap(),
        HEADER_SHA
    );
    let decls = inv
        .tensors
        .iter()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.type_id,
            nbytes: t.nbytes.expect("every Flint type has a known size"),
        })
        .collect();
    let layout = Layout::new(&inv.meta, decls).expect("strict tensor layout");
    let split = gguf::Split::header_only(path, &layout).expect("strict header-only Split");
    let r = model::arch::spec(&split).expect("Flint reader");
    let s = &r.spec;
    assert_eq!(s.arch, models::Arch::Flint);
    assert_eq!(
        (s.hidden, s.vocab, s.ctx_train, s.layers.len()),
        (2048, 150144, 262144, 49)
    );
    assert_eq!(s.rms_eps.to_bits(), 1e-6_f32.to_bits());
    assert_eq!(r.tensors.tensors.len(), 779);
    assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none() && s.encoder.is_none());
    assert_eq!(s.chat.pre, "flint");
    assert_eq!(s.chat.tools, Some(models::ToolFormat::GlmXml));
    assert_eq!(s.chat.reasoning, Some(models::ReasoningFormat::THINK));
    let mut routed = 0;
    for (l, layer) in s.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("layer {l} not GQA")
        };
        assert_eq!(
            (g.heads, g.kv_heads, g.head_dim, g.value_dim),
            (32, 4, 128, 128)
        );
        assert!(g.qk_norm && g.sinks);
        let full = FULL.contains(&l);
        assert_eq!(g.window, if full { None } else { Some(513) });
        assert_eq!(
            g.rope.mode,
            if full { RopeMode::None } else { RopeMode::Neox }
        );
        assert_eq!(g.rope.dims, if full { 0 } else { 128 });
        if l < 2 {
            assert!(matches!(layer.ffn, Ffn::Dense { ff: 6144, .. }));
        } else {
            routed += 1;
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} not routed")
            };
            assert_eq!((m.experts, m.top_k, m.expert_ff), (256, 6, 512));
            assert!(!m.router.bias);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSumFloor);
            assert_eq!(m.router.score, models::Score::Sigmoid);
            assert_eq!(m.router.scale, 1.0);
            let shared = m.shared.as_ref().unwrap();
            assert_eq!(shared.ff, 1024);
            assert!(!shared.sigmoid_gate);
        }
    }
    assert_eq!(routed, 47);
    let needs = models::needs(s);
    let norms = needs
        .iter()
        .filter(|(n, _)| matches!(n, Need::QkRope { qk_norm: true, .. }))
        .count();
    let rotary = needs
        .iter()
        .filter(|(n, _)| {
            matches!(
                n,
                Need::QkRope {
                    mode: RopeMode::Neox,
                    ..
                }
            )
        })
        .count();
    let sinks = needs
        .iter()
        .filter(|(n, _)| matches!(n, Need::Gqa { sinks: true, .. }))
        .count();
    assert_eq!((norms, rotary, sinks), (49, 35, 49));
    for (need, _) in &needs {
        if let Need::Router(shape) = need {
            let binding = model::arch::models::shape::select_router(*shape).unwrap();
            assert_eq!(
                binding.body,
                model::arch::models::shape::RouterBody::SigmoidFloor
            );
            assert_eq!(binding.rule, shape.rule);
            assert_eq!(
                (binding.experts(), binding.top_k_min, binding.top_k_max),
                (256, 6, 6)
            );
        }
    }
    let bytes: u64 = r.tensors.tensors.iter().map(|t| t.file_bytes).sum();
    assert_eq!(bytes, 25_049_218_176);
    let mut inputs = model::arch::flint::place::PlanInputs::read(&split)
        .expect("the validated header has genuine program and site bindings");
    inputs.room = (u64::MAX, model::placement::workstation::HostRead::Given);
    let machine = model::placement::workstation::plan_gate(s.layers.len());
    let levers = model::placement::PlanLevers::default();
    for ctx in [512, 1024, 4096] {
        for slots in [1, 3] {
            let plan = inputs
                .plan_with_slots(
                    &machine,
                    ctx,
                    &levers,
                    std::num::NonZeroUsize::new(slots).unwrap(),
                )
                .expect("the real mixed-quant header plans with resident slots");
            let cache = slots as u64 * 2048 * (14 * ctx + 35 * ctx.min(1024));
            assert_eq!(plan.cards.iter().map(|c| c.kv_bytes).sum::<u64>(), cache);
            assert!(inputs.prompt_batch_bytes(&machine).unwrap() > 0);
        }
    }
    let tier = model::arch::flint::place::tier_batch(&inputs).unwrap();
    assert!(tier.leg.unwrap().input_scales);
    assert!(tier.card().unwrap() > 0);
    println!(
        "flint header_sha={HEADER_SHA} tensors=779 layers=49 dense=2 routed=47 norms={norms} rotary={rotary} sinks={sinks} payload_bytes={bytes}"
    );
}
