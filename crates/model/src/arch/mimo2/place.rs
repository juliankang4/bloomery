//! A mimo2 file planned onto a machine: the common hybrid owner's plan
//! ([`place_hybrid`]) over this family's reading, as `glm5next::place` is for
//! GLM. The hyperparameters, every tensor's role and the typed description are
//! read here ([`PlanInputs::read`], which refuses a file whose next-token
//! layers carry tensors — the draft that reads them is a later round's — and
//! one with a feature no program runs, every one listed); what each layer
//! holds, the expert rule and the refusals are the owner's. The rule's arms
//! are split: a Q6_K down beside a gate and up the leg runs is a card layer,
//! and the down reads the leg's `Q8Act` columns. The expert tier card a
//! machine names (plan (b′): the A6000 the stage, the 3090 its tier) holds the
//! next ids of each card layer on the same leg; its prompt-batch bytes are
//! [`tier_batch`], and the stage card's own bytes of a tiered load are the
//! tier reserve ([`tier_reserve_bytes`]).

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::place_hybrid;
use crate::placement::PlacementError;
use crate::placement::workstation::{self, TierBatchBytes};

// The ring rule's owner is `runtime::stores`; these names stay on this path for
// `gpu-mimo2` and the gates, which import them from here.
pub use place_hybrid::{KvLayout, PlaceError};
pub use runtime::stores::{Rows, cut_depth_keeps};

/// What a plan of a mimo2 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "MiMo";

/// The positions the body's prompt batch holds at most: the host union's columns.
pub const PROMPT_ROWS: usize = crate::moe::UNION_MAX_COLS;

/// The flash's segments of a full layer's row, the keys a window layer's segment spans, the rows
/// one flash launch of a full layer runs at most and the columns of a projection chunk, as the
/// batch's buffers are sized by them; the body holds each equal to the kernels' by assertion.
pub const FLASH_SEGMENTS: usize = 80;
/// See [`FLASH_SEGMENTS`].
pub const SEGMENT_KEYS: usize = 64;
/// See [`FLASH_SEGMENTS`].
pub const FLASH_ROWS: usize = 16;
/// See [`FLASH_SEGMENTS`].
pub const COL_GROUP: usize = 8;

/// The card bytes of the prompt batch's buffers every unit of a prompt group shares, for batches of
/// up to `cap` positions of a file described by `hp`, four bytes a value:
///
/// - the walk's shared rows: the normed input, the attention's output and the routed block's
///   normed rows (`3 · embd` a position), the router's scores (`n_expert`), ids and weights
///   (`2 · n_used`), the fused q·k·v rows of the widest layer (`n_head · head_k + kv_heads ·
///   (head_k + head_v)`), the roped query heads (`n_head · head_k`) and the flash's output rows
///   (`n_head · head_v`);
/// - the dense block's gate·up rows of one chunk of [`COL_GROUP`] columns;
/// - the flash's partials at the widest launch of any layer: the whole batch on a window layer
///   over `window / SEGMENT_KEYS` segments, [`FLASH_ROWS`] rows on a full layer over
///   [`FLASH_SEGMENTS`], `head_v + 2` values a (row, head, segment);
/// - the host sums the batch port uploads (`embd` a position);
/// - with `card`, the card experts' batch: the card leg's buffers in the 128-value down form
///   ([`workstation::tier_chunk_scratch_bytes`]), the card sum and the sum plus the base (`2 ·
///   embd` a position) and the slots' places (`n_used`).
///
/// The one formula of those bytes: the plan keeps them out of the expert budget
/// ([`prompt_reserve_bytes`]) and the body's batch counts its allocation by them.
#[must_use]
pub fn prompt_shared_bytes(hp: &Hparams, cap: usize, card: bool) -> u64 {
    let n = hp.n_embd;
    let layers = hp.kinds.iter().zip(&hp.kv_heads).take(hp.n_trunk);
    let qkv = layers
        .clone()
        .map(|(_, &kv)| hp.n_head * hp.head_k + kv * (hp.head_k + hp.head_v))
        .max()
        .unwrap_or(0);
    let partials = |per_seg: usize| {
        layers
            .clone()
            .map(|(&kind, _)| {
                let (rows, segments) = match kind {
                    Kind::Full => (FLASH_ROWS.min(cap), FLASH_SEGMENTS),
                    Kind::Swa => (cap, hp.window.div_ceil(SEGMENT_KEYS)),
                };
                rows * hp.n_head * segments * per_seg
            })
            .max()
            .unwrap_or(0)
    };
    let values = 3 * cap * n
        + cap * (hp.n_expert + 2 * hp.n_used)
        + cap * qkv
        + cap * hp.n_head * (hp.head_k + hp.head_v)
        + COL_GROUP * hp.dense_ff.unwrap_or(0).max(1)
        + partials(hp.head_v)
        + partials(2)
        + cap * n;
    let batch = if card {
        let leg = workstation::tier_chunk_scratch_bytes(
            n as u64,
            hp.expert_ff as u64,
            hp.n_used as u64,
            cap as u64,
        ) as usize;
        leg + 2 * size_of::<f32>() * cap * n + size_of::<u32>() * cap * hp.n_used
    } else {
        0
    };
    (size_of::<f32>() * values + batch) as u64
}

/// Tokens one run of the tier's card leg covers (the tier's `CHUNK`, which holds itself equal to
/// this): a block of a prompt unit's rows runs in runs of at most this many.
pub const TIER_CHUNK: u64 = 16;

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for a file described by `hp` and
/// blocks of `cols` rows (the body's prompt unit): the tier card's staging for a block, one run of
/// the card leg's buffers over [`TIER_CHUNK`] of its rows in the 128-value form every down the
/// plan's rule carries reads (the rule's down arm is the leg's `Q8Act` downs), and the host's rows
/// and places ([`workstation::tier_leg_batch_bytes`]). The plan reserves them on the tier card and
/// the host, and the load checks its allocations against them.
pub fn tier_batch(hp: &Hparams, cols: u64) -> Result<TierBatchBytes, PlacementError> {
    workstation::tier_leg_batch_bytes(
        hp.n_embd as u64,
        hp.expert_ff as u64,
        hp.n_used as u64,
        TIER_CHUNK,
        cols,
    )
}

/// The stage card's bytes only a tiered load makes, beside [`prompt_reserve_bytes`], for a file
/// described by `hp` whose routed run is `routed` layers: the step's tier side (the slot map's
/// tier view, `n_expert` places a routed layer, and the step's `n_used` tier places) and the
/// prompt batch's tier places with the copies of the weights and places its join reads after the
/// next unit has routed (three words a slot, at [`PROMPT_ROWS`] positions). The one formula of
/// those bytes: the plan keeps them out of the expert budget on a machine that names a tier card
/// ([`PlanInputs::tier_reserve`]) and the body holds what it makes to them.
#[must_use]
pub fn tier_reserve_bytes(hp: &Hparams, routed: usize) -> u64 {
    let side = routed * hp.n_expert + hp.n_used;
    let batch = 3 * PROMPT_ROWS * hp.n_used;
    (size_of::<u32>() * (side + batch)) as u64
}

/// The card bytes of one unit of a prompt group's own buffers for batches of up to `cap` positions:
/// the layer's input rows and the residual after the attention (`2 · embd` values a position), the
/// positions and key counts (two words a position).
#[must_use]
pub fn prompt_unit_bytes(hp: &Hparams, cap: usize) -> u64 {
    (size_of::<f32>() * 2 * cap * hp.n_embd + size_of::<u32>() * 2 * cap) as u64
}

/// The card bytes the plan keeps out of the expert budget for a load's prompt batch: the shared
/// buffers with the card experts' batch, and the first unit, at [`PROMPT_ROWS`] positions, whatever
/// the context (the batch is made for `min(PROMPT_ROWS, ctx)`). The units a prompt group past one
/// adds come out of the card's margin.
#[must_use]
pub fn prompt_reserve_bytes(hp: &Hparams) -> u64 {
    prompt_shared_bytes(hp, PROMPT_ROWS, true) + prompt_unit_bytes(hp, PROMPT_ROWS)
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file whose
    /// next-token layers carry tensors is refused by the next-token count's
    /// name (the draft that reads them is a later round's; this plan loads
    /// the trunk without them), and one with a feature the engine does not
    /// run ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        if inputs.hp.n_layer > inputs.hp.n_trunk {
            return Err(PlacementError::Metadata {
                key: "nextn_predict_layers".to_string(),
                detail: format!(
                    "is {}; the draft that reads the next-token layers is a later round's, and \
                     this plan loads the trunk without them",
                    inputs.hp.n_layer - inputs.hp.n_trunk
                ),
            });
        }
        inputs.runnable()
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a
    /// gate that plans or describes a file whatever the coverage list holds,
    /// and the host's room and the arena's given bytes, a room that cannot be
    /// read refused by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::THINK))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let reserve = prompt_reserve_bytes(&hp);
        let routed = runtime::layer::hosted(&spec.layers).map_or(0, |run| run.len());
        let tier = tier_reserve_bytes(&hp, routed);
        Ok(PlanInputs::new(ENGINE, hp, model, spec)?
            .with_prompt_reserve(reserve)
            .with_tier_reserve(tier))
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::hparams::tests::{keys, tensors};
    use super::super::hparams::{Hparams, Kind};
    use super::{
        PROMPT_ROWS, PlaceError, PlanInputs, TIER_CHUNK, prompt_reserve_bytes, prompt_shared_bytes,
        prompt_unit_bytes, tier_batch, tier_reserve_bytes,
    };
    use crate::arch::synthetic::{V, header_shaped, host_run_experts};
    use crate::placement::{KvBytes, PlacementError, PlanLevers, workstation};

    /// A file whose next-token layers carry tensors is refused at
    /// [`PlanInputs::read`] by the next-token count's name — the draft that
    /// reads them is a later round's, and this plan loads the trunk without
    /// them — before the coverage list; the same file without its next-token
    /// layer is refused by that list alone (the synthetic head width is one no
    /// row of the mimo2 body runs), so the count is the named refusal.
    #[test]
    fn next_token_layers_are_refused_at_read() {
        let err = read_inputs("mimo2-place-nextn", &keys(), &tensors())
            .expect_err("a file with a next-token layer is refused");
        assert!(
            err.contains("metadata nextn_predict_layers: is 1")
                && err.contains("the draft that reads the next-token layers is a later round's"),
            "{err}"
        );
        let tensors: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| !n.starts_with("blk.4."))
            .collect();
        let err = read_inputs("mimo2-place-trunk", &trunk_keys(), &tensors)
            .expect_err("a trunk-only file is refused by the coverage list");
        assert!(
            err.contains("feature(s) of this file are not implemented"),
            "{err}"
        );
    }

    /// A machine that names an expert tier card (plan (b′)) is planned: the body serves one. The
    /// stage card keeps the tier reserve beside the prompt batch's on such a machine and nothing
    /// more on one without, and the tier's batch bytes are the common leg's: runs of [`TIER_CHUNK`]
    /// tokens over blocks of the unit's `cols` rows, in the 128-value form. The plan reaches the
    /// placement, which refuses the tier idle: the header's experts all fit the stage card. A
    /// context of no position is refused first. Mutant: the tier reserve left out of the
    /// description (the plan stops at [`PlaceError::Tier`]), or the run's tokens and the block's
    /// rows swapped.
    #[test]
    fn a_tier_card_is_planned_with_the_leg_batch() {
        const COLS: u64 = 3 * TIER_CHUNK;
        let path = header_shaped(
            "mimo2-place-tier",
            "mimo2",
            &keys(),
            &[
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        let hp = &inputs.hp;
        let routed = runtime::layer::hosted(&inputs.spec.layers).map_or(0, |run| run.len());
        let tiered = workstation::plan_bp(
            hp.n_trunk,
            None,
            tier_batch(hp, COLS).expect("valid tier geometry"),
        );
        let plain = workstation::plan_gate(hp.n_trunk);
        assert_eq!(
            inputs.prompt_batch_bytes(&plain).expect("a plain machine"),
            inputs.prompt_reserve
        );
        assert_eq!(
            inputs
                .prompt_batch_bytes(&tiered)
                .expect("a tiered machine"),
            inputs.prompt_reserve + tier_reserve_bytes(hp, routed)
        );
        let levers = PlanLevers::default();
        assert!(matches!(
            inputs.plan(&tiered, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
        match inputs.plan(&tiered, 4, &levers) {
            Err(PlaceError::Placement(PlacementError::IdleTier { tier: 0, .. })) => {}
            other => panic!("{other:?}, not the placement's idle-tier refusal"),
        }
    }

    /// The stage card's tier reserve at the real widths, 97,312 B [derived]: the step's tier side,
    /// 47 routed layers of 256 places and the step's 8 (12,040 words), and the prompt batch's tier
    /// places with the join's copies of the weights and places, 3 x 512 positions x 8 slots
    /// (12,288 words), four bytes a word. Mutant: the batch's tier words left out (48,160 B), or the
    /// side counted over the trunk's 48 layers (98,336 B).
    #[test]
    fn the_tier_reserve_is_the_side_and_the_batch_places() {
        let hp = real_widths();
        assert_eq!(tier_reserve_bytes(&hp, 47), 97_312);
    }

    /// The small header of the `hparams` tests' keys with its next-token
    /// layer taken out: four trunk blocks, the per-layer arrays four long.
    fn trunk_keys() -> Vec<(&'static str, V)> {
        keys()
            .into_iter()
            .map(|(k, v)| match k {
                "block_count" => ("block_count", V::U32(4)),
                "nextn_predict_layers" => ("nextn_predict_layers", V::U32(0)),
                "attention.head_count_kv" => ("attention.head_count_kv", V::I32s(vec![2, 1, 1, 2])),
                "attention.sliding_window_pattern" => (
                    "attention.sliding_window_pattern",
                    V::I32s(vec![0, 1, 1, 0]),
                ),
                _ => (k, v),
            })
            .collect()
    }

    /// The stage's rope rows: one table a distinct rope base the trunk turns
    /// by, `rope.dimension_count` f32 values a position, as the body copies
    /// them to its card. The small header's full and window bases differ (two
    /// tables of 4 values: 32 B a position); one base for both is one table.
    /// The plan counts them on the card that runs layer 0, at the plan's
    /// positions. FAIL-first: a layout that counts no row leaves the card's
    /// `stage_rows_bytes` at 0 and the load's scratch check refuses a context
    /// whose rope tables pass the fixed scratch term.
    #[test]
    fn the_stage_rope_rows_are_a_table_a_base() {
        let describe = |tag: &str, kv: &[(&'static str, V)]| {
            let global = [
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ];
            let path = header_shaped(tag, "mimo2", kv, &global, &tensors());
            let split = Split::open(&path).expect("the synthetic header opens");
            let mut inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
            let _ = std::fs::remove_file(&path);
            host_run_experts(&mut inputs.model);
            inputs
        };
        let two = describe("mimo2-rope-two", &keys());
        assert_eq!(two.kv.stage_position_bytes(10), 2 * 4 * 4 * 10);
        let one_base: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "rope.freq_base_swa" => (k, V::F32(1e7)),
                _ => (k, v),
            })
            .collect();
        let one = describe("mimo2-rope-one", &one_base);
        assert_eq!(one.kv.stage_position_bytes(10), 4 * 4 * 10);
        let machine = workstation::plan_gate(two.hp.n_trunk);
        let plan = two
            .plan(&machine, 1024, &PlanLevers::default())
            .expect("the small header plans at 1,024");
        assert_eq!(plan.cards[0].stage_rows_bytes, 32 * 1024);
        assert_eq!(
            two.seq_terms().slots_of(3).stage_position_bytes(1024),
            32 * 1024
        );
    }

    /// The small header's hyperparameters with the real file's widths in every field the prompt
    /// batch's formula reads (MiMo-V2.6-Flash: 48 trunk layers, the nine full ones at 0, 5, 11, 17,
    /// 23, 29, 35, 41 and 47 with 4 key heads and the rest window layers of 128 positions with 8;
    /// 4,096 wide, 64 query heads of key 192 and value 128; 256 experts, 8 used, 2,048 wide; one
    /// dense block 16,384 wide).
    fn real_widths() -> Hparams {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        // One file a call: the tests that read these widths run at once, and each removes its own.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("mimo2-place-batch-{n}");
        let path = header_shaped(&name, "mimo2", &keys(), &global, &tensors());
        let split = Split::open(&path).expect("the synthetic header opens");
        let hp = PlanInputs::describe(&split)
            .expect("the synthetic header describes")
            .hp;
        let _ = std::fs::remove_file(&path);
        let full = [0, 5, 11, 17, 23, 29, 35, 41, 47];
        let kinds: Vec<Kind> = (0..48)
            .map(|l| {
                if full.contains(&l) {
                    Kind::Full
                } else {
                    Kind::Swa
                }
            })
            .collect();
        let kv_heads = kinds
            .iter()
            .map(|k| if *k == Kind::Full { 4 } else { 8 })
            .collect();
        Hparams {
            n_trunk: 48,
            n_embd: 4096,
            n_head: 64,
            kv_heads,
            head_k: 192,
            head_v: 128,
            window: 128,
            kinds,
            n_expert: 256,
            n_used: 8,
            expert_ff: 2048,
            dense_ff: Some(16_384),
            ..hp
        }
    }

    /// The prompt batch's reserve at the union's 512 positions, 316,919,808 B [derived] from the
    /// file's widths, four bytes a value, term by term: the walk's shared rows 141,197,312 B (the
    /// normed input, output and routed rows 3 x 4,096, the scores and ids and weights 256 + 2 x 8,
    /// the fused q·k·v of the widest layer 64 x 192 + 8 x 320 = 14,848, the roped query heads 64 x
    /// 192, the flash output 64 x 128, all a position; the dense chunk 8 x 16,384; the partials'
    /// widest launch, a full layer's 16 rows x 64 heads x 80 segments x 128 = 10,485,760 values
    /// against a window layer's 512 x 64 x 2 x 128 = 8,388,608, and the (max, sum) pairs'
    /// 163,840) + the host sums 8,388,608 B (512 x 4,096) + the card experts' batch 150,552,576 B
    /// (the leg 133,758,976 at 512 columns of 8 slots through 4,096 x 2,048, the card sum and the
    /// sum plus the base 2 x 4 x 512 x 4,096, the slots' places 4 x 512 x 8) + one unit 16,781,312
    /// B (2 x 512 x 4,096 values and 2 x 512 words). Mutant: a full layer's launch counted at the
    /// batch's rows, or a term left out.
    #[test]
    fn the_prompt_batch_reserve_is_the_terms_at_the_unions_columns() {
        let hp = real_widths();
        assert_eq!(PROMPT_ROWS, 512);
        assert_eq!(
            prompt_shared_bytes(&hp, PROMPT_ROWS, false),
            141_197_312 + 8_388_608
        );
        assert_eq!(prompt_shared_bytes(&hp, PROMPT_ROWS, true), 300_138_496);
        assert_eq!(prompt_unit_bytes(&hp, PROMPT_ROWS), 16_781_312);
        assert_eq!(prompt_reserve_bytes(&hp), 316_919_808);
    }

    /// The plan keeps the prompt batch out of the expert budget: the inputs `describe` makes carry
    /// the formula's reserve, which is what the load's body holds its batch to. Mutant: `describe`
    /// leaves the reserve out (it reads 0).
    #[test]
    fn describe_reserves_the_prompt_batch() {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped("mimo2-place-reserve", "mimo2", &keys(), &global, &tensors());
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        assert!(inputs.prompt_reserve > 0, "{}", inputs.prompt_reserve);
        assert_eq!(inputs.prompt_reserve, prompt_reserve_bytes(&inputs.hp));
    }

    /// [`PlanInputs::read`] of a synthetic header, or its error's text.
    fn read_inputs(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<PlanInputs, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "mimo2", kv, &global, tensors);
        let split = Split::open(&path).expect("the synthetic header opens");
        let read = PlanInputs::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }
}
