//! What a Qwen3.8 run is held to against a reference forward of the same weights: the bound on a
//! layer output's distance, the router error an excused flip may carry, and the reference's routing
//! a flip is read against. The e2e gate holds the GGUF to ik's sets by them and the NVFP4 gate the
//! prepared split to the checkpoint's own forward, whose sets keep ik's names, so one change moves
//! both.

use bloomery_gpu::arch::qwen3moe::RouteTap;
use bloomery_gpu_gates::flip::{self, Flip};
use bloomery_gpu_gates::{GateError, RefManifest, topk_ids_logical_within};

use crate::e2e::tap;

/// PIN(2026-09-27): the free-running bound on a layer output's relative
/// distance from ik's, off every flip's path. Borrowed, not measured on
/// this model: GLM-5.3's gate's derivation (√45 · 1.415e-2, its forced
/// arm's worst layer error on the streams on ik's inputs) over this
/// model's 48 layers, √48 · 1.415e-2 ≈ 0.098, rounded up. GLM is the
/// analog and Qwen3.6 (0.26) is not: at every q8_0 projection our side
/// reads the f32 row, so the gap is ik's q8_2 activation alone, as in
/// GLM, where Qwen3.6's inputs are quantized on both sides; the host
/// experts run the q8_2_x4 rule on both. The e2e gate has no forced arm to
/// re-derive it on Qwen3.8; its (c) prints every layer's distance, the input
/// a forced arm would take.
pub const FREE_BAND: f64 = 0.10;

/// PIN(2026-09-28): [잠정 — 백로그] the most error, in router logit units, our
/// two logits may carry at an excused flip — a measured frontier, not a derivation.
/// In the batch set's free run the clean chain's largest pair error
/// was 0.942 (gap 0.483); the PLE site skipped (m06) reached 4.69, the other broken
/// chains 20.15 and 20.66; 2.0 sits 2.1x above clean and 2.3x below m06. The route tap
/// reading the next layer's logits (m14, 6.99) is left out: that error is the tap's,
/// not the router's. A flip-aware bound from a Qwen3.8 forced arm replaces it.
pub const FLIP_ERR_CAP: f64 = 2.0;

/// The reference's routing of one layer over a set's tokens: its router logits (`n_expert` a
/// token; the pick ranks their softmax, the same order) and its chosen ids (`n_used` a token).
pub struct IkRoute {
    logits: Vec<f32>,
    ids: Vec<i32>,
    n_expert: usize,
    n_used: usize,
}

impl IkRoute {
    /// Layer `l`'s routing in `man` (`ffn_moe_logits-<l>` and `ffn_moe_topk-<l>`, every id under
    /// `n_expert`), refused by name unless both hold the same whole tokens.
    pub fn read(
        man: &RefManifest,
        l: usize,
        n_expert: usize,
        n_used: usize,
    ) -> Result<IkRoute, GateError> {
        let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
        let row = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ids = topk_ids_logical_within(man, row, u32::try_from(n_expert)?)?;
        let whole = n_expert > 0
            && logits.len() % n_expert == 0
            && (logits.len() / n_expert).checked_mul(n_used) == Some(ids.len());
        if n_used == 0 || !whole {
            return Err(format!(
                "layer {l}: {} logits and {} ids; a token takes {n_expert} and {n_used}",
                logits.len(),
                ids.len()
            )
            .into());
        }
        Ok(IkRoute {
            logits,
            ids,
            n_expert,
            n_used,
        })
    }

    pub fn tokens(&self) -> usize {
        self.ids.len() / self.n_used
    }

    /// Token `t`'s logits and ids.
    pub fn at(&self, t: usize) -> (&[f32], &[i32]) {
        (
            &self.logits[t * self.n_expert..(t + 1) * self.n_expert],
            &self.ids[t * self.n_used..(t + 1) * self.n_used],
        )
    }

    /// The reference's own margin at token `t`: the lowest logit it kept less the highest it left.
    pub fn margin(&self, t: usize) -> f64 {
        let (v, ids) = self.at(t);
        flip::margin(v, ids)
    }
}

/// The flips of a run's routes (`[token][layer]`) against the reference's `ik_routes` (one per
/// layer): each (layer, token) whose chosen set is not the reference's. Refused by name when a
/// layer routes another count of tokens or a token has no tap.
pub fn route_flips(
    eager_routes: &[Vec<RouteTap>],
    ik_routes: &[IkRoute],
) -> Result<Vec<Flip>, GateError> {
    let mut flips = Vec::new();
    for (l, ik) in ik_routes.iter().enumerate() {
        if ik.tokens() != eager_routes.len() {
            return Err(format!(
                "layer {l}: ik routes {} tokens, the run {}",
                ik.tokens(),
                eager_routes.len()
            )
            .into());
        }
        for (t, routes) in eager_routes.iter().enumerate() {
            let ours = routes
                .get(l)
                .ok_or_else(|| format!("position {t}: no route tap for layer {l}"))?;
            let (iv, ids) = ik.at(t);
            flips.extend(Flip::between(
                (l, t),
                (&ours.ids, &ours.logits),
                (ids, iv),
                ik.margin(t),
            ));
        }
    }
    Ok(flips)
}
