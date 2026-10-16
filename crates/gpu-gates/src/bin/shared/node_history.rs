//! Token IDs and every logits bit from a node gate's stepped history.

use std::fmt::Display;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::Path;

use bloomery_gpu_gates::GateError;
use runtime::{Out, Target, Want};

/// Context capacity for the shared stepped history.
pub(crate) const CTX: u64 = 64;

/// Read the history runner's card budget and output path.
pub(crate) fn args() -> Result<(u64, std::path::PathBuf), GateError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [gib, out] = args.as_slice() else {
        return Err("usage: <card-budget-GiB> <history-file>".into());
    };
    let budget = gib
        .parse::<u64>()?
        .checked_mul(1 << 30)
        .ok_or("budget overflow")?;
    Ok((budget, std::path::PathBuf::from(out.as_str())))
}

/// Build the stage and one expert tier from their censuses.
pub fn machine(
    layers: usize,
    batch: model::placement::workstation::TierBatchBytes,
) -> Result<(model::placement::Machine, Option<SocketAddr>), GateError> {
    use bloomery_gpu::host::node::node_cards;
    use model::placement::{devices::spec_of_device, workstation};

    let own = bloomery_gpu::census()?;
    let stage = spec_of_device(own.first().ok_or("no stage card")?);
    let nodes = bloomery_levers::tier_nodes()?;
    let tier = match nodes.as_slice() {
        [] => stage,
        [_] => node_cards(&nodes)?.first().ok_or("no tier census")?.1,
        _ => return Err("the history requires exactly one tier node".into()),
    };
    let mut machine = workstation::plan_on(stage, layers);
    machine
        .tiers
        .push(workstation::tier_card(tier, None, batch));
    machine
        .host_mut()
        .reserves
        .push(("tier batch".into(), batch.host));
    Ok((machine, nodes.first().copied()))
}

/// Step `model`, loaded by `plan`, in graph mode through the shared history
/// and write it to `path`, once its first tier is the requested host tier on
/// `node` with no resident bytes; the history must page host experts.
pub fn write<B>(
    mut model: bloomery_gpu::GpuModel<B>,
    plan: &model::placement::Plan<'_>,
    path: &Path,
    node: Option<SocketAddr>,
) -> Result<(), GateError>
where
    B: bloomery_gpu::model::ChainBody + bloomery_gpu::host::served::TierBody,
    app::Session<B>: Target,
    <app::Session<B> as Target>::Error: Display,
{
    const WHAT: &str = "node history";
    println!(
        "plan: {} experts on stage, {} on tier",
        plan.cards[0].experts, plan.cards[1].experts
    );
    model.set_mode(bloomery_gpu::model::StepMode::Graph);
    let mut s = app::Session::from_model(model, CTX as u32);
    let t = s
        .model()
        .body(WHAT)?
        .hybrid()
        .tiers()
        .first()
        .ok_or("no loaded tier")?;
    if t.node() != node || t.resident_bytes() != 0 {
        return Err("the history did not open the requested host tier".into());
    }
    println!(
        "loaded: tier {:?}, resident bytes {}",
        t.node(),
        t.resident_bytes()
    );
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    let mut ids = Vec::with_capacity(19);
    for at in 0..19 {
        let input = if at < 3 { at + 1 } else { ids[ids.len() - 1] };
        let out = s
            .step(input, Want::Logits)
            .map_err(|e| format!("history step {at}: {e}"))?;
        let id = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the step did not return logits".into());
        };
        file.write_all(&id.to_le_bytes())?;
        file.write_all(&(row.len() as u64).to_le_bytes())?;
        for value in row {
            file.write_all(&value.to_bits().to_le_bytes())?;
        }
        ids.push(id);
    }
    file.flush()?;
    let t = s
        .model_mut()
        .body_parts(WHAT)?
        .2
        .hybrid_mut()
        .tiers_mut()
        .first_mut()
        .ok_or("no loaded tier")?;
    let (fills, bytes) = t.host_fills()?;
    println!(
        "history: {} ids {ids:?}; fills={fills} bytes={bytes}",
        ids.len()
    );
    if fills == 0 || bytes == 0 {
        return Err("the history did not exercise host paging".into());
    }
    Ok(())
}
