//! The serve gates' side of the seats' context choice (`placement::ctx_ask`,
//! `generate::choose_ctx`): the contexts an unset `--ctx` can take,
//! recomputed from this module's own literals, and the one `ctx` line a seat
//! prints, read and held to its word. Every serve gate reads a seat's choice
//! through here, so a change of the rule is one edit on the gate side too.

/// The least context an unset `--ctx` searches, the base a slot
/// (`generate::SEAT_STEPS`' floor).
pub const FLOOR: u64 = 4096;

/// The rounding of every context an unset `--ctx` searches
/// (`generate::CTX_GRAN`).
pub const STEP: u64 = 1024;

/// The context an unset `--ctx` asks a slot first under the model's `limit`:
/// the limit down to a [`STEP`], never under [`FLOOR`] a limit past it holds.
#[must_use]
pub fn target_of(limit: u64) -> u64 {
    (limit / STEP * STEP).max(FLOOR.min(limit))
}

/// The least context an unset `--ctx` takes under `limit`: [`FLOOR`], or
/// the target where that is less.
#[must_use]
pub fn floor_of(limit: u64) -> u64 {
    FLOOR.min(target_of(limit))
}

/// Whether an unset `--ctx` can take `ctx` a slot under `limit`: the target
/// ([`target_of`]), or a multiple of [`STEP`] from the floor ([`floor_of`])
/// up to the target.
#[must_use]
pub fn fits(limit: u64, ctx: u64) -> bool {
    let target = target_of(limit);
    ctx == target || (ctx.is_multiple_of(STEP) && floor_of(limit) <= ctx && ctx < target)
}

/// The context one [`STEP`] above `ctx`, the one whose plan was short when
/// an unset `--ctx` fitted to `ctx` under `limit`; `None` when `ctx` is the
/// target or no context the default takes ([`fits`]).
#[must_use]
pub fn step_above(limit: u64, ctx: u64) -> Option<u64> {
    (fits(limit, ctx) && ctx < target_of(limit)).then_some(ctx + STEP)
}

/// The one `ctx` line of a seat's stderr (`ctx rule=… ctx=… slots=… total=…
/// limit=… (…)[ from=… <memory>-short need=… room=…]`,
/// `placement::ctx_ask::Chosen::line`); `None` when there is none or more
/// than one.
#[must_use]
pub fn ctx_line(log: &str) -> Option<&str> {
    let mut lines = log.lines().filter(|l| l.starts_with("ctx rule="));
    let one = lines.next()?;
    lines.next().is_none().then_some(one)
}

/// The whole-number field `name` of a line of `name=value` words.
#[must_use]
pub fn field(line: &str, name: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(name)?.strip_prefix('='))?
        .parse()
        .ok()
}

/// The memory a `fitted` line was short of (`card` or `host`).
#[must_use]
pub fn short_of(line: &str) -> Option<&str> {
    line.split_whitespace()
        .find_map(|w| w.strip_suffix("-short"))
}

/// Whether the `ctx` line of an unset `--ctx` on `slots` slots under the
/// model's `limit`, named `source`, holds its word: `target` asks the target
/// and took it; `fitted` asked it (`from`), found the machine short of the
/// context one step above its own (`need` past `room`, on the card or the
/// host) and took a context under the target ([`step_above`]); the slots are
/// `slots`, the total their sum, the limit `limit (source)`. `None` when the
/// line or one of its terms is missing.
#[must_use]
pub fn default_line_held(line: &str, slots: u64, (limit, source): (u64, &str)) -> Option<bool> {
    let rule = line
        .split_whitespace()
        .find_map(|w| w.strip_prefix("rule="))?;
    let num = |k: &str| field(line, k);
    let (ctx, n, total) = (num("ctx")?, num("slots")?, num("total")?);
    let by_word = match rule {
        "target" => ctx == target_of(limit) && !line.contains(" from="),
        "fitted" => {
            let (from, need, room) = (num("from")?, num("need")?, num("room")?);
            from == target_of(limit)
                && step_above(limit, ctx).is_some()
                && short_of(line).is_some_and(|m| m == "card" || m == "host")
                && need > room
        }
        _ => false,
    };
    Some(
        by_word
            && n == slots
            && total == slots * ctx
            && line.contains(&format!(" limit={limit} ({source})")),
    )
}

/// Whether `line` is the seat's `ctx` line once an image-encoder reserve of
/// `bytes` is weighed ([`bloomery_levers::RungCost`]): a reserve that lowers
/// the context (`with` under `without`) holds only when the line is the bare
/// choice plus that cost in parentheses; the same context holds only when the
/// line names no reserve.
#[must_use]
pub fn reserve_named(line: &str, with: u64, without: u64, bytes: u64) -> bool {
    const MARK: &str = " (the image encoder's";
    let bare = line.split_once(MARK).map(|(head, _)| head).unwrap_or(line);
    let image = bloomery_levers::Reserve::ImageEncoder;
    match bloomery_levers::RungCost::of(image, with, without, bytes) {
        Some(note) => line == format!("{bare} ({note})"),
        None => !line.contains(MARK),
    }
}

#[cfg(test)]
mod tests {
    use super::{default_line_held, fits, floor_of, step_above, target_of};

    /// The target is the limit down to a step; the contexts under it are
    /// the steps down to the floor; a limit under the floor is its own one
    /// context.
    ///
    /// PIN(2026-10-27): replaces `the_ladder_ends_on_the_floor`; an unset
    /// context is fitted to the largest step that holds, not halved.
    #[test]
    fn the_default_takes_the_steps_under_the_limit() {
        assert_eq!(target_of(1_048_576), 1_048_576);
        assert_eq!(target_of(262_144 + 1000), 262_144);
        assert_eq!((target_of(2048), floor_of(2048)), (2048, 2048));
        assert_eq!(floor_of(262_144), 4096);
        assert!(fits(262_144, 262_144) && fits(262_144, 69_632) && fits(262_144, 4096));
        assert!(!fits(262_144, 69_000) && !fits(262_144, 3072) && !fits(262_144, 263_168));
        assert_eq!(step_above(262_144, 69_632), Some(70_656));
        assert_eq!(step_above(262_144, 261_120), Some(262_144));
        assert_eq!(step_above(262_144, 262_144), None);
        assert_eq!(step_above(262_144, 5000), None);
        assert_eq!(step_above(2048, 2048), None);
    }

    /// A line holds its word, and a line off the steps or of other slots
    /// does not.
    #[test]
    fn a_line_holds_its_word() {
        let limit = (262_144, "trained");
        let target = "ctx rule=target ctx=262144 slots=1 total=262144 limit=262144 (trained)";
        assert_eq!(default_line_held(target, 1, limit), Some(true));
        assert_eq!(default_line_held(target, 2, limit), Some(false));
        let fitted = "ctx rule=fitted ctx=69632 slots=1 total=69632 limit=262144 (trained) \
                      from=262144 host-short need=9 room=8";
        assert_eq!(default_line_held(fitted, 1, limit), Some(true));
        let off = "ctx rule=fitted ctx=65000 slots=1 total=65000 limit=262144 (trained) \
                   from=262144 card-short need=9 room=8";
        assert_eq!(default_line_held(off, 1, limit), Some(false));
        let halved = "ctx rule=halved ctx=65536 slots=1 total=65536 limit=262144 (trained) \
                      from=262144 card-short need=9 room=8";
        assert_eq!(default_line_held(halved, 1, limit), Some(false));
        assert_eq!(default_line_held("ctx rule=target", 1, limit), None);
    }

    /// The serve clause's predicate ([`super::reserve_named`]): a context the
    /// reserve lowers holds only with the cost named, a bare line does not;
    /// the same context holds only bare.
    ///
    /// PIN(2026-10-27): replaces `a_dropped_rung_names_the_reserve`; the
    /// default has no rungs.
    #[test]
    fn a_lowered_context_names_the_reserve() {
        let bytes = 111;
        let bare = "ctx rule=fitted ctx=16384 slots=1 total=16384 limit=32768 (CTX_MAX) \
                    from=32768 card-short need=9 room=8";
        let image = bloomery_levers::Reserve::ImageEncoder;
        let note = bloomery_levers::RungCost::of(image, 16_384, 32_768, bytes)
            .expect("the reserve lowers the context");
        let full = format!("{bare} ({note})");
        assert!(super::reserve_named(&full, 16_384, 32_768, bytes));
        // The clause leaves the line's fields readable.
        assert_eq!(super::field(&full, "room"), Some(8), "{full}");
        assert!(
            !super::reserve_named(bare, 16_384, 32_768, bytes),
            "a dropped reason"
        );
        assert!(
            super::reserve_named(bare, 32_768, 32_768, bytes),
            "the same context, bare"
        );
        assert!(
            !super::reserve_named(&full, 32_768, 32_768, bytes),
            "a reason with no drop"
        );
    }
}
