//! What a plan holds beside its own tensors: a model the process serves beside the planned one (the
//! image encoder's tower) that sits on the stage card, as a named reserve of fixed bytes the plan's
//! fit counts ([`model::placement::Card::reserves`]). One function enters it into a placed plan's
//! machine and a whole-fit probe's ([`reserve_on`]), one reads it back ([`reserve_in`]), and a
//! plan's refusal ([`refusal`]) and a verdict's line ([`note`]) name it. Free of the encoder crates,
//! so a placement module every build includes can take it.

use std::fmt::Display;

use model::placement::{Card, Machine};

/// What the plan's reserve for the image encoder's tower is called, as the plan's terms name it
/// (beside `workstation`'s draft and tier-batch reserves).
pub const IMAGE_ENCODER: &str = "image encoder";

/// The bytes a plan reserves on its stage card beside its own tensors, by what they are for;
/// `None` for one that sits off the plan or is not served.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Beside {
    /// The image encoder's tower.
    pub image: Option<u64>,
}

impl Beside {
    /// Each reserve by its name, in the order the stage card takes them.
    fn named(self) -> impl Iterator<Item = (&'static str, u64)> {
        [(IMAGE_ENCODER, self.image)]
            .into_iter()
            .filter_map(|(name, bytes)| Some((name, bytes?)))
    }
}

/// `machine` with `beside`'s reserves on its stage card. The one place they enter a machine: a
/// placed plan's builder and a whole-fit probe both call it, so both fit the same bytes.
pub fn reserve_on(machine: &mut Machine, beside: Beside) {
    for (name, bytes) in beside.named() {
        machine.cards[0].reserves.push((name.to_owned(), bytes));
    }
}

/// The reserves [`reserve_on`] put on `machine`'s stage card: what a plan built on it refuses
/// beside ([`refusal`]).
#[must_use]
pub fn reserve_in(machine: &Machine) -> Beside {
    let held = |name: &str| {
        machine
            .cards
            .first()?
            .reserves
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, bytes)| bytes)
    };
    Beside {
        image: held(IMAGE_ENCODER),
    }
}

/// A machine builder (`layers` to its [`Machine`]) with [`reserve_on`] applied to what it builds.
pub fn with_reserve(plain: impl Fn(usize) -> Machine, beside: Beside) -> impl Fn(usize) -> Machine {
    move |layers| {
        let mut machine = plain(layers);
        reserve_on(&mut machine, beside);
        machine
    }
}

/// What a verdict's line adds when its card carries the image encoder's reserve: the reserve by
/// name and bytes, beside the reserves term the verdict sums them in. Nothing for a card that
/// carries none, so a load without `--mmproj` prints its line as it did.
#[must_use]
pub fn note(card: &Card) -> String {
    card.reserves
        .iter()
        .find(|(name, _)| name == IMAGE_ENCODER)
        .map_or_else(String::new, |(name, bytes)| {
            format!("; the reserves term holds the {name}'s {bytes} B")
        })
}

/// A plan's refusal, beside the reserves that were in it: with none the refusal as it is.
pub fn refusal(beside: Beside, e: impl Display) -> String {
    let held: Vec<String> = beside
        .named()
        .map(|(name, bytes)| format!("the {name}'s {bytes} B as the stage card's {name} reserve"))
        .collect();
    if held.is_empty() {
        return e.to_string();
    }
    format!("the plan holds {}: {e}", held.join(" and "))
}

#[cfg(test)]
mod tests {
    use model::placement::workstation::plan_a;

    use super::*;

    const BYTES: u64 = 1_476_000_000;

    /// The whole-fit probe and the placed plan take the one reserve: the machines both build carry
    /// the same named bytes, the card sets aside exactly that much more, and none changes nothing.
    #[test]
    fn the_probe_and_the_placed_plan_see_the_same_reserve() {
        let image = Beside { image: Some(BYTES) };
        let plain = plan_a(40);
        let mut probe = plan_a(40);
        reserve_on(&mut probe, image);
        let placed = with_reserve(plan_a, image)(40);
        assert_eq!(probe, placed);
        assert_eq!(
            probe.cards[0].reserves.last(),
            Some(&(IMAGE_ENCODER.to_owned(), BYTES))
        );
        assert_eq!(
            probe.cards[0].reserve_bytes(),
            plain.cards[0].reserve_bytes() + BYTES
        );
        assert_eq!(
            probe.cards[0].set_aside_bytes(),
            plain.cards[0].set_aside_bytes() + BYTES
        );
        assert_eq!(reserve_in(&probe), image);
        assert_eq!(reserve_in(&plain), Beside::default());
        let mut none = plan_a(40);
        reserve_on(&mut none, Beside::default());
        assert_eq!(none, plain);
        assert_eq!(with_reserve(plan_a, Beside::default())(40), plain);
        assert_eq!(note(&plain.cards[0]), "");
        assert_eq!(
            note(&probe.cards[0]),
            "; the reserves term holds the image encoder's 1476000000 B"
        );
        assert_eq!(refusal(Beside::default(), "no fit"), "no fit");
        assert_eq!(
            refusal(image, "no fit"),
            "the plan holds the image encoder's 1476000000 B as the stage card's image encoder \
             reserve: no fit"
        );
    }
}
