//! The pieces the live view's tabs are drawn with: a box around lines, rows
//! of tiles that fill the width, a meter, a line with a part at each end,
//! and two columns side by side. They take and give painted lines and count
//! a line's columns without its escapes.

use crate::screen::{self, BOLD, DIM, RESET, Rgb, Style, TRACK, Text};

/// The columns between two columns of a tab.
pub(crate) const GAP: usize = 3;
/// The narrowest a tile is drawn when several share a row, and the columns
/// between two tiles.
const TILE: usize = 24;
const TILE_GAP: usize = 1;

/// The columns the painted line `painted` takes on a terminal.
pub(crate) fn shown(painted: &str) -> usize {
    screen::width(&screen::strip_escapes(painted))
}

/// `text` in `w` columns: whole when it fits, else cut and ended with `…`
/// (`...` in ASCII).
pub(crate) fn clip(text: &str, w: usize, style: Style) -> String {
    if screen::width(text) <= w {
        return text.to_owned();
    }
    let more = if style.utf8 { "…" } else { "..." };
    let room = w.saturating_sub(screen::width(more));
    let mut used = 0;
    let mut out: String = text
        .chars()
        .take_while(|&c| {
            used += screen::char_width(c);
            used <= room
        })
        .collect();
    if w >= screen::width(more) {
        out.push_str(more);
    }
    out
}

/// `text` painted with `code`.
pub(crate) fn paint(text: &str, code: &str) -> Text {
    let mut t = Text::default();
    t.push(text, code);
    t
}

/// A heading's escape: bold in the accent.
pub(crate) fn accent(style: Style) -> String {
    format!("{}{}", style.code(BOLD), style.fg(screen::ORANGE))
}

/// `lines` in a box `w` columns wide with a column of room inside each
/// side, a line longer than the room broken where the room runs out; ASCII
/// edges outside UTF-8.
pub(crate) fn boxed(lines: &[String], w: usize, style: Style) -> Vec<String> {
    let [tl, tr, bl, br, h, v] = if style.utf8 {
        ["┌", "┐", "└", "┘", "─", "│"]
    } else {
        ["+", "+", "+", "+", "-", "|"]
    };
    let edge = |s: &str| paint(s, style.code(DIM)).painted;
    let rule = h.repeat(w.saturating_sub(2));
    let inside = w.saturating_sub(4);
    let mut out = vec![edge(&format!("{tl}{rule}{tr}"))];
    for line in lines {
        for l in screen::wrap_painted(line, inside.max(1), false).split('\n') {
            let pad = " ".repeat(inside.saturating_sub(shown(l)));
            out.push(format!(
                "{} {l}{}{pad} {}",
                edge(v),
                style.code(RESET),
                edge(v)
            ));
        }
    }
    out.push(edge(&format!("{bl}{rule}{br}")));
    out
}

/// A bar of `n` cells, the share `part` of `whole` of them lit in `c` and
/// the rest dark; `thick` fills the cell, else a line through it. Without
/// colour the lit and the dark cells take different glyphs.
pub(crate) fn meter(part: u64, whole: u64, n: usize, c: Rgb, thick: bool, style: Style) -> Text {
    let lit = if whole == 0 {
        0
    } else {
        let n128 = n as u128;
        let lit = (u128::from(part.min(whole)) * n128 + u128::from(whole) / 2) / u128::from(whole);
        usize::try_from(lit).unwrap_or(n)
    };
    let (on, off) = match (style.utf8, thick) {
        (true, true) => ("█", if style.color { "█" } else { "░" }),
        (true, false) => ("━", if style.color { "━" } else { "─" }),
        (false, true) => ("#", if style.color { "#" } else { "." }),
        (false, false) => ("=", "-"),
    };
    let mut t = Text::default();
    t.push(&on.repeat(lit), &style.fg(c));
    t.push(&off.repeat(n - lit), &style.fg(TRACK));
    t
}

/// What a tile shows: its title, its value, a line under the value, and
/// the share its meter lights (the part, the whole, the colour).
pub(crate) struct Tile {
    pub(crate) title: &'static str,
    pub(crate) value: String,
    pub(crate) note: String,
    pub(crate) meter: Option<(u64, u64, Rgb)>,
}

/// `all` in boxes over `w` columns, in order: four to a row when four fit
/// [`TILE`] columns each, else two, else one; each row's boxes share its
/// full width, so a short last row ends in no empty box.
pub(crate) fn tiles(all: &[Tile], w: usize, style: Style) -> Vec<String> {
    if all.is_empty() {
        return Vec::new();
    }
    let per = [4, 2]
        .into_iter()
        .find(|&k| k * TILE + (k - 1) * TILE_GAP <= w)
        .unwrap_or(1);
    let mut out = Vec::new();
    for row in all.chunks(per) {
        let room = w.saturating_sub((row.len() - 1) * TILE_GAP);
        let metered = row.iter().any(|t| t.meter.is_some());
        let boxes: Vec<Vec<String>> = row
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let bw = room / row.len() + usize::from(i < room % row.len());
                tile(t, bw, metered, style)
            })
            .collect();
        for i in 0..boxes[0].len() {
            let line: Vec<&str> = boxes.iter().map(|b| b[i].as_str()).collect();
            out.push(line.join(&" ".repeat(TILE_GAP)));
        }
    }
    out
}

/// One tile in a box `w` columns wide, its meter's line kept (blank without
/// one) when `metered`.
fn tile(t: &Tile, w: usize, metered: bool, style: Style) -> Vec<String> {
    let inside = w.saturating_sub(4);
    let mut lines = vec![
        paint(&clip(t.title, inside, style), style.code(DIM)).painted,
        paint(&clip(&t.value, inside, style), style.code(BOLD)).painted,
        paint(&clip(&t.note, inside, style), style.code(DIM)).painted,
    ];
    if metered {
        lines.push(t.meter.map_or_else(String::new, |(part, whole, c)| {
            meter(part, whole, inside, c, false, style).painted
        }));
    }
    boxed(&lines, w, style)
}

/// `left` with `right` at the end of a line `w` columns wide; `right` on a
/// line of its own when the two do not fit one.
pub(crate) fn spread(left: &Text, right: &Text, w: usize) -> Vec<String> {
    let (l, r) = (screen::width(&left.shown), screen::width(&right.shown));
    if r == 0 {
        return vec![left.painted.clone()];
    }
    if l + GAP + r <= w {
        let mut line = left.clone();
        line.pad(w - r);
        line.append(right);
        return vec![line.painted];
    }
    vec![left.painted.clone(), right.painted.clone()]
}

/// The lines of `left`, each at most `lw` columns, with those of `right`
/// beside them [`GAP`] columns on.
pub(crate) fn beside(left: &[String], lw: usize, right: &[String]) -> Vec<String> {
    (0..left.len().max(right.len()))
        .map(|i| {
            let l = left.get(i).map_or("", String::as_str);
            match right.get(i) {
                Some(r) => format!("{l}{}{r}", " ".repeat(lw + GAP - shown(l).min(lw))),
                None => l.to_owned(),
            }
        })
        .collect()
}
