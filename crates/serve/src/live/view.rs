//! The live view's frames: a header (the wordmark beside a small furnace,
//! the model and the time it has served, the status, the seat, its address
//! and its load time), a row of tabs, the open tab's lines, and the keys. A
//! frame is built from a copy of the server's state ([`Now`]) and what the
//! start fixed ([`Fixed`]); it reads nothing else.
//!
//! Only what the server has numbers for is drawn: a tab, a tile, a bar or a
//! column whose source is missing is left off, and the tiles left fill their
//! rows. Placement is there only for a start that noted its records (the
//! start screen's facts), Records only for one that noted any. Numbers are
//! the server's own: the slots and the queue from the board, the rates of
//! the slots decoding now from the frames before, the throughput from
//! `/metrics`' totals (an average since start, which a request adds to when
//! it ends), the context against every slot's room, the requests that ended
//! from their timings, where the model went from the start's records.

use std::net::SocketAddr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::screen::{
    self, BOLD, DIM, Facts, HOST, INK, LABEL, MARGIN, Marks, ORANGE, RED, SLATE, Style, Text,
    UNDERLINE, YELLOW,
};
use crate::worker::rate;

use super::parts::{self, GAP, Tile, accent, paint};
use super::ring::{Ending, Finished};

/// What a slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Idle,
    /// A request runs its prompt: it has generated nothing yet.
    Prefill,
    /// A request generates.
    Decode,
    /// An action (a slot save or restore, a residency reset) holds it.
    Held,
}

/// A slot as one frame copied it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    pub(crate) phase: Phase,
    /// The positions it holds.
    pub(crate) n_past: usize,
    /// The request it runs, and the tokens that request generated so far.
    pub(crate) task: u64,
    pub(crate) n_decoded: usize,
}

/// The server's state: serving, stopping, or ended by a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Ready,
    Stopping,
    Failed,
}

/// A copy of the server's state, taken for one frame.
pub(crate) struct Now {
    pub(crate) status: Status,
    pub(crate) slots: Vec<Slot>,
    /// Each decoding slot's rate over the last frames ([`rates`]), tokens a
    /// second; empty on the first frame.
    pub(crate) rates: Vec<Option<f64>>,
    /// Requests waiting for a slot.
    pub(crate) queued: usize,
    /// The positions the slots can hold in all.
    pub(crate) room: usize,
    /// The prompt and generation throughput since start, tokens a second.
    pub(crate) prompt_rate: f64,
    pub(crate) decode_rate: f64,
    /// Prompt tokens the cache kept instead of evaluating, and prompt tokens
    /// evaluated, since start.
    pub(crate) cached: u64,
    pub(crate) evaluated: u64,
    /// The requests that ended, newest first.
    pub(crate) requests: Vec<Finished>,
}

/// What the start fixed: the address, the model, whether requests need an
/// API key, when the view started, the record lines the start noted and the
/// screen's facts of them, the routes the Connect tab lists.
pub(crate) struct Fixed {
    pub(crate) addr: SocketAddr,
    pub(crate) model: String,
    pub(crate) keyed: bool,
    pub(crate) since: Instant,
    pub(crate) records: Vec<String>,
    pub(crate) facts: Option<Facts>,
    pub(crate) routes: Vec<(&'static str, &'static str)>,
}

/// A tab of the view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tab {
    Overview,
    Placement,
    Requests,
    Records,
    Connect,
}

impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Placement => "Placement",
            Tab::Requests => "Requests",
            Tab::Records => "Records",
            Tab::Connect => "Connect",
        }
    }
}

impl Fixed {
    /// The tabs that have something to show, in order.
    pub(crate) fn tabs(&self) -> Vec<Tab> {
        let placed = self
            .facts
            .as_ref()
            .is_some_and(|f| f.split.is_some() || f.rows.iter().any(|(l, _)| *l != "loaded"));
        [
            (Tab::Overview, true),
            (Tab::Placement, placed),
            (Tab::Requests, true),
            (Tab::Records, !self.records.is_empty()),
            (Tab::Connect, true),
        ]
        .into_iter()
        .filter_map(|(t, there)| there.then_some(t))
        .collect()
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// The value of the start's fact row `label`, as shown.
    fn row(&self, label: &str, style: Style) -> Option<String> {
        let f = self.facts.as_ref()?;
        let (_, v) = f.rows.iter().find(|(l, _)| *l == label)?;
        Some(screen::plain(v, style))
    }
}

/// The widest the view draws, in columns.
const WIDEST: usize = 140;
/// The rows from which the header draws the furnace beside the wordmark.
const TALL: usize = 30;
/// The Overview's column of bars beside its requests.
const BARS: usize = 40;
/// The Connect tab's column of endpoints beside its commands.
const ROUTES: usize = 44;
/// The narrowest column of the Placement tab's sections when two share a
/// row.
const SECTION: usize = 36;
/// The rows the Overview tab lists of the requests that ended.
const RECENT: usize = 5;

/// The frame of `tab` (one of [`Fixed::tabs`], at `index`) at `columns` ×
/// `rows`, its lines from `scroll` on, and that start as it was clamped to
/// the lines there are.
pub(crate) fn frame(
    now: &Now,
    fixed: &Fixed,
    index: usize,
    scroll: usize,
    style: Style,
    columns: usize,
    rows: usize,
) -> (String, usize) {
    let tabs = fixed.tabs();
    let tab = tabs[index.min(tabs.len() - 1)];
    let margin = if columns > LABEL + MARGIN.len() {
        MARGIN
    } else {
        ""
    };
    let inner = columns.min(WIDEST) - margin.len();
    let fit = |lines: Vec<String>| -> Vec<String> {
        lines
            .into_iter()
            .flat_map(|l| {
                screen::wrap_painted(&format!("{margin}{l}"), columns, false)
                    .split('\n')
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let mut top = header(now, fixed, style, inner, rows >= TALL && inner >= 40);
    top.extend(tab_row(&tabs, tab, style, inner));
    top.push(String::new());
    let top = fit(top);

    let body = fit(match tab {
        Tab::Overview => overview(now, fixed, style, inner),
        Tab::Placement => placement(fixed, style, inner),
        Tab::Requests => requests(&now.requests, style, inner),
        Tab::Records => records(fixed, style, inner),
        Tab::Connect => connect(fixed, style, inner),
    });

    let key = |name: &str, what: &str| {
        let mut t = paint(name, "");
        t.push(&format!(" {what}"), style.code(DIM));
        t
    };
    let keys = [
        key(&format!("1-{}", tabs.len()), "views"),
        key(if style.utf8 { "↑↓" } else { "j k" }, "scroll"),
        key("q", "close view"),
        key("Ctrl+C", "stop server"),
    ];
    let mut bottom = vec![String::new()];
    bottom.extend(
        screen::key_rows(&keys, inner)
            .into_iter()
            .map(|t| t.painted),
    );
    let bottom = fit(bottom);

    // A short terminal keeps the tabs and the keys and as much of the body
    // as fits; the header goes first.
    let mut top = top;
    while top.len() + bottom.len() + 1 > rows && top.len() > 2 {
        top.remove(0);
    }
    let room = rows.saturating_sub(top.len() + bottom.len());
    let scroll = scroll.min(body.len().saturating_sub(room.max(1)));
    let mut shown: Vec<String> = body.iter().skip(scroll).take(room).cloned().collect();
    let below = body.len().saturating_sub(scroll + room);
    if below > 0
        && let Some(last) = shown.last_mut()
    {
        let more = paint(
            &format!("{margin}{} more lines below", below + 1),
            style.code(DIM),
        );
        let wrapped = screen::wrap_painted(&more.painted, columns, false);
        if let Some(first) = wrapped.split('\n').next() {
            *last = first.to_owned();
        }
    }
    while shown.len() < room {
        shown.push(String::new());
    }
    let mut lines = top;
    lines.extend(shown);
    lines.extend(bottom);
    lines.truncate(rows.max(1));
    let mut out = String::from("\x1b[H");
    out.push_str(&lines.join("\x1b[K\n"));
    out.push_str("\x1b[K\x1b[J");
    (out, scroll)
}

/// Each slot's rate from `before` to `after`, `took` apart: the tokens its
/// request generated in between, `None` for a slot that did not decode one
/// request in both.
pub(crate) fn rates(before: &[Slot], after: &[Slot], took: Duration) -> Vec<Option<f64>> {
    let secs = took.as_secs_f64();
    after
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let b = before.get(i)?;
            let same = b.phase == Phase::Decode && a.phase == Phase::Decode && b.task == a.task;
            #[allow(clippy::cast_precision_loss, reason = "a display rate to one decimal")]
            (same && a.n_decoded > b.n_decoded && secs > 0.0)
                .then(|| (a.n_decoded - b.n_decoded) as f64 / secs)
        })
        .collect()
}

/// The wordmark (beside the small furnace when `tall`) with the model, the
/// time served and the live mark at its end; then the status, the seat, the
/// address, and the load time at its end.
fn header(now: &Now, fixed: &Fixed, style: Style, inner: usize, tall: bool) -> Vec<String> {
    let dot = Marks::of(style).dot;
    let bright = format!("{}{}", style.code(BOLD), style.fg(YELLOW));
    const WORD: &str = "BLOOMERY";
    let mut word = paint(WORD, &bright);
    let mut out = Vec::new();
    if tall {
        let furnace = screen::blocks(&screen::hearth(), style);
        let pad = " ".repeat(screen::width(WORD) + 2);
        if let Some((last, above)) = furnace.split_last() {
            out.extend(above.iter().map(|f| format!("{pad}{f}")));
            word.push("  ", "");
            word.append(&Text {
                shown: screen::strip_escapes(last),
                painted: last.clone(),
            });
        }
    }
    let mut right = paint(
        &format!(
            "{} {dot} up {}  ",
            screen::plain(&fixed.model, style),
            uptime(fixed.since.elapsed())
        ),
        style.code(DIM),
    );
    right.push("[LIVE]", &bright);
    out.extend(parts::spread(&word, &right, inner));

    let mut status = screen::badge(
        match now.status {
            Status::Ready => "READY",
            Status::Stopping => "STOPPING",
            Status::Failed => "FAILED",
        },
        style,
    );
    if let Some(f) = &fixed.facts {
        status.push("  ", "");
        status.push(&screen::plain(&f.seat, style), style.code(BOLD));
    }
    status.push("  ", "");
    status.push(
        &fixed.url(),
        &format!(
            "{}{}{}",
            style.code(BOLD),
            style.code(UNDERLINE),
            style.fg(YELLOW)
        ),
    );
    let loaded = fixed
        .row("loaded", style)
        .map(|l| paint(&format!("loaded {l}"), style.code(DIM)))
        .unwrap_or_default();
    out.extend(parts::spread(&status, &loaded, inner));
    out
}

/// The tabs in rows of `inner` columns, the open one marked (on the accent,
/// in brackets without colour), over a rule.
fn tab_row(tabs: &[Tab], open: Tab, style: Style, inner: usize) -> Vec<String> {
    let mut rows = vec![Text::default()];
    for (i, t) in tabs.iter().enumerate() {
        let label = format!("{} {}", i + 1, t.name());
        let chip = match (*t == open, style.color) {
            (true, true) => {
                let on = format!("{}{}{}", style.bg(YELLOW), style.fg(INK), style.code(BOLD));
                paint(&format!(" {label} "), &on)
            }
            (true, false) => paint(&format!("[{label}]"), ""),
            (false, _) => paint(&format!(" {label} "), ""),
        };
        let at = rows.last().map_or(0, |r| screen::width(&r.shown));
        if at > 0 && at + 1 + screen::width(&chip.shown) > inner {
            rows.push(Text::default());
        } else if at > 0
            && let Some(row) = rows.last_mut()
        {
            row.push(" ", "");
        }
        if let Some(row) = rows.last_mut() {
            row.append(&chip);
        }
    }
    let rule = if style.utf8 { "─" } else { "-" };
    let mut out: Vec<String> = rows.into_iter().map(|r| r.painted).collect();
    out.push(paint(&rule.repeat(inner), style.code(DIM)).painted);
    out
}

/// The seat's states, the tiles, the context and experts bars, and the last
/// requests that ended (beside the bars when there is room).
fn overview(now: &Now, fixed: &Fixed, style: Style, inner: usize) -> Vec<String> {
    let mut out = seat(now, fixed, style, inner);
    out.push(String::new());
    out.extend(parts::tiles(&tiles(now, fixed, style), inner, style));
    out.push(String::new());
    let table = recent(&now.requests, style);
    let wide = table.iter().map(|l| parts::shown(l)).max().unwrap_or(0);
    if inner >= BARS + GAP + wide {
        out.extend(parts::beside(&bars(now, fixed, style, BARS), BARS, &table));
    } else {
        out.extend(bars(now, fixed, style, inner));
        out.push(String::new());
        out.extend(table);
    }
    out
}

/// The seat's states in a row, each one that holds now marked (in brackets),
/// with a count when there are several slots; the placement at its end.
fn seat(now: &Now, fixed: &Fixed, style: Style, inner: usize) -> Vec<String> {
    let many = now.slots.len() > 1;
    let mut chips = Vec::new();
    for (name, phase) in [
        ("Idle", Phase::Idle),
        ("Prefill", Phase::Prefill),
        ("Decode", Phase::Decode),
        ("Held", Phase::Held),
    ] {
        let n = now.slots.iter().filter(|s| s.phase == phase).count();
        if phase == Phase::Held && n == 0 {
            continue;
        }
        let label = if many && n > 0 {
            format!("{name} {n}")
        } else {
            name.to_owned()
        };
        chips.push((label, n > 0, YELLOW));
    }
    let queued = match now.queued {
        0 => "Queued".to_owned(),
        n => format!("Queued {n}"),
    };
    chips.push((queued, now.queued > 0, YELLOW));
    chips.push(("Fault".to_owned(), now.status == Status::Failed, RED));

    let mut row = paint("Seat", &accent(style));
    for (label, on, c) in chips {
        row.push(" ", "");
        if on {
            row.push(
                &format!("[{label}]"),
                &format!("{}{}", style.code(BOLD), style.fg(c)),
            );
        } else {
            row.push(&format!(" {label} "), style.code(DIM));
        }
    }
    let place = fixed
        .row("place", style)
        .map(|p| paint(&format!("place {p}"), style.code(DIM)))
        .unwrap_or_default();
    parts::spread(&row, &place, inner)
}

/// The tiles there are numbers for: decode and prefill, the card's memory,
/// the host's and the tier card's experts, the prompt cache, the NVMe tier.
fn tiles(now: &Now, fixed: &Fixed, style: Style) -> Vec<Tile> {
    let dot = Marks::of(style).dot;
    let average = |when: String, r: f64| {
        if r > 0.0 {
            format!("{when} {dot} {r:.1} avg")
        } else {
            when
        }
    };
    let decoding: Vec<f64> = now.rates.iter().flatten().copied().collect();
    let last = now.requests.first().map(|f| &f.timings);
    let (value, when) = match (decoding.len(), last) {
        (0, Some(t)) if t.predicted_n > 0 => (
            format!("{:.1} tok/s", rate(t.predicted_n as u64, t.predicted_ms)),
            "last request".to_owned(),
        ),
        (0, _) => ("none yet".to_owned(), "no request has ended".to_owned()),
        (n, _) => (
            format!("{:.1} tok/s", decoding.iter().sum::<f64>()),
            if n == 1 {
                "now".to_owned()
            } else {
                format!("now, {n} slots")
            },
        ),
    };
    let mut out = vec![Tile {
        title: "Decode",
        value,
        note: average(when, now.decode_rate),
        meter: None,
    }];

    let prompt = now
        .requests
        .iter()
        .map(|f| &f.timings)
        .find(|t| t.prompt_n > 0 && t.prompt_ms > 0.0);
    let (value, when) = match prompt {
        Some(t) => (
            format!("{:.1} tok/s", rate(t.prompt_n as u64, t.prompt_ms)),
            format!("{} tokens", t.prompt_n),
        ),
        None => ("none yet".to_owned(), "no prompt has run".to_owned()),
    };
    out.push(Tile {
        title: "Prefill",
        value,
        note: average(when, now.prompt_rate),
        meter: None,
    });

    let facts = fixed.facts.as_ref();
    let sizes = facts.map(|f| f.sizes).unwrap_or_default();
    let split = facts.and_then(|f| f.split);
    if let Some(resident) = sizes.resident {
        out.push(Tile {
            title: "Card memory",
            value: match sizes.card_total {
                Some(t) => format!("{} of {}", screen::gib(resident), screen::gib(t)),
                None => screen::gib(resident),
            },
            note: split.map_or_else(
                || "resident".to_owned(),
                |s| format!("{} experts resident", s.card),
            ),
            meter: sizes.card_total.map(|t| (resident, t, YELLOW)),
        });
    }
    if let Some(s) = split {
        let total = s.card.saturating_add(s.tier).saturating_add(s.host);
        out.push(Tile {
            title: "Host experts",
            value: s.host.to_string(),
            note: sizes.host.map_or_else(
                || format!("of {total} experts"),
                |b| format!("{} in RAM", screen::gib(b)),
            ),
            meter: Some((s.host, total, HOST)),
        });
        if s.tier > 0 {
            out.push(Tile {
                title: "Tier card",
                value: s.tier.to_string(),
                note: format!("of {total} experts"),
                meter: Some((s.tier, total, RED)),
            });
        }
    }

    let prompted = now.cached.saturating_add(now.evaluated);
    out.push(Tile {
        title: "Prompt cache",
        value: match percent(now.cached, prompted) {
            Some(p) => format!("{} reused ({p})", now.cached),
            None => format!("{} reused", now.cached),
        },
        note: match sizes.cache_ram {
            Some(0) => "no prompt cache".to_owned(),
            Some(r) => format!("{} in RAM", screen::gib(r)),
            None => "prompt tokens since start".to_owned(),
        },
        meter: (prompted > 0).then_some((now.cached, prompted, YELLOW)),
    });

    if let (Some(paged), Some(note)) = (sizes.paged, fixed.row("nvme", style)) {
        out.push(Tile {
            title: "NVMe tier",
            value: match paged {
                0 => "off".to_owned(),
                p => format!("{} paged", screen::gib(p)),
            },
            note,
            meter: None,
        });
    }
    out
}

/// The context the slots hold against their room, and where the plan put
/// the routed experts, each a label line over a bar `w` cells wide.
fn bars(now: &Now, fixed: &Fixed, style: Style, w: usize) -> Vec<String> {
    let dot = Marks::of(style).dot;
    let held: usize = now.slots.iter().map(|s| s.n_past).sum();
    let mut context = paint("Context", &accent(style));
    context.push(&format!("  {held} of {}", now.room), "");
    if let Some(p) = percent(held as u64, now.room as u64) {
        context.push(&format!("  {p}"), "");
    }
    let mut out = vec![
        context.painted,
        parts::meter(held as u64, now.room as u64, w, YELLOW, true, style).painted,
    ];
    if let Some(s) = fixed.facts.as_ref().and_then(|f| f.split)
        && let Some(cells) = screen::expert_cells(s, w, style)
    {
        let total = s.card.saturating_add(s.tier).saturating_add(s.host);
        let shares: Vec<String> = [("card", s.card), ("tier", s.tier), ("host", s.host)]
            .into_iter()
            .filter(|&(_, n)| n > 0)
            .filter_map(|(name, n)| Some(format!("{name} {}", percent(n, total)?)))
            .collect();
        let mut experts = paint("Experts", &accent(style));
        experts.push(&format!("  {}", shares.join(&format!(" {dot} "))), "");
        out.push(String::new());
        out.push(experts.painted);
        out.push(cells.painted);
    }
    out
}

/// A value of a request that ended, as a table's column shows it.
#[derive(Clone, Copy)]
enum Cell {
    Time,
    Status,
    Prompt,
    Reused,
    Output,
    Prefill,
    Decode,
    Engine,
}

/// `done` as a table of `cols` (each a title, a width and the value it
/// shows) under the titles dimmed: the time and the status from the left,
/// the counts from the right, the status in its colour (done in the accent,
/// a failure or a closed client in red, a stop dimmed).
fn table(done: &[Finished], cols: &[(&str, usize, Cell)], style: Style) -> Vec<String> {
    let place = |text: &str, w: usize, cell: Cell| match cell {
        Cell::Time | Cell::Status => format!("{text:<w$}"),
        _ => format!("{text:>w$}"),
    };
    let mut head = Text::default();
    for (i, &(title, w, cell)) in cols.iter().enumerate() {
        let gap = if i > 0 { "  " } else { "" };
        head.push(&format!("{gap}{}", place(title, w, cell)), style.code(DIM));
    }
    let mut out = vec![head.painted];
    for f in done {
        let t = &f.timings;
        let mut row = Text::default();
        for (i, &(_, w, cell)) in cols.iter().enumerate() {
            if i > 0 {
                row.push("  ", "");
            }
            let (text, code) = match cell {
                Cell::Time => (clock(f.at), String::new()),
                Cell::Status => (
                    f.ending.as_str().to_owned(),
                    match f.ending {
                        Ending::Done(_) => style.fg(YELLOW),
                        Ending::ClientClosed | Ending::EngineError | Ending::Error => style.fg(RED),
                        Ending::Stopped => style.code(DIM).to_owned(),
                    },
                ),
                Cell::Prompt => (t.n_prompt.to_string(), String::new()),
                Cell::Reused => (t.cache_n.to_string(), String::new()),
                Cell::Output => (t.predicted_n.to_string(), String::new()),
                Cell::Prefill => (tps(t.prompt_n, t.prompt_ms), String::new()),
                Cell::Decode => (tps(t.predicted_n, t.predicted_ms), String::new()),
                Cell::Engine => (
                    secs(t.cache_ms + t.prompt_ms + t.predicted_ms),
                    String::new(),
                ),
            };
            row.push(&place(&text, w, cell), &code);
        }
        out.push(row.painted);
    }
    out
}

/// `n` tokens over `ms` as a rate to one decimal, `-` for none.
fn tps(n: usize, ms: f64) -> String {
    let r = rate(n as u64, ms);
    if r > 0.0 {
        format!("{r:.1}")
    } else {
        "-".to_owned()
    }
}

/// `ms` in seconds to one decimal.
fn secs(ms: f64) -> String {
    format!("{:.1} s", ms / 1e3)
}

/// The last [`RECENT`] requests that ended, under their heading.
fn recent(done: &[Finished], style: Style) -> Vec<String> {
    let mut out = vec![paint("Recent requests", &accent(style)).painted];
    if done.is_empty() {
        out.push(paint("no request has ended yet", style.code(DIM)).painted);
        return out;
    }
    out.extend(table(
        &done[..done.len().min(RECENT)],
        &[
            ("TIME", 8, Cell::Time),
            ("STATUS", 13, Cell::Status),
            ("PROMPT", 7, Cell::Prompt),
            ("REUSED", 7, Cell::Reused),
            ("OUTPUT", 7, Cell::Output),
            ("TOK/S", 7, Cell::Decode),
            ("TOTAL", 8, Cell::Engine),
        ],
        style,
    ));
    out
}

/// The requests that ended, newest first, how the kept ones ended, and the
/// newest one's time split in a box.
fn requests(done: &[Finished], style: Style, inner: usize) -> Vec<String> {
    let dim = style.code(DIM);
    let dot = Marks::of(style).dot;
    let mut summary = paint("Requests", &accent(style));
    if done.is_empty() {
        summary.push("  no request has ended yet", dim);
        return vec![summary.painted];
    }
    let count = |keep: &dyn Fn(Ending) -> bool| done.iter().filter(|f| keep(f.ending)).count();
    let counts = [
        (count(&|e| matches!(e, Ending::Done(_))), "done"),
        (
            count(&|e| e == Ending::ClientClosed),
            "closed by the client",
        ),
        (
            count(&|e| matches!(e, Ending::EngineError | Ending::Error)),
            "failed",
        ),
        (count(&|e| e == Ending::Stopped), "stopped"),
    ];
    let said: Vec<String> = counts
        .iter()
        .filter(|&&(n, _)| n > 0)
        .map(|(n, what)| format!("{n} {what}"))
        .collect();
    summary.push(
        &format!(
            "  the last {}: {}",
            done.len(),
            said.join(&format!(" {dot} "))
        ),
        dim,
    );
    let mut out = vec![summary.painted, String::new()];

    out.extend(table(
        done,
        &[
            ("TIME", 8, Cell::Time),
            ("PROMPT", 7, Cell::Prompt),
            ("REUSED", 7, Cell::Reused),
            ("OUTPUT", 7, Cell::Output),
            ("PREFILL T/S", 11, Cell::Prefill),
            ("DECODE T/S", 10, Cell::Decode),
            ("ENGINE", 8, Cell::Engine),
            ("STATUS", 13, Cell::Status),
        ],
        style,
    ));
    out.push(String::new());
    out.extend(latest(&done[0], style, inner));
    out
}

/// The newest request's engine time split into the cache lookup, the
/// prefill and the decode, as a bar with its legend, in a box `w` wide.
fn latest(f: &Finished, style: Style, w: usize) -> Vec<String> {
    let inside = w.saturating_sub(4).max(1);
    let t = &f.timings;
    let mut head = paint(&format!("Latest {}", clock(f.at)), &accent(style));
    head.push(
        &format!(
            "  engine time {}",
            secs(t.cache_ms + t.prompt_ms + t.predicted_ms)
        ),
        style.code(DIM),
    );
    let mut lines = vec![head.painted];
    let glyphs = if style.utf8 {
        ["░", "▓", "█"]
    } else {
        [".", "=", "#"]
    };
    let spans = [
        (t.cache_ms, SLATE, "cache lookup".to_owned()),
        (
            t.prompt_ms,
            ORANGE,
            format!("prefill {} new tokens", t.prompt_n),
        ),
        (
            t.predicted_ms,
            YELLOW,
            format!("decode {} tokens", t.predicted_n),
        ),
    ];
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a share of a bar: whole microseconds, a negative time as none"
    )]
    let micros = spans.each_ref().map(|(ms, ..)| (ms.max(0.0) * 1e3) as u64);
    if let Some(cells) = screen::bar_cells(micros, inside) {
        let mut bar = Text::default();
        for ((n, glyph), (_, c, _)) in cells.iter().zip(glyphs).zip(&spans) {
            bar.push(&glyph.repeat(*n), &style.fg(*c));
        }
        lines.push(bar.painted);
    }
    let legend: Vec<Text> = spans
        .iter()
        .zip(glyphs)
        .map(|((ms, c, what), glyph)| {
            let mut l = paint(glyph, &style.fg(*c));
            l.push(&format!(" {what} {}", secs(*ms)), "");
            l
        })
        .collect();
    lines.extend(
        screen::key_rows(&legend, inside)
            .into_iter()
            .map(|l| l.painted),
    );
    let said = format!(
        "{} of {} prompt tokens came from the prompt cache and {} ran through prefill.",
        t.cache_n, t.n_prompt, t.prompt_n
    );
    lines.extend(
        screen::wrap(&said, inside, Marks::of(style).dot)
            .iter()
            .map(|l| paint(l, style.code(DIM)).painted),
    );
    parts::boxed(&lines, w, style)
}

/// Where the plan put the routed experts as one bar across the width, a
/// legend of the card, the tier card, the host and the NVMe tier (a part
/// that holds nothing dimmed and named none), then the start's sections
/// of labelled values: the card's and the host's in one column, the NVMe
/// tier's and the context's beside them when two fit, and any unread
/// record across the width.
fn placement(fixed: &Fixed, style: Style, inner: usize) -> Vec<String> {
    let Some(f) = &fixed.facts else {
        return Vec::new();
    };
    let dot = Marks::of(style).dot;
    let mut out = Vec::new();
    if let Some(s) = f.split
        && let Some(cells) = screen::expert_cells(s, inner, style)
    {
        let total = s.card.saturating_add(s.tier).saturating_add(s.host);
        let mut head = paint("Where the plan put the routed experts", &accent(style));
        let place = fixed
            .row("place", style)
            .map(|p| format!("place {p} {dot} "))
            .unwrap_or_default();
        head.push(&format!("  {place}{total} experts"), style.code(DIM));
        out.push(head.painted);
        out.push(cells.painted);
        let mut legend: Vec<Text> = screen::parts(style)
            .into_iter()
            .zip([("card", s.card), ("tier card", s.tier), ("host", s.host)])
            .map(|((glyph, _, c), (name, n))| {
                if n == 0 {
                    return paint(&format!("{glyph} {name} none"), style.code(DIM));
                }
                let mut l = paint(glyph, &style.fg(c));
                let share = percent(n, total).map(|p| format!(" ({p})"));
                l.push(&format!(" {name} {n}{}", share.unwrap_or_default()), "");
                l
            })
            .collect();
        legend.extend(f.sizes.paged.map(|p| match p {
            0 => paint("nvme none", style.code(DIM)),
            p => paint(&format!("nvme {} paged", screen::gib(p)), ""),
        }));
        out.extend(
            screen::key_rows(&legend, inner)
                .into_iter()
                .map(|l| l.painted),
        );
    }
    let two = inner >= 2 * SECTION + GAP;
    let cw = if two { (inner - GAP) / 2 } else { inner };
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (name, values) in &f.sections {
        let side = if two && ["nvme", "context"].contains(name) {
            &mut right
        } else {
            &mut left
        };
        if !side.is_empty() {
            side.push(String::new());
        }
        side.push(screen::heading(name, cw, style).painted);
        for (label, value) in values {
            let value = screen::plain(value, style);
            side.extend(
                screen::row_lines(label, &value, cw, style, dot)
                    .into_iter()
                    .map(|t| t.painted),
            );
        }
    }
    if !left.is_empty() || !right.is_empty() {
        out.push(String::new());
    }
    out.extend(parts::beside(&left, cw, &right));
    if let Some(unread) = fixed.row("unread", style) {
        out.push(String::new());
        out.extend(
            screen::row_lines("unread", &unread, inner, style, dot)
                .into_iter()
                .map(|t| t.painted),
        );
    }
    out
}

/// The record lines the start printed, each led by its kind, in a box.
fn records(fixed: &Fixed, style: Style, inner: usize) -> Vec<String> {
    let mut out = parts::spread(
        &paint("Records", &accent(style)),
        &paint(
            &format!("{} lines, as printed on stderr", fixed.records.len()),
            style.code(DIM),
        ),
        inner,
    );
    out.push(String::new());
    let inside = inner.saturating_sub(4).max(1);
    let dot = Marks::of(style).dot;
    let mut lines = Vec::new();
    for r in &fixed.records {
        let r = screen::plain(r, style);
        let kind = r.split(' ').next().map_or(0, str::len);
        for (i, piece) in screen::wrap(&r, inside, dot).into_iter().enumerate() {
            let line = match (i, piece.split_at_checked(kind.min(piece.len()))) {
                (0, Some((head, rest))) => {
                    let mut l = paint(head, style.code(BOLD));
                    l.push(rest, "");
                    l
                }
                _ => paint(&piece, ""),
            };
            lines.push(line.painted);
        }
    }
    out.extend(parts::boxed(&lines, inner, style));
    out
}

/// The routes and who can reach them; beside them (or under them when
/// narrow) a command that tries the server and what a client needs.
fn connect(fixed: &Fixed, style: Style, inner: usize) -> Vec<String> {
    let two = inner >= ROUTES + GAP + 50;
    let (lw, rw) = if two {
        (ROUTES, inner - ROUTES - GAP)
    } else {
        (inner, inner)
    };
    let dot = Marks::of(style).dot;
    let rows = |pairs: &[(&str, String)], w: usize| -> Vec<String> {
        pairs
            .iter()
            .flat_map(|(label, value)| screen::row_lines(label, value, w, style, dot))
            .map(|t| t.painted)
            .collect()
    };
    let model = screen::plain(&fixed.model, style);

    let mut left = vec![screen::heading("endpoints", lw, style).painted];
    for (method, path) in &fixed.routes {
        let code = if *method == "POST" {
            style.fg(YELLOW)
        } else {
            style.code(DIM).to_owned()
        };
        let mut l = paint(&format!("{method:<6}"), &code);
        l.push(path, "");
        left.push(l.painted);
    }
    left.push(String::new());
    left.push(screen::heading("access", lw, style).painted);
    let ip = fixed.addr.ip();
    let bind = if ip.is_loopback() {
        format!("{ip}, this machine only")
    } else if ip.is_unspecified() {
        format!("{ip}, every interface")
    } else {
        ip.to_string()
    };
    let key = if fixed.keyed {
        "required"
    } else {
        "not checked"
    };
    left.extend(rows(
        &[
            ("bind", bind),
            ("API key", key.to_owned()),
            ("model id", model.clone()),
        ],
        lw,
    ));
    let mut hint = Vec::new();
    if ip.is_loopback() {
        hint.push("Start with --host 0.0.0.0 to reach the server from another machine.");
    }
    if !fixed.keyed {
        hint.push("Set --api-key to require a key.");
    }
    if !hint.is_empty() {
        left.push(String::new());
        left.extend(
            screen::wrap(&hint.join(" "), lw, dot)
                .iter()
                .map(|l| paint(l, style.code(DIM)).painted),
        );
    }

    let inside = rw.saturating_sub(4).max(1);
    let url = fixed.url();
    let tries: Vec<String> = screen::try_lines(&url, false, style)
        .iter()
        .flat_map(|l| {
            screen::wrap_painted(l, inside, true)
                .split('\n')
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    let mut right = vec![paint("Try it", &accent(style)).painted];
    right.extend(parts::boxed(&tries, rw, style));
    right.push(String::new());
    right.push(paint("OpenAI-compatible clients", &accent(style)).painted);
    let key = if fixed.keyed {
        "a key set with --api-key"
    } else {
        "any text; the server does not check it"
    };
    let clients = rows(
        &[
            ("base URL", format!("{url}/v1")),
            ("API key", key.to_owned()),
            ("model", model),
        ],
        inside,
    );
    right.extend(parts::boxed(&clients, rw, style));

    if two {
        parts::beside(&left, lw, &right)
    } else {
        left.push(String::new());
        left.extend(right);
        left
    }
}

/// `part` of `whole` as a percentage to one decimal, `None` of nothing.
fn percent(part: u64, whole: u64) -> Option<String> {
    #[allow(clippy::cast_precision_loss, reason = "a display share to one decimal")]
    (whole > 0).then(|| format!("{:.1} %", 100.0 * part as f64 / whole as f64))
}

/// `d` in hours and minutes, or seconds below a minute.
fn uptime(d: Duration) -> String {
    let s = d.as_secs();
    match (s / 3600, s / 60 % 60) {
        (0, 0) => format!("{s} s"),
        (0, m) => format!("{m} m"),
        (h, m) => format!("{h} h {m} m"),
    }
}

/// `at` as the local time of day, `HH:MM:SS`; dashes for a time the clock
/// cannot name.
fn clock(at: SystemTime) -> String {
    let Ok(since) = at.duration_since(UNIX_EPOCH) else {
        return "--:--:--".to_owned();
    };
    let Ok(secs) = libc::time_t::try_from(since.as_secs()) else {
        return "--:--:--".to_owned();
    };
    // SAFETY: an all-zero tm is a valid value for localtime_r to fill.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: secs and tm are valid for the call, which fills tm.
    if unsafe { libc::localtime_r(&secs, &mut tm) }.is_null() {
        return "--:--:--".to_owned();
    }
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant, SystemTime};

    use super::super::ring::{Ending, Finished, KEPT, Ring};
    use super::{Fixed, Now, Phase, Slot, Status, Tab, WIDEST, frame, rates};
    use crate::genloop::Timings;
    use crate::screen::{self, Marks, Style, width};

    const START: &str = include_str!("../../tests/fixtures/screen-start.log");

    fn style(color: bool, utf8: bool) -> Style {
        Style {
            color,
            truecolor: color,
            utf8,
        }
    }

    fn finished(i: usize, ending: Ending) -> Finished {
        Finished {
            at: SystemTime::now(),
            timings: Timings {
                prompt_n: 100 + i,
                prompt_ms: 50.0,
                predicted_n: 10,
                predicted_ms: 500.0,
                cache_n: 7,
                n_prompt: 107 + i,
                ..Timings::default()
            },
            ending,
        }
    }

    fn fixed(records: &[&str], s: Style) -> Fixed {
        let records: Vec<String> = records.iter().map(|&r| r.to_owned()).collect();
        Fixed {
            addr: "127.0.0.1:8080".parse().unwrap(),
            model: "DeepSeek-V4.1".to_owned(),
            keyed: false,
            since: Instant::now(),
            facts: screen::facts(&records, Marks::of(s)),
            records,
            routes: crate::api::LISTED.to_vec(),
        }
    }

    fn now() -> Now {
        let mut ring = Ring::default();
        ring.push(finished(0, Ending::Done("stop")));
        ring.push(finished(1, Ending::ClientClosed));
        Now {
            status: Status::Ready,
            slots: vec![
                slot(Phase::Decode, 4812, 1, 812),
                slot(Phase::Prefill, 0, 2, 0),
                slot(Phase::Idle, 0, 0, 0),
            ],
            rates: vec![Some(31.3), None, None],
            queued: 1,
            room: 3 * 8192,
            prompt_rate: 812.04,
            decode_rate: 23.44,
            cached: 9,
            evaluated: 27,
            requests: ring.newest_first(),
        }
    }

    fn slot(phase: Phase, n_past: usize, task: u64, n_decoded: usize) -> Slot {
        Slot {
            phase,
            n_past,
            task,
            n_decoded,
        }
    }

    /// A slot's rate counts only the tokens one request generated in both
    /// frames: not a prompt's, not another request's that took the slot.
    #[test]
    fn rates_count_only_one_requests_tokens() {
        let before = [
            slot(Phase::Decode, 100, 1, 40),
            slot(Phase::Prefill, 0, 2, 0),
            slot(Phase::Decode, 900, 3, 300),
            slot(Phase::Idle, 0, 0, 0),
        ];
        let after = [
            slot(Phase::Decode, 150, 1, 90),
            slot(Phase::Decode, 512, 2, 1),
            slot(Phase::Decode, 1800, 4, 400),
            slot(Phase::Idle, 0, 0, 0),
        ];
        let got = rates(&before, &after, Duration::from_millis(500));
        assert_eq!(got, vec![Some(100.0), None, None, None]);
    }

    /// Each tab's frame holds its own rows and fits the terminal: every line
    /// within the columns, no more lines than the rows; without colour no
    /// escape but the cursor's, in ASCII no byte past it. Every row of tiles
    /// runs the full width, so no row ends in an empty box, and a row holds
    /// four tiles when four fit. A start that
    /// noted no record (a seat that prints none) has no Placement and no
    /// Records tab, and no tile or bar of theirs anywhere.
    #[test]
    fn frames_show_each_tab_within_the_terminal() {
        let lines: Vec<&str> = START.lines().collect();
        // The tops of the Overview's boxes, which are its tiles'.
        let tops = |text: &str| -> Vec<usize> {
            text.lines()
                .filter(|l| ["┌", "+-"].iter().any(|c| l.trim_start().starts_with(c)))
                .map(width)
                .collect()
        };
        for (s, cols, rows) in [
            (style(true, true), 120, 38),
            (style(false, true), 80, 24),
            (style(false, false), 60, 30),
            (style(true, true), 20, 12),
        ] {
            let f = fixed(&lines, s);
            let tabs = f.tabs();
            assert_eq!(
                tabs,
                [
                    Tab::Overview,
                    Tab::Placement,
                    Tab::Requests,
                    Tab::Records,
                    Tab::Connect
                ]
            );
            let full = cols.min(WIDEST);
            for (i, tab) in tabs.iter().enumerate() {
                let (got, _) = frame(&now(), &f, i, 0, s, cols, rows);
                let text = screen::strip_escapes(&got);
                let shown: Vec<&str> = text.split('\n').collect();
                assert!(
                    shown.len() <= rows,
                    "{tab:?} {cols}x{rows}: {} lines",
                    shown.len()
                );
                for l in &shown {
                    assert!(width(l) <= cols, "{tab:?} {cols}: {l:?}");
                }
                if !s.color {
                    assert!(!got.contains("\x1b[38"), "colour in {tab:?}");
                }
                if !s.utf8 {
                    assert!(got.is_ascii(), "{tab:?}: {got}");
                }
                if cols < 60 {
                    continue;
                }
                let (all, _) = frame(&now(), &f, i, 0, s, cols, 400);
                let all = screen::strip_escapes(&all);
                if *tab == Tab::Overview {
                    let rows = tops(&all);
                    assert!(
                        !rows.is_empty() && rows.iter().all(|&w| w == full),
                        "{rows:?}\n{all}"
                    );
                    let first = all
                        .lines()
                        .find(|l| {
                            ["│ Decode ", "| Decode "]
                                .iter()
                                .any(|p| l.trim_start().starts_with(p))
                        })
                        .unwrap_or("");
                    assert_eq!(
                        first.contains("Host experts"),
                        cols >= 100,
                        "four tiles to a row when they fit:\n{all}"
                    );
                }
                let compact: String = all.chars().filter(|c| !c.is_whitespace()).collect();
                let want: &[&str] = match tab {
                    Tab::Overview => &[
                        "BLOOMERY",
                        "[LIVE]",
                        "loadedin41.2s",
                        "[Decode1]",
                        "[Prefill1]",
                        "[Queued1]",
                        "31.3tok/s",
                        "23.4avg",
                        "2020.0tok/s",
                        "45.0GiBof47.5GiB",
                        "Hostexperts",
                        "199.3GiBinRAM",
                        "9reused(25.0%)",
                        "NVMetier",
                        "4812of24576",
                        "card17.4%",
                        "clientclosed",
                    ],
                    Tab::Placement => &[
                        "Wheretheplanput",
                        "card2668(17.4%)",
                        "tiercardnone",
                        "nvmenone",
                        "deviceA6000cuda0",
                        "resident45.0GiBof47.5GiB",
                        "budgetnone",
                        "shadow1.2GiB",
                        "ctx_max32768",
                        "slots2slots",
                    ],
                    Tab::Requests => &[
                        "PREFILLT/S",
                        "2000.0",
                        "20.0",
                        "stop",
                        "clientclosed",
                        "Latest",
                        "cachelookup",
                        "7of108prompttokens",
                    ],
                    Tab::Records => &["10lines,asprintedonstderr", "listeningon"],
                    Tab::Connect => &[
                        "/v1/chat/completions",
                        "POST",
                        "/metrics",
                        "Hello",
                        "thismachineonly",
                        "notchecked",
                        "OpenAI-compatibleclients",
                    ],
                };
                for w in want {
                    assert!(compact.contains(w), "{tab:?} {cols}: no {w}\n{all}");
                }
            }
        }

        let s = style(false, true);
        let bare = fixed(&[], s);
        assert_eq!(bare.tabs(), [Tab::Overview, Tab::Requests, Tab::Connect]);
        let (got, _) = frame(&now(), &bare, 0, 0, s, 120, 400);
        let got = screen::strip_escapes(&got);
        for gone in [
            "Card memory",
            "Host experts",
            "NVMe",
            "Experts",
            "placement",
        ] {
            assert!(!got.contains(gone), "{gone} without a record:\n{got}");
        }
        assert_eq!(tops(&got), [120], "three tiles in one full row:\n{got}");
    }

    /// A body longer than the room scrolls: the start clamps to the last
    /// screenful, and a line names what is left below.
    #[test]
    fn a_long_tab_scrolls_and_names_what_is_below() {
        let s = style(false, true);
        let lines: Vec<&str> = START.lines().collect();
        let f = fixed(&lines, s);
        let records = f.tabs().iter().position(|t| *t == Tab::Records).unwrap();
        let (got, at) = frame(&now(), &f, records, 0, s, 98, 16);
        assert_eq!(at, 0);
        assert!(got.contains("more lines below"), "{got}");
        let (got, at) = frame(&now(), &f, records, usize::MAX, s, 98, 16);
        assert!(at > 0 && at < usize::MAX);
        assert!(!got.contains("more lines below"), "{got}");
    }

    /// The ring keeps the newest [`KEPT`] requests, newest first.
    #[test]
    fn the_ring_keeps_the_newest() {
        let mut ring = Ring::default();
        for i in 0..KEPT + 5 {
            ring.push(finished(i, Ending::Done("stop")));
        }
        let kept = ring.newest_first();
        assert_eq!(kept.len(), KEPT);
        assert_eq!(kept[0].timings.prompt_n, 100 + KEPT + 4);
        assert_eq!(kept[KEPT - 1].timings.prompt_n, 100 + 5);
    }
}
