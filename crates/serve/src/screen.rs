//! The start screen: a view for a person over the record lines a seat printed
//! while it started, drawn once on stderr below them when the server begins
//! to serve. A seat arms it ([`arm`]) when `BLOOMERY_SCREEN` is on and stderr
//! is a terminal, and hands it each record line it prints ([`note`]); piped
//! stderr is never armed, so it carries the record lines alone.
//!
//! The screen reads the lines' text as printed and nothing else: every value
//! on it is a record's, shown as the record wrote it, a byte count in GiB to
//! one decimal. A row is read from the last line of its record's head, so the
//! order of lines of different heads does not move it; a fact whose record
//! the start did not print is left off, and two records that name different
//! placements are not joined. A record the start printed whose value the
//! screen could not read as a number is named on an `unread` row, never left
//! off as if it had not been printed. `NO_COLOR` (set and not empty) and
//! `TERM=dumb` draw it without colour; `TERM=dumb` or a locale that does not
//! name UTF-8 draws it in plain ASCII. `COLORTERM=truecolor` (or `24bit`)
//! draws its colours as they are; any other terminal gets the nearest of the
//! 256-colour cube.
//!
//! The screen is the wordmark and the furnace in half-block pixels, a status
//! line with the address, then two panels: where the model went (the card,
//! the host and the NVMe tier, with a bar of the experts each holds) and how
//! it serves (the placement, the context and slots, the prompt cache).
//! Below the art's width it uses a text wordmark and stacks the panels;
//! the width is read when it draws, after any resize during the load.

use std::cell::RefCell;
use std::io::IsTerminal;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// How the screen is drawn: with colour (24-bit, or the 256-colour cube),
/// and with UTF-8 glyphs or plain ASCII.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Style {
    color: bool,
    truecolor: bool,
    utf8: bool,
}

/// The terminal variables [`Style::of`] reads, each as set (`None` unset).
#[derive(Clone, Copy, Debug, Default)]
struct TermVars<'a> {
    term: Option<&'a str>,
    no_color: Option<&'a str>,
    colorterm: Option<&'a str>,
    lc_all: Option<&'a str>,
    lc_ctype: Option<&'a str>,
    lang: Option<&'a str>,
}

impl Style {
    /// The style of `v`: colour unless `NO_COLOR` is set and not empty or
    /// `TERM` is `dumb`, 24-bit when `COLORTERM` is `truecolor` or `24bit`;
    /// UTF-8 when the first non-empty of `LC_ALL`, `LC_CTYPE` and `LANG`
    /// names it, and never on a `dumb` terminal.
    #[must_use]
    fn of(v: TermVars<'_>) -> Style {
        fn set(x: Option<&str>) -> Option<&str> {
            x.filter(|x| !x.is_empty())
        }
        let dumb = set(v.term) == Some("dumb");
        let no_color = set(v.no_color).is_some();
        let utf8 = set(v.lc_all)
            .or(set(v.lc_ctype))
            .or(set(v.lang))
            .is_some_and(|l| {
                let l = l.to_ascii_lowercase();
                l.contains("utf-8") || l.contains("utf8")
            });
        let color = !no_color && !dumb;
        Style {
            color,
            truecolor: color && matches!(set(v.colorterm), Some("truecolor" | "24bit")),
            utf8: utf8 && !dumb,
        }
    }

    /// The style of the process's terminal variables ([`Style::of`]).
    #[must_use]
    fn of_terminal() -> Style {
        let term = std::env::var_os("TERM");
        let no_color = std::env::var_os("NO_COLOR");
        let colorterm = std::env::var_os("COLORTERM");
        let lc_all = std::env::var_os("LC_ALL");
        let lc_ctype = std::env::var_os("LC_CTYPE");
        let lang = std::env::var_os("LANG");
        let s =
            |v: &Option<std::ffi::OsString>| v.as_ref().map(|v| v.to_string_lossy().into_owned());
        let (term, no_color, colorterm) = (s(&term), s(&no_color), s(&colorterm));
        let (lc_all, lc_ctype, lang) = (s(&lc_all), s(&lc_ctype), s(&lang));
        Style::of(TermVars {
            term: term.as_deref(),
            no_color: no_color.as_deref(),
            colorterm: colorterm.as_deref(),
            lc_all: lc_all.as_deref(),
            lc_ctype: lc_ctype.as_deref(),
            lang: lang.as_deref(),
        })
    }

    /// The escape that sets `c` as the foreground (`layer` 3) or the
    /// background (`layer` 4), empty without colour.
    fn ink(self, layer: u8, c: Rgb) -> String {
        let Rgb(r, g, b) = c;
        match (self.color, self.truecolor) {
            (false, _) => String::new(),
            (true, true) => format!("\x1b[{layer}8;2;{r};{g};{b}m"),
            (true, false) => format!("\x1b[{layer}8;5;{}m", cube(c)),
        }
    }

    /// `code` with colour, nothing without.
    fn code(self, code: &str) -> &str {
        if self.color { code } else { "" }
    }

    fn fg(self, c: Rgb) -> String {
        self.ink(3, c)
    }

    fn bg(self, c: Rgb) -> String {
        self.ink(4, c)
    }
}

/// The index of the xterm 256-colour cube's colour nearest `c`, channel by
/// channel.
fn cube(Rgb(r, g, b): Rgb) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let near = |v: u8| {
        (0u8..6)
            .min_by_key(|&i| LEVELS[usize::from(i)].abs_diff(v))
            .unwrap_or(0)
    };
    16 + 36 * near(r) + 6 * near(g) + near(b)
}

/// Whether the screen is armed and not drawn yet: [`note`] reads it before
/// it takes the tape's lock, so a line printed after the draw costs one load.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// The lines noted since [`arm`] and the style to draw them in; `None`
/// before an arming, after [`draw`], and in a process that never armed.
static TAPE: Mutex<Option<(Style, Vec<String>)>> = Mutex::new(None);

/// Arms the screen when `on` (the seat's `BLOOMERY_SCREEN`) and stderr is a
/// terminal, in the terminal's [`Style`]; returns whether it armed. A seat
/// whose screen armed hands it each record line it prints.
pub fn arm(on: bool) -> bool {
    if !on || !std::io::stderr().is_terminal() {
        return false;
    }
    *tape() = Some((Style::of_terminal(), Vec::new()));
    ACTIVE.store(true, Ordering::Release);
    true
}

/// Keeps `line`, a record line the seat printed, for the screen; nothing
/// unless armed and not drawn yet.
pub fn note(line: &str) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    if let Some((_, lines)) = tape().as_mut() {
        lines.push(line.to_owned());
    }
}

/// Draws the screen of the noted lines on stderr, once, and disarms: the
/// server calls it when it begins to serve. Unarmed, or with no `listening`
/// line noted, it draws nothing.
pub(crate) fn draw() {
    ACTIVE.store(false, Ordering::Release);
    let Some((style, lines)) = tape().take() else {
        return;
    };
    if Start::new(&lines).listening().is_none() {
        return;
    }
    let mut size = libc::winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: stderr supplies a live fd and size is a writable winsize for this ioctl.
    let rc = unsafe { libc::ioctl(std::io::stderr().as_raw_fd(), libc::TIOCGWINSZ, &mut size) };
    if rc != 0 {
        eprintln!(
            "start screen: terminal width: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    if size.ws_col < 2 {
        eprintln!("start screen: terminal width must be at least 2 columns");
        return;
    }
    if let Some(screen) = render(&lines, style, usize::from(size.ws_col)) {
        eprint!("{screen}");
    }
}

fn tape() -> std::sync::MutexGuard<'static, Option<(Style, Vec<String>)>> {
    TAPE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The literal every seat's `listening` record carries before its address.
const LISTENING: &str = " listening on http://";

/// The records the screen could not read a value of, in the order met.
#[derive(Default)]
struct Unread(RefCell<Vec<String>>);

impl Unread {
    /// Names the record `line` is, once.
    fn note(&self, line: &str) {
        let name = if line.contains(LISTENING) {
            "listening"
        } else {
            line.split(' ').next().unwrap_or(line)
        };
        let mut names = self.0.borrow_mut();
        if !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    }
}

/// The noted record lines the screen is drawn from, and the records it could
/// not read a value of on the way.
struct Start<'a, S> {
    lines: &'a [S],
    unread: Unread,
}

impl<'a, S: AsRef<str>> Start<'a, S> {
    fn new(lines: &'a [S]) -> Self {
        Start {
            lines,
            unread: Unread::default(),
        }
    }

    fn line(&'a self, text: &'a str) -> Line<'a> {
        Line {
            text,
            unread: &self.unread,
        }
    }

    /// The last noted line that starts with `head` and carries ` key=`.
    fn last(&'a self, head: &str, key: &str) -> Option<Line<'a>> {
        self.lines
            .iter()
            .map(|l| self.line(l.as_ref()))
            .rfind(|l| l.text.starts_with(head) && l.text(key).is_some())
    }

    /// The last line that carries the `listening` literal.
    fn listening(&'a self) -> Option<Line<'a>> {
        self.lines
            .iter()
            .map(|l| self.line(l.as_ref()))
            .rfind(|l| l.text.contains(LISTENING))
    }
}

/// One record line as printed, read by key.
#[derive(Clone, Copy)]
struct Line<'a> {
    text: &'a str,
    unread: &'a Unread,
}

impl<'a> Line<'a> {
    /// The text after ` key=`, up to the next ` name=` or the line's end.
    fn text(self, key: &str) -> Option<&'a str> {
        let pat = format!(" {key}=");
        let start = self.text.find(&pat)? + pat.len();
        let rest = &self.text[start..];
        Some(rest[..next_key(rest)].trim_end())
    }

    /// The first word of `key`'s text.
    fn word(self, key: &str) -> Option<&'a str> {
        self.text(key)?.split_whitespace().next()
    }

    /// `key`'s first word as a whole number; a value that is empty or none is
    /// noted as unread.
    fn num(self, key: &str) -> Option<u64> {
        let n = self
            .text(key)?
            .split_whitespace()
            .next()
            .and_then(|w| w.parse().ok());
        if n.is_none() {
            self.unread.note(self.text);
        }
        n
    }

    /// The `(<n> B)` written beside `key`'s value, `None` where the value has
    /// no parenthesis; a byte figure that is none is noted as unread.
    fn bytes_beside(self, key: &str) -> Option<u64> {
        let (_, inner) = self.text(key)?.split_once('(')?;
        let n = inner.strip_suffix(" B)").and_then(|b| b.parse().ok());
        if n.is_none() {
            self.unread.note(self.text);
        }
        n
    }

    /// Whether this line may be joined to the `listening` line's placement
    /// `place`: it names the same one, or one of the two names none.
    fn same_place(self, place: Option<&str>) -> bool {
        match (self.word("place"), place) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => true,
        }
    }
}

/// Where the next ` name=` starts in `rest`, or its length.
fn next_key(rest: &str) -> usize {
    let b = rest.as_bytes();
    let mut i = 0;
    while let Some(off) = rest[i..].find(' ') {
        let at = i + off;
        let name = b[at + 1..]
            .iter()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || **c == b'_')
            .count();
        if name > 0 && b.get(at + 1 + name) == Some(&b'=') {
            return at;
        }
        i = at + 1;
    }
    rest.len()
}

/// `b` bytes in GiB, one decimal.
fn gib(b: u64) -> String {
    #[allow(
        clippy::cast_precision_loss,
        reason = "a display rounding to one decimal of GiB"
    )]
    let g = b as f64 / f64::from(1u32 << 30);
    format!("{g:.1} GiB")
}

/// What the screen shows, read from the record lines.
struct Facts {
    /// The seat's name, its `listening` line's head.
    seat: String,
    /// The address the server listens on.
    addr: String,
    /// Each row's label and value, in the screen's order.
    rows: Vec<(&'static str, String)>,
    /// The plan's routed experts on the card, the tier card and the host,
    /// when its card and host counts are numbers; the tier card's count is
    /// the plan's, else the load's (a plan record that has none), and a
    /// tier count that is no number leaves the split off.
    split: Option<Split>,
}

/// Where the plan put its routed experts, by count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Split {
    card: u64,
    tier: u64,
    host: u64,
}

/// The separators a row's value is written with, per [`Style`].
#[derive(Clone, Copy)]
struct Marks {
    dot: &'static str,
    times: &'static str,
}

/// The screen's facts from the records `lines` holds: the seat's name and
/// address from its `listening` line, `None` without one.
fn facts<S: AsRef<str>>(lines: &[S], m: Marks) -> Option<Facts> {
    let start = Start::new(lines);
    let listen = start.listening()?;
    let (head, addr) = listen.text.split_once(LISTENING)?;
    let seat = head.split(':').next()?.trim().to_owned();
    let addr = addr.trim().to_owned();
    let place = listen.word("place");
    let mut rows = seat_rows(&start, listen, m);
    let plan = start.last("plan ", "place").filter(|l| l.same_place(place));
    let load = start
        .last("load ", "resident_bytes")
        .filter(|l| l.same_place(place));
    let tier = [plan, load]
        .into_iter()
        .flatten()
        .find(|l| l.text("tier_experts").is_some())
        .map(|l| l.num("tier_experts"));
    rows.extend(where_rows(&start, plan, load, tier.flatten(), m));
    let split = plan.and_then(|p| {
        Some(Split {
            card: p.num("card_experts")?,
            tier: tier.unwrap_or(Some(0))?,
            host: p.num("host_experts")?,
        })
    });
    rows.extend(context_row(&start, listen, m));
    if let Some(ram) = start.last("cache ", "ram").and_then(|c| c.num("ram")) {
        rows.push(match ram {
            0 => ("cache", "no prompt cache".to_owned()),
            r => ("cache", format!("{} of prompt cache in RAM", gib(r))),
        });
    }
    if let Some(load) = load {
        let secs = load
            .text
            .rsplit_once(" in ")
            .and_then(|(_, t)| t.split_once(" s"))
            .and_then(|(s, _)| s.parse::<f64>().ok().map(|_| s));
        match secs {
            Some(s) => rows.push(("loaded", format!("in {s} s"))),
            None => start.unread.note(load.text),
        }
    }
    let unread = start.unread.0.borrow();
    if !unread.is_empty() {
        rows.push((
            "unread",
            format!(
                "{}: a value is no number, see the record line",
                unread.join(", ")
            ),
        ));
    }
    Some(Facts {
        seat,
        addr,
        rows,
        split,
    })
}

/// The rows the `listening` line itself holds: the placement (with the
/// `place unset` record's why when it names the same one), the
/// architecture, and a decide seat's model and head.
fn seat_rows<'a, S: AsRef<str>>(
    start: &'a Start<'a, S>,
    listen: Line<'_>,
    m: Marks,
) -> Vec<(&'static str, String)> {
    let mut rows = Vec::new();
    if let Some(place) = listen.word("place") {
        let why = start
            .last("place unset ", "why")
            .filter(|p| p.word("place") == Some(place))
            .and_then(|p| p.text("why"));
        rows.push((
            "place",
            match why {
                Some(w) => format!("{place} ({w})"),
                None => place.to_owned(),
            },
        ));
    }
    if let Some(arch) = listen.word("arch") {
        rows.push(("arch", arch.to_owned()));
    }
    if let Some(model) = listen.word("model") {
        let quant = listen.text("quant").map(|q| format!(" {} {q}", m.dot));
        rows.push(("model", format!("{model}{}", quant.unwrap_or_default())));
    }
    if let Some(head) = listen.word("head") {
        let row = listen.word("row").map(|r| format!(" {} row {r}", m.dot));
        rows.push(("head", format!("{head}{}", row.unwrap_or_default())));
    }
    rows
}

/// Where the model went: the card (the plan's card and experts, the load's
/// resident bytes), every plan card when there are several, the tier card,
/// the host's experts and the NVMe tier — the arena the load was granted
/// from its `nvtier` record, else the plan's, named as planned.
fn where_rows<'a, S: AsRef<str>>(
    start: &'a Start<'a, S>,
    plan: Option<Line<'_>>,
    load: Option<Line<'_>>,
    tier: Option<u64>,
    m: Marks,
) -> Vec<(&'static str, String)> {
    let experts = |l: Line<'_>, key: &str| {
        l.num(key).map(|n| match l.bytes_beside(key) {
            Some(b) => format!("{n} experts ({})", gib(b)),
            None => format!("{n} experts"),
        })
    };
    let mut rows = Vec::new();
    let mut card = Vec::new();
    if let Some(p) = plan {
        card.extend(p.word("card").map(str::to_owned));
        card.extend(experts(p, "card_experts"));
    }
    if let Some(b) = load.and_then(|l| l.num("resident_bytes")) {
        card.push(format!("{} resident", gib(b)));
    }
    if !card.is_empty() {
        rows.push(("card", card.join(&format!(" {} ", m.dot))));
    }
    if let Some(devices) = plan
        .and_then(|p| p.word("devices"))
        .filter(|d| d.contains(','))
    {
        rows.push(("devices", devices.to_owned()));
    }
    if let Some(n) = tier {
        let name = plan
            .and_then(|p| p.word("tier"))
            .map(|t| format!("{t} {} ", m.dot));
        rows.push(("tier", format!("{}{n} experts", name.unwrap_or_default())));
    }
    let Some(p) = plan else {
        return rows;
    };
    if let Some(host) = experts(p, "host_experts") {
        rows.push(("host", host));
    }
    let nvtier = start.last("nvtier ", "arena_granted");
    let granted = nvtier.and_then(|n| Some((n.num("paged")?, n.num("arena_granted")?)));
    match (granted, p.num("paged"), p.num("arena")) {
        (Some((paged, granted)), _, _) => {
            let planned = nvtier
                .and_then(|n| n.num("arena_planned"))
                .filter(|&a| a != granted)
                .map(|a| format!(" ({} planned)", gib(a)));
            rows.push((
                "nvme",
                format!(
                    "{} paged through a {} RAM arena{}",
                    gib(paged),
                    gib(granted),
                    planned.unwrap_or_default()
                ),
            ));
        }
        (None, Some(0), _) => rows.push(("nvme", "nothing paged".to_owned())),
        (None, Some(paged), Some(arena)) => rows.push((
            "nvme",
            format!("{} paged, a {} RAM arena planned", gib(paged), gib(arena)),
        )),
        (None, Some(paged), None) => rows.push(("nvme", format!("{} paged", gib(paged)))),
        (None, None, _) => {}
    }
    rows
}

/// The slots and their context: the `parallel` line's, else the
/// `listening` line's.
fn context_row<'a, S: AsRef<str>>(
    start: &'a Start<'a, S>,
    listen: Line<'_>,
    m: Marks,
) -> Option<(&'static str, String)> {
    let (dot, times) = (m.dot, m.times);
    let parallel = start.last("parallel ", "slots");
    let text = match parallel.and_then(|p| Some((p.num("slots")?, p.word("slot_ctx")?))) {
        Some((1, ctx)) => format!("1 slot {dot} {ctx} positions"),
        Some((n, ctx)) => match parallel.and_then(|p| p.word("total")) {
            Some(total) => format!("{n} slots {times} {ctx} positions ({total} in all)"),
            None => format!("{n} slots {times} {ctx} positions"),
        },
        None => {
            let ctx = listen.word("ctx")?;
            match listen.num("slots") {
                Some(1) => format!("1 slot {dot} {ctx} positions"),
                Some(n) => format!("{n} slots {times} {ctx} positions"),
                None => format!("{ctx} positions"),
            }
        }
    };
    Some(("context", text))
}

/// The columns `c` takes on a terminal: 0 for a combining mark, 2 for a wide
/// (East Asian wide or fullwidth) character, else 1.
fn char_width(c: char) -> usize {
    const ZERO: &[(u32, u32)] = &[
        (0x0300, 0x036F),
        (0x0483, 0x0489),
        (0x0591, 0x05BD),
        (0x0610, 0x061A),
        (0x064B, 0x065F),
        (0x1160, 0x11FF),
        (0x1AB0, 0x1AFF),
        (0x1DC0, 0x1DFF),
        (0x200B, 0x200F),
        (0x20D0, 0x20FF),
        (0x302A, 0x302F),
        (0x3099, 0x309A),
        (0xFE00, 0xFE0F),
        (0xFE20, 0xFE2F),
    ];
    const WIDE: &[(u32, u32)] = &[
        (0x1100, 0x115F),
        (0x2E80, 0x303E),
        (0x3041, 0x33FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xA000, 0xA4CF),
        (0xA960, 0xA97F),
        (0xAC00, 0xD7A3),
        (0xF900, 0xFAFF),
        (0xFE30, 0xFE4F),
        (0xFF00, 0xFF60),
        (0xFFE0, 0xFFE6),
        (0x1F1E6, 0x1F1FF),
        (0x1F300, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F900, 0x1F9FF),
        (0x20000, 0x3FFFD),
    ];
    let u = u32::from(c);
    let within = |t: &[(u32, u32)]| t.iter().any(|&(a, b)| (a..=b).contains(&u));
    if within(ZERO) {
        0
    } else if within(WIDE) {
        2
    } else {
        1
    }
}

/// The columns `s` takes on a terminal ([`char_width`]).
fn width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// The art's height in pixels, two to a terminal row (a half block).
const ART_H: usize = 24;
/// The art's width in columns, one pixel each: the two-panel screen's width.
const ART_W: usize = 98;
/// The columns the text is indented by.
const MARGIN: &str = "  ";
/// The columns of a row's label, its padding included.
const LABEL: usize = 9;
/// The columns between the two panels.
const GAP: usize = 3;
/// The placement panel's columns beside the serving panel.
const LEFT: usize = 44;
/// The experts bar's cells.
const BAR: usize = LEFT - LABEL - 1;

/// A colour of the art and the screen's accents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rgb(u8, u8, u8);

/// The heat ramp, hottest first: the wordmark is warmed down its rows by it,
/// and the bar's card cells along their length.
const HEAT: [Rgb; 4] = [
    Rgb(0xFC, 0xEF, 0xC7),
    Rgb(0xF5, 0xC5, 0x42),
    Rgb(0xF0, 0x8A, 0x3C),
    Rgb(0xE0, 0x53, 0x3A),
];
const CREAM: Rgb = HEAT[0];
const YELLOW: Rgb = HEAT[1];
const ORANGE: Rgb = HEAT[2];
const RED: Rgb = HEAT[3];
const WHITE_HOT: Rgb = Rgb(0xFF, 0xF8, 0xE8);
const SHADOW: Rgb = Rgb(0x5E, 0x2A, 0x18);
const CLAY: Rgb = Rgb(0xA5, 0x5A, 0x36);
const CLAY_LIT: Rgb = Rgb(0xC9, 0x82, 0x56);
const CLAY_DARK: Rgb = Rgb(0x73, 0x38, 0x1F);
const CLAY_BAND: Rgb = Rgb(0x8E, 0x4A, 0x2C);
const TUYERE: Rgb = Rgb(0x8C, 0x7B, 0x6E);
const TUYERE_LOW: Rgb = Rgb(0x6B, 0x5D, 0x52);
const GROUND: Rgb = Rgb(0x4A, 0x2A, 0x1C);
const GROUND_LOW: Rgb = Rgb(0x2E, 0x1A, 0x12);
const HOST: Rgb = Rgb(0x9A, 0x4A, 0x2C);
const INK: Rgb = Rgb(0x0E, 0x11, 0x19);
const KEY: Rgb = Rgb(0x23, 0x28, 0x38);

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const UNDERLINE: &str = "\x1b[4m";
const RESET: &str = "\x1b[0m";

/// Step `i` of `n` along [`HEAT`]: the first step its first colour, the
/// last its last.
fn heat(i: usize, n: usize) -> Rgb {
    let span = n.saturating_sub(1).max(1);
    let at = i.min(span) * (HEAT.len() - 1);
    let seg = (at / span).min(HEAT.len() - 2);
    let rem = at - seg * span;
    let (Rgb(r0, g0, b0), Rgb(r1, g1, b1)) = (HEAT[seg], HEAT[seg + 1]);
    let mix = |a: u8, b: u8| {
        let v = (usize::from(a) * (span - rem) + usize::from(b) * rem) / span;
        u8::try_from(v).unwrap_or(u8::MAX)
    };
    Rgb(mix(r0, r1), mix(g0, g1), mix(b0, b1))
}

/// The wordmark's letters B, L, O, M, E, R and Y, seven pixels wide and
/// twelve tall, `#` lit.
const LETTERS: [[&str; 12]; 7] = [
    [
        "######.", "#######", "##...##", "##...##", "##..##.", "#####..", "######.", "##...##",
        "##...##", "##...##", "#######", "######.",
    ],
    [
        "##.....", "##.....", "##.....", "##.....", "##.....", "##.....", "##.....", "##.....",
        "##.....", "##.....", "#######", "#######",
    ],
    [
        ".#####.", "#######", "##...##", "##...##", "##...##", "##...##", "##...##", "##...##",
        "##...##", "##...##", "#######", ".#####.",
    ],
    [
        "##...##", "###.###", "#######", "##.#.##", "##...##", "##...##", "##...##", "##...##",
        "##...##", "##...##", "##...##", "##...##",
    ],
    [
        "#######", "#######", "##.....", "##.....", "##.....", "######.", "######.", "##.....",
        "##.....", "##.....", "#######", "#######",
    ],
    [
        "######.", "#######", "##...##", "##...##", "##...##", "#######", "######.", "##.##..",
        "##..##.", "##..##.", "##...##", "##...##",
    ],
    [
        "##...##", "##...##", "##...##", ".##.##.", ".#####.", "..###..", "..###..", "..###..",
        "..###..", "..###..", "..###..", "..###..",
    ],
];

/// `bloomery` as indices into [`LETTERS`].
const WORD: [usize; 8] = [0, 1, 2, 2, 3, 4, 5, 6];

/// The art in pixels: the wordmark warmed down its rows over its shadow,
/// and beside it the furnace (a clay stack widening to its base, a flame at
/// its mouth, the bloom glowing in its arch, the tuyere at its side).
/// Without colour only the shapes stay: no shadow, no sparks, and the mouth
/// and the arch left open.
fn art(color: bool) -> Vec<Vec<Option<Rgb>>> {
    let mut px = vec![vec![None; ART_W]; ART_H];
    let (top, left) = (9, 2);
    let layers: &[bool] = if color { &[true, false] } else { &[false] };
    for &shadow in layers {
        for (i, &letter) in WORD.iter().enumerate() {
            for (r, row) in LETTERS[letter].iter().enumerate() {
                for (k, _) in row.bytes().enumerate().filter(|&(_, b)| b == b'#') {
                    let (y, x) = (top + r, left + 9 * i + k);
                    if shadow {
                        px[y + 1][x + 1] = Some(SHADOW);
                    } else {
                        px[y][x] = Some(heat(r, 12));
                    }
                }
            }
        }
    }

    let glow = |c: Rgb| color.then_some(c);
    let ox = ART_W - 20;
    for (y, row) in px.iter_mut().enumerate() {
        for x in 0..20usize {
            // The distance from the stack's middle, in quarter pixels.
            let q = (4 * x + 2).abs_diff(40);
            row[ox + x] = match y {
                0..=4 => {
                    let flame = [2, 2, 4, 6, 8][y];
                    (q <= flame).then_some(if y >= 3 && q <= 2 {
                        WHITE_HOT
                    } else if y >= 2 && q <= 4 {
                        YELLOW
                    } else if y <= 1 {
                        RED
                    } else {
                        ORANGE
                    })
                }
                5..=21 => {
                    let half = 12 + (y - 5);
                    let arch = match y {
                        15 => Some(2),
                        16 => Some(6),
                        17.. => Some(10),
                        _ => None,
                    };
                    if q > half {
                        ((y == 18 || y == 19) && x < 10).then_some(if y == 18 {
                            TUYERE
                        } else {
                            TUYERE_LOW
                        })
                    } else if y == 5 && q <= 6 {
                        glow(ORANGE)
                    } else if arch.is_some_and(|a| q <= a) {
                        glow(if y >= 18 && q <= 2 {
                            WHITE_HOT
                        } else if q <= 6 {
                            YELLOW
                        } else {
                            ORANGE
                        })
                    } else if q > half - 4 {
                        Some(if 4 * x + 2 < 40 { CLAY_LIT } else { CLAY_DARK })
                    } else if (y - 5) % 4 == 3 {
                        Some(CLAY_BAND)
                    } else {
                        Some(CLAY)
                    }
                }
                22 => Some(GROUND),
                _ => (2..18).contains(&x).then_some(GROUND_LOW),
            };
        }
    }
    if color {
        for (y, x, c) in [
            (0, 5, RED),
            (0, 15, YELLOW),
            (0, 19, ORANGE),
            (1, 4, ORANGE),
            (1, 12, WHITE_HOT),
            (1, 14, YELLOW),
            (2, 2, YELLOW),
            (2, 14, RED),
            (2, 18, YELLOW),
            (3, 0, ORANGE),
            (3, 17, RED),
            (4, 5, YELLOW),
            (4, 15, ORANGE),
            (5, 1, RED),
            (6, 18, ORANGE),
            (7, 16, YELLOW),
            (8, 19, RED),
            (9, 2, ORANGE),
        ] {
            px[y][ox + x] = Some(c);
        }
    }
    px
}

/// The terminal rows of the pixel rows `px`, two to a row. A cell lit above
/// and below draws `█` (with colour, `▀` over a background when the two
/// differ), above only `▀`, below only `▄`; in ASCII `#`, `"` and `.`. A
/// row's trailing blank cells are left off.
fn blocks(px: &[Vec<Option<Rgb>>], style: Style) -> Vec<String> {
    let (full, upper, lower) = if style.utf8 {
        ('█', '▀', '▄')
    } else {
        ('#', '"', '.')
    };
    px.chunks(2)
        .map(|pair| {
            let (top, low) = (&pair[0], pair.get(1));
            let mut line = String::new();
            let mut pen = (None, None);
            for (x, &a) in top.iter().enumerate() {
                let b = low.and_then(|l| l.get(x)).copied().flatten();
                let (glyph, ink) = match (a, b) {
                    (None, None) => (' ', (None, None)),
                    (Some(a), None) => (upper, (Some(a), None)),
                    (None, Some(b)) => (lower, (Some(b), None)),
                    (Some(a), Some(b)) if a == b || !style.color => (full, (Some(a), None)),
                    (Some(a), Some(b)) => (upper, (Some(a), Some(b))),
                };
                if style.color && ink != pen {
                    if pen != (None, None) {
                        line.push_str(RESET);
                    }
                    if let Some(fg) = ink.0 {
                        line.push_str(&style.fg(fg));
                    }
                    if let Some(bg) = ink.1 {
                        line.push_str(&style.bg(bg));
                    }
                    pen = ink;
                }
                line.push(glyph);
            }
            if pen != (None, None) {
                line.push_str(RESET);
            }
            line.truncate(line.trim_end_matches(' ').len());
            line
        })
        .collect()
}

/// A piece of the screen as a terminal shows it (`shown`, which sets its
/// width) and as it is written (`painted`, with its escapes).
#[derive(Clone, Default)]
struct Text {
    shown: String,
    painted: String,
}

impl Text {
    /// Appends `s`, painted with `code` (no escape when it is empty).
    fn push(&mut self, s: &str, code: &str) {
        self.shown.push_str(s);
        if code.is_empty() {
            self.painted.push_str(s);
        } else {
            self.painted.push_str(code);
            self.painted.push_str(s);
            self.painted.push_str(RESET);
        }
    }

    /// Pads with spaces to column `to`.
    fn pad(&mut self, to: usize) {
        let n = to.saturating_sub(width(&self.shown));
        self.push(&" ".repeat(n), "");
    }

    fn append(&mut self, other: &Text) {
        self.shown.push_str(&other.shown);
        self.painted.push_str(&other.painted);
    }
}

/// `text` broken into lines of at most `w` columns: at a `dot` separator
/// when the part after it fits a line, else at a space, and a word longer
/// than `w` where the columns run out.
fn wrap(text: &str, w: usize, dot: &str) -> Vec<String> {
    let used = |lines: &[String]| lines.last().map_or(0, |l| width(l));
    let sep = format!(" {dot} ");
    let parts: Vec<&str> = text.split(sep.as_str()).collect();
    let mut lines = vec![String::new()];
    for (i, part) in parts.iter().enumerate() {
        let part = if i + 1 < parts.len() {
            format!("{part} {dot}")
        } else {
            (*part).to_owned()
        };
        let at = used(&lines);
        if at > 0 && at + 1 + width(&part) > w && width(&part) <= w {
            lines.push(String::new());
        }
        for word in part.split(' ').filter(|w| !w.is_empty()) {
            let at = used(&lines);
            if at > 0 && at + 1 + width(word) > w {
                lines.push(String::new());
            } else if let Some(l) = lines.last_mut().filter(|l| !l.is_empty()) {
                l.push(' ');
            }
            for c in word.chars() {
                let at = used(&lines);
                if at > 0 && at + char_width(c) > w {
                    lines.push(String::new());
                }
                if let Some(l) = lines.last_mut() {
                    l.push(c);
                }
            }
        }
    }
    lines
}

/// The lines of the row `label`: its value wrapped to `w` columns less the
/// label's ([`wrap`]), the label dimmed before the first line and the rest
/// indented under it.
fn row_lines(label: &str, value: &str, w: usize, style: Style, dot: &str) -> Vec<Text> {
    if w <= LABEL {
        return [(label, style.code(DIM)), (value, "")]
            .into_iter()
            .flat_map(|(s, code)| {
                wrap(s, w, dot).into_iter().map(move |line| {
                    let mut t = Text::default();
                    t.push(&line, code);
                    t
                })
            })
            .collect();
    }
    wrap(value, w - LABEL, dot)
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let mut t = Text::default();
            if i == 0 {
                t.push(&format!("{label:<w$}", w = LABEL), style.code(DIM));
            } else {
                t.push(&" ".repeat(LABEL), "");
            }
            t.push(&v, "");
            t
        })
        .collect()
}

/// A section's heading: its name, then a rule to `w` columns.
fn heading(name: &str, w: usize, style: Style) -> Text {
    let mut t = Text::default();
    t.push(name, &style.fg(ORANGE));
    let rule = if style.utf8 { "─" } else { "-" };
    let rest = w.saturating_sub(width(name) + 1);
    t.push(&format!(" {}", rule.repeat(rest)), style.code(DIM));
    t
}

/// The cells of the `BAR`-cell bar the card, the tier card and the host take
/// of `s`: the nearest share each, at least one for a part that holds
/// experts, `None` when no part holds any or the counts overflow a `u64`.
fn bar_cells(s: Split) -> Option<[usize; 3]> {
    let parts = [s.card, s.tier, s.host];
    let total = parts.iter().try_fold(0u64, |t, &p| t.checked_add(p))?;
    if total == 0 {
        return None;
    }
    let (total, n128) = (u128::from(total), BAR as u128);
    let mut cells = parts.map(|p| match p {
        0 => 0,
        p => usize::try_from((u128::from(p) * n128 + total / 2) / total)
            .unwrap_or(BAR)
            .max(1),
    });
    while cells.iter().sum::<usize>() != BAR {
        let widest = (0..3).max_by_key(|&i| cells[i]).unwrap_or(0);
        if cells.iter().sum::<usize>() > BAR {
            cells[widest] -= 1;
        } else {
            cells[widest] += 1;
        }
    }
    Some(cells)
}

/// The experts bar and its legend, each a row under the `experts` label:
/// the card's cells warmed along their length, then the tier card's and
/// the host's.
fn bar(s: Split, style: Style) -> Option<[Text; 2]> {
    let [card, tier, host] = bar_cells(s)?;
    let glyphs = if style.utf8 {
        ["█", "▓", "░"]
    } else {
        ["#", "=", "."]
    };
    let mut cells = Text::default();
    cells.push(&format!("{:<w$}", "experts", w = LABEL), style.code(DIM));
    for i in 0..card {
        cells.push(glyphs[0], &style.fg(heat(i, card)));
    }
    cells.push(&glyphs[1].repeat(tier), &style.fg(RED));
    cells.push(&glyphs[2].repeat(host), &style.fg(HOST));

    let mut legend = Text::default();
    legend.push(&" ".repeat(LABEL), "");
    for (glyph, name, color, held) in [
        (glyphs[0], "card", YELLOW, s.card),
        (glyphs[1], "tier", RED, s.tier),
        (glyphs[2], "host", HOST, s.host),
    ] {
        if held > 0 {
            if width(&legend.shown) > LABEL {
                legend.push("   ", "");
            }
            legend.push(glyph, &style.fg(color));
            legend.push(&format!(" {name}"), "");
        }
    }
    Some([cells, legend])
}

/// The rows the placement panel shows, in the records' order.
const PLACED: [&str; 4] = ["card", "tier", "host", "nvme"];
/// The rows the serving panel shows.
const SERVED: [&str; 6] = ["place", "arch", "model", "head", "context", "cache"];

/// Hard wraps painted lines without counting their SGR escapes as columns.
/// Record rows use word wrapping first; this also bounds art, links and commands.
fn wrap_painted(line: &str, columns: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    let mut escape = false;
    for c in line.chars() {
        if c == '\x1b' {
            escape = true;
        }
        if escape {
            out.push(c);
            if c == 'm' {
                escape = false;
            }
            continue;
        }
        let n = char_width(c);
        if used + n > columns {
            out.push('\n');
            used = 0;
        }
        out.push(c);
        used += n;
    }
    out
}

/// The screen of the record lines `lines` in `style`, or `None` when they
/// hold no `listening` line.
#[must_use]
fn render<S: AsRef<str>>(lines: &[S], style: Style, columns: usize) -> Option<String> {
    let columns = columns.min(ART_W);
    let margin = if columns > LABEL + MARGIN.len() {
        MARGIN
    } else {
        ""
    };
    let inner = columns - margin.len();
    let stacked = columns < ART_W;
    let marks = if style.utf8 {
        Marks {
            dot: "·",
            times: "×",
        }
    } else {
        Marks {
            dot: "-",
            times: "x",
        }
    };
    let Facts {
        seat,
        addr,
        rows,
        split,
    } = facts(lines, marks)?;
    // A record value is shown without its control characters, and in ASCII
    // with `?` for every character past it.
    let plain = |text: &str| -> String {
        text.chars()
            .filter(|c| !c.is_control())
            .map(|c| if style.utf8 || c.is_ascii() { c } else { '?' })
            .collect()
    };
    let pick = |keep: &dyn Fn(&str) -> bool| -> Vec<(&str, String)> {
        rows.iter()
            .filter(|(l, _)| keep(l))
            .map(|(l, v)| (*l, plain(v)))
            .collect()
    };
    let dim = style.code(DIM);
    let mut out = vec![String::new()];
    if stacked {
        out.push(format!(
            "{margin}{}bloomery{}",
            style.fg(ORANGE),
            style.code(RESET)
        ));
    } else {
        out.extend(blocks(&art(style.color), style));
    }
    out.push(String::new());

    let seat = plain(&seat);
    let mut status = Text::default();
    let badge = format!("{}{}{}", style.bg(ORANGE), style.fg(INK), style.code(BOLD));
    status.push(" READY ", &badge);
    status.push("  ", "");
    status.push(&seat, style.code(BOLD));
    if let Some((_, loaded)) = pick(&|l| l == "loaded").first() {
        let right = format!("loaded {loaded}");
        if width(&status.shown) + 2 + width(&right) > inner {
            out.push(format!("{margin}{}", status.painted));
            status = Text::default();
        } else {
            status.pad(inner - width(&right));
        }
        status.push(&right, dim);
    }
    out.push(format!("{margin}{}", status.painted));
    let url = format!("http://{}", plain(&addr));
    let mut at = Text::default();
    at.push(&" ".repeat(LABEL), "");
    at.push(if style.utf8 { "→ " } else { "-> " }, dim);
    let link = format!(
        "{}{}{}",
        style.code(BOLD),
        style.code(UNDERLINE),
        style.fg(YELLOW)
    );
    at.push(&url, &link);
    out.push(format!("{margin}{}", at.painted));

    let placed = pick(&|l| PLACED.contains(&l));
    let served = pick(&|l| SERVED.contains(&l));
    let bar = split.and_then(|s| bar(s, style));
    let has_left = !placed.is_empty() || bar.is_some();
    let (lw, rw) = match (has_left, served.is_empty()) {
        (true, false) if !stacked => (LEFT, inner - LEFT - GAP),
        (true, false) => (inner, inner),
        (true, true) => (inner, 0),
        (false, _) => (0, inner),
    };
    let mut left = Vec::new();
    if has_left {
        left.push(heading("placement", lw, style));
        left.extend(bar.into_iter().flatten());
        for (label, value) in &placed {
            left.extend(row_lines(label, value, lw, style, marks.dot));
        }
    }
    let mut right = Vec::new();
    if !served.is_empty() {
        right.push(heading("serving", rw, style));
        for (label, value) in &served {
            right.extend(row_lines(label, value, rw, style, marks.dot));
        }
    }
    if !left.is_empty() || !right.is_empty() {
        out.push(String::new());
    }
    if stacked {
        if !left.is_empty() && !right.is_empty() {
            left.push(Text::default());
        }
        left.append(&mut right);
    }
    for i in 0..left.len().max(right.len()) {
        let mut line = left.get(i).cloned().unwrap_or_default();
        if let Some(r) = right.get(i) {
            if has_left {
                line.pad(lw + GAP);
            }
            line.append(r);
        }
        out.push(format!("{margin}{}", line.painted));
    }
    // Rows neither panel names (the plan's card list, the unread records)
    // run the full width below them.
    let wide = pick(&|l| l != "loaded" && !PLACED.contains(&l) && !SERVED.contains(&l));
    if !wide.is_empty() {
        out.push(String::new());
        for (label, value) in &wide {
            for t in row_lines(label, value, inner, style, marks.dot) {
                out.push(format!("{margin}{}", t.painted));
            }
        }
    }

    out.push(String::new());
    out.push(format!("{margin}{}", heading("try", inner, style).painted));
    let yellow = style.fg(YELLOW);
    let mut lines = Vec::new();
    if seat.ends_with("-decide") {
        let mut t = Text::default();
        t.push("POST ", dim);
        t.push(&format!("{url}/v1/systemone"), &yellow);
        lines.push(t);
    } else {
        let mut t = Text::default();
        t.push("$ ", dim);
        t.push("curl ", "");
        t.push(&format!("{url}/v1/chat/completions"), &yellow);
        t.push(" \\", dim);
        lines.push(t);
        let mut t = Text::default();
        t.push("    -H 'Content-Type: application/json' \\", dim);
        lines.push(t);
        let mut t = Text::default();
        t.push("    -d ", dim);
        t.push(r#"'{"messages":[{"role":"user","content":"Hello!"}]}'"#, "");
        lines.push(t);
    }
    out.extend(lines.iter().map(|t| format!("{margin}{}", t.painted)));

    out.push(String::new());
    let key = format!("{}{}", style.bg(KEY), style.fg(CREAM));
    let mut keys = Text::default();
    keys.push(" Ctrl+C ", &key);
    keys.push(" stops the server", dim);
    let mut hide = Text::default();
    hide.push(" BLOOMERY_SCREEN=off ", &key);
    hide.push(" hides this screen", dim);
    if width(&keys.shown) + 4 + width(&hide.shown) > inner {
        out.push(format!("{margin}{}", keys.painted));
        keys = Text::default();
    } else {
        keys.push("    ", "");
    }
    keys.append(&hide);
    out.push(format!("{margin}{}", keys.painted));
    out.push(String::new());
    Some(
        out.iter()
            .map(|line| wrap_painted(line, columns))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ART_H, GAP, LABEL, LEFT, MARGIN, Marks, Rgb, Split, Style, TermVars, bar_cells, blocks,
        facts, render, width,
    };

    /// The record lines of a V4.1 seat's start, as its log carries them.
    const START: &str = include_str!("../tests/fixtures/screen-start.log");
    /// The record lines of a Qwen3.8 seat's paged start, its arena granted
    /// smaller than planned.
    const START38: &str = include_str!("../tests/fixtures/screen-start-qwen38.log");

    const ASCII: Style = Style {
        color: false,
        truecolor: false,
        utf8: false,
    };
    const UTF8: Style = Style {
        color: false,
        truecolor: false,
        utf8: true,
    };
    const CUBE: Style = Style {
        color: true,
        truecolor: false,
        utf8: true,
    };
    const TRUE: Style = Style {
        color: true,
        truecolor: true,
        utf8: true,
    };

    fn lines(text: &str) -> Vec<&str> {
        text.lines().collect()
    }

    /// How the row `label` starts on the screen: its label padded to the
    /// label's columns, after a space.
    fn row(label: &str) -> String {
        format!(" {label:<w$}", w = LABEL)
    }

    fn golden_path(name: &str) -> String {
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn read_golden(name: &str) -> String {
        let path = golden_path(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// `got` is the golden text `tests/fixtures/<name>`. The goldens are
    /// written by this run itself, from the fixed starts above:
    /// `SCREEN_BLESS=1 cargo test -p bloomery-serve --lib screen::` rewrites
    /// them, and `git diff` shows what moved; any other value is refused.
    fn golden(name: &str, got: &str) {
        match std::env::var("SCREEN_BLESS").as_deref() {
            Err(_) => assert_eq!(got, read_golden(name), "the golden {name}"),
            Ok("1") => {
                let path = golden_path(name);
                std::fs::write(&path, got).unwrap_or_else(|e| panic!("{path}: {e}"));
            }
            Ok(other) => panic!("SCREEN_BLESS={other}: only 1 rewrites the goldens"),
        }
    }

    /// The screen of a fixed start, in UTF-8 and in ASCII, each the golden
    /// text; with colour, in 24 bits and in the 256-colour cube, the same
    /// text below the art once the escapes are taken out (the art keeps its
    /// shadow and glow only in colour).
    #[test]
    fn renders_the_golden_screens() {
        let plain =
            render(&lines(START), UTF8, super::ART_W).expect("the start has a listening line");
        golden("screen-utf8.txt", &plain);
        let got =
            render(&lines(START), ASCII, super::ART_W).expect("the start has a listening line");
        golden("screen-ascii.txt", &got);
        assert!(got.is_ascii(), "the ASCII screen holds a byte past ASCII");

        // PIN(2026-10-22): colour adds the art's shadow and glow; the text below must match.
        let below_art = |s: &str| {
            s.split('\n')
                .skip(1 + ART_H / 2)
                .collect::<Vec<_>>()
                .join("\n")
        };
        for (style, escape) in [(TRUE, "\x1b[38;2;"), (CUBE, "\x1b[38;5;")] {
            let colored = render(&lines(START), style, super::ART_W).unwrap();
            assert!(colored.contains(escape), "no {escape:?} in {style:?}");
            assert_eq!(below_art(&strip_escapes(&colored)), below_art(&plain));
        }
        assert!(
            !render(&lines(START), CUBE, super::ART_W)
                .unwrap()
                .contains("\x1b[38;2;")
        );
        assert_eq!(super::cube(Rgb(255, 0, 0)), 196);
    }

    /// A Qwen3.8 start that pages its experts: the NVMe row names the arena
    /// the `nvtier` record says the load was granted, beside the one planned,
    /// the golden text.
    #[test]
    fn renders_the_granted_arena() {
        let got =
            render(&lines(START38), UTF8, super::ART_W).expect("the start has a listening line");
        golden("screen-qwen38.txt", &got);
        let no_nvtier: Vec<&str> = lines(START38)
            .into_iter()
            .filter(|l| !l.starts_with("nvtier "))
            .collect();
        let got = render(&no_nvtier, UTF8, super::ART_W).unwrap();
        assert!(
            !got.contains("nvme"),
            "an arena from no nvtier record:\n{got}"
        );
    }

    /// The fixed start with its `plan`, `load` and `place unset` lines taken
    /// out: the screen leaves off the rows those records hold (card, host,
    /// nvme, the experts bar, the load time, the place's why) and keeps the
    /// rest, the golden text. The same lines in reverse order draw the same
    /// screen, and so does the whole start reversed: each head has one line
    /// here, and a row is read from the last line of its head, never from
    /// where it sits among the others.
    #[test]
    fn missing_and_reordered_lines_keep_to_the_records() {
        let missing: Vec<&str> = lines(START)
            .into_iter()
            .filter(|l| {
                !["plan ", "load ", "place unset "]
                    .iter()
                    .any(|h| l.starts_with(h))
            })
            .collect();
        golden(
            "screen-missing.txt",
            &render(&missing, ASCII, super::ART_W).unwrap(),
        );
        let reversed: Vec<&str> = missing.iter().rev().copied().collect();
        assert_eq!(
            render(&reversed, ASCII, super::ART_W).unwrap(),
            read_golden("screen-missing.txt")
        );

        let reversed: Vec<&str> = lines(START).into_iter().rev().collect();
        assert_eq!(
            render(&reversed, UTF8, super::ART_W).unwrap(),
            read_golden("screen-utf8.txt")
        );
    }

    /// Records that name another placement than the `listening` line are not
    /// joined to it: a `place unset` line gives no why, a `plan` and a `load`
    /// line no card, host, experts or load time.
    #[test]
    fn records_of_another_place_are_left_off() {
        let start: Vec<String> = lines(START)
            .into_iter()
            .map(|l| {
                if l.starts_with("place unset ") {
                    "place unset place=bp why=two cards, tier at or past the break-even".to_owned()
                } else if l.starts_with("plan ") || l.starts_with("load ") {
                    l.replace(" place=a ", " place=bp ")
                } else {
                    l.to_owned()
                }
            })
            .collect();
        let got = render(&start, ASCII, super::ART_W).unwrap();
        assert!(got.contains(&format!("{}a\n", row("place"))), "{got}");
        for gone in [
            "break-even".to_owned(),
            row("card"),
            row("host"),
            row("experts"),
            "loaded in".to_owned(),
        ] {
            assert!(
                !got.contains(&gone),
                "{gone} from a record of place bp:\n{got}"
            );
        }
    }

    /// A record the start did not print leaves its row off: a listening line
    /// alone shows the address and what that line says, nothing more.
    #[test]
    fn a_missing_record_leaves_its_row_off() {
        let only = [
            "bloomery-serve-qwen38: place=a ctx=8192 slots=2 slot_ctx=8192 listening on http://0.0.0.0:9000",
        ];
        let got = render(&only, ASCII, super::ART_W).unwrap();
        assert!(got.contains("-> http://0.0.0.0:9000\n"), "{got}");
        for label in ["place", "context"] {
            assert!(got.contains(&row(label)), "no {label} row:\n{got}");
        }
        for label in ["card", "host", "nvme", "cache", "experts"] {
            assert!(
                !got.contains(&row(label)),
                "a {label} row from no record:\n{got}"
            );
        }
        assert!(
            !got.contains("loaded in"),
            "a load time from no record:\n{got}"
        );
        assert!(got.contains("2 slots x 8192 positions"), "{got}");
    }

    /// `text` as lines, the first line that starts with `head` with `from`
    /// replaced by `to`.
    fn mutated(text: &str, head: &str, from: &str, to: &str) -> Vec<String> {
        let mut done = false;
        lines(text)
            .into_iter()
            .map(|l| {
                if !done && l.starts_with(head) {
                    assert!(l.contains(from), "{from} is not in {l}");
                    done = true;
                    l.replacen(from, to, 1)
                } else {
                    l.to_owned()
                }
            })
            .collect()
    }

    /// A value the screen shows as a number that is none (or, beside a
    /// count, a byte figure that is none) leaves what it would have shown
    /// off, and the `unread` row names the record: a record the start did
    /// print and the screen could not read is told, not left off as if the
    /// start had never printed it.
    #[test]
    fn a_value_that_is_no_number_is_named_not_filled_in() {
        for (start, head, from, to, gone, record) in [
            (
                START,
                "load ",
                " in 41.2 s ",
                " in soon s ",
                "loaded in".to_owned(),
                "load",
            ),
            (
                START,
                "load ",
                "resident_bytes=48318382080",
                "resident_bytes=lots",
                " resident".to_owned(),
                "load",
            ),
            (
                START,
                "plan ",
                "card_experts=2668 (",
                "card_experts=many (",
                "2668".to_owned(),
                "plan",
            ),
            (
                START,
                "plan ",
                "(44750684160 B)",
                "(lots B)",
                "(41.7 GiB)".to_owned(),
                "plan",
            ),
            (
                START,
                "parallel ",
                "slots=2 ",
                "slots=two ",
                "slots".to_owned(),
                "parallel",
            ),
            (
                START,
                "cache ",
                "ram=8589934592",
                "ram=lots",
                row("cache"),
                "cache",
            ),
            (
                START38,
                "nvtier ",
                "paged=107374182400",
                "paged=lots",
                row("nvme"),
                "nvtier",
            ),
            (
                START,
                "cache ",
                "ram=8589934592",
                "ram=",
                row("cache"),
                "cache",
            ),
            (
                START,
                "parallel ",
                "slots=2 ",
                "slots= ",
                "slots".to_owned(),
                "parallel",
            ),
        ] {
            let start = mutated(start, head, from, to);
            let got = render(&start, ASCII, super::ART_W).unwrap();
            assert!(
                !got.contains(&gone),
                "{gone} from a {record} value that is no number:\n{got}"
            );
            assert!(
                got.lines()
                    .any(|l| l.contains(&row("unread")) && l.contains(record)),
                "no unread row naming {record}:\n{got}"
            );
        }
        for start in [START, START38] {
            let got = render(&lines(start), ASCII, super::ART_W).unwrap();
            assert!(
                !got.contains(&row("unread")),
                "an unread row from a clean start:\n{got}"
            );
        }
    }

    /// Several plan cards: the `devices` row is the plan's whole list.
    #[test]
    fn several_cards_show_the_whole_list() {
        let cards = [
            "stage:A6000:cuda1:50952404992",
            "tier0:3090:cuda0:25350373376",
            "tier1:3090:cuda2:25350373376",
            "tier2:3090:cuda3:25350373376",
        ];
        for tiers in 0..cards.len() {
            let devices = format!("[{}]", cards[..=tiers].join(","));
            let start = [
                format!(
                    "plan place=bp card=A6000 ctx_max=16384 card_experts=2552 (38801506304 B) \
             host_experts=8045 (123885584384 B) host_shadow=0 B n_l=65..66 on 39 layers \
             card_budget=none paged=0 arena=0 arena_slots=0 card_free=50668240896 \
             devices={devices} cuda_order=unset(FASTEST_FIRST)"
                ),
                "bloomery-serve-glm: place=bp ctx=16384 listening on http://127.0.0.1:8080"
                    .to_owned(),
            ];
            let facts = facts(
                &start,
                Marks {
                    dot: ".",
                    times: "x",
                },
            )
            .unwrap();
            assert_eq!(
                facts.rows.iter().find(|r| r.0 == "devices").map(|r| &r.1),
                (tiers > 0).then_some(&devices)
            );
            let got = render(&start, ASCII, super::ART_W).unwrap();
            assert_eq!(got.contains(&row("devices")), tiers > 0, "{got}");
            if tiers == 1 {
                assert!(
                    got.contains(&format!("{}{devices}\n", row("devices"))),
                    "{got}"
                );
            }
        }
    }

    /// A value with Hangul and a combining mark pads by the columns a
    /// terminal draws, so the serving panel's rows start in one column beside
    /// a placement row that holds wide characters; a control character in a
    /// value is not drawn.
    #[test]
    fn wide_and_control_characters_keep_the_panels_aligned() {
        let start = [
            "plan place=a card=삽살개카드 card_experts=10 host_experts=30",
            "bloomery-serve-decide: place=a model=삽살이-Q4_K_M.gguf quant=q4_K×1988 q6_K×97 \
             head=e\u{301}tiquette.safetensors\x1b[31m row=clef listening on http://127.0.0.1:8080",
        ];
        let got = render(&start, UTF8, super::ART_W).unwrap();
        assert!(
            !got.contains('\x1b'),
            "an escape from a record value:\n{got}"
        );
        assert!(got.contains("삽살개카드"), "{got}");
        let column = width(MARGIN) + LEFT + GAP;
        let mut met = 0;
        for line in got.lines() {
            for label in ["place", "model", "head"] {
                if let Some(at) = line.find(&format!("{label:<w$}", w = LABEL)) {
                    assert_eq!(width(&line[..at]), column, "{label} in {line:?}\n{got}");
                    met += 1;
                }
            }
        }
        assert_eq!(met, 3, "{got}");
        assert_eq!(width("삽살이"), 6);
        assert_eq!(width("か\u{3099}"), width("が"));
        assert_eq!(width("\u{1F680}"), 2);
        assert_eq!(width("e\u{301}"), 1);
    }

    /// Two pixel rows make one terminal row: lit above and below `█` (with
    /// two colours, `▀` over the lower one's background), above only `▀`,
    /// below only `▄`, in ASCII `#`, `"` and `.`; an odd last pixel row
    /// draws as an upper half, and trailing blank cells are left off.
    #[test]
    fn half_blocks_pair_the_pixel_rows() {
        let (a, b) = (Some(Rgb(1, 2, 3)), Some(Rgb(4, 5, 6)));
        let px = vec![
            vec![a, a, a, None, None],
            vec![a, b, None, b, None],
            vec![a, None, None, None, None],
        ];
        assert_eq!(blocks(&px, UTF8), ["██▀▄", "▀"]);
        assert_eq!(blocks(&px, ASCII), ["##\".", "\""]);
        let colored = blocks(&px, TRUE);
        assert_eq!(
            colored[0],
            "\x1b[38;2;1;2;3m█\x1b[0m\x1b[38;2;1;2;3m\x1b[48;2;4;5;6m▀\x1b[0m\
             \x1b[38;2;1;2;3m▀\x1b[0m\x1b[38;2;4;5;6m▄\x1b[0m"
        );
        assert_eq!(strip_escapes(&colored[0]), "█▀▀▄");
    }

    /// The experts bar's cells: the nearest share each, a part that holds
    /// experts at least one cell however small, every cell taken, and no bar
    /// when no part holds any.
    #[test]
    fn the_bar_shares_its_cells() {
        let split = |card, tier, host| Split { card, tier, host };
        assert_eq!(bar_cells(split(2668, 0, 12692)), Some([6, 0, 28]));
        assert_eq!(bar_cells(split(1, 0, 100_000)), Some([1, 0, 33]));
        assert_eq!(bar_cells(split(10, 10, 10)), Some([11, 11, 12]));
        assert_eq!(bar_cells(split(0, 0, 0)), None);
        assert_eq!(bar_cells(split(u64::MAX, 0, 2)), None);
    }

    /// The tier card's experts come from the plan, else from the load when
    /// the plan record has no count (a placement whose plan names only its
    /// first card), so the bar keeps the tier's cells; a tier count that is
    /// no number leaves the bar off and the `unread` row names the record.
    #[test]
    fn the_bar_takes_the_tier_count_from_the_load() {
        let start = |tier: &str| {
            [
                "plan place=bp card=A6000 card_experts=10 host_experts=10".to_owned(),
                format!("load resident_bytes=1 place=bp tier_experts={tier} in 1.0 s"),
                "bloomery-serve-glm: place=bp listening on http://127.0.0.1:8080".to_owned(),
            ]
        };
        let got = render(&start("10"), UTF8, super::ART_W).unwrap();
        assert!(got.contains("▓ tier"), "{got}");
        assert!(got.contains(&"▓".repeat(11)), "{got}");
        assert!(
            got.contains(&format!("{}10 experts\n", row("tier"))),
            "{got}"
        );
        let got = render(&start("lots"), UTF8, super::ART_W).unwrap();
        assert!(!got.contains(&row("experts")), "{got}");
        assert!(!got.contains(&row("tier")), "{got}");
        assert!(got.contains("load: a value is no number"), "{got}");
    }

    #[test]
    fn tier_details_and_the_bar_keep_the_record_precedence() {
        for (plan, load, detail, count, unread) in [
            (
                Some("tier=3090 tier_experts=2"),
                Some("tier_experts=1"),
                Some("3090 · 2 experts"),
                Some(2),
                None,
            ),
            (
                Some("tier=3090"),
                Some("tier_experts=2"),
                Some("3090 · 2 experts"),
                Some(2),
                None,
            ),
            (
                Some("tier=3090 tier_experts=1"),
                None,
                Some("3090 · 1 experts"),
                Some(1),
                None,
            ),
            (
                Some("tier_experts=0"),
                Some("tier_experts=2"),
                Some("0 experts"),
                Some(0),
                None,
            ),
            (
                Some("tier_experts=1"),
                None,
                Some("1 experts"),
                Some(1),
                None,
            ),
            (
                Some("tier_experts=2"),
                None,
                Some("2 experts"),
                Some(2),
                None,
            ),
            (
                Some(""),
                Some("tier_experts=0"),
                Some("0 experts"),
                Some(0),
                None,
            ),
            (
                Some(""),
                Some("tier_experts=1"),
                Some("1 experts"),
                Some(1),
                None,
            ),
            (Some("tier=3090"), Some(""), None, None, None),
            (Some(""), None, None, None, None),
            (
                None,
                Some("tier_experts=2"),
                Some("2 experts"),
                Some(2),
                None,
            ),
            (
                Some("tier=3090 tier_experts=lots"),
                Some("tier_experts=2"),
                None,
                None,
                Some("plan"),
            ),
            (
                Some(""),
                Some("tier_experts=lots"),
                None,
                None,
                Some("load"),
            ),
            (
                Some("tier_experts=2"),
                Some("tier_experts=lots"),
                Some("2 experts"),
                Some(2),
                None,
            ),
            (None, Some("tier_experts=lots"), None, None, Some("load")),
        ] {
            let mut start = Vec::new();
            if let Some(fields) = plan {
                start.push(format!(
                    "plan place=bp card=A6000 card_experts=10 host_experts=10 {fields}"
                ));
            }
            if let Some(fields) = load {
                start.push(format!("load resident_bytes=1 place=bp {fields} in 1.0 s"));
            }
            start
                .push("bloomery-serve-glm: place=bp listening on http://127.0.0.1:8080".to_owned());
            let facts = facts(
                &start,
                Marks {
                    dot: "·",
                    times: "×",
                },
            )
            .unwrap();
            assert_eq!(
                facts
                    .rows
                    .iter()
                    .find(|r| r.0 == "tier")
                    .map(|r| r.1.as_str()),
                detail,
                "{start:?}"
            );
            let split = if plan.is_some() && unread.is_none() {
                Some(count.unwrap_or(0))
            } else {
                None
            };
            assert_eq!(facts.split.map(|s| s.tier), split, "{start:?}");
            let got = render(&start, UTF8, super::ART_W).unwrap();
            assert_eq!(got.contains(&row("tier")), detail.is_some(), "{got}");
            if let Some(detail) = detail {
                assert!(got.contains(detail), "{got}");
            }
            assert_eq!(
                got.contains("▓ tier"),
                split.is_some_and(|n| n > 0),
                "{got}"
            );
            assert_eq!(got.contains(&row("unread")), unread.is_some(), "{got}");
            if let Some(record) = unread {
                assert!(
                    got.contains(&format!("{record}: a value is no number")),
                    "{got}"
                );
            }
        }
    }

    #[test]
    fn tier_details_use_only_the_last_records_of_the_listening_place() {
        let start = "plan place=bp card=A6000 card_experts=10 host_experts=10\n\
                     load resident_bytes=1 place=bp tier_experts=2 in 1.0 s\n\
                     bloomery-serve-glm: place=bp listening on http://127.0.0.1:8080";
        let got = render(
            &mutated(start, "load ", "place=bp", "place=a"),
            UTF8,
            super::ART_W,
        )
        .unwrap();
        assert!(!got.contains(&row("tier")), "{got}");
        assert!(!got.contains("▓ tier"), "{got}");
        let got = render(
            &mutated(start, "plan ", "place=bp", "place=a"),
            UTF8,
            super::ART_W,
        )
        .unwrap();
        assert!(
            got.contains(&format!("{}2 experts\n", row("tier"))),
            "{got}"
        );
        assert!(!got.contains(&row("experts")), "{got}");
        let mut start = lines(start);
        start.push("load resident_bytes=1 place=bp in 2.0 s");
        let got = render(&start, UTF8, super::ART_W).unwrap();
        assert!(!got.contains(&row("tier")), "{got}");
        assert!(!got.contains("▓ tier"), "{got}");
    }

    /// No `listening` line, no screen: a start that did not reach its
    /// address draws nothing.
    #[test]
    fn no_listening_line_draws_nothing() {
        let start: Vec<&str> = lines(START)
            .into_iter()
            .filter(|l| !l.contains(" listening on http://"))
            .collect();
        assert_eq!(render(&start, TRUE, super::ART_W), None);
    }

    fn wide_start() -> Vec<String> {
        let start = mutated(START, "plan ", "card=A6000", "card=삽살개카드");
        let start = mutated(
            &start.join("\n"),
            "plan ",
            "devices=[stage:A6000:cuda0:50952404992]",
            "devices=[stage:A6000:cuda0:50952404992,tier0:3090:cuda1:25350373376,tier1:3090:cuda2:25350373376,tier2:3090:cuda3:25350373376]",
        );
        let mut start = mutated(
            &start.join("\n"),
            "bloomery-serve-ds41:",
            "place=a ",
            "place=a model=삽살이의아주긴모델이름-Q4_K_M.gguf head=e\u{301}tiquette.safetensors ",
        );
        let listening = start.last_mut().unwrap();
        *listening = listening.replace(
            "127.0.0.1:8080",
            &format!("{}example.org:8080", "inference.".repeat(16)),
        );
        start
    }

    #[test]
    fn narrow_screens_keep_every_record_value() {
        let start = wide_start();
        let compact =
            |text: &str| -> String { text.chars().filter(|c| !c.is_whitespace()).collect() };
        let facts = super::facts(
            &start,
            super::Marks {
                dot: "·",
                times: "×",
            },
        )
        .unwrap();
        for columns in [2, 9, 12, 20, 40, 80, 97] {
            for style in [UTF8, CUBE, TRUE] {
                let got = strip_escapes(&render(&start, style, columns).unwrap());
                for line in got.lines() {
                    assert!(width(line) <= columns, "{columns}: {line:?}");
                }
                let all = compact(&got);
                for value in [&facts.seat, &facts.addr]
                    .into_iter()
                    .chain(facts.rows.iter().map(|(_, v)| v))
                {
                    assert!(
                        all.contains(&compact(value)),
                        "{columns}: missing {value:?}\n{got}"
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn terminal_viewport() {
        use std::io::{BufRead, Read, Write};
        use std::os::fd::{AsRawFd, FromRawFd};
        use std::process::{Command, Stdio};

        if let Ok(fixture) = std::env::var("SCREEN_PTY_CHILD") {
            assert!(!super::arm(false));
            let armed = super::arm(true);
            println!("screen armed={armed}");
            std::io::stdout().flush().unwrap();
            let start = if fixture == "unicode" {
                wide_start()
            } else {
                lines(if fixture == "qwen38" { START38 } else { START })
                    .into_iter()
                    .map(str::to_owned)
                    .collect()
            };
            for line in &start {
                super::note(line);
            }
            if armed {
                std::io::stdin().read_exact(&mut [0]).unwrap();
            }
            super::draw();
            super::draw();
            return;
        }

        let command = |fixture: &str, term: &str, color: &str| {
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", "screen::tests::terminal_viewport", "--nocapture"])
                .env("SCREEN_PTY_CHILD", fixture)
                .env("TERM", term)
                .env("NO_COLOR", color)
                .env("COLORTERM", "truecolor")
                .env("LC_ALL", "ko_KR.UTF-8");
            cmd
        };
        let piped = command("v41", "xterm", "").output().unwrap();
        assert!(piped.status.success());
        assert!(piped.stderr.is_empty(), "piped stderr drew a screen");
        assert!(
            String::from_utf8(piped.stdout)
                .unwrap()
                .contains("screen armed=false")
        );

        for (cols, resized) in [
            (80, 80),
            (97, 97),
            (98, 98),
            (99, 99),
            (120, 120),
            (98, 80),
            (80, 98),
            (0, 0),
            (1, 1),
        ] {
            for fixture in ["v41", "qwen38", "unicode"] {
                for (term, color) in [("xterm", ""), ("xterm", "1"), ("dumb", "")] {
                    let mut size = libc::winsize {
                        ws_row: 100,
                        ws_col: cols,
                        ws_xpixel: 0,
                        ws_ypixel: 0,
                    };
                    let (mut master, mut slave) = (-1, -1);
                    // SAFETY: both fd outputs and the initialized window size live through the call; the other pointers are null.
                    let rc = unsafe {
                        libc::openpty(
                            &mut master,
                            &mut slave,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            &raw mut size,
                        )
                    };
                    assert_eq!(rc, 0);
                    // SAFETY: openpty succeeded and transferred ownership of this distinct master fd.
                    let mut master = unsafe { std::fs::File::from_raw_fd(master) };
                    // SAFETY: openpty succeeded and transferred ownership of this distinct slave fd.
                    let slave = unsafe { std::fs::File::from_raw_fd(slave) };
                    let mut child = command(fixture, term, color)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(slave.try_clone().unwrap())
                        .spawn()
                        .unwrap();
                    let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
                    let mut line = String::new();
                    loop {
                        assert!(
                            stdout.read_line(&mut line).unwrap() > 0,
                            "child never armed: {line}"
                        );
                        if line.contains("screen armed=true") {
                            break;
                        }
                    }
                    size.ws_col = resized;
                    // SAFETY: slave is an open PTY fd and size is an initialized winsize for this ioctl.
                    let rc = unsafe { libc::ioctl(slave.as_raw_fd(), libc::TIOCSWINSZ, &size) };
                    assert_eq!(rc, 0);
                    drop(slave);
                    child.stdin.take().unwrap().write_all(b"x").unwrap();
                    let mut bytes = Vec::new();
                    loop {
                        let mut buf = [0; 8192];
                        match master.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => bytes.extend_from_slice(&buf[..n]),
                            Err(e) if e.raw_os_error() == Some(libc::EIO) => break,
                            Err(e) => panic!("PTY read: {e}"),
                        }
                    }
                    assert!(child.wait().unwrap().success());
                    let got = String::from_utf8(bytes).unwrap().replace("\r\n", "\n");
                    let plain = strip_escapes(&got);
                    if resized < 2 {
                        assert!(
                            plain.contains("terminal width must be at least 2 columns"),
                            "{plain}"
                        );
                        assert!(!plain.contains("READY"), "{plain}");
                        continue;
                    }
                    assert_eq!(plain.matches(" READY ").count(), 1, "{plain}");
                    for line in plain.lines() {
                        assert!(
                            width(line) <= usize::from(resized),
                            "{cols}->{resized} {fixture} {term}: {} columns in {line:?}\n{plain}",
                            width(line)
                        );
                    }
                    for fact in [
                        "placement",
                        "serving",
                        "experts",
                        "context",
                        "cache",
                        "Ctrl+C",
                        "BLOOMERY_SCREEN=off",
                        "Hello!",
                    ] {
                        assert!(plain.contains(fact), "missing {fact}:\n{plain}");
                    }
                    if resized < 98 {
                        assert!(
                            plain.lines().any(|l| l.starts_with("  placement")),
                            "{plain}"
                        );
                        assert!(plain.lines().any(|l| l.starts_with("  serving")), "{plain}");
                    }
                    if resized >= 98 && fixture == "v41" && color == "1" {
                        assert_eq!(plain, read_golden("screen-utf8.txt"));
                    }
                    if cols == 80 && resized == 80 && fixture == "qwen38" && color == "1" {
                        println!("80-column PTY:\n{plain}");
                    }
                }
            }
        }
    }

    /// The terminal's variables: `NO_COLOR` set and not empty, or
    /// `TERM=dumb`, turn colour off; `COLORTERM` of `truecolor` or `24bit`
    /// draws 24-bit colour when there is colour; the first non-empty locale
    /// variable decides UTF-8, which `TERM=dumb` turns off too.
    #[test]
    fn the_style_follows_the_terminal_variables() {
        let utf8 = Some("ko_KR.UTF-8");
        let style = |v: TermVars<'_>| Style::of(v);
        assert_eq!(
            style(TermVars {
                lang: utf8,
                ..TermVars::default()
            }),
            CUBE
        );
        for colorterm in ["truecolor", "24bit"] {
            assert_eq!(
                style(TermVars {
                    colorterm: Some(colorterm),
                    lang: utf8,
                    ..TermVars::default()
                }),
                TRUE
            );
        }
        assert!(
            !style(TermVars {
                colorterm: Some("yes"),
                ..TermVars::default()
            })
            .truecolor,
            "COLORTERM=yes is not 24-bit"
        );
        assert_eq!(
            style(TermVars {
                no_color: Some("1"),
                colorterm: Some("truecolor"),
                lang: utf8,
                ..TermVars::default()
            }),
            UTF8
        );
        assert!(
            style(TermVars {
                no_color: Some(""),
                ..TermVars::default()
            })
            .color,
            "an empty NO_COLOR is unset"
        );
        assert_eq!(
            style(TermVars {
                term: Some("dumb"),
                colorterm: Some("truecolor"),
                lang: utf8,
                ..TermVars::default()
            }),
            ASCII
        );
        assert!(
            !style(TermVars {
                lc_all: Some("C"),
                lang: utf8,
                ..TermVars::default()
            })
            .utf8,
            "LC_ALL wins over LANG"
        );
        assert!(
            style(TermVars {
                lc_all: Some(""),
                lc_ctype: Some("en_US.utf8"),
                lang: Some("C"),
                ..TermVars::default()
            })
            .utf8,
            "an empty LC_ALL is unset, and LC_CTYPE comes before LANG"
        );
        assert!(!style(TermVars::default()).utf8, "no locale is not UTF-8");
    }

    fn strip_escapes(s: &str) -> String {
        let mut out = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c == '\x1b' {
                for c in it.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}
