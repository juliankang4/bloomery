//! The live view: while the server serves, `d` opens a view of what it is
//! doing on the terminal's alternate screen, and `q` or `Esc` closes it; the
//! server serves on throughout, and `Ctrl+C` stops it from either screen.
//!
//! It takes keys only where a person can press them and nothing else is
//! disturbed ([`term::Seen::takes_keys`]): `BLOOMERY_SCREEN` on ([`enable`]), stdin
//! and stderr terminals, the process in the terminal's foreground group,
//! and a terminal that is not `dumb`. Elsewhere it changes nothing and no
//! screen names its key. While it is closed its thread waits on stdin and
//! does nothing else; while it is open it draws a frame every
//! [`REDRAW_MS`] from a copy of the server's state, every lock held only to
//! copy ([`view::Now`]).
//!
//! The terminal is the view's to put back on every way out ([`term`]):
//! `q`, the orderly stop ([`restore`], first thing in its exit), an error that
//! ends the server, a second interrupt ([`restore_from_signal`]), a stop and
//! continue (`Ctrl+Z`, `fg`), and a panic. Any panic disables the view and
//! its keys for the rest of the process, even when the server catches it
//! and keeps serving.

mod parts;
mod ring;
mod term;
mod view;

use std::collections::VecDeque;
use std::io::{self, IsTerminal};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub(crate) use ring::{Ending, Finished, Ring};
pub(crate) use view::{Now, Phase, Slot, Status};

use crate::screen::{self, Marks, Style};
use term::Capture;
use view::Fixed;

/// How often the open view draws a frame, in milliseconds.
const REDRAW_MS: libc::c_int = 500;
/// How far back a decoding slot's rate looks: long enough to span several
/// steps of a draft that adds a few tokens at a time.
const RATE_WINDOW: Duration = Duration::from_secs(2);
/// How long an `Esc` waits for the rest of an arrow key's sequence.
const ESC_MS: libc::c_int = 30;
/// How long the view's thread waits before it looks again while the process
/// is in the background.
const BACKGROUND: Duration = Duration::from_millis(300);

/// `BLOOMERY_SCREEN` as the seat read it.
static ON: AtomicBool = AtomicBool::new(false);

/// Whether the keys were armed ([`start`]): what [`restore`] puts back.
static ARMED: AtomicBool = AtomicBool::new(false);

/// The pipe stderr is while the view is open; its lock also orders the
/// frames, the view's opening and closing, and [`restore`].
static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);

fn capture() -> MutexGuard<'static, Option<Capture>> {
    CAPTURE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Turns the live view on or off for this process: a seat's
/// `BLOOMERY_SCREEN`, which the start screen's arming also passes here.
pub fn enable(on: bool) {
    ON.store(on, Ordering::SeqCst);
}

/// Whether the server takes the view's keys ([`term::Seen::takes_keys`]).
pub(crate) fn keys() -> bool {
    let term = Style::term();
    term::Seen {
        on: ON.load(Ordering::SeqCst),
        stdin_tty: io::stdin().is_terminal(),
        stderr_tty: io::stderr().is_terminal(),
        foreground: term::foreground(),
        term: term.as_deref(),
    }
    .takes_keys()
}

/// Arms the keys and starts the view's thread for the server listening on
/// `addr` serving `model`, which lists `routes`, checks API keys when
/// `keyed` and reads its state from `now`. Called once [`keys`] said yes; a
/// failure leaves the terminal as it was.
pub(crate) fn start(
    addr: SocketAddr,
    model: String,
    routes: Vec<(&'static str, &'static str)>,
    keyed: bool,
    now: Box<dyn Fn() -> Now + Send>,
) -> io::Result<()> {
    let style = Style::of_terminal();
    let records = screen::records();
    let fixed = Fixed {
        addr,
        model,
        keyed,
        since: Instant::now(),
        facts: screen::facts(&records, Marks::of(style)),
        records,
        routes,
    };
    term::arm()?;
    ARMED.store(true, Ordering::SeqCst);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        term::restore_on_panic(style.utf8);
        previous(info);
    }));
    let mut v = Live {
        fixed,
        now,
        style,
        tab: 0,
        scroll: 0,
        stops: 0,
        seen: VecDeque::new(),
    };
    threads::helper::spawn_helper("serve-live", threads::helper::Placement::Float, move || {
        v.run();
    })
    .map_err(io::Error::other)
    .inspect_err(|_| restore())?;
    Ok(())
}

/// Puts the terminal back for good: the view closed (stderr back, the lines
/// it held written below a rule) and the settings as they were. Nothing in a
/// process whose keys were never armed.
pub(crate) fn restore() {
    if !ARMED.load(Ordering::SeqCst) {
        return;
    }
    let utf8 = Style::of_terminal().utf8;
    // Held through the restore, so a `d` pressed meanwhile cannot open the
    // view after it.
    let mut slot = capture();
    if let Err(e) = term::restore(slot.take(), utf8) {
        eprintln!("bloomery-serve: live view: putting the terminal back: {e}");
    }
}

/// [`restore`] for a signal handler: async-signal-safe, the lines the view
/// held left unwritten.
pub(crate) fn restore_from_signal() {
    term::restore_raw();
}

/// The view's thread: the keys, and the frames while the view is open.
struct Live {
    fixed: Fixed,
    now: Box<dyn Fn() -> Now + Send>,
    style: Style,
    tab: usize,
    scroll: usize,
    /// The stops ([`term::STOPS`]) the view entered again after.
    stops: u64,
    /// The slots at the frames of the last [`RATE_WINDOW`] and when they
    /// were copied, oldest first: what a decoding slot's rate is measured from.
    seen: VecDeque<(Instant, Vec<Slot>)>,
}

impl Live {
    fn run(&mut self) {
        loop {
            if term::GONE.load(Ordering::SeqCst) {
                return;
            }
            if !term::foreground() {
                std::thread::sleep(BACKGROUND);
                continue;
            }
            term::apply();
            let open = capture().is_some();
            match term::key(if open { REDRAW_MS } else { -1 }) {
                Ok(Some(b)) => self.press(b),
                Ok(None) => {}
                Err(_) if !term::foreground() => continue,
                Err(e) => {
                    restore();
                    eprintln!("bloomery-serve: live view: keys: {e}; the view is off");
                    return;
                }
            }
            self.draw();
        }
    }

    /// Acts on the key `b`: `d` opens the view; while it is open, `q` and a
    /// lone `Esc` close it, a digit or `Tab` (and the left and right arrows)
    /// picks a tab, and the up and down arrows (or `k` and `j`) scroll.
    fn press(&mut self, b: u8) {
        let open = capture().is_some();
        if !open {
            if b == screen::LIVE_KEY.as_bytes()[0] {
                self.open();
            }
            return;
        }
        let tabs = self.fixed.tabs().len();
        match b {
            b'q' => self.close(),
            0x1b => match self.sequence() {
                None => self.close(),
                Some(b'A') => self.scroll = self.scroll.saturating_sub(1),
                Some(b'B') => self.scroll = self.scroll.saturating_add(1),
                Some(b'C') => self.pick((self.tab + 1) % tabs),
                Some(b'D') => self.pick((self.tab + tabs - 1) % tabs),
                Some(_) => {}
            },
            b'k' => self.scroll = self.scroll.saturating_sub(1),
            b'j' => self.scroll = self.scroll.saturating_add(1),
            b'\t' => self.pick((self.tab + 1) % tabs),
            b'1'..=b'9' if usize::from(b - b'1') < tabs => self.pick(usize::from(b - b'1')),
            _ => {}
        }
    }

    /// After an `Esc`: the final byte of an arrow key's sequence, `Some(0)`
    /// for another sequence, `None` for a lone `Esc`.
    fn sequence(&self) -> Option<u8> {
        match term::key(ESC_MS) {
            Ok(Some(b'[' | b'O')) => match term::key(ESC_MS) {
                Ok(Some(b)) => Some(b),
                _ => Some(0),
            },
            Ok(Some(_)) => Some(0),
            _ => None,
        }
    }

    fn pick(&mut self, tab: usize) {
        self.tab = tab;
        self.scroll = 0;
    }

    fn open(&mut self) {
        let mut slot = capture();
        if term::GONE.load(Ordering::SeqCst) {
            return;
        }
        let cap = match term::capture() {
            Ok(cap) => cap,
            Err(e) => {
                eprintln!("bloomery-serve: live view: cannot open: {e}");
                return;
            }
        };
        self.stops = term::STOPS.load(Ordering::SeqCst);
        self.seen.clear();
        match term::enter(&cap) {
            Ok(()) => *slot = Some(cap),
            Err(e) => {
                if let Err(back) = term::leave(cap, self.style.utf8) {
                    eprintln!("bloomery-serve: live view: closing: {back}");
                }
                eprintln!("bloomery-serve: live view: cannot open: {e}");
            }
        }
    }

    fn close(&mut self) {
        let Some(cap) = capture().take() else {
            return;
        };
        if let Err(e) = term::leave(cap, self.style.utf8) {
            eprintln!("bloomery-serve: live view: closing: {e}");
        }
    }

    /// A frame of the open view, after entering again when a stop put the
    /// terminal back meanwhile.
    fn draw(&mut self) {
        if !term::foreground() || capture().is_none() {
            return;
        }
        let mut now = (self.now)();
        let at = Instant::now();
        while self
            .seen
            .front()
            .is_some_and(|(then, _)| at.duration_since(*then) > RATE_WINDOW)
        {
            self.seen.pop_front();
        }
        if let Some((then, before)) = self.seen.front() {
            now.rates = view::rates(before, &now.slots, at.duration_since(*then));
        }
        self.seen.push_back((at, now.slots.clone()));
        let slot = capture();
        let Some(cap) = slot.as_ref() else {
            return;
        };
        let stops = term::STOPS.load(Ordering::SeqCst);
        if stops != self.stops {
            self.stops = stops;
            if let Err(e) = term::enter(cap) {
                eprintln!("bloomery-serve: live view: entering again: {e}");
            }
        }
        let Some((columns, rows)) = term::size() else {
            if let Err(e) = term::draw(b"\x1b[H\x1b[Jterminal size unavailable\r\n") {
                eprintln!("bloomery-serve: live view: drawing: {e}");
            }
            return;
        };
        let (frame, scroll) = view::frame(
            &now,
            &self.fixed,
            self.tab,
            self.scroll,
            self.style,
            columns,
            rows,
        );
        self.scroll = scroll;
        if let Err(e) = term::draw(frame.as_bytes()) {
            eprintln!("bloomery-serve: live view: drawing: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// The child's server: the mock engine, the live view on, run as a seat
    /// runs it.
    fn child() -> ! {
        // SAFETY: report is an `extern "C" fn()` that lives as long as the process.
        assert_eq!(unsafe { libc::atexit(report) }, 0);
        // A stop must leave cbreak in force without interrupting the key
        // wait, so `SIGTSTP` goes to a thread other than the keys' one: the
        // parked thread takes it, and every thread the server starts
        // inherits the block below.
        threads::helper::spawn_helper("live-test-stop", threads::helper::Placement::Float, || {
            loop {
                std::thread::park();
            }
        })
        .unwrap();
        let tstp = super::term::only(libc::SIGTSTP);
        // SAFETY: tstp is read by the call; the old mask is not asked for.
        let rc = unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &tstp, std::ptr::null_mut()) };
        assert_eq!(rc, 0);
        super::enable(true);
        let config = crate::ServerConfig {
            model_alias: "mock".to_owned(),
            model_path: "mock.gguf".to_owned(),
            chat_template: "{%- for m in messages %}{{ m.content }}{%- endfor %}".to_owned(),
            chat_preparation: None,
            sampler: None,
            fatal_linger: crate::FATAL_LINGER,
            slot_save_path: None,
            api_keys: crate::flag::ApiKeys::default(),
        };
        let server =
            crate::Server::bind("127.0.0.1:0", Box::new(crate::MockEngine::new(64)), config)
                .expect("bind");
        let e = server.run();
        panic!("the server ended: {e}");
    }

    /// The terminal's settings as the process exits, on stdout: what the
    /// exit left them as, read before the terminal can go with the process.
    extern "C" fn report() {
        let t = termios(libc::STDIN_FILENO);
        let line = format!(
            "settings at exit: {:x} {:x} {:x} {:x} {:x?}\n",
            t.c_iflag, t.c_oflag, t.c_cflag, t.c_lflag, t.c_cc
        );
        // SAFETY: line is a live buffer of its length.
        unsafe { libc::write(libc::STDOUT_FILENO, line.as_ptr().cast(), line.len()) };
    }

    fn termios(fd: libc::c_int) -> libc::termios {
        // SAFETY: an all-zero termios is a valid value for tcgetattr to fill.
        let mut t: libc::termios = unsafe { std::mem::zeroed() };
        // SAFETY: fd is an open terminal and t a writable termios.
        assert_eq!(unsafe { libc::tcgetattr(fd, &mut t) }, 0);
        t
    }

    /// The live view on a terminal the server controls: the key named once
    /// the server listens (no start screen was drawn), cbreak in force, `d`
    /// opening the alternate screen with the Overview, a `Ctrl+Z` leaving and
    /// entering it again, `q` closing it, a `Ctrl+Z` with it closed, `d`
    /// again and `Ctrl+C` stopping the server the orderly way with the view
    /// closed first, and the terminal's settings at the exit as they were
    /// before the start.
    #[test]
    fn a_terminal_opens_and_closes_the_view_and_gets_its_settings_back() {
        if std::env::var_os("LIVE_PTY_CHILD").is_some() {
            child();
        }
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: 40,
            ws_col: 100,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: both fd outputs and the initialized window size live through the call; the
        // other pointers are null.
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
        let master = unsafe { std::fs::File::from_raw_fd(master) };
        // SAFETY: openpty succeeded and transferred ownership of this distinct slave fd.
        let slave = unsafe { std::fs::File::from_raw_fd(slave) };
        let before = termios(slave.as_raw_fd());
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "live::tests::a_terminal_opens_and_closes_the_view_and_gets_its_settings_back",
            "--nocapture",
        ])
        .env("LIVE_PTY_CHILD", "1")
        .env("TERM", "xterm")
        .env("LC_ALL", "C")
        .stdin(slave.try_clone().unwrap())
        .stdout(slave.try_clone().unwrap())
        .stderr(slave.try_clone().unwrap());
        // SAFETY: between fork and exec the closure makes only async-signal-safe calls:
        // a new session, then the terminal on fd 0 its controlling terminal.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = cmd.spawn().unwrap();

        let (tx, rx) = mpsc::channel();
        let mut reader = master.try_clone().unwrap();
        threads::helper::spawn_helper(
            "live-test-read",
            threads::helper::Placement::Float,
            move || {
                let mut buf = [0u8; 8192];
                while let Ok(n) = reader.read(&mut buf) {
                    if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                        return;
                    }
                }
            },
        )
        .unwrap();
        let mut seen = String::new();
        let mut wait_for = |what: &str, from: usize| -> usize {
            let until = Instant::now() + Duration::from_secs(20);
            loop {
                if let Some(at) = seen[from..].find(what) {
                    return from + at + what.len();
                }
                let left = until.saturating_duration_since(Instant::now());
                match rx.recv_timeout(left) {
                    Ok(b) => seen.push_str(&String::from_utf8_lossy(&b)),
                    Err(_) => panic!("no {what:?} after {from}:\n{seen}"),
                }
            }
        };
        let mut master = master;
        let at = wait_for("opens the live view", 0);
        let until = Instant::now() + Duration::from_secs(10);
        while termios(slave.as_raw_fd()).c_lflag & libc::ICANON != 0 {
            assert!(Instant::now() < until, "cbreak never applied");
            std::thread::sleep(Duration::from_millis(20));
        }
        let armed = termios(slave.as_raw_fd());
        assert_ne!(armed.c_lflag & libc::ISIG, 0, "ISIG stays");
        assert_eq!(armed.c_lflag & libc::ECHO, 0, "echo off");

        master.write_all(b"d").unwrap();
        let at = wait_for("\x1b[?1049h", at);
        let at = wait_for("Overview", at);
        wait_for("Decode", at);
        size.ws_col = 3;
        // SAFETY: slave is an open PTY and size is a readable winsize.
        let rc = unsafe { libc::ioctl(slave.as_raw_fd(), libc::TIOCSWINSZ, &size) };
        assert_eq!(rc, 0);
        let at = wait_for("terminal too small", at);
        size.ws_col = 100;
        // SAFETY: slave is an open PTY and size is a readable winsize.
        let rc = unsafe { libc::ioctl(slave.as_raw_fd(), libc::TIOCSWINSZ, &size) };
        assert_eq!(rc, 0);
        let at = wait_for("Overview", at);
        // The child leads its own session, so its group is orphaned and the
        // stop a Ctrl+Z asks for is discarded: the open view leaves and
        // enters again, and the closed one takes `d` without Enter.
        master.write_all(b"\x1a").unwrap();
        let at = wait_for("\x1b[?1049l", at);
        let at = wait_for("\x1b[?1049h", at);
        let at = wait_for("Overview", at);
        master.write_all(b"q").unwrap();
        let at = wait_for("\x1b[?1049l", at);
        // The closed view shows nothing of its key wait or of a stop: time
        // for its thread to reach the wait, then for the handler to run.
        std::thread::sleep(Duration::from_millis(300));
        master.write_all(b"\x1a").unwrap();
        std::thread::sleep(Duration::from_millis(300));
        master.write_all(b"d").unwrap();
        let at = wait_for("\x1b[?1049h", at);
        let at = wait_for("Overview", at);
        master.write_all(b"\x03").unwrap();
        let at = wait_for("\x1b[?1049l", at);
        let at = wait_for("shutdown (SIGINT)", at);
        let at = wait_for("settings at exit:", at);
        let end = wait_for("\n", at);
        let status = child.wait().unwrap();
        assert!(status.success(), "{status:?}\n{seen}");
        let want = format!(
            " {:x} {:x} {:x} {:x} {:x?}",
            before.c_iflag, before.c_oflag, before.c_cflag, before.c_lflag, before.c_cc
        );
        assert_eq!(
            seen[at..end].trim_end(),
            want,
            "the settings at exit, then before the start"
        );
    }
}
