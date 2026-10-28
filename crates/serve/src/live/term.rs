//! The terminal under the live view: the keys it reads (cbreak, the only
//! mode change), the alternate screen it draws on, and stderr held while the
//! view is open.
//!
//! [`arm`] keeps a copy of the terminal's settings and of stderr, then turns
//! off canonical input and echo only: `ISIG` and `OPOST` stay, so `Ctrl+C`
//! reaches the server's signal handlers as before and a line written to
//! stderr ends as before. While the view is open ([`enter`]), stderr is a
//! pipe whose lines a drain thread keeps ([`Spill`]); the view draws on the
//! copy, and [`leave`] puts stderr back and writes the kept lines below a
//! rule. Every way out puts the terminal back as it was: [`leave`] and
//! [`restore`] on the server's own paths, [`restore_raw`] (async-signal-safe)
//! on a second interrupt, a stop (`SIGTSTP`, undone once the process
//! continues or the stop is discarded) and a panic.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, mpsc};
use std::time::Duration;

use crate::api::handle;

/// Switches to the alternate screen and hides the cursor.
pub(super) const ENTER: &[u8] = b"\x1b[?1049h\x1b[?25l";
/// Shows the cursor and switches back to the main screen.
const LEAVE: &[u8] = b"\x1b[?25h\x1b[?1049l";
/// How long closing the view waits for the drain to read what the pipe still
/// holds; longer only when a child kept the pipe open.
const DRAIN_WAIT: Duration = Duration::from_millis(500);

/// The terminal as stderr reached it when the keys were armed, close-on-exec:
/// what the view draws on and what stderr is put back to.
static TTY: OnceLock<File> = OnceLock::new();
/// The terminal's settings when the keys were armed, and those with
/// canonical input and echo off.
static ORIG: OnceLock<libc::termios> = OnceLock::new();
static CBREAK: OnceLock<libc::termios> = OnceLock::new();
/// Whether the cbreak settings are in force.
static APPLIED: AtomicBool = AtomicBool::new(false);
/// Whether the view is open: the alternate screen shown, stderr the pipe.
pub(super) static OPEN: AtomicBool = AtomicBool::new(false);
/// How many stops, continued or discarded, put the terminal back while the
/// view was open: the view enters again when it changes.
pub(super) static STOPS: AtomicU64 = AtomicU64::new(0);
/// The terminal was put back for good (the server ends, or a panic): no
/// setting is applied again.
pub(super) static GONE: AtomicBool = AtomicBool::new(false);

/// What decides whether a process takes keys ([`Seen::takes_keys`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct Seen<'a> {
    /// `BLOOMERY_SCREEN`.
    pub(super) on: bool,
    pub(super) stdin_tty: bool,
    pub(super) stderr_tty: bool,
    /// The process is in the terminal's foreground group ([`foreground`]).
    pub(super) foreground: bool,
    /// `TERM`, `None` unset.
    pub(super) term: Option<&'a str>,
}

impl Seen<'_> {
    /// Whether the process may take keys: the screen on, stdin and stderr
    /// terminals, the process in the foreground (a background one that
    /// changes the settings is stopped by `SIGTTOU`), and a terminal that is
    /// not `dumb`.
    pub(super) fn takes_keys(self) -> bool {
        self.on && self.stdin_tty && self.stderr_tty && self.foreground && self.term != Some("dumb")
    }
}

/// Whether this process's group is the terminal's foreground group; `false`
/// when stdin is not the controlling terminal (`tcgetpgrp` fails).
pub(super) fn foreground() -> bool {
    // SAFETY: tcgetpgrp reads the group of the fd; it takes no pointer.
    let fg = unsafe { libc::tcgetpgrp(libc::STDIN_FILENO) };
    // SAFETY: getpgrp takes no argument and cannot fail.
    let own = unsafe { libc::getpgrp() };
    fg >= 0 && fg == own
}

/// Keeps the terminal's settings and a copy of stderr, ignores `SIGTTIN` (a
/// read from the background then fails instead of stopping the server),
/// takes the stop and continue signals, and applies cbreak.
pub(super) fn arm() -> io::Result<()> {
    // SAFETY: fcntl duplicates fd 2 to a new close-on-exec fd; it takes no pointer.
    let fd = unsafe { libc::fcntl(libc::STDERR_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is a new descriptor this call owns from here.
    let tty = unsafe { File::from_raw_fd(fd) };
    // SAFETY: an all-zero termios is a valid value for tcgetattr to fill.
    let mut orig: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: orig is a writable termios for the call.
    if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut orig) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut cbreak = orig;
    cbreak.c_lflag &= !(libc::ICANON | libc::ECHO);
    cbreak.c_cc[libc::VMIN] = 1;
    cbreak.c_cc[libc::VTIME] = 0;
    if TTY.set(tty).is_err() || ORIG.set(orig).is_err() || CBREAK.set(cbreak).is_err() {
        return Err(io::Error::other(
            "the live view's keys are armed once a process",
        ));
    }
    apply();
    if !APPLIED.load(Ordering::SeqCst) {
        return Err(io::Error::other("cbreak was not applied"));
    }
    handle(libc::SIGTTIN, libc::SIG_IGN);
    handle(libc::SIGTSTP, on_tstp as *const () as usize);
    handle(libc::SIGCONT, on_cont as *const () as usize);
    Ok(())
}

/// Applies cbreak when the process is in the foreground and the terminal
/// was not put back for good. Async-signal-safe.
pub(super) fn apply() {
    let Some(cbreak) = CBREAK.get() else {
        return;
    };
    if GONE.load(Ordering::SeqCst) || APPLIED.load(Ordering::SeqCst) || !foreground() {
        return;
    }
    // SAFETY: cbreak is an initialized termios read by the call.
    if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, cbreak) } == 0 {
        APPLIED.store(true, Ordering::SeqCst);
    }
}

/// The terminal's columns and rows, read from the copy of stderr.
pub(super) fn size() -> Option<(usize, usize)> {
    let tty = TTY.get()?;
    let mut size = libc::winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: tty is a live fd and size a writable winsize for this ioctl.
    let rc = unsafe { libc::ioctl(tty.as_raw_fd(), libc::TIOCGWINSZ, &mut size) };
    (rc == 0).then_some((usize::from(size.ws_col), usize::from(size.ws_row)))
}

/// Writes `bytes` to the terminal the view draws on.
pub(super) fn draw(bytes: &[u8]) -> io::Result<()> {
    let mut tty = TTY
        .get()
        .ok_or_else(|| io::Error::other("no terminal armed"))?;
    tty.write_all(bytes)
}

/// Waits up to `ms` (forever when negative) for a byte on stdin; `Ok(None)`
/// when none came. A read that fails (`EIO` in the background) is an error.
pub(super) fn key(ms: libc::c_int) -> io::Result<Option<u8>> {
    let mut fds = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: fds is one valid pollfd for the call.
    let n = unsafe { libc::poll(&mut fds, 1, ms) };
    if n < 0 {
        let e = io::Error::last_os_error();
        return if e.kind() == io::ErrorKind::Interrupted {
            Ok(None)
        } else {
            Err(e)
        };
    }
    if n == 0 || !foreground() {
        return Ok(None);
    }
    let mut byte = [0u8; 1];
    // SAFETY: byte is a one-byte buffer the call may write.
    match unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), 1) } {
        1 => Ok(Some(byte[0])),
        0 => Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
        _ => Err(io::Error::last_os_error()),
    }
}

/// The pipe stderr is while the view is open: its write end, which [`enter`]
/// puts on fd 2. Its read end belongs to the drain thread.
pub(super) struct Capture {
    w: File,
    /// Disconnected once the drain thread has ended.
    drained: mpsc::Receiver<()>,
}

/// Makes the pipe and starts its drain thread: while the view is open it
/// keeps what arrives in [`SPILL`], after that it writes it to the terminal
/// (not through stderr, whose lock a write blocked on the pipe holds), until
/// every write end (a child's inherited fd 2 among them) is closed.
pub(super) fn capture() -> io::Result<Capture> {
    let mut fds = [0 as libc::c_int; 2];
    // SAFETY: fds is a two-int array the call may write.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe gave two new descriptors, each owned once from here.
    let (mut r, w) = unsafe { (File::from_raw_fd(fds[0]), File::from_raw_fd(fds[1])) };
    for fd in fds {
        // SAFETY: fd is one of the pipe's open descriptors; no pointer.
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let (ends, drained) = mpsc::channel::<()>();
    std::thread::Builder::new()
        .name("serve-live-err".to_owned())
        .spawn(move || {
            let _ends = ends;
            let mut buf = [0u8; 4096];
            loop {
                let n = match r.read(&mut buf) {
                    Ok(0) => return,
                    Ok(n) => n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        eprintln!("bloomery-serve: live view: stderr pipe: {e}");
                        return;
                    }
                };
                let mut spill = spill();
                if spill.holding {
                    spill.take(&buf[..n]);
                } else if let Err(e) = draw(&buf[..n]) {
                    drop(spill);
                    eprintln!("bloomery-serve: live view: stderr: {e}");
                    return;
                }
            }
        })?;
    Ok(Capture { w, drained })
}

/// Opens the view on `cap`: stderr becomes its pipe, and the terminal shows
/// the alternate screen.
pub(super) fn enter(cap: &Capture) -> io::Result<()> {
    spill().holding = true;
    OPEN.store(true, Ordering::SeqCst);
    // SAFETY: both fds are open; dup2 replaces fd 2 atomically.
    if unsafe { libc::dup2(cap.w.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
        return Err(io::Error::last_os_error());
    }
    draw(ENTER)
}

/// Closes the view: stderr and the main screen back, then the lines stderr
/// carried meanwhile, in order, below a rule (`utf8` draws it in UTF-8).
/// Writers to stderr wait until those lines are out, so none comes before
/// them.
pub(super) fn leave(cap: Capture, utf8: bool) -> io::Result<()> {
    let stderr = io::stderr();
    let _writers = stderr.lock();
    let tty = TTY
        .get()
        .ok_or_else(|| io::Error::other("no terminal armed"))?;
    // SAFETY: both fds are open; dup2 replaces fd 2 atomically.
    if unsafe { libc::dup2(tty.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
        return Err(io::Error::last_os_error());
    }
    draw(LEAVE)?;
    OPEN.store(false, Ordering::SeqCst);
    // With its last write end closed the drain reads what the pipe holds and
    // ends; a child that inherited the pipe keeps it open, and what it
    // writes later is forwarded.
    let Capture { w, drained } = cap;
    drop(w);
    let _ = drained.recv_timeout(DRAIN_WAIT);
    let mut spill = spill();
    spill.holding = false;
    let text = spill.flush(utf8);
    draw(text.as_bytes())
}

/// Puts the terminal back for good, the server's last act on it: the view
/// closed as [`leave`] does, the settings as they were.
pub(super) fn restore(cap: Option<Capture>, utf8: bool) -> io::Result<()> {
    GONE.store(true, Ordering::SeqCst);
    let left = match cap {
        Some(cap) => leave(cap, utf8),
        None => Ok(()),
    };
    restore_raw();
    left
}

/// The terminal back without a lock or an allocation, for a signal handler
/// and a panic: stderr and the main screen when the view is open, then the
/// settings (with `SIGTTOU` blocked, so a background process may set them).
/// The lines the view held stay held.
pub(super) fn restore_raw() {
    let Some(tty) = TTY.get() else {
        return;
    };
    if OPEN.load(Ordering::SeqCst) {
        // SAFETY: both fds are open; dup2 replaces fd 2 atomically.
        unsafe { libc::dup2(tty.as_raw_fd(), libc::STDERR_FILENO) };
        // SAFETY: LEAVE is a static buffer of its length.
        unsafe { libc::write(tty.as_raw_fd(), LEAVE.as_ptr().cast(), LEAVE.len()) };
    }
    let Some(orig) = ORIG.get() else {
        return;
    };
    if !APPLIED.swap(false, Ordering::SeqCst) {
        return;
    }
    // SAFETY: an all-zero sigset_t is a valid set for sigemptyset to clear.
    let mut ttou: libc::sigset_t = unsafe { std::mem::zeroed() };
    // SAFETY: an all-zero sigset_t is a valid set for pthread_sigmask to fill.
    let mut old: libc::sigset_t = unsafe { std::mem::zeroed() };
    // SAFETY: ttou is a writable set.
    unsafe { libc::sigemptyset(&mut ttou) };
    // SAFETY: ttou is an initialized set.
    unsafe { libc::sigaddset(&mut ttou, libc::SIGTTOU) };
    // SAFETY: ttou is read and old written by the call.
    unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &ttou, &mut old) };
    // SAFETY: orig is an initialized termios read by the call.
    unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, orig) };
    // SAFETY: old is the mask the block call wrote.
    unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old, std::ptr::null_mut()) };
}

/// `SIGTSTP`: the terminal back as it was ([`restore_raw`]), then the stop
/// itself, on this thread so it takes effect before the call returns. Once
/// the process continues, or the stop is discarded (an orphaned group), the
/// handler and cbreak again, and only then may a view that was open enter
/// again from its thread ([`STOPS`]).
extern "C" fn on_tstp(_: libc::c_int) {
    restore_raw();
    handle(libc::SIGTSTP, libc::SIG_DFL);
    // SAFETY: an all-zero sigset_t is a valid set for sigemptyset to clear.
    let mut tstp: libc::sigset_t = unsafe { std::mem::zeroed() };
    // SAFETY: tstp is a writable set.
    unsafe { libc::sigemptyset(&mut tstp) };
    // SAFETY: tstp is an initialized set.
    unsafe { libc::sigaddset(&mut tstp, libc::SIGTSTP) };
    // SAFETY: tstp is read by the call; the old mask is not asked for.
    unsafe { libc::pthread_sigmask(libc::SIG_UNBLOCK, &tstp, std::ptr::null_mut()) };
    // SAFETY: raise takes no pointer; the default action stops here.
    unsafe { libc::raise(libc::SIGTSTP) };
    handle(libc::SIGTSTP, on_tstp as *const () as usize);
    apply();
    if OPEN.load(Ordering::SeqCst) {
        STOPS.fetch_add(1, Ordering::SeqCst);
    }
}

/// `SIGCONT`: cbreak again when the process continues in the foreground.
extern "C" fn on_cont(_: libc::c_int) {
    apply();
}

/// The most lines [`Spill`] keeps; an older one is dropped for a newer.
const SPILL_LINES: usize = 2000;
/// The longest line it keeps whole: a longer run without a newline is kept
/// as lines of this many bytes.
const SPILL_LINE_BYTES: usize = 64 * 1024;

/// What stderr carried while the view was open.
#[derive(Default)]
pub(super) struct Spill {
    /// Whether the drain keeps what arrives, or writes it on.
    holding: bool,
    lines: VecDeque<String>,
    partial: Vec<u8>,
    /// Lines dropped for newer ones.
    dropped: usize,
}

static SPILL: Mutex<Spill> = Mutex::new(Spill {
    holding: false,
    lines: VecDeque::new(),
    partial: Vec::new(),
    dropped: 0,
});

fn spill() -> MutexGuard<'static, Spill> {
    SPILL.lock().unwrap_or_else(|e| e.into_inner())
}

impl Spill {
    /// Keeps `bytes`, line by line.
    pub(super) fn take(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if b == b'\n' || self.partial.len() == SPILL_LINE_BYTES {
                let line = String::from_utf8_lossy(&self.partial).into_owned();
                self.partial.clear();
                if self.lines.len() == SPILL_LINES {
                    self.lines.pop_front();
                    self.dropped += 1;
                }
                self.lines.push_back(line);
            }
            if b != b'\n' {
                self.partial.push(b);
            }
        }
    }

    /// What it kept, emptied: nothing when it kept nothing, else a rule,
    /// the count of lines dropped, then the lines.
    pub(super) fn flush(&mut self, utf8: bool) -> String {
        if self.lines.is_empty() && self.partial.is_empty() && self.dropped == 0 {
            return String::new();
        }
        let rule = if utf8 { "──" } else { "--" };
        let mut out = format!("{rule} stderr while the live view was open {rule}\n");
        if self.dropped > 0 {
            out.push_str(&format!(
                "({} earlier lines were dropped: the view keeps the last {SPILL_LINES})\n",
                self.dropped
            ));
        }
        for line in self.lines.drain(..) {
            out.push_str(&line);
            out.push('\n');
        }
        if !self.partial.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.partial));
            out.push('\n');
            self.partial.clear();
        }
        self.dropped = 0;
        out
    }
}

/// Puts the terminal back from the panic hook: [`restore_raw`], then the
/// held lines when their lock is free.
pub(super) fn restore_on_panic(utf8: bool) {
    GONE.store(true, Ordering::SeqCst);
    let open = OPEN.load(Ordering::SeqCst);
    restore_raw();
    OPEN.store(false, Ordering::SeqCst);
    if open && let Ok(mut spill) = SPILL.try_lock() {
        spill.holding = false;
        let text = spill.flush(utf8);
        if let Err(e) = draw(text.as_bytes()) {
            eprintln!("bloomery-serve: live view: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SPILL_LINES, Seen, Spill};

    /// The keys are taken only with the screen on, both stdin and stderr
    /// terminals, the process in the foreground and a terminal not `dumb`.
    #[test]
    fn keys_need_every_condition() {
        let all = Seen {
            on: true,
            stdin_tty: true,
            stderr_tty: true,
            foreground: true,
            term: Some("xterm"),
        };
        assert!(all.takes_keys());
        assert!(Seen { term: None, ..all }.takes_keys());
        for seen in [
            Seen { on: false, ..all },
            Seen {
                stdin_tty: false,
                ..all
            },
            Seen {
                stderr_tty: false,
                ..all
            },
            Seen {
                foreground: false,
                ..all
            },
            Seen {
                term: Some("dumb"),
                ..all
            },
        ] {
            assert!(!seen.takes_keys(), "{seen:?}");
        }
    }

    /// The held lines come back in order below the rule, a line without its
    /// newline last; past the bound the oldest go and one line counts them.
    #[test]
    fn the_spill_keeps_order_and_names_what_it_dropped() {
        let mut s = Spill::default();
        assert_eq!(s.flush(true), "", "nothing held, nothing written");
        s.take(b"one\ntw");
        s.take(b"o\nthree");
        assert_eq!(
            s.flush(false),
            "-- stderr while the live view was open --\none\ntwo\nthree\n"
        );
        for i in 0..SPILL_LINES + 3 {
            s.take(format!("{i}\n").as_bytes());
        }
        let text = s.flush(true);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[1],
            format!("(3 earlier lines were dropped: the view keeps the last {SPILL_LINES})")
        );
        assert_eq!(lines[2], "3");
        assert_eq!(
            lines.last().copied(),
            Some(&*format!("{}", SPILL_LINES + 2))
        );
        assert_eq!(lines.len(), SPILL_LINES + 2);
        assert_eq!(s.flush(true), "", "a flush empties it");
    }
}
