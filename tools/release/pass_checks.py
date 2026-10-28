#!/usr/bin/env python3
"""The functional phase of a release sitting's cell: what a user's first requests check, on a live server.

    tools/release/pass_checks.py [URL] [--pid N] [--stderr FILE] [--dir DIR] [--tag TEXT] [--timeout S]
                                 [--max-tokens N] [--long-repeat N] [--client toktape [--toktape-bin PATH]]
    tools/release/pass_checks.py off-reason --stderr FILE
    tools/release/pass_checks.py --self-test

The server is the cell's `bloomery-serve`, already listening. URL, --pid (the server's process), --stderr (its
stderr) and --dir (where each request and its reply are kept, one JSON a check) default to
$BLOOMERY_SESSION_URL, _PID, _STDERR and _DIR. Every request is a greedy chat turn (temperature 0, --max-tokens
128, thinking off) on /v1/chat/completions, one at a time and in this order, each check printing
`check <name>: PASS|FAIL|SKIP <detail>`:

  pass_short                   a short prompt is answered
  pass_long                    a prompt of at least 4096 tokens, counted by /tokenize before it is sent (--long-repeat
                               repeats of one sentence; a count under 4096 is a FAIL whose detail starts `harness:`
                               and names it, and the prompt is not sent), is answered
  second_request_equals_first  pass_short's request replayed, after the long one, answers the same text
  tape_short, tape_long        only with --client toktape: the two prompts recorded as tapes by `toktape record` as the
                               release pass did (the flags are pass.sh's), after the checks above so no tape
                               changes what they saw; the long one is skipped when its token count failed
  off_reason                   only when the server is not up when the checks end: its refusal read from --stderr

A reply's text is `content`, or `reasoning_content` when content is empty; both empty is a FAIL. Anything but HTTP
200 is a FAIL. A FAIL does not stop the next check. After every check the server is probed (/health, and --pid
alive and not a zombie); a server that is not up skips every check left with `server not up: <why>`, the why
carrying its refusal. The last line is `SESSION rc=<n>`: 1 when any check is a FAIL, else 0 (and the exit status).
A line `session-plan: place=.. ctx_max=.. load_ctx=..` before the checks reads the server's `plan` and `load`
records from --stderr (tools/bloomery/records.py), the key the cell ran at.

An off cell is a server that never listened and said why. Its checks are all SKIP and `off_reason` is PASS with the
refusal text; SESSION rc is 0, because a named refusal is an outcome the release notes list, and the caller that
expects a cell to run sees `check off_reason:` and decides. A server that is down with no refusal line past the
start banner and the records is `off_reason: FAIL`. The refusal is the last line `bloomery-serve*: <text>` that is
neither the start banner nor a record, with the lines after it that are not records (`caused by:` ...); the process
prints it last, whatever the log holds before. `off-reason --stderr FILE` prints that text (exit 1 and no output when
the file has none).

Exit status: SESSION rc (0 or 1); 2 a named refusal (a file or the toktape binary is missing); 64 a usage error.
--self-test runs every case against tools/ref/stub-llama-server.py on a port of its own, then each mutant of the
code (the behaviour of the pass scripts this replaces, one at a time) and requires its case to fail.
"""
import argparse
import contextlib
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = HERE.parent / "ref" / "stub-llama-server.py"

SHORT_PROMPT = "Reply with one short sentence about the weather."
LONG_SENTENCE = "Local model serving keeps the weights on the card and the host. "
LONG_MIN_TOKENS = 4096
CHAT_PATH = "/v1/chat/completions"
PROBE_TIMEOUT = 5.0
HEAD = re.compile(r"^bloomery-serve[\w-]*: ")
BANNER = re.compile(r"^bloomery-serve[\w-]*: a first start on a new card\b")
# The binaries whose records a server's stderr can hold; `plan` and `load` are the generate_ds41 kinds.
RECORD_BINS = ("generate_ds41", "bloomery-serve-ds41", "bloomery-serve-qwen38")
REASON_LINES = 8


class CheckFail(Exception):
    pass


class CheckSkip(Exception):
    pass


class Unreachable(Exception):
    pass


def load_records():
    spec = importlib.util.spec_from_file_location("bloomery_records", HERE.parent / "bloomery" / "records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


records = load_records()
SCHEMAS = {}


def schema(bin_name):
    if bin_name not in SCHEMAS:
        SCHEMAS[bin_name] = records.load(bin_name)
    return SCHEMAS[bin_name]


def head(text, n=200):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def request(base, method, path, body, timeout):
    """(status, body text) of one HTTP request; Unreachable when no answer comes."""
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException) as e:
        raise Unreachable(str(e) or type(e).__name__)
    finally:
        conn.close()


def alive(pid):
    """Whether process `pid` runs: a zombie is not running."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError):
        return True
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(stat.strip()) and not stat.strip().startswith("Z")


def chat_body(prompt, max_tokens):
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}


def reply_text(obj):
    """(text, field) of a chat reply: content, or reasoning_content when content is empty; both empty is a CheckFail."""
    msg = ((obj.get("choices") or [{}])[0]).get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    text = content if content.strip() else reasoning
    if not text.strip():
        raise CheckFail("both content and reasoning_content are empty")
    return text, "content" if content.strip() else "reasoning_content"


def refusal_of(text):
    """What a server that ended printed last as its reason: the last `bloomery-serve*: <text>` line that is neither
    the start banner nor a record, and the non-record lines after it; None when the log has none."""
    lines = text.splitlines()
    schemas = [schema(b) for b in RECORD_BINS]

    def is_rec(line):
        return any(sc.read_line(line) is not None for sc in schemas)

    start = None
    for i, line in enumerate(lines):
        if HEAD.match(line) and not BANNER.match(line) and not is_rec(line):
            start = i
    if start is None:
        return None
    block = [HEAD.sub("", lines[start], count=1).strip()]
    for line in lines[start + 1:]:
        if line.strip() and not is_rec(line):
            block.append(line.strip())
    return head(" / ".join(block[:REASON_LINES]), 500)


class Session:
    def __init__(self, a):
        self.a = a
        self.fails = 0
        self.dead = None          # why the server is not up, once it is known not to be
        self.first = None         # (request body, text) of pass_short
        self.long_ok = False      # whether pass_long's token count reached the bound

    def say(self, name, status, detail=""):
        if status == "FAIL":
            self.fails += 1
        print(f"check {name}: {status}" + (f" {head(detail, 400)}" if detail else ""), flush=True)

    def stderr_text(self):
        try:
            return Path(self.a.stderr).read_text(errors="replace") if self.a.stderr else None
        except OSError:
            return None

    def save(self, name, body, status, text):
        if self.a.dir:
            (Path(self.a.dir) / f"{name}.json").write_text(
                json.dumps({"request": body, "status": status, "response": text[:20000]}, indent=1))

    def info(self):
        text = self.stderr_text()
        if text is None:
            return
        recs = records.read(text.splitlines(), schema=schema("generate_ds41"))
        plan, load = records.first(recs, "plan"), records.first(recs, "load")
        parts = []
        if plan is not None:
            parts += [f"place={plan.get('place')}", f"ctx_max={plan.get('ctx_max')}"]
        if load is not None:
            parts.append(f"load_ctx={load.get('ctx')}")
        if parts:
            print("session-plan: " + " ".join(parts), flush=True)

    def probe_down(self):
        """None while the server answers /health and its process runs; else why it is down."""
        why = None
        if self.a.pid and not alive(self.a.pid):
            why = f"process {self.a.pid} has exited"
        else:
            try:
                status, _ = request(self.a.url, "GET", "/health", None, PROBE_TIMEOUT)
                if status == 200:
                    return None
                why = f"/health answered HTTP {status}"
            except Unreachable as e:
                why = f"/health: {e}"
        text = self.stderr_text()
        said = refusal_of(text) if text is not None else None
        return f"{why}; its last words: {said}" if said else why

    def post_chat(self, name, body):
        """(text, field) of one chat request; CheckFail on a status other than 200, a reply that is no JSON, or no text."""
        status, text = request(self.a.url, "POST", CHAT_PATH, body, self.a.timeout)
        self.save(name, body, status, text)
        if status != 200:
            raise CheckFail(f"HTTP {status}: {head(text)}")
        try:
            obj = json.loads(text)
        except ValueError:
            raise CheckFail(f"the reply is not JSON: {head(text)}")
        return reply_text(obj)

    def tokenized(self, text):
        st, reply = request(self.a.url, "POST", "/tokenize", {"content": text, "add_special": False}, self.a.timeout)
        if st != 200:
            raise CheckFail(f"harness: /tokenize answered HTTP {st}: {head(reply)}")
        try:
            return len(json.loads(reply)["tokens"])
        except (ValueError, KeyError, TypeError):
            raise CheckFail(f"harness: /tokenize's reply has no tokens list: {head(reply)}")

    def pass_short(self):
        body = chat_body(SHORT_PROMPT, self.a.max_tokens)
        text, field = self.post_chat("pass_short", body)
        self.first = (body, text)
        return f"{len(text)} chars from {field}"

    def pass_long(self):
        prompt = (LONG_SENTENCE * self.a.long_repeat).strip()
        tokens = self.tokenized(prompt)
        if tokens < LONG_MIN_TOKENS:
            raise CheckFail(f"harness: the long prompt is {tokens} tokens by /tokenize, under {LONG_MIN_TOKENS}")
        self.long_ok = True
        text, field = self.post_chat("pass_long", chat_body(prompt, self.a.max_tokens))
        return f"{tokens} prompt tokens, {len(text)} chars from {field}"

    def second_request_equals_first(self):
        if self.first is None:
            raise CheckSkip("pass_short gave no reply to replay")
        body, want = self.first
        text, _ = self.post_chat("second_request_equals_first", body)
        if text != want:
            raise CheckFail(f"the replay's text differs: first {head(want, 60)!r}, replay {head(text, 60)!r}")
        return f"the replay of pass_short has the same {len(text)} chars"

    def tape(self, kind):
        a = self.a
        if kind == "long" and not self.long_ok:
            raise CheckSkip("pass_long did not reach its token count")
        prompt = SHORT_PROMPT if kind == "short" else (LONG_SENTENCE * a.long_repeat).strip()
        prompts = Path(a.dir) / f"{a.tag}-{kind}.jsonl"
        prompts.write_text(json.dumps({"name": kind, "prompt": prompt}, ensure_ascii=False) + "\n")
        cmd = [a.toktape_bin, "record", "--url", a.url, "--temp", "0", "--no-think", "--endpoint", "chat",
               "-n", str(a.max_tokens), "--for", "0", "--wait", "60s", "--host-label", "none",
               "--prompts", str(prompts), "--out", str(Path(a.dir) / "tapes"), "--tag", f"{a.tag}-{kind}",
               "-o", "json", "--quiet"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            raise CheckFail(f"toktape did not end within {a.timeout:g} s")
        (Path(a.dir) / f"{a.tag}-{kind}.json").write_text(r.stdout)
        if r.returncode != 0:
            raise CheckFail(f"toktape exit {r.returncode}: {head(r.stderr or r.stdout)}")
        try:
            s = json.loads(r.stdout)
            n = (s.get("timings") or {}).get("prompt_n", s.get("prompt_tokens"))
        except (ValueError, AttributeError):
            n = None
        return f"tape {a.tag}-{kind}" + (f", {n} prompt tokens" if n is not None else "")

    def off_reason(self):
        if not self.a.stderr:
            self.say("off_reason", "FAIL", "harness: no --stderr, so the server's refusal cannot be read")
            return
        text = self.stderr_text()
        said = refusal_of(text) if text is not None else None
        if said:
            self.say("off_reason", "PASS", said)
        else:
            self.say("off_reason", "FAIL", f"no refusal line in {self.a.stderr} (the start banner and records are no reason)")

    def steps(self):
        out = [("pass_short", self.pass_short), ("pass_long", self.pass_long),
               ("second_request_equals_first", self.second_request_equals_first)]
        if self.a.client == "toktape":
            out += [("tape_short", lambda: self.tape("short")), ("tape_long", lambda: self.tape("long"))]
        return out

    def run(self):
        self.info()
        self.dead = self.probe_down()
        for name, fn in self.steps():
            if self.dead:
                self.say(name, "SKIP", f"server not up: {self.dead}")
                continue
            try:
                self.say(name, "PASS", fn())
            except CheckFail as e:
                self.say(name, "FAIL", str(e))
            except CheckSkip as e:
                self.say(name, "SKIP", str(e))
            except Unreachable as e:
                self.say(name, "FAIL", f"no answer: {e}")
            self.dead = self.probe_down()
        if self.dead is not None:
            self.off_reason()
        rc = 1 if self.fails else 0
        print(f"SESSION rc={rc}", flush=True)
        return rc


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"pass_checks.py: {message}", file=sys.stderr)
        sys.exit(64)


def refuse(msg):
    print(f"pass_checks.py: {msg}", file=sys.stderr)
    sys.exit(2)


def main(argv, env=None):
    env = os.environ if env is None else env
    if argv[:1] == ["off-reason"]:
        ap = Parser(prog="pass_checks.py off-reason")
        ap.add_argument("--stderr", required=True)
        a = ap.parse_args(argv[1:])
        try:
            said = refusal_of(Path(a.stderr).read_text(errors="replace"))
        except OSError as e:
            refuse(f"cannot read {a.stderr}: {e}")
        if said:
            print(said)
        return 0 if said else 1
    ap = Parser(prog="pass_checks.py")
    ap.add_argument("url", nargs="?", default=env.get("BLOOMERY_SESSION_URL"))
    ap.add_argument("--pid", default=env.get("BLOOMERY_SESSION_PID"))
    ap.add_argument("--stderr", default=env.get("BLOOMERY_SESSION_STDERR"))
    ap.add_argument("--dir", default=env.get("BLOOMERY_SESSION_DIR"))
    ap.add_argument("--tag", default="pass")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-repeat", type=int, default=450)
    ap.add_argument("--client", choices=("http", "toktape"), default="http")
    ap.add_argument("--toktape-bin")
    a = ap.parse_args(argv)
    if not a.url:
        ap.error("no URL: give one, or set BLOOMERY_SESSION_URL")
    if not re.match(r"^http://[^/:]+(:\d+)?/?$", a.url):
        ap.error(f"the URL {a.url!r} is not http://host[:port]")
    if a.client == "toktape":
        if not a.dir:
            ap.error("--client toktape needs --dir (the tapes and their prompt files go there)")
        a.toktape_bin = a.toktape_bin or shutil.which("toktape")
        if not a.toktape_bin:
            refuse("--client toktape: toktape is not on PATH (--toktape-bin names it)")
    if a.dir:
        Path(a.dir).mkdir(parents=True, exist_ok=True)
    if a.stderr and not Path(a.stderr).is_file():
        refuse(f"--stderr {a.stderr} is not a file")
    return Session(a).run()


# ---- self-test ----

class Stub:
    """tools/ref/stub-llama-server.py as a child: its stdout read for the listening line, its stderr in a file."""

    def __init__(self, tmp, env, args):
        e = dict(os.environ)
        e["TMPDIR"] = str(tmp)
        e.update(env)
        self.err = Path(tmp) / "stub.err"
        self.errf = open(self.err, "wb")
        self.proc = subprocess.Popen([sys.executable, "-I", str(STUB), "--port", "0", *args], stdout=subprocess.PIPE,
                                     stderr=self.errf, env=e, text=True)
        self.port = None
        timer = threading.Timer(30, self.proc.kill)
        timer.start()
        for line in self.proc.stdout:
            m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", line)
            if m:
                self.port = int(m.group(1))
                break
        timer.cancel()
        if self.port is None:
            self.proc.wait()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
        self.proc.wait()
        self.proc.stdout.close()
        self.errf.close()


SCRATCH = []


def scratch(prefix):
    """A temporary directory the self-test removes when it ends."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    SCRATCH.append(d)
    return d


def closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse(lines):
    by, order, rc = {}, [], None
    for line in lines:
        m = re.match(r"check (\w+): (PASS|FAIL|SKIP)(?: (.*))?$", line)
        if m:
            by[m.group(1)] = (m.group(2), m.group(3) or "")
            order.append(m.group(1))
        m = re.match(r"SESSION rc=(\d+)$", line)
        if m:
            rc = int(m.group(1))
    return by, order, rc


class Run:
    """One session of `main` against a fresh stub child."""

    def __init__(self, main, stub_env=None, stub_args=(), argv=(), env_style=False, pid=True):
        self.tmp = scratch("pass-checks-st.")
        self.stub = Stub(self.tmp, stub_env or {}, stub_args)
        try:
            port = self.stub.port or closed_port()
            url = f"http://127.0.0.1:{port}"
            pid_s = str(self.stub.proc.pid)
            if env_style:
                env = {"BLOOMERY_SESSION_URL": url, "BLOOMERY_SESSION_PID": pid_s,
                       "BLOOMERY_SESSION_STDERR": str(self.stub.err), "BLOOMERY_SESSION_DIR": str(self.tmp / "out")}
                args = list(argv)
            else:
                env = {}
                args = [url, "--stderr", str(self.stub.err), "--dir", str(self.tmp / "out")] + list(argv)
                if pid:
                    args += ["--pid", pid_s]
            args += ["--timeout", "20"]
            self.rc, self.lines, self.raised = call(main, args, env)
            self.by, self.order, self.rc_line = parse(self.lines)
        finally:
            self.stub.close()

    def chats(self):
        f = self.tmp / "stub-srv-chats"
        return [x.split() for x in f.read_text().splitlines()] if f.exists() else []


def call(main, argv, env):
    out = io.StringIO()
    rc, raised = None, None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = main(argv, env)
    except SystemExit as e:
        rc = e.code
    except Exception as e:
        raised = f"{type(e).__name__}: {e}"
    return rc, out.getvalue().splitlines(), raised


CASES = {}


def case(fn):
    CASES[fn.__name__.replace("_", "-")] = fn
    return fn


def verdicts(run, **want):
    """Problems of `run` against `name=STATUS` wishes, and that the run did not raise."""
    out = [f"raised {run.raised}"] if run.raised else []
    for name, status in want.items():
        got = run.by.get(name, ("missing", ""))[0]
        if got != status:
            out.append(f"{name} is {got}, want {status}")
    return out


def rc_problem(r, want):
    return [] if r.rc_line == want and r.rc == want else [f"SESSION rc {r.rc_line}, exit {r.rc}, want {want}"]


@case
def healthy_server_through_the_environment(main):
    r = Run(main, {"STUB_SRV_PLAN": "16384"}, ["--place", "a"], env_style=True)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if r.order != ["pass_short", "pass_long", "second_request_equals_first"]:
        p.append(f"checks {r.order}: an off_reason or a tape on a server that is up")
    if "session-plan: place=a ctx_max=16384 load_ctx=16384" not in r.lines:
        p.append(f"no session-plan line from the plan and load records: {r.lines[:2]}")
    ch = r.chats()
    if len(ch) != 3 or ch[0][2] != ch[2][2] or ch[0][2] == ch[1][2] or {c[1] for c in ch} != {"128"}:
        p.append(f"the requests are not short, long, the short again at 128 tokens: {ch}")
    kept = sorted(f.name for f in (r.tmp / "out").glob("*.json"))
    if kept != ["pass_long.json", "pass_short.json", "second_request_equals_first.json"]:
        p.append(f"the requests kept in --dir: {kept}")
    return p


@case
def reasoning_only_reply(main):
    r = Run(main, {"STUB_SRV_CHAT_MODE": "reasoning"})
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if "reasoning_content" not in r.by.get("pass_short", ("", ""))[1]:
        p.append("pass_short's detail does not name reasoning_content")
    return p


@case
def both_empty_reply(main):
    p = []
    for mode in ("empty", "null"):
        r = Run(main, {"STUB_SRV_CHAT_MODE": mode})
        p += [f"{mode}: {x}" for x in verdicts(r, pass_short="FAIL", pass_long="FAIL", second_request_equals_first="SKIP")
              + rc_problem(r, 1)]
    return p


@case
def long_prompt_under_4096_tokens(main):
    r = Run(main, {"STUB_SRV_TOKENIZE_N": "4061"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="PASS") + rc_problem(r, 1)
    detail = r.by.get("pass_long", ("", ""))[1]
    if not detail.startswith("harness:") or "4061" not in detail:
        p.append(f"pass_long's detail is not a `harness:` line naming 4061: {detail!r}")
    if len(r.chats()) != 2:
        p.append(f"the long prompt was sent anyway: {len(r.chats())} chat requests, want 2")
    r = Run(main, {"STUB_SRV_TOKENIZE_N": "4096"})
    p += [f"4096: {x}" for x in verdicts(r, pass_long="PASS") + rc_problem(r, 0)]
    return p


@case
def second_request_500(main):
    r = Run(main, {"STUB_SRV_FAIL_AT": "3"})
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    if "500" not in r.by.get("second_request_equals_first", ("", ""))[1]:
        p.append("second_request_equals_first's detail does not name the 500")
    return p


@case
def a_fail_does_not_stop_the_next_check(main):
    r = Run(main, {"STUB_SRV_FAIL_AT": "2"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="PASS") + rc_problem(r, 1)
    if r.order[:3] != ["pass_short", "pass_long", "second_request_equals_first"]:
        p.append(f"order {r.order}")
    return p


@case
def a_dead_server_skips_the_rest_naming_why(main):
    r = Run(main, {"STUB_SRV_DIE_AT": "2"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="SKIP", off_reason="PASS")
    p += rc_problem(r, 1)
    for name in ("second_request_equals_first", "off_reason"):
        if "engine fault at request 2" not in r.by.get(name, ("", ""))[1]:
            p.append(f"{name}'s detail does not name the server's last words")
    return p


@case
def a_replay_that_differs_fails(main):
    r = Run(main, {"STUB_SRV_DRIFT": "1"})
    return verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)


REFUSAL = "--ctx 16384 over 1 slot(s) asks 16384 positions a slot, and the card holds 11502 (stub)"


@case
def an_off_cell_reads_the_refusal_not_the_banner(main):
    r = Run(main, {"STUB_SRV_BANNER": "1", "STUB_SRV_PLAN": "16384", "STUB_SRV_REFUSE": REFUSAL}, ["--place", "a"])
    p = verdicts(r, pass_short="SKIP", pass_long="SKIP", second_request_equals_first="SKIP", off_reason="PASS")
    p += rc_problem(r, 0)
    if r.by.get("off_reason", ("", ""))[1] != REFUSAL:
        p.append(f"off_reason's detail is {r.by.get('off_reason', ('', ''))[1]!r}, want the refusal text")
    if "session-plan: place=a ctx_max=16384" not in r.lines:
        p.append(f"the refused server's plan record is not read, or its load is: {r.lines[:1]}")
    for line in r.lines:
        if "first start" in line or "place unset" in line or "plan place" in line:
            p.append(f"a banner or record line in the output: {line!r}")
    return p


@case
def an_off_cell_with_only_the_banner_is_unnamed(main):
    r = Run(main, {"STUB_SRV_BANNER": "1", "STUB_SRV_EXIT": "1"})
    p = verdicts(r, pass_short="SKIP", off_reason="FAIL") + rc_problem(r, 1)
    if "first start" in r.by.get("off_reason", ("", ""))[1]:
        p.append("the banner is given as the reason")
    return p


@case
def toktape_client_records_the_tapes_after_the_checks(main):
    fake = scratch("pass-checks-tt.") / "toktape"
    log = fake.parent / "argv"
    fake.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\necho \'{{"timings": {{"prompt_n": 4500}}}}\'\nexit ${{FAKE_TOKTAPE_RC:-0}}\n')
    fake.chmod(0o755)
    extra = ["--client", "toktape", "--toktape-bin", str(fake), "--tag", "rel"]
    p = []
    r = Run(main, argv=extra)
    p += verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS", tape_short="PASS",
                  tape_long="PASS") + rc_problem(r, 0)
    calls = log.read_text().splitlines() if log.exists() else []
    if len(calls) != 2 or not all(
            re.match(r"record --url http://127\.0\.0\.1:\d+ --temp 0 --no-think --endpoint chat -n 128 --for 0 "
                     r"--wait 60s --host-label none --prompts \S+ --out \S+ --tag rel-(short|long) -o json --quiet$", c)
            for c in calls):
        p.append(f"toktape's argv is not the release pass's: {calls}")
    if "prompt tokens" not in r.by.get("tape_long", ("", ""))[1]:
        p.append("tape_long's detail reads no prompt count from toktape's summary")
    os.environ["FAKE_TOKTAPE_RC"] = "3"
    try:
        r = Run(main, argv=extra)
    finally:
        del os.environ["FAKE_TOKTAPE_RC"]
    p += [f"rc 3: {x}" for x in verdicts(r, pass_short="PASS", tape_short="FAIL", tape_long="FAIL") + rc_problem(r, 1)]
    if "toktape exit 3" not in r.by.get("tape_long", ("", ""))[1]:
        p.append("tape_long's detail does not name toktape's exit status")
    r = Run(main, {"STUB_SRV_TOKENIZE_N": "4061"}, argv=extra)
    p += [f"4061: {x}" for x in verdicts(r, pass_long="FAIL", tape_short="PASS", tape_long="SKIP")]
    return p


OFF_LOG = """bloomery-serve: a first start on a new card can compile the GPU code for it (tens of seconds); later starts take seconds
place unset place=bp why=set tier_experts=0 break_even=0 basis=none
bloomery-serve-ds41: place=bp ctx=65536 listening on http://127.0.0.1:18080
bloomery-serve: tier card 3090 (tier 0) holds no expert
  caused by: the cards before it hold every routed expert
"""
LISTENED_THEN_PANICKED = """bloomery-serve: a first start on a new card can compile the GPU code for it (tens of seconds); later starts take seconds
bloomery-serve-ds41: place=bp ctx=65536 listening on http://127.0.0.1:18080
thread 'main' panicked at crates/serve/src/engine.rs:1:1
"""


@case
def off_reason_command_reads_the_refusal(main):
    tmp = scratch("pass-checks-or.")
    (tmp / "refused").write_text(OFF_LOG)
    (tmp / "panicked").write_text(LISTENED_THEN_PANICKED)
    p = []
    for name, want_rc, want_out in (
            ("refused", 0, ["tier card 3090 (tier 0) holds no expert / caused by: the cards before it hold every routed expert"]),
            ("panicked", 1, [])):
        rc, lines, raised = call(main, ["off-reason", "--stderr", str(tmp / name)], {})
        if raised or rc != want_rc or lines != want_out:
            p.append(f"{name}: exit {rc}, output {lines}, raised {raised}; want {want_rc}, {want_out}")
    return p


@case
def a_server_down_with_no_stderr_is_a_harness_fail(main):
    rc, lines, raised = call(main, [f"http://127.0.0.1:{closed_port()}", "--timeout", "5"], {})
    by, _, rc_line = parse(lines)
    detail = by.get("off_reason", ("missing", ""))[1]
    p = [f"raised {raised}"] if raised else []
    if by.get("off_reason", ("",))[0] != "FAIL" or not detail.startswith("harness:") or rc != 1 or rc_line != 1:
        p.append(f"off_reason {by.get('off_reason')}, exit {rc}, SESSION rc {rc_line}")
    return p


@case
def usage_errors_exit_64_and_a_missing_file_2(main):
    p = []
    for argv in ([], ["http://127.0.0.1:1", "--client", "toktape"], ["http://127.0.0.1:1", "--bogus"]):
        rc, _, raised = call(main, argv, {})
        if rc != 64:
            p.append(f"{argv}: exit {rc} {raised or ''}, want 64")
    rc, _, _ = call(main, ["http://127.0.0.1:1", "--stderr", "/nonexistent/stderr"], {})
    if rc != 2:
        p.append(f"a --stderr that is no file: exit {rc}, want 2")
    cli = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve())], capture_output=True, text=True,
                         env={"PATH": os.environ.get("PATH", "")})
    if cli.returncode != 64 or "no URL" not in cli.stderr:
        p.append(f"the script with no URL: exit {cli.returncode}, stderr {cli.stderr!r}")
    return p


def run_cases(main, only=None):
    problems = {}
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        got = fn(main)
        if got:
            problems[name] = got
    return problems


# Each mutant puts back, in the code above this line, what one of the pass scripts this replaces did; the case named
# beside it must fail on it. An anchor that no longer matches fails the self-test by name.
MUTANTS = (
    ("takes any HTTP status", (("if status != 200:", "if False:", 1),), "second-request-500"),
    ("reads content only", (("text = content if content.strip() else reasoning", "text = content", 1),),
     "reasoning-only-reply"),
    ("sends a long prompt of any size", (("if tokens < LONG_MIN_TOKENS:", "if False:", 1),),
     "long-prompt-under-4096-tokens"),
    ("stops at the first FAIL", (("if self.dead:", "if self.dead or self.fails:", 1),),
     "a-fail-does-not-stop-the-next-check"),
    ("never finds the server dead", (("self.dead = self.probe_down()", "self.dead = None", 2),),
     "a-dead-server-skips-the-rest-naming-why"),
    ("reads the log's last five lines as the reason",
     (("def refusal_of(text):", 'def refusal_of(text):\n    return head(" ".join(text.splitlines()[-5:]), 400)\n\n\ndef refusal_of_whole(text):', 1),),
     "an-off-cell-reads-the-refusal-not-the-banner"),
    ("takes a record for the refusal", (("not BANNER.match(line) and not is_rec(line)", "not BANNER.match(line)", 1),), "off-reason-command-reads-the-refusal"),
    ("compares no text", (("if text != want:", "if False:", 1),), "a-replay-that-differs-fails"),
    ("ends 0 on a FAIL", (("rc = 1 if self.fails else 0", "rc = 0", 1),), "second-request-500"),
)


def mutant_main(pairs):
    text = Path(__file__).read_text()
    cut = text.index("# Each mutant puts back")
    src, rest = text[:cut], text[cut:]
    for old, new, count in pairs:
        if src.count(old) != count:
            raise AssertionError(f"mutant anchor {old!r} is in the code {src.count(old)} times, want {count}")
        src = src.replace(old, new)
    src += rest
    ns = {"__file__": str(Path(__file__).resolve()), "__name__": "pass_checks_mutant"}
    exec(compile(src, "<mutant>", "exec"), ns)
    return ns["main"]


def self_test():
    try:
        return self_test_run()
    finally:
        for d in SCRATCH:
            shutil.rmtree(d, ignore_errors=True)


def self_test_run():
    problems = run_cases(main)
    for name, p in problems.items():
        for x in p:
            print(f"FAIL {name}: {x}")
    if problems:
        print(f"pass_checks self-test: {len(problems)} of {len(CASES)} cases FAILED")
        return 1
    bad = 0
    for what, pairs, name in MUTANTS:
        red = run_cases(mutant_main(pairs), only={name}).get(name)
        print(f"mutant {what}: {name} {'red: ' + head(red[0], 110) if red else 'GREEN'}")
        bad += not red
    if bad:
        print(f"pass_checks self-test: {bad} of {len(MUTANTS)} mutants left their case green")
        return 1
    print(f"pass_checks self-test: {len(CASES)} cases ok, {len(MUTANTS)} mutants red")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
