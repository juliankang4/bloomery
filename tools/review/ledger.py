#!/usr/bin/env python3
"""The review bot's slag ledger: one open issue labelled LABEL whose body lists what the bot has learned. The
loop runs without people: a lesson goes into force when the learner proposes it, keeps itself alive only by measured
use, and a maintainer's one move is a veto.

    tools/review/ledger.py accepted --repo OWNER/NAME --out FILE        (GH_TOKEN, read)
    tools/review/ledger.py fetch    --repo OWNER/NAME --out FILE        (GH_TOKEN, read)
    tools/review/ledger.py update   --repo OWNER/NAME --candidates FILE --stats FILE --usage FILE
                                    [--secret-env NAME]... [--dry-run FILE]  (GH_TOKEN, issues write)
    tools/review/ledger.py --self-test

A lesson is one line, `- [x] **L<n>** · <kind> · until <date> · used <n> · <title>: <rule> — evidence: <links>`.
Ticked and not struck through, it is in force: every review reads it (`accepted`, at most MAX_ACCEPTED). Kinds:
- `escape` (found from a fix the review missed) adds scrutiny. Its use is a finding it raised that a later commit of
  the pull request answered (`acted` in lesson-stats.json).
- `noise` (found from findings the pull requests ignored or the validator removed) takes scrutiny away, so it is
  bounded by code here: it may only drop ⚠ items (the reviewer instructions say so), its evidence names at least two
  pull requests, and its text may not touch GUARDED (security, credentials, unsafe, gates, lints, silent failure,
  must-fix). Its use is a finding it suppressed.

Every update renews a lesson in force whose `until` has passed when it was used since the last renewal (until +
LIFE_DAYS, used back to 0), and strikes it through as expired when it was not. `used` adds up the period's use.
A maintainer vetoes a lesson by striking it through or pauses it by unticking it; `update` copies every line it does
not renew or expire exactly as it is in the issue just before the write, so a veto made while the learner ran stays,
and a struck or unticked line is never touched again. New candidates (at most MAX_NEW) enter ticked, after the
checks above, a check for credentials (exit 3, nothing written) and a check that the title and evidence are not
already in the ledger. A malformed candidates file is exit 65.

`accepted` writes the reviewer's view to FILE and prints the ledger version (a hash of those lines, `none` without a
ledger). The ledger is found by label, by the marker in its body and by an author in the trusted set (the bot or the
repository owner); anything else is ignored. `fetch` writes the current body for the learner.
"""

import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post import TOKEN_RE, api  # noqa: E402

LABEL = "review-ledger"
MARK = "<!-- bloomery-review-ledger v1 -->"
TITLE = "Shingler's slag ledger"
TRUSTED_EXTRA = ("github-actions[bot]",)
KINDS = ("escape", "noise")
USE = {"escape": "acted", "noise": "suppressed"}
LIFE_DAYS = 28
MAX_ACCEPTED = 30
MAX_NEW = 3
MAX_RUNS = 8
GUARDED = re.compile(r"(?i)secur|credential|secret|token|api.?key|passw|unsafe|safety|\bgate|\blint|\bpin\b|pin\(|"
                     r"toleran|\bband|relax|silent|panic|fault|must.?fix|kind a|kind b|inject|untrusted|escape catalog")
LESSON_RE = re.compile(r"^- \[([ xX])\] (.*\*\*L(\d+)\*\*.*)$")
META_RE = re.compile(r"· (escape|noise) · until (\d{4}-\d\d-\d\d) · used (\d+) ·")
RUNS_HEAD = "## Learner runs"

INTRO = f"""{MARK}
The slag Shingler, the review bot, has learned to hammer out of this repository's pull requests. Every review reads the ticked lines that are not
struck through (at most {MAX_ACCEPTED}). The weekly learner (`.github/workflows/review-learn.yml`) runs this list on its
own; nobody has to approve anything.

- A new lesson is in force at once for {LIFE_DAYS} days. It renews itself only when it was used in that time (an
  `escape` lesson's finding that a later commit answered, a `noise` lesson's suppressed ⚠), and expires otherwise.
- A `noise` lesson can only drop ⚠ items, needs two pull requests as evidence, and may not touch security,
  credentials, unsafe code, gates, lints, silent failure or must-fix items.
- To veto a lesson, strike it through (`~~…~~`); to pause it, untick it. The learner never touches either again.
- A lesson that keeps renewing can move to `.github/review/escape-catalog.md`; delete its line here then.

## Lessons
"""


def lessons(body):
    """[(ticked, text, n)] for every lesson line of body, in order."""
    out = []
    for line in (body or "").splitlines():
        m = LESSON_RE.match(line.rstrip())
        if m:
            out.append((m.group(1) in "xX", m.group(2), int(m.group(3))))
    return out


def in_force(ticked, text):
    return ticked and "~~" not in text


def accepted(body):
    return [text for ticked, text, _ in lessons(body) if in_force(ticked, text)][:MAX_ACCEPTED]


def version(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12] if lines else "none"


def clean(s, cap, fields=True):
    s = re.sub(r"\s+", " ", str(s or ""))
    # No checkbox, lesson id, strike or comment syntax; in a lesson's text, no field separator either.
    s = re.sub(r"\[[ xX]\]|\*\*|~~|<!--|-->" + ("|·" if fields else ""), "", s)
    return re.sub(r"\s+", " ", s).strip()[:cap].rstrip()


def norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def keep_candidates(items, repo, body):
    """The candidates update may add, and how many it dropped (malformed, guarded, or already in the ledger)."""
    if not isinstance(items, list):
        raise ValueError("candidates is not a JSON array")
    link_re = re.compile(rf"^https://github\.com/{re.escape(repo)}/(pull|issues|commit)/[A-Za-z0-9#/_.=-]+$")
    pr_re = re.compile(rf"^https://github\.com/{re.escape(repo)}/pull/(\d+)")
    have = set()
    for _, text, _ in lessons(body):
        m = re.search(r"· used \d+ · (.*?):", text) or re.search(r"· (?:escape|noise) · (.*?):", text)
        if m:
            have.add(norm_title(m.group(1)))
    kept, dropped = [], 0
    for it in items:
        if not isinstance(it, dict) or it.get("kind") not in KINDS:
            dropped += 1
            continue
        title, rule = clean(it.get("title"), 120), clean(it.get("rule"), 500)
        ev = [e for e in (it.get("evidence") or []) if isinstance(e, str) and link_re.match(e)][:4]
        prs = {pr_re.match(e).group(1) for e in ev if pr_re.match(e)}
        bad = not title or not rule or not ev or norm_title(title) in have or any(e in (body or "") for e in ev)
        if it["kind"] == "noise" and (len(prs) < 2 or GUARDED.search(title + " " + rule)):
            bad = True
        if bad:
            dropped += 1
            continue
        have.add(norm_title(title))
        kept.append({"kind": it["kind"], "title": title, "rule": rule, "evidence": ev})
    dropped += max(0, len(kept) - MAX_NEW)
    return kept[:MAX_NEW], dropped


def age(body, stats, today):
    """The lesson lines after this run's use: (lines, renewed ids, expired ids)."""
    out, renewed, expired = [], [], []
    for line in (body or "").splitlines():
        m = LESSON_RE.match(line.rstrip())
        if not m:
            continue
        ticked, text, n = m.group(1) in "xX", m.group(2), f"L{m.group(3)}"
        meta = META_RE.search(text)
        if not in_force(ticked, text) or not meta:
            out.append(line.rstrip())  # vetoed, paused or hand-written: as it is
            continue
        kind, until, used = meta.group(1), dt.date.fromisoformat(meta.group(2)), int(meta.group(3))
        used += int((stats.get(n) or {}).get(USE[kind], 0))
        if today >= until and used == 0:
            out.append(f"- [ ] ~~{text}~~ (expired {today}: no measured use)")
            expired.append(n)
            continue
        if today >= until:
            until, used = today + dt.timedelta(days=LIFE_DAYS), 0
            renewed.append(n)
        out.append("- [x] " + text.replace(meta.group(0), f"· {kind} · until {until} · used {used} ·", 1))
    return out, renewed, expired


def render(body, new, stats, today, run_line):
    lines, renewed, expired = age(body, stats, today)
    n = max([k for _, _, k in lessons(body)], default=0)
    for it in new:
        n += 1
        lines.append(f"- [x] **L{n}** · {it['kind']} · until {today + dt.timedelta(days=LIFE_DAYS)} · used 0 · "
                     f"{it['title']}: {it['rule']} — evidence: {', '.join(it['evidence'])}")
    runs = []
    if body and RUNS_HEAD in body:
        runs = [x for x in body.split(RUNS_HEAD, 1)[1].splitlines() if x.startswith("- ")]
    if run_line:
        tail = "".join(f" · {k} {', '.join(v)}" for k, v in (("new", [f"L{n - len(new) + i + 1}" for i in
                       range(len(new))]), ("renewed", renewed), ("expired", expired)) if v)
        runs = [f"- {run_line}{tail}"] + runs
    out = INTRO.splitlines() + [""] + (lines or ["(none yet)"]) + ["", RUNS_HEAD, ""] + runs[:MAX_RUNS]
    return "\n".join(out) + "\n"


def find(repo, token, call=api):
    """The ledger issue, or None: open, labelled LABEL, with MARK in its body, opened by a trusted author."""
    trusted = set(TRUSTED_EXTRA) | {repo.split("/")[0]}
    got = call("GET", f"https://api.github.com/repos/{repo}/issues?labels={LABEL}&state=open&per_page=100", token)
    for i in sorted(got, key=lambda i: i["number"]):
        if MARK in (i.get("body") or "") and (i.get("user") or {}).get("login") in trusted:
            return i
    return None


def read(path, default):
    try:
        return open(path, encoding="utf-8").read() if path and os.path.exists(path) else default
    except OSError:
        return default


def update(repo, token, cand_path, stats_path, usage_path=None, secret_env=(), dry=None, call=api, today=None):
    today = today or dt.datetime.now(dt.timezone.utc).date()
    raw = read(cand_path, "[]")
    try:
        items = json.loads(raw or "[]")
        stats = json.loads(read(os.path.join(os.path.dirname(stats_path), "lesson-stats.json"), "{}") or "{}")
    except ValueError as e:
        print(f"update: not JSON: {e}", file=sys.stderr)
        return 65
    secrets = [os.environ.get(n, "") for n in secret_env]
    if TOKEN_RE.search(raw) or any(s and s in raw for s in secrets):
        print("update: a candidate carries a credential; the ledger is not written", file=sys.stderr)
        return 3
    run_line = clean(read(stats_path, "").strip().splitlines()[0] if read(stats_path, "").strip() else "", 400, False)
    try:
        usage = json.loads(read(usage_path, "{}") or "{}")
    except ValueError:
        usage = {}
    cost = usage.get("cost_usd")
    # A Codex run on a ChatGPT login has no price: it is named instead.
    run_line += (f" · learner ${cost:.2f}" if isinstance(cost, (int, float))
                 else f" · learner {usage['engine']}" if usage.get("subtype") == "success" and usage.get("engine")
                 else " · learner not run")
    # Read the issue as late as possible, so a veto made while the learner ran is the one copied.
    issue = find(repo, token, call)
    body = issue["body"] if issue else ""
    try:
        new, dropped = keep_candidates(items, repo, body)
    except ValueError as e:
        print(f"update: {e}", file=sys.stderr)
        return 65
    nxt = render(body, new, stats, today, run_line + (f" · {dropped} candidate(s) dropped" if dropped else ""))
    if dry:
        open(dry, "w", encoding="utf-8").write(nxt)
    elif issue:
        call("PATCH", f"https://api.github.com/repos/{repo}/issues/{issue['number']}", token, {"body": nxt})
    else:
        try:
            call("POST", f"https://api.github.com/repos/{repo}/labels", token,
                 {"name": LABEL, "color": "5319e7", "description": "Shingler's slag ledger (the review bot's lessons)"})
        except Exception:  # the label exists
            pass
        issue = call("POST", f"https://api.github.com/repos/{repo}/issues", token,
                     {"title": TITLE, "body": nxt, "labels": [LABEL]})
    print(f"update: ledger #{issue['number'] if issue else '-'} new={len(new)} dropped={dropped}")
    return 0


def self_test():
    repo, day = "o/r", dt.date(2026, 10, 19)
    body = render("", [], {}, day, "run 1")
    assert MARK in body and "(none yet)" in body and lessons(body) == [], body
    esc = {"kind": "escape", "title": "Exit code after a pipe", "rule": "flag `x | tail` in a script that reports rc",
           "evidence": ["https://github.com/o/r/commit/abc", "https://evil.example/x"]}
    c = [esc,
         {"kind": "noise", "title": "Doc wording", "rule": "do not flag doc sentence style",
          "evidence": ["https://github.com/o/r/pull/1"]},                                   # one pull request
         {"kind": "noise", "title": "Unsafe blocks", "rule": "stop flagging unsafe without SAFETY",
          "evidence": ["https://github.com/o/r/pull/1", "https://github.com/o/r/pull/2"]},  # guarded
         {"kind": "noise", "title": "Doc wording - [x] **L9** · x", "rule": "do not flag doc sentence style",
          "evidence": ["https://github.com/o/r/pull/1", "https://github.com/o/r/pull/2"]},
         {"kind": "praise", "title": "a", "rule": "b", "evidence": ["https://github.com/o/r/pull/8"]}]
    new, dropped = keep_candidates(c, repo, body)
    assert [n["title"] for n in new] == ["Exit code after a pipe", "Doc wording - L9 x"] and dropped == 3, (new, dropped)
    assert new[0]["evidence"] == ["https://github.com/o/r/commit/abc"]
    b2 = render(body, new, {}, day, "run 2")
    assert [(t, n) for t, _, n in lessons(b2)] == [(True, 1), (True, 2)] and len(accepted(b2)) == 2, b2
    assert "until 2026-11-16 · used 0" in b2 and "new L1, L2" in b2 and b2.index("run 2") < b2.index("run 1"), b2
    # A veto (strike) and a pause (untick) stay; a lesson in force renews when used and expires when not.
    vet = b2.replace("- [x] **L2**", "- [ ] **L2**")
    b3 = render(vet, [], {"L1": {"acted": 1}}, dt.date(2026, 11, 1), "run 3")
    assert "**L1** · escape · until 2026-11-16 · used 1 ·" in b3 and "- [ ] **L2**" in b3, b3
    b4 = render(b3, [], {}, dt.date(2026, 11, 16), "run 4")
    assert "**L1** · escape · until 2026-12-14 · used 0 ·" in b4 and "renewed L1" in b4, b4
    b5 = render(b4, [], {}, dt.date(2026, 12, 14), "run 5")
    assert "- [ ] ~~**L1**" in b5 and "(expired 2026-12-14: no measured use)" in b5 and accepted(b5) == [], b5
    b6 = render(b5, [], {"L1": {"acted": 5}}, dt.date(2027, 1, 1), "run 6")
    assert b6.split("## Lessons")[1].split(RUNS_HEAD)[0] == b5.split("## Lessons")[1].split(RUNS_HEAD)[0]
    struck = b2.replace("- [x] **L1** · escape · until 2026-11-16 · used 0 · Exit code after a pipe",
                        "- [x] ~~**L1** · escape · until 2026-11-16 · used 0 · Exit code after a pipe~~")
    assert "~~" in render(struck, [], {}, dt.date(2026, 12, 1), "r").split(RUNS_HEAD)[0].split("**L2**")[0]
    again, d = keep_candidates([esc], repo, b2)
    assert again == [] and d == 1  # same title and evidence
    calls = []

    def fake(method, url, token, payload=None):
        calls.append((method, url))
        if method == "GET":
            return [{"number": 5, "user": {"login": "mallory"}, "body": b2},
                    {"number": 9, "user": {"login": "o"}, "body": b2}]
        return {"number": 9}
    with tempfile.TemporaryDirectory() as d:
        cp, sp, up, out = (os.path.join(d, x) for x in ("c.json", "stats.md", "u.json", "out.md"))
        json.dump([c[3]], open(cp, "w"))
        open(sp, "w").write("2026-10-19 reviews=1\n")
        json.dump({"L1": {"acted": 1}}, open(os.path.join(d, "lesson-stats.json"), "w"))
        json.dump({"cost_usd": 0.1984}, open(up, "w"))
        assert update(repo, "t", cp, sp, up, dry=out, call=fake, today=day) == 0
        o = open(out).read()
        assert "2026-10-19 reviews=1 · learner $0.20 · 1 candidate(s) dropped" in o and "used 1" in o, o
        open(sp, "w").write("2026-10-19 · reviews 1 · ~~x~~\n")
        assert update(repo, "t", cp, sp, up, dry=out, call=fake, today=day) == 0
        assert "- 2026-10-19 · reviews 1 · x · learner $0.20" in open(out).read()
        json.dump({"cost_usd": None, "engine": "codex", "subtype": "success"}, open(up, "w"))
        assert update(repo, "t", cp, sp, up, dry=out, call=fake, today=day) == 0
        assert "· x · learner codex" in open(out).read()
        assert update(repo, "t", cp, sp, call=fake, today=day) == 0
        assert calls[-1] == ("PATCH", "https://api.github.com/repos/o/r/issues/9")
        json.dump([{"kind": "escape", "title": "k", "rule": "sk-ant-" + "a" * 20, "evidence": []}], open(cp, "w"))
        assert update(repo, "t", cp, sp, dry=out, call=fake) == 3
        open(cp, "w").write("{not json")
        assert update(repo, "t", cp, sp, dry=out, call=fake) == 65
    assert find(repo, "t", lambda *a, **k: [{"number": 5, "user": {"login": "mallory"}, "body": MARK}]) is None
    print("ledger.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) < 2 or argv[1] not in ("accepted", "fetch", "update"):
        print(__doc__, file=sys.stderr)
        return 64
    opts, secret_env, rest = {}, [], argv[2:]
    while rest:
        k = rest.pop(0)
        if not rest:
            print(f"{k} needs a value", file=sys.stderr)
            return 64
        v = rest.pop(0)
        if k == "--secret-env":
            secret_env.append(v)
        else:
            opts[k] = v
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print(f"{argv[1]}: GH_TOKEN is not set", file=sys.stderr)
        return 64
    try:
        repo = opts["--repo"]
        if argv[1] == "update":
            return update(repo, token, opts.get("--candidates"), opts["--stats"], opts.get("--usage"), secret_env,
                          opts.get("--dry-run"))
        issue = find(repo, token)
        body = issue["body"] if issue else ""
        if argv[1] == "fetch":
            open(opts["--out"], "w", encoding="utf-8").write(body)
            print(f"fetch: ledger #{issue['number'] if issue else '-'}, {len(lessons(body))} lesson(s)")
            return 0
        lines = accepted(body)
        with open(opts["--out"], "w", encoding="utf-8") as f:
            if lines:
                f.write("# Lessons in force (the slag ledger issue)\n\n")
                f.write("\n".join(f"- {x}" for x in lines) + "\n")
            else:
                f.write("No lessons in force.\n")
        print(version(lines))
        return 0
    except KeyError as e:
        print(f"missing option {e}", file=sys.stderr)
        return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
