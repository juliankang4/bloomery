#!/usr/bin/env python3
"""What the review bot's weekly learner reads (.github/workflows/review-learn.yml). Every signal is an outcome the
machine can see; no person has to react to anything.

    tools/review/signals.py --repo OWNER/NAME --git DIR --days N --out DIR [--lookback DAYS] [--ref REF]
                            [--all-merged 1]                                     (GH_TOKEN; `gh` for artifacts)
    tools/review/signals.py --self-test

- escapes: a commit on REF in the last N days whose subject names a fix (FIX_RE) removed or changed lines that, by
  `git blame` at its parent, came from a pull request the bot reviewed. A pull request's commits on REF
  (pr_commits): a merge commit's second-parent side, a rebase's first-parent run whose subjects are the pull
  request's, or a squash commit alone. Each escape gets its patch (escapes/<sha>.patch).
- outcomes: for each pull request merged or closed in the window, every finding of every bot review on it (a must-fix
  item, a ⚠ item, an inline comment; its first link to a line of the reviewed head) is `acted` when a later commit of
  the pull request changed that line (within LINE_SLACK), `ignored` when the pull request merged with the line as it
  was, and left out when it closed unmerged or the comparison is not available.
- removals and lesson use: the `review` artifact of each review's run (the footer's `run=`) holds the validator's
  removed.json (findings it removed, with the reason) and the reviewer's lessons.json (ledger lessons that raised or
  suppressed a finding). A raised finding takes its outcome from the line it cites.
- tool aims: the same artifact's aims.json names the findings dryer or crapper pointed the reviewer at; each takes
  the outcome of the line it cites, so the stats line says whether each tool earns its place in this repository.

Writes OUT/signals.md (the learner's index, with reviews/<pr>.md), OUT/lesson-stats.json ({"L3": {"raised",
"acted", "suppressed"}} for the window), OUT/stats.md (one line for the ledger, with each tool's
acted and ignored counts) and OUT/has_signals.txt (`true` when
there is an escape, an ignored finding or a removal, so the workflow can skip the model run otherwise).
`--all-merged 1` treats every merged pull request as reviewed, to try the escape search on history.
"""

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post import MARKER, api  # noqa: E402

BOT = "github-actions[bot]"
FIX_RE = re.compile(r"(?i)\b(fix|fixes|bug|regress\w*|broke\w*|wrong|crash\w*|revert\w*)\b")
MAX_FIXES = 120
MAX_ESCAPES = 10
MAX_LISTED = 40
LINE_SLACK = 2
TOOLS = ("dryer", "crapper")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
FOOT_RE = re.compile(rf"<!-- {MARKER} head_sha=(\w+) verdict=(\w+) must_fix=(\d+) warn=(\d+)")
RUN_RE = re.compile(rf"<!-- {MARKER} head_sha=[^>]* run=(\d+)")
COST_RE = re.compile(r"cost_usd=([0-9.]+)")


def git(d, *args):
    return subprocess.run(["git", "-C", d, *args], capture_output=True, text=True, check=True).stdout


def old_ranges(patch):
    """{old path: [(start, count)]} for the lines a unified diff removes or changes (count 0: a pure insertion after
    start)."""
    out, path = {}, None
    for line in patch.splitlines():
        if line.startswith("--- "):
            path = None if line[4:] == "/dev/null" else line[4:].removeprefix("a/")
        m = HUNK_RE.match(line)
        if m and path:
            out.setdefault(path, []).append((int(m.group(1)), 1 if m.group(2) is None else int(m.group(2))))
    return out


def blamed(d, commit, ranges):
    """{origin sha: [(path, line)]} for the given old-side ranges, blamed at commit's parent."""
    out = {}
    for path, rs in ranges.items():
        for start, n in rs:
            if not n:
                continue
            try:
                text = git(d, "blame", "--porcelain", "-L", f"{start},+{n}", f"{commit}^", "--", path)
            except subprocess.CalledProcessError:
                continue
            for line in text.splitlines():
                m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
                if m:
                    out.setdefault(m.group(1), []).append((path, int(m.group(2))))
    return out


def pr_commits(d, merge_sha, subjects):
    """The commits a merged pull request put on the branch (see the module doc)."""
    parents = git(d, "rev-list", "--parents", "-n", "1", merge_sha).split()[1:]
    if len(parents) == 2:
        return [merge_sha] + git(d, "rev-list", f"{parents[0]}..{parents[1]}").split()
    run = []
    for line in git(d, "log", "--first-parent", f"--max-count={max(1, len(subjects))}", "--format=%H%x09%s",
                    merge_sha).splitlines():
        sha, subject = line.split("\t", 1)
        if subject not in subjects:
            break
        run.append(sha)
    return run or [merge_sha]


def findings(repo, review_body):
    """[(kind, path, line, text)] for a review body's must-fix and ⚠ items, each by its first line link."""
    link = re.compile(rf"https://github\.com/{re.escape(repo)}/blob/[0-9a-f]{{7,40}}/([^#\s)]+)#L(\d+)")
    out, blocks, cur = [], [], None
    for line in review_body.splitlines():
        if re.match(r"^#### \d+\.", line):
            cur = ["must", line]
            blocks.append(cur)
        elif line.startswith("⚠"):
            cur = ["warn", line]
            blocks.append(cur)
        elif line.startswith("#") or line.startswith("<!--"):
            cur = None
        elif cur is not None:
            cur[1] += "\n" + line
    for kind, text in blocks:
        m = link.search(text)
        if m:
            out.append((kind, urllib.parse.unquote(m.group(1)), int(m.group(2)), re.sub(r"\s+", " ", text)[:400]))
    return out


def line_changed(compare_files, path, line):
    """True when the comparison changed path near line, False when it left it, None when it cannot tell."""
    for f in compare_files:
        if f.get("filename") != path and f.get("previous_filename") != path:
            continue
        if f.get("status") in ("removed", "renamed") and f.get("filename") != path:
            return True
        if "patch" not in f:
            return None  # a binary or an oversized diff
        for start, n in old_ranges("--- a/" + path + "\n" + f["patch"]).get(path, []):
            if start - LINE_SLACK <= line <= start + max(n, 1) - 1 + LINE_SLACK:
                return True
        return False
    return False


def outcome_of(outcomes, pr, rec):
    """The outcome of the finding a lessons.json or aims.json record cites by path and line, or None."""
    for o in outcomes:
        if o["pr"] == pr and o["path"] == rec.get("path") and o["line"] == rec.get("line"):
            return o["outcome"]
    return None


def pages(call, url, token):
    got, page = [], 1
    while True:
        sep = "&" if "?" in url else "?"
        part = call("GET", f"{url}{sep}per_page=100&page={page}", token)
        items = part["items"] if isinstance(part, dict) else part
        got += items
        if len(items) < 100 or page >= 10:
            return got
        page += 1


def gh_download(repo, run_id, dest):
    """The run's `review` artifact into dest; False when it is gone or never existed."""
    r = subprocess.run(["gh", "run", "download", str(run_id), "-R", repo, "-n", "review", "-D", dest],
                       capture_output=True, text=True)
    return r.returncode == 0


def read_json_list(path):
    try:
        v = json.load(open(path, encoding="utf-8"))
        return v if isinstance(v, list) else []
    except (OSError, ValueError):
        return []


def collect(repo, d, days, lookback, ref, token, out, call=api, download=gh_download, now=None, all_merged=False):
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(days=days)
    s19 = since.isoformat()[:19]
    q = urllib.parse.quote(f"repo:{repo} is:pr updated:>={(now - dt.timedelta(days=lookback)).date()}")
    prs = pages(call, f"https://api.github.com/search/issues?q={q}", token)
    for sub in ("reviews", "escapes"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    reviewed, origin, stats, outcomes, removals, lessons, aims = {}, {}, [], [], [], {}, {}
    for p in sorted(prs, key=lambda p: p["number"]):
        n = p["number"]
        revs = [r for r in pages(call, f"https://api.github.com/repos/{repo}/pulls/{n}/reviews", token)
                if (r.get("user") or {}).get("login") == BOT and f"<!-- {MARKER} head_sha=" in (r.get("body") or "")]
        if not revs and not all_merged:
            continue
        reviewed[n] = revs
        with open(os.path.join(out, "reviews", f"{n}.md"), "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(f"Review {r['html_url']} at {r['submitted_at']}\n\n{r['body']}" for r in revs)
                    or "The bot did not review this pull request (a history trial, --all-merged).\n")
        for r in revs:
            if r["submitted_at"] >= s19:
                m = FOOT_RE.search(r["body"])
                stats.append((m.group(2) if m else "?", int(m.group(3)) if m else 0, int(m.group(4)) if m else 0,
                              sum(float(x) for x in COST_RE.findall(r["body"])), "cost_usd=None" in r["body"]))
        pr = call("GET", f"https://api.github.com/repos/{repo}/pulls/{n}", token)
        if pr.get("merged") and pr.get("merge_commit_sha"):
            try:
                git(d, "merge-base", "--is-ancestor", pr["merge_commit_sha"], ref)
                subjects = {c["commit"]["message"].split("\n", 1)[0]
                            for c in pages(call, f"https://api.github.com/repos/{repo}/pulls/{n}/commits", token)}
                for sha in pr_commits(d, pr["merge_commit_sha"], subjects):
                    origin[sha] = n
            except subprocess.CalledProcessError:
                pass  # not on REF
        closed = pr.get("merged_at") or pr.get("closed_at") or ""
        if not revs or closed < s19:
            continue  # outcomes are counted once, in the window the pull request closed in
        merged, final = bool(pr.get("merged")), pr["head"]["sha"]
        inline = [c for c in pages(call, f"https://api.github.com/repos/{repo}/pulls/{n}/comments", token)
                  if (c.get("user") or {}).get("login") == BOT]
        compares = {}
        for r in revs:
            m = FOOT_RE.search(r["body"])
            head = m.group(1) if m else r.get("commit_id")
            items = [(k, path, line, text, r["html_url"]) for k, path, line, text in findings(repo, r["body"])]
            items += [("inline", c["path"], c.get("original_line") or c.get("line"), c.get("body", "")[:400],
                       c["html_url"]) for c in inline if c.get("original_commit_id") == head
                      and (c.get("original_line") or c.get("line"))]
            if head not in compares:
                compares[head] = [] if head == final else None
                if head != final:
                    try:
                        compares[head] = call("GET", f"https://api.github.com/repos/{repo}/compare/{head}...{final}",
                                              token).get("files") or []
                    except Exception:  # a force-pushed head the network no longer has
                        compares[head] = None
            for kind, path, line, text, url in items:
                changed = None if compares[head] is None else line_changed(compares[head], path, line)
                if changed is None or (not changed and not merged):
                    continue
                outcomes.append({"pr": n, "kind": kind, "path": path, "line": line, "text": text, "url": url,
                                 "outcome": "acted" if changed else "ignored"})
            run = RUN_RE.search(r["body"])
            if run:
                with tempfile.TemporaryDirectory() as t:
                    if download(repo, run.group(1), t):
                        art = os.path.join(t, "out")
                        for x in read_json_list(os.path.join(art, "removed.json"))[:20]:
                            if isinstance(x, dict):
                                removals.append({"pr": n, "url": r["html_url"], **{k: str(x.get(k, ""))[:300] for k in
                                                 ("what", "path", "line", "reason", "lesson")}})
                        for x in read_json_list(os.path.join(art, "lessons.json"))[:20]:
                            if not isinstance(x, dict) or not re.fullmatch(r"L\d+", str(x.get("id", ""))):
                                continue
                            s = lessons.setdefault(x["id"], {"raised": 0, "acted": 0, "suppressed": 0})
                            if x.get("effect") == "suppressed":
                                s["suppressed"] += 1
                            elif x.get("effect") == "raised":
                                s["raised"] += 1
                                s["acted"] += outcome_of(outcomes, n, x) == "acted"
                        for x in read_json_list(os.path.join(art, "aims.json"))[:40]:
                            if isinstance(x, dict) and x.get("tool") in TOOLS:
                                a = aims.setdefault(x["tool"], {"aimed": 0, "acted": 0, "ignored": 0})
                                a["aimed"] += 1
                                o = outcome_of(outcomes, n, x)
                                if o:
                                    a[o] += 1
    escapes = []
    log = git(d, "log", f"--since={since.isoformat()}", "--format=%H%x09%s", ref).splitlines()
    fixes = [line.split("\t", 1) for line in log if FIX_RE.search(line.split("\t", 1)[-1])][:MAX_FIXES]
    for sha, subject in fixes:
        try:
            patch = git(d, "diff", "-U0", f"{sha}^", sha)
        except subprocess.CalledProcessError:
            continue
        for osha, where in blamed(d, sha, old_ranges(patch)).items():
            n = origin.get(osha)
            if n is None or origin.get(sha) == n:
                continue  # not from a reviewed pull request, or a fix inside that same pull request
            escapes.append((sha, subject, n, osha, where))
    escapes = escapes[:MAX_ESCAPES]
    for sha, *_ in escapes:
        text = git(d, "show", "--format=commit %H%nSubject: %s%n", sha).splitlines()
        with open(os.path.join(out, "escapes", f"{sha}.patch"), "w", encoding="utf-8") as f:
            f.write("\n".join(text[:600]) + ("\n(cut at 600 lines)\n" if len(text) > 600 else "\n"))
    write(out, repo, since, now, days, reviewed, escapes, outcomes, removals, lessons, aims, stats)
    return len(escapes), len(outcomes), len(removals)


def write(out, repo, since, now, days, reviewed, escapes, outcomes, removals, lessons, aims, stats):
    v = {k: sum(1 for s in stats if s[0] == k) for k in ("CLEAN", "MUST_FIX")}
    uncosted = sum(s[4] for s in stats)  # a Codex run on a ChatGPT login reports no cost
    acted = [o for o in outcomes if o["outcome"] == "acted"]
    ignored = [o for o in outcomes if o["outcome"] == "ignored"]
    line = (f"{now.date()} · last {days} d · reviews {len(stats)} (CLEAN {v['CLEAN']}, MUST_FIX {v['MUST_FIX']}) · "
            f"must-fix {sum(s[1] for s in stats)}, ⚠ {sum(s[2] for s in stats)} · "
            f"review cost ${sum(s[3] for s in stats):.2f}{f' ({uncosted} with no cost)' if uncosted else ''} · "
            f"findings acted {len(acted)}, ignored {len(ignored)} · "
            f"validator removed {len(removals)} · escapes {len(escapes)}"
            + "".join(f" · {t} aimed {a['aimed']} (acted {a['acted']}, ignored {a['ignored']})"
                      for t, a in sorted(aims.items())))
    open(os.path.join(out, "stats.md"), "w", encoding="utf-8").write(line + "\n")
    json.dump(lessons, open(os.path.join(out, "lesson-stats.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(out, "has_signals.txt"), "w").write("true\n" if escapes or ignored or removals else "false\n")
    L = [f"# Signals since {since.date()} ({days} days)", "", line, "",
         f"Pull requests with a bot review in the lookback: {', '.join(f'#{n}' for n in sorted(reviewed)) or 'none'}.",
         "Their reviews are in `reviews/<number>.md`.", "", "## Escapes", ""]
    if not escapes:
        L.append("None: no fix commit in the window changed lines that came from a reviewed pull request.")
    for sha, subject, n, osha, where in escapes:
        locs = ", ".join(f"{p}:{ln}" for p, ln in where[:8])
        L.append(f"- fix https://github.com/{repo}/commit/{sha} ({subject}) changed lines from #{n} "
                 f"(commit {osha[:12]}): {locs}. Patch: `escapes/{sha}.patch`.")
    for title, rows in (("Findings the pull request merged without changing", ignored),
                        ("Findings a later commit of the pull request answered", acted)):
        L += ["", f"## {title} ({len(rows)})", ""]
        L += [f"- #{o['pr']} {o['kind']} {o['path']}:{o['line']} ({o['url']}): {o['text']}" for o in rows[:MAX_LISTED]]
        if not rows:
            L.append("None.")
    L += ["", f"## Findings the validator removed ({len(removals)})", ""]
    L += [f"- #{x['pr']} {x['path']}:{x['line']} ({x['url']}) reason={x['reason']} lesson={x['lesson'] or '-'}: "
          f"{x['what']}" for x in removals[:MAX_LISTED]] or ["None."]
    L += ["", "## Ledger lessons used in the window", ""]
    L += [f"- {k}: raised {s['raised']} (acted {s['acted']}), suppressed {s['suppressed']}"
          for k, s in sorted(lessons.items())] or ["None."]
    open(os.path.join(out, "signals.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def self_test():
    patch = ("diff --git a/x.sh b/x.sh\n--- a/x.sh\n+++ b/x.sh\n@@ -3 +3 @@\n-a\n+b\n@@ -7,2 +6,0 @@\n-c\n-d\n"
             "@@ -9,0 +8,1 @@\n+e\n--- /dev/null\n+++ b/new.sh\n@@ -0,0 +1 @@\n+f\n")
    assert old_ranges(patch) == {"x.sh": [(3, 1), (7, 2), (9, 0)]}, old_ranges(patch)
    files = [{"filename": "a.rs", "status": "modified", "patch": "@@ -40,3 +40,4 @@\n x\n-y\n+z\n+w\n x"},
             {"filename": "b.rs", "status": "modified"}]
    assert line_changed(files, "a.rs", 41) and line_changed(files, "a.rs", 44) and not line_changed(files, "a.rs", 50)
    assert line_changed(files, "b.rs", 1) is None and line_changed(files, "c.rs", 1) is False
    body = ("**Verdict: MUST_FIX**: x\n\n### Must fix\n\n#### 1. Bad rc\n[t.sh:5](https://github.com/o/r/blob/"
            + "a" * 40 + "/t.sh#L5) reads the wrong rc.\n\n### Non-blocking\n⚠ Slow loop: [l.rs:9](https://github.com/"
            "o/r/blob/" + "a" * 40 + "/src/l.rs#L9).\n\n⚠ No link here.\n\n### Checked\n- [z.rs:1](https://github.com/o/"
            "r/blob/" + "a" * 40 + "/z.rs#L1)\n")
    got = [(k, p, ln) for k, p, ln, _ in findings("o/r", body)]
    assert got == [("must", "t.sh", 5), ("warn", "src/l.rs", 9)], got
    with tempfile.TemporaryDirectory() as d:
        repo, out = os.path.join(d, "r"), os.path.join(d, "out")
        os.makedirs(repo)

        def g(*a):
            subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True)
        g("init", "-q", "-b", "main")
        g("config", "user.email", "t@t")
        g("config", "user.name", "t")
        open(os.path.join(repo, "x.sh"), "w").write("one\n")
        g("add", ".")
        g("commit", "-qm", "base")
        open(os.path.join(repo, "x.sh"), "w").write("one\ncmd | tail\n")
        g("commit", "-qam", "pr 7: add a step")
        pr_sha = git(repo, "rev-parse", "HEAD").strip()
        open(os.path.join(repo, "x.sh"), "w").write("one\ncmd; rc=$?\n")
        g("commit", "-qam", "x.sh: fix the exit code read after a pipe")
        fix_sha = git(repo, "rev-parse", "HEAD").strip()
        now = dt.datetime.now(dt.timezone.utc)
        ts = (now - dt.timedelta(days=1)).isoformat()[:19] + "Z"
        h1, h2 = "1" * 40, "2" * 40
        rv = (body + f"\n<!-- {MARKER} head_sha={h1} verdict=MUST_FIX must_fix=1 warn=2 mode=full ledger=x run=55 -->\n"
              "<!-- bloomery-usage reviewer: cost_usd=None engine=codex; validator: cost_usd=0.25 engine=claude -->")

        def fake(method, url, token, payload=None):
            u = url.split("?")[0]
            if "/search/issues" in url:
                return {"items": [{"number": 7}, {"number": 8}]}
            if u.endswith("/pulls/7/reviews"):
                return [{"user": {"login": BOT}, "body": rv, "html_url": "u7", "submitted_at": ts, "commit_id": h1},
                        {"user": {"login": "o"}, "body": "lgtm", "html_url": "h", "submitted_at": ts}]
            if u.endswith("/pulls/8/reviews"):
                return []
            if u.endswith("/pulls/7/commits"):
                return [{"commit": {"message": "pr 7: add a step\n\nbody"}}]
            if u.endswith("/pulls/7/comments"):
                return [{"user": {"login": BOT}, "path": "z.rs", "original_line": 3, "original_commit_id": h1,
                         "body": "use X", "html_url": "c1"}]
            if u.endswith("/pulls/7"):
                return {"merged": True, "merge_commit_sha": pr_sha, "commits": 1, "merged_at": ts,
                        "head": {"sha": h2}}
            if u.endswith(f"/compare/{h1}...{h2}"):
                return {"files": [{"filename": "t.sh", "status": "modified", "patch": "@@ -5 +5 @@\n-a\n+b"}]}
            raise AssertionError(url)

        def dl(repo_, run, dest):
            assert run == "55"
            os.makedirs(os.path.join(dest, "out"))
            json.dump([{"what": "a guess", "path": "y.rs", "line": 2, "reason": "speculation", "lesson": None}],
                      open(os.path.join(dest, "out", "removed.json"), "w"))
            json.dump([{"tool": "crapper", "path": "t.sh", "line": 5}, {"tool": "dryer", "path": "src/l.rs", "line": 9},
                       {"tool": "dryer", "path": "q.rs", "line": 1}, {"tool": "lint", "path": "t.sh", "line": 5}],
                      open(os.path.join(dest, "out", "aims.json"), "w"))
            json.dump([{"id": "L3", "effect": "raised", "path": "t.sh", "line": 5, "what": "rc"},
                       {"id": "L4", "effect": "suppressed", "what": "style"}, {"id": "bad"}],
                      open(os.path.join(dest, "out", "lessons.json"), "w"))
            return True
        assert collect("o/r", repo, 8, 60, "main", "t", out, call=fake, download=dl, now=now) == (1, 3, 1)
        sig = open(os.path.join(out, "signals.md")).read()
        assert f"commit/{fix_sha}" in sig and "from #7" in sig and "x.sh:2" in sig, sig
        assert "#7 must t.sh:5" in sig.split("answered")[1] and "#7 warn src/l.rs:9" in sig.split("answered")[0], sig
        assert "#7 inline z.rs:3" in sig.split("answered")[0] and "reason=speculation" in sig, sig
        assert json.load(open(os.path.join(out, "lesson-stats.json"))) == {
            "L3": {"raised": 1, "acted": 1, "suppressed": 0}, "L4": {"raised": 0, "acted": 0, "suppressed": 1}}
        stats = open(os.path.join(out, "stats.md")).read()
        assert "reviews 1 (CLEAN 0, MUST_FIX 1)" in stats and "acted 1, ignored 2" in stats, stats
        assert "review cost $0.25 (1 with no cost) · " in stats, stats
        assert "crapper aimed 1 (acted 1, ignored 0) · dryer aimed 2 (acted 0, ignored 1)" in stats, stats
        assert open(os.path.join(out, "has_signals.txt")).read() == "true\n"
        # A squash commit stands alone; a merge commit brings its second parent's side.
        assert pr_commits(repo, fix_sha, {"something else"}) == [fix_sha]
        g("checkout", "-qb", "side", pr_sha)
        open(os.path.join(repo, "y.sh"), "w").write("y\n")
        g("add", ".")
        g("commit", "-qm", "side change")
        side = git(repo, "rev-parse", "HEAD").strip()
        g("checkout", "-q", "main")
        g("merge", "-q", "--no-ff", "-m", "Merge side", "side")
        merge = git(repo, "rev-parse", "HEAD").strip()
        assert pr_commits(repo, merge, set()) == [merge, side], pr_commits(repo, merge, set())
    print("signals.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    opts, rest = {}, argv[1:]
    while rest:
        k = rest.pop(0)
        if not rest:
            print(__doc__, file=sys.stderr)
            return 64
        opts[k] = rest.pop(0)
    token = os.environ.get("GH_TOKEN", "")
    try:
        if not token:
            raise KeyError("GH_TOKEN")
        e, o, r = collect(opts["--repo"], opts["--git"], int(opts["--days"]), int(opts.get("--lookback", "60")),
                          opts.get("--ref", "HEAD"), token, opts["--out"], all_merged=opts.get("--all-merged") == "1")
    except KeyError as k:
        print(f"signals.py: missing {k}\n{__doc__}", file=sys.stderr)
        return 64
    print(f"signals: escapes={e} outcomes={o} removals={r}; "
          + open(os.path.join(opts["--out"], "stats.md")).read().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
