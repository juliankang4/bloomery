#!/usr/bin/env bash
# 주석 규약 점검(맥, 빌드 없음). AGENTS.md Conventions: 주석에 이슈 번호·날짜를 쓰지 않는다 —
# 이력은 rig-log와 커밋 메시지의 것이다. 예외는 재핀 귀속 한 줄 `PIN(YYYY-MM-DD):`.
# 대상은 crates/*/src와 crates/*/tests 전부다. 목록이 아니라 제외로 적어서 새 크레이트가 조용히 빠지지 않게 한다. 제외는
# 워크스페이스 밖의 재현기 oxide-ice-unroll뿐이다.
# The comments are read by tools/check-comment-only.py --comments (one lexer: `//` and `/* */`, never a `//`
# inside a string). Two rules, each outside a `PIN(` line:
#   crates/*/src and crates/*/tests: no issue number (MUL-N) and no date in a comment;
#   crates/*/src: no measured value in a comment — a number with a time, rate or bandwidth unit.
# A third rule reads every text file under tools/, comment or not: no `<doc>.md:<line>` citation (ranges and
# comma lists start the same way). A line number goes silently wrong the moment the doc is edited above it;
# cite a section heading plus a table row's first cell or a quoted phrase, or the rig-log
# `log/<date>.md#<anchor>` the value was measured in. An exception is a CITE_ALLOW entry, named with its reason.
# It sees the `.md:` form only: a doc named without its extension (`<report>:<line>`) is not caught.
# A fourth rule reads every file git sees (tracked, or untracked and not ignored): no word of a deleted
# feature (RETIRED_WORDS, a case-blind ERE spelled so this file cannot match it). An exception is a WORD_ALLOW entry: a whole file (`<path>||<reason>`) or one line
# (`<path>|<fixed text on the line>|<reason>`); an entry that matches nothing is an error.
# Exit: 0 clean, 1 a violation, 2 no verdict: a source listing, the comment reader or a retired-word scan that cannot
# read a file or a directory, or a grep that fails. 64 a malformed allow entry or an unknown argument, 69 not a git
# work tree. `--self-test` runs the listing, the comment reader and the retired-word scan on fixed input and on input
# that cannot be read, and exits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=tools/grep-answer.sh
source "$ROOT/tools/grep-answer.sh"
# The router-frequency card list and its lever are deleted: the placement ranks no card set.
RETIRED_WORDS='h[o]t.?list|뜨[거]운 목록|핫 ?리[스]트'

nov() { echo "check-comments: $*, no verdict" >&2; exit 2; }
# The wrapped-word scan's file list (NUL-separated, so it is a file and not a variable).
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

# rs_files CRATES : every .rs under CRATES/<crate>/src and CRATES/<crate>/tests, sorted, into files (global), the
# reproducer oxide-ice-unroll left out. The whole CRATES directory is listed, so any directory under it that find cannot
# enter ends the check, a crate directory included.
rs_files() {
  local listing f
  listing=$(find "$1" -path "$1/oxide-ice-unroll" -prune -o -name '*.rs' -print | LC_ALL=C sort) ||
    nov "cannot list the .rs files under $1 (exit $?, above)"
  files=()
  while IFS= read -r f; do
    case ${f#"$1"/*/} in src/* | tests/*) files+=("$f") ;; esac
  done <<<"$listing"
}

# comment_lines FILE… : into comments (global), every comment line of FILE… as `path:line:text`. A reader that fails
# (it stops on a file it cannot open) ends the check.
comment_lines() {
  comments=$(python3 tools/check-comment-only.py --comments "$@") || nov "the comment reader failed (rc $?, above)"
}

# gg ARGS… : the output of `git grep --untracked ARGS…`, and 0 on a match or none. git grep says only on stderr that
# it cannot open a file or a directory, and exits 0 or 1 as if it had read it, so any message there, or another
# status, ends the check.
gg() {
  local err rc=0
  { err=$(git grep --untracked "$@" 2>&1 >&3); } 3>&1 || rc=$?
  [ "$rc" -le 1 ] && [ -z "$err" ] && return 0
  [ -z "$err" ] || printf '%s\n' "$err" >&2
  nov "git grep cannot read every file of the tree (above)"
}

# retired : into words (global), every line of the files git sees that carries a retired word.
retired() {
  local split
  words=$(gg -nIiE -- "$RETIRED_WORDS") || exit $?
  # The English word wrapped across a line break ("… a h""ot" / "// list …"), which a line grep cannot see; reported
  # at the line the word starts on. A listed file awk cannot read ends the check.
  gg -zIl -e . >"$LIST"
  split=$(xargs -0 awk '
    FNR == 1 { prev = "" }
    { cur = tolower($0) }
    prev ~ /h[o]t[ -]?$/ && cur ~ /^[[:space:]]*[\/!#*-]*[[:space:]]*list/ { print FILENAME ":" FNR - 1 ":" last " / " $0 }
    { prev = cur; last = $0 }' <"$LIST") || nov "the wrapped-word scan cannot read a file git grep listed (above)"
  [ -z "$split" ] || words=$(printf '%s\n%s\n' "$words" "$split" | sed '/^$/d')
}

self_test() {
  local fails=0 out rc d
  expect() {
    [ "$2" = "$3" ] || { echo "check-comments: self-test $1: got [$2], want [$3]" >&2; fails=$((fails + 1)); }
  }
  rc=0 out=$( (rs_files /nonexistent) 2>&1) || rc=$?
  expect missing-dir "$rc ${out##*$'\n'}" \
    "2 check-comments: cannot list the .rs files under /nonexistent (exit 1, above), no verdict"
  d=$(mktemp -d)
  mkdir -p "$d/x/src/m" "$d/x/tests" "$d/x/benches" "$d/oxide-ice-unroll/src"
  touch "$d/x/src/m/a.rs" "$d/x/tests/t.rs" "$d/x/benches/b.rs" "$d/x/build.rs" "$d/oxide-ice-unroll/src/o.rs"
  rc=0 out=$( (rs_files "$d" && printf '%s ' "${files[@]#"$d"/}") 2>&1) || rc=$?
  expect listing "$rc $out" "0 x/src/m/a.rs x/tests/t.rs "
  rm -rf "$d"
  rc=0 out=$( (comment_lines /nonexistent/a.rs) 2>&1) || rc=$?
  expect reader "$rc ${out##*$'\n'}" "2 check-comments: the comment reader failed (rc 1, above), no verdict"
  # git stands in for each git grep of the scan: the line search, then the listing the wrapped-word awk reads.
  local warn="warning: could not open directory 'a/': Permission denied"
  rc=0 out=$( (git() { case $* in (*-nIiE*) echo "$warn" >&2 ;; esac; }; retired) 2>&1) || rc=$?
  expect unreadable "$rc ${out##*$'\n'}" "2 check-comments: git grep cannot read every file of the tree (above), no verdict"
  rc=0 out=$( (git() { case $* in (*-zIl*) echo "$warn" >&2 ;; esac; }; retired) 2>&1) || rc=$?
  expect unreadable-listing "$rc ${out##*$'\n'}" "2 check-comments: git grep cannot read every file of the tree (above), no verdict"
  rc=0 out=$( (git() { return 128; }; retired) 2>&1) || rc=$?
  expect git-failed "$rc ${out##*$'\n'}" "2 check-comments: git grep cannot read every file of the tree (above), no verdict"
  rc=0 out=$( (git() { case $* in (*-zIl*) printf '/nonexistent/a.rs\0' ;; esac; }; retired) 2>&1) || rc=$?
  expect listed-unreadable "$rc ${out##*$'\n'}" \
    "2 check-comments: the wrapped-word scan cannot read a file git grep listed (above), no verdict"
  rc=0 out=$(bash "$ROOT/tools/check-comments.sh" '' extra 2>&1) || rc=$?
  expect extra-argument "$rc $out" "64 check-comments: unknown argument ' extra' (none, --self-test)"
  [ "$fails" = 0 ] || { echo "check-comments: self-test: $fails failed" >&2; return 1; }
  echo "check-comments: self-test ok"
}

case "$*" in
  --self-test) self_test; exit $? ;;
  '') ;;
  *) echo "check-comments: unknown argument '$*' (none, --self-test)" >&2; exit 64 ;;
esac

rs_files crates
comment_lines "${files[@]}"
# Each line is `path:line:text`; a rule matches the text after the line number.
text() { grep_answer check-comments -E "^[^:]+:[0-9]+:.*($1)" | grep_answer check-comments -v 'PIN('; }
fail=0
bad=$(printf '%s\n' "$comments" | text 'MUL-[0-9]+|20[0-9]{2}-[0-9]{2}-[0-9]{2}')
if [ -n "$bad" ]; then
  n=$(printf '%s\n' "$bad" | wc -l | tr -d ' ')
  printf '%s\n' "$bad" | head -40 >&2
  echo "check-comments: $n comment line(s) carry an issue number or a date — history belongs in rig-log (AGENTS.md Conventions)" >&2
  fail=1
fi
unit='(^|[^0-9A-Za-z_])[0-9][0-9,.]*[[:space:]]?(ms|µs|μs|us|ns|GB/s|GiB/s|tok/s|t/s|MB/s)([^0-9A-Za-z_]|$)'
meas=$(printf '%s\n' "$comments" | grep_answer check-comments -E '^crates/[^/]+/src/' | text "$unit")
if [ -n "$meas" ]; then
  n=$(printf '%s\n' "$meas" | wc -l | tr -d ' ')
  printf '%s\n' "$meas" | head -40 >&2
  echo "check-comments: $n comment line(s) in crates/*/src carry a measured value — the number belongs in rig-log (AGENTS.md Conventions: measured ms/GB/s/tok/s)" >&2
  fail=1
fi
# CITE_ALLOW: `<path>|<fixed text on the line>|<reason>`, one per allowed line; an entry that matches no
# citation is an error, so a stale exception cannot linger.
CITE_ALLOW=()
cites=$(grep_answer check-comments -rnIE '[A-Za-z0-9_./-]+\.md:[0-9]+' tools)
unmatched=()
allowed=0
for a in ${CITE_ALLOW[@]+"${CITE_ALLOW[@]}"}; do
  p=${a%%|*}; rest=${a#*|}; t=${rest%%|*}
  [ -n "$p" ] && [ -n "$t" ] && [ "$rest" != "$t" ] && [ -n "${rest#*|}" ] \
    || { echo "check-comments: CITE_ALLOW entry '$a' is not <path>|<text>|<reason>" >&2; exit 64; }
  hit=$(printf '%s\n' "$cites" | grep_answer check-comments -F -- "$t" | while IFS= read -r l; do case $l in ("$p:"*) printf '%s\n' "$l" ;; esac; done)
  if [ -z "$hit" ]; then unmatched+=("$a"); continue; fi
  allowed=$((allowed + $(printf '%s\n' "$hit" | wc -l)))
  cites=$(printf '%s\n' "$cites" | grep_answer check-comments -v -xF -- "$hit")
done
if [ "${#unmatched[@]}" -gt 0 ]; then
  printf 'check-comments: CITE_ALLOW entry matches no citation: %s\n' "${unmatched[@]}" >&2
  fail=1
fi
if [ -n "$cites" ]; then
  n=$(printf '%s\n' "$cites" | wc -l | tr -d ' ')
  printf '%s\n' "$cites" | head -40 >&2
  echo "check-comments: $n line(s) under tools/ cite a doc by line number — cite its section heading plus a table row's first cell or a quoted phrase, or a rig-log anchor" >&2
  fail=1
fi
WORD_ALLOW=(
  'crates/levers/src/registry.rs|name: "BLOOMERY_HOT_|the retired row refuses the name when it is set'
)
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "check-comments: not a git work tree — the retired-word rule reads the tracked files; no verdict" >&2; exit 69; }
retired
wunmatched=()
wallowed=0
for a in "${WORD_ALLOW[@]}"; do
  p=${a%%|*}; rest=${a#*|}; t=${rest%%|*}
  [ -n "$p" ] && [ "$rest" != "$t" ] && [ -n "${rest#*|}" ] \
    || { echo "check-comments: WORD_ALLOW entry '$a' is not <path>|<text>|<reason>" >&2; exit 64; }
  hit=$(printf '%s\n' "$words" | while IFS= read -r l; do
    case $l in ("$p:"*) [ -z "$t" ] || case $l in (*"$t"*) ;; (*) continue ;; esac; printf '%s\n' "$l" ;; esac
  done)
  if [ -z "$hit" ]; then wunmatched+=("$a"); continue; fi
  wallowed=$((wallowed + $(printf '%s\n' "$hit" | wc -l)))
  words=$(printf '%s\n' "$words" | grep_answer check-comments -v -xF -- "$hit")
done
if [ "${#wunmatched[@]}" -gt 0 ]; then
  printf 'check-comments: WORD_ALLOW entry matches no line: %s\n' "${wunmatched[@]}" >&2
  fail=1
fi
if [ -n "$words" ]; then
  n=$(printf '%s\n' "$words" | wc -l | tr -d ' ')
  nf=$(printf '%s\n' "$words" | cut -d: -f1 | LC_ALL=C sort -u | wc -l | tr -d ' ')
  sed -n 1,40p <<<"$words" >&2
  echo "check-comments: $n line(s) in $nf file(s) of the tree carry a retired word ($RETIRED_WORDS, any case) — the feature is deleted; describe what runs now" >&2
  fail=1
fi
[ "$fail" = 0 ] || exit 1
echo "check-comments: ok (${#files[@]} files, $(printf '%s\n' "$comments" | wc -l | tr -d ' ') comment lines; tools/ cites no doc by line number, $allowed allowed by CITE_ALLOW; no retired word, $wallowed line(s) allowed by WORD_ALLOW)"
