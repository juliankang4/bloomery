#!/usr/bin/env bash
# The one-bundle-load ratchet (Mac, grep only, no build): a `#[cuda_module]` family's generated
# `load` reads its crate's whole embedded bundle, so calling it outside the shared slot is one more
# whole-bundle cuModuleLoadData per call — the class the shared_module! path closed (one load per
# (device, crate bundle) per process, counted by bloomery_gpu::bundle_loads). This file owns the
# detection rule:
#   families every `#[cuda_module]`-attributed `mod NAME` under crates/ except oxide-ice-unroll
#            (the reproducer outside the workspace)
#   uses     every code line under crates/<dir>/src that calls `<family>::load(` literally
#   allowed  the bundle-path closure idiom alone — `::load(anchor).map(|loaded|
#            loaded.as_cuda_module().clone())`, the body shared_module! and start_bundle_load's
#            closures spell — and pure `//` comment lines; the macro's own `$module::load` is a
#            metavariable and never matches a family name
# The scan keeps the lines holding the fixed string `::load(` (grep -F) and runs the family alternation on
# those lines alone, never on every source line.
# Every other family load goes through `crate::shared_module!` (inside bloomery-gpu) or
# `bloomery_gpu::shared_module!` (outside it), which binds to the slot's one module.
# Exit: 0 no raw load (a line naming the family count). 1 each raw load, its file:line and the
# family. 70 the family scan found nothing (the collector rotted), or a source file the family or load scan cannot list
# or read (no silent pass). `--self-test` runs both scans on fixed input and a missing file and exits.
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LC_ALL=C

# shellcheck source=tools/grep-answer.sh
source "$ROOT/tools/grep-answer.sh"

say() { printf 'check-loads: %s\n' "$*"; }

# The generated load's one legitimate caller shape: the closure that hands the loaded module to
# bundle_module/start_bundle_load. Matching it (not a file exemption) keeps lib.rs itself covered.
idiom='::load\(anchor\)\.map\(\|loaded\| loaded\.as_cuda_module\(\)\.clone\(\)\)'

# raw_loads FILE… : into hits, the lines of FILE… that call a family's (pattern's) generated load outside the
# idiom; none with no file (stdin is never read). A file grep cannot read, or any stage that fails, ends the check.
raw_loads() {
    hits=
    [ $# -gt 0 ] || return 0
    hits=$(grep_answer check-loads -HnF '::load(' "$@" |
        grep_answer check-loads -E "^[^:]+:[0-9]+:(.*[^A-Za-z0-9_])?(${pattern})::load\(" |
        grep_answer check-loads -vE '^[^:]+:[0-9]+:[[:space:]]*//' |
        grep_answer check-loads -vE "$idiom") ||
        { say "cannot read a source file to scan for loads (exit $?, above)" >&2; exit 70; }
}

# families_of FILE… : into families, the name of every `#[cuda_module]`-attributed `mod` in FILE…, sorted, one per
# line; none with no file (stdin is never read). A file awk cannot read ends the check.
families_of() {
    families=
    [ $# -gt 0 ] || return 0
    families=$(awk '
        /^[[:space:]]*#[[:space:]]*\[[[:space:]]*cuda_module\]/ { want = 1; next }
        want {
            if ($0 ~ /^[[:space:]]*(\/\/|#!?\[)/) next
            if ($0 ~ /^[[:space:]]*(pub([[:space:]]*\([a-z]+\))?[[:space:]]+)?mod[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                name = $0
                sub(/.*mod[[:space:]]+/, "", name)
                sub(/[^A-Za-z0-9_].*/, "", name)
                print name
            }
            want = 0
        }
        ' "$@" | sort -u) ||
        { say "cannot read a source file to collect the #[cuda_module] families (awk failed, above)" >&2; exit 70; }
}

self_test() {
    local fails=0 out rc pattern=kern
    expect() {
        [ "$2" = "$3" ] || { say "self-test $1: got [$2], want [$3]" >&2; fails=$((fails + 1)); }
    }
    rc=0 out=$( (raw_loads <(printf 'let m = kern::load(ctx)?;\n') <(printf 'fn f() {}\n') &&
        printf '%s' "${hits#*:*:}") 2>&1) || rc=$?
    expect load "$rc $out" "0 let m = kern::load(ctx)?;"
    rc=0 out=$( (raw_loads <(printf 'kern::load(x)\n') && printf '%s' "${hits#*:*:}") 2>&1) || rc=$?
    expect one-file "$rc $out" "0 kern::load(x)"
    rc=0 out=$( (raw_loads <(printf 'fn f() {}\n') <(printf 'fn g() {}\n') && printf '%s' "$hits") 2>&1) || rc=$?
    expect no-load "$rc $out" "0 "
    rc=0 out=$( (raw_loads <(printf 'fn f() {}\n') <(printf 'other::load(x)\n') && printf '%s' "$hits") 2>&1) || rc=$?
    expect other-family "$rc $out" "0 "
    rc=0 out=$( (raw_loads <(printf '// kern::load(x)\n') \
        <(printf 'kern::load(anchor).map(|loaded| loaded.as_cuda_module().clone())\n') && printf '%s' "$hits") 2>&1) || rc=$?
    expect comment-and-idiom "$rc $out" "0 "
    rc=0 out=$( (raw_loads <<< '2:kern::load(x)' && printf '%s' "$hits") 2>&1) || rc=$?
    expect no-file "$rc $out" "0 "
    rc=0 out=$( (raw_loads /nonexistent/a.rs <(printf 'kern::load(x)\n')) 2>&1) || rc=$?
    expect missing-file "$rc ${out##*$'\n'}" "70 check-loads: cannot read a source file to scan for loads (exit 2, above)"
    rc=0 out=$( (families_of <(printf '#[cuda_module]\npub mod kern {\n') <(printf '#[cuda_module]\n// x\nmod other;\n') &&
        printf '%s ' $families) 2>&1) || rc=$?
    expect families "$rc $out" "0 kern other "
    rc=0 out=$( (families_of && printf '%s' "$families") 2>&1 <<< $'#[cuda_module]\nmod kern {') || rc=$?
    expect families-no-file "$rc $out" "0 "
    rc=0 out=$( (families_of <(printf '#[cuda_module]\nmod kern {\n') /nonexistent/a.rs) 2>&1) || rc=$?
    expect families-missing-file "$rc ${out##*$'\n'}" \
        "70 check-loads: cannot read a source file to collect the #[cuda_module] families (awk failed, above)"
    rc=0 out=$(bash "$ROOT/tools/check-loads.sh" '' extra 2>&1) || rc=$?
    expect extra-argument "$rc $out" "64 check-loads: unknown argument ' extra' (none, --self-test)"
    [ "$fails" = 0 ] || { say "self-test: $fails failed" >&2; return 1; }
    say "self-test ok"
}

case "$*" in
    --self-test) self_test; exit $? ;;
    '') ;;
    *) say "unknown argument '$*' (none, --self-test)" >&2; exit 64 ;;
esac

# Every .rs under crates/ but the reproducer, listed once: the family scan reads them all, the load scan those under a
# src/ directory.
listing=$(find crates -type f -name '*.rs' -not -path 'crates/oxide-ice-unroll/*' | sort) ||
    { say "cannot list the .rs files under crates (find failed, above)" >&2; exit 70; }
all=() files=()
while IFS= read -r f; do
    [ -n "$f" ] || continue
    all+=("$f")
    case $f in */src/*) files+=("$f") ;; esac
done <<<"$listing"
families_of ${all[@]+"${all[@]}"}
[ -n "$families" ] || { say "no #[cuda_module] family found under crates/ — the scan collector rotted" >&2; exit 70; }
n_families="$(printf '%s\n' "$families" | wc -l | tr -d ' ')"

pattern="$(printf '%s\n' "$families" | paste -sd'|' -)"
raw_loads ${files[@]+"${files[@]}"}
status=0
while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    say "raw generated-module load: $hit"
    status=1
done <<<"$hits"

if [ "$status" -eq 0 ]; then
    say "$n_families #[cuda_module] families, every load through the shared slot"
fi
exit "$status"
