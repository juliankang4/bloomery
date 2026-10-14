#!/usr/bin/env bash
# PTX spill ratchet — the gate over ptx-scan's `spill` (ptxas spill stores, bytes) and `jit_local`
# (the driver JIT's local bytes per thread) columns. ptx-scan is an instrument and asserts nothing;
# this script reads its table for each named binary and holds every entry to the value pinned in the
# table file. Both columns are fixed at build time, so this is a compile-time ratchet: no timing.
#
# The failure it blocks: a kernel that starts spilling to local memory passes every bit gate — the
# spill changes speed, not bits — and nothing read those columns (an 8-byte ptxas spill in
# ds41_attn_seg_sel was found by eye). A new entry is a failure until it is pinned, so a new kernel
# cannot arrive spilling unseen.
#
# A scan that skipped the driver JIT (`ptx-scan.sh --no-jit`: banner `jit=skipped(no-jit)`, the
# jit_local cells `skipped(no-jit)`) holds no jit_local reading, and this ratchet pins that column: it is
# refused by name, never read as a value. The script runs the scan itself without the flag, so this
# guards a caller that hands it another scan.
#
# Usage: tools/ptx-spill-check.sh <table>... <binary name>[:<entry substring>]...
#        tools/ptx-spill-check.sh --self-test   (a stub ptx-scan.sh on fixed scans, no build; check-recipes runs it)
#   Runs `tools/ptx-scan.sh <binary>` for each binary (the recipe builds them first) and compares
#   its entry rows with that binary's rows in the table its scan reads. A `<binary>:<substring>` argument scans only
#   the entries whose name holds the substring (ptx-scan's own filter) and the table's rows for the binary are
#   those entries: a binary that links a bundle another named binary already pins (the gpu bundle) pins only
#   its own device crate's entries. A table (tools/ref/ptx-shapes.tsv) holds
#   `<binary> <entry> <spill> <jit_local>`, whitespace-separated; `#` starts a comment. A pinned
#   nonzero value carries a `# PIN(YYYY-MM-DD): <reason>` comment on the line above its row.
#
#   One table per PTX arch and card capability, because `spill` follows the arch ptxas assembles for and
#   `jit_local` the capability the driver JITs for. The leading arguments that hold a `/` are the tables, and one that
#   ends in `/` is every `*.tsv` in that directory (the recipe names tools/ref/ptx-shapes.tsv and tools/ref/ptx-shapes/);
#   the rest are binaries. A table names its arch on one `# arch: <sm_NN>` line and its capability on
#   one `# jit-cc: <M.m>` line of its header. A table with no such line, with two, or with one whose value is not one
#   arch or capability, and two tables of one arch and capability, are refused. Each binary is read against the table
#   of its own scan's banner, `arch=` and `jit-cc=`; a scan with no such table fails by name. An arch no table pins
#   (BLOOMERY_ARCH, sm_86 unset: tools/arch.sh) ends the script before any scan, so a scan is never read against
#   another arch's pins.
#   Every table is read once, into a copy every binary reads: a table given as a pipe or a process
#   substitution (`<(grep … tools/ref/ptx-shapes.tsv)`) is empty on a second read.
#
# Every violation prints one line, and the script runs every binary before it decides:
#   <bin> <entry> spill=<got> pinned=<want> ABOVE|BELOW   (either column; BELOW = lower the pin)
#   <bin> <entry> unpinned spill=<got> jit_local=<got>     (in the scan, not in the table)
#   <bin> <entry> stale                                    (in the table, not in the scan)
#   ptx-spill bin=<bin> scan is no-jit (jit_local not read) FAIL   (a scan that skipped the JIT)
#   ptx-spill bin=<bin> no table for <arch> on cc <M.m> FAIL       (no table of the scan's arch and capability)
# Before its verdict each binary names the table it read: `ptx-spill bin=<bin> table=<path> arch=<arch> jit-cc=<M.m>`.
# Exit status: 0 when every binary's scan read and matched its pins; 1 on a violation or a scan
# that failed (ptx-scan's own nonzero exit, or a table it printed without rows); 2 on a usage error
# (a table whose arch or capability lines are refused above included); 64 a BLOOMERY_ARCH that is no arch or that no
# table pins.
set -uo pipefail

self_test() {
  local t fails=0 out rc
  t=$(mktemp -d)
  cp "${BASH_SOURCE[0]}" "$t/ptx-spill-check.sh"
  cp "${BASH_SOURCE[0]%/*}/arch.sh" "$t/arch.sh"
  unset BLOOMERY_ARCH
  # The stub scan prints the fixture scan of its binary, or fails as ptx-scan does on a missing binary.
  printf '%s\n' '#!/usr/bin/env bash' \
    'f="${BASH_SOURCE[0]%/*}/$1${2:+.$2}.scan"' \
    '[ -f "$f" ] || { echo "ptx-scan bin=target/release/$1 missing scan=failed"; exit 1; }' \
    'cat "$f"' > "$t/ptx-scan.sh"
  # A scan's banner names the arch and capability it was read on: SA and SC, sm_86 and 8.6 unset.
  scan() { # scan <bin> <entry spill jit_local>...
    local b=$1 r
    shift
    { echo "ptx-scan bin=target/release/$b arch=${SA:-sm_86} modules=1 jit-card=fixture jit-cc=${SC:-8.6} jit-cuda=13.4"; echo "entry regs spill jit_local"
      for r in "$@"; do set -- $r; echo "$1 32 $2 $3"; done
      echo "ptx-scan-md5: method=m"; echo "k 0123 1"; } > "$t/$b.scan"
  }
  marked() { # marked <bin> <banner tail> <jit_local cell> <entry spill>...: a scan whose banner tail and jit_local cell are given
    local b=$1 banner=$2 cell=$3 r
    shift 3
    { echo "ptx-scan bin=target/release/$b arch=sm_86 modules=1$banner"; echo "entry regs spill jit_local"
      for r in "$@"; do set -- $r; echo "$1 32 $2 $cell"; done
      echo "ptx-scan-md5: method=m"; echo "k 0123 1"; } > "$t/$b.scan"
  }
  scan a "k1 0 0" "k2 8 0"
  scan b "k3 0 0"
  scan c "k4 0 0"
  scan d "k1 16 0"
  marked f " jit=skipped(no-jit)" "skipped(no-jit)" "k1 0"
  marked g " jit=skipped(no-jit)" 0 "k1 0"
  marked h "" "skipped(no-jit)" "k1 0"
  scan v.vis_ "vis_a 0 0"
  # Scans on a cc 12.1 card: w at sm_86, x at sm_86 with only a cc 8.6 row, y on a capability no table pins, z at sm_121.
  SC=12.1 scan w "k1 0 0" "k2 8 24"
  SC=12.1 scan x "k1 0 0"
  SC=9.0 scan y "k1 0 0"
  SA=sm_121 SC=12.1 scan z "k1 0 16"
  printf '%s\n' '# fixture' '# arch: sm_86' '# jit-cc: 8.6' 'a k1 0 0' '# PIN(2000-01-01): fixture' 'a k2 8 0' 'b k3 0 0' \
    'd k1 0 0' 'd k9 0 0' 'f k1 0 0' 'g k1 0 0' 'h k1 0 0' 'v vis_a 0 0' 'w k1 0 0' 'w k2 8 0' 'x k1 0 0' 'y k1 0 0' 'z k1 0 0' \
    > "$t/table"
  mkdir "$t/side" "$t/empty"
  printf '%s\n' '# fixture' '# arch: sm_86' '# jit-cc: 12.1' 'w k1 0 0' '# PIN(2000-01-01): fixture' 'w k2 8 24' 'y k1 0 0' \
    > "$t/side/sm_86-cc12.1.tsv"
  printf '%s\n' '# fixture' '# arch: sm_121' '# jit-cc: 12.1' '# PIN(2000-01-01): fixture' 'z k1 0 16' > "$t/side/sm_121-cc12.1.tsv"
  tables=("$t/table" "$t/side/")
  check() { # check <name> <want rc> <want line or -> <args...>
    local name=$1 want=$2 line=$3
    shift 3
    out=$(bash "$t/ptx-spill-check.sh" "$@" 2>&1); rc=$?
    if [ "$rc" != "$want" ] || { [ "$line" != - ] && ! grep -qxF -- "$line" <<< "$out"; }; then
      echo "ptx-spill-check self-test: $name: rc $rc (want $want), want line '$line' in:" >&2
      sed 's/^/  /' <<< "$out" >&2
      fails=$((fails + 1))
    fi
  }
  check "a table read twice through a process substitution" 0 "ptx-spill bin=b entries=1 pinned=1 nonzero=none violations=0 PASS" \
    <(cat "$t/table") a b
  check "a binary with no pinned row reads unpinned, not stale" 1 "c k4 unpinned spill=0 jit_local=0" "$t/table" a c
  check "a spill above its pin" 1 "d k1 spill=16 pinned=0 ABOVE" "$t/table" d
  check "a pinned row the scan lacks" 1 "d k9 stale" "$t/table" d
  check "a failed scan" 1 "ptx-spill bin=e scan rc=1 FAIL" "$t/table" e
  check "a scan that skipped the JIT (banner and cells)" 1 "ptx-spill bin=f scan is no-jit (jit_local not read) FAIL" "$t/table" f
  check "a no-jit banner over a numeric jit_local" 1 "ptx-spill bin=g scan is no-jit (jit_local not read) FAIL" "$t/table" g
  check "a skipped jit_local under a banner that ran the JIT" 1 "ptx-spill bin=h scan is no-jit (jit_local not read) FAIL" "$t/table" h
  check "a binary scanned under an entry filter pins only the filtered entries" 0 "ptx-spill bin=v entries=1 pinned=1 nonzero=none violations=0 PASS" "$t/table" v:vis_
  check "no binary" 2 - "$t/table"
  check "no table" 2 - a
  # The table: the scan's arch and capability pick it among the tables named.
  check "a cc 8.6 scan reads the cc 8.6 table" 0 "ptx-spill bin=a table=$t/table arch=sm_86 jit-cc=8.6" "${tables[@]}" a
  check "a cc 12.1 scan reads the cc 12.1 table" 0 "ptx-spill bin=w entries=2 pinned=2 nonzero=k2(spill=8,jit_local=24) violations=0 PASS" \
    "${tables[@]}" w
  check "a cc 12.1 scan is not read against a cc 8.6 row" 1 "x k1 unpinned spill=0 jit_local=0" "${tables[@]}" x
  check "a capability no table pins" 1 "ptx-spill bin=y no table for sm_86 on cc 9.0 FAIL" "${tables[@]}" y
  BLOOMERY_ARCH=sm_121 check "an sm_121 scan reads the sm_121 table" 0 \
    "ptx-spill bin=z table=$t/side/sm_121-cc12.1.tsv arch=sm_121 jit-cc=12.1" "${tables[@]}" z
  check "a directory with no table" 2 "ptx-spill-check: cannot read the table $t/empty/*.tsv" "$t/table" "$t/empty/" a
  # A table's header: one `# arch:` and one `# jit-cc:` line, each with one value of its shape.
  header() { # header <name> <header line>...: a table of a's two rows under those header lines
    local n=$1
    shift
    printf '%s\n' '# fixture' "$@" 'a k1 0 0' '# PIN(2000-01-01): fixture' 'a k2 8 0' > "$t/$n"
  }
  header noarch '# jit-cc: 8.6'
  header twoarch '# arch: sm_86' '# arch: sm_90' '# jit-cc: 8.6'
  header nocc '# arch: sm_86'
  header twocc '# arch: sm_86' '# jit-cc: 8.6' '# jit-cc: 12.1'
  header nocc-value '# arch: sm_86' '# jit-cc:'
  header twocc-values '# arch: sm_86' '# jit-cc: 12.1 8.6'
  header ccshape '# arch: sm_86' '# jit-cc: 86'
  header archshape '# arch: sm86' '# jit-cc: 8.6'
  header dup '# arch: sm_86' '# jit-cc: 8.6'
  check "a table that names no arch" 2 "ptx-spill-check: the table $t/noarch declares no \`# arch:\` line: its rows pin no arch" "$t/noarch" a
  check "a table that names two archs" 2 "ptx-spill-check: the table $t/twoarch declares 2 \`# arch:\` lines: a table pins one arch" "$t/twoarch" a
  check "a table that names no capability" 2 "ptx-spill-check: the table $t/nocc declares no \`# jit-cc:\` line: its rows pin no jit-cc" "$t/nocc" a
  check "a table that names a second capability" 2 \
    "ptx-spill-check: the table $t/twocc declares 2 \`# jit-cc:\` lines: a table pins one jit-cc" "$t/twocc" a
  check "a capability line with no value" 2 \
    "ptx-spill-check: the table $t/nocc-value declares '# jit-cc:', not one jit-cc as in '# jit-cc: 8.6'" "$t/nocc-value" a
  check "a capability line with two values" 2 \
    "ptx-spill-check: the table $t/twocc-values declares '# jit-cc: 12.1 8.6', not one jit-cc as in '# jit-cc: 8.6'" "$t/twocc-values" a
  check "a capability of another shape" 2 "ptx-spill-check: the table $t/ccshape declares '# jit-cc: 86', not one jit-cc as in '# jit-cc: 8.6'" "$t/ccshape" a
  check "an arch of another shape" 2 "ptx-spill-check: the table $t/archshape declares '# arch: sm86', not one arch as in '# arch: sm_86'" "$t/archshape" a
  check "two tables of one arch and capability" 2 \
    "ptx-spill-check: the tables $t/table and $t/dup both pin sm_86 on cc 8.6" "$t/table" "$t/dup" a
  # The arch: the scan is of the build's arch. An arch no table pins ends it before any scan.
  BLOOMERY_ARCH=sm_90 check "an arch no table pins" 64 \
    "ptx-spill-check: no ptx-shapes table for sm_90; the tables pin sm_86 on cc 8.6, sm_121 on cc 12.1, sm_86 on cc 12.1" "${tables[@]}" a
  if grep -q 'ptx-scan bin=' <<< "$out"; then
    echo "ptx-spill-check self-test: the refusal of another arch ran a scan:" >&2
    sed 's/^/  /' <<< "$out" >&2
    fails=$((fails + 1))
  fi
  BLOOMERY_ARCH=sm_86 check "the table's own arch is read against the table" 0 "ptx-spill bin=a entries=2 pinned=2 nonzero=k2(spill=8,jit_local=0) violations=0 PASS" "$t/table" a
  BLOOMERY_ARCH=sm86 check "a BLOOMERY_ARCH that is no arch" 64 "ptx-spill-check.sh: BLOOMERY_ARCH is a CUDA arch, sm_ and the capability's digits with an optional a or f (unset: sm_86), got 'sm86'" "$t/table" a
  rm -rf "$t"
  [ "$fails" = 0 ] && echo "ptx-spill-check: self-test ok" || { echo "ptx-spill-check: self-test $fails failed" >&2; return 1; }
}
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi

TABLES=()
while [ $# -gt 0 ]; do
  case $1 in */) TABLES+=("$1"*.tsv); shift ;; */*) TABLES+=("$1"); shift ;; *) break ;; esac
done
if [ "${#TABLES[@]}" = 0 ] || [ $# = 0 ]; then
  echo "usage: ptx-spill-check.sh <table>... <binary name>..." >&2
  exit 2
fi
SCAN="${BASH_SOURCE[0]%/*}/ptx-scan.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# shellcheck source=tools/arch.sh
source "${BASH_SOURCE[0]%/*}/arch.sh"
bloomery_arch ptx-spill-check.sh || exit $?
KEYS=()
for i in "${!TABLES[@]}"; do
  F=${TABLES[$i]}
  cat -- "$F" > "$TMP/table.$i" || { echo "ptx-spill-check: cannot read the table $F" >&2; exit 2; }
  # one `# <key>: <value>` line each, the value one arch (tools/arch.sh's shape) or one capability as a banner prints it
  for K in arch jit-cc; do
    if [ "$K" = arch ]; then S=$ARCH_SHAPE X=sm_86; else S='^[1-9][0-9]?\.[0-9]$' X=8.6; fi
    D=$(grep "^# $K:" "$TMP/table.$i")
    n=$(grep -c "^# $K:" "$TMP/table.$i")
    case $n in
      0) echo "ptx-spill-check: the table $F declares no \`# $K:\` line: its rows pin no $K" >&2; exit 2 ;;
      1) ;;
      *) echo "ptx-spill-check: the table $F declares $n \`# $K:\` lines: a table pins one $K" >&2; exit 2 ;;
    esac
    V=${D#"# $K: "}
    if ! [[ $V =~ $S ]]; then
      echo "ptx-spill-check: the table $F declares '$D', not one $K as in '# $K: $X'" >&2
      exit 2
    fi
    if [ "$K" = arch ]; then T_ARCH=$V; else T_CC=$V; fi
  done
  for j in "${!KEYS[@]}"; do
    if [ "${KEYS[$j]}" = "$T_ARCH $T_CC" ]; then
      echo "ptx-spill-check: the tables ${TABLES[$j]} and $F both pin $T_ARCH on cc $T_CC" >&2
      exit 2
    fi
  done
  KEYS+=("$T_ARCH $T_CC")
done
NAMED=$(for K in "${KEYS[@]}"; do echo "${K% *} on cc ${K#* }"; done | paste -sd, - | sed 's/,/, /g')
if ! printf '%s\n' "${KEYS[@]}" | grep -q "^$ARCH "; then
  echo "ptx-spill-check: no ptx-shapes table for $ARCH; the tables pin $NAMED" >&2
  exit 64
fi
rc=0
for ARG in "$@"; do
  BIN=${ARG%%:*}
  FILTER=
  case "$ARG" in *:*) FILTER=${ARG#*:} ;; esac
  bash "$SCAN" "$BIN" ${FILTER:+"$FILTER"} > "$TMP/$BIN.scan" 2> "$TMP/$BIN.err"
  src=$?
  banner=$(grep -m1 '^ptx-scan bin=' "$TMP/$BIN.scan")
  echo "${banner:-ptx-scan bin=target/release/$BIN (no banner)}"
  if [ "$src" -ne 0 ]; then
    echo "ptx-spill bin=$BIN scan rc=$src FAIL"
    sed 's/^/  ptx-scan stderr: /' "$TMP/$BIN.err" | head -5
    rc=1
    continue
  fi
  # A skipped JIT is no reading, and `skipped(no-jit)` in awk's arithmetic is 0: refused before any compare.
  if grep -qF 'skipped(no-jit)' "$TMP/$BIN.scan"; then
    echo "ptx-spill bin=$BIN scan is no-jit (jit_local not read) FAIL"
    rc=1
    continue
  fi
  # The scan's own arch and capability pick the table.
  BARCH=$(sed -n 's/.* arch=\([^ ]*\).*/\1/p' <<< "$banner")
  BCC=$(sed -n 's/.* jit-cc=\([^ ]*\).*/\1/p' <<< "$banner")
  T=
  for i in "${!KEYS[@]}"; do [ "${KEYS[$i]}" = "$BARCH $BCC" ] && T=$i; done
  if [ -z "$T" ]; then
    echo "ptx-spill bin=$BIN no table for ${BARCH:-(no arch)} on cc ${BCC:-(no jit-cc)} FAIL"
    rc=1
    continue
  fi
  echo "ptx-spill bin=$BIN table=${TABLES[$T]} arch=$BARCH jit-cc=$BCC"
  # The table rows: between the header line (`entry … jit_local`) and the md5 section.
  awk '
    /^entry[[:space:]]/ { h = 1; for (i = 1; i <= NF; i++) { if ($i == "spill") s = i; if ($i == "jit_local") j = i }; next }
    /^ptx-scan-md5:/ { exit }
    h && NF > 1 { print $1, $s, $j }
  ' "$TMP/$BIN.scan" > "$TMP/$BIN.got"
  awk -v b="$BIN" '!/^[[:space:]]*#/ && NF >= 4 && $1 == b { print $2, $3, $4 }' "$TMP/table.$T" > "$TMP/$BIN.want"
  if [ ! -s "$TMP/$BIN.got" ]; then
    echo "ptx-spill bin=$BIN scan printed no entry rows FAIL"
    rc=1
    continue
  fi
  # The pins by file name, not NR == FNR: a binary with no pinned row has an empty first file, and
  # NR == FNR would then read its scan as the pins and print every entry as stale.
  out=$(awk -v b="$BIN" '
    FILENAME == ARGV[1] { ws[$1] = $2; wj[$1] = $3; next }
    {
      seen[$1] = 1
      if (!($1 in ws)) { printf "%s %s unpinned spill=%s jit_local=%s\n", b, $1, $2, $3; next }
      if ($2 != ws[$1]) printf "%s %s spill=%s pinned=%s %s\n", b, $1, $2, ws[$1], ($2 + 0 > ws[$1] + 0 ? "ABOVE" : "BELOW")
      if ($3 != wj[$1]) printf "%s %s jit_local=%s pinned=%s %s\n", b, $1, $3, wj[$1], ($3 + 0 > wj[$1] + 0 ? "ABOVE" : "BELOW")
    }
    END { for (e in ws) if (!(e in seen)) printf "%s %s stale\n", b, e }
  ' "$TMP/$BIN.want" "$TMP/$BIN.got" | sort)
  n=$(wc -l < "$TMP/$BIN.got" | tr -d ' ')
  p=$(wc -l < "$TMP/$BIN.want" | tr -d ' ')
  nz=$(awk '$2 != 0 || $3 != 0 { printf "%s%s(spill=%s,jit_local=%s)", c, $1, $2, $3; c = "," }' "$TMP/$BIN.got")
  if [ -n "$out" ]; then
    echo "$out"
    v=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=$v FAIL"
    rc=1
  else
    echo "ptx-spill bin=$BIN entries=$n pinned=$p nonzero=${nz:-none} violations=0 PASS"
  fi
done
exit "$rc"
