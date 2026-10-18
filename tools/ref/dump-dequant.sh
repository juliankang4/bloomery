#!/usr/bin/env bash
# dump-dequant.sh — the dequantization oracle's three sets, written once and read by `just gate-1-1`
# (and, for the synthetic blocks, the i-quant row gates). The writer behind `just dump-ref-dequant`;
# the readers open each set through its refset family (crates/refset/src/arch/dequant.rs), which
# refuse a set that another harness, another ggml library or another model file wrote.
#
# Runs under tools/box.sh (the toolchain env is sourced). It builds the harness (build-dequant.sh),
# then runs it three times:
#
#   $BLOOMERY_DATA/ref-dequant-v2lite         the default profile's file (V2-Lite), every type it holds
#   $BLOOMERY_DATA/ref-dequant-v41<suffix>    the V4.1 file the tree runs ($BLOOMERY_V41_MODEL), every
#                                             type it holds; the suffix is the profile's V41_SET_SUFFIX
#   $BLOOMERY_DATA/ref-synth                  `dequant_ref --synthetic`: the types no model file holds
#
# and writes DEQUANT.tsv into each, last: the harness executable and the libggml.so it loads (path,
# md5), the model file (or `(synthetic rows)`), one row per file dequant_ref wrote (bytes, md5) and
# the `# complete` trailer. The old DEQUANT.tsv goes first, so a set is unfinished while its files
# change.
#
# The two model sets have directories of their own: the `ref` and `ref-v41<suffix>` directories are
# where the gate before the identity file dumped them, and a checkout that still runs that gate
# rewrites them. `ref-synth` stays where it was, because the i-quant row gates read its `.blocks`;
# that gate rewrites the same bytes into it (the harness is deterministic), and its manifest.txt
# is written by rename.
#
# Not a measurement: no lease, no GPU, a few seconds.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/tools/ref/ref-paths.sh"

# dump <dir> <model> <dequant_ref argument>… — the harness into <dir>, then <dir>/DEQUANT.tsv. Every digest and size is
# read into a variable before anything is written, so a failed read ends the script with no DEQUANT.tsv.
dump() {
  local dir=$1 model=$2 types f n=0 bin_md5 lib_md5 bytes sum rows=''
  shift 2
  rm -f "$dir/DEQUANT.tsv"
  "$BIN" "$@"
  types=$(awk '{ print $2 }' "$dir/manifest.txt")
  bin_md5=$(file_md5 "$BIN")
  lib_md5=$(file_md5 "$LIBGGML")
  for f in manifest.txt $(for t in $types; do for x in raw meta blocks; do [ -f "$dir/$t.$x" ] && echo "$t.$x"; done; done); do
    bytes=$(wc -c < "$dir/$f")
    sum=$(file_md5 "$dir/$f")
    rows+=$f$'\t'$bytes$'\t'$sum$'\n'
    n=$((n + 1))
  done
  {
    printf '# dequant_ref\t%s\t%s\n' "$BIN" "$bin_md5"
    printf '# libggml\t%s\t%s\n' "$LIBGGML" "$lib_md5"
    printf '# model\t%s\n' "$model"
    printf 'file\tbytes\tmd5\n'
    printf '%s' "$rows"
    printf '# complete\t%s\n' "$n"
  } > "$dir/DEQUANT.tsv.tmp.$$"
  mv "$dir/DEQUANT.tsv.tmp.$$" "$dir/DEQUANT.tsv"
  head -4 "$dir/DEQUANT.tsv"
}

# `bash tools/ref/dump-dequant.sh --self-test`: dump() on a stand-in harness, with md5sum and wc stand-ins first on PATH
# (tools/ref/read-stubs.sh). A whole set gets its rows and trailer; a digest or size read that fails, in the header or
# in a row, ends dump() with no DEQUANT.tsv and no temporary file. No box, no ggml.
if [ "${1:-}" = --self-test ]; then
  set +e
  t=$(mktemp -d "${TMPDIR:-/tmp}/dump-dequant-test.XXXXXX") || exit 70
  trap 'rm -rf "$t"' EXIT
  # shellcheck source=tools/ref/read-stubs.sh
  source "$HERE/tools/ref/read-stubs.sh"
  mkdir "$t/bin"
  read_stubs "$t/bin"
  PATH=$t/bin:$PATH
  BIN=$t/dequant_ref LIBGGML=$t/libggml.so bad=0 tab=$'\t'
  # The harness: one type, q4, whose raw file holds $RAW.
  printf '#!/bin/sh\necho "0 q4" > "$1/manifest.txt"\nprintf %%s "$RAW" > "$1/q4.raw"\necho m > "$1/q4.meta"\n' > "$BIN"
  chmod +x "$BIN"
  # run <name> <libggml text> <raw text>: dump into $t/<name>; RC and LEFT (the set's files after it).
  run() {
    mkdir "$t/$1"
    printf %s "$2" > "$LIBGGML"
    (set -e; export RAW=$3; dump "$t/$1" model "$t/$1") > "$t/$1.out" 2>&1
    RC=$?
    LEFT=$(cd "$t/$1" && ls | tr '\n' ' ')
  }
  run whole lib r
  check whole-rc "$RC" 0
  check whole-trailer "$(tail -n 1 "$t/whole/DEQUANT.tsv")" "# complete${tab}3"
  check whole-row "$(grep '^q4.raw' "$t/whole/DEQUANT.tsv")" "q4.raw${tab}7${tab}0123456789abcdef0123456789abcdef"
  check whole-lib "$(grep '^# libggml' "$t/whole/DEQUANT.tsv")" "# libggml${tab}$LIBGGML${tab}0123456789abcdef0123456789abcdef"
  for c in "header-md5 fail-md5 r" "row-md5 lib fail-md5" "row-wc lib fail-wc"; do
    read -r name lib raw <<< "$c"
    run "$name" "$lib" "$raw"
    check "$name-rc" "$([ "$RC" != 0 ] && echo failed || echo "0: $(tail -n 1 "$t/$name/DEQUANT.tsv" 2> /dev/null)")" failed
    check "$name-no-manifest" "$LEFT" "manifest.txt q4.meta q4.raw "
  done
  echo "dump-dequant self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
  exit
fi

[ "$BLOOMERY_MODEL" = deepseek2 ] || {
  echo "dump-dequant: the V2-Lite set is the default profile's (deepseek2); BLOOMERY_MODEL=$BLOOMERY_MODEL would dump another file as it" >&2
  exit 64
}
V41=${BLOOMERY_V41_MODEL:?BLOOMERY_V41_MODEL unset — run through tools/box.sh, which exports it from the deepseek41 profile}
SUFFIX=$(. "$HERE/tools/ref/models/deepseek41.sh" && printf %s "$V41_SET_SUFFIX")

bash "$HERE/tools/ref/build-dequant.sh"
BIN=$BLOOMERY_DATA/bin/dequant_ref
LIBGGML=$(ldd "$BIN" | awk '$1 == "libggml.so" { print $3 }')
[ -f "$LIBGGML" ] || { echo "dump-dequant: $BIN loads no libggml.so (ldd names '$LIBGGML')" >&2; exit 66; }

dump "$BLOOMERY_DATA/ref-dequant-v2lite" "$MODEL" "$MODEL" "$BLOOMERY_DATA/ref-dequant-v2lite"
dump "$BLOOMERY_DATA/ref-dequant-v41$SUFFIX" "$V41" "$V41" "$BLOOMERY_DATA/ref-dequant-v41$SUFFIX"
dump "$BLOOMERY_DATA/ref-synth" "(synthetic rows)" --synthetic "$BLOOMERY_DATA/ref-synth"
