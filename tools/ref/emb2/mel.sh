#!/usr/bin/env bash
# Build the host-only gemma4a oracle against the existing pinned llama.cpp
# tree, then write a new synthetic PCM set. No fetch or model is needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source tools/ref/models/gemma-embedding2.sh
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
head=$(git -c safe.directory="$LCPP" -C "$LCPP" rev-parse HEAD)
[ "$head" = "$LCPP_PIN" ] || { echo "mel: llama.cpp is $head, want $LCPP_PIN" >&2; exit 2; }
status=$(git -c safe.directory="$LCPP" -C "$LCPP" status --porcelain --untracked-files=no)
[ -z "$status" ] || { echo "mel: llama.cpp has tracked changes" >&2; exit 2; }
OUT=$DATA/ref_gemma_embedding2_mel
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || { echo "mel: $OUT exists; move it aside before dumping" >&2; exit 2; }
mkdir -p target "$DATA"
CXX=${CXX:-c++}
"$CXX" -std=c++17 -O2 -ffp-contract=off -pthread \
    -I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/tools/mtmd" \
    tools/ref/emb2/dump_mel.cpp "$LCPP/tools/mtmd/mtmd-audio.cpp" \
    -L"$LCPP/build/bin" -Wl,-rpath,"$LCPP/build/bin" -lmtmd -lggml-base -o target/dump_mel
scratch=$(mktemp -d "$DATA/.emb2-mel.XXXXXX")
echo "mel: staging set in $scratch"
LD_LIBRARY_PATH="$LCPP/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    BLOOMERY_REF_BUILD=${LCPP_PIN:0:9} target/dump_mel "$scratch"
mv -Tn -- "$scratch" "$OUT"
[ ! -e "$scratch" ] || { echo "mel: $OUT appeared while dumping; set not published" >&2; exit 2; }
echo "mel: published $OUT from $head"
