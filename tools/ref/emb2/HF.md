# HF text reference for EmbeddingGemma 2

## 1. Generate the second reference

`hf_vectors.py` runs the official sentence-transformers pipeline on the E3 text
cases with the original Hugging Face (HF) weights. It writes the
`hf-gemma-embedding2` refset family beside the mainline llama.cpp sets.
This workflow needs a Linux NVIDIA machine, an isolated Python environment,
and the existing `ref-tokenizer-gemma4` set from `just dump-ref-tokenizer gemma4`.

The package pins are in `crates/refset/src/emb2/hf-build.txt`. The model revision
is in `crates/refset/src/arch/gemma_embedding2/hf-checkpoint.txt`. Install those
versions into a venv, then run the generator while holding the machine's GPU
lock. Set `DATA` to the directory containing the E3 reference sets:

```sh
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install $(tr ' ' '\n' < crates/refset/src/emb2/hf-build.txt | sed 's/=/==/')
.venv-hf/bin/python tools/ref/emb2/hf_vectors.py \
  --ids-dir "$DATA/ref-tokenizer-gemma4" \
  --checkpoint-dir "$HOME/.cache/bloomery/embeddinggemma2-hf" \
  --out "$DATA/ref_gemma_embedding2_hf"
(cd "$DATA/ref_gemma_embedding2_hf" && sha256sum -c SHA256SUMS)
```

The output directory must be empty. The manifest is completed after every
vector has been written.
The generator checks the original weights against the pinned HF revision's LFS
digest and records the checkpoint files' SHA256 values. It uses float32, eager
attention, no task prefix, and the checkpoint's sentence-transformers pooling
and normalization modules. Vision and audio encoders are not loaded.

## 2. Compare the vectors

```sh
BLOOMERY_DATA="$DATA" BLOOMERY_BOX_LOCAL=1 just gate-gpu-gemma-embedding2-e2e --hf
```

The normal v2 encoder gate also runs this clause. `--hf` limits the encoder
binary to the HF comparisons and window check; the recipe still runs its unit
checks. The gate prints cosine, maximum absolute difference and relative L2
distance for bloomery versus HF and mainline llama.cpp versus HF.

HF uses the original weights, while the mainline and bloomery references use
Q8_0. The HF clause freezes its reference-precision budget from the minimum
normalized mainline/HF cosine, 0.9991290441, calibrated over 84 E3 cases on
DGX Spark and RTX 5090. The derived unit-vector L2 budget is
`sqrt(2 * (1 - 0.9991290441))`, approximately 0.04174. The existing CPU-twin
rule uses 2.5 times that frozen budget, at least the existing 1e-2 floor.
An independent relative L2 ceiling of 5e-2 also applies to every comparison arm.
Neither bound depends on the vector under test. The existing per-layer CPU-twin
checks remain unchanged. Cosine and maximum absolute difference are diagnostic
values.

## 3. Sliding-window convention

The pinned HF text configuration has `sliding_window = 512`. Its eager text
model calls `create_bidirectional_sliding_window_mask`, whose predicate is
`abs(query - key) <= sliding_window`. The generator exercises the inclusive
boundary and the next excluded key on both sides, and records the implementation
files' hashes and source locations.

HF's value is a one-sided reach. The GGUF value is the whole span, 1024, which
bloomery divides by two. Both retain distances through 512 on either side,
including the query itself. The gate checks the HF reach against every sliding
layer and requires a case that extends beyond both window edges.

## 4. Exceptional text inputs

Invalid UTF-8 cases cannot enter the text API and receive explicit `skipped`
records. Literal BOS/EOS text can produce different token sequences in HF and
mainline. For these cases, the set retains both outputs: the sentence-transformers
forward pipeline on the exact E3 ids, and an extra `_hf` case from ordinary
`SentenceTransformer.encode(prompt="")` with HF's own ids. For matching ids,
the generator checks that forward and encode agree. The gate compares bloomery
against both sequences and mainline against the E3-aligned sequence. Extra
encode rows use the same frozen precision budget and independent ceiling.
