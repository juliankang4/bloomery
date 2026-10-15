# EmbeddingGemma reference sets

Both encoder versions use mainline llama.cpp at `6868a52cb337450658006ad2f9281c2eb63f1666`.
`build-embedding-ref <architecture>` and `dump-ref-embedding <architecture> [steps...]`
share the linker, dumper and set writer. Model profiles own the vocabulary, context,
files, digests, available steps and optional image projector.

For v1, with the pinned mainline tree already built:

```sh
just build-emb1-ref
just dump-ref-tokenizer llama
just dump-ref-emb1
```

`build-emb1-ref` selects `gemma-embedding`; `build-emb2-ref` selects
`gemma-embedding2`. The v2 recipe names and step arguments remain available.
Both use `tools/ref/emb2/dump_emb2.cpp` and the same mainline build.
Set `LCPP` to that tree on another machine.

V1 writes `ref_gemma_embedding_text`, `ref_gemma_embedding_text.cpu` and
`ref_gemma_embedding_server`. Their families are `text-gemma-embedding`,
`text-gemma-embedding-cpu` and `server-gemma-embedding`. They pin the v1 model
file, its SHA256, architecture, oracle commit, set kind and device. The text-only
profile has no tower step or projector requirement.

In a v1 text set, `embd` holds the API's token rows after the two learned dense
projections, equal to the tap `result_embd`. `result_norm` holds the RMS-normalized
rows before that dense head. `pooled` holds the API's mean-pooled vector after
the dense head. The taps also include every `l_out-<layer>`. `tap_effect` compares
`result_embd_pooled` (stored as `result_embd`) with the clean pass in both versions.
The dense projections follow pooling in the pooled pass and operate on individual
rows when pooling is NONE.

The shared tokenizer cases precede `cases-emb1.txt`. The extra file includes
paired distant word substitutions and an input of 1500 words. In the pair whose
ids differ at token 512, compare the first-layer rows at query 256 (distance 256,
inside the window) and query 255 (distance 257, outside). This inspects the
symmetric window before later layers propagate information from farther away.
The server set contains raw and normalized vectors; its MRL rows are derived
and marked `DERIVED`, as in the v2 set.

The tokenizer set includes the five corpus snapshots as well as the fixed cases.
Its default profile's `IK` tree must contain `src/`, `docs/`, `README.md` and
`github-data/`, as for the v2 tokenizer set. `LCPP` selects the tokenizer executable;
`IK` selects those corpus texts.
