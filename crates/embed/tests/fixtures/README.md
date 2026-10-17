# 1. Embedding response fixtures

These fixtures let `cargo test -p bloomery-embed --test oracle -- --nocapture` compare the host embedding path with llama.cpp without loading a model or opening a GPU.

`llama-server.json` contains four `/v1/embeddings` responses from llama.cpp revision `6868a52cb337450658006ad2f9281c2eb63f1666`, the [EmbeddingGemma 2 reference pin](../../../refset/src/emb2.rs). The requests use token ids, float and base64 encoding, and normalization modes -1 and 2. `pooled_bits` is the raw base64 response decoded as little-endian f32 bit patterns. The model SHA256 is in the fixture and checked against [the reference profile](../../../refset/src/arch/gemma_embedding2/mod.rs).

# 2. Capture and comparison

With that revision built and the profile's Q8_0 model in `$MODEL`, start the CPU oracle with:

```sh
llama-server -m "$MODEL" --alias embeddinggemma-2 --embedding --pooling mean \
    -ngl 0 -c 128 -b 128 -ub 128 -t 4 --host 127.0.0.1 --port 8097
```

Send each stored `request` to `/v1/embeddings` and retain the response object. The test sorts JSON object keys before comparing serialized bytes. Float array values round-trip through f32 before serialization, since the contract is f32 values rather than a decimal spelling. Base64 strings are compared unchanged. The test checks every response field, including model, index, encoding and token usage.

# 3. MRL reference vectors

`mrl.json` holds the raw vector of case `cases/01` of `ref_gemma_embedding2_server.cpu`. Its manifest SHA256 identifies the source set. The expected 128, 256 and 512 dimensional vectors come from the same pinned llama.cpp tree's `common_embd_normalize`, linked from its common library. They are not HTTP responses or the Python-derived MRL tensors in the reference set. The test compares f32 base64 bytes after the production path cuts the raw vector and renormalizes it.

Regenerate the raw source set with `BLOOMERY_MODEL=gemma-embedding2 just dump-ref-embedding gemma-embedding2 server.cpu`. For each width `n`, pass the first `n` raw f32 values to `common_embd_normalize(raw, out, n, 2)` and retain `out` as little-endian f32 base64. The reference-set Python derivation uses f64 arithmetic for scaling, so it is not a byte oracle for the C++ normalizer.
