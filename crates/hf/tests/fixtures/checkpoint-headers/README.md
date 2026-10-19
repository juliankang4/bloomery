# Checkpoint shard headers

Real safetensors headers, gzipped, for `crates/hf/src/safetensors.rs`'s folder tests. Each `.header.json.gz` is the JSON header of one shard exactly as the file stores it (the bytes after its 8-byte length), read by an HTTP range request. The tests write each one as a shard whose data section is a hole, so the files have the real layout and no tensor bytes.

| file | source |
|---|---|
| `glm-5.3-flash-nvfp4-model-00001-of-00033.header.json.gz`, `...-00002-of-00033...` | `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348`, `model-00001-of-00033.safetensors` and `model-00002-of-00033.safetensors` |
| `glm-5.3-flash-nvfp4-index-00001-00002.json.gz` | the same revision's `model.safetensors.index.json`, its `weight_map` cut to the entries of those two shards (5,681 of 147,661) |
| `deepseek-v4.1-flash-model-00024-of-00048.header.json.gz` | `deepseek-ai/DeepSeek-V4.1-Flash` at `314113f8a7dadb388215396ba5fbee2c610a04d6`, `model-00024-of-00048.safetensors` |
| `glm-5.3-flash-fp4-local-layers-0-3-4.tensors.json.gz` | `nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`: the dtype and shape of every tensor of layers 0, 3 and 4 and every unlayered `model.language_model.` tensor and `lm_head`, gathered from all 42 safetensors headers (the vision tower left out) |
| `qwen3.8-flash-next-fp4-local-ple-part-0.tensors.json.gz` | `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6887e18c089fdea8509a33e63429a3300`: the dtype and shape of the first of the 128 parts of layer 1's PLE table (`ple.ple_embedding.ngram_embedding.shard_0`) and of the one `weight_scale_2` the parts share |
