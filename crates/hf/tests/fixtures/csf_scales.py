"""The compressed-scale fixture for crates/hf/src/csf.rs.

usage (Python 3, standard library only):
    python3 csf_scales.py fixture csf-scales.safetensors
    python3 csf_scales.py plain

`fixture` writes two kinds of scales, each under its checkpoint names:

- `qwen`: layer 0 expert 0 gate_proj's `weight_scale.nvfp4_csf_fixed` U8 [40, 1296] and
  `weight_scale.nvfp4_csf_exceptions` U32 of nvidia/Qwen3.8-Flash-Next-FP4-Local at REV, read by HTTP
  range reads from tensors/model-00001-of-00021.safetensors (640 rows by 160 scale groups);
- `syn_csf` (48 by 20) and `syn_csf32` (384 by 43: an odd column count, so the last block column is one
  wide, and 33 tiles, so two pages), seeded scale bytes around a row centre with a few far ones, encoded
  here by the encoders' rules (a row's base the start of the 16-wide window covering most of its bytes, the
  lowest on a tie, at most 240; a byte the window misses an exception), beside the bytes they encode as
  `<stem>.weight_scale` F8_E4M3.

`plain` prints the sha256 of the same gate_proj's plain `weight_scale` in
local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at PLAIN_REV, the uncompressed copy of the same weights, which
the decode of `qwen` must equal.
"""
import hashlib
import json
import random
import struct
import sys
import urllib.request

from modelopt_nvfp4 import write

REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
SHARD = "tensors/model-00001-of-00021.safetensors"
PLAIN_REV = "6909a5be"
PLAIN = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
STEM = "model.language_model.layers.0.mlp.experts.0.gate_proj"


def get(repo, rev, path, a=None, b=None):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    headers = {} if a is None else {"Range": f"bytes={a}-{b - 1}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as f:
        data = f.read()
    assert a is None or len(data) == b - a, (path, a, b, len(data))
    return data


def tensor(repo, rev, path, name):
    (n,) = struct.unpack("<Q", get(repo, rev, path, 0, 8))
    entry = json.loads(get(repo, rev, path, 8, 8 + n))[name]
    a, b = entry["data_offsets"]
    return entry, get(repo, rev, path, 8 + n + a, 8 + n + b)


def scales(rng, rows, cols):
    out = []
    for _ in range(rows):
        centre = rng.randrange(20, 110)
        row = [centre + rng.randrange(-5, 10) for _ in range(cols)]
        for c in range(cols):
            if rng.random() < 0.04:
                row[c] = rng.randrange(0, 127)
        out.append(row)
    return out


def base(row):
    cover = [sum(b <= v < b + 16 for v in row) for b in range(241)]
    return cover.index(max(cover))


def pack(codes):
    codes = codes + [0] * (len(codes) % 2)
    return bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))


def code(v, b):
    return v - b if 0 <= v - b < 16 else 0


def csf(m):
    rows, cols = len(m), len(m[0])
    bases = [base(r) for r in m]
    fixed = b""
    for s in range(0, rows, 16):
        fixed += bytes(bases[s : s + 16])
        fixed += pack([code(v, bases[r]) for r in range(s, s + 16) for v in m[r]])
    words = [(r * cols + c) | m[r][c] << 24 for r in range(rows) for c in range(cols) if code(m[r][c], bases[r]) + bases[r] != m[r][c]]
    return fixed, struct.pack(f"<{len(words)}I", *words)


def csf32(m):
    rows, cols = len(m), len(m[0])
    bases = [base(r) for r in m]
    block_rows, block_cols = rows // 32, (cols + 1) // 2
    tile_cols = (block_cols + 1) // 2
    tiles = (block_rows + 3) // 4 * tile_cols
    fields, records, page_bases, offsets = [], [], [], []
    for t in range(tiles):
        if t % 32 == 0:
            page_bases.append(2 * len(records))
            offsets.append(0)
            first = len(records)
        r, c = t // tile_cols * 4, t % tile_cols * 2
        for bc in range(c, min(c + 2, block_cols)):
            for br in range(r, min(r + 4, block_rows)):
                r0, c0 = 32 * br, 2 * bc
                w = min(2, cols - c0)
                for i in range(32):
                    for j in range(w):
                        v, b = m[r0 + i][c0 + j], bases[r0 + i]
                        fields.append(code(v, b))
                        if code(v, b) + b != v:
                            records.append((2 * i + j) | v << 8)
                offsets.append(len(records) - first)
    return (
        bytes(bases),
        pack(fields),
        struct.pack(f"<{len(page_bases)}I", *page_bases),
        struct.pack(f"<{len(offsets)}H", *offsets),
        struct.pack(f"<{len(records)}H", *records),
    )


def fixture(path):
    tensors = []
    for part in ("nvfp4_csf_fixed", "nvfp4_csf_exceptions"):
        name = f"{STEM}.weight_scale.{part}"
        entry, raw = tensor("nvidia/Qwen3.8-Flash-Next-FP4-Local", REV, SHARD, name)
        tensors.append((f"qwen.weight_scale.{part}", entry["dtype"], entry["shape"], raw))
    rng = random.Random(20261019)
    m = scales(rng, 48, 20)
    fixed, exc = csf(m)
    tensors += [
        ("syn_csf.weight_scale", "F8_E4M3", [48, 20], bytes(v for r in m for v in r)),
        ("syn_csf.weight_scale.nvfp4_csf_fixed", "U8", [3, 16 * 11], fixed),
        ("syn_csf.weight_scale.nvfp4_csf_exceptions", "U32", [len(exc) // 4], exc),
    ]
    m = scales(rng, 384, 43)
    parts = csf32(m)
    tensors.append(("syn_csf32.weight_scale", "F8_E4M3", [384, 43], bytes(v for r in m for v in r)))
    for part, dtype, size, raw in zip(("bases", "codes", "page_bases", "offsets", "exceptions"), ("U8", "U8", "U32", "U16", "U16"), (1, 1, 4, 2, 2), parts):
        tensors.append((f"syn_csf32.weight_scale.csf32_{part}", dtype, [len(raw) // size], raw))
    write(path, tensors, f"nvidia/Qwen3.8-Flash-Next-FP4-Local {REV} {SHARD} {STEM}; syn_* seeded 20261019")


def plain():
    path = json.loads(get(PLAIN, PLAIN_REV, "model.safetensors.index.json"))["weight_map"][f"{STEM}.weight_scale"]
    _, raw = tensor(PLAIN, PLAIN_REV, path, f"{STEM}.weight_scale")
    print(len(raw), hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    {"fixture": lambda: fixture(sys.argv[2]), "plain": plain}[sys.argv[1]]()
