"""The ModelOpt NVFP4 fixture and its reference for crates/hf/src/modelopt.rs.

usage (Python 3, standard library only):
    python3 modelopt_nvfp4.py slice modelopt-nvfp4.safetensors
    python3 modelopt_nvfp4.py ref modelopt-nvfp4.safetensors > modelopt-nvfp4-ref.json

`slice` writes the fixture: the first ROWS rows of `gate_proj` of experts 0 and 1 of layer 0 (weight U8,
weight_scale F8_E4M3, weight_scale_2 and input_scale F32), read by HTTP range reads from
model-00001-of-00010.safetensors of nvidia/Qwen3.8-Flash-Next-NVFP4 at REV under their own names, and one
synthetic tensor `synthetic` whose sub-block scales run through every byte E4M3 defines as a positive
scale (0x00..=0x7E) and whose 16-value groups each hold all 16 codes, every value at another position.

`ref` dequantizes every NVFP4 tensor of the fixture by the ModelOpt rule, value c of row r being
e2m1(nibble c) * e4m3(weight_scale[r][c // 16]) * weight_scale_2, where nibble c is the low (c even) or
high (c odd) nibble of weight byte c // 2. Each format is decoded from its definition (E2M1: sign, two
exponent bits of bias 1, one mantissa bit; E4M3: sign, four exponent bits of bias 7, three mantissa bits,
0x7F and 0xFF NaN). The product is formed in f64, where it is exact (a 2-bit and a 4-bit significand times
an f32), and rounded once to f32. E2M1's -0 (code 8) is taken as +0, the 0 ggml's code table holds for it;
every other sign is the product's. It prints, per tensor, its scale and the sha256 of each row's f32 bits,
little-endian.
"""
import hashlib
import json
import struct
import sys
import urllib.request

REV = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
SHARD = "model-00001-of-00010.safetensors"
URL = f"https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/{REV}/{SHARD}"
ROWS = 4
EXPERTS = [f"model.language_model.layers.0.mlp.experts.{e}.gate_proj" for e in (0, 1)]


def get(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=120) as f:
        data = f.read()
    assert len(data) == b - a, (a, b, len(data))
    return data


def header(blob):
    (n,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8 : 8 + n]), 8 + n


def write(path, tensors, source):
    meta = {"format": "pt", "source": source}
    entries, at = {"__metadata__": meta}, 0
    for name, dtype, shape, raw in tensors:
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
    text = json.dumps(entries, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"".join(t[3] for t in tensors))


def synthetic():
    rows, k = 4, 512
    groups = rows * k // 16
    scales = bytes(g % 0x7F for g in range(groups))
    weight = bytearray()
    for g in range(groups):
        vals = [(j * 7 + g) & 15 for j in range(16)]
        weight += bytes(vals[2 * i] | vals[2 * i + 1] << 4 for i in range(8))
    return [
        ("synthetic.weight", "U8", [rows, k // 2], bytes(weight)),
        ("synthetic.weight_scale", "F8_E4M3", [rows, k // 16], scales),
        ("synthetic.weight_scale_2", "F32", [], struct.pack("<f", 0.3712)),
        ("synthetic.input_scale", "F32", [], struct.pack("<f", 0.5)),
    ]


def slice_fixture(out):
    (n,) = struct.unpack("<Q", get(0, 8))
    head, base = header(get(0, 8) + get(8, 8 + n))
    tensors = []
    for stem in EXPERTS:
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            name = f"{stem}.{suffix}"
            e = head[name]
            a, b = e["data_offsets"]
            shape = e["shape"]
            if shape:
                row = (b - a) // shape[0]
                shape, b = [ROWS] + shape[1:], a + ROWS * row
            tensors.append((name, e["dtype"], shape, get(base + a, base + b)))
    source = f"nvidia/Qwen3.8-Flash-Next-NVFP4@{REV} {SHARD} rows 0..{ROWS}; synthetic"
    write(out, tensors + synthetic(), source)


def e2m1(n):
    if n == 8:
        return 0.0
    sign = -1.0 if n & 8 else 1.0
    e, m = (n >> 1) & 3, n & 1
    return sign * (m * 0.5 if e == 0 else (1 + m / 2) * 2.0 ** (e - 1))


def e4m3(b):
    if b & 0x7F == 0x7F:
        raise ValueError(f"scale byte {b:#04x} is E4M3's NaN")
    sign = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 15, b & 7
    return sign * (m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def reference(path):
    with open(path, "rb") as f:
        blob = f.read()
    head, base = header(blob)

    def data(name):
        a, b = head[name]["data_offsets"]
        return blob[base + a : base + b]

    out = {"source": head["__metadata__"]["source"], "tensors": {}}
    for name in head:
        if not name.endswith(".weight"):
            continue
        stem = name[: -len(".weight")]
        rows, half = head[name]["shape"]
        w, s = data(name), data(f"{stem}.weight_scale")
        (s2,) = struct.unpack("<f", data(f"{stem}.weight_scale_2"))
        k = 2 * half
        digests = []
        for r in range(rows):
            row = bytearray()
            for c in range(k):
                nib = w[r * half + c // 2] >> (4 * (c % 2)) & 15
                x = e2m1(nib) * e4m3(s[r * (k // 16) + c // 16]) * s2
                row += struct.pack("<f", x)
            digests.append(hashlib.sha256(row).hexdigest())
        out["tensors"][stem] = {"scale": s2, "rows": digests}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    {"slice": slice_fixture, "ref": reference}[sys.argv[1]](sys.argv[2])
