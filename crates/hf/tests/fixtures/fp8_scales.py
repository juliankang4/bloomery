"""The FP8 scale-layout fixture and its reference for crates/hf/src/quant_config.rs.

usage (Python 3, standard library only):
    python3 fp8_scales.py slice fp8-scales.safetensors
    python3 fp8_scales.py ref fp8-scales.safetensors > fp8-scales-ref.json
    python3 fp8_scales.py --self-test

`slice` writes the fixture from HTTP range reads of three checkpoints, each tensor under its own
name, cut to the rows (and for the 32 by 32 blocks the values) listed in SLICES:

- openbmb/MiniCPM-V-4.7-35B-A3B-FP8: two routed experts' down_proj, F8_E4M3 with a BF16
  weight_scale [rows, 1], one scale a row (compressed-tensors, strategy channel);
- nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8: a mixer out_proj, F8_E4M3 with one F32 weight_scale
  (ModelOpt FP8), and one row of an in_proj, whose 3136 values a row are not whole 128-value
  blocks;
- deepseek-ai/DeepSeek-V4.1-Flash: a shared expert's w2, F8_E4M3 with an F8_E8M0 scale, one a
  32 by 32 block, cut to rows 2400..2440 (two rows of blocks) and values 768..1024 (eight
  blocks), where the scales differ along both axes.

`ref` dequantizes every F8_E4M3 weight of the fixture by its producer's rule, value c of row r
being e4m3(weight[r][c]) times the scale of the block holding (r, c): the row's for a channel
scale, the one for a tensor scale, 2^(e - 127) for the E8M0 byte e of a 32 by 32 block. Each
format is decoded from its definition (E4M3: sign, four exponent bits of bias 7, three mantissa
bits, 0x7F and 0xFF NaN; BF16: the high half of an f32). The product is formed in f64, where it is
exact (a 4-bit significand times at most 24 bits), and rounded once to f32. It prints, per weight,
the sha256 of each row's f32 bits, little-endian.
"""
import hashlib
import json
import math
import struct
import sys

from csf_scales import get
from modelopt_nvfp4 import e4m3, header, write

# (repository, revision, shard, [(stem, first row, rows, first value, values or None for whole
# rows)]); a weight's scale companions follow it, cut to the same rows and blocks.
SLICES = [
    (
        "openbmb/MiniCPM-V-4.7-35B-A3B-FP8",
        "25f6715c91b8e528d45532d4a3cbcb206c59fb47",
        "model-00002-of-00002.safetensors",
        [
            ("model.language_model.layers.20.mlp.experts.148.down_proj", 0, 4, 0, None),
            ("model.language_model.layers.20.mlp.experts.149.down_proj", 0, 4, 0, None),
        ],
    ),
    (
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
        "3fe6dab75665a93884214ad4b1b95cf02717d081",
        "model-00001-of-00002.safetensors",
        [
            ("backbone.layers.0.mixer.in_proj", 0, 1, 0, None),
            ("backbone.layers.0.mixer.out_proj", 0, 2, 0, None),
        ],
    ),
    (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "314113f8a7dadb388215396ba5fbee2c610a04d6",
        "model-00024-of-00048.safetensors",
        [("layers.21.ffn.shared_experts.w2", 2400, 40, 768, 256)],
    ),
]
SCALES = ("weight_scale", "scale", "input_scale")


def cut(raw, shape, rows, c0, cols, unit):
    """Values c0..c0 + cols of the `rows` rows of a row-major [r, c] tensor of `unit`-byte items."""
    c = shape[1]
    return b"".join(raw[(r * c + c0) * unit : (r * c + c0 + cols) * unit] for r in range(rows))


def slice_fixture(out):
    tensors, sources = [], []
    for repo, rev, shard, stems in SLICES:
        (n,) = struct.unpack("<Q", get(repo, rev, shard, 0, 8))
        head, base = header(get(repo, rev, shard, 0, 8) + get(repo, rev, shard, 8, 8 + n))
        for stem, r0, rows, c0, cols in stems:
            r, c = head[f"{stem}.weight"]["shape"]
            cols = cols or c
            for suffix in ("weight",) + SCALES:
                name = f"{stem}.{suffix}"
                if name not in head:
                    continue
                e = head[name]
                a, b = e["data_offsets"]
                shape = e["shape"]
                unit = (b - a) // math.prod(shape)
                if shape in ([], [1]):
                    raw = get(repo, rev, shard, base + a, base + b)
                    tensors.append((name, e["dtype"], shape, raw))
                    continue
                if suffix == "weight":
                    first, keep = [r0, c0], [rows, cols]
                elif shape == [r, 1]:
                    first, keep = [r0, 0], [rows, 1]
                else:
                    # One scale a 32 by 32 block: the blocks the kept rows and values lie in.
                    assert r0 % 32 == 0 and c0 % 32 == 0, stem
                    first = [r0 // 32, c0 // 32]
                    keep = [-(-(r0 + rows) // 32) - first[0], -(-(c0 + cols) // 32) - first[1]]
                row = (b - a) // shape[0]
                raw = get(repo, rev, shard, base + a + first[0] * row, base + a + (first[0] + keep[0]) * row)
                if keep[1] != shape[1]:
                    raw = cut(raw, shape, keep[0], first[1], keep[1], unit)
                tensors.append((name, e["dtype"], keep, raw))
            values = f" values {c0}..{c0 + cols}" if cols != c else ""
            sources.append(f"{repo}@{rev} {shard} {stem} rows {r0}..{r0 + rows}{values}")
    write(out, tensors, "; ".join(sources))


def scale_values(dtype, raw):
    if dtype == "F32":
        values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    elif dtype == "BF16":
        values = [struct.unpack("<f", struct.pack("<I", h << 16))[0] for (h,) in struct.iter_unpack("<H", raw)]
    elif dtype == "F8_E8M0":
        if 0xFF in raw:
            raise ValueError("scale byte 0xff is E8M0's NaN")
        values = [2.0 ** (e - 127) for e in raw]
    else:
        raise ValueError(dtype)
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{dtype} scale is not finite")
    return values


def reference(path):
    with open(path, "rb") as f:
        blob = f.read()
    head, base = header(blob)

    def data(name):
        a, b = head[name]["data_offsets"]
        return blob[base + a : base + b]

    out = {"source": head["__metadata__"]["source"], "weights": {}}
    for name in head:
        if not name.endswith(".weight") or head[name]["dtype"] != "F8_E4M3":
            continue
        stem = name[: -len(".weight")]
        rows, k = head[name]["shape"]
        sname = next(f"{stem}.{s}" for s in SCALES if f"{stem}.{s}" in head)
        shape = head[sname]["shape"]
        s = scale_values(head[sname]["dtype"], data(sname))
        if shape in ([], [1]):
            block = (rows, k)
        elif shape == [rows, 1]:
            block = (1, k)
        else:
            block = (32, 32)
        c = -(-k // block[1])
        w = data(name)
        digests = []
        for r in range(rows):
            row = bytearray()
            for j in range(k):
                x = e4m3(w[r * k + j]) * s[r // block[0] * c + j // block[1]]
                row += struct.pack("<f", x)
            digests.append(hashlib.sha256(row).hexdigest())
        out["weights"][name] = digests
    json.dump(out, sys.stdout, indent=1)
    print()


def self_test():
    import contextlib
    import io
    from pathlib import Path
    import tempfile
    from unittest.mock import patch

    here = Path(__file__).parent
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        reference(here / "fp8-scales.safetensors")
    assert json.loads(output.getvalue()) == json.loads((here / "fp8-scales-ref.json").read_text())
    for dtype, raw, expected in (
        ("F32", struct.pack("<3f", 0, -1, 2**127), [0, -1, 2**127]),
        ("BF16", struct.pack("<3H", 0, 0xBF80, 0x7F00), [0, -1, 2**127]),
        ("F8_E8M0", bytes([0, 127, 254]), [2**-127, 1, 2**127]),
    ):
        assert scale_values(dtype, raw) == expected, dtype

    accepted = []
    with tempfile.TemporaryDirectory() as tmp:
        source, sliced = Path(tmp) / "source", Path(tmp) / "sliced"
        tensors = []
        for stem, dtype, shape, raw in (
            ("tensor", "F32", [], struct.pack("<f", 2)),
            ("channel", "BF16", [64, 1], struct.pack("<64H", *range(0x3F80, 0x3FC0))),
            ("block", "F8_E8M0", [2, 2], bytes([127, 128, 129, 130])),
        ):
            tensors.extend([
                (f"{stem}.weight", "F8_E4M3", [64, 64], bytes(range(64)) * 64),
                (f"{stem}.weight_scale", dtype, shape, raw),
            ])
        write(source, tensors, "test")
        blob = source.read_bytes()
        slices = [("repo", "rev", "shard", [(stem, 32, 2, 32, 32) for stem in ("tensor", "channel", "block")])]
        with patch.dict(slice_fixture.__globals__, SLICES=slices), patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = lambda req, timeout: io.BytesIO(
                blob[int(req.headers["Range"].split("=")[1].split("-")[0]):
                     int(req.headers["Range"].split("-")[1]) + 1]
            )
            slice_fixture(sliced)
        sliced_blob = sliced.read_bytes()
        head, base = header(sliced_blob)
        for name, dtype, shape, raw in tensors:
            entry = head[name]
            if name.endswith(".weight"):
                shape, raw = [2, 32], bytes(range(32, 64)) * 2
            elif shape == [64, 1]:
                shape, raw = [2, 1], raw[64:68]
            elif shape == [2, 2]:
                shape, raw = [1, 1], raw[3:4]
            a, b = entry["data_offsets"]
            assert (entry["dtype"], entry["shape"], sliced_blob[base + a:base + b]) == (dtype, shape, raw), name

        for suffix in SCALES:
            for dtype, raw in (
                ("F8_E8M0", b"\xff"),
                *(("F32", struct.pack("<f", v)) for v in (math.nan, math.inf, -math.inf)),
                *(("BF16", struct.pack("<H", v)) for v in (0x7FC0, 0x7F80, 0xFF80)),
            ):
                write(source, [("test.weight", "F8_E4M3", [1, 1], b"\x01"),
                               (f"test.{suffix}", dtype, [], raw)], "test")
                output = io.StringIO()
                try:
                    with contextlib.redirect_stdout(output):
                        reference(source)
                except ValueError as error:
                    assert "NaN" in str(error) or "not finite" in str(error), error
                    assert not output.getvalue()
                else:
                    accepted.append(f"{suffix} {dtype} {raw.hex()}")
    assert not accepted, f"undefined scales accepted: {accepted}"
    print("fp8_scales: fixture hashes, range slices and 21 undefined-scale refusals ok")


if __name__ == "__main__":
    if sys.argv[1] == "--self-test":
        self_test()
    else:
        {"slice": slice_fixture, "ref": reference}[sys.argv[1]](sys.argv[2])
