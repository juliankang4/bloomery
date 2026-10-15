#!/usr/bin/env python3
"""Collect llama-server's EmbeddingGemma vectors into an oracle set in dump_ref's set format: the end-to-end answer
bloomery's /v1/embeddings is compared against.

  server_vectors.py --url <http://host:port> --pid <server pid> --model <gguf> --model-sha256 <hex> --arch <architecture>
                    --card <name> --build-record <file> --ids-dir <dir> --case <name> ... --out <dir>
                    [--mmproj <gguf> --mmproj-sha256 <hex> --image <name>=<png> ...] [--mix-image <name>]
                    [--server-flags <text>]

The server is the caller's (tools/ref/emb2/emb2.sh starts it): llama-server --embeddings with the file's mean pooling
and a ubatch that holds the longest case. Before any request and after the last, the pid the caller started (--pid) must
be alive and the server at --url must answer /props with the model it was given and a build_info of the commit
BLOOMERY_REF_BUILD names; whether that pid is the one answering is not checked here (emb2.sh refuses a port already
taken). The model and optional projector's sha256 are computed here and must be the ones given.
Per text case (<ids-dir>/<case>.txt and its ids, the profile's tokenizer set), two /v1/embeddings requests, both
encoding_format base64 so the f32 bytes are the server's own:
  <case>/embd    the default answer: the mean-pooled vector, L2 normalized (embd_normalize 2)          [768, 1]
  <case>/raw     the same request with embd_normalize -1: the pooled vector before normalization       [768, 1]
  <case>/mrl<k>  derived, not the server's: the first k values of raw, L2 normalized in f64 and rounded to f32, for
                 each Matryoshka width k the model card names (128, 256, 512); llama-server has no `dimensions` field
A case that is not UTF-8 cannot be a JSON string: it gets a `# skipped` line naming it and no rows. Each other case's ids
are checked against /tokenize (add_special, parse_special: what the embedding handler does) and the oracle's ids; a
`# tokenize` line records the comparison and the usage line the server's prompt_tokens, and a difference stops the run.
Per --image, the request is one {"content": [{"type": "image_url", ...}]} object (a data: URL), rows `img_<name>/embd`
and `img_<name>/raw`; --mix-image adds one request with a text part before that image (`mix_<name>`).
The header carries what the node-dump reader and the server family check: `# model`, `# arch`, `# build`
(BLOOMERY_REF_BUILD), `# emb2 server`, `# device` (cuda and the card), `# model_sha256`, and, when a projector is used,
`# mmproj <path> sha256 <hex>`;
then the server's flags and build, the build record's lines, and the `# complete` trailer, written last through a .partial
name. Python's standard library only.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import urllib.request
from pathlib import Path

MRL = (128, 256, 512)


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def embed(url: str, entry, normalize: int) -> tuple[list[float], int]:
    body = {"input": [entry], "encoding_format": "base64", "embd_normalize": normalize}
    r = post(url, "/v1/embeddings", body)
    raw = base64.b64decode(r["data"][0]["embedding"])
    return list(struct.unpack(f"<{len(raw) // 4}f", raw)), r["usage"]["prompt_tokens"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SetWriter:
    def __init__(self, out: Path):
        self.out, self.header, self.rows = out, [], []
        out.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, xs: list[float], op: str) -> None:
        stem = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        data = struct.pack(f"<{len(xs)}f", *xs)
        if not all(math.isfinite(x) for x in xs):
            raise SystemExit(f"server_vectors: {name} holds a value that is not finite")
        (self.out / f"{stem}.0.f32").write_bytes(data)
        total = sum(struct.unpack(f"<{len(xs)}f", data))
        self.rows.append(f"tensor\t{name}\t0\tf32\t{len(xs)}\t1\t1\t1\t{len(data)}\t{total:.6f}\t{op}\t1\t0\t-\t-")

    def finish(self) -> None:
        cols = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
        lines = self.header + [cols] + self.rows + [f"# complete\t{len(self.rows)}\t0"]
        partial = self.out / "MANIFEST.tsv.partial"
        partial.write_text("\n".join(lines) + "\n")
        partial.rename(self.out / "MANIFEST.tsv")


def pooled_rows(w: SetWriter, url: str, name: str, entry, mrl: bool) -> int:
    norm, n = embed(url, entry, 2)
    raw, n_raw = embed(url, entry, -1)
    if n != n_raw:
        raise SystemExit(f"server_vectors: {name}: prompt_tokens {n} and {n_raw} for one input")
    w.write(f"{name}/embd", norm, "SERVER")
    w.write(f"{name}/raw", raw, "SERVER")
    if mrl:
        for k in MRL:
            cut = raw[:k]
            s = math.sqrt(sum(x * x for x in cut))
            w.write(f"{name}/mrl{k}", [x / s for x in cut], "DERIVED")
    return n


def check_server(a: argparse.Namespace, build: str) -> dict:
    """The caller's pid is alive and the server at the url runs the given model, built at the commit `build` names."""
    try:
        os.kill(a.pid, 0)
    except OSError as e:
        raise SystemExit(f"server_vectors: the server's pid {a.pid} is not running ({e}): another server answers") from e
    props = json.loads(urllib.request.urlopen(a.url + "/props", timeout=60).read())
    if props.get("model_path") != a.model:
        raise SystemExit(f"server_vectors: the server runs {props.get('model_path')!r}, not {a.model}")
    info = props.get("build_info", "")
    commit = info.rsplit("-", 1)[-1]
    n = min(len(commit), len(build))
    if n < 7 or commit[:n] != build[:n]:
        raise SystemExit(f"server_vectors: the server's build_info is {info!r}, not the commit {build}")
    return props


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pid", required=True, type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--mmproj")
    ap.add_argument("--mmproj-sha256")
    ap.add_argument("--arch", choices=("gemma-embedding", "gemma-embedding2"), required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--build-record", required=True, type=Path)
    ap.add_argument("--ids-dir", required=True, type=Path)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--mix-image")
    ap.add_argument("--server-flags", default="")
    a = ap.parse_args()

    build = os.environ.get("BLOOMERY_REF_BUILD", "")
    if not build:
        raise SystemExit("server_vectors: BLOOMERY_REF_BUILD names no commit")
    if bool(a.mmproj) != bool(a.mmproj_sha256) or (a.image and not a.mmproj):
        raise SystemExit("server_vectors: images need a projector path and sha256")
    files = [(a.model, a.model_sha256)]
    if a.mmproj:
        files.append((a.mmproj, a.mmproj_sha256))
    for path, want in files:
        got = sha256(Path(path))
        if got != want:
            raise SystemExit(f"server_vectors: {path} has sha256 {got}, want {want}")
    props = check_server(a, build)
    w = SetWriter(a.out)
    w.header += [
        "# server_vectors: llama.cpp mainline llama-server /v1/embeddings, raw f32, little-endian",
        f"# model\t{a.model}",
        f"# build\t{build}",
        f"# arch\t{a.arch}",
        "# emb2\tserver",
        f"# device\tcuda\tthreads\t-\tcard\t{a.card}",
        f"# model_sha256\t{a.model_sha256}",
        f"# server_flags\t{a.server_flags}",
        f"# server_build\t{props['build_info']}",
    ]
    if a.mmproj:
        w.header.append(f"# mmproj\t{a.mmproj}\tsha256\t{a.mmproj_sha256}")
    w.header += [f"# build_record\t{line}" for line in a.build_record.read_text().splitlines() if line]
    for name in a.case:
        data = (a.ids_dir / f"{name}.txt").read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            w.header.append(f"# skipped\t{name}\tnot UTF-8, so not a JSON string")
            continue
        want = [int(x) for x in (a.ids_dir / f"{name}.ids").read_text().split()]
        got = post(a.url, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
        n = pooled_rows(w, a.url, name, text, True)
        same = "equal" if got == want else f"differs server {len(got)} ids, oracle {len(want)}"
        w.header.append(f"# tokenize\t{name}\t{same}\tprompt_tokens {n}")
        if got != want or n != len(want):
            raise SystemExit(f"server_vectors: {name}: the server's ids are not the oracle's ({same}, prompt_tokens {n})")

    images = dict(i.split("=", 1) for i in a.image)
    for name, png in images.items():
        data = Path(png).read_bytes()
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        part = {"type": "image_url", "image_url": {"url": url}}
        n = pooled_rows(w, a.url, f"img_{name}", {"content": [part]}, False)
        w.header.append(f"# image\timg_{name}\t{png}\t{hashlib.sha256(data).hexdigest()}\tprompt_tokens {n}")
        if name == a.mix_image:
            text = {"type": "text", "text": "A picture of coloured rings and ramps: "}
            n = pooled_rows(w, a.url, f"mix_{name}", {"content": [text, part]}, False)
            w.header.append(f"# image\tmix_{name}\t{png}\t{hashlib.sha256(data).hexdigest()}\tprompt_tokens {n}\ttext first")
    check_server(a, build)
    w.finish()
    print(f"server_vectors: {len(w.rows)} row(s) into {a.out}")


if __name__ == "__main__":
    main()
