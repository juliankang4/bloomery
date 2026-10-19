"""The GLM-5.3-Flash Hugging Face map oracle (crates/refset/src/hfmap.rs), by ranged reads of the hub
(tools/bloomery/hf.py) at the revisions below: nvidia/GLM-5.3-Flash-NVFP4's shard headers
(<shard>.header.json.gz) and its small F32 and BF16 tensors of layers 0 to 44 (A_log, dt_bias, conv
taps, hc base and scale, e_score_correction_bias) into hf.safetensors; unsloth's UD-Q4_K_XL tensor
table (tensors.tsv: name and ggml dims) and the glm5next counterparts of those tensors into
gguf.safetensors (F32, 1-D). MANIFEST.tsv is written last, `# complete` its last line.

    glm5next-hf-map.py OUT    OUT must not exist

Exit 65 on an answer this does not read (a header, a split set, a tensor type), 69 when the network
or the hub is down, 64 on a usage error."""
import gzip, hashlib, importlib.util, json, os, re, struct, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("bloomery_hf", os.path.join(ROOT, "tools", "bloomery", "hf.py"))
hf = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(hf)

CHECKPOINT = ("nvidia/GLM-5.3-Flash-NVFP4", "da920bb0b9f4a06727223a349e55468e38352348")
GGUF = ("unsloth/GLM-5.3-Flash-GGUF", "a38483c8cd5df544f53d70fb281afe97369d5ab6", "UD-Q4_K_XL")
LAYERS = 45
HF_WANT = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(A_log|dt_bias|[qkv]_conv1d\.weight)"
                     r"|hc_(attn|ffn)_(base|scale)|mlp\.gate\.e_score_correction_bias)$")
GGUF_WANT = re.compile(r"^blk\.(\d+)\.(ssm_a|ssm_dt\.bias|ssm_conv1d_[qkv]\.weight|hc_(attn|ffn)_(base|scale)\.weight"
                       r"|exp_probs_b\.bias)$")


def wanted(pattern, name):
    m = pattern.match(name)
    return m is not None and int(m.group(1)) < LAYERS


def safetensors(path, tensors):
    head, data = {}, b""
    for name, (dtype, shape, b) in tensors.items():
        head[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    h = json.dumps(head).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + data)


def write(out, hub, scratch):
    written = []

    def put(name, data):
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
        written.append(name)

    repo, rev = CHECKPOINT
    sizes = {e.path: e.size for e in hub.listing(repo, rev)}
    read = lambda path, a, n: hub.range(repo, path, a, a + n, rev)
    index = json.loads(read("model.safetensors.index.json", 0, sizes["model.safetensors.index.json"]))["weight_map"]
    small = {}
    for shard in sorted(set(index.values())):
        n = struct.unpack("<Q", read(shard, 0, 8))[0]
        raw = read(shard, 8, n)
        put(shard + ".header.json.gz", gzip.compress(raw, mtime=0))
        for name, e in sorted(json.loads(raw).items()):
            if name != "__metadata__" and wanted(HF_WANT, name):
                a, b = e["data_offsets"]
                small[name] = (e["dtype"], e["shape"], read(shard, 8 + n + a, b - a))
    safetensors(os.path.join(out, "hf.safetensors"), small)
    written.append("hf.safetensors")

    repo, rev, quant = GGUF
    sets = [s for s in hf.gguf_sets(repo, hub.listing(repo, rev))
            if all(f.path.startswith(quant + "/") for f in s["files"])]
    if len(sets) != 1:
        raise hf.Red(f"{repo}@{rev}: {len(sets)} GGUF sets under {quant}/, not one")
    rows, values = ["tensor\tdimensions"], {}
    local = os.path.join(scratch, "header.bin")
    for f in sets[0]["files"]:
        fetch = lambda a, b, p=f.path: hub.range(repo, p, a, b, rev)
        meta, tensors, got = hf.read_header(fetch, local, f.size, 64 * hf.MIB)
        if meta is None:
            raise hf.Red(f"{f.path}: the header is unfinished at {got} bytes")
        start = hf.gguf_ranges().header(local, size=f.size)[2]
        for name, ty, dims, off in tensors:
            rows.append(f"{name}\t{'x'.join(map(str, dims))}")
            if not wanted(GGUF_WANT, name):
                continue
            if ty != 0:
                raise hf.Red(f"{name}: ggml type {hf.ggml_type_name(ty, f.path)}, not F32")
            count = 1
            for d in dims:
                count *= d
            values[name] = ("F32", [count], fetch(start + off, start + off + 4 * count))
    put("tensors.tsv", ("\n".join(rows) + "\n").encode())
    safetensors(os.path.join(out, "gguf.safetensors"), values)
    written.append("gguf.safetensors")
    return written, len(small), len(values), len(rows) - 1


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 64
    out = argv[0]
    try:
        os.makedirs(out)
    except OSError as e:
        print(f"glm5next-hf-map: {out}: {e.strerror}; move the old set away first", file=sys.stderr)
        return 64
    try:
        with tempfile.TemporaryDirectory() as scratch:
            written, n_hf, n_gguf, n_rows = write(out, hf.Hub(scratch), scratch)
    except hf.Refuse as e:
        print(f"glm5next-hf-map: {e}", file=sys.stderr)
        return e.code
    except (hf.Red, KeyError, ValueError) as e:
        print(f"glm5next-hf-map: {e!r}", file=sys.stderr)
        return 65
    lines = [
        f"# harness\t{md5(os.path.abspath(__file__))}",
        f"# checkpoint\t{CHECKPOINT[0]}@{CHECKPOINT[1]}",
        f"# model\t{GGUF[0]}@{GGUF[1]}/{GGUF[2]}",
        "# arch\tglm5next",
        "# file\tname\tbytes\tmd5",
    ] + [f"file\t{n}\t{os.path.getsize(os.path.join(out, n))}\t{md5(os.path.join(out, n))}" for n in written]
    lines.append(f"# complete\t{len(written)}")
    with open(os.path.join(out, "MANIFEST.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{out}: {len(written)} files, hf tensors {n_hf}, gguf tensors {n_gguf}, table rows {n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
