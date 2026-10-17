#!/usr/bin/env python3
"""Dump official sentence-transformers vectors for the E3 UTF-8 text cases.

Use an isolated venv with the versions in crates/refset/src/emb2/hf-build.txt.
The checkpoint revision lives beside its refset family, not at a moving HF main.
"""

import argparse
import importlib.metadata
import inspect
import shutil
from pathlib import Path

from server_vectors import SetWriter, metrics, sha256

ROOT = Path(__file__).resolve().parents[3]
BUILD = (ROOT / "crates/refset/src/emb2/hf-build.txt").read_text()
CHECKPOINT = (ROOT / "crates/refset/src/arch/gemma_embedding2/hf-checkpoint.txt").read_text()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    for pin in BUILD.split():
        package, version = pin.split("=")
        if importlib.metadata.version(package) != version:
            ap.error(f"requires {package}=={version}")
    if a.out.exists() and any(a.out.iterdir()):
        ap.error(f"output must be empty: {a.out}")

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer
    from transformers import masking_utils
    from transformers.models.embedding_gemma2 import modeling_embedding_gemma2

    repo, revision = CHECKPOINT.split("@")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    checkpoint = Path(snapshot_download(repo, revision=revision, local_dir=a.checkpoint_dir))
    weight = checkpoint / "model.safetensors"
    expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == weight.name)
    digest = sha256(weight)
    if digest != expected:
        raise ValueError(f"{weight}: sha256 {digest}, HF revision requires {expected}")
    model = SentenceTransformer(
        str(checkpoint), device="cuda",
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.eval()
    config = model[0].auto_model.config.text_config
    window = config.sliding_window
    mask = masking_utils.sliding_window_bidirectional_mask_function(window)
    # Exercise the actual HF mask at both edges, including the center position.
    for offset in (-window - 1, -window, -window + 1, 0, window - 1, window, window + 1):
        query = torch.tensor(window + 1)
        got = bool(mask(0, 0, query, query + offset))
        if got != (abs(offset) <= window):
            raise ValueError(f"HF mask: offset {offset}, allowed {got}, window {window}")
        print(f"hf-window offset {offset} allowed {got}", flush=True)
    w = SetWriter(a.out)
    w.header = [
        f"# model\t{CHECKPOINT}", "# arch\tgemma-embedding2", f"# build\t{BUILD}",
        "# emb2\thf", f"# device\tcuda\tthreads\t{torch.get_num_threads()}\tcard\t{torch.cuda.get_device_name()}",
        f"# model_sha256\t{digest}", f"# sliding_window\t{window}",
        f"# torch\t{torch.__version__}", "# dtype\tfloat32", "# attention\teager",
        "# prompt\tempty", "# normalization\tsentence-transformers",
    ]
    for module in (masking_utils, modeling_embedding_gemma2):
        path = Path(inspect.getsourcefile(module))
        w.header.append(f"# source\t{module.__name__}\t{sha256(path)}")
        for name in ("sliding_window_bidirectional_overlay", "create_bidirectional_sliding_window_mask", "EmbeddingGemma2TextModel"):
            if hasattr(module, name):
                function = getattr(module, name)
                line = inspect.getsourcelines(function)[1]
                print(f"hf-source {inspect.getsourcefile(function)}:{line} {name}", flush=True)
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts:
            w.header.append(f"# checkpoint_file\t{path.relative_to(checkpoint)}\t{sha256(path)}")
    manifest = a.ids_dir / "MANIFEST.tsv"
    lines = manifest.read_text().splitlines()
    names = [line.split("\t")[0] for line in lines if line.startswith("cases/")]
    if [line.split("\t")[:2] for line in lines if line.split("\t")[0] == "# complete"] != [["# complete", str(len(names))]]:
        raise ValueError(f"{manifest}: unfinished set or incorrect case count")
    if not names or len(names) != len(set(names)) or any(not n.removeprefix("cases/").isdigit() for n in names):
        raise ValueError("the tokenizer set must contain unique numeric E3 cases")
    count = 0
    for name in names:
        text_path = a.ids_dir / f"{name}.txt"
        try:
            text = text_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            w.header.append(f"# skipped\t{name}\tnot UTF-8")
            continue
        source_ids = a.ids_dir / f"{name}.ids"
        ids = [int(x) for x in source_ids.read_text().split()]
        actual = model.tokenize([text])["input_ids"][0].tolist()
        encoded = model.encode([text], prompt="", normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        features = {
            "input_ids": torch.tensor([ids], device="cuda"),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
        }
        with torch.inference_mode():
            vector = model(features)["sentence_embedding"][0].float().cpu().tolist()
        if actual == ids:
            _, maximum = metrics(encoded, vector)
            if maximum > 1e-7:
                raise ValueError(f"{name}: encode and forward differ by {maximum}")
        else:
            hf_name = f"{name}_hf"
            hf_ids = a.out / f"{hf_name}.ids"
            hf_ids.parent.mkdir(parents=True, exist_ok=True)
            hf_ids.write_text("\n".join(map(str, actual)) + "\n")
            w.header.append(f"# case\t{hf_name}\t{len(actual)}\t{hf_name}.ids\t{sha256(hf_ids)}")
            w.header.append(f"# tokenize\t{name}\tE3 {len(ids)}\tHF {len(actual)}\tseparate encode row {hf_name}")
            w.write(f"{hf_name}/embd", encoded, "SENTENCE_TRANSFORMERS_ENCODE")
            print(f"hf-tokenize {name}: E3 {len(ids)}, HF {len(actual)}; official encode stored as {hf_name}", flush=True)
        target = a.out / f"{name}.ids"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_ids, target)
        w.header.append(f"# case\t{name}\t{len(ids)}\t{name}.ids\t{sha256(target)}")
        w.header.append(f"# text\t{name}\t{sha256(text_path)}")
        w.write(f"{name}/embd", vector, "SENTENCE_TRANSFORMERS_FORWARD")
        count += 1
        print(f"hf-vector {name} ids {len(ids)} dimensions {len(vector)}", flush=True)
    if not count:
        raise ValueError("no UTF-8 cases were encoded")
    w.finish()
    paths = sorted(p for p in a.out.rglob("*") if p.is_file())
    (a.out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(a.out)}\n" for p in paths))
    print(f"hf-vectors: {count} cases; window one-sided inclusive {window}; set-sha256 {sha256(a.out / 'MANIFEST.tsv')}")


if __name__ == "__main__":
    main()
