#!/usr/bin/env python3
"""One-time setup for the local MiniLM ONNX model.

This is a standalone helper, NOT part of the authorship_attribution package.
The package itself never downloads anything at runtime; this script is the one
deliberate, manual place where files are fetched from the internet.

Usage (from the repo root, inside the project virtualenv):
    python setup_minilm_onnx.py
    python setup_minilm_onnx.py --dest models/all-MiniLM-L6-v2-onnx
    python setup_minilm_onnx.py --skip-download   # files already placed by hand
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
# local filename -> path inside the Hugging Face repo
REMOTE_FILES = {"model.onnx": "onnx/model.onnx", "vocab.txt": "vocab.txt"}
EMBEDDING_DIM = 384
SAMPLE_IDS = [101, 2023, 2003, 1037, 3231, 102]  # [CLS] this is a test [SEP]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    partial.replace(dest)


def check_vocab(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems = []
    if len(lines) != 30522:
        problems.append(f"expected 30522 lines, found {len(lines)}")
    if len(set(lines)) != len(lines):
        problems.append("duplicate tokens")
    for token, index in (("[PAD]", 0), ("[CLS]", 101), ("[SEP]", 102)):
        if len(lines) <= index or lines[index] != token:
            problems.append(f"{token} is not at line index {index}")
    if problems:
        sys.exit("vocab.txt check FAILED: " + "; ".join(problems))
    print("vocab.txt: OK (30522 unique tokens, special tokens in the expected places)")


def check_model(path: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime/numpy not installed here: skipping the model load test.")
        print("Install the project requirements and rerun to run this check.")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_inputs()]
    print(f"model inputs: {names}")
    if "input_ids" not in names or "attention_mask" not in names:
        sys.exit("model check FAILED: expected inputs named input_ids and attention_mask")
    ids = np.array([SAMPLE_IDS], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if "token_type_ids" in names:
        feeds["token_type_ids"] = np.zeros_like(ids)
    output = np.asarray(session.run(None, feeds)[0])
    print(f"first output shape for a {ids.shape[1]}-token input: {output.shape}")
    if output.ndim == 2:
        sys.exit(
            "model check FAILED: output is 2-D (already pooled). The engine needs the "
            "token-level export with output shape (batch, tokens, 384). Use the "
            "onnx/model.onnx from sentence-transformers/all-MiniLM-L6-v2."
        )
    if output.shape != (1, ids.shape[1], EMBEDDING_DIM) or not np.isfinite(output).all():
        sys.exit(f"model check FAILED: expected shape (1, {ids.shape[1]}, {EMBEDDING_DIM}) with finite values")
    print("model.onnx: OK (loads, correct inputs, token-level 384-dim output)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("models/all-MiniLM-L6-v2-onnx"))
    parser.add_argument("--skip-download", action="store_true", help="use files already in --dest")
    parser.add_argument("--force", action="store_true", help="overwrite an existing, different manifest.json")
    args = parser.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    for local, remote in REMOTE_FILES.items():
        target = dest / local
        if target.is_file():
            print(f"{local}: already present, not downloading")
        elif args.skip_download:
            sys.exit(f"{local} is missing from {dest} and --skip-download was given")
        else:
            download(BASE_URL + remote, target)

    check_vocab(dest / "vocab.txt")
    check_model(dest / "model.onnx")

    manifest = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": EMBEDDING_DIM,
        "maximum_token_length": 256,
        "license": "Apache-2.0",
        "onnx_model": "model.onnx",
        "model_files": {"model.onnx": sha256_file(dest / "model.onnx")},
        "tokenizer_vocab": "vocab.txt",
        "tokenizer_vocab_sha256": sha256_file(dest / "vocab.txt"),
        "do_lower_case": True,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        if manifest_path.read_text(encoding="utf-8") == text:
            print("manifest.json: already up to date (left untouched)")
        elif not args.force:
            sys.exit(
                "manifest.json already exists and differs from the new one. Its exact bytes feed the "
                "model fingerprint, so changing it invalidates existing baselines. Rerun with --force "
                "only if you really mean to."
            )
        else:
            manifest_path.write_text(text, encoding="utf-8")
            print("manifest.json: overwritten (rebuild your baselines)")
    else:
        manifest_path.write_text(text, encoding="utf-8")
        print("manifest.json: written")
    print(f"\nDone. Point the CLI at it with:  --model-dir {dest}")


if __name__ == "__main__":
    main()
