#!/usr/bin/env python3
"""Prepare evaluation datasets for nanoGPT tasks.

Downloads WikiText-2, WikiText-103, and LAMBADA test sets, tokenizes them with
GPT-2 BPE, and produces headerless uint16 .bin files.

Output: {data_root}/eval/{wikitext2,wikitext103,lambada}.bin (~2 MB total)

Usage:
    python vendor/data_scripts/nanoGPT/prepare_eval.py --data-root vendor/data
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tiktoken

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional fallback
    load_dataset = None


def _download_url_text(url: str) -> str:
    return urllib.request.urlopen(url).read().decode("utf-8")


def _download_zip_member(url: str, member: str) -> str:
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(member).decode("utf-8")


def _load_wikitext(name: str) -> str:
    if load_dataset is not None:
        dataset = load_dataset("wikitext", name, split="test")
        text = "\n".join(dataset["text"])
        if text.strip():
            return text

    legacy_sources = {
        "wikitext-2-raw-v1": (
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
            "wikitext-2-raw/wiki.test.raw",
        ),
        "wikitext-103-raw-v1": (
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
            "wikitext-103-raw/wiki.test.raw",
        ),
    }
    url, member = legacy_sources[name]
    return _download_zip_member(url, member)


def main():
    parser = argparse.ArgumentParser(description="Prepare eval data for nanoGPT")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    # --- WikiText-2 ---
    wt2_path = out_dir / "wikitext2.bin"
    if not wt2_path.exists():
        print("Downloading WikiText-2...")
        text = _load_wikitext("wikitext-2-raw-v1")
        tokens = enc.encode(text)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(str(wt2_path))
        print(f"  WikiText-2 test: {len(tokens):,} tokens")
    else:
        print("WikiText-2: already exists")

    # --- WikiText-103 ---
    wt103_path = out_dir / "wikitext103.bin"
    if not wt103_path.exists():
        print("Downloading WikiText-103...")
        text = _load_wikitext("wikitext-103-raw-v1")
        tokens = enc.encode(text)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(str(wt103_path))
        print(f"  WikiText-103 test: {len(tokens):,} tokens")
    else:
        print("WikiText-103: already exists")

    # --- LAMBADA ---
    lambada_path = out_dir / "lambada.bin"
    if not lambada_path.exists():
        print("Downloading LAMBADA...")
        url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
        data = _download_url_text(url)
        tokens = []
        for line in data.strip().split("\n"):
            obj = json.loads(line)
            tokens.extend(enc.encode(" " + obj["text"]))
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(str(lambada_path))
        print(f"  LAMBADA test: {len(tokens):,} tokens")
    else:
        print("LAMBADA: already exists")

    print(f"\nEval data ready at {out_dir}/")


if __name__ == "__main__":
    main()
