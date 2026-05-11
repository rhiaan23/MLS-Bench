"""Prepare WikiText-103 for ttt-memory (Titans MAC paper-aligned eval).

Downloads the raw WikiText-103 corpus from HuggingFace (Salesforce/wikitext)
and tokenizes it with tiktoken GPT-2 BPE. Produces three flat uint16
binaries: train.bin, valid.bin, test.bin under /data/titans-lmm/wikitext103/.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np


# HuggingFace parquet mirrors for wikitext-103-raw-v1.
HF_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
SPLIT_FILES = {
    "train": [
        f"{HF_BASE}/train-00000-of-00002.parquet",
        f"{HF_BASE}/train-00001-of-00002.parquet",
    ],
    "valid": [f"{HF_BASE}/validation-00000-of-00001.parquet"],
    "test":  [f"{HF_BASE}/test-00000-of-00001.parquet"],
}


def _download(url: str, dest: Path) -> bool:
    print(f"    GET {url}", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mlsbench-prepare"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            dest.write_bytes(resp.read())
        return dest.stat().st_size > 1024
    except Exception as e:  # noqa: BLE001
        print(f"      failed: {e}", flush=True)
        if dest.exists():
            dest.unlink()
        return False


def _read_parquet_text(path: Path) -> list[str]:
    """Read the 'text' column from a wikitext parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import pandas as pd
        return pd.read_parquet(path)["text"].tolist()
    table = pq.read_table(path, columns=["text"])
    return table.column("text").to_pylist()


def _tokenize_split(urls: list[str], out_bin: Path, tmp_dir: Path) -> bool:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    if out_bin.exists() and out_bin.stat().st_size > 1024:
        print(f"  [skip] {out_bin.name} ({out_bin.stat().st_size} bytes)")
        return True

    all_ids: list[int] = []
    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        tmp = tmp_dir / fname
        if not tmp.exists() or tmp.stat().st_size < 1024:
            if not _download(url, tmp):
                return False
        rows = _read_parquet_text(tmp)
        print(f"    tokenizing {fname} ({len(rows)} rows)...", flush=True)
        for text in rows:
            if not text:
                continue
            all_ids.extend(enc.encode_ordinary(text))
        # streaming is possible but the corpus fits easily in RAM
    arr = np.asarray(all_ids, dtype=np.uint32)
    if arr.max() >= 2 ** 16:
        raise ValueError(f"Token id {arr.max()} doesn't fit in uint16")
    arr.astype(np.uint16).tofile(out_bin)
    print(f"  [ok]   {out_bin.name} ({out_bin.stat().st_size} bytes, "
          f"{len(all_ids)} tokens)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "titans-lmm" / "wikitext103"
    tmp_dir = out_dir / "_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing WikiText-103 (GPT-2 BPE) in {out_dir}")
    failures = []
    for split, urls in SPLIT_FILES.items():
        out = out_dir / f"{split}.bin"
        if not _tokenize_split(urls, out, tmp_dir):
            failures.append(split)

    if failures:
        print(f"\nFAILED: {failures}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
