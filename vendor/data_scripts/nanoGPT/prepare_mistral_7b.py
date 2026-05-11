#!/usr/bin/env python3
"""Download Mistral-7B-v0.1 weights and WikiText-2 raw text for PTQ tasks.

Downloads:
  1. ``mistralai/Mistral-7B-v0.1`` model (~14GB) for quantization
  2. WikiText-2 raw text for calibration and evaluation

Output:
  {data_root}/mistral-7b-v01/     (HuggingFace model snapshot)
  {data_root}/mistral-7b-v01/wikitext2_raw.txt  (raw text for calibration/eval)

Usage:
    python vendor/data_scripts/nanoGPT/prepare_mistral_7b.py --data-root vendor/data
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO_ID = "mistralai/Mistral-7B-v0.1"


def download_wikitext2_raw(dest_dir: Path) -> None:
    """Download WikiText-2 raw text for calibration and evaluation."""
    out_file = dest_dir / "wikitext2_raw.txt"
    if out_file.exists() and out_file.stat().st_size > 0:
        print("WikiText-2 raw text: already exists, skipping")
        return

    print("Downloading WikiText-2 raw text...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])
        if text.strip():
            out_file.write_text(text, encoding="utf-8")
            print(f"  WikiText-2 raw: {len(text):,} chars")
            return
    except Exception as e:
        print(f"  datasets fallback: {e}")

    # Fallback: download from S3
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read("wikitext-2-raw/wiki.test.raw").decode("utf-8")
    out_file.write_text(text, encoding="utf-8")
    print(f"  WikiText-2 raw: {len(text):,} chars")

    # Also save train split for calibration
    train_file = dest_dir / "wikitext2_train.txt"
    if not train_file.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            train_text = "\n".join(ds["text"])
            if train_text.strip():
                train_file.write_text(train_text, encoding="utf-8")
                print(f"  WikiText-2 train: {len(train_text):,} chars")
        except Exception:
            try:
                train_text = z.read("wikitext-2-raw/wiki.train.raw").decode("utf-8")
                train_file.write_text(train_text, encoding="utf-8")
                print(f"  WikiText-2 train: {len(train_text):,} chars")
            except Exception:
                print("  WARNING: Could not download WikiText-2 train split")


def main():
    parser = argparse.ArgumentParser(
        description="Download Mistral-7B-v0.1 weights and calibration data"
    )
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "mistral-7b-v01"
    dest.mkdir(parents=True, exist_ok=True)

    # Check if model weights already exist
    model_files = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
    if model_files:
        print("mistral-7b-v01: model weights already downloaded, skipping")
    else:
        print(f"Downloading {REPO_ID}...", flush=True)
        try:
            snapshot_download(repo_id=REPO_ID, local_dir=str(dest))
            print("mistral-7b-v01: done")
        except Exception as e:
            print(f"mistral-7b-v01: FAILED - {e}", file=sys.stderr)
            sys.exit(1)

    # Download WikiText-2 raw text
    download_wikitext2_raw(dest)

    print(f"\nMistral-7B-v0.1 weights and calibration data ready at {dest}/")


if __name__ == "__main__":
    main()
