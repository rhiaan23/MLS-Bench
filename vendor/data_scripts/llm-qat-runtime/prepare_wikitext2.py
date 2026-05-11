#!/usr/bin/env python3
"""Download and cache WikiText-2 dataset for QAT finetune + evaluation.

Pre-downloads ``wikitext/wikitext-2-raw-v1`` train+test splits so
compute nodes without network can use them for QAT training and
perplexity evaluation.

Output: {data_root}/wikitext2/  (HF datasets cache)

Usage:
    python vendor/data_scripts/llm-qat-runtime/prepare_wikitext2.py --data-root vendor/data
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download WikiText-2 dataset")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    dest = Path(args.data_root) / "wikitext2"
    dest.mkdir(parents=True, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = str(dest)

    from datasets import load_dataset

    marker = dest / ".done"
    if marker.exists():
        print("wikitext2: already downloaded, skipping")
        return

    print("Downloading WikiText-2 (raw v1)...", flush=True)
    try:
        ds_train = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=str(dest)
        )
        ds_test = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=str(dest)
        )
        print(f"  train: {len(ds_train)} examples")
        print(f"  test:  {len(ds_test)} examples")
        marker.touch()
        print("wikitext2: done")
    except Exception as e:
        print(f"wikitext2: FAILED - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWikiText-2 data ready at {dest}/")


if __name__ == "__main__":
    main()
