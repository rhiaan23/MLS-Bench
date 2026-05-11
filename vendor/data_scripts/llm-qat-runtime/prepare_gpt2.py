#!/usr/bin/env python3
"""Download GPT-2 124M HuggingFace weights for QAT tasks.

Downloads the ``gpt2`` (124M parameter) model snapshot so that compute
nodes (which have no network access) can load pretrained weights directly.

Output: {data_root}/gpt2/  (HuggingFace snapshot)

Usage:
    python vendor/data_scripts/llm-qat-runtime/prepare_gpt2.py --data-root vendor/data
"""

import argparse
import os
import sys
from pathlib import Path


REPO_ID = "gpt2"


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 124M weights")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "gpt2"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        print("gpt2: already downloaded, skipping")
        return

    print(f"Downloading {REPO_ID}...", flush=True)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(dest),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "merges.txt",
                "vocab.json",
                "tokenizer*.json",
                "config.json",
                "generation_config.json",
                "pytorch_model.bin",
                "model.safetensors",
            ],
        )
        marker.touch()
        print("gpt2: done")
    except Exception as e:
        print(f"gpt2: FAILED - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nGPT-2 124M weights ready at {dest}/")


if __name__ == "__main__":
    main()
