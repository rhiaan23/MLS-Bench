#!/usr/bin/env python3
"""Download GPT-2 Medium weights from HuggingFace for TTT adaptation tasks.

Downloads the ``gpt2-medium`` model (355M params) so that compute nodes
(which have no network access) can load pretrained weights directly.

Output: {data_root}/gpt2-medium/  (HuggingFace snapshot)

Usage:
    python vendor/data_scripts/nanoGPT/prepare_gpt2_medium.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data nanoGPT
"""

import argparse
import os
import sys
from pathlib import Path


REPO_ID = "openai-community/gpt2-medium"


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 Medium weights")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "gpt2-medium"
    dest.mkdir(parents=True, exist_ok=True)

    if dest.exists() and any(dest.iterdir()):
        print("gpt2-medium: already downloaded, skipping")
        return

    print(f"Downloading {REPO_ID}...", flush=True)
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(dest))
        print("gpt2-medium: done")
    except Exception as e:
        print(f"gpt2-medium: FAILED - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nGPT-2 Medium weights ready at {dest}/")


if __name__ == "__main__":
    main()
