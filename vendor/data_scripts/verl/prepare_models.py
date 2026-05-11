#!/usr/bin/env python3
"""Download Qwen3-1.7B for verl RL fine-tuning tasks.

Output: {data_root}/models/Qwen3-1.7B/

Usage:
    python vendor/data_scripts/verl/prepare_models.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data verl
"""

import argparse
import os
import sys
from pathlib import Path

MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]


def main():
    parser = argparse.ArgumentParser(description="Download Qwen models for verl")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    model_dir = Path(args.data_root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in MODELS:
        name = repo_id.split("/")[-1]
        dest = model_dir / name

        if dest.exists() and any(dest.iterdir()):
            print(f"{name}: already downloaded, skipping")
            continue

        print(f"Downloading {repo_id}...", flush=True)
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(dest))
            print(f"{name}: done")
        except Exception as e:
            print(f"{name}: FAILED - {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
