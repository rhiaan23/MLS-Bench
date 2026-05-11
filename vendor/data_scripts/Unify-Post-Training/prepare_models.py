#!/usr/bin/env python3
"""Download the small Qwen2.5-Math model used by the MLS-Bench HPT task."""

import argparse
import os
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Math-1.5B")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "models" / "Qwen2.5-Math-1.5B"
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        print("Qwen2.5-Math-1.5B: already downloaded, skipping")
        return

    print(f"Downloading {MODEL_ID}...", flush=True)
    try:
        snapshot_download(repo_id=MODEL_ID, local_dir=str(dest))
    except Exception as exc:
        print(f"Qwen2.5-Math-1.5B download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Qwen2.5-Math-1.5B: done")


if __name__ == "__main__":
    main()
