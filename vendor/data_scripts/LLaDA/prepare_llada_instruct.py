#!/usr/bin/env python3
"""Download LLaDA-8B-Instruct weights for downstream task evaluation.

Output: {data_root}/llada-instruct/ (HuggingFace snapshot)

Usage:
    python vendor/data_scripts/LLaDA/prepare_llada_instruct.py --data-root vendor/data
"""

import argparse
import sys
from pathlib import Path

REPO_ID = "GSAI-ML/LLaDA-8B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / "llada-instruct"
    dest.mkdir(parents=True, exist_ok=True)

    # Check for required model files (not just any file)
    required = ["config.json", "tokenizer_config.json"]
    has_weights = any(
        (dest / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]
    ) or any(dest.glob("model-*.safetensors"))
    if all((dest / f).exists() for f in required) and has_weights:
        print(f"llada-instruct: already downloaded at {dest}, skipping")
        return

    print(f"Downloading {REPO_ID} to {dest}...", flush=True)
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(dest))
        print("llada-instruct: done")
    except Exception as e:
        print(f"llada-instruct: FAILED - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
