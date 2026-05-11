#!/usr/bin/env python3
"""Download mistralai/Mistral-7B-Instruct-v0.2 weights.

Output: {data_root}/mistral-7b-instruct-v0.2/  (HuggingFace snapshot, ~14 GB)

This is the primary backbone used by the SnapKV paper (Li et al., NeurIPS 2024)
and is used here for direct numerical comparability with the published H2O /
SnapKV / StreamingLLM long-context numbers. Instruction-tuned, 32K native
context (sliding-window attention disabled in this checkpoint).

Idempotent: skips if a complete snapshot already exists (config.json plus
weight files totalling >= MIN_BYTES_DONE) or if a ``.done`` marker is present.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DIR_NAME = "mistral-7b-instruct-v0.2"
# Mistral-7B-Instruct-v0.2 safetensors total ~14.5 GB; require ≥12 GB to consider done.
MIN_BYTES_DONE = int(12 * 1024 * 1024 * 1024)


def _weight_bytes(path: Path) -> int:
    total = 0
    for cand in ("pytorch_model.bin", "model.safetensors"):
        f = path / cand
        if f.exists():
            total += f.stat().st_size
    for shard in path.glob("pytorch_model-*.bin"):
        total += shard.stat().st_size
    for shard in path.glob("model-*.safetensors"):
        total += shard.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    dest = Path(args.data_root) / DIR_NAME
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists() and (dest / "config.json").exists() and _weight_bytes(dest) >= MIN_BYTES_DONE:
        print(f"{DIR_NAME} ({REPO_ID}): already downloaded "
              f"({_weight_bytes(dest)/1e9:.2f} GB), skipping")
        return
    if (dest / "config.json").exists() and _weight_bytes(dest) >= MIN_BYTES_DONE:
        marker.touch()
        print(f"{DIR_NAME} ({REPO_ID}): existing snapshot detected "
              f"({_weight_bytes(dest)/1e9:.2f} GB), marking done")
        return

    print(f"Downloading {REPO_ID} -> {dest} (Mistral-7B-Instruct-v0.2, ~14 GB)...", flush=True)
    try:
        # Prefer safetensors over .bin to halve disk usage.
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(dest),
            allow_patterns=[
                "*.json", "*.txt", "tokenizer*", "*.model",
                "model.safetensors", "model-*.safetensors",
            ],
        )
        marker.touch()
        print(f"{DIR_NAME}: done ({dest}, {_weight_bytes(dest)/1e9:.2f} GB)")
    except Exception as e:
        print(f"{DIR_NAME}: FAILED - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
