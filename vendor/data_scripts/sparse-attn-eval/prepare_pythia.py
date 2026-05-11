#!/usr/bin/env python3
"""Download Qwen/Qwen2.5-1.5B-Instruct weights for long-context inference.

Output: {data_root}/qwen2.5-1.5b-instruct/  (HuggingFace snapshot, ~3 GB)

(Filename is left as ``prepare_pythia.py`` for backward compat with the
package config; this previously fetched Pythia-1.4B but instruction-following
matters for NIAH / LongBench Qasper, so we now use Qwen2.5-1.5B-Instruct.)

Idempotent: skips if a complete snapshot already exists (config.json plus
weight files totalling >= MIN_BYTES_DONE) or if a ``.done`` marker is present.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_NAME = "qwen2.5-1.5b-instruct"
# Qwen2.5-1.5B-Instruct safetensors is ~3.1 GB; require ≥2.5 GB to consider done.
MIN_BYTES_DONE = int(2.5 * 1024 * 1024 * 1024)


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

    print(f"Downloading {REPO_ID} -> {dest} (Qwen2.5-1.5B-Instruct, ~3 GB)...", flush=True)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(dest),
            allow_patterns=[
                "*.json", "*.txt", "tokenizer*", "*.model", "merges.txt",
                "pytorch_model.bin", "pytorch_model-*.bin",
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
