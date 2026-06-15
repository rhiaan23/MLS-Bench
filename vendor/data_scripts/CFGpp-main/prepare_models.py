#!/usr/bin/env python3
"""Prepare Stable Diffusion model weights for CFGpp-main tasks.

Downloads model snapshots from HuggingFace Hub into the host cache directory
so they can be bind-mounted into containers at runtime.

Output: {data_root}/huggingface_cache/hub/models--*

Usage:
    python vendor/data_scripts/CFGpp-main/prepare_models.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data CFGpp-main
"""

import argparse
import os
import sys
from pathlib import Path


# Models to download: (repo_id, repo_type)
MODELS = [
    ("runwayml/stable-diffusion-v1-5", "model"),
    ("Manojb/stable-diffusion-2-base", "model"),
    ("madebyollin/sdxl-vae-fp16-fix", "model"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "model"),
]

# diffusers only loads the PyTorch/safetensors weights; the SDXL repo also
# ships ONNX / OpenVINO / Flax copies that bloat the baked image by tens of
# GB and were the files that intermittently failed mid-snapshot. Skip them.
IGNORE_PATTERNS = ["*.onnx", "*.onnx_data", "*openvino*", "*.msgpack"]


def _snapshot_ok(model_cache: Path) -> bool:
    """A snapshot is ready only if it has no leftover .incomplete blobs AND
    every file the latest snapshot references actually resolves. The old
    "snapshots dir non-empty + no .incomplete" check passed for an
    interrupted download whose weight symlinks dangle into never-finished
    blobs — exactly the SDXL failure that shipped an image with only
    unet/config.json and no weights."""
    snapshots = model_cache / "snapshots"
    if not (snapshots.exists() and any(snapshots.iterdir())):
        return False
    blobs = model_cache / "blobs"
    if blobs.exists() and list(blobs.glob("*.incomplete")):
        return False
    snap = sorted(p for p in snapshots.iterdir() if p.is_dir())[-1]
    for f in snap.rglob("*"):
        if f.is_symlink() and not f.resolve().exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SD model weights")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "huggingface_cache"
    os.environ["HF_HOME"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use mirror if HF_ENDPOINT is set, or default to hf-mirror for accessibility
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    for i, (repo_id, repo_type) in enumerate(MODELS, 1):
        name = repo_id.split("/")[-1]
        # Check if already downloaded AND complete (resolvable weights).
        model_cache = cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}"
        if _snapshot_ok(model_cache):
            print(f"[{i}/{len(MODELS)}] {name}: already downloaded, skipping")
            continue
        if (model_cache / "snapshots").exists():
            print(f"[{i}/{len(MODELS)}] {name}: incomplete/partial cache, re-fetching")

        print(f"[{i}/{len(MODELS)}] Downloading {repo_id}...", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                cache_dir=str(cache_dir / "hub"),
                ignore_patterns=IGNORE_PATTERNS,
            )
            print(f"[{i}/{len(MODELS)}] {name}: done")
        except Exception as e:
            print(f"[{i}/{len(MODELS)}] {name}: FAILED - {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
