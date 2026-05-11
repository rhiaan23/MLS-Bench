#!/usr/bin/env python3
"""Download diffusion models and utility weights for SpargeAttn benchmarks.

Downloads:
  - CogVideoX-2b            (~10GB, video generation, bfloat16)
  - PixArt-XL-2-1024-MS     (~2.3GB, image generation, float16)
  - Wan2.1-T2V-1.3B         (~5GB, video generation, bfloat16)
  - AlexNet torchvision      (~233MB, for LPIPS quality eval)

All models are saved under {data_root}/SpargeAttn/models/.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print(f"  $ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, **kwargs)


def download_hf_model(repo_id: str, local_dir: Path):
    """Download a HuggingFace model using huggingface_hub."""
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [skip] {repo_id} already exists at {local_dir}")
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        print(f"  Downloading {repo_id} -> {local_dir} ...", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    except ImportError:
        # Fallback to git clone
        print(f"  huggingface_hub not available, using git clone for {repo_id}", flush=True)
        run(f"git lfs install && git clone https://huggingface.co/{repo_id} {local_dir}")


def download_torchvision_alexnet(cache_dir: Path):
    """Download AlexNet weights for LPIPS evaluation."""
    ckpt_dir = cache_dir / "hub" / "checkpoints"
    ckpt_file = ckpt_dir / "alexnet-owt-7be5be79.pth"
    if ckpt_file.exists():
        print(f"  [skip] AlexNet weights already at {ckpt_file}")
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    print(f"  Downloading AlexNet weights ...", flush=True)
    run(f"wget -q -O {ckpt_file} {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    models_dir = args.data_root / "SpargeAttn" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== SpargeAttn: Preparing models ===", flush=True)

    # 1. Diffusion models
    models = [
        ("THUDM/CogVideoX-2b", "CogVideoX-2b"),
        ("PixArt-alpha/PixArt-XL-2-1024-MS", "PixArt-XL-2-1024-MS"),
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B-Diffusers"),
    ]
    for repo_id, dirname in models:
        download_hf_model(repo_id, models_dir / dirname)

    # 2. Torchvision AlexNet (for LPIPS)
    torch_cache = models_dir / "torch_cache"
    download_torchvision_alexnet(torch_cache)

    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
