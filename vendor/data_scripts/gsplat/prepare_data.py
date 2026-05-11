#!/usr/bin/env python3
"""Prepare Mip-NeRF 360 dataset for 3DGS densification task.

Downloads the full 360_v2 dataset (~12.5GB) and extracts only the scenes
needed for evaluation (garden, bicycle).

Output layout:
  {data_root}/gsplat_data/360_v2/
  ├── garden/
  │   ├── images/
  │   ├── images_2/
  │   ├── images_4/
  │   ├── images_8/
  │   └── sparse/0/
  └── bicycle/
      ├── images/
      ├── images_2/
      ├── images_4/
      ├── images_8/
      └── sparse/0/

Usage:
    python vendor/data_scripts/gsplat/prepare_data.py --data-root vendor/data
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENES = ["garden", "bicycle", "bonsai", "stump"]
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ready] {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-q", "--show-progress", "-O", str(tmp), url])
    else:
        urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"[downloaded] {dest}")


def download_and_extract(data_root: str):
    out_dir = Path(data_root) / "gsplat_data" / "360_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    all_present = all((out_dir / s / "sparse" / "0").exists() for s in SCENES)
    if all_present:
        print(f"All scenes already present in {out_dir}")
        return

    zip_path = Path(data_root) / "gsplat_data" / "360_v2.zip"

    if not zip_path.exists():
        print(f"Downloading Mip-NeRF 360 dataset (~12.5GB)...")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["wget", "-q", "--show-progress", "-O", str(zip_path), DATASET_URL],
        )
    else:
        print(f"Found existing {zip_path}")

    # Extract only needed scenes
    print(f"Extracting scenes: {SCENES}")
    for scene in SCENES:
        scene_dir = out_dir / scene
        if scene_dir.exists() and (scene_dir / "sparse" / "0").exists():
            print(f"  {scene}: already extracted")
            continue
        print(f"  Extracting {scene}...")
        subprocess.check_call([
            "unzip", "-q", "-o", str(zip_path),
            f"{scene}/*", "-d", str(out_dir),
        ])

    # Clean up zip to save space (optional)
    # zip_path.unlink()
    print(f"Done. Scenes available at {out_dir}")


def prepare_torch_cache(data_root: str):
    """Cache AlexNet weights used by torchmetrics LPIPS."""
    ckpt = Path(data_root) / "gsplat_data" / "torch_cache" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    download(ALEXNET_URL, ckpt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True,
                   help="Root data directory (e.g., vendor/data)")
    args = p.parse_args()
    download_and_extract(args.data_root)
    prepare_torch_cache(args.data_root)
