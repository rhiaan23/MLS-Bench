#!/usr/bin/env python3
"""Prepare offline data/cache for diffusers-main CV tasks.

Output layout:
  {data_root}/diffusers_data/
  ├── cifar10/cifar-10-batches-py/...
  ├── cleanfid/
  │   ├── inception-2015-12-05.pt
  │   └── cifar10_clean_train_32.npz
  └── pretrained/hub/checkpoints/vgg16-397923af.pth

The package config bind-mounts this directory at /data, so runtime scripts can
use /data/cifar10, /data/cleanfid, and TORCH_HOME=/data/pretrained without
network access.
"""

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLEANFID_INCEPTION_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
    "pretrained/metrics/inception-2015-12-05.pt"
)
CLEANFID_CIFAR10_STATS_URL = (
    "https://www.cs.cmu.edu/~clean-fid/stats/cifar10_clean_train_32.npz"
)
VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


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


def prepare_cifar10(base: Path) -> None:
    cifar_dir = base / "cifar10"
    extracted = cifar_dir / "cifar-10-batches-py"
    required = [extracted / "data_batch_1", extracted / "test_batch", extracted / "batches.meta"]
    if all(p.exists() and p.stat().st_size > 0 for p in required):
        print(f"[cifar10] ready: {extracted}")
        return

    archive = cifar_dir / "cifar-10-python.tar.gz"
    download(CIFAR10_URL, archive)
    print("[cifar10] extracting...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(cifar_dir)
    if not all(p.exists() and p.stat().st_size > 0 for p in required):
        raise RuntimeError(f"CIFAR-10 extraction incomplete under {cifar_dir}")
    print(f"[cifar10] ready: {extracted}")


def prepare_cleanfid(base: Path) -> None:
    cache = base / "cleanfid"
    download(CLEANFID_INCEPTION_URL, cache / "inception-2015-12-05.pt")
    download(CLEANFID_CIFAR10_STATS_URL, cache / "cifar10_clean_train_32.npz")


def prepare_pretrained(base: Path) -> None:
    ckpt_dir = base / "pretrained" / "hub" / "checkpoints"
    download(VGG16_URL, ckpt_dir / "vgg16-397923af.pth")


def main(data_root: str) -> None:
    base = Path(data_root) / "diffusers_data"
    base.mkdir(parents=True, exist_ok=True)
    prepare_cifar10(base)
    prepare_cleanfid(base)
    prepare_pretrained(base)
    print(f"[done] diffusers data prepared at {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()
    main(args.data_root)
