#!/usr/bin/env python3
"""Prepare metric model weights used by CFGpp-main tasks.

Output:
  {data_root}/cfgpp_model_weights/
  ├── ViT-B-32.pt
  └── inception_v3_google.pth

This directory is mounted at /opt/model_weights for Docker/Apptainer and
translated to the same host path for local/Conda runtime.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve


CLIP_VIT_B32_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
    "ViT-B-32.pt"
)
INCEPTION_V3_URL = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CFGpp metric model weights")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "cfgpp_model_weights"
    download(CLIP_VIT_B32_URL, out_dir / "ViT-B-32.pt")
    download(INCEPTION_V3_URL, out_dir / "inception_v3_google.pth")
    print(f"[done] CFGpp model weights prepared at {out_dir}")


if __name__ == "__main__":
    main()
