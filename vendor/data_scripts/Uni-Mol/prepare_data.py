"""Prepare data for Uni-Mol package.

Downloads the official Uni-Mol pre-split molecular property prediction data
(LMDB format with train/valid/test splits and pre-computed conformers),
plus pretrained weights.

Run via: mlsbench data Uni-Mol

Creates:
  <data_root>/Uni-Mol/molecular_property_prediction/<dataset>/{train,valid,test}.lmdb
  <data_root>/Uni-Mol/unimol_weights/mol_pre_all_h_220816.pt
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


OFFICIAL_DATA_URL = (
    "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/"
    "unimol_data/finetune/molecular_property_prediction.tar.gz"
)

PRETRAINED_WEIGHTS = {
    "mol_pre_all_h_220816.pt": (
        "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/"
        "mol_pre_all_h_220816.pt"
    ),
}

# Datasets we use (subset of what the tarball contains)
REQUIRED_DATASETS = ["bbbp", "bace", "tox21", "esol", "freesolv", "lipo"]


def download(url: str, dest: str) -> None:
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root) / "Uni-Mol"
    root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download and extract official pre-split molecular property data
    # ------------------------------------------------------------------
    mol_prop_dir = root / "molecular_property_prediction"

    # Check if already extracted
    all_present = True
    for ds in REQUIRED_DATASETS:
        for split in ["train", "valid", "test"]:
            if not (mol_prop_dir / ds / f"{split}.lmdb").exists():
                all_present = False
                break
        if not all_present:
            break

    if all_present:
        print("  [SKIP] Official molecular property data already extracted")
    else:
        tarball_path = root / "molecular_property_prediction.tar.gz"
        if not tarball_path.exists():
            download(OFFICIAL_DATA_URL, str(tarball_path))
        print("  Extracting molecular_property_prediction.tar.gz ...")
        with tarfile.open(str(tarball_path), "r:gz") as tar:
            tar.extractall(path=str(root))
        print("  Extraction complete.")
        # Clean up tarball to save space
        tarball_path.unlink()

    # ------------------------------------------------------------------
    # Download pretrained weights
    # ------------------------------------------------------------------
    weights_dir = root / "unimol_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for name, url in PRETRAINED_WEIGHTS.items():
        dest = weights_dir / name
        if dest.exists():
            print(f"  [SKIP] unimol_weights/{name} already exists")
        else:
            download(url, str(dest))
            print(f"  Downloaded unimol_weights/{name}")

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    checks = []
    for ds in REQUIRED_DATASETS:
        for split in ["train", "valid", "test"]:
            checks.append(mol_prop_dir / ds / f"{split}.lmdb")
    for name in PRETRAINED_WEIGHTS:
        checks.append(root / "unimol_weights" / name)

    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    print("All Uni-Mol data verified.")


if __name__ == "__main__":
    main()
