"""Prepare data for Uni-3DAR package.

Downloads molecular/crystal datasets from HuggingFace (dptech/Uni-3DAR).
Run via: mlsbench data Uni-3DAR

Creates:
  <data_root>/Uni-3DAR/qm9/{train.lmdb, valid.lmdb, ...}
  <data_root>/Uni-3DAR/geom_drug/{train.lmdb, valid.lmdb, ...}
  <data_root>/Uni-3DAR/mp20/{train.lmdb, valid.lmdb, ...}
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


HF_REPO = "dptech/Uni-3DAR"
DATASETS = {
    "qm9": {"archive": "qm9_data.tar.gz", "subdir": "qm9_data", "dest": "qm9"},
    "geom_drug": {"archive": "drug_data.tar.gz", "subdir": "drug_data", "dest": "geom_drug"},
    "mp20": {"archive": "mp20_data.tar.gz", "subdir": "mp20_data", "dest": "mp20"},
}


def download_hf(repo_id: str, filename: str, local_dir: str) -> str:
    """Download a file from HuggingFace Hub."""
    print(f"  Downloading {repo_id}/{filename} -> {local_dir}")
    subprocess.run(
        [
            sys.executable, "-c",
            f"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='{repo_id}', filename='{filename}', local_dir='{local_dir}')"
        ],
        check=True,
    )
    return os.path.join(local_dir, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root) / "Uni-3DAR"
    root.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/uni3dar_dl")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        dest_dir = root / info["dest"]
        # Check if data already exists (look for .lmdb files)
        if dest_dir.exists() and any(dest_dir.glob("*.lmdb")):
            print(f"  [SKIP] {name} already exists at {dest_dir}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Download archive
        archive_path = tmp_dir / info["archive"]
        if not archive_path.exists():
            download_hf(HF_REPO, info["archive"], str(tmp_dir))

        # Extract
        print(f"  Extracting {info['archive']}...")
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(path=str(tmp_dir))

        # Move contents from extracted subdir to destination
        src_dir = tmp_dir / info["subdir"]
        if src_dir.exists():
            for item in src_dir.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    item.rename(target)
            src_dir.rmdir()
        else:
            print(f"  WARNING: Expected subdir {info['subdir']} not found after extraction")

        # Clean up archive
        if archive_path.exists():
            archive_path.unlink()

        print(f"  Extracted {name} to {dest_dir}")

    # Verify
    checks = []
    for name, info in DATASETS.items():
        dest_dir = root / info["dest"]
        for split in ["train", "valid"]:
            checks.append(dest_dir / f"{split}.lmdb")

    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    print("All Uni-3DAR data verified.")


if __name__ == "__main__":
    main()
