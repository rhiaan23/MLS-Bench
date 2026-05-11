"""Prepare data for CBGBench package.

Downloads CrossDocked2020 processed datasets from Google Drive.
Run via: mlsbench data CBGBench

Creates:
  <data_root>/CBGBench/crossdocked_v1.1_rmsd1.0_pocket10/  (raw pocket data)
  <data_root>/CBGBench/pl/  (processed LMDB for de novo design)
  <data_root>/CBGBench/pl_decomp/  (processed LMDB for fragment/linker/scaffold)
  <data_root>/CBGBench/split_by_name_10m.pt  (train/test split)

Data source: https://drive.google.com/drive/folders/1wm5_rMbemxqMiTxoBr_V-Vt5NyNtdZT7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Google Drive file IDs for CBGBench CrossDocked2020 processed data
# From the official CBGBench repo README
GDRIVE_FILES = {
    # Processed LMDB for de novo (fullatom) -- pl/ directory
    "pl": {
        "id": "1wm5_rMbemxqMiTxoBr_V-Vt5NyNtdZT7",
        "type": "folder",
    },
}

# Individual files to download via gdown
# These are the specific files needed for our tasks
DOWNLOADS = [
    {
        "name": "pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb",
        "gdrive_id": "1-4JTaw8DotSLqYJG_m_BjVqAV-jJPUeH",
        "check": "pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb",
    },
    {
        "name": "pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb",
        "gdrive_id": "18J1FJ0rpKGxN95PiLkfSh8JyxYjjPdSp",
        "check": "pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb",
    },
    {
        "name": "pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb",
        "gdrive_id": "1Vc-ePEPVXVnYSN8OGJOCMbLMkWMxbrz7",
        "check": "pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb",
    },
    {
        "name": "split_by_name_10m.pt",
        "gdrive_id": "1UJE2n-fFIxkMnMLogicXeSVKImageqJd1",
        "check": "split_by_name_10m.pt",
    },
]

# Raw pocket data (from TargetDiff Google Drive)
RAW_DATA = {
    "name": "crossdocked_v1.1_rmsd1.0_pocket10",
    "gdrive_id": "1r2sJhPMGjKjHMJuYPBagjR3KJnFLaNze",
    "archive": "crossdocked_v1.1_rmsd1.0_pocket10.tar.gz",
}


def gdown_file(file_id: str, dest: str) -> None:
    """Download a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {url} -> {dest}")
    subprocess.run(
        [sys.executable, "-m", "gdown", url, "-O", dest],
        check=True,
    )


def gdown_folder(folder_id: str, dest: str) -> None:
    """Download a folder from Google Drive using gdown."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"  Downloading folder {url} -> {dest}")
    subprocess.run(
        [sys.executable, "-m", "gdown", url, "-O", dest, "--folder"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root) / "CBGBench"
    root.mkdir(parents=True, exist_ok=True)

    # Ensure subdirectories exist
    (root / "pl").mkdir(parents=True, exist_ok=True)
    (root / "pl_decomp").mkdir(parents=True, exist_ok=True)

    # Download processed LMDB files
    for item in DOWNLOADS:
        dest = root / item["check"]
        if dest.exists():
            print(f"  [SKIP] {item['name']} already exists")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown_file(item["gdrive_id"], str(dest))
        print(f"  Downloaded {item['name']}")

    # Download raw pocket data
    raw_dir = root / RAW_DATA["name"]
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"  [SKIP] {RAW_DATA['name']} already exists")
    else:
        archive_path = f"/tmp/{RAW_DATA['archive']}"
        gdown_file(RAW_DATA["gdrive_id"], archive_path)
        print(f"  Extracting {RAW_DATA['archive']}...")
        subprocess.run(
            ["tar", "-xzf", archive_path, "-C", str(root)],
            check=True,
        )
        os.remove(archive_path)
        print(f"  Extracted {RAW_DATA['name']}")

    # Verify
    checks = [
        root / "pl" / "crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb",
        root / "pl_decomp" / "crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb",
        root / "pl_decomp" / "crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb",
        root / "split_by_name_10m.pt",
        root / "crossdocked_v1.1_rmsd1.0_pocket10",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    print("All CBGBench data verified.")


if __name__ == "__main__":
    main()
