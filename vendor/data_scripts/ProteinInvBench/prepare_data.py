#!/usr/bin/env python3
"""Prepare data for ProteinInvBench.

Downloads CATH4.2 / CATH4.3 / TS datasets from the upstream GitHub release.
Output:
    <data_root>/ProteinInvBench/cath4.2/...
    <data_root>/ProteinInvBench/cath4.3/...
    <data_root>/ProteinInvBench/ts/*.json
The pkg config binds {data_root}/ProteinInvBench into /workspace/data
inside the container (matching --data-root /workspace/data in scripts).

Run via:
    mlsbench data ProteinInvBench
"""

import argparse
import glob
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATA_TARBALL_URL = (
    "https://github.com/A4Bio/ProteinInvBench/releases/download/"
    "dataset_release/data.tar.gz"
)
TS_ZIP_URL = (
    "https://github.com/A4Bio/ProteinInvBench/releases/download/"
    "dataset_release/TSDataset.zip"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest} already downloaded")
        return
    print(f"  Downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


def prepare_cath(root: Path) -> None:
    if all(
        (root / d).exists() and any((root / d).iterdir())
        for d in ("cath4.2", "cath4.3")
    ):
        print("  [SKIP] CATH4.2/CATH4.3 already populated")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "data.tar.gz"
        download(DATA_TARBALL_URL, tarball)
        with tarfile.open(str(tarball), "r:gz") as tar:
            tar.extractall(tmp)
        for sub in ("cath4.2", "cath4.3"):
            src = Path(tmp) / "data" / sub
            dst = root / sub
            dst.mkdir(parents=True, exist_ok=True)
            if not src.is_dir():
                raise RuntimeError(f"Expected {src} in tarball")
            for fname in os.listdir(src):
                shutil.copy2(src / fname, dst / fname)
            print(f"  {sub}: {sorted(os.listdir(dst))}")


def prepare_ts(root: Path) -> None:
    ts_dir = root / "ts"
    if ts_dir.exists() and list(ts_dir.glob("*.json")):
        print("  [SKIP] TS dataset already populated")
        return
    ts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "TSDataset.zip"
        download(TS_ZIP_URL, zpath)
        with zipfile.ZipFile(str(zpath)) as zf:
            zf.extractall(tmp)
        for f in glob.glob(os.path.join(tmp, "**", "*.json"), recursive=True):
            shutil.copy2(f, ts_dir / os.path.basename(f))
        print(f"  TS: {sorted(os.listdir(ts_dir))}")


def verify(root: Path) -> None:
    needed = []
    for sub in ("cath4.2", "cath4.3"):
        d = root / sub
        if not (d.exists() and any(d.iterdir())):
            needed.append(f"{sub}/")
    if not list((root / "ts").glob("*.json")):
        needed.append("ts/*.json")
    if needed:
        print("ERROR: missing artifacts:", needed, file=sys.stderr)
        sys.exit(1)
    print("All ProteinInvBench data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    root = Path(args.data_root) / "ProteinInvBench"
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing ProteinInvBench data at {root} ===")
    prepare_cath(root)
    prepare_ts(root)
    verify(root)


if __name__ == "__main__":
    main()
