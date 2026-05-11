#!/usr/bin/env python3
"""Prepare data for Neural-Solver-Library tasks (PDE benchmarks).

Downloads the datasets used by tasks that reference this package
(currently `pde-design-solver`, which uses AirfRANS, mlcfd, and AirCraft)
into {data_root}/Neural-Solver-Library/. The pkg config binds that dir
into /data inside the container.

Sources:
  - mlcfd: http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
    (Multi-physics CFD demo dataset, public download from the author site.)
  - AirfRANS: https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
    (Bonnet et al., NeurIPS 2022; public release.)
  - AirCraft: Google Drive id 1UDGgtOM8UYBFbDe_t2FP7Ij9N5SA3w-g — the
    LSM/Transolver-style internal aerodynamics CFD dataset used by
    Neural-Solver-Library `Custom_Car`-style tasks. The bundle is a
    public Google Drive share (no login required) but is not on a
    formal data portal; the file id is pinned here so re-runs are
    deterministic.

Run via:
    mlsbench data Neural-Solver-Library
"""

import argparse
import os
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_DOWNLOADS = [
    (
        "http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip",
        "mlcfd_data.zip",
        "PDE_data",
    ),
    (
        "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
        "AirfRANS.zip",
        "AirfRANS",
    ),
]

AIRCRAFT_GDRIVE_ID = "1UDGgtOM8UYBFbDe_t2FP7Ij9N5SA3w-g"

EXPECTED_DIRS = [
    "PDE_data/mlcfd_data",
    "AirfRANS",
    "AirCraft",
]


def gdrive_download(file_id: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest} already downloaded ({dest.stat().st_size} bytes)")
        return
    try:
        import gdown  # type: ignore

        gdown.download(id=file_id, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 10_000:
            return
    except ImportError:
        pass
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            page = urllib.request.urlopen(
                "https://drive.google.com/uc?export=download&id=" + file_id
            ).read().decode(errors="ignore")
            if "Quota exceeded" in page:
                wait = 60 * attempt
                print(f"  Quota exceeded, retry in {wait}s ({attempt}/{retries})", flush=True)
                time.sleep(wait)
                continue
            m = re.search(r'uuid.*?value="([^"]+)', page)
            if not m:
                raise RuntimeError("No uuid in confirm page")
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t&uuid={m.group(1)}"
            )
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  Error: {e}, retry in {60 * attempt}s")
            time.sleep(60 * attempt)
    raise RuntimeError(f"Failed Google Drive id={file_id} -> {dest}: {last_err}")


def url_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest} already downloaded")
        return
    print(f"  Downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


def extract_zip(zip_path: Path, target_dir: Path, remove: bool = True) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} -> {target_dir}")
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(target_dir))
    if remove:
        zip_path.unlink()


def extract_tarballs_in(parent: Path) -> None:
    """Extract any *.tar.gz / *.tgz archives in `parent` to sibling dirs.

    The mlcfd_data.zip ships nested per-param tarballs at
    `mlcfd_data/training_data/param{0..8}.tar.gz`. The Car loader expects
    them already extracted as directories `param0/`, etc. We leave the
    source tarballs in place so the data dir can be re-validated cheaply.
    """
    if not parent.exists():
        return
    for archive in sorted(parent.iterdir()):
        if not archive.is_file():
            continue
        name = archive.name
        if name.endswith(".tar.gz"):
            stem = name[:-len(".tar.gz")]
        elif name.endswith(".tgz"):
            stem = name[:-len(".tgz")]
        else:
            continue
        out_dir = parent / stem
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [SKIP] {out_dir} already extracted")
            continue
        print(f"  Extracting {archive} -> {parent}/")
        with tarfile.open(str(archive), "r:gz") as tf:
            try:
                tf.extractall(str(parent), filter="data")  # py3.12+
            except TypeError:
                tf.extractall(str(parent))


def prepare_archives(root: Path) -> None:
    for url, name, target_sub in ARCHIVE_DOWNLOADS:
        target = root / target_sub
        if target.exists() and any(target.iterdir()):
            print(f"  [SKIP] {target} already populated")
            continue
        zip_path = root / name
        url_download(url, zip_path)
        extract_zip(zip_path, target, remove=True)

    # mlcfd_data ships nested per-param tarballs that the Car loader expects
    # as already-extracted directories. Extract them here (idempotent).
    mlcfd_train = root / "PDE_data" / "mlcfd_data" / "training_data"
    extract_tarballs_in(mlcfd_train)


def prepare_aircraft(root: Path) -> None:
    target = root / "AirCraft"
    if target.exists() and any(target.iterdir()):
        print(f"  [SKIP] {target} already populated")
        return
    zip_path = root / "AirCraft.zip"
    gdrive_download(AIRCRAFT_GDRIVE_ID, zip_path)
    extract_zip(zip_path, target, remove=True)


def verify(root: Path) -> None:
    missing = []
    for rel in EXPECTED_DIRS:
        path = root / rel
        if not path.exists() or not any(path.iterdir()):
            missing.append(rel + "/")
    if missing:
        print("\nERROR: missing artifacts:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    # AirCraft loader (vendor/pkg_configs/Neural-Solver-Library/pre_edit.py)
    # expects an airplane_dataset.json manifest plus HDF5 files containing
    # "pos", "normals", and "values" keys. Verify those, not just nonempty.
    aircraft = root / "AirCraft"
    manifest = aircraft / "airplane_dataset.json"
    if not manifest.exists():
        print(
            "\nERROR: AirCraft/airplane_dataset.json missing. "
            "The bundle extracted at the wrong layout — look for it under "
            f"{aircraft} or its subdirs.",
            file=sys.stderr,
        )
        sys.exit(1)
    h5_files = list(aircraft.glob("*.h5"))
    if not h5_files:
        # Some bundles wrap content in a subdir; surface that explicitly.
        nested = [p for p in aircraft.glob("**/*.h5")][:1]
        hint = f" (found nested at {nested[0]})" if nested else ""
        print(f"\nERROR: no .h5 files at top level of {aircraft}{hint}", file=sys.stderr)
        sys.exit(1)
    try:
        import h5py  # type: ignore

        with h5py.File(str(h5_files[0]), "r") as f:
            keys = set(f.keys())
        needed = {"pos", "normals", "values"}
        if not needed.issubset(keys):
            print(
                f"\nERROR: AirCraft sample {h5_files[0].name} missing keys "
                f"{needed - keys} (has {sorted(keys)})",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        # h5py not on host; skip the deep check, the runtime container has it.
        pass

    print("\nAll Neural-Solver-Library data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "Neural-Solver-Library"
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing Neural-Solver-Library data at {root} ===")

    prepare_archives(root)
    prepare_aircraft(root)
    verify(root)


if __name__ == "__main__":
    main()
