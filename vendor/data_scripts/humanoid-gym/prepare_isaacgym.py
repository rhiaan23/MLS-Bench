#!/usr/bin/env python3
"""Prepare Isaac Gym Preview 4 for humanoid-gym tasks.

Downloads the public Isaac Gym Preview 4 tarball from NVIDIA's CDN (the
`developer.nvidia.com/isaac-gym-preview-4` URL redirects to a time-signed
download link on developer.download.nvidia.com, no login required) and
extracts it to `{data_root}/isaacgym` so that the humanoid-gym container
can bind-mount it at /tmp/isaacgym and `pip install -e /tmp/isaacgym/python`.

Output layout (after a successful run):
    {data_root}/isaacgym/
        python/                 # `pip install -e python` target
        docs/ assets/ ...
        .ready                  # sentinel

Usage:
    python vendor/data_scripts/humanoid-gym/prepare_isaacgym.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data humanoid-gym
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


DOWNLOAD_URL = "https://developer.nvidia.com/isaac-gym-preview-4"
# ~200 MB tarball; signed-token 302 redirect points to developer.download.nvidia.com.
EXPECTED_MIN_BYTES = 150_000_000


def download(tarball: Path) -> None:
    if tarball.exists() and tarball.stat().st_size >= EXPECTED_MIN_BYTES:
        print(f"  Tarball already downloaded: {tarball} ({tarball.stat().st_size} bytes)")
        return

    tarball.parent.mkdir(parents=True, exist_ok=True)
    tmp = tarball.with_suffix(tarball.suffix + ".part")
    print(f"  Downloading Isaac Gym Preview 4 from {DOWNLOAD_URL} ...", flush=True)
    # -L: follow signed-token redirect; -A: NVIDIA's CDN rejects empty/curl UA.
    cmd = [
        "curl", "-L", "-A", "Mozilla/5.0", "--fail", "--retry", "3",
        "--connect-timeout", "30", "-o", str(tmp), DOWNLOAD_URL,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"curl failed (exit {result.returncode}). The NVIDIA signed-token "
            f"URL may have expired or network is unavailable."
        )
    size = tmp.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        raise RuntimeError(
            f"Downloaded tarball is too small ({size} bytes). "
            f"Expected >={EXPECTED_MIN_BYTES}. File may be an HTML error page."
        )
    tmp.replace(tarball)
    print(f"  Downloaded {tarball} ({size} bytes)")


def extract(tarball: Path, target: Path) -> None:
    sentinel = target / ".ready"
    if sentinel.exists() and (target / "python" / "setup.py").exists():
        print(f"  Already extracted at {target}")
        return

    # The tarball contains a top-level `isaacgym/` directory; we want its
    # contents to live directly under {data_root}/isaacgym (so the python
    # package sits at {data_root}/isaacgym/python).
    staging = target.parent / f".isaacgym_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"  Extracting to {staging} ...", flush=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(staging)

    # Locate the extracted top-level dir (expected: staging/isaacgym)
    entries = [p for p in staging.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        src_root = entries[0]
    else:
        # Fallback: tarball may extract flat
        src_root = staging

    if not (src_root / "python" / "setup.py").exists():
        raise RuntimeError(
            f"Extracted tarball does not contain python/setup.py at {src_root}. "
            f"Contents: {list(src_root.iterdir())}"
        )

    # Move contents into target
    target.mkdir(parents=True, exist_ok=True)
    for child in src_root.iterdir():
        dest = target / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))

    shutil.rmtree(staging)
    sentinel.touch()
    print(f"  Ready at {target}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Isaac Gym Preview 4 for humanoid-gym")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    target = data_root / "isaacgym"
    tarball = data_root / "_downloads" / "IsaacGym_Preview_4_Package.tar.gz"

    if (target / ".ready").exists() and (target / "python" / "setup.py").exists():
        print(f"[OK] Isaac Gym already prepared at {target}")
        return

    print(f"Target: {target}")
    print(f"Tarball cache: {tarball}")

    download(tarball)
    extract(tarball, target)

    print(f"[OK] Isaac Gym prepared at {target}")


if __name__ == "__main__":
    main()
