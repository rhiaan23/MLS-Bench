#!/usr/bin/env python3
"""Prepare server_data for stabletoolbench.

Downloads the ``stabletoolbench/Cache`` HF dataset and extracts
``server_cache.zip`` into the host data path that the runtime data_bind
mounts into the container. This must run on the host: the pkg_config
install_cmds run inside the image build, where the literal ``{data_root}``
template is NOT expanded by the apptainer or docker build path and the
host data dir is not writable, so an install-time download never lands
on the host.

This script is wired into ``vendor/pkg_configs/stabletoolbench/config.json``
as a ``data_deps`` entry, so ``mlsbench data stabletoolbench`` /
``mlsbench build`` / the agent auto-pipeline invoke it automatically. It
is idempotent (skips the download when ``server_data/tools`` is already
populated).

Run manually via:
    python vendor/data_scripts/stabletoolbench/prepare_data.py \\
        --data-root <data_root>

Output:
    <data_root>/stabletoolbench/server_data/tools/...
    <data_root>/stabletoolbench/server_data/answer/...
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "huggingface_hub"]
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "stabletoolbench" / "server_data"
    if root.exists() and any(root.iterdir()):
        # Cheap check: if `tools/` is present, treat as ready.
        tools = root / "tools"
        if tools.exists() and any(tools.iterdir()):
            print(f"[stabletoolbench] server_data already populated at {root}")
            return 0

    _ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    download_dir = root.parent / "cache_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stabletoolbench] snapshot_download stabletoolbench/Cache -> {download_dir}",
          flush=True)
    snapshot_download(
        "stabletoolbench/Cache",
        repo_type="dataset",
        local_dir=str(download_dir),
    )
    server_zip = download_dir / "server_cache.zip"
    if not server_zip.exists():
        # Some snapshots ship the zip elsewhere; search.
        for candidate in download_dir.rglob("server_cache.zip"):
            server_zip = candidate
            break
    if not server_zip.exists():
        print(f"[stabletoolbench] ERROR: server_cache.zip not found in {download_dir}",
              file=sys.stderr)
        return 1
    print(f"[stabletoolbench] extracting {server_zip} -> {root}", flush=True)
    with zipfile.ZipFile(str(server_zip)) as zf:
        zf.extractall(str(root))
    print("[stabletoolbench] server_data ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
