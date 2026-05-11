#!/usr/bin/env python3
"""Pre-download HF datasets needed by dlm-dkv-policy evaluation.

Outputs go to {data_root}/huggingface_cache/datasets/ (matches HF_DATASETS_CACHE
in vendor/pkg_configs/dLLM-cache/config.json).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    cache_root = Path(args.data_root) / "huggingface_cache"
    datasets_cache = cache_root / "datasets"
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    marker = cache_root / ".dlm-dkv-datasets-ready"
    # Treat 0-byte markers as missing (the readiness check in
    # cli._path_has_content() rejects zero-byte files), so a previous
    # half-completed prep run does NOT short-circuit a retry.
    if marker.exists() and marker.stat().st_size > 0:
        print(f"dlm-dkv HF datasets already prepared at {datasets_cache}")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "datasets"])
        from datasets import load_dataset

    targets = [
        ("HuggingFaceH4/MATH-500", None, "test", {}),
        ("openai_humaneval", None, "test", {"trust_remote_code": True}),
        ("allenai/ai2_arc", "ARC-Challenge", "test", {}),
    ]

    for repo, name, split, extra in targets:
        label = f"{repo}{(':' + name) if name else ''}/{split}"
        print(f"Downloading {label}", flush=True)
        if name is None:
            load_dataset(repo, split=split, cache_dir=str(datasets_cache), **extra)
        else:
            load_dataset(repo, name, split=split, cache_dir=str(datasets_cache), **extra)

    # Use ``write_text`` so the marker is non-empty; the readiness check
    # treats 0-byte files as MISSING.
    marker.write_text("ready\n")
    print(f"dlm-dkv HF datasets ready at {datasets_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
