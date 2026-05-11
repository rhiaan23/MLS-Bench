#!/usr/bin/env python3
"""Prepare the shared Hugging Face cache for transformers-kv-lab tasks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)


def ensure_prepare_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "datasets",
            "huggingface_hub",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare transformers-kv-lab HF cache")
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    cache_root = Path(args.data_root) / "huggingface_cache"
    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    marker = cache_root / ".transformers-kv-lab-ready"
    if marker.exists():
        print(f"transformers-kv-lab HF cache already prepared at {cache_root}")
        return 0

    try:
        ensure_prepare_dependencies()

        from datasets import load_dataset
        from huggingface_hub import hf_hub_download, snapshot_download

        for repo_id in MODELS:
            print(f"Downloading model snapshot: {repo_id}", flush=True)
            snapshot_download(repo_id=repo_id, cache_dir=str(hub_cache))

        print("Downloading THUDM/LongBench data.zip", flush=True)
        hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=str(hub_cache),
        )

        print("Caching THUDM/LongBench-v2 train split", flush=True)
        load_dataset("THUDM/LongBench-v2", split="train", cache_dir=str(datasets_cache))

        print("Caching openai/gsm8k main test split", flush=True)
        load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(datasets_cache))

        print("Downloading opencompass/NeedleBench/PaulGrahamEssays.jsonl", flush=True)
        hf_hub_download(
            repo_id="opencompass/NeedleBench",
            filename="PaulGrahamEssays.jsonl",
            repo_type="dataset",
            cache_dir=str(hub_cache),
        )
    except Exception as exc:
        print(f"transformers-kv-lab HF cache preparation failed: {exc}", file=sys.stderr)
        return 1

    marker.touch()
    print(f"transformers-kv-lab HF cache ready at {cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
