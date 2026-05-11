#!/usr/bin/env python3
"""Download the SLDBench subsets used by the task at build time."""

import json
import os
from pathlib import Path

from datasets import load_dataset


# Container default is /data/scaling_law; local mode overrides via env.
ROOT = Path(os.environ.get("SCALING_LAW_DATA_DIR", "/data/scaling_law"))
ROOT.mkdir(parents=True, exist_ok=True)


def dump_dataset(dataset_name: str, config_name: str, split: str, prefix: str) -> int:
    ds = load_dataset(dataset_name, config_name, split=split)
    path = ROOT / f"{prefix}__{config_name}__{split}.jsonl"
    ds.to_json(str(path))
    print(f"Saved {dataset_name}/{config_name}/{split} -> {path} ({len(ds)} rows)", flush=True)
    return len(ds)


# Harder SLDBench subsets recommended by sldbench authors (see SLDAgent paper).
# NOTE: `lr_bsz_scaling_law_modified` is declared in the HF README but its
# parquet files are missing upstream (404 on resolve/main). We fall back to the
# published `lr_bsz_scaling_law` subset, which has the same schema. Revisit if
# upstream publishes the modified variant.
manifest = {}
for cfg in (
    "vocab_scaling_law",
    "lr_bsz_scaling_law",
    "data_constrained_scaling_law",
):
    for split in ("train", "test"):
        manifest[f"{cfg}/{split}"] = dump_dataset("pkuHaowei/sldbench", cfg, split, "sldbench")

(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
