#!/usr/bin/env python3
"""Prepare the OpenR1 parquet used by the MLS-Bench HPT task."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenR1 math training data")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"datasets import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.data_root) / "upt-data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "openr1.parquet"

    if out_path.exists():
        print(f"{out_path}: already exists, skipping")
        return

    print("Downloading Elliott/Openr1-Math-46k-8192...", flush=True)
    dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

    rows = []
    for item in dataset:
        record = dict(item)
        record["prompt"] = record["prompt"][1:]
        rows.append(record)

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
