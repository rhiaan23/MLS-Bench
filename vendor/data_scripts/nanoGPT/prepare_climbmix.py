#!/usr/bin/env python3
"""Prepare ClimbMix training data for nanoGPT tasks.

Downloads selected shards from nvidia/Nemotron-ClimbMix, tokenizes them with
GPT-2 BPE, and produces train.bin / val.bin in headerless uint16 nanoGPT format.

Output: {data_root}/climbmix/train.bin, {data_root}/climbmix/val.bin (~58 GB)

Usage:
    python vendor/data_scripts/nanoGPT/prepare_climbmix.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data nanoGPT
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# Selected ClimbMix main parts (balanced across domains, ~10B tokens total)
SELECTED_PARTS = [9, 10, 18, 21, 23, 28, 29, 33]
EOT_TOKEN = 50256  # GPT-2 <|endoftext|>
VAL_TOKENS = 1_000_000


def download_parts(raw_dir: Path):
    """Download selected ClimbMix parts from HuggingFace."""
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)

    for part_id in SELECTED_PARTS:
        fname = f"part_{part_id}.tokenized.jsonl"
        dest = raw_dir / fname
        if dest.exists():
            print(f"  Part {part_id}: already downloaded")
            continue
        print(f"  Downloading part_{part_id}...", flush=True)
        hf_hub_download(
            repo_id="nvidia/Nemotron-ClimbMix",
            filename=fname,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"  Part {part_id}: done")


def tokenize_part(jsonl_path: Path, bin_path: Path):
    """Convert a .tokenized.jsonl part to headerless uint16 .bin."""
    if bin_path.exists():
        print(f"  {bin_path.name}: already exists, skipping")
        return

    print(f"  Processing {jsonl_path.name}...", flush=True)

    # First pass: count total tokens
    total_len = 0
    with open(jsonl_path) as fh:
        for line in fh:
            row = json.loads(line)
            total_len += len(row["tokens"]) + 1  # +1 for EOT

    print(f"    Total tokens (with EOT): {total_len:,}")

    # Second pass: write to bin
    arr = np.memmap(str(bin_path), dtype=np.uint16, mode="w+", shape=(total_len,))
    idx = 0
    with open(jsonl_path) as fh:
        for line_num, line in enumerate(fh):
            row = json.loads(line)
            tokens = row["tokens"]
            tokens.append(EOT_TOKEN)
            batch = np.array(tokens, dtype=np.uint16)
            arr[idx : idx + len(batch)] = batch
            idx += len(batch)
            if line_num % 100000 == 0:
                print(f"    {line_num} docs, {idx:,} tokens written", flush=True)

    arr.flush()
    print(f"  {bin_path.name}: {total_len:,} tokens written")


def merge_bins(bin_dir: Path, out_dir: Path):
    """Merge individual part bins into train.bin + val.bin."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    if train_path.exists() and val_path.exists():
        print("  train.bin and val.bin already exist, skipping merge")
        return

    # Calculate total tokens
    total_tokens = 0
    part_files = []
    for part_id in SELECTED_PARTS:
        bf = bin_dir / f"part_{part_id}_train.bin"
        if not bf.exists():
            print(f"  WARNING: {bf} not found, skipping")
            continue
        n = os.path.getsize(bf) // 2  # uint16 = 2 bytes
        total_tokens += n
        part_files.append((bf, n))
        print(f"  Part {part_id}: {n:,} tokens")

    train_tokens = total_tokens - VAL_TOKENS
    print(f"\n  Total: {total_tokens:,} tokens")
    print(f"  Train: {train_tokens:,} | Val: {VAL_TOKENS:,}")

    # Write merged files
    print(f"  Writing {train_path}...")
    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        written = 0
        for bf, n in part_files:
            data = np.fromfile(str(bf), dtype=np.uint16)
            remaining_train = train_tokens - written
            if remaining_train >= len(data):
                f_train.write(data.tobytes())
                written += len(data)
            else:
                if remaining_train > 0:
                    f_train.write(data[:remaining_train].tobytes())
                    f_val.write(data[remaining_train:].tobytes())
                    written += len(data)
                else:
                    f_val.write(data.tobytes())
                    written += len(data)

    print(f"  Done: train.bin ({os.path.getsize(train_path) / 1e9:.1f} GB), "
          f"val.bin ({os.path.getsize(val_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare ClimbMix data for nanoGPT")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "climbmix_raw"
    bin_dir = data_root / "climbmix_bin"
    out_dir = data_root / "climbmix"

    bin_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Download ClimbMix parts ===")
    download_parts(raw_dir)

    print("\n=== Step 2: Tokenize parts to .bin ===")
    for part_id in SELECTED_PARTS:
        jsonl = raw_dir / f"part_{part_id}.tokenized.jsonl"
        binf = bin_dir / f"part_{part_id}_train.bin"
        if jsonl.exists():
            tokenize_part(jsonl, binf)

    print("\n=== Step 3: Merge into train.bin / val.bin ===")
    merge_bins(bin_dir, out_dir)

    print("\n=== ClimbMix data preparation complete ===")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
