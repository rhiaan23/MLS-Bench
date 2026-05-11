#!/usr/bin/env python3
"""Download small slices of LongBench long-context QA subsets.

LongBench (Bai et al., 2024) is the standard long-context QA benchmark. We
use two of its tasks:

  - ``qasper``           single-doc scientific paper QA (4K-16K context)
  - ``multifieldqa_en``  long-document multi-field QA (4-8K context)

Both subsets are saved as deterministic JSONL slices so the harness can
stream them without HF datasets at runtime.

Outputs:
  {data_root}/longbench-qasper/qasper.jsonl
  {data_root}/longbench-qasper/multifieldqa_en.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

NUM_EXAMPLES = 100
SUBSETS = ("qasper", "multifieldqa_en")


def _download_subset(subset, dest, cache_dir, num_examples):
    out_path = dest / f"{subset}.jsonl"
    marker = dest / f".done.{subset}"

    if marker.exists() and out_path.exists():
        n = sum(1 for _ in out_path.open())
        print(f"longbench/{subset}: already downloaded ({n} examples), skipping")
        return

    from datasets import load_dataset

    try:
        ds = load_dataset(
            "THUDM/LongBench", subset, split="test",
            cache_dir=str(cache_dir), trust_remote_code=True,
        )
    except Exception as e:
        print(f"longbench/{subset}: FAILED - {e}", file=sys.stderr)
        sys.exit(1)

    n_take = min(num_examples, len(ds))
    with out_path.open("w") as f:
        for i in range(n_take):
            row = ds[i]
            # Standard LongBench schema: input, context, answers, length, ...
            obj = {
                "input": row.get("input", ""),
                "context": row.get("context", ""),
                "answers": row.get("answers", []),
                "length": row.get("length", 0),
                "_id": row.get("_id", str(i)),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    marker.touch()
    print(f"longbench/{subset}: wrote {n_take} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    dest = Path(args.data_root) / "longbench-qasper"
    dest.mkdir(parents=True, exist_ok=True)

    cache_dir = dest / "_hfcache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))

    # Backwards-compat: an older ``.done`` marker only signaled the qasper
    # subset; remove it so we re-check both subsets.
    legacy_marker = dest / ".done"
    if legacy_marker.exists() and not (dest / ".done.qasper").exists():
        legacy_marker.unlink()

    for subset in SUBSETS:
        _download_subset(subset, dest, cache_dir, args.num_examples)

    # Aggregate marker for upstream "is this dep ready?" checks.
    (dest / ".done").touch()


if __name__ == "__main__":
    main()
