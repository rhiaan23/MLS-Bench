#!/usr/bin/env python3
"""Generate a deterministic Needle-In-A-Haystack (NIAH) test set.

NIAH (Kamradt 2023) inserts a single fact (the "needle") at a controlled depth
in a long filler context (the "haystack") and asks the model to retrieve it.
This is the canonical benchmark for evaluating KV-cache compression / sparse
attention in LLM long-context inference (used in H2O, SnapKV, Quest,
MInference).

We generate the haystack from a deterministic large filler (Paul Graham essays
proxy: long repetitive English text drawn from WikiText-103 if available, else
synthetic filler). We pick 5 needles x 10 depths = 50 cases.

Output: {data_root}/niah/cases.jsonl  (each line: needle, haystack_seed,
        depth_pct, question, answer)

We don't store the inflated 8K haystack here — the runtime harness rebuilds it
deterministically from the haystack source plus the (needle, depth_pct).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

NUM_NEEDLES = 5
NUM_DEPTHS = 10  # 0%, 11%, 22%, ..., 99%

# Each entry: (city, magic_number)
NEEDLES = [
    ("San Francisco", "9472831"),
    ("Tokyo", "1837465"),
    ("Buenos Aires", "5294817"),
    ("Reykjavik", "6385129"),
    ("Mumbai", "2748391"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    dest = Path(args.data_root) / "niah"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "cases.jsonl"
    marker = dest / ".done"

    if marker.exists() and out_path.exists():
        n = sum(1 for _ in out_path.open())
        print(f"niah: already prepared ({n} cases), skipping")
        return

    cases = []
    for ni, (city, num) in enumerate(NEEDLES[:NUM_NEEDLES]):
        for di in range(NUM_DEPTHS):
            depth_pct = di / max(NUM_DEPTHS - 1, 1)  # 0.0 to 1.0
            sentence = (
                f"The special magic number for {city} is {num}."
            )
            question = f"What is the special magic number for {city}?"
            answer = num
            cases.append({
                "case_id": f"{ni}_{di}",
                "needle": sentence,
                "depth_pct": round(depth_pct, 4),
                "question": question,
                "answer": answer,
                "city": city,
            })

    with out_path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    marker.touch()
    print(f"niah: wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
