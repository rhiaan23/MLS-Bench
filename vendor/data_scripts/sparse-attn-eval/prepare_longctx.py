#!/usr/bin/env python3
"""Download a long-context corpus for LLM perplexity evaluation.

Primary: WikiText-103 raw test split (~180MB raw text in HF cache).
Fallback: WikiText-2 raw test split (smaller, still works).

Output: {data_root}/longctx/<HF cache layout>
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    dest = Path(args.data_root) / "longctx"
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(dest)

    marker = dest / ".done"
    if marker.exists():
        print("longctx: already downloaded, skipping")
        return

    from datasets import load_dataset

    try:
        ds_test = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="test",
            cache_dir=str(dest),
        )
        # Also save a flat plaintext file so the harness can read it without HF datasets.
        text = "\n\n".join(ds_test["text"])
        (dest / "wikitext103_test.txt").write_text(text)
        print(f"longctx: wikitext-103 test -> {len(ds_test)} examples, "
              f"{len(text)} chars")
        marker.touch()
        return
    except Exception as e:
        print(f"wikitext-103 download failed: {e}; falling back to wikitext-2",
              file=sys.stderr)

    try:
        ds_test = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test",
            cache_dir=str(dest),
        )
        text = "\n\n".join(ds_test["text"])
        (dest / "wikitext2_test.txt").write_text(text)
        marker.touch()
        print(f"longctx: wikitext-2 test fallback ({len(ds_test)} examples)")
    except Exception as e:
        print(f"longctx: FAILED - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
