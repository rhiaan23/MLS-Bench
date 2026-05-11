"""Prepare data for AbBiBench package.

Downloads ESM-2 3B model weights to the host-side transformers_cache so the
container can find them via the data_bind at runtime. The container's
install_cmds also caches the model into the writable sandbox, but that
sandbox cache is shadowed by the data_bind at runtime, so we must populate
the host cache too.

Run via: mlsbench data AbBiBench

Creates:
  <data_root>/transformers_cache/hub/models--facebook--esm2_t36_3B_UR50D/...
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Host data root (mlsbench data_root)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_dir = data_root / "transformers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use the same env vars HF respects
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    from huggingface_hub import snapshot_download

    for model_id in ("facebook/esm2_t36_3B_UR50D",
                     "facebook/esm2_t33_650M_UR50D"):
        print(f"[AbBiBench data_prep] Downloading {model_id} -> {cache_dir}",
              flush=True)
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir / "hub"),
        )

    print("[AbBiBench data_prep] Done.")


if __name__ == "__main__":
    main()
