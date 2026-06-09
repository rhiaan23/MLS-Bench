#!/usr/bin/env python3
"""Pre-download openai/clip-vit-large-patch14 into the HF cache layout that
``robomimic.utils.lang_utils`` expects.

``lang_utils.py`` runs at module-import time:

    CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip")),
    )

so we populate ``{data_root}/robomimic/hf_cache_clip/`` such that with
``HF_HOME=/data/huggingface`` in the container and a bind/COPY of that host dir
to ``/data/huggingface/clip``, the cache lookup hits a baked snapshot instead
of trying to fetch from the Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "openai/clip-vit-large-patch14"

# Files the HF cache layout will contain after a successful snapshot_download.
# We check a couple of representative ones rather than re-walking the whole
# snapshot, matching the pattern used by prepare_llada_instruct.py.
_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _snapshot_dir(cache_dir: Path) -> Path | None:
    """Return the snapshots/<sha> dir inside the HF cache, or None."""
    repo_dir = cache_dir / "models--openai--clip-vit-large-patch14"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for sub in snapshots.iterdir():
        if sub.is_dir():
            return sub
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "robomimic" / "hf_cache_clip"
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = _snapshot_dir(cache_dir)
    if existing is not None and all((existing / f).exists() for f in _REQUIRED_FILES):
        has_weights = (existing / "pytorch_model.bin").exists() or any(
            existing.glob("model*.safetensors")
        ) or any(existing.glob("pytorch_model*.bin"))
        if has_weights:
            print(f"CLIP already cached at {existing}, skipping")
            return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub not installed on host; declare it in "
            "pkg_config.host_data_prepare_requirements",
            file=sys.stderr,
        )
        return 1

    # Only fetch what the PyTorch path needs. The HF repo ships flax + tf +
    # pytorch_model.bin + safetensors snapshots; together that's ≈6 GB. Image
    # size matters here, and transformers' from_pretrained picks safetensors
    # first when both are present.
    allow_patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "model.safetensors",
    ]

    print(f"Downloading {REPO_ID} into HF cache at {cache_dir}...", flush=True)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        print(f"CLIP snapshot_download failed: {e}", file=sys.stderr)
        return 1

    final = _snapshot_dir(cache_dir)
    if final is None:
        print("CLIP snapshot_download finished but no snapshots/ dir present", file=sys.stderr)
        return 1
    print(f"CLIP cached at {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
