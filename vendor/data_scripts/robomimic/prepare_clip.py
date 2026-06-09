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


def _is_complete(cache_dir: Path) -> bool:
    """True only when the snapshot has the required config/tokenizer files AND
    model weights — i.e. a finished download, not a half-populated HF cache that
    ``snapshot_download`` created before fetching the blobs."""
    snap = _snapshot_dir(cache_dir)
    if snap is None or not all((snap / f).exists() for f in _REQUIRED_FILES):
        return False
    return (
        (snap / "pytorch_model.bin").exists()
        or any(snap.glob("model*.safetensors"))
        or any(snap.glob("pytorch_model*.bin"))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    cache_dir = Path(args.data_root) / "robomimic" / "hf_cache_clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready_marker = cache_dir / ".ready"

    if _is_complete(cache_dir):
        # Backfill the readiness marker for caches populated before this script
        # wrote one (e.g. the snapshot baked into older images).
        if not ready_marker.exists():
            ready_marker.write_text(REPO_ID + "\n", encoding="utf-8")
        print(f"CLIP already cached at {_snapshot_dir(cache_dir)}, skipping")
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

    if not _is_complete(cache_dir):
        print(
            "CLIP snapshot_download finished but the cache is incomplete "
            "(missing required files or weights); not marking ready",
            file=sys.stderr,
        )
        return 1
    # Only now is the cache guaranteed complete — write a non-empty readiness
    # marker so _data_dep_exists (ready_files) treats a partial/interrupted
    # download as MISSING and retries instead of mounting a half-baked snapshot.
    ready_marker.write_text(REPO_ID + "\n", encoding="utf-8")
    print(f"CLIP cached at {_snapshot_dir(cache_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
