"""Prepare data for chimera-bench package.

Downloads the CHIMERA-Bench v1.0 dataset from HuggingFace, sets up the
expected directory layout (`processed/`, `raw/structures/`), and runs each
baseline's preprocess.py inside the chimera-bench container to produce the
per-baseline trans_baselines/* caches needed by chimera_trainer.py.

Layout produced under <data_root>/chimera-bench/:
    metadata/final_summary.csv          (HF source)
    structures/*.pdb                    (HF source)
    splits/{epitope_group,antigen_fold,temporal}.json (HF source)
    complex_features/*.pt               (HF source)
    processed/final_summary.csv  -> ../metadata/final_summary.csv (symlink)
    raw/structures               -> ../structures (symlink)
    trans_baselines/mean/{all.jsonl, idx_to_cid.json, all_processed/*.pkl}
    trans_baselines/dymean/...
    trans_baselines/diffab/processed/structures.lmdb, idx_to_cid.json

Run via: mlsbench data chimera-bench
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "mansoorbaloch/chimera-bench"
PATTERNS = [
    "metadata/*",
    "structures/*",
    "splits/*",
    "complex_features/*",
    "README.md",
]
MIN_CACHE_ENTRIES = 100


def run(cmd, **kwargs):
    print(f"[prepare_data] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def hf_download(target: Path):
    """Download the CHIMERA-Bench HF dataset.

    Uses huggingface_hub.snapshot_download (Python API) instead of the CLI
    so we get automatic resume + retries on partial downloads. The CLI's
    --include filter has been observed to silently miss small directories
    (notably metadata/) when the connection is rate-limited mid-fetch.
    """
    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    print(f"[prepare_data] snapshot_download {REPO_ID} -> {target}",
          flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=PATTERNS,
        max_workers=4,
    )

    # Sanity-check that metadata + structures actually landed; the snapshot
    # download has been silent about partial failures in the past.
    metadata_csv = target / "metadata" / "final_summary.csv"
    structures_dir = target / "structures"
    if not metadata_csv.exists():
        raise RuntimeError(
            f"HF download finished but {metadata_csv} is missing. "
            "Re-run `mlsbench data chimera-bench` (resume) or fetch "
            "manually with: huggingface-cli download "
            f"{REPO_ID} --repo-type dataset --local-dir {target} "
            "--include 'metadata/*'")
    if not structures_dir.exists() or not any(structures_dir.iterdir()):
        raise RuntimeError(
            f"HF download finished but {structures_dir} is empty. "
            "Re-run `mlsbench data chimera-bench` to resume.")


def link_compat_dirs(root: Path):
    """Create the `processed/` and `raw/structures` symlinks expected by
    chimera_utils.py and the per-baseline preprocess.py scripts."""
    processed = root / "processed"
    processed.mkdir(exist_ok=True)
    fs_src = root / "metadata" / "final_summary.csv"
    fs_dst = processed / "final_summary.csv"
    if fs_src.exists() and not fs_dst.exists():
        try:
            os.symlink(fs_src, fs_dst)
        except OSError:
            shutil.copy2(fs_src, fs_dst)

    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    structures_dst = raw / "structures"
    structures_src = root / "structures"
    if structures_src.exists() and not structures_dst.exists():
        try:
            os.symlink(structures_src, structures_dst)
        except OSError:
            shutil.copytree(structures_src, structures_dst)


def find_sif():
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    sif = project_root / "vendor" / "images" / "chimera-bench.sif"
    if sif.exists():
        return sif
    return None


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return None


def _remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _validate_indexed_pickle_cache(output_dir: Path, baseline: str) -> bool:
    cache_dir = output_dir / "all_processed"
    idx_to_cid = _load_json(output_dir / "idx_to_cid.json")
    complex_ids = _load_json(output_dir / "complex_ids.json")
    part_files = list(cache_dir.glob("*.pkl")) if cache_dir.exists() else []

    if not part_files:
        print(f"[prepare_data] {baseline}: missing all_processed/*.pkl",
              flush=True)
        return False
    if not isinstance(idx_to_cid, list) or len(idx_to_cid) < MIN_CACHE_ENTRIES:
        print(f"[prepare_data] {baseline}: invalid idx_to_cid.json",
              flush=True)
        return False
    if not isinstance(complex_ids, dict) or len(complex_ids) != len(idx_to_cid):
        print(f"[prepare_data] {baseline}: invalid complex_ids.json",
              flush=True)
        return False
    return True


def _clean_indexed_pickle_cache(output_dir: Path):
    for rel in ("all_processed", "idx_to_cid.json", "complex_ids.json"):
        _remove_path(output_dir / rel)


def _validate_diffab_cache(output_dir: Path) -> bool:
    processed_dir = output_dir / "processed"
    lmdb_path = processed_dir / "structures.lmdb"
    ids = _load_pickle(Path(str(lmdb_path) + "-ids"))
    idx_to_cid = _load_json(output_dir / "idx_to_cid.json")
    complex_ids = _load_json(output_dir / "complex_ids.json")

    if not lmdb_path.exists() or lmdb_path.stat().st_size <= 8192:
        print("[prepare_data] diffab: missing or empty structures.lmdb",
              flush=True)
        return False
    if not isinstance(ids, list) or len(ids) < MIN_CACHE_ENTRIES:
        print("[prepare_data] diffab: invalid structures.lmdb-ids",
              flush=True)
        return False
    if not isinstance(idx_to_cid, list) or idx_to_cid != ids:
        print("[prepare_data] diffab: idx_to_cid.json does not match LMDB ids",
              flush=True)
        return False
    if not isinstance(complex_ids, dict) or len(complex_ids) != len(ids):
        print("[prepare_data] diffab: invalid complex_ids.json",
              flush=True)
        return False
    return True


def _clean_diffab_cache(output_dir: Path):
    processed_dir = output_dir / "processed"
    lmdb_path = processed_dir / "structures.lmdb"
    for path in (
        lmdb_path,
        Path(str(lmdb_path) + "-lock"),
        Path(str(lmdb_path) + "-ids"),
        output_dir / "idx_to_cid.json",
        output_dir / "complex_ids.json",
    ):
        _remove_path(path)


def run_preprocess_in_container(data_root: Path):
    """Run each baseline's preprocess.py inside the chimera-bench SIF."""
    sif = find_sif()
    if sif is None:
        raise FileNotFoundError(
            "chimera-bench.sif not found. Build the image first via "
            "`mlsbench build chimera-bench`, then re-run this script.")

    project_root = Path(__file__).resolve().parents[3]
    pkg_dir = project_root / "vendor" / "external_packages" / "chimera-bench"
    if not pkg_dir.exists():
        raise FileNotFoundError(
            "chimera-bench source not fetched; run "
            "`mlsbench fetch chimera-bench` first.")

    script_dir = Path(__file__).resolve().parent
    chimera_root = data_root / "chimera-bench"
    trans_root = chimera_root / "trans_baselines"

    common_binds = [
        f"{pkg_dir}:/workspace/chimera-bench",
        f"{chimera_root}:/data/chimera-bench-v1.0",
        f"{script_dir}:/workspace/chimera-data-scripts",
    ]

    # Each baseline's preprocess.py inserts its own _SCRIPT_DIR into sys.path
    # (e.g. dymean/preprocess.py:23 inserts baselines/dymean/). Scoping the
    # PYTHONPATH per-baseline avoids name collisions between baselines'
    # `data` packages (mean/data, dymean/data, diffab/data are all distinct).
    for baseline in ("mean", "dymean"):
        output_dir = trans_root / baseline
        if _validate_indexed_pickle_cache(output_dir, baseline):
            print(f"[prepare_data] {baseline} cache already valid; skipping",
                  flush=True)
            continue

        _clean_indexed_pickle_cache(output_dir)
        baseline_dir = f"/workspace/chimera-bench/baselines/{baseline}"
        env_args = [
            "--env", "CHIMERA_DATA_ROOT=/data/chimera-bench-v1.0",
            "--env", f"PYTHONPATH={baseline_dir}",
        ]
        script = f"{baseline_dir}/preprocess.py"
        cmd = ["apptainer", "exec",
               *env_args,
               "--bind", common_binds[0],
               "--bind", common_binds[1],
               "--bind", common_binds[2],
               "--pwd", baseline_dir,
               str(sif), "python", script]
        print(f"[prepare_data] === Preprocessing baseline={baseline} ===", flush=True)
        subprocess.run(cmd, check=True)
        if not _validate_indexed_pickle_cache(output_dir, baseline):
            raise RuntimeError(f"{baseline} preprocess finished but cache is invalid")

    diffab_output = trans_root / "diffab"
    if _validate_diffab_cache(diffab_output):
        print("[prepare_data] diffab cache already valid; skipping", flush=True)
        return

    _clean_diffab_cache(diffab_output)
    baseline_dir = "/workspace/chimera-bench/baselines/diffab"
    env_args = [
        "--env", "CHIMERA_DATA_ROOT=/data/chimera-bench-v1.0",
        "--env", (
            "PYTHONPATH=/workspace/chimera-bench/baselines/diffab:"
            "/workspace/chimera-bench/baselines"
        ),
    ]
    script = "/workspace/chimera-data-scripts/diffab_preprocess_chunked.py"
    cmd = [
        "apptainer", "exec",
        *env_args,
        "--bind", common_binds[0],
        "--bind", common_binds[1],
        "--bind", common_binds[2],
        "--pwd", baseline_dir,
        str(sif), "python", script,
        "--data-root", "/data/chimera-bench-v1.0",
        "--output-dir", "/data/chimera-bench-v1.0/trans_baselines/diffab",
        "--min-success", str(MIN_CACHE_ENTRIES),
    ]
    print("[prepare_data] === Preprocessing baseline=diffab ===", flush=True)
    subprocess.run(cmd, check=True)
    if not _validate_diffab_cache(diffab_output):
        raise RuntimeError("diffab preprocess finished but cache is invalid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Host data root (mlsbench data_root)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    chimera_root = data_root / "chimera-bench"

    needs_download = not (chimera_root / "metadata" / "final_summary.csv").exists() \
        or not (chimera_root / "structures").exists()
    if needs_download:
        hf_download(chimera_root)
    else:
        print(f"[prepare_data] HF dataset already present at {chimera_root}, skipping download")

    link_compat_dirs(chimera_root)
    run_preprocess_in_container(data_root)
    print("[prepare_data] Done.")


if __name__ == "__main__":
    main()
