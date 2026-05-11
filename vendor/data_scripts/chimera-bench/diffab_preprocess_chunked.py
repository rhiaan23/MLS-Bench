"""Build CHIMERA DiffAb caches with resumable, bounded-memory writes.

The upstream DiffAb dataset builder materializes all processed structures
before writing LMDB. This helper keeps the same DiffAb structure parser and
Chothia renumbering code, but commits LMDB entries incrementally so reruns can
resume after login-node OOM kills or SLURM timeouts.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import shutil
from pathlib import Path

import lmdb
from tqdm import tqdm

from diffab.datasets.sabdab import preprocess_sabdab_structure
from diffab.tools.renumber.run import renumber as renumber_pdb


MAP_SIZE = 32 * (1024 * 1024 * 1024)
LOG = logging.getLogger("chimera_diffab_preprocess")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def atomic_pickle(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_json(path: Path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def load_pickle(path: Path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return default


def remove_lmdb(lmdb_path: Path):
    for path in (
        lmdb_path,
        Path(str(lmdb_path) + "-lock"),
        Path(str(lmdb_path) + "-ids"),
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def load_chimera_entries(data_root: Path, max_entries: int | None = None):
    summary_path = data_root / "processed" / "final_summary.csv"
    structures_dir = data_root / "raw" / "structures"
    entries = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = row["pdb"]
            pdb_path = structures_dir / f"{pdb}.pdb"
            if not pdb_path.exists():
                continue
            ag_chains = [
                c.strip() for c in row["antigen_chain"].split("|") if c.strip()
            ]
            entries.append({
                "complex_id": row["complex_id"],
                "pdb": pdb,
                "pdb_path": str(pdb_path),
                "H_chain": row["Hchain"] or None,
                "L_chain": row["Lchain"] or None,
                "ag_chains": ag_chains,
            })
            if max_entries is not None and len(entries) >= max_entries:
                break
    return entries


def save_renumber_log(path: Path, renumber_log):
    atomic_json(path, renumber_log)


def renumber_all_resumable(
    entries,
    chothia_dir: Path,
    renumber_log_path: Path,
    log_interval: int,
):
    chothia_dir.mkdir(parents=True, exist_ok=True)
    renumber_log = load_json(renumber_log_path, {})

    pdb_to_entry = {}
    for entry in entries:
        pdb_to_entry.setdefault(entry["pdb"], entry)

    changed = 0
    for pdb, entry in tqdm(pdb_to_entry.items(), desc="Renumber"):
        out_path = chothia_dir / f"{pdb}.pdb"
        if out_path.exists():
            if renumber_log.get(pdb, {}).get("status") != "cached":
                renumber_log[pdb] = {"status": "cached"}
                changed += 1
            continue

        try:
            heavy_chains, light_chains, other_chains = renumber_pdb(
                entry["pdb_path"], str(out_path), return_other_chains=True)
            renumber_log[pdb] = {
                "status": "ok",
                "heavy_chains": heavy_chains,
                "light_chains": light_chains,
                "other_chains": other_chains,
            }
        except Exception as exc:
            LOG.warning("Renumber failed for %s: %s", pdb, exc)
            renumber_log[pdb] = {"status": "failed", "error": str(exc)}
        changed += 1

        if changed >= log_interval:
            save_renumber_log(renumber_log_path, renumber_log)
            changed = 0

    save_renumber_log(renumber_log_path, renumber_log)
    return renumber_log


def verify_existing_ids(lmdb_path: Path, ids):
    if not lmdb_path.exists() or not ids:
        return []
    try:
        env = lmdb.open(
            str(lmdb_path), subdir=False, readonly=True, lock=False,
            readahead=False, max_readers=1)
        with env.begin(buffers=True) as txn:
            for cid in (ids[0], ids[-1]):
                if txn.get(cid.encode("utf-8")) is None:
                    env.close()
                    return []
        env.close()
    except lmdb.Error:
        return []
    return ids


def make_task(entry, chothia_dir: Path):
    entry_id = entry["complex_id"]
    return {
        "id": entry_id,
        "entry": {
            "id": entry_id,
            "pdbcode": entry["pdb"],
            "H_chain": entry["H_chain"],
            "L_chain": entry["L_chain"],
            "ag_chains": entry["ag_chains"],
        },
        "pdb_path": str(chothia_dir / f"{entry['pdb']}.pdb"),
    }


def commit_ids(ids_path: Path, succeeded_ids):
    atomic_pickle(ids_path, succeeded_ids)


def build_lmdb_chunked(
    entries,
    chothia_dir: Path,
    lmdb_path: Path,
    commit_every: int,
):
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path = Path(str(lmdb_path) + "-ids")
    existing_ids = verify_existing_ids(lmdb_path, load_pickle(ids_path, []))
    if lmdb_path.exists() and not existing_ids:
        LOG.info("Removing empty or inconsistent LMDB at %s", lmdb_path)
        remove_lmdb(lmdb_path)

    succeeded_ids = list(existing_ids)
    seen = set(succeeded_ids)
    env = lmdb.open(
        str(lmdb_path), map_size=MAP_SIZE, create=True, subdir=False,
        readonly=False, lock=True, readahead=False)

    txn = env.begin(write=True)
    pending = 0
    try:
        for entry in tqdm(entries, desc="Preprocess"):
            cid = entry["complex_id"]
            if cid in seen:
                continue
            pdb_path = chothia_dir / f"{entry['pdb']}.pdb"
            if not pdb_path.exists():
                LOG.warning("Chothia PDB missing for %s, skipping", cid)
                continue

            task = make_task(entry, chothia_dir)
            try:
                data = preprocess_sabdab_structure(task)
            except Exception as exc:
                LOG.warning("Preprocess failed for %s: %s", cid, exc)
                continue
            if data is None:
                LOG.warning("Preprocess returned None for %s", cid)
                continue
            if data.get("heavy") is None and data.get("light") is None:
                LOG.warning("No valid antibody chains for %s", cid)
                continue

            txn.put(cid.encode("utf-8"),
                    pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            succeeded_ids.append(cid)
            seen.add(cid)
            pending += 1

            if pending >= commit_every:
                txn.commit()
                env.sync()
                commit_ids(ids_path, succeeded_ids)
                txn = env.begin(write=True)
                pending = 0

        txn.commit()
        env.sync()
        commit_ids(ids_path, succeeded_ids)
    except BaseException:
        if pending:
            txn.commit()
            env.sync()
            commit_ids(ids_path, succeeded_ids)
        else:
            txn.abort()
        raise
    finally:
        env.close()

    return succeeded_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit-every", type=int, default=25)
    parser.add_argument("--renumber-log-interval", type=int, default=25)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--min-success", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chothia_dir = output_dir / "chothia"
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    entries = load_chimera_entries(data_root, args.max_entries)
    LOG.info("Found %d CHIMERA complexes", len(entries))

    renumber_log = renumber_all_resumable(
        entries,
        chothia_dir,
        output_dir / "renumber_log.json",
        args.renumber_log_interval,
    )
    failed = sum(1 for item in renumber_log.values()
                 if item.get("status") == "failed")
    LOG.info("Renumbered/cacheable PDBs: %d total, %d failed",
             len(renumber_log), failed)

    lmdb_path = processed_dir / "structures.lmdb"
    succeeded_ids = build_lmdb_chunked(
        entries, chothia_dir, lmdb_path, args.commit_every)
    if len(succeeded_ids) < args.min_success:
        raise RuntimeError(
            f"DiffAb preprocessing produced only {len(succeeded_ids)} "
            f"structures; expected at least {args.min_success}")

    atomic_json(output_dir / "idx_to_cid.json", succeeded_ids)
    atomic_json(output_dir / "complex_ids.json", {
        cid: idx for idx, cid in enumerate(succeeded_ids)
    })
    LOG.info("DiffAb preprocessing complete: %d structures", len(succeeded_ids))


if __name__ == "__main__":
    main()
