"""Prepare data for HypSeek (ai4sci-vs-contrastive-scoring).

Fetches every data dependency the task needs:
  (1) Pretrained Uni-Mol weights + ESM2 cache (no large data).
  (2) HypSeek author-released training/validation bundles from figshare
      (lmdbs, label sequences, cluster files).
  (3) Evaluation test datasets (DUD-E, DEKOIS 2.0x, LIT-PCBA) from the
      same upstream sources the original setup session used.
  (4) JSON manifests + UniProt FASTA cache derived from the test
      datasets via rcsb.org -> rest.uniprot.org lookups.

Output:
    <data_root>/HypSeek/pretrain/{mol_pre_no_h_220816.pt, pocket_pre_220816.pt}
    <data_root>/HypSeek/hf_cache/                                  (ESM2 cache)
    <data_root>/HypSeek/vs_data/{train_lig_all_blend.lmdb, ...}     (figshare)
    <data_root>/HypSeek/test_datasets/{DUD-E, DEKOIS_2.0x, lit_pcba, *.json}
    <data_root>/HypSeek/uniport_fasta/<accession>.fasta             (UniProt)

Source pinning (recovered from the original setup session, see
``docs/ai4sci_data_audit.md`` plus the historical install_cmds in
vendor/pkg_configs/HypSeek/config.json):
  * DUD-E raw: GDrive id 1ftr3UjpjhN6ISLsFJb7-RSPOxAWbmpV2
  * LIT-PCBA raw: GDrive id 1ApPfAx8-terj8nxrx63egiGuY8p6j_cu
  * DEKOIS_2.0x: https://ndownloader.figshare.com/files/50994021

Run via:
    mlsbench data HypSeek
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


PRETRAINED_WEIGHTS = {
    "mol_pre_no_h_220816.pt": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt",
    "pocket_pre_220816.pt": "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt",
}

VS_DATA_FILES = {
    "train_lig_all_blend.lmdb": "https://ndownloader.figshare.com/files/50992728",
    "train_prot_all_blend.lmdb": "https://ndownloader.figshare.com/files/50992194",
    "valid_label_seq.json": "https://ndownloader.figshare.com/files/50992173",
    "valid_lig.lmdb": "https://ndownloader.figshare.com/files/50992176",
    "valid_prot.lmdb": "https://ndownloader.figshare.com/files/50992182",
    "uniport80.clstr": "https://ndownloader.figshare.com/files/54239555",
    "uniport40.clstr": "https://ndownloader.figshare.com/files/54239558",
    "pocket_name2idx_train_blend.json": "https://ndownloader.figshare.com/files/50992185",
    "mol_smi2idx_train_blend.json": "https://ndownloader.figshare.com/files/50992191",
}

TRAIN_LABEL = {
    "url": "https://ndownloader.figshare.com/files/50992179",
    "archive": "train_label.zip",
}

# Test-dataset bundles. Each is (gdrive_id-or-url, archive_name, layout_dest)
# where layout_dest is the final subdir under test_datasets/.
DUDE_GDRIVE_ID = "1ftr3UjpjhN6ISLsFJb7-RSPOxAWbmpV2"
PCBA_GDRIVE_ID = "1ApPfAx8-terj8nxrx63egiGuY8p6j_cu"
DEKOIS_URL = "https://ndownloader.figshare.com/files/50994021"

TEST_DATASET_DIRS = ["DUD-E", "DEKOIS_2.0x", "lit_pcba"]
TEST_DATASET_MANIFESTS = ["dude.json", "dekois.json", "PCBA.json"]

_ESM2_CACHE_INLINE = (
    "import os; "
    "os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/data/hf_cache'); "
    "from transformers import AutoTokenizer, AutoModelForMaskedLM; "
    "AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D'); "
    "AutoModelForMaskedLM.from_pretrained('facebook/esm2_t12_35M_UR50D'); "
    "print('ESM2 cached')"
)


def download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {url} -> {dest}", flush=True)
            urllib.request.urlretrieve(url, str(dest))
            return
        except (urllib.error.URLError, OSError) as e:
            print(f"  attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url}")


def gdrive_download(file_id: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest} already downloaded")
        return
    try:
        import gdown  # type: ignore

        gdown.download(id=file_id, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 10_000:
            return
    except ImportError:
        pass
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            page = urllib.request.urlopen(
                "https://drive.google.com/uc?export=download&id=" + file_id
            ).read().decode(errors="ignore")
            if "Quota exceeded" in page or "Cannot retrieve" in page:
                wait = 60 * attempt
                print(f"  Quota issue, retry in {wait}s ({attempt}/{retries})")
                time.sleep(wait)
                continue
            m = re.search(r'uuid.*?value="([^"]+)', page)
            if not m:
                raise RuntimeError("No uuid in confirm page")
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t&uuid={m.group(1)}"
            )
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  Error: {e}, retry in {60 * attempt}s")
            time.sleep(60 * attempt)
    raise RuntimeError(f"Failed to fetch GDrive id={file_id}: {last_err}")


def _run_cache_cmd(label: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    print(f"  Caching ESM2 model via {label}...", flush=True)
    res = subprocess.run(cmd, env=env, check=False)
    if res.returncode == 0:
        return True
    print(f"  {label} ESM2 cache failed with exit {res.returncode}; trying next backend.",
          file=sys.stderr)
    return False


def cache_esm2(hf_cache_dir: Path) -> None:
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]

    sif = project_root / "vendor" / "images" / "HypSeek.sif"
    if sif.exists() and shutil.which("apptainer"):
        cmd = [
            "apptainer", "exec",
            "--bind", f"{hf_cache_dir}:/data/hf_cache",
            "--env", "HF_HOME=/data/hf_cache",
            str(sif),
            "python", "-c", _ESM2_CACHE_INLINE,
        ]
        if _run_cache_cmd("Apptainer", cmd):
            return

    docker_tag = "mlsbench/hypseek:latest"
    if shutil.which("docker"):
        inspect = subprocess.run(
            ["docker", "image", "inspect", docker_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode == 0:
            cmd = [
                "docker", "run", "--rm", "--entrypoint", "",
                "-v", f"{hf_cache_dir}:/data/hf_cache",
                "-e", "HF_HOME=/data/hf_cache",
                docker_tag,
                "python", "-c", _ESM2_CACHE_INLINE,
            ]
            if _run_cache_cmd("Docker", cmd):
                return

    env = os.environ.copy()
    env["HF_HOME"] = str(hf_cache_dir)
    if _run_cache_cmd("local Python", [sys.executable, "-c", _ESM2_CACHE_INLINE], env=env):
        return

    raise RuntimeError(
        "Could not cache facebook/esm2_t12_35M_UR50D via Apptainer, Docker, "
        "or local Python. Build the HypSeek runtime first or install "
        "transformers in the local runtime."
    )


def prepare_pretrain(root: Path) -> None:
    pretrain_dir = root / "pretrain"
    for name, url in PRETRAINED_WEIGHTS.items():
        dest = pretrain_dir / name
        if dest.exists():
            print(f"  [SKIP] pretrain/{name}")
            continue
        download(url, dest)


def prepare_hf_cache(root: Path) -> None:
    hf_cache_dir = root / "hf_cache"
    hub_dir = hf_cache_dir / "hub"
    if hub_dir.exists() and any(hub_dir.glob("models--facebook--esm2*")):
        print("  [SKIP] ESM2 model already cached")
        return
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_esm2(hf_cache_dir)


def prepare_vs_data(root: Path) -> None:
    vs_dir = root / "vs_data"
    vs_dir.mkdir(parents=True, exist_ok=True)
    for name, url in VS_DATA_FILES.items():
        dest = vs_dir / name
        if dest.exists():
            print(f"  [SKIP] vs_data/{name}")
            continue
        download(url, dest)

    train_label_dir = vs_dir / "train_label_seq"
    if not (train_label_dir.exists() and any(train_label_dir.iterdir())):
        zip_path = Path(f"/tmp/{TRAIN_LABEL['archive']}")
        download(TRAIN_LABEL["url"], zip_path)
        subprocess.run(["unzip", "-oq", str(zip_path), "-d", str(vs_dir)], check=True)
        zip_path.unlink()
        print("  Extracted train_label_seq")
    for name in ("fep_repeat_ligands_can.json", "fep_assays.json"):
        dest = vs_dir / name
        if not dest.exists():
            dest.write_text("[]")
    (vs_dir / "cache").mkdir(parents=True, exist_ok=True)


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(target_dir))


def prepare_test_datasets(root: Path) -> bool:
    """Download DUD-E + LIT-PCBA + DEKOIS_2.0x raw bundles into test_datasets/.

    Returns True iff all three subdirs end up populated. The DUD-E
    GDrive share has been reported as deprecated upstream
    (the file id was once reachable, then revoked). If it returns
    "Cannot retrieve" the script logs a clear warning and continues —
    DEKOIS and LIT-PCBA are the larger parts of the test bundle and
    should still succeed.
    """
    td = root / "test_datasets"
    td.mkdir(parents=True, exist_ok=True)

    # DUD-E (GDrive zip; layout: data/protein/DUD-E/raw/all/<target>/)
    # Stale DUD-E zip from a previous quota-exceeded download (2 KB) blocks
    # subsequent attempts; clear it before retrying.
    stale_zip = td / "_dude.zip"
    if stale_zip.exists() and stale_zip.stat().st_size < 1_000_000:
        stale_zip.unlink()
    if not (td / "DUD-E").exists() or not any((td / "DUD-E").iterdir()):
        try:
            zip_path = td / "_dude.zip"
            gdrive_download(DUDE_GDRIVE_ID, zip_path)
            tmp = td / "_dude_tmp"
            _safe_extract_zip(zip_path, tmp)
            zip_path.unlink()
            inner = tmp / "data" / "protein" / "DUD-E" / "raw" / "all"
            if inner.exists():
                if (td / "DUD-E").exists():
                    import shutil
                    shutil.rmtree(td / "DUD-E")
                inner.rename(td / "DUD-E")
            else:
                # Fallback: take the first dir under tmp as DUD-E root.
                roots = [p for p in tmp.iterdir() if p.is_dir()]
                if roots:
                    roots[0].rename(td / "DUD-E")
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: DUD-E GDrive download failed ({e}). Falling "
                  "back to the official upstream source dude.docking.org.",
                  flush=True)
            # Official upstream — slower (~2.8GB tgz) but no quota.
            try:
                import shutil
                import tarfile
                tgz_path = td / "_dude_all.tar.gz"
                if not tgz_path.exists() or tgz_path.stat().st_size < 100_000_000:
                    download(
                        "https://dude.docking.org/db/subsets/all/all.tar.gz",
                        tgz_path,
                    )
                tmp = td / "_dude_tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                print(f"  Extracting {tgz_path} (this is large) ...", flush=True)
                with tarfile.open(str(tgz_path), "r:gz") as tar:
                    tar.extractall(str(tmp))
                tgz_path.unlink()
                inner = tmp / "all"
                if inner.exists() and any(inner.iterdir()):
                    if (td / "DUD-E").exists():
                        shutil.rmtree(td / "DUD-E")
                    inner.rename(td / "DUD-E")
                else:
                    # Some archives nest under a single root directory.
                    roots = [p for p in tmp.iterdir() if p.is_dir()]
                    if roots:
                        roots[0].rename(td / "DUD-E")
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception as inner_e:  # noqa: BLE001
                print(
                    f"  ERROR: DUD-E upstream fallback also failed: {inner_e}",
                    flush=True,
                )

    # LIT-PCBA (GDrive zip)
    if not (td / "lit_pcba").exists() or not any((td / "lit_pcba").iterdir()):
        try:
            zip_path = td / "_pcba.zip"
            gdrive_download(PCBA_GDRIVE_ID, zip_path)
            _safe_extract_zip(zip_path, td)
            zip_path.unlink()
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: LIT-PCBA download failed: {e}", flush=True)

    # DEKOIS 2.0x (figshare zip; ~17 GB)
    if not (td / "DEKOIS_2.0x").exists() or not any((td / "DEKOIS_2.0x").iterdir()):
        try:
            zip_path = td / "_dekois.zip"
            download(DEKOIS_URL, zip_path)
            _safe_extract_zip(zip_path, td)
            zip_path.unlink()
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: DEKOIS_2.0x download failed: {e}", flush=True)

    return all((td / d).exists() and any((td / d).iterdir()) for d in TEST_DATASET_DIRS)


def _read_pdb_header_compnd(pdb_path: Path) -> str | None:
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HEADER") or line.startswith("COMPND"):
                    parts = line.split()
                    for tok in parts:
                        if len(tok) == 4 and tok.isalnum():
                            return tok.upper()
    except Exception:
        return None
    return None


def _scan_target_pdb_id(target_dir: Path) -> str | None:
    """Best-effort PDB id extraction for a target directory."""
    for name in ("receptor.pdb", "protein.pdb"):
        cand = target_dir / name
        if cand.exists():
            pid = _read_pdb_header_compnd(cand)
            if pid:
                return pid
    # Look for *_protein.mol2 / *_protein.pdb with the PDB id baked in.
    for cand in list(target_dir.glob("*_protein.pdb")) + list(target_dir.glob("*_protein.mol2")):
        stem = cand.stem.split("_protein")[0]
        if len(stem) == 4 and stem.isalnum():
            return stem.upper()
    # Sometimes the dir name is itself a PDB id.
    if len(target_dir.name) == 4 and target_dir.name.isalnum():
        return target_dir.name.upper()
    return None


def _pdb_to_uniprot(pdb: str) -> str | None:
    url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdb}/1"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        # The RCSB response carries rcsb_uniprot_container_identifiers.uniprot_id.
        if isinstance(data, list) and data:
            data = data[0]
        ids = (
            data.get("rcsb_uniprot_container_identifiers", {})
            .get("uniprot_ids")
            or data.get("rcsb_id")
        )
        if isinstance(ids, list) and ids:
            return ids[0]
        if isinstance(ids, str) and ids:
            return ids
    except Exception:
        return None
    return None


def _fetch_uniprot_fasta(acc: str, fasta_dir: Path) -> bool:
    dest = fasta_dir / f"{acc}.fasta"
    if dest.exists() and dest.stat().st_size > 50:
        return True
    try:
        download(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", dest)
        return dest.exists() and dest.stat().st_size > 50
    except Exception as e:  # noqa: BLE001
        print(f"    {acc}: {e}", flush=True)
        return False


def _validate_manifest_triples(
    manifest_path: Path,
    triples: object,
    fasta_dir: Path,
) -> None:
    if not isinstance(triples, list) or not triples:
        raise RuntimeError(f"{manifest_path.name} is empty or not a list")
    bad = []
    missing_fasta = []
    for idx, triple in enumerate(triples):
        if (
            not isinstance(triple, list)
            or len(triple) < 3
            or not triple[0]
            or not triple[1]
            or not triple[2]
        ):
            bad.append(idx)
            continue
        acc = str(triple[0])
        fasta = fasta_dir / f"{acc}.fasta"
        if not (fasta.exists() and fasta.stat().st_size > 50):
            missing_fasta.append(acc)
    if bad:
        raise RuntimeError(f"{manifest_path.name} has invalid triples at indices {bad[:10]}")
    if missing_fasta:
        raise RuntimeError(
            f"{manifest_path.name} missing FASTA files for "
            + ", ".join(sorted(set(missing_fasta))[:10])
        )


def build_manifests(root: Path) -> None:
    """Generate dude.json / dekois.json / PCBA.json + uniport_fasta/.

    Each manifest is a list of [uniprot, pdb, target] triples; the FASTA
    cache is one file per unique UniProt accession. This mirrors the
    bootstrap loop the original setup session ran in-image.
    """
    td = root / "test_datasets"
    fasta_dir = root / "uniport_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "dude.json": td / "DUD-E",
        "dekois.json": td / "DEKOIS_2.0x",
        "PCBA.json": td / "lit_pcba",
    }

    # Reuse existing manifests when present + their accessions still resolve.
    for fname, ds_dir in manifests.items():
        manifest_path = td / fname
        if manifest_path.exists() and manifest_path.stat().st_size > 2:
            try:
                triples = json.loads(manifest_path.read_text())
                bad_fetch = []
                for triple in triples:
                    if isinstance(triple, list) and len(triple) >= 3 and triple[0]:
                        if not _fetch_uniprot_fasta(str(triple[0]), fasta_dir):
                            bad_fetch.append(str(triple[0]))
                if bad_fetch:
                    raise RuntimeError(
                        f"{fname}: failed to fetch FASTA for "
                        + ", ".join(sorted(set(bad_fetch))[:10])
                    )
                _validate_manifest_triples(manifest_path, triples, fasta_dir)
                continue
            except Exception:
                pass  # rebuild on parse failure
        if not ds_dir.exists():
            continue
        triples = []
        failures = []
        targets = sorted(p for p in ds_dir.iterdir() if p.is_dir())
        print(f"  Bootstrapping {fname} ({len(targets)} targets)...")
        for tdir in targets:
            pdb = _scan_target_pdb_id(tdir)
            uid = _pdb_to_uniprot(pdb) if pdb else None
            triples.append([uid or "", pdb or "", tdir.name])
            if not pdb or not uid:
                failures.append(f"{tdir.name}: pdb={pdb or '?'} uniprot={uid or '?'}")
            elif not _fetch_uniprot_fasta(uid, fasta_dir):
                failures.append(f"{tdir.name}: FASTA fetch failed for {uid}")
        manifest_path.write_text(json.dumps(triples, indent=2))
        print(f"  wrote {manifest_path}")
        if failures:
            # PDB->UniProt resolution depends on rest.uniprot.org being
            # reachable; only the test labels (dude/dekois/lit_pcba) need
            # the resulting fasta, while the `train` label is independent.
            # Warn instead of aborting so partial data still unblocks the
            # training smoke; verify() catches missing test artifacts when
            # they're actually required.
            print(
                f"  WARN: {fname} incomplete; first failures: "
                + "; ".join(failures[:10]),
                flush=True,
            )
        else:
            _validate_manifest_triples(manifest_path, triples, fasta_dir)


def verify(root: Path, test_ok: bool) -> None:
    pretrain_dir = root / "pretrain"
    vs_dir = root / "vs_data"
    td = root / "test_datasets"
    checks = [
        pretrain_dir / "mol_pre_no_h_220816.pt",
        pretrain_dir / "pocket_pre_220816.pt",
        vs_dir / "train_lig_all_blend.lmdb",
        vs_dir / "train_prot_all_blend.lmdb",
        vs_dir / "valid_label_seq.json",
        vs_dir / "valid_lig.lmdb",
        vs_dir / "valid_prot.lmdb",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"ERROR: Missing: {missing}", file=sys.stderr)
        sys.exit(1)
    # Task scripts (dude.sh / dekois.sh / lit_pcba.sh) all read from
    # /data/test_datasets. Any missing subdir is a fatal prep failure.
    missing_test = [d for d in TEST_DATASET_DIRS
                    if not (td / d).exists() or not any((td / d).iterdir())]
    if missing_test:
        print(
            "ERROR: test_datasets incomplete: " + ", ".join(missing_test)
            + ".\n  See warnings above for which upstream source failed.\n"
            "  All three of DUD-E, DEKOIS_2.0x, lit_pcba are required by the "
            "task evaluation scripts.",
            file=sys.stderr,
        )
        sys.exit(1)
    manifest_errors = []
    fasta_dir = root / "uniport_fasta"
    for name in TEST_DATASET_MANIFESTS:
        path = td / name
        if not path.exists():
            manifest_errors.append(f"{name} missing")
            continue
        try:
            triples = json.loads(path.read_text())
            _validate_manifest_triples(path, triples, fasta_dir)
        except Exception as exc:  # noqa: BLE001
            manifest_errors.append(f"{name}: {exc}")
    if manifest_errors:
        # Manifest+FASTA are only consumed by the test labels
        # (dude/dekois/lit_pcba). Demote to a warning so the train smoke
        # path stays unblocked when rest.uniprot.org is flaky; tests will
        # surface the missing pieces when their scripts try to read them.
        print(
            "WARN: HypSeek manifests/FASTA incomplete (test labels only):\n  - "
            + "\n  - ".join(manifest_errors),
            flush=True,
        )
    print("All HypSeek data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "HypSeek"
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== Preparing HypSeek data at {root} ===")

    prepare_pretrain(root)
    prepare_hf_cache(root)
    prepare_vs_data(root)
    test_ok = prepare_test_datasets(root)
    if test_ok:
        build_manifests(root)
    verify(root, test_ok)


if __name__ == "__main__":
    main()
