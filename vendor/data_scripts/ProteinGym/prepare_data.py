#!/usr/bin/env python3
"""Prepare data for ProteinGym (ai4bio-mutation-effect-prediction).

Downloads the v1.1 substitution benchmark from the official Marks-Lab
distribution server, extracts the DMS assays and cv_folds, and (best
effort) produces ESM2-650M mean-pooled embeddings for the three assays
the task uses (BLAT_ECOLX_Firnberg_2014, ESTA_BACSU_Nutschel_2020,
RASH_HUMAN_Bandaru_2017). Embedding generation uses the already-built
ProteinGym runtime when available: Apptainer image, Docker image, or
the current local/Conda Python.

Output:
    <data_root>/proteingym/DMS_substitutions.csv
    <data_root>/proteingym/DMS_assays/DMS_ProteinGym_substitutions/*.csv  (217)
    <data_root>/proteingym/cv_folds/cv_folds_singles_substitutions/*.csv  (217)
    <data_root>/proteingym_embeddings/{<assay_id>}.pt                     (3)

Sources:
    https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_ProteinGym_substitutions.zip
    https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/cv_folds_singles_substitutions.zip
    https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_substitutions.csv
    Embedding model: facebook/esm2_t33_650M_UR50D via fair-esm.

Run via:
    mlsbench data ProteinGym
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REFERENCE_CSV_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/"
    "DMS_substitutions.csv"
)
DMS_ZIP_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/"
    "DMS_ProteinGym_substitutions.zip"
)
CV_FOLDS_ZIP_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/"
    "cv_folds_singles_substitutions.zip"
)

# Three assays consumed by tasks/ai4bio-mutation-effect-prediction.
EMBED_ASSAYS = [
    "BLAT_ECOLX_Firnberg_2014",
    "ESTA_BACSU_Nutschel_2020",
    "RASH_HUMAN_Bandaru_2017",
]

# Hard cap mirrors the existing embedding script.
MAX_TOKENS = 1022
EMBED_DIM = 1280


def docker_image_tag(name: str) -> str:
    return f"mlsbench/{name.lower()}:latest"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000:
        print(f"  [SKIP] {dest} already downloaded")
        return
    print(f"  Downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


def extract(zip_path: Path, target: Path, expected_subdir: str | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} -> {target}", flush=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(target))
    # If the zip wraps content in an extra directory, flatten it.
    if expected_subdir is not None and not (target / expected_subdir).exists():
        roots = [p for p in target.iterdir() if p.is_dir() and p.name != expected_subdir]
        if len(roots) == 1:
            inner = roots[0] / expected_subdir
            if inner.exists():
                shutil.move(str(inner), str(target / expected_subdir))


def prepare_dms(data_root: Path) -> Path:
    pg = data_root / "proteingym"
    pg.mkdir(parents=True, exist_ok=True)

    csv_path = pg / "DMS_substitutions.csv"
    download(REFERENCE_CSV_URL, csv_path)

    assay_dir = pg / "DMS_assays" / "DMS_ProteinGym_substitutions"
    if not (assay_dir.exists() and len(list(assay_dir.glob("*.csv"))) >= 200):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "DMS_ProteinGym_substitutions.zip"
            download(DMS_ZIP_URL, zip_path)
            extract(zip_path, pg / "DMS_assays", expected_subdir="DMS_ProteinGym_substitutions")
    print(f"  DMS_assays: {len(list(assay_dir.glob('*.csv')))} files")

    cv_dir = pg / "cv_folds" / "cv_folds_singles_substitutions"
    if not (cv_dir.exists() and len(list(cv_dir.glob("*.csv"))) >= 200):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "cv_folds_singles_substitutions.zip"
            download(CV_FOLDS_ZIP_URL, zip_path)
            extract(zip_path, pg / "cv_folds", expected_subdir="cv_folds_singles_substitutions")
    print(f"  cv_folds: {len(list(cv_dir.glob('*.csv')))} files")
    return pg


def embedding_inline_py() -> str:
    return r"""
import os, csv, torch, esm
import pandas as pd

DATA_ROOT = os.environ.get('PROTEINGYM_DATA', '/data/proteingym')
EMBED_ROOT = os.environ.get('PROTEINGYM_EMBEDDINGS', '/data/esm2_embeddings')

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

assays = ASSAYS_PLACEHOLDER

with open(os.path.join(DATA_ROOT, 'DMS_substitutions.csv')) as f:
    ref = {row['DMS_id']: row for row in csv.DictReader(f)}

for assay in assays:
    out = os.path.join(EMBED_ROOT, f'{assay}.pt')
    if os.path.exists(out):
        print(f'SKIP {assay}'); continue
    row = ref[assay]
    target_seq = row['target_seq']
    dms_dir = os.path.join(DATA_ROOT, 'DMS_assays')
    csv_files = []
    for r, _, files in os.walk(dms_dir):
        for fn in files:
            if fn.startswith(assay) and fn.endswith('.csv'):
                csv_files.append(os.path.join(r, fn))
    if not csv_files:
        raise RuntimeError(f'No DMS CSV for {assay}')
    df = pd.read_csv(csv_files[0])
    df = df[~df['mutant'].str.contains(':')].reset_index(drop=True)
    seqs = [target_seq]
    seq_to_idx = {target_seq: 0}
    for _, r in df.iterrows():
        ms = r['mutated_sequence']
        if ms not in seq_to_idx:
            seq_to_idx[ms] = len(seqs)
            seqs.append(ms)
    embed_dim = 1280
    all_emb = torch.zeros(len(seqs), embed_dim)
    bs = 16
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i+bs]
            data = [(f'seq_{j}', s) for j, s in enumerate(batch)]
            _, _, tok = batch_converter(data)
            tok = tok.to(device)
            if tok.size(1) > 1024:
                tok = tok[:, :1024]
            out_feat = model(tok, repr_layers=[33], return_contacts=False)['representations'][33]
            for j in range(len(batch)):
                L = min(len(batch[j]), 1022)
                all_emb[i+j] = out_feat[j, 1:L+1].mean(dim=0).cpu()
    mut_emb = torch.zeros(len(df), embed_dim)
    scores = torch.zeros(len(df))
    mut_ids = []
    for idx, (_, r) in enumerate(df.iterrows()):
        mut_emb[idx] = all_emb[seq_to_idx[r['mutated_sequence']]]
        scores[idx] = r['DMS_score']
        mut_ids.append(r['mutant'])
    torch.save({
        'embeddings': mut_emb,
        'scores': scores,
        'mutant_ids': mut_ids,
        'wt_embedding': all_emb[0],
        'embed_dim': embed_dim,
    }, out)
    print(f'  saved {out} {tuple(mut_emb.shape)}')
""".replace("ASSAYS_PLACEHOLDER", repr(EMBED_ASSAYS))


def run_embedding_cmd(label: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    print(f"  Running ESM2 embedding generation via {label}...", flush=True)
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"  {label} embedding generation exited {res.returncode}; trying next backend.",
              file=sys.stderr)
        return False
    return True


def embeddings_via_runtime(data_root: Path) -> bool:
    """Generate embeddings via Apptainer, Docker, or local/Conda Python."""
    project_root = Path(__file__).resolve().parents[3]
    pg = data_root / "proteingym"
    emb = data_root / "proteingym_embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    if all((emb / f"{a}.pt").exists() for a in EMBED_ASSAYS):
        print(f"  [SKIP] All {len(EMBED_ASSAYS)} embeddings already present")
        return True

    inline = embedding_inline_py()

    sif = project_root / "vendor" / "images" / "ProteinGym.sif"
    if sif.exists() and shutil.which("apptainer"):
        cmd = [
            "apptainer", "exec", "--nv",
            "--bind", f"{pg}:/data/proteingym",
            "--bind", f"{emb}:/data/esm2_embeddings",
            str(sif),
            "python", "-c", inline,
        ]
        if run_embedding_cmd("Apptainer", cmd):
            return True

    docker_tag = docker_image_tag("ProteinGym")
    if shutil.which("docker"):
        inspect = subprocess.run(
            ["docker", "image", "inspect", docker_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode == 0:
            cmd = ["docker", "run", "--rm", "--entrypoint", ""]
            if shutil.which("nvidia-smi"):
                cmd.extend(["--gpus", "all"])
            cmd.extend([
                "-v", f"{pg}:/data/proteingym",
                "-v", f"{emb}:/data/esm2_embeddings",
                docker_tag,
                "python", "-c", inline,
            ])
            if run_embedding_cmd("Docker", cmd):
                return True

    env = os.environ.copy()
    env["PROTEINGYM_DATA"] = str(pg)
    env["PROTEINGYM_EMBEDDINGS"] = str(emb)
    return run_embedding_cmd("local Python", [sys.executable, "-c", inline], env=env)


def verify(data_root: Path, require_embeddings: bool) -> None:
    pg = data_root / "proteingym"
    missing = []
    for rel in (
        "DMS_substitutions.csv",
        "DMS_assays/DMS_ProteinGym_substitutions",
        "cv_folds/cv_folds_singles_substitutions",
    ):
        path = pg / rel
        if not path.exists():
            missing.append(str(path))
    if require_embeddings:
        for a in EMBED_ASSAYS:
            if not (data_root / "proteingym_embeddings" / f"{a}.pt").exists():
                missing.append(f"proteingym_embeddings/{a}.pt")
    if missing:
        print("ERROR: missing artifacts:", missing, file=sys.stderr)
        sys.exit(1)
    print("All ProteinGym data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Don't try to generate ESM2 embeddings (data download only).",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    print(f"=== Preparing ProteinGym data at {data_root} ===")
    prepare_dms(data_root)

    embedded = False
    if not args.skip_embeddings:
        embedded = embeddings_via_runtime(data_root)

    # Embeddings are required at runtime. Always verify them unless the
    # caller explicitly said --skip-embeddings.
    require = not args.skip_embeddings
    if require and not embedded:
        # Don't claim success when embeddings are missing — runtime will fail.
        emb_dir = data_root / "proteingym_embeddings"
        already_have = all(
            (emb_dir / f"{a}.pt").exists() for a in EMBED_ASSAYS
        )
        if not already_have:
            print(
                "\nERROR: ProteinGym ESM2 embeddings missing.\n"
                "  Run `mlsbench build ProteinGym` for your configured runtime, then re-run\n"
                f"  `python vendor/data_scripts/ProteinGym/prepare_data.py --data-root {args.data_root}`\n"
                "  (or `mlsbench data ProteinGym`).",
                file=sys.stderr,
            )
            sys.exit(2)
    verify(data_root, require_embeddings=require)


if __name__ == "__main__":
    main()
