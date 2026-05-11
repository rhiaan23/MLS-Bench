#!/usr/bin/env python3
"""Prepare data for ClimSim (ai4sci-climate-emulation).

Downloads three months of E3SM-MMF low-resolution real-geography
simulation data from the LEAP HuggingFace dataset, then converts the
raw .mli/.mlo NetCDFs into the .npy arrays the task expects.

Output:
    <data_root>/ClimSim/raw/train/<month>/E3SM-MMF.{mli,mlo}.<...>.nc
    <data_root>/ClimSim/processed/{train,val}_{inputs,outputs}.npy
    <data_root>/ClimSim/processed/{inp,out}_{mean,std}.npy

Source:
    https://huggingface.co/datasets/LEAP/ClimSim_low-res
    (E3SM-MMF.mli.<year>-<month>-<day>-<seconds>.nc input pairs and
     matching E3SM-MMF.mlo. output pairs)

Run via:
    mlsbench data ClimSim
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path


HF_DATASET = "LEAP/ClimSim_low-res"
TRAIN_MONTHS = ["0001-02", "0001-03"]
VAL_MONTHS = ["0001-07"]
ALL_MONTHS = TRAIN_MONTHS + VAL_MONTHS

# ---- preprocessing constants (mirror vendor/data/ClimSim/preprocess.py) -------
N_LEVELS = 60
INPUT_ML_VARS = [
    "state_t", "state_q0001", "state_q0002", "state_q0003",
    "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O",
]
INPUT_SL_VARS = [
    "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX",
    "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS",
    "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR",
    "cam_in_LWUP", "cam_in_ICEFRAC", "cam_in_LANDFRAC",
    "cam_in_OCNFRAC", "cam_in_SNOWHICE",
]
TEND_MAP = {
    "ptend_t": "state_t", "ptend_q0001": "state_q0001",
    "ptend_q0002": "state_q0002", "ptend_q0003": "state_q0003",
    "ptend_u": "state_u", "ptend_v": "state_v",
}
OUTPUT_ML_VARS = list(TEND_MAP.keys())
OUTPUT_SL_VARS = [
    "cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC",
    "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD",
]
INP_DIM = len(INPUT_ML_VARS) * N_LEVELS + len(INPUT_SL_VARS)  # 556
OUT_DIM = len(OUTPUT_ML_VARS) * N_LEVELS + len(OUTPUT_SL_VARS)  # 368

PROCESSED_FILES = [
    "train_inputs.npy", "train_outputs.npy",
    "val_inputs.npy", "val_outputs.npy",
    "inp_mean.npy", "inp_std.npy", "out_mean.npy", "out_std.npy",
]


def have_processed(processed: Path) -> bool:
    return all((processed / f).exists() for f in PROCESSED_FILES)


def download_month(month: str, raw_train: Path) -> None:
    # huggingface.co rate-limits unauthenticated tree listings; honour
    # HF_ENDPOINT (e.g. https://hf-mirror.com) before importing the hub
    # client. Default to the mirror if nothing is set.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    target = raw_train / month
    target.mkdir(parents=True, exist_ok=True)

    mli_files = sorted(target.glob("E3SM-MMF.mli.*.nc"))
    if mli_files and (target / mli_files[0].name.replace(".mli.", ".mlo.")).exists():
        print(f"  [SKIP] {month}: {len(mli_files)} input files already present")
        return

    print(
        f"  Downloading {month} from {HF_DATASET} via {os.environ['HF_ENDPOINT']}...",
        flush=True,
    )
    snap_root = raw_train.parent / "_hf_snapshot"
    pattern = f"train/{month}/*.nc"
    # snapshot_download retries internally but still gives up on 5xx storms;
    # retry the whole call so transient mirror outages don't kill prep.
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            snap_dir = snapshot_download(
                repo_id=HF_DATASET,
                repo_type="dataset",
                allow_patterns=[pattern],
                local_dir=str(snap_root),
                max_workers=4,
            )
            break
        except Exception as exc:
            last_err = exc
            wait = min(60, 5 * attempt)
            print(f"  HF download attempt {attempt}/5 failed: {exc}; retry in {wait}s", flush=True)
            import time as _t
            _t.sleep(wait)
    else:
        raise RuntimeError(f"HF snapshot for {month} failed after 5 attempts") from last_err
    src = Path(snap_dir) / "train" / month
    if not src.exists():
        raise RuntimeError(
            f"HF snapshot at {snap_dir} did not produce {src}; check that "
            f"{HF_DATASET} still hosts {month}."
        )
    for nc in src.glob("*.nc"):
        dest = target / nc.name
        if not dest.exists():
            shutil.move(str(nc), str(dest))
    if snap_root.exists():
        shutil.rmtree(snap_root, ignore_errors=True)
    print(
        f"  {month}: now {len(list(target.glob('E3SM-MMF.mli.*.nc')))} input files",
        flush=True,
    )


def process_nc_files(months, split_name, raw_dir):
    import numpy as np
    import xarray as xr

    all_inputs, all_outputs = [], []
    for month in months:
        month_dir = raw_dir / "train" / month
        mli_files = sorted(glob.glob(os.path.join(str(month_dir), "E3SM-MMF.mli.*.nc")))
        print(f"{split_name} month {month}: {len(mli_files)} input files")
        if not mli_files:
            raise RuntimeError(f"No .mli files in {month_dir}")
        for mli_f in mli_files:
            mlo_f = mli_f.replace(".mli.", ".mlo.")
            if not os.path.exists(mlo_f):
                continue
            try:
                ds_in = xr.open_dataset(mli_f)
                ds_out = xr.open_dataset(mlo_f)
            except Exception as e:  # noqa: BLE001
                print(f"  Skip {os.path.basename(mli_f)}: {e}")
                continue
            inp_parts = []
            for var in INPUT_ML_VARS:
                v = ds_in[var].values
                if v.ndim == 2 and v.shape[0] == N_LEVELS:
                    v = v.T
                inp_parts.append(v.astype(np.float32))
            for var in INPUT_SL_VARS:
                v = ds_in[var].values.flatten().astype(np.float32)
                inp_parts.append(v[:, None])
            inp = np.concatenate(inp_parts, axis=1)
            out_parts = []
            for tend_var in OUTPUT_ML_VARS:
                state_var = TEND_MAP[tend_var]
                v = (ds_out[state_var].values - ds_in[state_var].values) / 1200.0
                if v.ndim == 2 and v.shape[0] == N_LEVELS:
                    v = v.T
                out_parts.append(v.astype(np.float32))
            for var in OUTPUT_SL_VARS:
                v = ds_out[var].values.flatten().astype(np.float32)
                out_parts.append(v[:, None])
            out = np.concatenate(out_parts, axis=1)
            assert inp.shape[1] == INP_DIM, f"{inp.shape[1]} != {INP_DIM}"
            assert out.shape[1] == OUT_DIM, f"{out.shape[1]} != {OUT_DIM}"
            all_inputs.append(inp)
            all_outputs.append(out)
            ds_in.close()
            ds_out.close()
    X = np.concatenate(all_inputs, axis=0).astype(np.float32)
    Y = np.concatenate(all_outputs, axis=0).astype(np.float32)
    print(f"{split_name}: {X.shape[0]} samples, inp={X.shape}, out={Y.shape}")
    return X, Y


def run_preprocess(climsim_dir: Path) -> None:
    import numpy as np

    raw_dir = climsim_dir / "raw"
    out_dir = climsim_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Processing training data...")
    train_inp, train_out = process_nc_files(TRAIN_MONTHS, "train", raw_dir)
    print("Processing validation data...")
    val_inp, val_out = process_nc_files(VAL_MONTHS, "val", raw_dir)

    inp_mean = train_inp.mean(axis=0).astype(np.float32)
    inp_std = train_inp.std(axis=0).astype(np.float32)
    inp_std[inp_std < 1e-12] = 1.0
    out_mean = train_out.mean(axis=0).astype(np.float32)
    out_std = train_out.std(axis=0).astype(np.float32)
    out_std[out_std < 1e-12] = 1.0

    np.save(out_dir / "train_inputs.npy", train_inp)
    np.save(out_dir / "train_outputs.npy", train_out)
    np.save(out_dir / "val_inputs.npy", val_inp)
    np.save(out_dir / "val_outputs.npy", val_out)
    np.save(out_dir / "inp_mean.npy", inp_mean)
    np.save(out_dir / "inp_std.npy", inp_std)
    np.save(out_dir / "out_mean.npy", out_mean)
    np.save(out_dir / "out_std.npy", out_std)
    print(f"All files saved to {out_dir}")


def verify(climsim_dir: Path) -> None:
    processed = climsim_dir / "processed"
    if not have_processed(processed):
        print(f"ERROR: missing processed .npy files under {processed}", file=sys.stderr)
        sys.exit(1)
    print("All ClimSim data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    climsim_dir = Path(args.data_root) / "ClimSim"
    raw_train = climsim_dir / "raw" / "train"
    processed = climsim_dir / "processed"
    raw_train.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing ClimSim data at {climsim_dir} ===")
    if have_processed(processed):
        print("  [SKIP] processed/*.npy already complete")
        return

    for month in ALL_MONTHS:
        download_month(month, raw_train)

    run_preprocess(climsim_dir)
    verify(climsim_dir)


if __name__ == "__main__":
    main()
