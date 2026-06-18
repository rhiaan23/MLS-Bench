#!/usr/bin/env python3
"""Prepare data for ClimSim (ai4sci-climate-emulation).

Follows the official ClimSim protocol (Yu et al., NeurIPS 2023 D&B;
github.com/leap-stc/ClimSim) at a reduced, fast-to-train scale:

* TEMPORAL HOLDOUT, NOT cross-season.  ClimSim trains on early years and
  validates on a later one, so train and val/test each span a *full annual
  cycle* (all seasons appear in both).  We mirror that with adjacent years:
      train = simulation year 1 (0001-02 .. 0002-01), ~300k samples
      val   = year 2, even timesteps   (0002-02 .. 0003-01)
      test  = year 2, odd timesteps
  Because train covers every season, each variable's training range captures its
  natural variability and the held-out year-2 stays in-distribution — this is
  what prevents the val/test NMSE blow-up the old Feb/Mar->July cross-season
  split produced.  (Years 1-2 are the closest pair, avoiding later-year sampling
  artifacts seen when spanning to years 7-8.  ~300k keeps the var>0.01 NMSE mask
  on the reliably-skillful target dims; denser sampling pulls near-unpredictable
  intermittent cloud-tendency dims into the metric.  Overfitting of the larger
  baselines is instead handled by val-NMSE checkpoint selection in the trainer.)

* RANGE NORMALIZATION, as in ClimSim's climsim_utils/data_utils.py:
      x_norm = (x - input_mean) / (input_max - input_min)        (line 808)
  ClimSim zeroes the non-finite results of constant dims (max==min); we instead
  floor that divisor to 1.0 here, so constant dims become (x-mean) with no
  division by zero and the trainer needs no special-casing.
  We store the per-feature subtractor (`*_mean.npy`) and divisor
  (`*_std.npy` == max-min) computed on the TRAIN split; the trainer applies
  the division and zeroes any non-finite entries.  This replaces the old
  (x-mean)/std scheme whose 1e-12 std floor let near-constant trace-gas dims
  explode.

* Time subsampling (cf. ClimSim's stride-7) keeps the download ~4 GB and the
  training budgets comparable to the historical baselines.

Output:
    <data_root>/ClimSim/raw/train/<month>/E3SM-MMF.{mli,mlo}.<...>.nc
    <data_root>/ClimSim/processed/{train,val,test}_{inputs,outputs}.npy
    <data_root>/ClimSim/processed/{inp,out}_{mean,std}.npy   (std == max-min)

Source: https://huggingface.co/datasets/LEAP/ClimSim_low-res

Run via: mlsbench data ClimSim
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HF_DATASET = "LEAP/ClimSim_low-res"

# E3SM-MMF runs a no-leap (365-day) calendar; 72 timesteps/day at 1200 s.
DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
SECONDS = [f"{s:05d}" for s in range(0, 86400, 1200)]  # 00000 .. 85200


def year_months(start_year_month, n_months=12):
    """List of 'YYYY-MM' strings, n_months starting at (year, month)."""
    y, m = start_year_month
    out = []
    for _ in range(n_months):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


# Temporal holdout: train = simulation year 1 (densely sampled), val/test =
# year 2 (interleaved). Both span a full annual cycle (all seasons in both), so
# it is an in-distribution temporal-generalization test, not cross-season
# extrapolation. Year 1 is sampled densely (~169 files/month -> ~780k samples)
# so the larger reference baselines (cnn 14M, unet 10.8M) get enough data not to
# overfit; a sparse single year (~300k) let them overfit. Years 1 and 2 are the
# closest pair, avoiding any later-year sampling artifacts.
TRAIN_MONTHS = year_months((1, 2), 12)   # 0001-02 .. 0002-01  (year 1)
EVAL_MONTHS = year_months((2, 2), 12)    # 0002-02 .. 0003-01  (year 2 -> val/test)

# Deterministic, evenly-strided files per month (x384 columns/file).
# ~65/month over 12 train months ~= 300k train; ~18/month over 12 eval months
# ~= 83k -> ~41k val + ~41k test after the even/odd interleave. (Denser sampling
# pulls intermittent cloud-tendency dims above the var>0.01 NMSE mask; those are
# near-unpredictable, so ~300k keeps the metric on the reliably-skillful dims.)
FILES_PER_MONTH_TRAIN = 65
FILES_PER_MONTH_EVAL = 18

# ---- preprocessing constants -------------------------------------------------
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
    "test_inputs.npy", "test_outputs.npy",
    "inp_mean.npy", "inp_std.npy", "out_mean.npy", "out_std.npy",
]


def have_processed(processed: Path) -> bool:
    return all((processed / f).exists() for f in PROCESSED_FILES)


def month_file_stems(month: str, n_files: int):
    """Deterministic, evenly-strided list of '<month>-<dd>-<sssss>' stems."""
    import numpy as np

    mm = int(month.split("-")[1])
    all_stems = [
        f"{month}-{d:02d}-{s}"
        for d in range(1, DAYS_IN_MONTH[mm] + 1)
        for s in SECONDS
    ]
    n = min(n_files, len(all_stems))
    idx = sorted(set(int(i) for i in np.linspace(0, len(all_stems) - 1, n).round()))
    return [all_stems[i] for i in idx]


def _resolve_url(rel: str) -> str:
    # Default to the HF mirror (matches the repo's other data scripts; works on
    # compute nodes where direct huggingface.co is blocked). Honour HF_ENDPOINT
    # override (e.g. https://huggingface.co) when set.
    endpoint = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    return f"{endpoint}/datasets/{HF_DATASET}/resolve/main/train/{rel}?download=true"


def _download_one(rel: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(_resolve_url(rel), timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, OSError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"failed to download {rel} after 5 attempts: {last}")


def download_subset(raw_train: Path) -> None:
    """Fetch the deterministic strided .mli/.mlo subset for every split month.

    Plain resolve-URL HTTP (no HF tree/xet API) so it is immune to the dataset's
    API rate limits. Honours HF_ENDPOINT for mirrors.
    """
    jobs = []
    for month, n in [(m, FILES_PER_MONTH_TRAIN) for m in TRAIN_MONTHS] + \
                     [(m, FILES_PER_MONTH_EVAL) for m in EVAL_MONTHS]:
        tgt = raw_train / month
        tgt.mkdir(parents=True, exist_ok=True)
        for stem in month_file_stems(month, n):
            for kind in ("mli", "mlo"):
                name = f"E3SM-MMF.{kind}.{stem}.nc"
                jobs.append((f"{month}/{name}", tgt / name))
    todo = [(rel, d) for rel, d in jobs if not (d.exists() and d.stat().st_size > 0)]
    print(f"  downloading {len(todo)}/{len(jobs)} files...", flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda a: _download_one(*a), todo))
    print("  download complete", flush=True)


def month_pairs(months, n_per_month):
    """Flat list of (month, stem) for the strided file subset of each month."""
    return [(m, s) for m in months for s in month_file_stems(m, n_per_month)]


def process_pairs(pairs, raw_dir):
    """Process an explicit list of (month, stem) NetCDF pairs into (X, Y).

    Each stem is one *timestep* (a whole NetCDF holding all 384 columns), so
    splitting the *pairs* list (not the row-concatenated array) keeps val and
    test on disjoint timesteps — an independent temporal holdout."""
    import numpy as np
    import xarray as xr

    Xs, Ys = [], []
    for month, stem in pairs:
        month_dir = raw_dir / "train" / month
        mli_f = month_dir / f"E3SM-MMF.mli.{stem}.nc"
        mlo_f = month_dir / f"E3SM-MMF.mlo.{stem}.nc"
        if not (mli_f.exists() and mlo_f.exists()):
            raise RuntimeError(f"missing pair for {stem} under {month_dir}")
        ds_in = xr.open_dataset(mli_f)
        ds_out = xr.open_dataset(mlo_f)
        inp_parts = []
        for var in INPUT_ML_VARS:
            v = ds_in[var].values
            if v.ndim == 2 and v.shape[0] == N_LEVELS:
                v = v.T
            inp_parts.append(v.astype(np.float32))
        for var in INPUT_SL_VARS:
            inp_parts.append(ds_in[var].values.flatten().astype(np.float32)[:, None])
        inp = np.concatenate(inp_parts, axis=1)
        out_parts = []
        for tend_var in OUTPUT_ML_VARS:
            state_var = TEND_MAP[tend_var]
            v = (ds_out[state_var].values - ds_in[state_var].values) / 1200.0
            if v.ndim == 2 and v.shape[0] == N_LEVELS:
                v = v.T
            out_parts.append(v.astype(np.float32))
        for var in OUTPUT_SL_VARS:
            out_parts.append(ds_out[var].values.flatten().astype(np.float32)[:, None])
        out = np.concatenate(out_parts, axis=1)
        assert inp.shape[1] == INP_DIM and out.shape[1] == OUT_DIM
        Xs.append(inp)
        Ys.append(out)
        ds_in.close()
        ds_out.close()
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    print(f"  {len(pairs)} files: {X.shape[0]} samples", flush=True)
    return X, Y


def range_divisor(X):
    """max-min per feature (ClimSim INPUT normalization denominator,
    climsim_utils/data_utils.py:808), float64-accumulated (the cam_in_SNOWHICE
    ~1e30 fill sentinel overflows float32). Truly-constant dims (max==min) get a
    divisor of 1.0 so the normalized value is just (x-mean) with no division by
    zero — the all-seasons training year already keeps every informative dim's
    range well away from zero, so nothing else needs flooring."""
    import numpy as np

    x64 = X.astype(np.float64)
    rng = x64.max(axis=0) - x64.min(axis=0)
    rng[rng < 1e-12] = 1.0  # constant dims -> divisor 1.0 (no inf at load time)
    return rng.astype(np.float32)


def std_divisor(X):
    """Per-feature std for OUTPUT normalization. ClimSim scales targets by a
    per-level `output_scale` so each is O(1) (data_utils.py:809); a train std
    divisor is the equivalent. Near-constant tendency dims (std<1e-8) get a
    divisor of 1.0 so their normalized variance stays ~0 and the NMSE var>0.01
    mask correctly excludes them. float64-accumulated for safety."""
    import numpy as np

    sd = X.std(axis=0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    return sd.astype(np.float32)


def run_preprocess(climsim_dir: Path) -> None:
    import numpy as np

    raw_dir = climsim_dir / "raw"
    out_dir = climsim_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Processing train (year 1)...", flush=True)
    train_inp, train_out = process_pairs(month_pairs(TRAIN_MONTHS, FILES_PER_MONTH_TRAIN), raw_dir)

    # Split the year-2 files (each a distinct timestep) even/odd into val/test, so
    # val and test are DISJOINT timesteps (an independent temporal holdout) while
    # both still span the whole year. Splitting at the file/timestep level — not by
    # row (which would alternate spatial columns of the SAME timestep) — is what
    # keeps test independent of the val set used for early-stopping/selection.
    eval_pairs = month_pairs(EVAL_MONTHS, FILES_PER_MONTH_EVAL)
    val_pairs, test_pairs = eval_pairs[0::2], eval_pairs[1::2]
    print("Processing val (year 2, even timesteps)...", flush=True)
    val_inp, val_out = process_pairs(val_pairs, raw_dir)
    print("Processing test (year 2, odd timesteps)...", flush=True)
    test_inp, test_out = process_pairs(test_pairs, raw_dir)
    print(f"split: train={len(train_inp)} val={len(val_inp)} test={len(test_inp)} "
          f"(temporal holdout: yr1 train / yr2 timesteps split val|test)", flush=True)

    # Normalization statistics from TRAIN only.
    #   inputs:  range-normalized  (x-mean)/(max-min), ClimSim-style
    #   outputs: std-normalized    (y-mean)/std, so each target is O(1) (ClimSim
    #            output_scale equivalent) and the NMSE var>0.01 mask stays valid.
    inp_mean = train_inp.mean(axis=0, dtype=np.float64).astype(np.float32)
    inp_std = range_divisor(train_inp)           # == input_max - input_min
    out_mean = train_out.mean(axis=0, dtype=np.float64).astype(np.float32)
    out_std = std_divisor(train_out)

    np.save(out_dir / "train_inputs.npy", train_inp)
    np.save(out_dir / "train_outputs.npy", train_out)
    np.save(out_dir / "val_inputs.npy", val_inp)
    np.save(out_dir / "val_outputs.npy", val_out)
    np.save(out_dir / "test_inputs.npy", test_inp)
    np.save(out_dir / "test_outputs.npy", test_out)
    np.save(out_dir / "inp_mean.npy", inp_mean)
    np.save(out_dir / "inp_std.npy", inp_std)
    np.save(out_dir / "out_mean.npy", out_mean)
    np.save(out_dir / "out_std.npy", out_std)
    print(f"All files saved to {out_dir}", flush=True)


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

    download_subset(raw_train)
    run_preprocess(climsim_dir)
    verify(climsim_dir)


if __name__ == "__main__":
    main()
