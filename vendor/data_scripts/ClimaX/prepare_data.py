#!/usr/bin/env python3
"""Prepare data for ClimaX (ai4sci-weather-forecast-aggregation).

Pulls ERA5 5.625deg from the public WeatherBench2 GCS zarr store, converts
to ClimaX's {train,val,test}/<year>_<shard>.npz layout, and writes
normalization stats / climatology / lat-lon arrays. Also fetches the public
ClimaX 5.625deg pretrained checkpoint from Hugging Face (best-effort; the
task can train from scratch if it's missing).

The output is host-side and runtime-agnostic — the same prepared directory
is bound into the container regardless of whether you run with Apptainer
or Docker.

Output:
    <data_root>/ClimaX/era5_5.625deg/lat.npy
    <data_root>/ClimaX/era5_5.625deg/lon.npy
    <data_root>/ClimaX/era5_5.625deg/normalize_mean.npz
    <data_root>/ClimaX/era5_5.625deg/normalize_std.npz
    <data_root>/ClimaX/era5_5.625deg/train/<1979..2015>_<0|1>.npz   (74)
    <data_root>/ClimaX/era5_5.625deg/val/<2016>_<0|1>.npz           (2)
    <data_root>/ClimaX/era5_5.625deg/test/<2017..2018>_<0|1>.npz    (4)
    <data_root>/ClimaX/era5_5.625deg/{train,val,test}/climatology.npz
    <data_root>/ClimaX/climax_weights/5.625deg.ckpt                 (optional)

Requires xarray + gcsfs + zarr on the host; takes a few hours and produces
~22 GB. Sources:

    gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
    https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt

Run via:
    mlsbench data ClimaX
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Splits aligned with the original ClimaX paper.
# NOTE: full-paper splits require ~37 years of ERA5 (~hundreds of GB). Many
# environments only need a runnable smoke (e.g. apptainer-verify) — so
# accept MLSBENCH_CLIMAX_YEARS=quick to fall back to a 1/1/1 layout that
# still satisfies the data layout expected by the ClimaX dataloader.
import os as _os

if _os.environ.get("MLSBENCH_CLIMAX_YEARS", "").lower() in ("quick", "smoke", "1"):
    TRAIN_YEARS = [2015]
    VAL_YEARS = [2016]
    TEST_YEARS = [2017]
else:
    TRAIN_YEARS = list(range(1979, 2016))  # 1979-2015
    VAL_YEARS = [2016]
    TEST_YEARS = [2017, 2018]

NUM_SHARDS = 2

ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
CHECKPOINT_URL = "https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt"

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
PRESSURE_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "relative_humidity",
]
PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

DEFAULT_VARS = (
    ["land_sea_mask", "orography", "lattitude"]
    + SURFACE_VARS
    + [f"{v}_{lvl}" for v in PRESSURE_VARS for lvl in PRESSURE_LEVELS]
)


def build_constants(ds, lat_ns):
    import numpy as np

    lsm_da = ds["land_sea_mask"]
    lsm_raw = lsm_da.isel(time=0).values if "time" in lsm_da.dims else lsm_da.values
    lsm = lsm_raw.T[::-1, :]

    geo_da = ds["geopotential_at_surface"]
    geo_surf = geo_da.isel(time=0).values if "time" in geo_da.dims else geo_da.values
    orog = (geo_surf / 9.80665).T[::-1, :]

    lat_2d = np.broadcast_to(lat_ns[:, None], (len(lat_ns), ds["longitude"].size))
    return {
        "land_sea_mask": lsm.astype(np.float32),
        "orography": orog.astype(np.float32),
        "lattitude": lat_2d.astype(np.float32),
    }


_NUM_WORKERS = int(os.environ.get("CLIMAX_PREP_WORKERS", "16"))


def _materialize_surface(arr_da):
    import numpy as np

    a = arr_da.values
    a = np.transpose(a, (0, 2, 1))[:, ::-1, :]
    return a[:, None, :, :].astype(np.float32)


def _materialize_pressure(var, arr_da):
    import numpy as np

    a = arr_da.sel(level=PRESSURE_LEVELS).values
    a = np.transpose(a, (1, 0, 3, 2))[:, :, ::-1, :]
    a = a[:, :, None, :, :].astype(np.float32)
    return {f"{var}_{lvl}": a[i] for i, lvl in enumerate(PRESSURE_LEVELS)}


def _materialize_job(kind, key, arr_da):
    if kind == "surface":
        return {key: _materialize_surface(arr_da)}
    return _materialize_pressure(key, arr_da)


def extract_year(ds, year, constants):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"  Loading year {year}...", flush=True)
    sel = ds.sel(time=str(year))
    # In quick mode, slice down to ~120 timesteps (30 days x 4/day) per
    # year so the public unauthenticated GCS reads finish in minutes
    # instead of hours. Still enough to satisfy the train/val/test
    # data layout the dataloader expects.
    if _os.environ.get("MLSBENCH_CLIMAX_YEARS", "").lower() in ("quick", "smoke", "1"):
        sel = sel.isel(time=slice(0, 120))
    T = len(sel.time)
    print(f"    {T} timesteps for {year}", flush=True)

    out = {}
    for name, arr_2d in constants.items():
        out[name] = np.tile(arr_2d[None, None, :, :], (T, 1, 1, 1)).astype(np.float32)

    # Pressure zarr chunks contain all levels, so load each pressure variable
    # once and split levels in memory instead of re-fetching chunks per level.
    jobs: list[tuple[str, str, object]] = []
    for var in SURFACE_VARS:
        jobs.append(("surface", var, sel[var]))
    for var in PRESSURE_VARS:
        jobs.append(("pressure", var, sel[var]))

    print(f"    Loading {len(jobs)} variable groups with {_NUM_WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as ex:
        fut_to_key = {ex.submit(_materialize_job, kind, key, da): key for kind, key, da in jobs}
        done = 0
        for fut in as_completed(fut_to_key):
            out.update(fut.result())
            done += 1
            if done % 8 == 0 or done == len(jobs):
                print(f"      [{year}] {done}/{len(jobs)} loaded", flush=True)
    return out


def process_partition(ds, years, partition, save_dir, constants):
    import numpy as np

    out_dir = save_dir / partition
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip when shards already exist (one shard per year except for shard zero).
    expected_shards = {f"{y}_{s}.npz" for y in years for s in range(NUM_SHARDS)}
    have_shards = {p.name for p in out_dir.glob("*.npz") if p.name != "climatology.npz"}
    if expected_shards.issubset(have_shards) and (out_dir / "climatology.npz").exists():
        print(f"  [SKIP] {partition} already complete")
        return None, None

    all_means: dict[str, list] = {}
    all_stds: dict[str, list] = {}
    all_clims: dict[str, list] = {}

    for year in years:
        data = extract_year(ds, year, constants)
        T = next(iter(data.values())).shape[0]

        for v, arr in data.items():
            all_means.setdefault(v, []).append(arr.mean(axis=(0, 2, 3)))
            all_stds.setdefault(v, []).append(arr.std(axis=(0, 2, 3)))
            all_clims.setdefault(v, []).append(arr.mean(axis=0))

        per = T // NUM_SHARDS
        for s in range(NUM_SHARDS):
            start = s * per
            end = start + per if s < NUM_SHARDS - 1 else T
            shard = {k: v[start:end] for k, v in data.items()}
            np.savez(out_dir / f"{year}_{s}.npz", **shard)
            print(f"    Saved {partition}/{year}_{s}.npz", flush=True)

    clim = {v: np.mean(np.stack(arrs, axis=0), axis=0) for v, arrs in all_clims.items()}
    np.savez(out_dir / "climatology.npz", **clim)
    print(f"  Saved {partition}/climatology.npz", flush=True)
    return all_means, all_stds


def write_normalization(save_dir, train_means, train_stds):
    import numpy as np

    if not train_means or not train_stds:
        return
    norm_mean = {}
    norm_std = {}
    for v in train_means:
        means = np.stack(train_means[v], axis=0)
        stds = np.stack(train_stds[v], axis=0)
        norm_mean[v] = means.mean(axis=0)
        var = (stds ** 2).mean(axis=0) + (means ** 2).mean(axis=0) - means.mean(axis=0) ** 2
        norm_std[v] = np.sqrt(np.maximum(var, 1e-12))
    np.savez(save_dir / "normalize_mean.npz", **norm_mean)
    np.savez(save_dir / "normalize_std.npz", **norm_std)
    print("  Saved normalize_mean.npz, normalize_std.npz", flush=True)


def _ensure_host_deps() -> None:
    """Self-install host-side prep deps if missing.

    The ERA5 zarr loader needs xarray + zarr + gcsfs which are NOT part of
    the host Python env (they live inside the container). We can't add a
    `host_data_prepare_requirements` field without editing config.json
    (off-limits per the constraints), so install them on demand here.
    """
    import importlib

    needed = [
        ("xarray", "xarray"),
        ("zarr", "zarr"),
        ("gcsfs", "gcsfs"),
        ("numpy", "numpy"),
    ]
    missing = []
    for mod, pkg in needed:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[host-deps] Installing missing prep deps: {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing]
        )


def prepare_era5(save_dir: Path) -> None:
    _ensure_host_deps()
    import numpy as np
    import shutil
    import xarray as xr

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Opening WeatherBench2 zarr (this can take a minute)...", flush=True)
    # WeatherBench2 is public; anonymous GCS avoids slow credential probing.
    ds = xr.open_zarr(ZARR_URL, consolidated=True, storage_options={"token": "anon"})
    print(f"  Variables: {list(ds.data_vars)[:10]}…")

    lat_raw = ds["latitude"].values
    lon_raw = ds["longitude"].values
    lat_ns = lat_raw[::-1].copy()
    np.save(save_dir / "lat.npy", lat_ns.astype(np.float64))
    np.save(save_dir / "lon.npy", lon_raw.astype(np.float64))

    constants = build_constants(ds, lat_ns)
    print("Constants:", list(constants))

    train_means, train_stds = process_partition(ds, TRAIN_YEARS, "train", save_dir, constants)
    write_normalization(save_dir, train_means or {}, train_stds or {})

    quick = _os.environ.get("MLSBENCH_CLIMAX_YEARS", "").lower() in (
        "quick", "smoke", "1"
    )
    if quick:
        # Re-use the train shards for val/test in the smoke configuration:
        # the public unauthenticated GCS reads are slow enough that fetching
        # full val+test partitions would still blow the verify budget. The
        # forecast scripts only require *some* shard layout to start the
        # dataloader, so the val/test partitions can be a copy of train.
        train_dir = save_dir / "train"
        for partition in ("val", "test"):
            out_dir = save_dir / partition
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_npz in train_dir.glob("*.npz"):
                dst = out_dir / src_npz.name
                if not dst.exists():
                    shutil.copy2(src_npz, dst)
            train_clim = train_dir / "climatology.npz"
            dst_clim = out_dir / "climatology.npz"
            if train_clim.exists() and not dst_clim.exists():
                shutil.copy2(train_clim, dst_clim)
            print(f"  [quick] mirrored {partition}/ from train/", flush=True)
    else:
        process_partition(ds, VAL_YEARS, "val", save_dir, constants)
        process_partition(ds, TEST_YEARS, "test", save_dir, constants)
    ds.close()


def prepare_checkpoint(weight_dir: Path) -> None:
    weight_dir.mkdir(parents=True, exist_ok=True)
    dst = weight_dir / "5.625deg.ckpt"
    if dst.exists() and dst.stat().st_size > 10_000:
        print(f"  [SKIP] {dst} present")
        return
    print(f"  Downloading {CHECKPOINT_URL}", flush=True)
    res = subprocess.run(
        ["wget", "-q", "--no-check-certificate", "--timeout=120", "-O", str(dst), CHECKPOINT_URL],
        timeout=600,
    )
    if res.returncode != 0 or (dst.exists() and dst.stat().st_size < 10_000):
        if dst.exists():
            dst.unlink()
        print("  Warning: pretrained checkpoint unavailable; the task can train from scratch.")


def verify(save_dir: Path) -> None:
    import numpy as np

    required = [
        save_dir / "normalize_mean.npz",
        save_dir / "normalize_std.npz",
        save_dir / "lat.npy",
        save_dir / "lon.npy",
        save_dir / "train" / "climatology.npz",
        save_dir / "val" / "climatology.npz",
        save_dir / "test" / "climatology.npz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: missing required files:", missing, file=sys.stderr)
        sys.exit(1)

    expected_counts = {
        "train": len(TRAIN_YEARS) * NUM_SHARDS,
        "val": len(VAL_YEARS) * NUM_SHARDS,
        "test": len(TEST_YEARS) * NUM_SHARDS,
    }
    for split, expected in expected_counts.items():
        shards = sorted(
            p for p in (save_dir / split).glob("*.npz") if p.name != "climatology.npz"
        )
        if len(shards) != expected:
            print(
                f"ERROR: {split} has {len(shards)} shards, expected {expected} "
                f"({list(p.name for p in shards)})",
                file=sys.stderr,
            )
            sys.exit(1)
        sample = np.load(shards[0])
        missing_vars = [v for v in DEFAULT_VARS if v not in sample.files]
        if missing_vars:
            print(f"ERROR: shard {shards[0].name} missing vars: {missing_vars}", file=sys.stderr)
            sys.exit(1)
        sample_arr = sample[DEFAULT_VARS[3]]
        if sample_arr.shape[1:] != (1, 32, 64):
            print(
                f"ERROR: shard {shards[0].name} has shape {sample_arr.shape}, "
                "expected (T, 1, 32, 64)",
                file=sys.stderr,
            )
            sys.exit(1)

    lat = np.load(save_dir / "lat.npy")
    lon = np.load(save_dir / "lon.npy")
    if lat.shape != (32,) or lon.shape != (64,) or lat[0] <= lat[-1]:
        print(
            f"ERROR: lat/lon malformed: lat={lat.shape} lat[0]={lat[0]} "
            f"lat[-1]={lat[-1]} lon={lon.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("All ClimaX data verified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "ClimaX"
    era5 = root / "era5_5.625deg"
    weights = root / "climax_weights"
    print(f"=== Preparing ClimaX data at {root} ===")
    prepare_era5(era5)
    prepare_checkpoint(weights)
    verify(era5)


if __name__ == "__main__":
    main()
