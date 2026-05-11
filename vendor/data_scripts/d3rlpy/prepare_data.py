"""Download and convert RL Unplugged Atari data locally using multiprocessing.

Converts GCS TFRecord shards → numpy .gz files matching d4rl-atari layout:
  {outdir}/{Game}/1/{epoch}/observation.gz, action.gz, reward.gz, terminal.gz

Usage:
  python prepare_data.py [--workers 8] [--data-root vendor/data]

Requires: gsutil, tensorflow, numpy, Pillow
"""

import argparse
import gzip
import io
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

GAMES = ["Breakout", "Pong", "Qbert"]
N_SHARDS = 50
MAX_TRANSITIONS = 1_000_000


def convert_shard(args):
    """Download one shard from GCS, convert TFRecord → numpy, delete raw."""
    game, shard_idx, outdir = args
    epoch = shard_idx + 1
    out_path = Path(outdir) / game / "1" / str(epoch)

    # Skip if already done
    if (out_path / "observation.gz").exists():
        print(f"[{game}] epoch {epoch}: already exists, skipping", flush=True)
        return f"{game}/{epoch}: skipped"

    import tensorflow as tf

    shard_name = f"run_1-{shard_idx:05d}-of-00050"
    tmp_dir = os.path.join(outdir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"rlu_shard_{game}_{shard_idx}_{os.getpid()}")
    url = f"gs://rl_unplugged/atari_episodes_ordered/{game}/{shard_name}"

    try:
        subprocess.run(
            ["gsutil", "cp", url, tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print(f"[{game}] epoch {epoch}: download failed: {e}", flush=True)
        return f"{game}/{epoch}: download failed"

    try:
        ds = tf.data.TFRecordDataset(tmp_path, compression_type="GZIP")
        obs, act, rew, trm = [], [], [], []
        total = 0
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            f = ex.features.feature
            actions = list(f["actions"].int64_list.value)
            rewards = list(f["clipped_rewards"].float_list.value)
            discounts = list(f["discounts"].float_list.value)
            n = len(actions)
            pngs = list(f["observations"].bytes_list.value[:n])
            frames = [np.array(Image.open(io.BytesIO(p)), dtype=np.uint8) for p in pngs]
            arr = np.array(frames, dtype=np.uint8)
            if arr.ndim == 4:
                arr = arr[:, :, :, 0]
            obs.append(arr)
            act.append(np.array(actions[:n], dtype=np.int32))
            rew.append(np.array(rewards[:n], dtype=np.float32))
            trm.append(np.array([1.0 - d for d in discounts[:n]], dtype=np.float32))
            total += n
            if total >= MAX_TRANSITIONS:
                break

        arrays = {
            "observation": np.concatenate(obs)[:MAX_TRANSITIONS],
            "action": np.concatenate(act)[:MAX_TRANSITIONS],
            "reward": np.concatenate(rew)[:MAX_TRANSITIONS],
            "terminal": np.concatenate(trm)[:MAX_TRANSITIONS],
        }
        got = arrays["observation"].shape[0]
        # Pad if needed (some shards may have fewer transitions)
        if got < MAX_TRANSITIONS:
            for k in arrays:
                arrays[k] = np.concatenate([arrays[k], arrays[k][: MAX_TRANSITIONS - got]])

        out_path.mkdir(parents=True, exist_ok=True)
        for name, a in arrays.items():
            with gzip.open(str(out_path / f"{name}.gz"), "wb") as fp:
                np.save(fp, a)

        print(f"[{game}] epoch {epoch}: {got} transitions", flush=True)
        return f"{game}/{epoch}: {got} transitions"
    except Exception as e:
        print(f"[{game}] epoch {epoch}: conversion failed: {e}", flush=True)
        return f"{game}/{epoch}: failed: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data"),
    )
    args = parser.parse_args()

    outdir = str(Path(args.data_root) / "data_staging" / "d4rl_atari")

    tasks = [(game, i, outdir) for game in GAMES for i in range(N_SHARDS)]
    print(f"Processing {len(tasks)} shards with {args.workers} workers → {outdir}")

    with Pool(args.workers) as pool:
        results = pool.map(convert_shard, tasks)

    failed = [r for r in results if "failed" in r]
    print(f"\nDone: {len(results) - len(failed)}/{len(results)} succeeded")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
