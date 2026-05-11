#!/usr/bin/env python3
"""Prepare COCO val2014 images and pre-compute inception stats for FID.

Memory-efficient streaming version:
  - Stream-downloads val2014.zip via curl (no whole-file in RAM).
  - Extracts entries one at a time (not extractall) so peak RSS stays bounded.
  - Computes Inception statistics with online sums (sum_x, sum_xx) so we
    never materialize the full N x 2048 activation tensor.
  - Uses CUDA when available; falls back to CPU. Frees per-batch buffers
    explicitly so peak RSS stays bounded regardless of dataset size.

Output: {data_root}/coco_val2014/
    val2014/              # ~40K JPEG images
    inception_stats.npz   # pre-computed (mu, sigma) for FID

Usage:
    python vendor/data_scripts/CFGpp-main/prepare_coco.py --data-root vendor/data
    # or via mlsbench:
    mlsbench data CFGpp-main
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"


def run_cmd(cmd, cwd=None):
    """Run a shell command, raising on failure."""
    print(f"  $ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def stream_download(url: str, dest: Path):
    """Stream-download via curl so the whole file never lives in RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        # Resume partial download
        run_cmd(f"curl -L --fail --retry 5 --retry-delay 10 -C - -o {tmp} {url}")
    else:
        run_cmd(f"curl -L --fail --retry 5 --retry-delay 10 -o {tmp} {url}")
    tmp.rename(dest)


def prepare_coco_images(coco_dir: Path):
    """Download and extract COCO val2014 images."""
    img_dir = coco_dir / "val2014"

    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 40000:
        print("[coco_val2014] Images already exist, skipping download")
        return img_dir

    coco_dir.mkdir(parents=True, exist_ok=True)
    zip_path = coco_dir / "val2014.zip"

    if not zip_path.exists():
        print("[coco_val2014] Stream-downloading COCO val2014 (~6GB) via curl...", flush=True)
        stream_download(COCO_VAL_URL, zip_path)
    else:
        print(
            f"[coco_val2014] Zip already at {zip_path} "
            f"({zip_path.stat().st_size / 1e9:.2f} GB), skipping download"
        )

    print("[coco_val2014] Extracting (one entry at a time)...", flush=True)
    # Iterate entries instead of extractall(): keeps memory usage flat regardless
    # of zip size.
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members):
            zf.extract(member, str(coco_dir))
            if (i + 1) % 5000 == 0:
                print(f"  extracted {i + 1}/{len(members)} files", flush=True)
    # Clean up zip to save space
    zip_path.unlink(missing_ok=True)
    print("[coco_val2014] Extraction done")

    assert img_dir.exists(), "COCO val2014 extraction failed"
    return img_dir


def prepare_inception_stats(img_dir: Path, npz_path: Path, batch_size: int = 32):
    """Compute Inception-v3 activation statistics with streaming online updates.

    Uses the pair-wise summing identities so we never store all activations:
        sum_x[d]      += pred[i, d]
        sum_xx[d1,d2] += pred[i, d1] * pred[i, d2]
    then mu = sum_x / N, sigma = (sum_xx - N * mu mu^T) / (N - 1).

    The 2048 x 2048 sum_xx matrix is ~32 MB float64 — flat regardless of N.
    """
    if npz_path.exists():
        print("[coco_val2014] Inception stats already exist, skipping")
        return

    import gc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from pytorch_fid.inception import InceptionV3
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[coco_val2014] Computing inception stats on {device} (streaming)...", flush=True)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    class ResizedImageDataset(Dataset):
        def __init__(self, img_dir):
            self.files = sorted([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)

    dataset = ResizedImageDataset(img_dir)
    # Bound RAM: 0 workers on CPU; 2 on GPU. Activation buffers are released per batch.
    n_workers = 2 if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[coco_val2014] Processing {len(dataset)} images (batch={batch_size})...", flush=True)

    sum_x = np.zeros(dims, dtype=np.float64)
    sum_xx = np.zeros((dims, dims), dtype=np.float64)
    n = 0
    log_every = max(1, len(dataloader) // 50)
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)[0]               # (B, dims, 1, 1)
            pred = pred.squeeze(-1).squeeze(-1)  # (B, dims)
            pred_np = pred.detach().to(torch.float32).cpu().numpy().astype(np.float64)
            sum_x += pred_np.sum(axis=0)
            sum_xx += pred_np.T @ pred_np
            n += pred_np.shape[0]
            del pred, pred_np, batch
            if (bi + 1) % log_every == 0:
                print(f"  batch {bi + 1}/{len(dataloader)} (n={n})", flush=True)
                gc.collect()

    if n < 2:
        raise RuntimeError(f"Too few images processed: {n}")
    mu = sum_x / n
    # Unbiased covariance: (sum_xx - n * mu mu^T) / (n - 1)
    sigma = (sum_xx - n * np.outer(mu, mu)) / (n - 1)

    np.savez(str(npz_path), mu=mu, sigma=sigma)
    print(f"[coco_val2014] Saved inception stats to {npz_path} (n={n})")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO val2014 images and inception stats for FID"
    )
    parser.add_argument("--data-root", type=str, required=True, help="Root data directory")
    args = parser.parse_args()

    coco_dir = Path(args.data_root) / "coco_val2014"
    img_dir = prepare_coco_images(coco_dir)

    npz_path = coco_dir / "inception_stats.npz"
    prepare_inception_stats(img_dir, npz_path)

    print("\n[coco_val2014] Done")
    print(f"  Images:  {img_dir}")
    print(f"  Stats:   {npz_path}")


if __name__ == "__main__":
    main()
