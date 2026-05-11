"""Prepare script for CFGpp-main data dependencies.

Downloads Stable Diffusion model weights to the host cache directory.
Run via: mlsbench data CFGpp-main
"""

import argparse
import os
import sys
from pathlib import Path

# Use HF mirror to avoid connectivity issues
HF_MIRROR = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    # If HF_HOME is already set (e.g. inside container), use it directly;
    # otherwise compute from --data-root.
    if os.environ.get("HF_HOME"):
        cache_dir = Path(os.environ["HF_HOME"])
    elif args.data_root:
        cache_dir = Path(args.data_root) / "huggingface_cache"
        os.environ["HF_HOME"] = str(cache_dir)
    else:
        print("ERROR: Either set HF_HOME or pass --data-root")
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set mirror endpoint before importing diffusers
    os.environ["HF_ENDPOINT"] = HF_MIRROR

    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

    failed = []

    print(f"[1/3] Downloading SD v1.5 to {cache_dir} ...")
    try:
        StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        print("SD v1.5 done.")
    except Exception as e:
        print(f"SD v1.5 FAILED: {e}")
        failed.append("SD v1.5")

    print(f"[2/3] Downloading SDXL VAE fix to {cache_dir} ...")
    try:
        AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
        print("SDXL VAE fix done.")
    except Exception as e:
        print(f"SDXL VAE fix FAILED: {e}")
        failed.append("SDXL VAE fix")

    print(f"[3/3] Downloading SDXL base to {cache_dir} ...")
    try:
        StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        print("SDXL base done.")
    except Exception as e:
        print(f"SDXL base FAILED: {e}")
        failed.append("SDXL base")

    if failed:
        print(f"\nWARNING: {len(failed)} download(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
