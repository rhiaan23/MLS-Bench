"""Prepare katakomba memmap cache on the host filesystem.

Runs inside the katakomba container to decompress HDF5 datasets into
numpy memmap files. The resulting cache directory is bind-mounted into
containers at runtime, avoiding tmpfs overflow from on-the-fly decompression.
"""
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: prepare_cache.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the container image
    project_root = Path(__file__).resolve().parents[3]
    image = project_root / "vendor" / "images" / "katakomba.sif"
    if not image.exists():
        print(f"Container image not found: {image}")
        print("Run 'mlsbench build katakomba' first.")
        sys.exit(1)

    # Run memmap pre-heat inside the container, writing cache to bind-mounted host dir
    script = (
        "from katakomba.utils.roles import Role, Race, Alignment; "
        "from katakomba.utils.datasets.small_scale import load_nld_aa_small_dataset; "
        "import os; "
        "os.environ['KATAKOMBA_CACHE_DIR'] = '/katakomba-cache'; "
        "[load_nld_aa_small_dataset(Role(r), Race('hum'), Alignment(a), mode='memmap')[0].close() "
        "for r, a in [('mon','neu'),('val','neu'),('ran','neu')]]; "
        "print('Memmap cache ready')"
    )

    cmd = [
        "apptainer", "exec", "--writable-tmpfs", "--no-home",
        "--env", f"HOME=/root,KATAKOMBA_DATA_DIR=/root/.katakomba/datasets,"
                 f"KATAKOMBA_CACHE_DIR=/katakomba-cache,"
                 f"PYTHONPATH=/workspace/katakomba",
        "--bind", f"{output_dir}:/katakomba-cache",
        str(image),
        "python3", "-c", script,
    ]

    print(f"Generating memmap cache in {output_dir} ...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Cache generation failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"Cache ready at {output_dir}")


if __name__ == "__main__":
    main()
