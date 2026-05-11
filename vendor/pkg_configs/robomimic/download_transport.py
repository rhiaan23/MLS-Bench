"""Download robomimic Transport PH dataset to host (for data_bind mount)."""
import os
import shutil
from huggingface_hub import hf_hub_download

HOST_PATH = os.environ.get(
    "MLSBENCH_DATA_HOST_PATH",
    "vendor/data/robomimic/transport",
)
out_dir = os.path.join(HOST_PATH, "ph")
os.makedirs(out_dir, exist_ok=True)
fname = "low_dim_v15.hdf5"
out_path = os.path.join(out_dir, fname)

if not os.path.exists(out_path):
    print(f"Downloading transport/ph/{fname}...")
    tmp = hf_hub_download(
        repo_id="robomimic/robomimic_datasets",
        filename=f"v1.5/transport/ph/{fname}",
        repo_type="dataset",
    )
    # Resolve symlinks from HF cache
    shutil.copy2(os.path.realpath(tmp), out_path)
    print(f"  -> {out_path}")
else:
    print(f"Already exists: {out_path}")

print("Transport dataset ready")
