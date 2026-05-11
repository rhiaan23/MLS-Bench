"""Download robomimic datasets from HuggingFace."""
import os
import shutil
from huggingface_hub import hf_hub_download

REPO = "robomimic/robomimic_datasets"
BASE = "/data/datasets"

for task in ["lift", "can", "square"]:
    out_dir = os.path.join(BASE, task, "ph")
    os.makedirs(out_dir, exist_ok=True)
    fname = "low_dim_v15.hdf5"
    out_path = os.path.join(out_dir, fname)
    if not os.path.exists(out_path):
        print(f"Downloading {task}/ph/{fname}...")
        tmp = hf_hub_download(
            repo_id=REPO,
            filename=f"v1.5/{task}/ph/{fname}",
            repo_type="dataset",
        )
        # hf_hub_download returns a symlink to a cached blob; copy the
        # real file so the container image doesn't contain dangling links.
        shutil.copy2(os.path.realpath(tmp), out_path)
        print(f"  -> {out_path}")
    else:
        print(f"Already exists: {out_path}")

print("Datasets downloaded OK")
