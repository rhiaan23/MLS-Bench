"""Download robomimic low-dim datasets for Lift, Can, Square from HuggingFace."""
import os
import urllib.request

HF_BASE = "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5"

DATASETS = {
    "lift": {
        "url": f"{HF_BASE}/lift/ph/low_dim_v15.hdf5",
        "out": "/data/robomimic/datasets/lift/ph/low_dim_abs.hdf5",
    },
    "can": {
        "url": f"{HF_BASE}/can/ph/low_dim_v15.hdf5",
        "out": "/data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
    },
    "square": {
        "url": f"{HF_BASE}/square/ph/low_dim_v15.hdf5",
        "out": "/data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
    },
}


def main():
    for task, info in DATASETS.items():
        out_path = info["out"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            print(f"Already exists: {out_path}")
            continue
        print(f"Downloading {task} -> {out_path} ...")
        urllib.request.urlretrieve(info["url"], out_path)
        print(f"  Done: {out_path}")

    print("All robomimic datasets ready.")


if __name__ == "__main__":
    main()
