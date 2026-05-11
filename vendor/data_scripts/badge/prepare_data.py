"""Prepare OpenML datasets for the badge active-learning task."""

import argparse
import pickle
from pathlib import Path


DATASET_IDS = (6, 44, 46)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "badge" / "oml"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openml
    except ImportError as exc:
        raise RuntimeError(
            "Preparing badge data requires the openml Python package on the host."
        ) from exc

    for did in DATASET_IDS:
        out_path = out_dir / f"data_{did}.pk"
        if out_path.exists():
            print(f"OpenML-{did} already prepared at {out_path}", flush=True)
            continue

        print(f"Fetching OpenML-{did}...", flush=True)
        dataset = openml.datasets.get_dataset(did)
        x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        with out_path.open("wb") as fh:
            pickle.dump({"data": (x.values, y.values)}, fh)
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
