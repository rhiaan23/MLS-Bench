"""Prepare data for scikit-learn package.

Downloads sklearn/AIF360 datasets (OpenML + built-in + fairness benchmarks) to
the host data directory so they are available at runtime via bind mount
(compute nodes have no network).

Run via: mlsbench data scikit-learn

Creates:
  <data_root>/sklearn/  — sklearn/OpenML/AIF360 cache
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    data_home = Path(args.data_root) / "sklearn"
    data_home.mkdir(parents=True, exist_ok=True)
    data_home_str = str(data_home)

    print(f"Preparing sklearn datasets in {data_home_str}")

    # Check if OpenML datasets already exist (avoid sklearn import on envs without it)
    openml_dir = data_home / "openml" / "openml.org" / "data" / "v1"
    openml_populated = openml_dir.is_dir() and any(openml_dir.iterdir())

    # Download OpenML datasets
    from sklearn.datasets import fetch_openml, load_breast_cancer

    datasets = [
        ("mnist_784", 1),
        ("Fashion-MNIST", 1),
        ("madelon", 1),
    ]

    for name, version in datasets:
        if openml_populated:
            print(f"  OpenML cache populated, verifying {name} v{version}...", flush=True)
        else:
            print(f"  Fetching {name} v{version}...", flush=True)
        try:
            fetch_openml(name, version=version, data_home=data_home_str,
                         parser="auto", as_frame=False)
            print(f"    OK", flush=True)
        except Exception as e:
            print(f"    WARNING: {e}", flush=True)

    # Built-in datasets (no download needed, but call to verify)
    print("  Verifying built-in datasets...", flush=True)
    load_breast_cancer()
    print("    OK", flush=True)

    # Also fetch 20newsgroups
    from sklearn.datasets import fetch_20newsgroups
    print("  Fetching 20newsgroups...", flush=True)
    try:
        fetch_20newsgroups(subset='all', data_home=data_home_str)
        print("    OK", flush=True)
    except Exception as e:
        print(f"    WARNING: {e}", flush=True)

    # Fetch california housing
    from sklearn.datasets import fetch_california_housing
    print("  Fetching california_housing...", flush=True)
    try:
        fetch_california_housing(data_home=data_home_str)
        print("    OK", flush=True)
    except Exception as e:
        print(f"    WARNING: {e}", flush=True)

    # Fetch fairness / high-stakes tabular datasets used by subgroup calibration
    # and selective deferral tasks.
    print("  Fetching AIF360 fairness datasets...", flush=True)
    try:
        from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_lawschool_gpa

        fetch_adult(data_home=data_home_str, cache=True, binary_race=True, dropna=True)
        print("    adult OK", flush=True)
        fetch_compas(data_home=data_home_str, cache=True, binary_race=True, dropna=True)
        print("    compas OK", flush=True)
        fetch_lawschool_gpa(data_home=data_home_str, cache=True, binary_race=True, dropna=True)
        print("    law_school OK", flush=True)
    except Exception as e:
        print(f"    WARNING: could not fetch all AIF360 datasets: {e}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
