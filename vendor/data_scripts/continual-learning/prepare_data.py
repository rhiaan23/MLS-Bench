"""Prepare data for continual-learning package.

Downloads MNIST and CIFAR-100 datasets to host data directory so they are
available at runtime via bind mount (compute nodes have no network).

Run via: mlsbench data continual-learning

Creates:
  <data_root>/continual-learning/MNIST/       — MNIST dataset
  <data_root>/continual-learning/CIFAR100/    — CIFAR-100 dataset
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    # The code uses store="/data/continual-learning" and then
    # data_dir="{store}/datasets", with dataset paths "{data_dir}/{name}"
    # e.g. /data/continual-learning/datasets/MNIST
    # On the host, /data/continual-learning maps to {data_root}/continual-learning
    data_dir = Path(args.data_root) / "continual-learning" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing continual-learning datasets in {data_dir}")

    # Check if datasets already exist (avoid torchvision import on envs without it)
    mnist_ready = (data_dir / "MNIST" / "MNIST" / "raw" / "train-images-idx3-ubyte").exists()
    cifar_ready = (data_dir / "CIFAR100" / "cifar-100-python").is_dir()
    if mnist_ready and cifar_ready:
        print("  All datasets already present, skipping download.")
        print("Done.")
        return

    import torchvision

    # torchvision downloads into {root}/MNIST/raw/ and {root}/CIFAR100/
    # The code passes "{data_dir}/MNIST" as root, so we download to data_dir/MNIST
    mnist_dir = data_dir / "MNIST"
    mnist_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading MNIST...", flush=True)
    torchvision.datasets.MNIST(str(mnist_dir), download=True)
    print("    OK", flush=True)

    cifar_dir = data_dir / "CIFAR100"
    cifar_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading CIFAR-100...", flush=True)
    torchvision.datasets.CIFAR100(str(cifar_dir), download=True)
    print("    OK", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
