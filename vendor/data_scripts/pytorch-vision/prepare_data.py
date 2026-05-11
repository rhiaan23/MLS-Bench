"""Prepare CIFAR and FashionMNIST data for pytorch-vision tasks."""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve() / "pytorch-vision"
    root.mkdir(parents=True, exist_ok=True)

    import torchvision

    torchvision.datasets.CIFAR10(root / "cifar", train=True, download=True)
    torchvision.datasets.CIFAR10(root / "cifar", train=False, download=True)
    torchvision.datasets.CIFAR100(root / "cifar", train=True, download=True)
    torchvision.datasets.CIFAR100(root / "cifar", train=False, download=True)
    torchvision.datasets.FashionMNIST(root / "fmnist", train=True, download=True)
    torchvision.datasets.FashionMNIST(root / "fmnist", train=False, download=True)

    checks = [
        root / "cifar" / "cifar-10-batches-py",
        root / "cifar" / "cifar-100-python",
        root / "fmnist" / "FashionMNIST",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"Missing pytorch-vision data: {missing}", file=sys.stderr)
        sys.exit(1)
    print("pytorch-vision data ready")


if __name__ == "__main__":
    main()
