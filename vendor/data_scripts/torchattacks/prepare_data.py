"""Prepare torchvision datasets and CIFAR model cache for torchattacks tasks."""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve() / "torchattacks"
    root.mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision

    torchvision.datasets.CIFAR10(root / "cifar10", train=True, download=True)
    torchvision.datasets.CIFAR10(root / "cifar10", train=False, download=True)
    torchvision.datasets.CIFAR100(root / "cifar100", train=True, download=True)
    torchvision.datasets.CIFAR100(root / "cifar100", train=False, download=True)
    torchvision.datasets.MNIST(root / "mnist", train=True, download=True)
    torchvision.datasets.MNIST(root / "mnist", train=False, download=True)

    cache = root / "torch_cache"
    cache.mkdir(parents=True, exist_ok=True)
    names = [
        "cifar10_resnet20",
        "cifar10_vgg11_bn",
        "cifar10_mobilenetv2_x1_0",
        "cifar100_resnet20",
        "cifar100_vgg11_bn",
        "cifar100_mobilenetv2_x1_0",
    ]
    old_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(cache / "hub"))
    try:
        for name in names:
            torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True, trust_repo=True)
    finally:
        torch.hub.set_dir(old_hub)

    checks = [
        root / "cifar10" / "cifar-10-batches-py",
        root / "cifar100" / "cifar-100-python",
        root / "mnist" / "MNIST",
        cache / "hub" / "checkpoints",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"Missing torchattacks data: {missing}", file=sys.stderr)
        sys.exit(1)
    print("torchattacks data ready")


if __name__ == "__main__":
    main()
