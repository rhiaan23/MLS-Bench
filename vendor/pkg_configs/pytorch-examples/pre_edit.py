"""Pre-edit operations for pytorch-examples package.

Patches torch.accelerator calls (requires PyTorch >=2.6) to torch.cuda
equivalents for compatibility with PyTorch 2.4.x base image.
"""

OPS = [
    # Line 97: torch.accelerator.is_available() -> torch.cuda.is_available()
    {"op": "replace", "file": "pytorch-examples/mnist/main.py",
     "start_line": 97, "end_line": 97,
     "content": "    use_accel = not args.no_accel and torch.cuda.is_available()\n"},
    # Line 102: torch.accelerator.current_accelerator() -> torch.device("cuda")
    {"op": "replace", "file": "pytorch-examples/mnist/main.py",
     "start_line": 102, "end_line": 102,
     "content": '        device = torch.device("cuda")\n'},
    # Lines 120,122: Use pre-built MNIST data from image instead of downloading
    {"op": "replace", "file": "pytorch-examples/mnist/main.py",
     "start_line": 120, "end_line": 120,
     "content": "    dataset1 = datasets.MNIST('/data/mnist', train=True, download=False,\n"},
    {"op": "replace", "file": "pytorch-examples/mnist/main.py",
     "start_line": 122, "end_line": 122,
     "content": "    dataset2 = datasets.MNIST('/data/mnist', train=False,\n"},
]
