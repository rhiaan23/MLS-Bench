#!/usr/bin/env python3
"""Prepare VICON datasets via the package shell script."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    script = project_root / "vendor" / "pkg_configs" / "VICON" / "prepare_data.sh"
    cmd = ["bash", str(script), args.data_root]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
