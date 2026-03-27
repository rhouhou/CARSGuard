#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def summarize_value(name: str, value) -> None:
    if name.startswith("__"):
        return

    print(f"\nVariable: {name}")
    print(f"  Type: {type(value)}")

    if isinstance(value, np.ndarray):
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")

        if np.issubdtype(value.dtype, np.number) and value.size > 0:
            flat = value.ravel()
            print(f"  Min: {np.nanmin(flat):.6g}")
            print(f"  Max: {np.nanmax(flat):.6g}")
            print(f"  Mean: {np.nanmean(flat):.6g}")

            preview = flat[:10]
            print(f"  First values: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect contents of a MATLAB .mat file")
    parser.add_argument("mat_file", type=Path)
    args = parser.parse_args()

    data = loadmat(args.mat_file)

    print(f"File: {args.mat_file}")
    print(f"Keys: {list(data.keys())}")

    for key, value in data.items():
        summarize_value(key, value)


if __name__ == "__main__":
    main()