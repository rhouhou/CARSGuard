#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


SIGNAL_PATH = Path("data/raw/real_cars/Experimental BCARS data/HepG2 cells/HepG2_cells_500nm_pixel_size_1ms_cars.mat")
AXIS_PATH = Path("data/raw/real_cars/Experimental BCARS data/HepG2 cells/wavenumber_calibrated.mat")
OUTPUT_DIR = Path("data/raw/real_cars/extracted_hepg2")


def main() -> None:
    signal_mat = loadmat(SIGNAL_PATH)
    axis_mat = loadmat(AXIS_PATH)

    cube = np.asarray(signal_mat["carsMatrix"], dtype=float)
    wn = np.asarray(axis_mat["new_WN"], dtype=float).squeeze()

    if wn[0] > wn[-1]:
        wn = wn[::-1]
        cube = cube[:, :, ::-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Choose some representative pixels manually for now
    selected_pixels = [
        (50, 50),
        (75, 100),
        (100, 100),
        (125, 80),
        (150, 150),
    ]

    for i, j in selected_pixels:
        spec = cube[i, j, :]
        df = pd.DataFrame({
            "x": wn,
            "y": spec,
        })

        out_path = OUTPUT_DIR / f"hepg2_pixel_{i}_{j}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()