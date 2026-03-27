#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


SIGNAL_PATH = Path("data/raw/real_cars/Experimental BCARS data/Spectra toluene/toluene_1ms_cars.mat")
AXIS_PATH = Path("data/raw/real_cars/Experimental BCARS data/Spectra toluene/calibrated wavenumber.mat")
OUTPUT_DIR = Path("data/raw/real_cars/extracted_toluene")


def main() -> None:
    signal_mat = loadmat(SIGNAL_PATH)
    axis_mat = loadmat(AXIS_PATH)

    cars = np.asarray(signal_mat["carsMatrix"], dtype=float)
    wn = np.asarray(axis_mat["new_WN"], dtype=float).squeeze()

    cars = np.squeeze(cars, axis=0)  # (100, 1340)

    if wn[0] > wn[-1]:
        wn = wn[::-1]
        cars = cars[:, ::-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indices = [0, 10, 25, 50, 75]

    for idx in indices:
        df = pd.DataFrame({
            "x": wn,
            "y": cars[idx],
        })
        out_path = OUTPUT_DIR / f"toluene_1ms_cars_spec_{idx:03d}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()