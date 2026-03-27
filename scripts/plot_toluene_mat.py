#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SIGNAL_PATH = Path("data/raw/real_cars/Experimental BCARS data/Spectra toluene/toluene_1ms_cars.mat")
AXIS_PATH = Path("data/raw/real_cars/Experimental BCARS data/Spectra toluene/calibrated wavenumber.mat")


def load_data(signal_path: Path, axis_path: Path) -> tuple[np.ndarray, np.ndarray]:
    signal_mat = loadmat(signal_path)
    axis_mat = loadmat(axis_path)

    cars = np.asarray(signal_mat["carsMatrix"], dtype=float)   # shape (1, 100, 1340)
    wn = np.asarray(axis_mat["new_WN"], dtype=float).squeeze() # shape (1340,)

    # remove singleton dimension -> (100, 1340)
    cars = np.squeeze(cars, axis=0)

    # make axis increasing for easier plotting/processing
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        cars = cars[:, ::-1]

    return wn, cars


def plot_single_spectrum(wn: np.ndarray, cars: np.ndarray, idx: int = 0) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(wn, cars[idx])
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(f"Single toluene CARS spectrum (index {idx})")
    plt.tight_layout()
    plt.show()


def plot_multiple_spectra(wn: np.ndarray, cars: np.ndarray, n: int = 10) -> None:
    plt.figure(figsize=(8, 4))
    for i in range(min(n, cars.shape[0])):
        plt.plot(wn, cars[i], alpha=0.7)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(f"First {min(n, cars.shape[0])} toluene CARS spectra")
    plt.tight_layout()
    plt.show()


def plot_mean_spectrum(wn: np.ndarray, cars: np.ndarray) -> None:
    mean_spec = np.mean(cars, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(wn, mean_spec)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean intensity")
    plt.title("Mean toluene CARS spectrum")
    plt.tight_layout()
    plt.show()


def plot_heatmap(wn: np.ndarray, cars: np.ndarray) -> None:
    plt.figure(figsize=(9, 5))
    plt.imshow(
        cars,
        aspect="auto",
        extent=[wn[0], wn[-1], 0, cars.shape[0]],
        origin="lower",
    )
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Spectrum index")
    plt.title("Toluene CARS spectra heatmap")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()


def main() -> None:
    wn, cars = load_data(SIGNAL_PATH, AXIS_PATH)

    print("Wavenumber shape:", wn.shape)
    print("CARS shape:", cars.shape)
    print("Wavenumber min/max:", wn.min(), wn.max())

    plot_single_spectrum(wn, cars, idx=0)
    plot_multiple_spectra(wn, cars, n=10)
    plot_mean_spectrum(wn, cars)
    plot_heatmap(wn, cars)


if __name__ == "__main__":
    main()