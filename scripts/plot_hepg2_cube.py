#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SIGNAL_PATH = Path("data/raw/real_cars/Experimental BCARS data/HepG2 cells/HepG2_cells_500nm_pixel_size_1ms_cars.mat")
AXIS_PATH = Path("data/raw/real_cars/Experimental BCARS data/HepG2 cells/wavenumber_calibrated.mat")


def load_cube(signal_path: Path, axis_path: Path) -> tuple[np.ndarray, np.ndarray]:
    signal_mat = loadmat(signal_path)
    axis_mat = loadmat(axis_path)

    cube = np.asarray(signal_mat["carsMatrix"], dtype=float)   # (200, 200, 1340)
    wn = np.asarray(axis_mat["new_WN"], dtype=float).squeeze() # (1340,)

    # Make spectral axis increasing
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        cube = cube[:, :, ::-1]

    return wn, cube


def plot_integrated_image(cube: np.ndarray) -> None:
    img = np.sum(cube, axis=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", origin="lower")
    plt.title("Integrated HepG2 BCARS image")
    plt.colorbar(label="Integrated intensity")
    plt.tight_layout()
    plt.show()


def plot_mean_spectrum(wn: np.ndarray, cube: np.ndarray) -> None:
    mean_spec = np.mean(cube, axis=(0, 1))

    plt.figure(figsize=(8, 4))
    plt.plot(wn, mean_spec)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean intensity")
    plt.title("Mean HepG2 BCARS spectrum")
    plt.tight_layout()
    plt.show()


def plot_center_pixel_spectrum(wn: np.ndarray, cube: np.ndarray) -> None:
    i = cube.shape[0] // 2
    j = cube.shape[1] // 2
    spec = cube[i, j, :]

    plt.figure(figsize=(8, 4))
    plt.plot(wn, spec)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(f"Center pixel spectrum ({i}, {j})")
    plt.tight_layout()
    plt.show()


def plot_random_pixel_spectra(wn: np.ndarray, cube: np.ndarray, n: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    plt.figure(figsize=(8, 4))
    for _ in range(n):
        i = rng.integers(0, cube.shape[0])
        j = rng.integers(0, cube.shape[1])
        plt.plot(wn, cube[i, j, :], alpha=0.8, label=f"({i},{j})")

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(f"{n} random HepG2 pixel spectra")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    wn, cube = load_cube(SIGNAL_PATH, AXIS_PATH)

    print("Cube shape:", cube.shape)
    print("Axis shape:", wn.shape)
    print("Wavenumber min/max:", wn.min(), wn.max())

    plot_integrated_image(cube)
    plot_mean_spectrum(wn, cube)
    plot_center_pixel_spectrum(wn, cube)
    plot_random_pixel_spectra(wn, cube, n=5, seed=42)


if __name__ == "__main__":
    main()