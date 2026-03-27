#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def find_numeric_arrays(mat_dict: dict) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arrays[key] = value
    return arrays


def choose_array(arrays: dict[str, np.ndarray], preferred_names: list[str]) -> Optional[tuple[str, np.ndarray]]:
    lowered = {k.lower(): k for k in arrays.keys()}
    for pref in preferred_names:
        if pref.lower() in lowered:
            real_key = lowered[pref.lower()]
            return real_key, arrays[real_key]
    return None


def squeeze_vector(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector after squeeze, got shape {arr.shape}")
    return arr.astype(float)


def load_wavenumber_axis(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    arrays = find_numeric_arrays(data)

    candidate = choose_array(
        arrays,
        preferred_names=[
            "wavenumber_calibrated",
            "calibrated_wavenumber",
            "wavenumber",
            "wn",
            "axis",
        ],
    )

    if candidate is None:
        # fallback: choose the smallest 1D numeric array
        one_d = [(k, v) for k, v in arrays.items() if np.asarray(v).squeeze().ndim == 1]
        if not one_d:
            raise ValueError(f"No 1D numeric axis found in {mat_path}")
        key, arr = sorted(one_d, key=lambda kv: np.asarray(kv[1]).size)[0]
        print(f"[INFO] Using fallback axis variable: {key}")
        return squeeze_vector(arr)

    key, arr = candidate
    print(f"[INFO] Using axis variable: {key}")
    return squeeze_vector(arr)


def load_signal_array(mat_path: Path) -> tuple[str, np.ndarray]:
    data = loadmat(mat_path)
    arrays = find_numeric_arrays(data)

    candidate = choose_array(
        arrays,
        preferred_names=[
            "carsMatrix",
            "carsmatrix",
            "cars",
            "spectrum",
            "spectra",
            "nrb",
            "dark",
        ],
    )

    if candidate is not None:
        key, arr = candidate
        print(f"[INFO] Using signal variable: {key}")
        return key, np.asarray(arr)

    # fallback: choose the largest numeric array
    key, arr = sorted(arrays.items(), key=lambda kv: np.asarray(kv[1]).size, reverse=True)[0]
    print(f"[INFO] Using fallback signal variable: {key}")
    return key, np.asarray(arr)


def plot_vector(x: np.ndarray, y: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_cube_summary(cube: np.ndarray, axis: Optional[np.ndarray], title_prefix: str) -> None:
    cube = np.asarray(cube).astype(float)

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    # README says first row is saturated in images
    image = np.sum(cube[:, 1:, :], axis=2) if cube.shape[1] > 1 else np.sum(cube, axis=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(image, aspect="equal", cmap="gray")
    plt.title(f"{title_prefix} - integrated image")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    mean_spectrum = np.mean(cube, axis=(0, 1))
    if axis is not None and len(axis) == len(mean_spectrum):
        plot_vector(axis, mean_spectrum, f"{title_prefix} - mean spectrum")
    else:
        plot_vector(np.arange(len(mean_spectrum)), mean_spectrum, f"{title_prefix} - mean spectrum (index axis)")

    center_i = cube.shape[0] // 2
    center_j = cube.shape[1] // 2
    pixel_spectrum = cube[center_i, center_j, :]

    if axis is not None and len(axis) == len(pixel_spectrum):
        plot_vector(axis, pixel_spectrum, f"{title_prefix} - center pixel spectrum")
    else:
        plot_vector(np.arange(len(pixel_spectrum)), pixel_spectrum, f"{title_prefix} - center pixel spectrum (index axis)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spectra or hyperspectral cubes from BCARS .mat files")
    parser.add_argument("signal_mat", type=Path, help="Path to signal .mat file")
    parser.add_argument("--axis-mat", type=Path, default=None, help="Path to wavenumber axis .mat file")
    args = parser.parse_args()

    axis = None
    if args.axis_mat is not None:
        axis = load_wavenumber_axis(args.axis_mat)

    var_name, signal = load_signal_array(args.signal_mat)
    signal = np.asarray(signal).squeeze()

    print(f"[INFO] Final signal shape after squeeze: {signal.shape}")

    if signal.ndim == 1:
        x = axis if axis is not None and len(axis) == len(signal) else np.arange(len(signal))
        plot_vector(x, signal.astype(float), f"{args.signal_mat.name} - {var_name}")

    elif signal.ndim == 2:
        # could be [n_spectra, n_channels] or [n_channels, n_spectra]
        arr = signal.astype(float)

        plt.figure(figsize=(6, 4))
        plt.imshow(arr, aspect="auto")
        plt.title(f"{args.signal_mat.name} - matrix view")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        if axis is not None:
            if arr.shape[1] == len(axis):
                mean_spectrum = arr.mean(axis=0)
                plot_vector(axis, mean_spectrum, f"{args.signal_mat.name} - mean spectrum")
            elif arr.shape[0] == len(axis):
                mean_spectrum = arr.mean(axis=1)
                plot_vector(axis, mean_spectrum, f"{args.signal_mat.name} - mean spectrum")
            else:
                print("[WARN] Axis length does not match either matrix dimension.")
        else:
            print("[INFO] No axis provided; matrix shown only.")

    elif signal.ndim == 3:
        plot_cube_summary(signal, axis, args.signal_mat.name)

    else:
        raise ValueError(f"Unsupported signal dimensionality: {signal.ndim}")


if __name__ == "__main__":
    main()