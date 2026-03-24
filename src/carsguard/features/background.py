from __future__ import annotations

from typing import Dict

import numpy as np

from carsguard.core.spectrum import Spectrum


def moving_average(y: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(y, kernel, mode="same")


def extract_background_features(
    spectrum: Spectrum,
    background_window: int = 31,
) -> Dict[str, float]:
    """
    Estimate how dominant the slow-varying background is relative to residual detail.
    """
    y = spectrum.y.astype(float)

    if background_window >= len(y):
        background_window = max(3, len(y) // 5)
        if background_window % 2 == 0:
            background_window += 1

    background = moving_average(y, background_window)
    residual = y - background

    background_std = float(np.std(background))
    residual_std = float(np.std(residual))
    background_energy = float(np.sum(background ** 2))
    residual_energy = float(np.sum(residual ** 2))
    dominance_ratio = float(background_std / (residual_std + 1e-12))

    return {
        "background_std": background_std,
        "residual_std": residual_std,
        "background_energy": background_energy,
        "residual_energy": residual_energy,
        "background_dominance_ratio": dominance_ratio,
    }