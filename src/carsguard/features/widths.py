from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.features.peaks import find_local_peaks


def estimate_peak_widths(
    spectrum: Spectrum,
    peak_indices: np.ndarray,
    height_fraction: float = 0.5,
) -> np.ndarray:
    """
    Estimate peak widths at a given height fraction (default ~FWHM).
    """
    x = spectrum.x
    y = spectrum.y
    widths: List[float] = []

    for idx in peak_indices:
        peak_height = y[idx]
        level = peak_height * height_fraction

        # Search left
        left = idx
        while left > 0 and y[left] > level:
            left -= 1

        # Search right
        right = idx
        while right < len(y) - 1 and y[right] > level:
            right += 1

        width = x[right] - x[left] if right > left else 0.0
        widths.append(float(width))

    return np.asarray(widths, dtype=float)


def extract_width_features(
    spectrum: Spectrum,
    min_prominence: float = 0.01,
    min_distance: int = 3,
) -> Dict[str, Any]:
    """
    Extract width-related features from detected peaks.
    """
    peak_indices = find_local_peaks(
        y=spectrum.y,
        min_prominence=min_prominence,
        min_distance=min_distance,
    )

    if len(peak_indices) == 0:
        return {
            "mean_peak_width": 0.0,
            "std_peak_width": 0.0,
            "min_peak_width": 0.0,
            "max_peak_width": 0.0,
            "peak_widths": [],
        }

    widths = estimate_peak_widths(
        spectrum=spectrum,
        peak_indices=peak_indices,
        height_fraction=0.5,
    )

    return {
        "mean_peak_width": float(np.mean(widths)) if len(widths) > 0 else 0.0,
        "std_peak_width": float(np.std(widths)) if len(widths) > 0 else 0.0,
        "min_peak_width": float(np.min(widths)) if len(widths) > 0 else 0.0,
        "max_peak_width": float(np.max(widths)) if len(widths) > 0 else 0.0,
        "peak_widths": widths.tolist(),
    }