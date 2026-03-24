from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.core.exceptions import SpectrumValidationError


BaselineMethod = Literal["none", "moving_minimum"]


@dataclass
class BaselineResult:
    corrected_spectrum: Spectrum
    baseline: np.ndarray


def estimate_baseline(
    spectrum: Spectrum,
    method: BaselineMethod = "none",
    window_size: int = 31,
) -> np.ndarray:
    """
    Estimate a simple baseline.

    V1 method:
    - none: zero baseline
    - moving_minimum: local minimum baseline estimate
    """
    y = spectrum.y

    if method == "none":
        return np.zeros_like(y)

    if method != "moving_minimum":
        raise ValueError(f"Unsupported baseline method: {method}")

    if window_size < 3:
        raise ValueError("window_size must be >= 3")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for moving minimum.")

    if window_size >= len(y):
        raise SpectrumValidationError(
            f"Baseline window_size={window_size} is too large for spectrum "
            f"'{spectrum.spectrum_id}'."
        )

    half = window_size // 2
    baseline = np.empty_like(y, dtype=float)

    for i in range(len(y)):
        start = max(0, i - half)
        stop = min(len(y), i + half + 1)
        baseline[i] = np.min(y[start:stop])

    return baseline


def subtract_baseline(
    spectrum: Spectrum,
    method: BaselineMethod = "none",
    window_size: int = 31,
    clip_zero: bool = False,
) -> BaselineResult:
    """
    Estimate and subtract a baseline.
    """
    baseline = estimate_baseline(
        spectrum=spectrum,
        method=method,
        window_size=window_size,
    )

    corrected = spectrum.y - baseline
    if clip_zero:
        corrected = np.maximum(corrected, 0.0)

    metadata = dict(spectrum.metadata)
    metadata["baseline_method"] = method
    metadata["baseline_window_size"] = window_size
    metadata["baseline_subtracted"] = method != "none"

    corrected_spectrum = spectrum.copy(y=corrected, metadata=metadata)
    return BaselineResult(corrected_spectrum=corrected_spectrum, baseline=baseline)