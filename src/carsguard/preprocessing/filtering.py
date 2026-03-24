from __future__ import annotations

from typing import Literal

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.core.exceptions import SpectrumValidationError


FilterMethod = Literal["none", "moving_average"]


def smooth_spectrum(
    spectrum: Spectrum,
    method: FilterMethod = "none",
    window_size: int = 5,
) -> Spectrum:
    """
    Apply light smoothing to a spectrum.

    V1 supports:
    - none
    - moving_average
    """
    if method == "none":
        return spectrum.copy()

    if method != "moving_average":
        raise ValueError(f"Unsupported filter method: {method}")

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered moving average.")

    if window_size >= len(spectrum.y):
        raise SpectrumValidationError(
            f"window_size={window_size} is too large for spectrum "
            f"'{spectrum.spectrum_id}' with {len(spectrum.y)} points."
        )

    kernel = np.ones(window_size, dtype=float) / window_size
    y_smooth = np.convolve(spectrum.y, kernel, mode="same")

    metadata = dict(spectrum.metadata)
    metadata["filter_method"] = method
    metadata["filter_window_size"] = window_size

    return spectrum.copy(y=y_smooth, metadata=metadata)