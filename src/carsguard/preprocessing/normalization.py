from __future__ import annotations

from typing import Literal

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.core.exceptions import SpectrumValidationError


NormalizationMethod = Literal["none", "minmax", "max", "area", "zscore", "vector"]


def normalize_spectrum(
    spectrum: Spectrum,
    method: NormalizationMethod = "none",
) -> Spectrum:
    """
    Normalize spectrum intensity values.
    """
    y = spectrum.y.astype(float).copy()

    if method == "none":
        return spectrum.copy()

    elif method == "minmax":
        y_min = np.min(y)
        y_max = np.max(y)
        denom = y_max - y_min
        if np.isclose(denom, 0.0):
            raise SpectrumValidationError(
                f"Cannot min-max normalize flat spectrum '{spectrum.spectrum_id}'."
            )
        y = (y - y_min) / denom

    elif method == "max":
        y_max = np.max(np.abs(y))
        if np.isclose(y_max, 0.0):
            raise SpectrumValidationError(
                f"Cannot max normalize zero spectrum '{spectrum.spectrum_id}'."
            )
        y = y / y_max

    elif method == "area":
        area = np.trapz(np.abs(y), spectrum.x)
        if np.isclose(area, 0.0):
            raise SpectrumValidationError(
                f"Cannot area normalize zero-area spectrum '{spectrum.spectrum_id}'."
            )
        y = y / area

    elif method == "zscore":
        mean = np.mean(y)
        std = np.std(y)
        if np.isclose(std, 0.0):
            raise SpectrumValidationError(
                f"Cannot z-score normalize constant spectrum '{spectrum.spectrum_id}'."
            )
        y = (y - mean) / std

    elif method == "vector":
        norm = np.linalg.norm(y)
        if np.isclose(norm, 0.0):
            raise SpectrumValidationError(
                f"Cannot vector normalize zero spectrum '{spectrum.spectrum_id}'."
            )
        y = y / norm

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    metadata = dict(spectrum.metadata)
    metadata["normalization_method"] = method

    return spectrum.copy(y=y, metadata=metadata)