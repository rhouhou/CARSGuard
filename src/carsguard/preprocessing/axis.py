from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.core.exceptions import SpectrumValidationError


InterpolationMethod = Literal["linear"]


@dataclass
class AxisGrid:
    start: float
    stop: float
    num_points: int

    def to_array(self) -> np.ndarray:
        if self.num_points < 2:
            raise ValueError("AxisGrid.num_points must be at least 2.")
        return np.linspace(self.start, self.stop, self.num_points, dtype=float)


def is_monotonic_increasing(x: np.ndarray) -> bool:
    return np.all(np.diff(x) > 0)


def sort_spectrum_by_axis(spectrum: Spectrum) -> Spectrum:
    """
    Sort spectrum points by ascending x.
    """
    order = np.argsort(spectrum.x)
    return spectrum.copy(
        x=spectrum.x[order],
        y=spectrum.y[order],
    )


def crop_spectrum(
    spectrum: Spectrum,
    x_min: float | None = None,
    x_max: float | None = None,
) -> Spectrum:
    """
    Crop a spectrum to a specified x-range.
    """
    x = spectrum.x
    y = spectrum.y

    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max

    if mask.sum() < 2:
        raise SpectrumValidationError(
            f"Cropping spectrum '{spectrum.spectrum_id}' leaves fewer than 2 points."
        )

    return spectrum.copy(x=x[mask], y=y[mask])


def resample_spectrum(
    spectrum: Spectrum,
    new_x: np.ndarray,
    method: InterpolationMethod = "linear",
) -> Spectrum:
    """
    Resample a spectrum onto a new x-axis using interpolation.
    """
    if method != "linear":
        raise ValueError(f"Unsupported interpolation method: {method}")

    x = spectrum.x
    y = spectrum.y

    if not is_monotonic_increasing(x):
        spectrum = sort_spectrum_by_axis(spectrum)
        x = spectrum.x
        y = spectrum.y

    eps = 1e-9
    if new_x.min() < x.min() - eps or new_x.max() > x.max() + eps:
        raise SpectrumValidationError(
            f"New axis [{new_x.min():.3f}, {new_x.max():.3f}] exceeds "
            f"original spectrum range [{x.min():.3f}, {x.max():.3f}] "
            f"for spectrum '{spectrum.spectrum_id}'."
        )

    new_y = np.interp(new_x, x, y)

    metadata = dict(spectrum.metadata)
    metadata["resampled"] = True
    metadata["resampled_num_points"] = int(len(new_x))

    return spectrum.copy(x=new_x, y=new_y, metadata=metadata)


def build_common_grid(
    spectra: list[Spectrum],
    num_points: int,
    mode: Literal["intersection", "union"] = "intersection",
) -> np.ndarray:
    """
    Build a common x-grid across multiple spectra.

    intersection:
        uses the overlapping x-range of all spectra
    union:
        uses the full combined x-range of all spectra
    """
    if not spectra:
        raise ValueError("Cannot build common grid from an empty spectrum list.")

    mins = [float(np.min(s.x)) for s in spectra]
    maxs = [float(np.max(s.x)) for s in spectra]

    if mode == "intersection":
        start = max(mins)
        stop = min(maxs)
        if stop <= start:
            raise SpectrumValidationError(
                "No overlapping spectral range across spectra."
            )
    elif mode == "union":
        start = min(mins)
        stop = max(maxs)
    else:
        raise ValueError(f"Unsupported grid mode: {mode}")

    return AxisGrid(start=start, stop=stop, num_points=num_points).to_array()