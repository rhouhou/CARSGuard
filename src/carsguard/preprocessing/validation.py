from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.core.exceptions import SpectrumValidationError


@dataclass
class SpectrumValidationReport:
    spectrum_id: str
    is_valid: bool
    has_nan: bool
    has_inf: bool
    x_is_monotonic: bool
    has_duplicate_x: bool
    n_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dynamic_range: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_spectrum(spectrum: Spectrum, raise_on_error: bool = True) -> SpectrumValidationReport:
    """
    Run basic structural checks on a spectrum.
    """
    x = spectrum.x
    y = spectrum.y

    has_nan = bool(np.isnan(x).any() or np.isnan(y).any())
    has_inf = bool(np.isinf(x).any() or np.isinf(y).any())
    x_diff = np.diff(x)
    x_is_monotonic = bool(np.all(x_diff > 0))
    has_duplicate_x = bool(np.any(x_diff == 0))

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dynamic_range = float(y_max - y_min)

    is_valid = (
        len(x) >= 2
        and not has_nan
        and not has_inf
        and x_is_monotonic
        and not has_duplicate_x
    )

    report = SpectrumValidationReport(
        spectrum_id=spectrum.spectrum_id,
        is_valid=is_valid,
        has_nan=has_nan,
        has_inf=has_inf,
        x_is_monotonic=x_is_monotonic,
        has_duplicate_x=has_duplicate_x,
        n_points=len(x),
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        y_min=y_min,
        y_max=y_max,
        dynamic_range=dynamic_range,
    )

    if raise_on_error and not is_valid:
        reasons = []
        if has_nan:
            reasons.append("contains NaN values")
        if has_inf:
            reasons.append("contains infinite values")
        if not x_is_monotonic:
            reasons.append("x-axis is not strictly increasing")
        if has_duplicate_x:
            reasons.append("x-axis contains duplicate values")

        reason_text = "; ".join(reasons) if reasons else "unknown validation error"
        raise SpectrumValidationError(
            f"Spectrum '{spectrum.spectrum_id}' failed validation: {reason_text}"
        )

    return report