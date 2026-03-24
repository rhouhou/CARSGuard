from __future__ import annotations

from typing import Dict

import numpy as np

from carsguard.core.spectrum import Spectrum


def extract_intensity_features(spectrum: Spectrum) -> Dict[str, float]:
    """
    Extract basic intensity-distribution features.
    """
    y = spectrum.y.astype(float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_median = float(np.median(y))
    y_q05 = float(np.quantile(y, 0.05))
    y_q25 = float(np.quantile(y, 0.25))
    y_q75 = float(np.quantile(y, 0.75))
    y_q95 = float(np.quantile(y, 0.95))
    dynamic_range = float(y_max - y_min)

    max_abs = float(np.max(np.abs(y)))
    if np.isclose(max_abs, 0.0):
        max_to_mean_abs_ratio = 0.0
    else:
        max_to_mean_abs_ratio = float(max_abs / (np.mean(np.abs(y)) + 1e-12))

    return {
        "y_min": y_min,
        "y_max": y_max,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_median": y_median,
        "y_q05": y_q05,
        "y_q25": y_q25,
        "y_q75": y_q75,
        "y_q95": y_q95,
        "dynamic_range": dynamic_range,
        "max_abs_intensity": max_abs,
        "max_to_mean_abs_ratio": max_to_mean_abs_ratio,
    }