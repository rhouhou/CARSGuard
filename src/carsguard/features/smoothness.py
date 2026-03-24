from __future__ import annotations

from typing import Dict

import numpy as np

from carsguard.core.spectrum import Spectrum


def extract_smoothness_features(spectrum: Spectrum) -> Dict[str, float]:
    """
    Extract roughness / smoothness related features using derivatives.
    """
    x = spectrum.x
    y = spectrum.y

    if len(y) < 3:
        return {
            "first_derivative_std": 0.0,
            "second_derivative_std": 0.0,
            "total_variation": 0.0,
            "roughness_ratio": 0.0,
        }

    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    total_variation = float(np.sum(np.abs(np.diff(y))))
    y_std = float(np.std(y))
    roughness_ratio = float(np.std(dy) / (y_std + 1e-12))

    return {
        "first_derivative_std": float(np.std(dy)),
        "second_derivative_std": float(np.std(d2y)),
        "total_variation": total_variation,
        "roughness_ratio": roughness_ratio,
    }