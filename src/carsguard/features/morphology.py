from __future__ import annotations

from typing import Dict

import numpy as np

from carsguard.core.spectrum import Spectrum


def extract_morphology_features(spectrum: Spectrum) -> Dict[str, float]:
    """
    Extract broad shape descriptors for the spectrum.
    """
    x = spectrum.x
    y = spectrum.y.astype(float)

    abs_y = np.abs(y)
    total_mass = float(np.sum(abs_y))

    if np.isclose(total_mass, 0.0):
        center_of_mass = float(np.mean(x))
        spread = 0.0
    else:
        center_of_mass = float(np.sum(x * abs_y) / total_mass)
        spread = float(np.sqrt(np.sum(((x - center_of_mass) ** 2) * abs_y) / total_mass))

    # Estimate asymmetry around the center
    midpoint = (x.min() + x.max()) / 2.0
    left_mask = x < midpoint
    right_mask = x >= midpoint

    left_energy = float(np.sum(abs_y[left_mask])) if np.any(left_mask) else 0.0
    right_energy = float(np.sum(abs_y[right_mask])) if np.any(right_mask) else 0.0
    asymmetry = float((right_energy - left_energy) / (left_energy + right_energy + 1e-12))

    return {
        "center_of_mass": center_of_mass,
        "spectral_spread": spread,
        "left_energy": left_energy,
        "right_energy": right_energy,
        "left_right_asymmetry": asymmetry,
    }