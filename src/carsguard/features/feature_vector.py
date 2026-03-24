from __future__ import annotations

from typing import Any, Dict

from carsguard.core.spectrum import Spectrum
from carsguard.features.background import extract_background_features
from carsguard.features.intensity import extract_intensity_features
from carsguard.features.morphology import extract_morphology_features
from carsguard.features.peaks import extract_peak_features
from carsguard.features.smoothness import extract_smoothness_features
from carsguard.features.widths import extract_width_features


def extract_feature_vector(
    spectrum: Spectrum,
    peak_min_prominence: float = 0.01,
    peak_min_distance: int = 3,
    background_window: int = 31,
) -> Dict[str, Any]:
    """
    Build a combined feature dictionary for a spectrum.
    """
    features: Dict[str, Any] = {
        "spectrum_id": spectrum.spectrum_id,
        "domain": spectrum.domain,
        "source_type": spectrum.source_type,
        "sample_class": spectrum.sample_class,
        "sample_name": spectrum.sample_name,
        "n_points": spectrum.n_points,
        "spectral_range_min": spectrum.spectral_range[0],
        "spectral_range_max": spectrum.spectral_range[1],
    }

    features.update(extract_intensity_features(spectrum))
    features.update(extract_smoothness_features(spectrum))
    features.update(extract_background_features(spectrum, background_window=background_window))
    features.update(extract_morphology_features(spectrum))
    features.update(
        extract_peak_features(
            spectrum,
            min_prominence=peak_min_prominence,
            min_distance=peak_min_distance,
        )
    )
    features.update(
        extract_width_features(
            spectrum,
            min_prominence=peak_min_prominence,
            min_distance=peak_min_distance,
        )
    )

    return features


def flatten_feature_vector(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Keep only numeric scalar features for statistics / nearest-neighbor use.
    """
    flat: Dict[str, float] = {}

    for key, value in features.items():
        if isinstance(value, bool):
            flat[key] = float(value)
        elif isinstance(value, (int, float)):
            flat[key] = float(value)

    return flat