import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.features.feature_vector import extract_feature_vector, flatten_feature_vector


def make_test_spectrum():
    x = np.linspace(800, 1800, 300)
    y = (
        0.8 * np.exp(-0.5 * ((x - 950) / 20) ** 2)
        + 1.0 * np.exp(-0.5 * ((x - 1200) / 35) ** 2)
        + 0.6 * np.exp(-0.5 * ((x - 1600) / 25) ** 2)
        + 0.1
    )
    return Spectrum(
        spectrum_id="feat_test",
        x=x,
        y=y,
        domain="Raman",
        source_type="test",
    )


def test_feature_extraction_basic():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    assert "peak_count" in features
    assert "mean_peak_width" in features
    assert "background_dominance_ratio" in features
    assert features["n_points"] == 300
    assert features["spectral_range_min"] == 800.0
    assert features["spectral_range_max"] == 1800.0


def test_feature_values_are_numeric():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    numeric_keys = [
        "dynamic_range",
        "y_std",
        "first_derivative_std",
        "background_dominance_ratio",
        "center_of_mass",
        "spectral_spread",
    ]

    for key in numeric_keys:
        assert isinstance(features[key], float)


def test_flatten_feature_vector_keeps_scalar_numeric_values():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)
    flat = flatten_feature_vector(features)

    assert "dynamic_range" in flat
    assert "peak_count" in flat
    assert "n_points" in flat

    assert isinstance(flat["dynamic_range"], float)
    assert isinstance(flat["peak_count"], float)
    assert isinstance(flat["n_points"], float)

    # list-valued keys should not be present in the flattened output
    assert "peak_positions" not in flat
    assert "peak_heights" not in flat