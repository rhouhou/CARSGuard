import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.preprocessing.axis import crop_spectrum, resample_spectrum
from carsguard.preprocessing.baseline import subtract_baseline
from carsguard.preprocessing.filtering import smooth_spectrum
from carsguard.preprocessing.normalization import normalize_spectrum


def make_test_spectrum():
    x = np.linspace(800, 1800, 200)
    y = np.sin((x - 800) / 80.0) + 1.5
    return Spectrum(
        spectrum_id="test",
        x=x,
        y=y,
        domain="Raman",
        source_type="test",
    )


def test_crop_spectrum():
    spec = make_test_spectrum()
    cropped = crop_spectrum(spec, x_min=900, x_max=1500)

    assert cropped.x[0] >= 900
    assert cropped.x[-1] <= 1500
    assert cropped.n_points < spec.n_points


def test_resample():
    spec = make_test_spectrum()
    new_x = np.linspace(900, 1500, 50)

    cropped = crop_spectrum(spec, x_min=900, x_max=1500)
    resampled = resample_spectrum(cropped, new_x=new_x)

    assert resampled.n_points == 50
    assert np.isclose(resampled.x[0], 900.0)
    assert np.isclose(resampled.x[-1], 1500.0)


def test_normalization_max():
    spec = make_test_spectrum()
    norm = normalize_spectrum(spec, method="max")

    assert np.isclose(np.max(np.abs(norm.y)), 1.0)


def test_smoothing_preserves_length():
    spec = make_test_spectrum()
    smooth = smooth_spectrum(spec, method="moving_average", window_size=5)

    assert smooth.n_points == spec.n_points


def test_baseline_subtraction_preserves_length():
    spec = make_test_spectrum()
    result = subtract_baseline(spec, method="moving_minimum", window_size=11)

    assert result.corrected_spectrum.n_points == spec.n_points
    assert len(result.baseline) == spec.n_points