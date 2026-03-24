import numpy as np
from pathlib import Path

from carsguard.io.loaders import load_spectrum


def test_load_csv_spectrum(tmp_path: Path):
    file_path = tmp_path / "test.csv"

    x = np.linspace(800, 1800, 50)
    y = np.sin(x / 100.0)

    data = np.column_stack([x, y])
    np.savetxt(file_path, data, delimiter=",")

    spectrum = load_spectrum(
        file_path=file_path,
        spectrum_id="test_spec",
        domain="Raman",
        source_type="test",
    )

    assert spectrum.n_points == 50
    assert spectrum.domain == "Raman"
    assert spectrum.source_type == "test"
    assert np.allclose(spectrum.x, x)
    assert np.allclose(spectrum.y, y)


def test_load_npy_spectrum(tmp_path: Path):
    file_path = tmp_path / "test.npy"

    x = np.linspace(800, 1800, 20)
    y = np.cos(x / 200.0)
    arr = np.column_stack([x, y])
    np.save(file_path, arr)

    spectrum = load_spectrum(
        file_path=file_path,
        spectrum_id="test_npy",
        domain="BCARS",
        source_type="test",
    )

    assert spectrum.n_points == 20
    assert spectrum.domain == "BCARS"
    assert np.allclose(spectrum.x, x)
    assert np.allclose(spectrum.y, y)