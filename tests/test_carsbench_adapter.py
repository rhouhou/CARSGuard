import numpy as np
from pathlib import Path

from carsguard.integration.carsbench_adapter import load_carsbench_spectrum, spectrum_from_carsbench_record


def test_carsbench_adapter(tmp_path: Path):
    file_path = tmp_path / "sim.csv"

    x = np.linspace(800, 1800, 100)
    y = np.cos(x / 100.0)

    data = np.column_stack([x, y])
    np.savetxt(file_path, data, delimiter=",")

    spectrum = load_carsbench_spectrum(
        file_path=file_path,
        domain="BCARS",
        simulation_metadata={"test_param": 1.0},
    )

    assert spectrum.source_type == "simulated_CARSBench"
    assert spectrum.metadata["producer"] == "CARSBench"
    assert spectrum.metadata["test_param"] == 1.0
    assert spectrum.n_points == 100


def test_carsbench_adapter_from_record(tmp_path: Path):
    file_path = tmp_path / "sim_record.csv"

    x = np.linspace(800, 1800, 60)
    y = np.sin(x / 120.0)
    np.savetxt(file_path, np.column_stack([x, y]), delimiter=",")

    spectrum = spectrum_from_carsbench_record(
        {
            "file_path": file_path,
            "spectrum_id": "sim_record",
            "domain": "BCARS",
            "sample_class": "lipid",
            "metadata": {"noise_level": 0.1},
        }
    )

    assert spectrum.spectrum_id == "sim_record"
    assert spectrum.sample_class == "lipid"
    assert spectrum.metadata["producer"] == "CARSBench"
    assert spectrum.metadata["noise_level"] == 0.1