from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from carsguard.core.spectrum import Spectrum
from carsguard.io.loaders import load_spectrum


def load_carsbench_spectrum(
    file_path: str | Path,
    spectrum_id: Optional[str] = None,
    domain: str = "BCARS",
    sample_class: Optional[str] = None,
    sample_name: Optional[str] = None,
    preprocessing_status: Optional[str] = None,
    simulation_metadata: Optional[Dict[str, Any]] = None,
    x_col: int | str = 0,
    y_col: int | str = 1,
    delimiter: Optional[str] = None,
) -> Spectrum:
    """
    Load a spectrum produced by CARSBench into the internal Spectrum format.

    This is intentionally generic for V1. Later you can extend it to parse:
    - simulation domain
    - generator settings
    - NRB settings
    - noise settings
    - phase settings
    - recovery settings
    """
    file_path = Path(file_path)

    if spectrum_id is None:
        spectrum_id = file_path.stem

    metadata = simulation_metadata.copy() if simulation_metadata is not None else {}
    metadata["producer"] = "CARSBench"

    return load_spectrum(
        file_path=file_path,
        spectrum_id=spectrum_id,
        domain=domain,
        source_type="simulated_CARSBench",
        sample_class=sample_class,
        sample_name=sample_name,
        preprocessing_status=preprocessing_status,
        metadata=metadata,
        x_col=x_col,
        y_col=y_col,
        delimiter=delimiter,
    )


def spectrum_from_carsbench_record(record: Dict[str, Any]) -> Spectrum:
    """
    Convenience loader from a plain dictionary describing a CARSBench output.

    Expected keys:
    - file_path
    Optional keys:
    - spectrum_id
    - domain
    - sample_class
    - sample_name
    - preprocessing_status
    - metadata
    - x_col
    - y_col
    - delimiter
    """
    return load_carsbench_spectrum(
        file_path=record["file_path"],
        spectrum_id=record.get("spectrum_id"),
        domain=record.get("domain", "BCARS"),
        sample_class=record.get("sample_class"),
        sample_name=record.get("sample_name"),
        preprocessing_status=record.get("preprocessing_status"),
        simulation_metadata=record.get("metadata"),
        x_col=record.get("x_col", 0),
        y_col=record.get("y_col", 1),
        delimiter=record.get("delimiter"),
    )