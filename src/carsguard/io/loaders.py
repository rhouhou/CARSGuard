from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from carsguard.core.dataset import SpectrumRecord
from carsguard.core.exceptions import SpectrumValidationError
from carsguard.core.spectrum import Spectrum


SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".npy", ".npz"}


def load_spectrum(
    file_path: str | Path,
    spectrum_id: str,
    domain: str,
    source_type: str,
    sample_class: Optional[str] = None,
    sample_name: Optional[str] = None,
    preprocessing_status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    x_col: int | str = 0,
    y_col: int | str = 1,
    delimiter: Optional[str] = None,
) -> Spectrum:
    """
    Load a spectrum from disk and return a Spectrum object.

    Supported formats:
    - CSV / TSV / TXT with two columns
    - NPY with shape (N, 2)
    - NPZ containing arrays named 'x' and 'y'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise SpectrumValidationError(f"Spectrum file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise SpectrumValidationError(
            f"Unsupported spectrum file format: {suffix}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix in {".csv", ".txt", ".tsv"}:
        x, y = _load_text_spectrum(file_path, x_col=x_col, y_col=y_col, delimiter=delimiter)
    elif suffix == ".npy":
        x, y = _load_npy_spectrum(file_path)
    elif suffix == ".npz":
        x, y = _load_npz_spectrum(file_path)
    else:
        raise SpectrumValidationError(f"Unsupported spectrum file format: {suffix}")

    spectrum = Spectrum(
        spectrum_id=spectrum_id,
        x=x,
        y=y,
        domain=domain,
        source_type=source_type,
        sample_class=sample_class,
        sample_name=sample_name,
        preprocessing_status=preprocessing_status,
        metadata=metadata or {},
    )

    return spectrum


def load_spectrum_from_record(
    record: SpectrumRecord,
    base_dir: str | Path | None = None,
    x_col: int | str = 0,
    y_col: int | str = 1,
    delimiter: Optional[str] = None,
) -> Spectrum:
    """
    Load a Spectrum object from a SpectrumRecord.
    """
    path = Path(record.file_path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path

    metadata = dict(record.metadata)
    metadata["paired_to_id"] = record.paired_to_id
    metadata["notes"] = record.notes
    metadata["x_axis_type"] = record.x_axis_type

    return load_spectrum(
        file_path=path,
        spectrum_id=record.spectrum_id,
        domain=record.domain,
        source_type=record.source_type,
        sample_class=record.sample_class,
        sample_name=record.sample_name,
        preprocessing_status=record.preprocessing_status,
        metadata=metadata,
        x_col=x_col,
        y_col=y_col,
        delimiter=delimiter,
    )


def _load_text_spectrum(
    file_path: Path,
    x_col: int | str = 0,
    y_col: int | str = 1,
    delimiter: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a 2-column spectrum from text/CSV/TSV.
    """
    if delimiter is None:
        if file_path.suffix.lower() == ".tsv":
            delimiter = "\t"
        elif file_path.suffix.lower() == ".csv":
            delimiter = ","
        else:
            delimiter = None  # pandas auto-detect engine='python' with sep=None

    try:
        if delimiter is None:
            df = pd.read_csv(file_path, sep=None, engine="python", comment="#", header=None)
        else:
            df = pd.read_csv(file_path, sep=delimiter, comment="#", header=None)
    except Exception as exc:
        raise SpectrumValidationError(f"Failed to read text spectrum: {file_path}") from exc

    if df.shape[1] < 2:
        raise SpectrumValidationError(
            f"Text spectrum must contain at least two columns: {file_path}"
        )

    try:
        x = df.iloc[:, x_col].to_numpy(dtype=float) if isinstance(x_col, int) else df[x_col].to_numpy(dtype=float)
        y = df.iloc[:, y_col].to_numpy(dtype=float) if isinstance(y_col, int) else df[y_col].to_numpy(dtype=float)
    except Exception as exc:
        raise SpectrumValidationError(
            f"Failed to extract x/y columns from spectrum file: {file_path}"
        ) from exc

    _validate_loaded_arrays(x, y, file_path)
    return x, y


def _load_npy_spectrum(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a spectrum from .npy with expected shape (N, 2).
    """
    try:
        arr = np.load(file_path)
    except Exception as exc:
        raise SpectrumValidationError(f"Failed to read NPY spectrum: {file_path}") from exc

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise SpectrumValidationError(
            f"NPY spectrum must have shape (N, 2). Got: {arr.shape} for {file_path}"
        )

    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)

    _validate_loaded_arrays(x, y, file_path)
    return x, y


def _load_npz_spectrum(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a spectrum from .npz containing arrays named 'x' and 'y'.
    """
    try:
        data = np.load(file_path)
    except Exception as exc:
        raise SpectrumValidationError(f"Failed to read NPZ spectrum: {file_path}") from exc

    if "x" not in data or "y" not in data:
        raise SpectrumValidationError(
            f"NPZ spectrum must contain 'x' and 'y' arrays: {file_path}"
        )

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    _validate_loaded_arrays(x, y, file_path)
    return x, y


def _validate_loaded_arrays(x: np.ndarray, y: np.ndarray, file_path: Path) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise SpectrumValidationError(
            f"Loaded spectrum arrays must be 1D: {file_path}"
        )

    if len(x) != len(y):
        raise SpectrumValidationError(
            f"Loaded x and y arrays have different lengths: {file_path}"
        )

    if len(x) < 2:
        raise SpectrumValidationError(
            f"Loaded spectrum must contain at least 2 points: {file_path}"
        )

    if np.isnan(x).any() or np.isnan(y).any():
        raise SpectrumValidationError(
            f"Loaded spectrum contains NaN values: {file_path}"
        )

    if np.isinf(x).any() or np.isinf(y).any():
        raise SpectrumValidationError(
            f"Loaded spectrum contains infinite values: {file_path}"
        )