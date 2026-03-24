from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from carsguard.core.spectrum import Spectrum


def save_spectrum_csv(spectrum: Spectrum, path: str | Path) -> None:
    """
    Save a Spectrum object as CSV with columns x and y.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "x": spectrum.x,
        "y": spectrum.y,
    })
    df.to_csv(path, index=False)


def save_spectrum_npy(spectrum: Spectrum, path: str | Path) -> None:
    """
    Save a Spectrum object as NPY with shape (N, 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.column_stack([spectrum.x, spectrum.y])
    np.save(path, arr)


def save_spectrum_npz(spectrum: Spectrum, path: str | Path) -> None:
    """
    Save a Spectrum object as NPZ with named arrays x and y and metadata JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(spectrum.to_dict(), default=_json_default)
    np.savez(path, x=spectrum.x, y=spectrum.y, metadata=metadata_json)


def save_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save a dictionary as a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_default)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=index)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)