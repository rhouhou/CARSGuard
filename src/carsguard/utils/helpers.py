from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy-heavy dictionaries into JSON-safe Python types.
    """
    out: Dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, dict):
            out[key] = to_serializable_dict(value)
        elif isinstance(value, list):
            out[key] = [_convert_scalar(v) for v in value]
        else:
            out[key] = value

    return out


def _convert_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return to_serializable_dict(value)
    return value


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary.
    """
    items: List[tuple[str, Any]] = []

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.std(arr))


def dicts_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a DataFrame.
    """
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)