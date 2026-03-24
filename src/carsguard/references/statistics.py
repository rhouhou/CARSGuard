from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureStatistics:
    feature_name: str
    mean: float
    std: float
    minimum: float
    maximum: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_feature_statistics(values: Iterable[float], feature_name: str) -> FeatureStatistics:
    """
    Compute robust summary statistics for one numeric feature.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return FeatureStatistics(
            feature_name=feature_name,
            mean=0.0,
            std=0.0,
            minimum=0.0,
            maximum=0.0,
            q05=0.0,
            q25=0.0,
            q50=0.0,
            q75=0.0,
            q95=0.0,
            count=0,
        )

    return FeatureStatistics(
        feature_name=feature_name,
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        q05=float(np.quantile(arr, 0.05)),
        q25=float(np.quantile(arr, 0.25)),
        q50=float(np.quantile(arr, 0.50)),
        q75=float(np.quantile(arr, 0.75)),
        q95=float(np.quantile(arr, 0.95)),
        count=int(len(arr)),
    )


def compute_dataframe_statistics(
    df: pd.DataFrame,
    exclude_columns: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics for all numeric columns in a DataFrame.
    """
    exclude_columns = exclude_columns or []
    stats: Dict[str, Dict[str, Any]] = {}

    numeric_df = df.select_dtypes(include=[np.number])

    for column in numeric_df.columns:
        if column in exclude_columns:
            continue
        feature_stats = compute_feature_statistics(numeric_df[column].values, column)
        stats[column] = feature_stats.to_dict()

    return stats


def zscore_distance(value: float, mean: float, std: float, eps: float = 1e-12) -> float:
    """
    Compute absolute z-score distance from a reference mean/std.
    """
    return abs(value - mean) / (std + eps)


def quantile_range_membership_score(
    value: float,
    q25: float,
    q75: float,
    q05: float,
    q95: float,
) -> float:
    """
    Heuristic score in [0, 1] based on where a value falls relative to
    reference quantile ranges.

    - inside [q25, q75] -> 1.0
    - inside [q05, q95] but outside IQR -> 0.5
    - outside [q05, q95] -> decays toward 0
    """
    if q25 <= value <= q75:
        return 1.0

    if q05 <= value <= q95:
        return 0.5

    # decay outside the 5-95 range
    if value < q05:
        width = max(q25 - q05, 1e-12)
        return max(0.0, 0.5 - (q05 - value) / (2.0 * width))
    else:
        width = max(q95 - q75, 1e-12)
        return max(0.0, 0.5 - (value - q95) / (2.0 * width))