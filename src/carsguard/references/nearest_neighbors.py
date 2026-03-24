from __future__ import annotations

from typing import Any, Dict, List, Sequence

import math


DEFAULT_EXCLUDE_KEYS = {
    "spectrum_id",
    "sample_class",
    "sample_name",
    "domain",
    "source_type",
}


def _shared_numeric_keys(
    query_features: Dict[str, Any],
    reference_features: Dict[str, Any],
    exclude_keys: set[str] | None = None,
) -> List[str]:
    exclude_keys = exclude_keys or set()
    keys: List[str] = []

    for key, value in query_features.items():
        if key in exclude_keys:
            continue
        if key not in reference_features:
            continue

        ref_value = reference_features[key]

        if isinstance(value, (int, float)) and isinstance(ref_value, (int, float)):
            keys.append(key)

    return keys


def euclidean_distance(
    query_features: Dict[str, Any],
    reference_features: Dict[str, Any],
    exclude_keys: set[str] | None = None,
) -> float:
    """
    Euclidean distance over shared numeric scalar features.
    """
    keys = _shared_numeric_keys(
        query_features,
        reference_features,
        exclude_keys=exclude_keys or DEFAULT_EXCLUDE_KEYS,
    )

    if not keys:
        return float("inf")

    total = 0.0
    for key in keys:
        diff = float(query_features[key]) - float(reference_features[key])
        total += diff * diff

    return math.sqrt(total)


def nearest_neighbors(
    query_features: Dict[str, Any],
    reference_feature_table: Sequence[Dict[str, Any]],
    k: int = 5,
    exclude_keys: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Return the k nearest reference examples by Euclidean distance.
    """
    results: List[Dict[str, Any]] = []

    for ref in reference_feature_table:
        dist = euclidean_distance(
            query_features,
            ref,
            exclude_keys=exclude_keys or DEFAULT_EXCLUDE_KEYS,
        )
        results.append(
            {
                "spectrum_id": ref.get("spectrum_id"),
                "sample_class": ref.get("sample_class"),
                "sample_name": ref.get("sample_name"),
                "domain": ref.get("domain"),
                "distance": float(dist),
            }
        )

    results.sort(key=lambda row: row["distance"])
    return results[:k]


def mean_neighbor_distance(
    query_features: Dict[str, Any],
    reference_feature_table: Sequence[Dict[str, Any]],
    k: int = 5,
    exclude_keys: set[str] | None = None,
) -> float:
    """
    Mean distance to the k nearest neighbors.
    """
    neighbors = nearest_neighbors(
        query_features,
        reference_feature_table,
        k=k,
        exclude_keys=exclude_keys,
    )

    if not neighbors:
        return float("inf")

    return float(sum(n["distance"] for n in neighbors) / len(neighbors))