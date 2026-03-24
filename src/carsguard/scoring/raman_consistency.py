from __future__ import annotations

from typing import Any, Dict, List, Optional

from carsguard.references.nearest_neighbors import mean_neighbor_distance, nearest_neighbors
from carsguard.references.statistics import quantile_range_membership_score


DEFAULT_RAMAN_FEATURES = [
    "peak_count",
    "mean_peak_width",
    "std_peak_width",
    "highest_peak_position",
    "mean_peak_height",
    "dynamic_range",
    "spectral_spread",
    "center_of_mass",
]


def score_raman_consistency(
    features: Dict[str, Any],
    reference_profile: Dict[str, Any],
    selected_features: Optional[List[str]] = None,
    neighbor_k: int = 5,
) -> Dict[str, Any]:
    """
    Score how consistent a recovered Raman-like spectrum is relative to a Raman reference.
    """
    selected_features = selected_features or DEFAULT_RAMAN_FEATURES
    feature_stats = reference_profile["feature_statistics"]
    reference_table = reference_profile.get("feature_table", [])

    per_feature_scores: Dict[str, float] = {}
    warnings: List[str] = []

    for feature_name in selected_features:
        if feature_name not in features:
            continue
        if feature_name not in feature_stats:
            continue
        if features[feature_name] is None:
            continue

        value = float(features[feature_name])
        stats = feature_stats[feature_name]

        score = quantile_range_membership_score(
            value=value,
            q25=float(stats["q25"]),
            q75=float(stats["q75"]),
            q05=float(stats["q05"]),
            q95=float(stats["q95"]),
        )
        per_feature_scores[feature_name] = score

        if score < 0.25:
            warnings.append(
                f"{feature_name} is inconsistent with Raman reference behavior"
            )
        elif score < 0.5:
            warnings.append(
                f"{feature_name} is only weakly consistent with Raman reference behavior"
            )

    distribution_score = (
        sum(per_feature_scores.values()) / len(per_feature_scores)
        if per_feature_scores
        else 0.0
    )

    nn_distance = mean_neighbor_distance(
        query_features=features,
        reference_feature_table=reference_table,
        k=neighbor_k,
    )

    neighbors = nearest_neighbors(
        query_features=features,
        reference_feature_table=reference_table,
        k=neighbor_k,
    )

    neighbor_score = 1.0 / (1.0 + nn_distance) if nn_distance != float("inf") else 0.0
    final_score = 0.7 * distribution_score + 0.3 * neighbor_score

    return {
        "score_name": "raman_consistency",
        "score": float(max(0.0, min(1.0, final_score))),
        "distribution_score": float(distribution_score),
        "neighbor_score": float(neighbor_score),
        "mean_neighbor_distance": float(nn_distance) if nn_distance != float("inf") else None,
        "per_feature_scores": per_feature_scores,
        "nearest_references": neighbors,
        "warnings": warnings,
        "selected_features": selected_features,
        "neighbor_k": neighbor_k,
    }