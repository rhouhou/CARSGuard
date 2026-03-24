from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS = {
    "roughness_divisor": 5.0,
    "curvature_divisor": 10.0,
    "background_divisor": 10.0,
    "narrow_peak_width_threshold": 3.0,
    "spike_ratio_threshold": 10.0,
}


def score_artifact_risk(
    features: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Heuristic artifact / warning detector.

    High score = higher artifact risk.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    warnings: List[str] = []
    risk_components: Dict[str, float] = {}

    roughness_ratio = float(features.get("roughness_ratio", 0.0))
    second_derivative_std = float(features.get("second_derivative_std", 0.0))
    background_dominance_ratio = float(features.get("background_dominance_ratio", 0.0))
    peak_count = float(features.get("peak_count", 0.0))
    mean_peak_width = float(features.get("mean_peak_width", 0.0))
    max_to_mean_abs_ratio = float(features.get("max_to_mean_abs_ratio", 0.0))

    roughness_divisor = float(thresholds["roughness_divisor"])
    curvature_divisor = float(thresholds["curvature_divisor"])
    background_divisor = float(thresholds["background_divisor"])
    narrow_peak_width_threshold = float(thresholds["narrow_peak_width_threshold"])
    spike_ratio_threshold = float(thresholds["spike_ratio_threshold"])

    roughness_risk = min(1.0, roughness_ratio / max(roughness_divisor, 1e-12))
    risk_components["roughness_risk"] = roughness_risk
    if roughness_risk > 0.6:
        warnings.append("spectrum appears highly oscillatory or noisy")

    curvature_risk = min(1.0, second_derivative_std / max(curvature_divisor, 1e-12))
    risk_components["curvature_risk"] = curvature_risk
    if curvature_risk > 0.6:
        warnings.append("spectrum shows strong high-frequency curvature")

    background_risk = min(1.0, background_dominance_ratio / max(background_divisor, 1e-12))
    risk_components["background_risk"] = background_risk
    if background_risk > 0.7:
        warnings.append("background may dominate resonant structure")

    narrow_peak_risk = 0.0
    if peak_count > 0 and mean_peak_width < narrow_peak_width_threshold:
        narrow_peak_risk = min(
            1.0,
            (narrow_peak_width_threshold - mean_peak_width)
            / max(narrow_peak_width_threshold, 1e-12),
        )
        warnings.append("peaks appear unusually narrow; possible synthetic or numerical artifact")
    risk_components["narrow_peak_risk"] = narrow_peak_risk

    spike_risk = 0.0
    if max_to_mean_abs_ratio > spike_ratio_threshold:
        spike_risk = min(
            1.0,
            (max_to_mean_abs_ratio - spike_ratio_threshold)
            / max(spike_ratio_threshold, 1e-12),
        )
        warnings.append("spectrum may contain sharp spikes or intensity outliers")
    risk_components["spike_risk"] = spike_risk

    final_risk = sum(risk_components.values()) / len(risk_components) if risk_components else 0.0

    return {
        "score_name": "artifact_risk",
        "score": float(max(0.0, min(1.0, final_risk))),
        "risk_components": risk_components,
        "warnings": warnings,
        "thresholds": thresholds,
    }