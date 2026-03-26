from __future__ import annotations

from typing import Any, Dict, List, Optional

from carsguard.physics.constraints import PhysicsThresholds, PhysicsWeights


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _peak_width_score(mean_peak_width: float, min_peak_width: float) -> float:
    """
    Score is high when mean peak width is at or above the minimum plausible width.
    """
    if mean_peak_width >= min_peak_width:
        return 1.0
    if min_peak_width <= 0:
        return 1.0
    return _clamp01(mean_peak_width / min_peak_width)


def _background_score(
    background_dominance_ratio: float,
    min_background_dominance: float,
    max_background_dominance: float,
) -> float:
    """
    Score is high when background dominance lies within a plausible interval.
    """
    value = background_dominance_ratio

    if min_background_dominance <= value <= max_background_dominance:
        return 1.0

    if value < min_background_dominance:
        if min_background_dominance <= 0:
            return 1.0
        return _clamp01(value / min_background_dominance)

    # value > max
    if value <= 0:
        return 0.0
    return _clamp01(max_background_dominance / value)


def _spike_score(max_to_mean_abs_ratio: float, max_spike_ratio: float) -> float:
    """
    Score is high when the spectrum is not dominated by sharp isolated spikes.
    """
    if max_to_mean_abs_ratio <= max_spike_ratio:
        return 1.0
    if max_to_mean_abs_ratio <= 0:
        return 1.0
    return _clamp01(max_spike_ratio / max_to_mean_abs_ratio)


def _roughness_score(roughness_ratio: float, max_roughness_ratio: float) -> float:
    """
    Score is high when roughness is not excessively large.
    """
    if roughness_ratio <= max_roughness_ratio:
        return 1.0
    if roughness_ratio <= 0:
        return 1.0
    return _clamp01(max_roughness_ratio / roughness_ratio)


def score_physics_plausibility(
    features: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Physics-informed plausibility score.

    This does not prove physical correctness. It provides configurable
    sanity checks for suspicious spectra based on:
    - minimum plausible peak width
    - plausible NRB/background dominance range
    - spike/outlier intensity behavior
    - excessive roughness / oscillatory behavior
    """
    threshold_obj = PhysicsThresholds.from_dict(thresholds)
    weight_obj = PhysicsWeights.from_dict(weights).normalized()

    warnings: List[str] = []
    component_scores: Dict[str, float] = {}

    mean_peak_width = float(features.get("mean_peak_width", 0.0))
    background_dominance_ratio = float(features.get("background_dominance_ratio", 0.0))
    max_to_mean_abs_ratio = float(features.get("max_to_mean_abs_ratio", 0.0))
    roughness_ratio = float(features.get("roughness_ratio", 0.0))

    component_scores["peak_width_plausibility"] = _peak_width_score(
        mean_peak_width=mean_peak_width,
        min_peak_width=threshold_obj.min_peak_width,
    )

    component_scores["background_plausibility"] = _background_score(
        background_dominance_ratio=background_dominance_ratio,
        min_background_dominance=threshold_obj.min_background_dominance,
        max_background_dominance=threshold_obj.max_background_dominance,
    )

    component_scores["spike_plausibility"] = _spike_score(
        max_to_mean_abs_ratio=max_to_mean_abs_ratio,
        max_spike_ratio=threshold_obj.max_spike_ratio,
    )

    component_scores["roughness_plausibility"] = _roughness_score(
        roughness_ratio=roughness_ratio,
        max_roughness_ratio=threshold_obj.max_roughness_ratio,
    )

    if component_scores["peak_width_plausibility"] < 0.5:
        warnings.append("peak widths appear narrower than the configured plausible measurement limit")

    if component_scores["background_plausibility"] < 0.5:
        if background_dominance_ratio < threshold_obj.min_background_dominance:
            warnings.append("background appears too weak for a plausible coherent measurement")
        elif background_dominance_ratio > threshold_obj.max_background_dominance:
            warnings.append("background appears too dominant for a plausible coherent measurement")

    if component_scores["spike_plausibility"] < 0.5:
        warnings.append("spectrum may contain spike-like or clipping-related intensity artifacts")

    if component_scores["roughness_plausibility"] < 0.5:
        warnings.append("spectrum roughness appears too high for stable physically plausible structure")

    final_score = (
        weight_obj.peak_width * component_scores["peak_width_plausibility"]
        + weight_obj.background * component_scores["background_plausibility"]
        + weight_obj.spikes * component_scores["spike_plausibility"]
        + weight_obj.roughness * component_scores["roughness_plausibility"]
    )

    return {
        "score_name": "physics_plausibility",
        "score": _clamp01(final_score),
        "component_scores": component_scores,
        "warnings": warnings,
        "thresholds": {
            "min_peak_width": threshold_obj.min_peak_width,
            "max_background_dominance": threshold_obj.max_background_dominance,
            "min_background_dominance": threshold_obj.min_background_dominance,
            "max_spike_ratio": threshold_obj.max_spike_ratio,
            "max_roughness_ratio": threshold_obj.max_roughness_ratio,
        },
        "weights": {
            "peak_width": weight_obj.peak_width,
            "background": weight_obj.background,
            "spikes": weight_obj.spikes,
            "roughness": weight_obj.roughness,
        },
    }