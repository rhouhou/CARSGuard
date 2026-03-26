from __future__ import annotations

from typing import Any, Dict, Optional

from carsguard.features.feature_vector import extract_feature_vector
from carsguard.physics.sanity import score_physics_plausibility
from carsguard.scoring.artifact_detection import score_artifact_risk
from carsguard.scoring.bcars_realism import score_bcars_realism
from carsguard.scoring.confidence import score_confidence
from carsguard.scoring.raman_consistency import score_raman_consistency


DEFAULT_LABEL_THRESHOLDS = {
    "high_threshold": 0.75,
    "medium_threshold": 0.45,
}


def label_score(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    thresholds = {**DEFAULT_LABEL_THRESHOLDS, **(thresholds or {})}
    high_threshold = float(thresholds["high_threshold"])
    medium_threshold = float(thresholds["medium_threshold"])

    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def evaluate_spectrum(
    spectrum,
    bcars_reference_profile: Optional[Dict[str, Any]] = None,
    raman_reference_profile: Optional[Dict[str, Any]] = None,
    peak_min_prominence: float = 0.01,
    peak_min_distance: int = 3,
    background_window: int = 31,
    bcars_selected_features: Optional[list[str]] = None,
    raman_selected_features: Optional[list[str]] = None,
    bcars_neighbor_k: int = 5,
    raman_neighbor_k: int = 5,
    artifact_thresholds: Optional[Dict[str, float]] = None,
    physics_thresholds: Optional[Dict[str, float]] = None,
    physics_weights: Optional[Dict[str, float]] = None,
    label_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for one spectrum.
    """
    features = extract_feature_vector(
        spectrum,
        peak_min_prominence=peak_min_prominence,
        peak_min_distance=peak_min_distance,
        background_window=background_window,
    )

    bcars_result = None
    if bcars_reference_profile is not None:
        bcars_result = score_bcars_realism(
            features=features,
            reference_profile=bcars_reference_profile,
            selected_features=bcars_selected_features,
            neighbor_k=bcars_neighbor_k,
        )

    raman_result = None
    if raman_reference_profile is not None:
        raman_result = score_raman_consistency(
            features=features,
            reference_profile=raman_reference_profile,
            selected_features=raman_selected_features,
            neighbor_k=raman_neighbor_k,
        )

    artifact_result = score_artifact_risk(
        features=features,
        thresholds=artifact_thresholds,
    )

    physics_result = score_physics_plausibility(
        features=features,
        thresholds=physics_thresholds,
        weights=physics_weights,
    )

    confidence_result = score_confidence(
        bcars_result=bcars_result,
        raman_result=raman_result,
        artifact_result=artifact_result,
    )

    summary = {
        "spectrum_id": spectrum.spectrum_id,
        "domain": spectrum.domain,
        "source_type": spectrum.source_type,
        "sample_class": spectrum.sample_class,
        "sample_name": spectrum.sample_name,
        "features": features,
        "bcars_realism": bcars_result,
        "raman_consistency": raman_result,
        "artifact_risk": artifact_result,
        "physics_plausibility": physics_result,
        "confidence": confidence_result,
    }

    score_labels = {}
    if bcars_result is not None:
        score_labels["bcars_realism"] = label_score(
            float(bcars_result["score"]),
            thresholds=label_thresholds,
        )
    if raman_result is not None:
        score_labels["raman_consistency"] = label_score(
            float(raman_result["score"]),
            thresholds=label_thresholds,
        )

    score_labels["artifact_risk"] = label_score(
        1.0 - float(artifact_result["score"]),
        thresholds=label_thresholds,
    )
    score_labels["physics_plausibility"] = label_score(
        float(physics_result["score"]),
        thresholds=label_thresholds,
    )
    score_labels["confidence"] = label_score(
        float(confidence_result["score"]),
        thresholds=label_thresholds,
    )

    summary["score_labels"] = score_labels
    return summary