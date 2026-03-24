from __future__ import annotations

from typing import Any, Dict


def score_confidence(
    bcars_result: Dict[str, Any] | None = None,
    raman_result: Dict[str, Any] | None = None,
    artifact_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Estimate confidence in the overall evaluation.
    """
    components: Dict[str, float] = {}
    notes = []

    if bcars_result is not None:
        bcars_score = float(bcars_result.get("score", 0.0))
        bcars_dist = float(bcars_result.get("distribution_score", 0.0))
        components["bcars_support"] = 0.5 * bcars_score + 0.5 * bcars_dist
    else:
        notes.append("BCARS realism result unavailable")

    if raman_result is not None:
        raman_score = float(raman_result.get("score", 0.0))
        raman_dist = float(raman_result.get("distribution_score", 0.0))
        components["raman_support"] = 0.5 * raman_score + 0.5 * raman_dist
    else:
        notes.append("Raman consistency result unavailable")

    if artifact_result is not None:
        artifact_score = float(artifact_result.get("score", 0.0))
        components["artifact_cleanliness"] = 1.0 - artifact_score
    else:
        notes.append("Artifact result unavailable")

    if bcars_result is not None and raman_result is not None:
        agreement = 1.0 - abs(
            float(bcars_result.get("score", 0.0)) - float(raman_result.get("score", 0.0))
        )
        components["cross_score_agreement"] = agreement

    confidence = sum(components.values()) / len(components) if components else 0.0

    return {
        "score_name": "confidence",
        "score": float(max(0.0, min(1.0, confidence))),
        "components": components,
        "notes": notes,
    }