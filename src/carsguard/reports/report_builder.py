from __future__ import annotations

from typing import Any, Dict, List

from carsguard.reports.recommendations import build_recommendations


def _safe_score_block(block: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if block is None:
        return None

    return {
        "score": block.get("score"),
        "score_name": block.get("score_name"),
        "warnings": block.get("warnings", []),
        "mean_neighbor_distance": block.get("mean_neighbor_distance"),
        "nearest_references": block.get("nearest_references", []),
        "per_feature_scores": block.get("per_feature_scores", {}),
        "distribution_score": block.get("distribution_score"),
        "neighbor_score": block.get("neighbor_score"),
        "components": block.get("components", {}),
        "risk_components": block.get("risk_components", {}),
        "component_scores": block.get("component_scores", {}),
        "notes": block.get("notes", []),
        "thresholds": block.get("thresholds", {}),
        "weights": block.get("weights", {}),
    }


def _collect_warnings(evaluation: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    for key in ("bcars_realism", "raman_consistency", "artifact_risk", "physics_plausibility"):
        block = evaluation.get(key)
        if block is None:
            continue
        for warning in block.get("warnings", []):
            warnings.append(warning)

    deduped: List[str] = []
    seen = set()
    for item in warnings:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def build_report(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw evaluation dictionary into a compact report object.
    """
    report = {
        "spectrum_id": evaluation.get("spectrum_id"),
        "domain": evaluation.get("domain"),
        "source_type": evaluation.get("source_type"),
        "sample_class": evaluation.get("sample_class"),
        "sample_name": evaluation.get("sample_name"),
        "score_labels": evaluation.get("score_labels", {}),
        "bcars_realism": _safe_score_block(evaluation.get("bcars_realism")),
        "raman_consistency": _safe_score_block(evaluation.get("raman_consistency")),
        "artifact_risk": _safe_score_block(evaluation.get("artifact_risk")),
        "physics_plausibility": _safe_score_block(evaluation.get("physics_plausibility")),
        "confidence": _safe_score_block(evaluation.get("confidence")),
        "warnings": _collect_warnings(evaluation),
        "recommendations": build_recommendations(evaluation),
    }

    return report