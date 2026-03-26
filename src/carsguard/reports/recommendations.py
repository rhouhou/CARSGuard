from __future__ import annotations

from typing import Any, Dict, List


def build_recommendations(evaluation: Dict[str, Any]) -> List[str]:
    """
    Generate short actionable recommendations from evaluation results.
    """
    recommendations: List[str] = []

    bcars = evaluation.get("bcars_realism")
    raman = evaluation.get("raman_consistency")
    artifact = evaluation.get("artifact_risk")
    physics = evaluation.get("physics_plausibility")

    if bcars is not None:
        bcars_score = float(bcars.get("score", 0.0))
        per_feature = bcars.get("per_feature_scores", {})

        if bcars_score < 0.45:
            recommendations.append(
                "Compare the spectrum more closely against real BCARS/CARS references before using it as a benchmark example."
            )

        if per_feature.get("background_dominance_ratio", 1.0) < 0.5:
            recommendations.append(
                "Inspect the non-resonant background level; it may be outside the experimentally plausible range."
            )

        if per_feature.get("mean_peak_width", 1.0) < 0.5:
            recommendations.append(
                "Check spectral broadening and instrument-response assumptions; peak widths may be unrealistic."
            )

        if per_feature.get("total_variation", 1.0) < 0.5:
            recommendations.append(
                "Inspect local spectral variability; the spectrum may be too smooth or too irregular relative to real coherent data."
            )

    if raman is not None:
        raman_score = float(raman.get("score", 0.0))
        per_feature = raman.get("per_feature_scores", {})

        if raman_score < 0.45:
            recommendations.append(
                "Recheck the recovery or NRB-removal step; the Raman-like component is weakly consistent with Raman references."
            )

        if per_feature.get("highest_peak_position", 1.0) < 0.5:
            recommendations.append(
                "Inspect peak-position calibration; dominant Raman-like peaks may be shifted from expected reference behavior."
            )

        if per_feature.get("peak_count", 1.0) < 0.5:
            recommendations.append(
                "Inspect whether peaks are being lost or hallucinated during recovery."
            )

    if artifact is not None:
        artifact_score = float(artifact.get("score", 0.0))

        if artifact_score > 0.60:
            recommendations.append(
                "Inspect the spectrum for numerical artifacts, oscillations, or spike-like behavior before trusting the evaluation."
            )

        for warning in artifact.get("warnings", []):
            if "oscillatory" in warning.lower():
                recommendations.append(
                    "Consider reducing numerical ringing or excessive noise amplification."
                )
            if "background" in warning.lower():
                recommendations.append(
                    "Consider revisiting baseline or NRB modeling."
                )
            if "narrow" in warning.lower():
                recommendations.append(
                    "Check whether spectral resolution or simulated linewidth assumptions are too optimistic."
                )
            if "spikes" in warning.lower():
                recommendations.append(
                    "Inspect for interpolation errors, clipping, or isolated intensity outliers."
                )

    if physics is not None:
        physics_score = float(physics.get("score", 0.0))
        comp = physics.get("component_scores", {})

        if physics_score < 0.45:
            recommendations.append(
                "Inspect physics-informed plausibility checks before treating this spectrum as a realistic coherent measurement."
            )

        if comp.get("peak_width_plausibility", 1.0) < 0.5:
            recommendations.append(
                "Increase or verify effective spectral broadening; current peak widths may be below a plausible instrument-limited regime."
            )

        if comp.get("background_plausibility", 1.0) < 0.5:
            recommendations.append(
                "Revisit the balance between structured signal and non-resonant background."
            )

        if comp.get("spike_plausibility", 1.0) < 0.5:
            recommendations.append(
                "Inspect isolated intensity spikes, clipping, or interpolation artifacts."
            )

        if comp.get("roughness_plausibility", 1.0) < 0.5:
            recommendations.append(
                "Inspect whether the spectrum is too oscillatory or numerically unstable to be physically plausible."
            )

    deduped: List[str] = []
    seen = set()
    for rec in recommendations:
        if rec not in seen:
            deduped.append(rec)
            seen.add(rec)

    return deduped