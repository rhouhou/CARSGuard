import numpy as np

from carsguard.core.spectrum import Spectrum
from carsguard.features.feature_vector import extract_feature_vector
from carsguard.physics.sanity import score_physics_plausibility
from carsguard.scoring.artifact_detection import score_artifact_risk
from carsguard.scoring.bcars_realism import score_bcars_realism
from carsguard.scoring.raman_consistency import score_raman_consistency
from carsguard.scoring.summary import evaluate_spectrum, label_score


def make_test_spectrum(domain: str = "BCARS", source_type: str = "test") -> Spectrum:
    x = np.linspace(800, 1800, 300)
    y = (
        0.7 * np.exp(-0.5 * ((x - 1000) / 25) ** 2)
        + 1.0 * np.exp(-0.5 * ((x - 1250) / 40) ** 2)
        + 0.5 * np.exp(-0.5 * ((x - 1550) / 30) ** 2)
        + 0.1
    )
    return Spectrum(
        spectrum_id="score_test",
        x=x,
        y=y,
        domain=domain,
        source_type=source_type,
    )


def make_reference_profile_from_features(features: dict) -> dict:
    feature_statistics = {}
    feature_table = []

    numeric_items = {
        k: float(v)
        for k, v in features.items()
        if isinstance(v, (int, float)) and v is not None
    }

    for key, value in numeric_items.items():
        spread = max(abs(value) * 0.1, 1e-3)
        feature_statistics[key] = {
            "mean": value,
            "std": spread,
            "minimum": value - 2 * spread,
            "maximum": value + 2 * spread,
            "q05": value - 1.5 * spread,
            "q25": value - 0.5 * spread,
            "q50": value,
            "q75": value + 0.5 * spread,
            "q95": value + 1.5 * spread,
            "count": 10,
        }

    for i in range(3):
        row = {"spectrum_id": f"ref_{i}", "domain": "BCARS", "sample_class": "test"}
        row.update(numeric_items)
        feature_table.append(row)

    return {
        "feature_statistics": feature_statistics,
        "feature_table": feature_table,
    }


def test_artifact_score_runs():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    result = score_artifact_risk(features)

    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["warnings"], list)
    assert "thresholds" in result


def test_artifact_score_respects_custom_thresholds():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    default_result = score_artifact_risk(features)

    custom_result = score_artifact_risk(
        features,
        thresholds={
            "roughness_divisor": 0.01,
            "curvature_divisor": 0.01,
            "background_divisor": 0.01,
            "narrow_peak_width_threshold": 1000.0,
            "spike_ratio_threshold": 0.01,
        },
    )

    assert custom_result["score"] >= default_result["score"]
    assert len(custom_result["warnings"]) >= len(default_result["warnings"])


def test_physics_plausibility_runs():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    result = score_physics_plausibility(features)

    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert "component_scores" in result
    assert "warnings" in result


def test_physics_plausibility_respects_custom_thresholds():
    spec = make_test_spectrum()
    features = extract_feature_vector(spec)

    default_result = score_physics_plausibility(features)

    strict_result = score_physics_plausibility(
        features,
        thresholds={
            "min_peak_width": 1000.0,
            "max_background_dominance": 0.01,
            "min_background_dominance": 0.9,
            "max_spike_ratio": 0.01,
            "max_roughness_ratio": 0.01,
        },
        weights={
            "peak_width": 0.4,
            "background": 0.3,
            "spikes": 0.2,
            "roughness": 0.1,
        },
    )

    assert strict_result["score"] <= default_result["score"]
    assert len(strict_result["warnings"]) >= len(default_result["warnings"])


def test_label_score_default_thresholds():
    assert label_score(0.90) == "high"
    assert label_score(0.60) == "medium"
    assert label_score(0.20) == "low"


def test_label_score_custom_thresholds():
    thresholds = {"high_threshold": 0.8, "medium_threshold": 0.3}

    assert label_score(0.85, thresholds=thresholds) == "high"
    assert label_score(0.50, thresholds=thresholds) == "medium"
    assert label_score(0.10, thresholds=thresholds) == "low"


def test_bcars_realism_respects_selected_features():
    spec = make_test_spectrum(domain="BCARS")
    features = extract_feature_vector(spec)
    reference_profile = make_reference_profile_from_features(features)

    result = score_bcars_realism(
        features=features,
        reference_profile=reference_profile,
        selected_features=["peak_count", "dynamic_range"],
        neighbor_k=2,
    )

    assert result["selected_features"] == ["peak_count", "dynamic_range"]
    assert result["neighbor_k"] == 2
    assert set(result["per_feature_scores"].keys()).issubset({"peak_count", "dynamic_range"})
    assert 0.0 <= result["score"] <= 1.0


def test_raman_consistency_respects_selected_features():
    spec = make_test_spectrum(domain="Raman")
    features = extract_feature_vector(spec)
    reference_profile = make_reference_profile_from_features(features)

    result = score_raman_consistency(
        features=features,
        reference_profile=reference_profile,
        selected_features=["peak_count", "center_of_mass"],
        neighbor_k=3,
    )

    assert result["selected_features"] == ["peak_count", "center_of_mass"]
    assert result["neighbor_k"] == 3
    assert set(result["per_feature_scores"].keys()).issubset({"peak_count", "center_of_mass"})
    assert 0.0 <= result["score"] <= 1.0


def test_evaluate_spectrum_includes_physics_plausibility():
    spec = make_test_spectrum(domain="BCARS")
    features = extract_feature_vector(spec)

    bcars_reference = make_reference_profile_from_features(features)
    raman_reference = make_reference_profile_from_features(features)

    result = evaluate_spectrum(
        spectrum=spec,
        bcars_reference_profile=bcars_reference,
        raman_reference_profile=raman_reference,
        peak_min_prominence=0.02,
        peak_min_distance=4,
        background_window=21,
        bcars_selected_features=["peak_count", "dynamic_range"],
        raman_selected_features=["peak_count", "center_of_mass"],
        bcars_neighbor_k=2,
        raman_neighbor_k=2,
        artifact_thresholds={
            "roughness_divisor": 5.0,
            "curvature_divisor": 10.0,
            "background_divisor": 10.0,
            "narrow_peak_width_threshold": 3.0,
            "spike_ratio_threshold": 10.0,
        },
        physics_thresholds={
            "min_peak_width": 3.0,
            "max_background_dominance": 10.0,
            "min_background_dominance": 0.05,
            "max_spike_ratio": 10.0,
            "max_roughness_ratio": 5.0,
        },
        physics_weights={
            "peak_width": 0.3,
            "background": 0.3,
            "spikes": 0.2,
            "roughness": 0.2,
        },
        label_thresholds={"high_threshold": 0.8, "medium_threshold": 0.3},
    )

    assert "physics_plausibility" in result
    assert result["physics_plausibility"]["score_name"] == "physics_plausibility"
    assert result["score_labels"]["physics_plausibility"] in {"low", "medium", "high"}
    assert result["bcars_realism"]["selected_features"] == ["peak_count", "dynamic_range"]
    assert result["raman_consistency"]["selected_features"] == ["peak_count", "center_of_mass"]