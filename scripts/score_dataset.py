#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from carsguard.core.config import load_project_configs
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.loaders import load_spectrum_from_record
from carsguard.reports.report_builder import build_report
from carsguard.scoring.summary import evaluate_spectrum


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all spectra in a benchmark table.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--raman-reference", type=Path, default=None)
    parser.add_argument("--cars-reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    paths = cfg["paths"]
    scfg = cfg["scoring"]

    extract_cfg = scfg["features"]["extraction"]
    bcars_cfg = scfg["bcars_realism"]
    raman_cfg = scfg["raman_consistency"]
    artifact_cfg = scfg["artifact_detection"]
    label_cfg = scfg["score_labels"]

    benchmark = args.benchmark or Path(paths["benchmark_table"])
    output = args.output or (Path(paths["scores_output_dir"]) / "dataset_scores.csv")
    raman_reference = args.raman_reference or (Path(paths["references_output_dir"]) / "raman_reference.json")
    cars_reference = args.cars_reference or (Path(paths["references_output_dir"]) / "cars_reference.json")

    dataset = load_benchmark_table(benchmark)
    raman_ref = _load_json(raman_reference) if raman_reference.exists() else None
    cars_ref = _load_json(cars_reference) if cars_reference.exists() else None

    rows = []
    for record in dataset.records:
        spectrum = load_spectrum_from_record(record, base_dir=args.base_dir)
        evaluation = evaluate_spectrum(
            spectrum=spectrum,
            bcars_reference_profile=cars_ref if bcars_cfg["enabled"] else None,
            raman_reference_profile=raman_ref if raman_cfg["enabled"] else None,
            peak_min_prominence=extract_cfg["peak_min_prominence"],
            peak_min_distance=extract_cfg["peak_min_distance"],
            background_window=extract_cfg["background_window"],
            bcars_selected_features=bcars_cfg["selected_features"],
            raman_selected_features=raman_cfg["selected_features"],
            bcars_neighbor_k=bcars_cfg["neighbor_k"],
            raman_neighbor_k=raman_cfg["neighbor_k"],
            artifact_thresholds=artifact_cfg["thresholds"],
            label_thresholds=label_cfg,
        )
        report = build_report(evaluation)

        rows.append(
            {
                "spectrum_id": report["spectrum_id"],
                "domain": report["domain"],
                "source_type": report["source_type"],
                "sample_class": report["sample_class"],
                "bcars_realism_score": None if report["bcars_realism"] is None else report["bcars_realism"]["score"],
                "raman_consistency_score": None if report["raman_consistency"] is None else report["raman_consistency"]["score"],
                "artifact_risk_score": None if report["artifact_risk"] is None else report["artifact_risk"]["score"],
                "confidence_score": None if report["confidence"] is None else report["confidence"]["score"],
                "bcars_realism_label": report["score_labels"].get("bcars_realism"),
                "raman_consistency_label": report["score_labels"].get("raman_consistency"),
                "artifact_risk_label": report["score_labels"].get("artifact_risk"),
                "confidence_label": report["score_labels"].get("confidence"),
                "n_warnings": len(report.get("warnings", [])),
                "n_recommendations": len(report.get("recommendations", [])),
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved dataset scoring summary with {len(df)} rows to {output}")


if __name__ == "__main__":
    main()