#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carsguard.core.config import load_project_configs
from carsguard.io.loaders import load_spectrum
from carsguard.reports.report_builder import build_report
from carsguard.reports.serializers import report_to_text, save_report_json
from carsguard.scoring.summary import evaluate_spectrum


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single spectrum.")
    parser.add_argument("file_path", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--source-type", type=str, default=None)
    parser.add_argument("--spectrum-id", type=str, default=None)
    parser.add_argument("--raman-reference", type=Path, default=None)
    parser.add_argument("--cars-reference", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    defaults = cfg["defaults"]
    paths = cfg["paths"]
    scfg = cfg["scoring"]

    extract_cfg = scfg["features"]["extraction"]
    bcars_cfg = scfg["bcars_realism"]
    raman_cfg = scfg["raman_consistency"]
    artifact_cfg = scfg["artifact_detection"]
    label_cfg = scfg["score_labels"]
    physics_cfg = scfg["physics_plausibility"]

    domain = args.domain or defaults["domain"]
    source_type = args.source_type or defaults["source_type"]
    spectrum_id = args.spectrum_id or args.file_path.stem

    raman_reference = args.raman_reference or (Path(paths["references_output_dir"]) / "raman_reference.json")
    cars_reference = args.cars_reference or (Path(paths["references_output_dir"]) / "cars_reference.json")

    spectrum = load_spectrum(
        file_path=args.file_path,
        spectrum_id=spectrum_id,
        domain=domain,
        source_type=source_type,
    )

    raman_ref = _load_json(raman_reference) if raman_reference.exists() else None
    cars_ref = _load_json(cars_reference) if cars_reference.exists() else None

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
        physics_thresholds=physics_cfg["thresholds"] if physics_cfg["enabled"] else None,
        physics_weights=physics_cfg["weights"] if physics_cfg["enabled"] else None,
        label_thresholds=label_cfg,
    )
    report = build_report(evaluation)

    print(report_to_text(report))

    if args.output_json is not None:
        save_report_json(report, args.output_json)
        print(f"\nSaved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()