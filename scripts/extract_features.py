#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from carsguard.core.config import load_project_configs
from carsguard.features.feature_vector import extract_feature_vector, flatten_feature_vector
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.loaders import load_spectrum_from_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract feature vectors for all spectra in benchmark table.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    scfg = cfg["scoring"]["features"]["extraction"]

    benchmark = args.benchmark or Path(cfg["paths"]["benchmark_table"])
    output = args.output or Path(cfg["paths"]["features_output"])

    dataset = load_benchmark_table(benchmark)
    rows = []

    for record in dataset.records:
        spectrum = load_spectrum_from_record(record, base_dir=args.base_dir)
        features = extract_feature_vector(
            spectrum,
            peak_min_prominence=scfg["peak_min_prominence"],
            peak_min_distance=scfg["peak_min_distance"],
            background_window=scfg["background_window"],
        )
        flat = flatten_feature_vector(features)
        flat["spectrum_id"] = record.spectrum_id
        flat["source_type"] = record.source_type
        flat["domain"] = record.domain
        flat["sample_class"] = record.sample_class
        rows.append(flat)

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved feature table with {len(df)} rows to {output}")


if __name__ == "__main__":
    main()