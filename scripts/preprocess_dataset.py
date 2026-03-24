#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from carsguard.core.config import load_project_configs
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.loaders import load_spectrum_from_record
from carsguard.io.writers import save_spectrum_csv
from carsguard.preprocessing.axis import crop_spectrum, resample_spectrum
from carsguard.preprocessing.baseline import subtract_baseline
from carsguard.preprocessing.filtering import smooth_spectrum
from carsguard.preprocessing.normalization import normalize_spectrum
from carsguard.preprocessing.validation import validate_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess spectra from benchmark table.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    pcfg = cfg["preprocessing"]

    benchmark = args.benchmark or Path(cfg["paths"]["benchmark_table"])
    output_dir = args.output_dir or Path(cfg["paths"]["processed_data_dir"])

    crop_cfg = pcfg["axis"]["crop"]
    resample_cfg = pcfg["axis"]["resample"]
    norm_cfg = pcfg["normalization"]
    filter_cfg = pcfg["filtering"]
    baseline_cfg = pcfg["baseline"]

    dataset = load_benchmark_table(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_x = None
    if resample_cfg["enabled"]:
        x_min = crop_cfg["x_min"]
        x_max = crop_cfg["x_max"]
        num_points = resample_cfg["num_points"]
        if x_min is None or x_max is None:
            raise ValueError("Resampling requires crop.x_min and crop.x_max in preprocessing config.")
        new_x = np.linspace(x_min, x_max, num_points)

    count = 0
    for record in dataset.records:
        spectrum = load_spectrum_from_record(record, base_dir=args.base_dir)
        validate_spectrum(spectrum, raise_on_error=pcfg["validation"]["require_strictly_increasing_x"])

        if crop_cfg["enabled"]:
            spectrum = crop_spectrum(
                spectrum,
                x_min=crop_cfg["x_min"],
                x_max=crop_cfg["x_max"],
            )

        if new_x is not None:
            spectrum = resample_spectrum(
                spectrum,
                new_x=new_x,
                method=resample_cfg["interpolation_method"],
            )

        spectrum = normalize_spectrum(spectrum, method=norm_cfg["method"])
        spectrum = smooth_spectrum(
            spectrum,
            method=filter_cfg["method"],
            window_size=filter_cfg["window_size"],
        )

        baseline_result = subtract_baseline(
            spectrum,
            method=baseline_cfg["method"],
            window_size=baseline_cfg["window_size"],
            clip_zero=baseline_cfg["clip_zero"],
        )
        spectrum = baseline_result.corrected_spectrum

        out_path = output_dir / f"{spectrum.spectrum_id}.csv"
        save_spectrum_csv(spectrum, out_path)
        count += 1

    print(f"Preprocessed {count} spectra into {output_dir}")


if __name__ == "__main__":
    main()