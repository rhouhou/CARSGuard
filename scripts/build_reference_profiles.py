#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from carsguard.core.config import load_project_configs
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.writers import save_json
from carsguard.references.cars_reference import build_cars_reference_profile
from carsguard.references.raman_reference import build_raman_reference_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Raman and CARS/BCARS reference profiles.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)
    rcfg = cfg["references"]

    benchmark = args.benchmark or Path(cfg["paths"]["benchmark_table"])
    output_dir = args.output_dir or Path(cfg["paths"]["references_output_dir"])

    dataset = load_benchmark_table(benchmark)

    raman_ref = build_raman_reference_profile(
        dataset=dataset,
        base_dir=str(args.base_dir),
        source_type_filter=rcfg["raman_reference"]["source_type"],
        domain_filter=rcfg["raman_reference"]["domain"],
        peak_min_prominence=rcfg["raman_reference"]["peak_min_prominence"],
        peak_min_distance=rcfg["raman_reference"]["peak_min_distance"],
        background_window=rcfg["raman_reference"]["background_window"],
    )

    cars_ref = build_cars_reference_profile(
        dataset=dataset,
        base_dir=str(args.base_dir),
        source_type_filter=rcfg["cars_reference"]["source_type"],
        domain_filter=rcfg["cars_reference"]["domain"],
        peak_min_prominence=rcfg["cars_reference"]["peak_min_prominence"],
        peak_min_distance=rcfg["cars_reference"]["peak_min_distance"],
        background_window=rcfg["cars_reference"]["background_window"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(raman_ref.to_dict(), output_dir / "raman_reference.json")
    save_json(cars_ref.to_dict(), output_dir / "cars_reference.json")

    print(f"Saved Raman reference to {output_dir / 'raman_reference.json'}")
    print(f"Saved CARS reference to {output_dir / 'cars_reference.json'}")


if __name__ == "__main__":
    main()