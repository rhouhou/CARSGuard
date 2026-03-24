#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from carsguard.core.config import load_project_configs


def infer_domain(source_type: str) -> str:
    source_type_lower = source_type.lower()
    if "raman" in source_type_lower:
        return "Raman"
    if "bcars" in source_type_lower:
        return "BCARS"
    if "cars" in source_type_lower:
        return "CARS"
    return "unknown"


def collect_files(root: Path, supported_suffixes: set[str]) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_suffixes
    )


def build_rows(data_root: Path, supported_suffixes: set[str]) -> List[dict]:
    rows = []

    for source_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        source_type = source_dir.name
        domain = infer_domain(source_type)

        for file_path in collect_files(source_dir, supported_suffixes):
            rel_path = file_path.relative_to(data_root.parent)

            rows.append(
                {
                    "spectrum_id": file_path.stem,
                    "source_type": source_type,
                    "domain": domain,
                    "file_path": str(rel_path).replace("\\", "/"),
                    "sample_class": None,
                    "sample_name": None,
                    "x_axis_type": None,
                    "spectral_range": None,
                    "n_points": None,
                    "preprocessing_status": "raw",
                    "label_group": (
                        "real"
                        if "real" in source_type.lower() or "raman" in source_type.lower()
                        else "simulated"
                    ),
                    "paired_to_id": None,
                    "notes": None,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build initial benchmark_table.csv from data/raw/")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_project_configs(args.config_dir)

    data_root = args.data_root or Path(cfg["paths"]["raw_data_dir"])
    output = args.output or Path(cfg["paths"]["benchmark_table"])
    supported_suffixes = set(cfg["data"]["supported_suffixes"])

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    rows = build_rows(data_root, supported_suffixes)
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved benchmark table with {len(df)} rows to {output}")


if __name__ == "__main__":
    main()