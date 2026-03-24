from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from carsguard.core.dataset import SpectrumDataset, SpectrumRecord
from carsguard.core.exceptions import BenchmarkTableError


REQUIRED_COLUMNS = [
    "spectrum_id",
    "source_type",
    "domain",
    "file_path",
]

OPTIONAL_COLUMNS = [
    "sample_class",
    "sample_name",
    "x_axis_type",
    "spectral_range",
    "n_points",
    "preprocessing_status",
    "label_group",
    "paired_to_id",
    "notes",
]


def validate_benchmark_table(df: pd.DataFrame) -> None:
    """
    Validate that the benchmark table contains the minimum required columns
    and no duplicate spectrum IDs.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise BenchmarkTableError(
            f"Benchmark table is missing required columns: {missing}"
        )

    if df["spectrum_id"].isna().any():
        raise BenchmarkTableError("Benchmark table contains missing spectrum_id values.")

    if df["spectrum_id"].duplicated().any():
        duplicates = df.loc[df["spectrum_id"].duplicated(), "spectrum_id"].tolist()
        raise BenchmarkTableError(
            f"Benchmark table contains duplicate spectrum_id values: {duplicates}"
        )


def load_benchmark_table(path: str | Path) -> SpectrumDataset:
    """
    Load a benchmark CSV file into a SpectrumDataset.
    """
    path = Path(path)

    if not path.exists():
        raise BenchmarkTableError(f"Benchmark table not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise BenchmarkTableError(f"Failed to read benchmark table: {path}") from exc

    validate_benchmark_table(df)
    records = dataframe_to_records(df)

    return SpectrumDataset(records)


def dataframe_to_records(df: pd.DataFrame) -> List[SpectrumRecord]:
    """
    Convert a benchmark table DataFrame to SpectrumRecord objects.
    """
    records: List[SpectrumRecord] = []

    known_cols = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    for _, row in df.iterrows():
        metadata = {
            key: row[key]
            for key in df.columns
            if key not in known_cols and pd.notna(row[key])
        }

        record = SpectrumRecord(
            spectrum_id=str(row["spectrum_id"]),
            source_type=str(row["source_type"]),
            domain=str(row["domain"]),
            file_path=str(row["file_path"]),
            sample_class=_optional_str(row, "sample_class"),
            sample_name=_optional_str(row, "sample_name"),
            x_axis_type=_optional_str(row, "x_axis_type"),
            spectral_range=_optional_str(row, "spectral_range"),
            n_points=_optional_int(row, "n_points"),
            preprocessing_status=_optional_str(row, "preprocessing_status"),
            label_group=_optional_str(row, "label_group"),
            paired_to_id=_optional_str(row, "paired_to_id"),
            notes=_optional_str(row, "notes"),
            metadata=metadata,
        )
        records.append(record)

    return records


def records_to_dataframe(dataset: SpectrumDataset) -> pd.DataFrame:
    """
    Convert a SpectrumDataset back to a DataFrame.
    """
    rows = []
    for record in dataset.records:
        row = {
            "spectrum_id": record.spectrum_id,
            "source_type": record.source_type,
            "domain": record.domain,
            "file_path": record.file_path,
            "sample_class": record.sample_class,
            "sample_name": record.sample_name,
            "x_axis_type": record.x_axis_type,
            "spectral_range": record.spectral_range,
            "n_points": record.n_points,
            "preprocessing_status": record.preprocessing_status,
            "label_group": record.label_group,
            "paired_to_id": record.paired_to_id,
            "notes": record.notes,
        }
        row.update(record.metadata)
        rows.append(row)

    return pd.DataFrame(rows)


def save_benchmark_table(dataset: SpectrumDataset, path: str | Path) -> None:
    """
    Save a SpectrumDataset to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = records_to_dataframe(dataset)
    df.to_csv(path, index=False)


def _optional_str(row: pd.Series, key: str) -> str | None:
    value = row[key] if key in row else None
    if pd.isna(value):
        return None
    return str(value)


def _optional_int(row: pd.Series, key: str) -> int | None:
    value = row[key] if key in row else None
    if pd.isna(value):
        return None
    return int(value)