from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from carsguard.core.dataset import SpectrumDataset
from carsguard.features.feature_vector import extract_feature_vector, flatten_feature_vector
from carsguard.io.loaders import load_spectrum_from_record
from carsguard.references.statistics import compute_dataframe_statistics


@dataclass
class CARSReferenceProfile:
    """
    Reference profile built from real CARS / BCARS spectra.
    """
    domain: str
    source_type: str
    n_spectra: int
    feature_statistics: Dict[str, Dict[str, Any]]
    feature_table: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_cars_reference_profile(
    dataset: SpectrumDataset,
    base_dir: str | None = None,
    source_type_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
    peak_min_prominence: float = 0.01,
    peak_min_distance: int = 3,
    background_window: int = 31,
) -> CARSReferenceProfile:
    """
    Build a real coherent-spectroscopy reference profile from a dataset.
    """
    feature_rows: List[Dict[str, Any]] = []

    for record in dataset.records:
        if domain_filter is not None and record.domain != domain_filter:
            continue

        if source_type_filter is not None and record.source_type != source_type_filter:
            continue

        spectrum = load_spectrum_from_record(record, base_dir=base_dir)

        features = extract_feature_vector(
            spectrum,
            peak_min_prominence=peak_min_prominence,
            peak_min_distance=peak_min_distance,
            background_window=background_window,
        )
        flat_features = flatten_feature_vector(features)

        row = {
            "spectrum_id": record.spectrum_id,
            "sample_class": record.sample_class,
            "sample_name": record.sample_name,
            "domain": record.domain,
            **flat_features,
        }
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    stats = compute_dataframe_statistics(
        df,
        exclude_columns=[],
    )

    return CARSReferenceProfile(
        domain=domain_filter or "all",
        source_type=source_type_filter or "all",
        n_spectra=len(feature_rows),
        feature_statistics=stats,
        feature_table=feature_rows,
        metadata={
            "peak_min_prominence": peak_min_prominence,
            "peak_min_distance": peak_min_distance,
            "background_window": background_window,
        },
    )