from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SpectrumRecord:
    """
    One row from the benchmark table.
    """

    spectrum_id: str
    source_type: str
    domain: str
    file_path: str
    sample_class: Optional[str] = None
    sample_name: Optional[str] = None
    x_axis_type: Optional[str] = None
    spectral_range: Optional[str] = None
    n_points: Optional[int] = None
    preprocessing_status: Optional[str] = None
    label_group: Optional[str] = None
    paired_to_id: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.file_path)


@dataclass
class SpectrumDataset:
    """
    Lightweight container for benchmark records.
    """

    records: List[SpectrumRecord]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterable[SpectrumRecord]:
        return iter(self.records)

    def filter(
        self,
        source_type: Optional[str] = None,
        domain: Optional[str] = None,
        sample_class: Optional[str] = None,
        label_group: Optional[str] = None,
    ) -> "SpectrumDataset":
        filtered = self.records

        if source_type is not None:
            filtered = [r for r in filtered if r.source_type == source_type]

        if domain is not None:
            filtered = [r for r in filtered if r.domain == domain]

        if sample_class is not None:
            filtered = [r for r in filtered if r.sample_class == sample_class]

        if label_group is not None:
            filtered = [r for r in filtered if r.label_group == label_group]

        return SpectrumDataset(filtered)

    def get_by_id(self, spectrum_id: str) -> Optional[SpectrumRecord]:
        for record in self.records:
            if record.spectrum_id == spectrum_id:
                return record
        return None

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                "spectrum_id": r.spectrum_id,
                "source_type": r.source_type,
                "domain": r.domain,
                "file_path": r.file_path,
                "sample_class": r.sample_class,
                "sample_name": r.sample_name,
                "x_axis_type": r.x_axis_type,
                "spectral_range": r.spectral_range,
                "n_points": r.n_points,
                "preprocessing_status": r.preprocessing_status,
                "label_group": r.label_group,
                "paired_to_id": r.paired_to_id,
                "notes": r.notes,
                "metadata": r.metadata,
            }
            for r in self.records
        ]