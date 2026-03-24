from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Spectrum:
    """
    Represents a single spectrum and its metadata.
    """

    spectrum_id: str
    x: np.ndarray
    y: np.ndarray
    domain: str
    source_type: str
    sample_class: Optional[str] = None
    sample_name: Optional[str] = None
    preprocessing_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("Spectrum x and y must be 1D arrays.")

        if len(self.x) != len(self.y):
            raise ValueError("Spectrum x and y must have the same length.")

        if len(self.x) < 2:
            raise ValueError("Spectrum must contain at least two points.")

    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def spectral_range(self) -> tuple[float, float]:
        return float(np.min(self.x)), float(np.max(self.x))

    def copy(self, **updates: Any) -> "Spectrum":
        """
        Return a shallow copy with optional field updates.
        """
        data = {
            "spectrum_id": self.spectrum_id,
            "x": self.x.copy(),
            "y": self.y.copy(),
            "domain": self.domain,
            "source_type": self.source_type,
            "sample_class": self.sample_class,
            "sample_name": self.sample_name,
            "preprocessing_status": self.preprocessing_status,
            "metadata": dict(self.metadata),
        }
        data.update(updates)
        return Spectrum(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Metadata-only representation. Does not serialize full arrays.
        """
        return {
            "spectrum_id": self.spectrum_id,
            "domain": self.domain,
            "source_type": self.source_type,
            "sample_class": self.sample_class,
            "sample_name": self.sample_name,
            "preprocessing_status": self.preprocessing_status,
            "n_points": self.n_points,
            "spectral_range_min": self.spectral_range[0],
            "spectral_range_max": self.spectral_range[1],
            "metadata": self.metadata,
        }