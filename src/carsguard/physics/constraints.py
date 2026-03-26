from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PhysicsThresholds:
    """
    Physics-informed plausibility thresholds.

    These are not universal laws. They are configurable sanity bounds
    meant to catch spectra that are suspicious relative to expected
    coherent Raman / instrument behavior.
    """
    min_peak_width: float = 3.0
    max_background_dominance: float = 10.0
    min_background_dominance: float = 0.05
    max_spike_ratio: float = 10.0
    max_roughness_ratio: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None = None) -> "PhysicsThresholds":
        data = data or {}
        return cls(
            min_peak_width=float(data.get("min_peak_width", 3.0)),
            max_background_dominance=float(data.get("max_background_dominance", 10.0)),
            min_background_dominance=float(data.get("min_background_dominance", 0.05)),
            max_spike_ratio=float(data.get("max_spike_ratio", 10.0)),
            max_roughness_ratio=float(data.get("max_roughness_ratio", 5.0)),
        )


@dataclass
class PhysicsWeights:
    """
    Weights for aggregating physics plausibility components.
    """
    peak_width: float = 0.3
    background: float = 0.3
    spikes: float = 0.2
    roughness: float = 0.2

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None = None) -> "PhysicsWeights":
        data = data or {}
        return cls(
            peak_width=float(data.get("peak_width", 0.3)),
            background=float(data.get("background", 0.3)),
            spikes=float(data.get("spikes", 0.2)),
            roughness=float(data.get("roughness", 0.2)),
        )

    def normalized(self) -> "PhysicsWeights":
        total = self.peak_width + self.background + self.spikes + self.roughness
        if total <= 0:
            return PhysicsWeights()
        return PhysicsWeights(
            peak_width=self.peak_width / total,
            background=self.background / total,
            spikes=self.spikes / total,
            roughness=self.roughness / total,
        )