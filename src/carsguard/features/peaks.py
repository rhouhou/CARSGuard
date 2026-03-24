from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from carsguard.core.spectrum import Spectrum


def find_local_peaks(
    y: np.ndarray,
    min_prominence: float = 0.01,
    min_distance: int = 3,
) -> np.ndarray:
    """
    Simple local peak detector without SciPy.

    A point is considered a peak if:
    - it is greater than its immediate neighbors
    - its height above the local surrounding minimum exceeds min_prominence
    - it is at least min_distance away from the previous accepted peak
    """
    if len(y) < 3:
        return np.array([], dtype=int)

    candidate_indices: List[int] = []

    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_min = min(y[i - 1], y[i + 1])
            prominence = y[i] - local_min
            if prominence >= min_prominence:
                candidate_indices.append(i)

    if not candidate_indices:
        return np.array([], dtype=int)

    # Enforce minimum distance by greedy selection on descending peak height
    candidate_indices = sorted(candidate_indices, key=lambda idx: y[idx], reverse=True)
    selected: List[int] = []

    for idx in candidate_indices:
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)

    selected.sort()
    return np.array(selected, dtype=int)


def extract_peak_features(
    spectrum: Spectrum,
    min_prominence: float = 0.01,
    min_distance: int = 3,
    max_peaks_to_store: int = 20,
) -> Dict[str, Any]:
    """
    Extract simple peak-based features.
    """
    y = spectrum.y
    x = spectrum.x

    peak_indices = find_local_peaks(
        y=y,
        min_prominence=min_prominence,
        min_distance=min_distance,
    )

    peak_positions = x[peak_indices] if len(peak_indices) > 0 else np.array([], dtype=float)
    peak_heights = y[peak_indices] if len(peak_indices) > 0 else np.array([], dtype=float)

    # Sort stored peaks by descending height for stable summaries
    if len(peak_heights) > 0:
        order = np.argsort(peak_heights)[::-1]
        peak_positions_sorted = peak_positions[order]
        peak_heights_sorted = peak_heights[order]
    else:
        peak_positions_sorted = peak_positions
        peak_heights_sorted = peak_heights

    top_peak_positions = peak_positions_sorted[:max_peaks_to_store].tolist()
    top_peak_heights = peak_heights_sorted[:max_peaks_to_store].tolist()

    features: Dict[str, Any] = {
        "peak_count": int(len(peak_indices)),
        "peak_positions": peak_positions.tolist(),
        "peak_heights": peak_heights.tolist(),
        "top_peak_positions": top_peak_positions,
        "top_peak_heights": top_peak_heights,
        "highest_peak_position": float(peak_positions_sorted[0]) if len(peak_positions_sorted) > 0 else None,
        "highest_peak_height": float(peak_heights_sorted[0]) if len(peak_heights_sorted) > 0 else None,
        "mean_peak_height": float(np.mean(peak_heights)) if len(peak_heights) > 0 else 0.0,
        "std_peak_height": float(np.std(peak_heights)) if len(peak_heights) > 0 else 0.0,
    }

    return features