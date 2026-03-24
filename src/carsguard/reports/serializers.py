from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from carsguard.io.writers import save_json


def report_to_json(report: Dict[str, Any], indent: int = 2) -> str:
    """
    Serialize a report dictionary to a JSON string.
    """
    return json.dumps(report, indent=indent, ensure_ascii=False)


def save_report_json(report: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save a report dictionary to a JSON file.
    """
    save_json(report, path=path, indent=indent)


def report_to_text(report: Dict[str, Any]) -> str:
    """
    Render a simple human-readable text report.
    """
    lines = []
    lines.append(f"Spectrum ID: {report.get('spectrum_id')}")
    lines.append(f"Domain: {report.get('domain')}")
    lines.append(f"Source type: {report.get('source_type')}")

    if report.get("sample_class"):
        lines.append(f"Sample class: {report.get('sample_class')}")
    if report.get("sample_name"):
        lines.append(f"Sample name: {report.get('sample_name')}")

    lines.append("")

    labels = report.get("score_labels", {})
    if labels:
        lines.append("Score labels:")
        for key, value in labels.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

    for key in ("bcars_realism", "raman_consistency", "artifact_risk", "confidence"):
        block = report.get(key)
        if block is None:
            continue

        lines.append(f"{key}:")
        if block.get("score") is not None:
            lines.append(f"  score: {block['score']:.3f}")
        warnings = block.get("warnings", [])
        if warnings:
            lines.append("  warnings:")
            for warning in warnings:
                lines.append(f"    - {warning}")
        lines.append("")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("All warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("Recommendations:")
        for rec in recommendations:
            lines.append(f"  - {rec}")

    return "\n".join(lines).strip()