from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from carsguard.io.loaders import load_spectrum
from carsguard.reports.report_builder import build_report
from carsguard.scoring.summary import evaluate_spectrum


def evaluate_uploaded_spectrum(
    file_path: str | Path,
    domain: str,
    source_type: str = "uploaded",
    spectrum_id: Optional[str] = None,
    sample_class: Optional[str] = None,
    sample_name: Optional[str] = None,
    preprocessing_status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    bcars_reference_profile: Optional[Dict[str, Any]] = None,
    raman_reference_profile: Optional[Dict[str, Any]] = None,
    x_col: int | str = 0,
    y_col: int | str = 1,
    delimiter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and evaluate a user-uploaded spectrum in one step.
    """
    file_path = Path(file_path)

    if spectrum_id is None:
        spectrum_id = file_path.stem

    spectrum = load_spectrum(
        file_path=file_path,
        spectrum_id=spectrum_id,
        domain=domain,
        source_type=source_type,
        sample_class=sample_class,
        sample_name=sample_name,
        preprocessing_status=preprocessing_status,
        metadata=metadata,
        x_col=x_col,
        y_col=y_col,
        delimiter=delimiter,
    )

    evaluation = evaluate_spectrum(
        spectrum=spectrum,
        bcars_reference_profile=bcars_reference_profile,
        raman_reference_profile=raman_reference_profile,
    )

    return build_report(evaluation)