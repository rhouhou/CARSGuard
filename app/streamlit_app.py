from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from carsguard.core.config import load_project_configs
from carsguard.features.peaks import find_local_peaks
from carsguard.integration.upload_api import evaluate_uploaded_spectrum
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.loaders import load_spectrum, load_spectrum_from_record
from carsguard.reports.serializers import report_to_text


@st.cache_data
def load_reference(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON reference profile if it exists."""
    reference_path = Path(path)

    if not reference_path.exists():
        return None

    with reference_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_benchmark_dataset(path: str | Path):
    """Load the benchmark table if it exists."""
    benchmark_path = Path(path)

    if not benchmark_path.exists():
        return None

    return load_benchmark_table(benchmark_path)


def get_spectrum_by_id(dataset, spectrum_id: str, base_dir: str | Path = "."):
    """Load a spectrum from the benchmark dataset by spectrum ID."""
    if dataset is None:
        return None

    for record in dataset.records:
        if record.spectrum_id == spectrum_id:
            return load_spectrum_from_record(record, base_dir=base_dir)

    return None


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize a signal for visual comparison."""
    return signal / (np.max(np.abs(signal)) + 1e-12)


def get_score(report: dict[str, Any], key: str) -> float | None:
    """Safely extract a score from a report block."""
    block = report.get(key)

    if block is None:
        return None

    return block.get("score")


def format_score(score: float | None) -> str:
    """Format a score for Streamlit metric cards."""
    if score is None:
        return "N/A"

    return f"{score:.2f}"


def interpret_score(score: float | None) -> str:
    """Convert a numeric score to a simple interpretation label."""
    if score is None:
        return "Not available"

    if score > 0.75:
        return "Strong"

    if score > 0.45:
        return "Moderate"

    return "Weak"


def plot_uploaded_spectrum(spectrum, show_peaks: bool):
    """Plot uploaded spectrum and optional detected peaks."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(spectrum.x, spectrum.y, label="Spectrum")

    if show_peaks:
        peak_indices = find_local_peaks(spectrum.y)
        ax.scatter(
            spectrum.x[peak_indices],
            spectrum.y[peak_indices],
            s=30,
            label="Peaks",
        )

    ax.set_xlabel("Wavenumber / spectral axis")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


def plot_reference_overlay(spectrum, reference_spectrum, reference_id: str):
    """Plot uploaded spectrum against a selected reference spectrum."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        spectrum.x,
        normalize_signal(spectrum.y),
        label="Uploaded",
        linewidth=2,
    )

    ax.plot(
        reference_spectrum.x,
        normalize_signal(reference_spectrum.y),
        label=f"Reference ({reference_id})",
        linestyle="--",
    )

    ax.set_xlabel("Spectral axis")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

st.set_page_config(page_title="CARSGuard", layout="wide")

st.title("CARSGuard")

st.markdown(
    "Validation of CARS/BCARS spectra for **physical realism**, "
    "**Raman consistency**, and **artifact detection**."
)

cfg = load_project_configs("configs")
paths = cfg["paths"]

raman_ref = load_reference(
    Path(paths["references_output_dir"]) / "raman_reference.json"
)

cars_ref = load_reference(
    Path(paths["references_output_dir"]) / "cars_reference.json"
)

dataset = load_benchmark_dataset(paths["benchmark_table"])

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("Settings")

domain = st.sidebar.selectbox(
    "Domain",
    ["BCARS", "CARS", "Raman"],
)

show_peaks = st.sidebar.checkbox("Show detected peaks", True)
show_json = st.sidebar.checkbox("Show raw JSON", False)

if raman_ref is None:
    st.sidebar.warning("Raman reference profile not found.")

if cars_ref is None:
    st.sidebar.warning("CARS/BCARS reference profile not found.")

if dataset is None:
    st.sidebar.info("Benchmark table not found. Reference overlays are disabled.")

# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload spectrum",
    type=["csv", "txt", "tsv", "npy"],
)

if uploaded_file is None:
    st.info("Upload a spectrum to start.")
    st.stop()

tmp_dir = Path("outputs/tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

tmp_path = tmp_dir / uploaded_file.name

with tmp_path.open("wb") as file:
    file.write(uploaded_file.getbuffer())

spectrum = load_spectrum(
    file_path=tmp_path,
    spectrum_id=tmp_path.stem,
    domain=domain,
    source_type="uploaded",
)

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

with st.spinner("Evaluating..."):
    report = evaluate_uploaded_spectrum(
        file_path=tmp_path,
        domain=domain,
        bcars_reference_profile=cars_ref,
        raman_reference_profile=raman_ref,
    )

# ---------------------------------------------------------------------
# Plot spectrum
# ---------------------------------------------------------------------

st.subheader("Spectrum")
st.pyplot(plot_uploaded_spectrum(spectrum, show_peaks))

# ---------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------

st.subheader("Scores")

bcars_score = get_score(report, "bcars_realism")
raman_score = get_score(report, "raman_consistency")
artifact_score = get_score(report, "artifact_risk")
confidence_score = get_score(report, "confidence")

col1, col2, col3, col4 = st.columns(4)

col1.metric("BCARS realism", format_score(bcars_score))
col2.metric("Raman consistency", format_score(raman_score))
col3.metric("Artifact risk", format_score(artifact_score))
col4.metric("Confidence", format_score(confidence_score))

# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------

st.subheader("Interpretation")

st.write(f"**BCARS realism:** {interpret_score(bcars_score)}")
st.write(f"**Raman consistency:** {interpret_score(raman_score)}")

if artifact_score is None:
    artifact_label = "Not available"
elif artifact_score < 0.3:
    artifact_label = "Low"
elif artifact_score < 0.6:
    artifact_label = "Moderate"
else:
    artifact_label = "High"

st.write(f"**Artifact risk:** {artifact_label}")
st.write(f"**Confidence:** {interpret_score(confidence_score)}")

# ---------------------------------------------------------------------
# Nearest references
# ---------------------------------------------------------------------

st.subheader("Nearest reference spectra")

neighbors = []

if report.get("bcars_realism"):
    neighbors = report["bcars_realism"].get("nearest_references", [])

if not neighbors:
    st.info("No reference neighbors available.")
else:
    for neighbor in neighbors[:5]:
        st.write(
            f"{neighbor['spectrum_id']} | "
            f"distance = {neighbor['distance']:.3f}"
        )

    selected_id = st.selectbox(
        "Select reference to overlay",
        options=[neighbor["spectrum_id"] for neighbor in neighbors],
    )

    reference_spectrum = get_spectrum_by_id(dataset, selected_id)

    if reference_spectrum is not None:
        st.subheader("Overlay: uploaded spectrum vs reference")
        st.pyplot(plot_reference_overlay(spectrum, reference_spectrum, selected_id))
    else:
        st.warning("Could not load reference spectrum from dataset.")

# ---------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------

st.subheader("Warnings")

warnings = report.get("warnings", [])

if warnings:
    for warning in warnings:
        st.warning(warning)
else:
    st.success("No major warnings.")

# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

st.subheader("Recommendations")

recommendations = report.get("recommendations", [])

if recommendations:
    for recommendation in recommendations:
        st.info(recommendation)
else:
    st.success("No recommendations.")

# ---------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------

st.subheader("Text report")
st.text(report_to_text(report))

# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

if show_json:
    st.subheader("Raw JSON")
    st.json(report)