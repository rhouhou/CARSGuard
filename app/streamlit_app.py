from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from carsguard.core.config import load_project_configs
from carsguard.integration.upload_api import evaluate_uploaded_spectrum
from carsguard.io.loaders import load_spectrum
from carsguard.reports.serializers import report_to_text
from carsguard.features.peaks import find_local_peaks
from carsguard.io.benchmark_table import load_benchmark_table
from carsguard.io.loaders import load_spectrum_from_record

@st.cache_data
def load_benchmark_dataset(path="data/benchmark_table.csv"):
    return load_benchmark_table(path)


def get_spectrum_by_id(dataset, spectrum_id, base_dir="."):
    for record in dataset.records:
        if record.spectrum_id == spectrum_id:
            return load_spectrum_from_record(record, base_dir=base_dir)
    return None


# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="CARSGuard", layout="wide")

st.title("CARSGuard")
st.markdown(
    "Validation of CARS/BCARS spectra for **physical realism**, "
    "**Raman consistency**, and **artifact detection**."
)


# -------------------------------
# Load config
# -------------------------------
cfg = load_project_configs("configs")
paths = cfg["paths"]


def load_reference(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


raman_ref = load_reference(Path(paths["references_output_dir"]) / "raman_reference.json")
cars_ref = load_reference(Path(paths["references_output_dir"]) / "cars_reference.json")

dataset = load_benchmark_dataset(paths["benchmark_table"])

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")

domain = st.sidebar.selectbox("Domain", ["BCARS", "CARS", "Raman"])
show_peaks = st.sidebar.checkbox("Show detected peaks", True)
show_json = st.sidebar.checkbox("Show raw JSON", False)


# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload spectrum",
    type=["csv", "txt", "tsv", "npy"],
)

if uploaded_file is None:
    st.info("Upload a spectrum to start.")
    st.stop()


# Save temp
tmp_dir = Path("outputs/tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)
tmp_path = tmp_dir / uploaded_file.name

with tmp_path.open("wb") as f:
    f.write(uploaded_file.getbuffer())


# Load spectrum separately (for plotting)
spectrum = load_spectrum(
    file_path=tmp_path,
    spectrum_id=tmp_path.stem,
    domain=domain,
    source_type="uploaded",
)


# -------------------------------
# Evaluate
# -------------------------------
with st.spinner("Evaluating..."):
    report = evaluate_uploaded_spectrum(
        file_path=tmp_path,
        domain=domain,
        bcars_reference_profile=cars_ref,
        raman_reference_profile=raman_ref,
    )


# -------------------------------
# Plot spectrum
# -------------------------------
st.subheader("Spectrum")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(spectrum.x, spectrum.y, label="Spectrum")

if show_peaks:
    peak_idx = find_local_peaks(spectrum.y)
    ax.scatter(
        spectrum.x[peak_idx],
        spectrum.y[peak_idx],
        s=30,
        label="Peaks",
    )

ax.set_xlabel("Wavenumber / Spectral axis")
ax.set_ylabel("Intensity")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)


# -------------------------------
# Scores
# -------------------------------
st.subheader("Scores")

col1, col2, col3, col4 = st.columns(4)

def s(block):
    return None if block is None else block.get("score")

col1.metric("BCARS realism", f"{s(report['bcars_realism']):.2f}" if report["bcars_realism"] else "N/A")
col2.metric("Raman consistency", f"{s(report['raman_consistency']):.2f}" if report["raman_consistency"] else "N/A")
col3.metric("Artifact risk", f"{s(report['artifact_risk']):.2f}")
col4.metric("Confidence", f"{s(report['confidence']):.2f}")


# -------------------------------
# Interpretation
# -------------------------------
st.subheader("Interpretation")

def interpret(score):
    if score is None:
        return "Not available"
    if score > 0.75:
        return "Strong"
    if score > 0.45:
        return "Moderate"
    return "Weak"

st.write(f"**BCARS realism:** {interpret(s(report['bcars_realism']))}")
st.write(f"**Raman consistency:** {interpret(s(report['raman_consistency']))}")
st.write(f"**Artifact risk:** {'Low' if s(report['artifact_risk']) < 0.3 else 'High'}")
st.write(f"**Confidence:** {interpret(s(report['confidence']))}")


# -------------------------------
# Nearest references
# -------------------------------
# -------------------------------
# Nearest references + overlay
# -------------------------------
st.subheader("Nearest reference spectra")

neighbors = []
if report["bcars_realism"]:
    neighbors = report["bcars_realism"].get("nearest_references", [])

if not neighbors:
    st.info("No reference neighbors available.")
else:
    # Show list
    for n in neighbors[:5]:
        st.write(f"{n['spectrum_id']} | distance = {n['distance']:.3f}")

    # Select one
    selected_id = st.selectbox(
        "Select reference to overlay",
        options=[n["spectrum_id"] for n in neighbors],
    )

    ref_spectrum = get_spectrum_by_id(dataset, selected_id)

    if ref_spectrum is not None:
        st.subheader("Overlay: Uploaded vs Reference")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Normalize both for visual comparison
        def normalize(y):
            return y / (np.max(np.abs(y)) + 1e-12)

        ax.plot(
            spectrum.x,
            normalize(spectrum.y),
            label="Uploaded",
            linewidth=2,
        )

        ax.plot(
            ref_spectrum.x,
            normalize(ref_spectrum.y),
            label=f"Reference ({selected_id})",
            linestyle="--",
        )

        ax.set_xlabel("Spectral axis")
        ax.set_ylabel("Normalized intensity")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    else:
        st.warning("Could not load reference spectrum from dataset.")


# -------------------------------
# Warnings
# -------------------------------
st.subheader("Warnings")

if report["warnings"]:
    for w in report["warnings"]:
        st.warning(w)
else:
    st.success("No major warnings.")


# -------------------------------
# Recommendations
# -------------------------------
st.subheader("Recommendations")

if report["recommendations"]:
    for r in report["recommendations"]:
        st.info(r)
else:
    st.success("No recommendations.")


# -------------------------------
# Text report
# -------------------------------
st.subheader("Text report")
st.text(report_to_text(report))


# -------------------------------
# JSON
# -------------------------------
if show_json:
    st.subheader("Raw JSON")
    st.json(report)