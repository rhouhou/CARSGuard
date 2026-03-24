from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from carsguard.core.config import load_project_configs
from carsguard.integration.upload_api import evaluate_uploaded_spectrum
from carsguard.reports.serializers import report_to_text


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
CONFIG_DIR = Path("configs")
cfg = load_project_configs(CONFIG_DIR)
paths = cfg["paths"]


# -------------------------------
# Helper: load reference profiles
# -------------------------------
def load_reference(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


raman_ref_path = Path(paths["references_output_dir"]) / "raman_reference.json"
cars_ref_path = Path(paths["references_output_dir"]) / "cars_reference.json"

raman_ref = load_reference(raman_ref_path)
cars_ref = load_reference(cars_ref_path)


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Settings")

domain = st.sidebar.selectbox(
    "Domain",
    options=["BCARS", "CARS", "Raman"],
    index=0,
)

source_type = st.sidebar.text_input(
    "Source type",
    value="uploaded",
)

show_json = st.sidebar.checkbox("Show raw JSON report", value=False)


# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a spectrum file (CSV, TXT, NPY)",
    type=["csv", "txt", "tsv", "npy"],
)

if uploaded_file is None:
    st.info("Upload a spectrum file to begin.")
    st.stop()


# -------------------------------
# Save uploaded file temporarily
# -------------------------------
temp_dir = Path("outputs/tmp")
temp_dir.mkdir(parents=True, exist_ok=True)

temp_path = temp_dir / uploaded_file.name
with temp_path.open("wb") as f:
    f.write(uploaded_file.getbuffer())


# -------------------------------
# Run evaluation
# -------------------------------
with st.spinner("Evaluating spectrum..."):
    report = evaluate_uploaded_spectrum(
        file_path=temp_path,
        domain=domain,
        source_type=source_type,
        bcars_reference_profile=cars_ref,
        raman_reference_profile=raman_ref,
    )


# -------------------------------
# Display results
# -------------------------------
st.subheader("Scores")

col1, col2, col3, col4 = st.columns(4)

def safe_score(block):
    return None if block is None else block.get("score")


bcars_score = safe_score(report.get("bcars_realism"))
raman_score = safe_score(report.get("raman_consistency"))
artifact_score = safe_score(report.get("artifact_risk"))
confidence_score = safe_score(report.get("confidence"))

col1.metric("BCARS realism", f"{bcars_score:.2f}" if bcars_score is not None else "N/A")
col2.metric("Raman consistency", f"{raman_score:.2f}" if raman_score is not None else "N/A")
col3.metric("Artifact risk", f"{artifact_score:.2f}" if artifact_score is not None else "N/A")
col4.metric("Confidence", f"{confidence_score:.2f}" if confidence_score is not None else "N/A")


# -------------------------------
# Labels
# -------------------------------
st.subheader("Score labels")

labels = report.get("score_labels", {})
st.json(labels)


# -------------------------------
# Warnings
# -------------------------------
warnings = report.get("warnings", [])

st.subheader("Warnings")

if warnings:
    for w in warnings:
        st.warning(w)
else:
    st.success("No major warnings detected.")


# -------------------------------
# Recommendations
# -------------------------------
recommendations = report.get("recommendations", [])

st.subheader("Recommendations")

if recommendations:
    for r in recommendations:
        st.info(r)
else:
    st.success("No specific recommendations.")


# -------------------------------
# Text report
# -------------------------------
st.subheader("Text report")

text_report = report_to_text(report)
st.text(text_report)


# -------------------------------
# Raw JSON
# -------------------------------
if show_json:
    st.subheader("Raw JSON report")
    st.json(report)


#how to run the app
# pip install streamlit
#streamlit run app/streamlit_app.py