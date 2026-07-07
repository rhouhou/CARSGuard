# CARSGuard Streamlit app

This folder contains a prototype Streamlit interface for CARSGuard.

The app is intended as a lightweight interactive demo for validating CARS/BCARS or Raman-like spectra.

---

## Purpose

The Streamlit app provides a simple browser-based interface for:

* uploading a spectrum
* selecting the spectrum type or domain
* running CARSGuard validation
* displaying validation scores
* showing warnings and recommendations
* viewing JSON or text reports
* inspecting the uploaded spectrum visually

This app is useful for demonstration, portfolio presentation, and early workflow testing.

---

## Run the app

From the repository root:

```bash
source .venv/bin/activate
python3 -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

If `streamlit` is not available, install the app dependencies first:

```bash
python3 -m pip install -e ".[app]"
```

---

## Expected usage

A typical app workflow is:

```text
Upload spectrum
Choose spectrum/domain type
Run validation
Inspect scores
Review warnings
Review recommendations
Export or copy report
```

---

## Current status

The app is currently prototype-stage.

It is not required for the core Python package or command-line workflows.

The main tested functionality lives in:

```text
src/carsguard/
tests/
scripts/
```

---

## Future improvements

Possible improvements include:

* clearer upload templates
* example spectra
* better score cards
* interactive spectrum plots
* downloadable JSON reports
* downloadable text reports
* support for reference-profile upload
* batch validation interface
* integration with CARSBench and prCARS outputs

---

## Notes

The app is for research, education, benchmarking, and portfolio demonstration.

It is not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings.
