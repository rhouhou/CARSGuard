# CARSGuard

**Validation framework for physical realism and Raman consistency in CARS/BCARS spectra.**

CARSGuard is a modular validation and quality-control framework for assessing simulated, recovered, or uploaded CARS/BCARS spectra. It compares spectra against real coherent Raman references for experimental plausibility and against Raman references for resonant-content consistency. The framework outputs interpretable scores, warnings, nearest-reference matches, and actionable recommendations.

## Project purpose

CARSGuard is designed as the validation companion to simulation pipelines such as **CARSBench**.

- **CARSBench** generates spectra
- **CARSGuard** evaluates whether those spectra are experimentally plausible, chemically consistent, and free from obvious artifacts

The framework is intentionally modular so that scoring rules, reference datasets, preprocessing pipelines, and reporting methods can be refined over time.

## Current scope

Version 1 focuses on:

- loading real Raman, real CARS/BCARS, and simulated spectra
- building a benchmark table
- preprocessing and harmonizing spectra
- extracting interpretable spectral features
- building reference profiles from real datasets
- scoring spectra using:
  - BCARS/CARS realism
  - Raman consistency
  - artifact risk
  - confidence
- exporting reports in JSON and text form

## Repository structure

```text
carsguard/
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/carsguard/
├── tests/
├── outputs/
└── app/

Core code lives in:
src/carsguard/
├── core/
├── io/
├── preprocessing/
├── features/
├── references/
├── scoring/
├── reports/
├── integration/
└── utils/

Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CARSGuard.git
cd CARSGuard
2. Create and activate an environment

Using venv:
python -m venv .venv
source .venv/bin/activate

On Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install the package
pip install -e .

4. Optional development dependencies
pip install pytest

Data layout

Expected raw data structure:
data/raw/
├── ramanbiolib/
├── real_cars/
├── carsbench/
└── external/

The initial benchmark table is stored in:
data/benchmark_table.csv

Each row corresponds to one spectrum and contains metadata such as source type, domain, file path, preprocessing status, and optional pairing information.

Basic workflow
1. Build the benchmark table
python scripts/build_benchmark_table.py

2. Preprocess spectra

Example:
python scripts/preprocess_dataset.py \
  --x-min 800 \
  --x-max 1800 \
  --num-points 1000 \
  --normalization max

3. Extract features
python scripts/extract_features.py

4. Build reference profiles
python scripts/build_reference_profiles.py \
  --raman-source-type ramanbiolib \
  --cars-source-type real_cars
5. Score a single spectrum
python scripts/score_single_spectrum.py \
  data/raw/carsbench/example.csv \
  --domain BCARS \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
6. Score all spectra in the benchmark table
python scripts/score_dataset.py \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
Main scores

CARSGuard uses multiple interpretable scores rather than a single opaque number.

1. BCARS/CARS realism

Measures how experimentally plausible a spectrum is relative to real coherent Raman references.

2. Raman consistency

Measures how well a recovered or Raman-like spectrum agrees with Raman reference behavior.

3. Artifact risk

Detects suspicious behavior such as oscillations, spikes, unrealistic narrow peaks, or excessive background dominance.

4. Confidence

Summarizes how reliable the overall evaluation is based on support from the other scoring modules.

Example output

A typical report contains:

BCARS realism score
Raman consistency score
artifact risk score
confidence score
warnings
recommendations
nearest reference spectra

Example structure:

{
  "spectrum_id": "sim_001",
  "bcars_realism": {
    "score": 0.73
  },
  "raman_consistency": {
    "score": 0.61
  },
  "artifact_risk": {
    "score": 0.22
  },
  "confidence": {
    "score": 0.68
  },
  "warnings": [
    "background may dominate resonant structure"
  ],
  "recommendations": [
    "Inspect the non-resonant background level; it may be outside the experimentally plausible range."
  ]
}
Running tests
pytest tests/
Configuration

Project settings are stored in:

configs/
├── default.yaml
├── preprocessing.yaml
├── references.yaml
└── scoring.yaml

These files define default paths, preprocessing settings, reference-profile settings, and scoring settings.

Design philosophy

CARSGuard is intentionally:

modular
interpretable
dataset-aware
conservative in its claims

It is not meant to replace physical understanding or expert judgment. It is meant to provide a transparent validation layer that helps identify unrealistic simulations, inconsistent recoveries, and suspicious spectra.

Limitations

Current version limitations include:

simple peak detection without SciPy
heuristic scoring rules
simple baseline and smoothing methods
no class-conditional reference modeling yet
no advanced uncertainty calibration yet

These are acceptable for a first GitHub-ready prototype and can be improved iteratively.

Suggested future extensions
better peak detection and linewidth estimation
class-specific reference profiles
richer artifact taxonomy
learned out-of-distribution detection
Streamlit upload interface
integration with broader CARSBench metadata
PDF/HTML report export
Citation / project description

CARSGuard: Validation framework for physical realism and Raman consistency in CARS/BCARS spectra.

License

This project is released under the MIT License. See LICENSE for details.