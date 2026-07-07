# CARSGuard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange)
![Project Type](https://img.shields.io/badge/project-spectroscopy%20validation-purple)

**CARSGuard** is a validation and quality-control framework for assessing the physical realism, Raman consistency, and artifact risk of CARS/BCARS spectra.

It is designed for simulated, recovered, or uploaded coherent Raman spectra and provides interpretable scores, warnings, reference comparisons, and actionable recommendations.

---

## Why this project matters

CARS/BCARS spectra can look visually plausible while still containing unrealistic artifacts, unstable retrieval behavior, poor Raman consistency, or suspicious background dominance.

CARSGuard provides a transparent validation layer for checking whether spectra are:

* experimentally plausible
* chemically consistent with Raman-like references
* free from obvious artifacts
* suitable for downstream benchmarking
* reliable enough to inspect, compare, or report

The goal is not to replace expert judgment, but to support reproducible and interpretable quality control for spectroscopy workflows.

---

## What this repository demonstrates

This project demonstrates:

* scientific validation framework design
* interpretable scoring for spectroscopy data
* quality-control workflows for simulated and recovered spectra
* reference-profile based comparison
* artifact-risk detection
* dataset-aware validation pipelines
* modular Python package engineering
* foundations for integration with CARSBench and prCARS

---

## Project status

CARSGuard is currently an **alpha-stage research and portfolio project**.

| Component                       | Status              |
| ------------------------------- | ------------------- |
| CARS/BCARS spectrum loading     | Implemented         |
| Raman spectrum loading          | Implemented         |
| CARSBench adapter               | Implemented         |
| Preprocessing and harmonization | Implemented         |
| Spectral feature extraction     | Implemented         |
| Reference profile construction  | Implemented         |
| BCARS/CARS realism scoring      | Implemented         |
| Raman consistency scoring       | Implemented         |
| Artifact-risk scoring           | Implemented         |
| Confidence scoring              | Implemented         |
| JSON/text report export         | Implemented         |
| Unit tests                      | Implemented         |
| Streamlit/app interface         | Prototype / planned |
| CI workflow                     | Planned             |
| Full documentation site         | Planned             |
| Real-data validation report     | Planned             |

---

## Key features

* Load real Raman, real CARS/BCARS, and simulated spectra
* Build benchmark tables from multiple spectrum sources
* Preprocess and harmonize spectra onto a common axis
* Extract interpretable spectral features
* Build reference profiles from real or curated spectra
* Score spectra using multiple validation dimensions:

  * CARS/BCARS realism
  * Raman consistency
  * artifact risk
  * confidence
* Generate warnings and recommendations
* Export validation reports in JSON and text form
* Support integration with CARSBench-generated spectra
* Provide modular components for future validation extensions

---

## Main validation scores

CARSGuard uses multiple interpretable scores instead of a single opaque quality score.

| Score              | Purpose                                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| BCARS/CARS realism | Measures how experimentally plausible a CARS/BCARS spectrum is relative to coherent Raman references                  |
| Raman consistency  | Measures how well a recovered or Raman-like spectrum agrees with Raman reference behavior                             |
| Artifact risk      | Detects suspicious behavior such as spikes, oscillations, unrealistic narrow peaks, or excessive background dominance |
| Confidence         | Summarizes how reliable the validation result is based on support from the scoring modules                            |

---

## Intended workflow

A typical CARSGuard workflow is:

```text
Load spectra
Build benchmark table
Preprocess and harmonize spectra
Extract spectral features
Build reference profiles
Score spectra
Generate reports
Inspect warnings and recommendations
```

In the broader CARS/BCARS ecosystem:

```text
CARSBench  → generate simulated benchmark spectra
prCARS     → retrieve Raman-like spectra
CARSGuard  → validate plausibility and Raman consistency
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rhouhou/CARSGuard.git
cd CARSGuard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On some macOS/Linux systems, you may need to use `python3` instead of `python`.

Install the package in editable mode:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install development dependencies:

```bash
python -m pip install pytest
```

---

## Installation check

After installation, verify that the package can be imported:

```bash
python -c "import carsguard; print(carsguard.__name__)"
```

Expected output:

```text
carsguard
```

Run the test suite:

```bash
python -m pytest
```

---

## Data layout

CARSGuard expects raw and generated data to be organized under `data/`.

Recommended structure:

```text
data/
  raw/
    ramanbiolib/
    real_cars/
    carsbench/
    external/

  benchmark_table.csv
```

The benchmark table stores metadata for each spectrum, such as:

* spectrum ID
* source type
* domain
* file path
* preprocessing status
* optional pairing or reference information

Large raw datasets should usually not be committed to Git.

---

## Basic workflow

### 1. Build the benchmark table

```bash
python scripts/build_benchmark_table.py
```

### 2. Preprocess spectra

```bash
python scripts/preprocess_dataset.py \
  --x-min 800 \
  --x-max 1800 \
  --num-points 1000 \
  --normalization max
```

### 3. Extract features

```bash
python scripts/extract_features.py
```

### 4. Build reference profiles

```bash
python scripts/build_reference_profiles.py \
  --raman-source-type ramanbiolib \
  --cars-source-type real_cars
```

### 5. Score a single spectrum

```bash
python scripts/score_single_spectrum.py \
  data/raw/carsbench/example.csv \
  --domain BCARS \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
```

### 6. Score all spectra

```bash
python scripts/score_dataset.py \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
```

---

## Example report output

A typical validation report contains:

* BCARS/CARS realism score
* Raman consistency score
* artifact-risk score
* confidence score
* warnings
* recommendations
* nearest reference matches

Example JSON-like structure:

```json
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
```

---

## Repository structure

```text
CARSGuard/
  app/
    Prototype app or interface components

  configs/
    Default validation, preprocessing, reference, and scoring settings

  data/
    Lightweight data documentation and benchmark metadata

  docs/
    Project documentation and design notes

  notebooks/
    Exploratory notebooks

  scripts/
    Command-line workflows for building tables, preprocessing, references, scoring, and reports

  src/carsguard/
    core/
      Core data models and validation objects

    io/
      Spectrum loading and data input utilities

    preprocessing/
      Resampling, smoothing, normalization, and harmonization

    features/
      Spectral feature extraction

    references/
      Reference profile construction and nearest-reference logic

    scoring/
      Realism, Raman consistency, artifact-risk, and confidence scoring

    reports/
      JSON/text report generation

    integration/
      CARSBench and external workflow adapters

    utils/
      Shared utilities

  tests/
    Unit tests for adapters, config, features, loaders, preprocessing, and scoring
```

---

## Configuration

Project settings are stored in:

```text
configs/
  default.yaml
  preprocessing.yaml
  references.yaml
  scoring.yaml
```

These files define default paths, preprocessing parameters, reference-profile settings, and scoring behavior.

---

## Relationship to the CARS ecosystem

CARSGuard is designed to be the validation layer in a three-part CARS/BCARS workflow:

| Project   | Role                                                                            |
| --------- | ------------------------------------------------------------------------------- |
| CARSBench | Simulates CARS/BCARS spectra under controlled domain shifts                     |
| prCARS    | Retrieves Raman-like signals from CARS/BCARS spectra                            |
| CARSGuard | Validates spectra and retrieval outputs for realism, consistency, and artifacts |

Together, these projects support simulation, retrieval, and validation experiments for spectroscopy-aware machine learning.

---

## Design philosophy

CARSGuard is intentionally:

* modular
* interpretable
* dataset-aware
* conservative in its claims
* easy to extend with new references and scoring rules

The framework is meant to flag suspicious spectra and guide inspection, not to make final scientific or clinical decisions automatically.

---

## Limitations

CARSGuard is an alpha-stage validation framework.

Current limitations include:

* scoring rules are partly heuristic
* reference profiles are simple and should be expanded
* no advanced uncertainty calibration yet
* no class-conditional reference modeling yet
* no full real-data validation report yet
* app/interface components are still prototype-stage

This project is **not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings**.

---

## Roadmap

Planned improvements include:

* Add GitHub Actions CI for tests and linting
* Add a polished `pyproject.toml` with dependencies and optional extras
* Add documentation pages for scoring, preprocessing, references, and reports
* Add example validation reports
* Add example figures to the README
* Add class-specific reference profiles
* Add richer artifact taxonomy
* Add uncertainty and confidence calibration
* Add integration examples with CARSBench and prCARS
* Add Streamlit upload interface
* Add PDF/HTML report export
* Add release notes and citation metadata

---

## Citation

If you use CARSGuard in research, education, or benchmarking work, please cite:

```bibtex
@misc{carsguard2026,
  title={CARSGuard: Validation Framework for Physical Realism and Raman Consistency in CARS/BCARS Spectra},
  author={Houhou, Rola},
  year={2026},
  note={Alpha research software},
  url={https://github.com/rhouhou/CARSGuard}
}
```

---

## License

This project is licensed under the MIT License.