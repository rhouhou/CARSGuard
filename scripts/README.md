# Scripts

This folder contains command-line workflows for running CARSGuard validation steps.

The scripts are intended for small experiments, local testing, and reproducible portfolio demonstrations.

---

## Typical workflow

A typical CARSGuard workflow is:

```text
Build benchmark table
Preprocess spectra
Extract features
Build reference profiles
Score spectra
Generate validation reports
```

---

## Script overview

| Script                        | Purpose                                                                |
| ----------------------------- | ---------------------------------------------------------------------- |
| `build_benchmark_table.py`    | Builds or updates the benchmark metadata table from available spectra  |
| `preprocess_dataset.py`       | Harmonizes spectra by cropping, resampling, smoothing, and normalizing |
| `extract_features.py`         | Extracts spectral features used for scoring and reference comparison   |
| `build_reference_profiles.py` | Builds Raman and CARS/BCARS reference profiles                         |
| `score_single_spectrum.py`    | Scores one spectrum and generates a validation output                  |
| `score_dataset.py`            | Scores a full benchmark table or dataset                               |
| `generate_report.py`          | Generates JSON or text validation reports, if available                |

Some scripts may be experimental depending on the current development stage.

---

## Example usage

### Build benchmark table

```bash
python scripts/build_benchmark_table.py
```

### Preprocess spectra

```bash
python scripts/preprocess_dataset.py \
  --x-min 800 \
  --x-max 1800 \
  --num-points 1000 \
  --normalization max
```

### Extract features

```bash
python scripts/extract_features.py
```

### Build reference profiles

```bash
python scripts/build_reference_profiles.py \
  --raman-source-type ramanbiolib \
  --cars-source-type real_cars
```

### Score a single spectrum

```bash
python scripts/score_single_spectrum.py \
  data/raw/carsbench/example.csv \
  --domain BCARS \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
```

### Score a dataset

```bash
python scripts/score_dataset.py \
  --raman-reference outputs/references/raman_reference.json \
  --cars-reference outputs/references/cars_reference.json
```

---

## Notes

Run scripts from the repository root:

```bash
cd CARSGuard
```

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

On some systems, use `python3` instead of `python`.

Example:

```bash
python3 scripts/score_dataset.py
```

---

## Data and output policy

Large raw datasets and generated outputs should not be committed to Git.

Recommended local output locations:

```text
outputs/
results/
data/raw/
data/processed/
```

These paths should usually remain ignored by Git.

Commit only lightweight examples, documentation, metadata, and scripts.
