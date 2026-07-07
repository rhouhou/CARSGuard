# CARSGuard documentation

This folder contains lightweight documentation for CARSGuard.

CARSGuard is an alpha-stage validation and quality-control framework for assessing physical realism, Raman consistency, artifact risk, and confidence in CARS/BCARS spectra.

---

## Documentation pages

| Page                                   | Description                                                                              |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`scoring.md`](scoring.md)             | Explains CARSGuard validation scores and how to interpret them                           |
| [`preprocessing.md`](preprocessing.md) | Explains spectrum preprocessing, resampling, smoothing, normalization, and harmonization |
| [`references.md`](references.md)       | Explains Raman and CARS/BCARS reference-profile construction                             |
| [`reports.md`](reports.md)             | Explains validation report structure, warnings, and recommendations                      |
| [`integration.md`](integration.md)     | Explains how CARSGuard connects with CARSBench and prCARS                                |

---

## Recommended reading order

Start with:

1. [`scoring.md`](scoring.md)
2. [`preprocessing.md`](preprocessing.md)
3. [`references.md`](references.md)
4. [`reports.md`](reports.md)
5. [`integration.md`](integration.md)
6. the main repository [`README.md`](../README.md)

---

## What each page covers

### Scoring

`scoring.md` explains the main validation scores used by CARSGuard:

* BCARS/CARS realism
* Raman consistency
* artifact risk
* confidence

### Preprocessing

`preprocessing.md` explains how spectra are prepared before validation, including:

* axis validation
* cropping
* resampling
* smoothing
* normalization
* harmonization

### References

`references.md` explains how reference profiles are used for comparison, including:

* Raman reference profiles
* CARS/BCARS reference profiles
* nearest-reference comparison
* reference metadata

### Reports

`reports.md` explains the structure of CARSGuard validation reports, including:

* scores
* warnings
* recommendations
* nearest references
* report interpretation

### Integration

`integration.md` explains how CARSGuard fits into the broader CARS/BCARS project ecosystem:

```text
CARSBench  → generate simulated benchmark spectra
prCARS     → retrieve Raman-like spectra
CARSGuard  → validate plausibility and Raman consistency
```

---

## Notes

The documentation is intentionally lightweight at this stage.

Future documentation may include:

* Streamlit app usage
* example validation reports
* real-data validation summaries
* class-specific reference profiles
* PDF/HTML report export
