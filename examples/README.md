# CARSGuard examples

This folder contains lightweight examples showing the type of validation output CARSGuard can produce.

---

## Example validation report

The file [`example_validation_report.json`](example_validation_report.json) is a representative CARSGuard report.

It demonstrates the expected structure of a validation output, including:

- spectrum metadata
- preprocessing settings
- reference profile paths
- validation scores
- score labels
- warnings
- recommendations
- nearest-reference matches

This file is intended for documentation and portfolio demonstration.

It is not a real scientific validation result.

---

## Example report fields

| Field | Meaning |
|---|---|
| `spectrum_id` | Unique identifier for the validated spectrum |
| `source_type` | Source of the spectrum, such as CARSBench, real CARS, Raman, or external |
| `domain` | Simulation or acquisition domain, if available |
| `preprocessing` | Settings used before scoring |
| `references` | Reference profiles used for comparison |
| `scores` | Numeric validation scores |
| `score_labels` | Human-readable score categories |
| `warnings` | Issues detected by CARSGuard |
| `recommendations` | Suggested next actions |
| `nearest_references` | Closest reference spectra, when available |

---

## Notes

Large outputs and raw datasets should not be committed to Git.

Only lightweight examples, small metadata files, and documentation-safe outputs should be included in this folder.
