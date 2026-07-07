# Documentation assets

This folder contains lightweight visual assets used in the CARSGuard documentation and README.

---

## Current assets

| File                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `carsguard_workflow.svg` | Workflow diagram showing the main CARSGuard validation pipeline |

---

## Asset policy

Keep this folder lightweight.

Recommended files:

* small SVG diagrams
* small PNG figures for documentation
* simple workflow illustrations
* README-safe visual summaries

Avoid committing:

* large plots
* raw spectra
* generated benchmark outputs
* private or restricted data
* large binary result files

---

## Notes

Assets in this folder should support documentation, portfolio explanation, and reproducibility.

Large generated outputs should be saved locally under ignored folders such as:

```text
outputs/
results/
data/processed/
```
