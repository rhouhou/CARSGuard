# Validation reports

CARSGuard reports are designed to make validation results interpretable and reproducible.

A report should not only contain scores. It should also explain warnings, recommendations, reference matches, and preprocessing settings.

---

## Report purpose

A CARSGuard report helps users answer:

- Is the spectrum physically plausible?
- Is the Raman-like output consistent with references?
- Are there artifacts that need manual inspection?
- How confident is the validation result?
- Which preprocessing and reference settings were used?

---

## Typical report contents

A typical report may include:

| Section | Description |
|---|---|
| Spectrum metadata | Source, ID, file path, domain, and modality |
| Preprocessing settings | Axis range, resampling, smoothing, and normalization |
| Reference settings | Raman and CARS/BCARS reference profile paths |
| Scores | Realism, Raman consistency, artifact risk, and confidence |
| Warnings | Human-readable warning messages |
| Recommendations | Suggested next actions |
| Nearest references | Closest matching reference spectra, when available |

---

## Example JSON report

```json
{
  "spectrum_id": "sim_001",
  "source_type": "carsbench",
  "domain": "A_typical",
  "preprocessing": {
    "x_min": 800,
    "x_max": 1800,
    "num_points": 1000,
    "normalization": "max"
  },
  "scores": {
    "bcars_realism": 0.73,
    "raman_consistency": 0.61,
    "artifact_risk": 0.22,
    "confidence": 0.68
  },
  "warnings": [
    "background may dominate resonant structure"
  ],
  "recommendations": [
    "Inspect the non-resonant background level before using this spectrum in downstream benchmarking."
  ],
  "nearest_references": {
    "raman": "raman_ref_012",
    "cars": "cars_ref_004"
  }
}
```

---

## Warning messages

Warnings should be specific and actionable.

Examples:

```text
High artifact risk: sharp spikes detected.
Low Raman consistency: recovered signal is far from available Raman references.
Low confidence: reference profile is missing or incomplete.
Possible background dominance: resonant structure may be weak relative to background.
```

Good warnings help the user understand what to inspect next.

---

## Recommendations

Recommendations should suggest practical next steps.

Examples:

```text
Inspect the raw spectrum before preprocessing.
Try a different normalization method.
Check whether the spectrum axis matches the reference profile.
Compare against additional Raman references.
Run prCARS retrieval with an alternative background correction method.
Exclude this spectrum from benchmark summaries until manually reviewed.
```

---

## JSON reports

JSON reports are useful for:

- automated scoring pipelines
- batch validation
- downstream dashboards
- reproducible experiments
- integration with CARSBench and prCARS

JSON output should preserve enough metadata to reproduce the validation result.

---

## Text reports

Text reports are useful for:

- quick inspection
- README examples
- lightweight summaries
- command-line workflows

A text report should summarize:

- final interpretation
- score values
- warnings
- recommendations
- confidence level

---

## Recommended report interpretation

A good report should make clear whether the result is:

| Status | Meaning |
|---|---|
| Pass | Spectrum appears plausible with low artifact risk |
| Inspect | Spectrum is usable but needs manual review |
| Warning | Spectrum has suspicious behavior or weak reference agreement |
| Fail | Spectrum should not be used without further investigation |

These labels should be conservative and should not replace scientific review.

---

## Future improvements

Planned report improvements include:

- PDF export
- HTML report export
- batch-level summary reports
- visual score dashboards
- example spectra plots
- integration with Streamlit app output
