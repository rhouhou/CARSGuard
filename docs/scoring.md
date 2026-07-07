# Scoring system

CARSGuard validates CARS/BCARS spectra using multiple interpretable scores.

The goal is not to produce one opaque pass/fail label. Instead, CARSGuard separates validation into different dimensions so that users can understand why a spectrum may be plausible, suspicious, or unreliable.

---

## Main score categories

| Score | Purpose |
|---|---|
| BCARS/CARS realism | Checks whether a spectrum looks experimentally plausible compared with coherent Raman references |
| Raman consistency | Checks whether a recovered Raman-like signal agrees with expected Raman-like behavior |
| Artifact risk | Flags suspicious spectral behavior such as spikes, oscillations, or unrealistic structures |
| Confidence | Summarizes how reliable the validation result is based on available evidence |

---

## BCARS/CARS realism score

The BCARS/CARS realism score measures whether an input CARS/BCARS spectrum appears physically and experimentally plausible.

This score may consider features such as:

- spectral smoothness
- signal dynamic range
- background dominance
- peak structure
- noise behavior
- similarity to real CARS/BCARS reference profiles

A low realism score may indicate that the spectrum is simulated poorly, corrupted, overprocessed, or outside the expected experimental distribution.

---

## Raman consistency score

The Raman consistency score measures how well a recovered Raman-like signal agrees with Raman reference behavior.

This score is useful when validating outputs from retrieval methods such as Kramers-Kronig, MEM, or neural-network retrieval.

It may consider:

- similarity to Raman reference profiles
- peak-position plausibility
- peak-width plausibility
- spectral shape consistency
- agreement with expected biological or chemical signatures

A low Raman consistency score does not always mean the result is wrong. It may also indicate that the reference profile is incomplete, the sample is outside the known reference space, or preprocessing needs adjustment.

---

## Artifact-risk score

The artifact-risk score estimates whether a spectrum contains suspicious patterns that may reduce trust in the result.

Possible artifacts include:

- sharp spikes
- unstable oscillations
- unrealistic narrow peaks
- excessive baseline drift
- unusually flat spectra
- suspiciously high noise
- edge artifacts
- background-dominated spectra

A high artifact-risk score means the spectrum should be inspected manually before being used in downstream analysis.

---

## Confidence score

The confidence score summarizes how much trust should be placed in the validation output.

Confidence may depend on:

- whether reference profiles are available
- whether the spectrum lies within the expected axis range
- whether preprocessing succeeded
- whether feature extraction was stable
- whether different scoring modules agree

A low confidence score means CARSGuard may not have enough evidence to make a strong validation statement.

---

## Example interpretation

Example output:

```json
{
  "bcars_realism": {
    "score": 0.74
  },
  "raman_consistency": {
    "score": 0.63
  },
  "artifact_risk": {
    "score": 0.18
  },
  "confidence": {
    "score": 0.71
  }
}
```

Possible interpretation:

```text
The spectrum appears reasonably plausible.
The recovered Raman-like signal has moderate agreement with references.
Artifact risk is low.
The validation result has acceptable confidence.
```

This would usually be considered a result worth keeping, while still requiring scientific inspection.

---

## Recommended score interpretation

A practical interpretation guide is:

| Score range | Interpretation |
|---:|---|
| 0.80–1.00 | Strong / high-quality evidence |
| 0.60–0.79 | Reasonable but should be inspected |
| 0.40–0.59 | Uncertain or borderline |
| 0.20–0.39 | Weak or suspicious |
| 0.00–0.19 | Very weak or likely problematic |

For artifact risk, the interpretation is reversed:

| Artifact-risk score | Interpretation |
|---:|---|
| 0.00–0.19 | Low artifact risk |
| 0.20–0.39 | Mild artifact risk |
| 0.40–0.59 | Moderate artifact risk |
| 0.60–0.79 | High artifact risk |
| 0.80–1.00 | Very high artifact risk |

---

## Important note about thresholds

The score ranges above are practical interpretation guidelines.

They should not be treated as fixed scientific thresholds until they are validated on larger real datasets.

For now, CARSGuard should be used as a conservative inspection and quality-control tool.

---

## Recommended reporting

When reporting CARSGuard results, include:

- spectrum source
- preprocessing configuration
- reference profile used
- BCARS/CARS realism score
- Raman consistency score
- artifact-risk score
- confidence score
- warnings
- recommendations

Avoid reporting only a single final score, because the individual score categories explain different failure modes.

---

## Limitations

The current scoring system is alpha-stage.

Limitations include:

- scoring rules are partly heuristic
- reference profiles are still limited
- thresholds are not fully calibrated
- real-data validation should be expanded
- uncertainty estimation is still planned

CARSGuard scores should support expert review, not replace it.
