# Integration with CARSBench and prCARS

CARSGuard is designed to work as the validation layer in a broader CARS/BCARS spectroscopy workflow.

The three related projects are:

| Project | Role |
|---|---|
| `CARSBench` | Simulates CARS/BCARS spectra under controlled domain shifts |
| `prCARS` | Retrieves Raman-like spectra from CARS/BCARS input |
| `CARSGuard` | Validates whether spectra and retrieval outputs are physically plausible |

Together, they form a small research ecosystem for simulation, retrieval, and validation.

---

## Ecosystem workflow

A typical workflow is:

```text
CARSBench simulated spectrum
        ↓
prCARS retrieval
        ↓
CARSGuard validation
        ↓
Benchmark metrics and QC reports
```

---

## Using CARSGuard with CARSBench

CARSBench can generate synthetic CARS/BCARS spectra and Raman-equivalent targets.

CARSGuard can then evaluate whether the generated spectra are plausible and whether the associated Raman-like targets or retrieval outputs are consistent.

A useful experiment is:

```text
Generate spectra from each CARSBench domain
Run the same validation settings on each domain
Summarize realism, artifact risk, and confidence by domain
Identify which simulated domain is most difficult or least realistic
```

This helps answer questions such as:

- Which domains generate suspicious spectra?
- Does noise or calibration shift increase artifact risk?
- Does NRB-family shift reduce realism?
- Are some synthetic spectra too easy or too unrealistic?

---

## Using CARSGuard with prCARS

prCARS retrieves Raman-like signals from CARS/BCARS spectra.

CARSGuard can validate:

- the original CARS/BCARS input
- the recovered Raman-like output
- the consistency between retrieval output and Raman references
- whether retrieval artifacts are present

Example workflow:

```text
Measured or simulated CARS spectrum
        ↓
prCARS retrieval
        ↓
Recovered Raman-like signal
        ↓
CARSGuard plausibility and consistency checks
```

---

## Combined benchmark workflow

For a complete experiment:

```text
1. Generate synthetic spectra with CARSBench
2. Run prCARS retrieval on each spectrum
3. Compare retrieved signals with known Raman targets
4. Run CARSGuard validation on spectra and retrievals
5. Summarize results by domain and seed
6. Report where retrieval methods succeed or fail
```

---

## Example result table

A combined benchmark could produce a table like:

| Method | Source domain | Test domain | Raman consistency | Artifact risk | CARSGuard status |
|---|---|---|---:|---:|---|
| KK + rolling-ball | `A_typical` | `C_low_res_noisy` | 0.71 | 0.19 | pass |
| KK + ALS | `A_typical` | `F_nrb_family_shift` | 0.52 | 0.43 | inspect |
| MEM + SNIP | `G_biochemical_source` | `H_biochemical_target` | 0.66 | 0.25 | pass |

This type of table makes retrieval quality and validation status easier to interpret together.

---

## Notes

The integration workflow is currently a planned direction rather than a fully packaged end-to-end pipeline.

At this stage:

- `CARSBench` provides simulation and benchmark data generation.
- `prCARS` provides retrieval and preprocessing tools.
- `CARSGuard` provides validation and plausibility-checking tools.

Future work may include shared examples, common data loaders, and an end-to-end pipeline that connects all three projects.
