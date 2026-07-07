# Reference profiles

CARSGuard uses reference profiles to compare spectra against expected Raman-like or CARS/BCARS-like behavior.

Reference profiles are not meant to be perfect ground truth. They provide a practical comparison baseline for detecting spectra that look unusual, inconsistent, or suspicious.

---

## Why reference profiles matter

A spectrum may pass simple numeric checks but still be chemically or experimentally implausible.

Reference profiles help answer questions such as:

- Does this spectrum resemble known Raman-like behavior?
- Does this CARS/BCARS spectrum look similar to real coherent Raman data?
- Is the recovered Raman-like signal consistent with expected peak patterns?
- Is the spectrum far away from the available reference space?

---

## Supported reference types

CARSGuard can use different reference sources.

| Reference type | Purpose |
|---|---|
| Raman references | Used to assess Raman consistency of recovered or Raman-like spectra |
| CARS/BCARS references | Used to assess experimental realism of coherent Raman spectra |
| CARSBench references | Used to compare generated spectra across simulated domains |
| Curated project references | Used for task-specific validation and benchmarking |

---

## Typical reference-building workflow

A typical workflow is:

```text
Load reference spectra
Validate axis and intensity arrays
Preprocess and harmonize spectra
Normalize spectra consistently
Extract reference features
Build average or distributional profiles
Save reference profile files
Use references during scoring
```

---

## Reference metadata

Each reference spectrum should ideally include metadata such as:

- source dataset
- spectrum ID
- molecule, tissue, or sample type if known
- acquisition modality
- axis range
- preprocessing settings
- normalization method
- license or usage note
- whether the spectrum is raw or processed

Metadata is important because reference quality affects validation quality.

---

## Raman reference profiles

Raman reference profiles are used to evaluate recovered Raman-like spectra.

They can help detect:

- unrealistic peak positions
- missing expected bands
- excessive oscillations
- poor agreement with biological or chemical reference behavior
- retrieval outputs that look mathematically smooth but chemically weak

A Raman consistency score should always be interpreted in relation to the available reference set.

---

## CARS/BCARS reference profiles

CARS/BCARS reference profiles are used to evaluate the physical realism of coherent Raman spectra.

They can help detect:

- unrealistic intensity scale
- excessive background dominance
- suspiciously flat spectra
- unrealistic peak contrast
- strong detector artifacts
- spectra that differ strongly from real CARS/BCARS behavior

This is especially useful for validating simulations from CARSBench or recovered/intermediate spectra from prCARS workflows.

---

## Nearest-reference comparison

Nearest-reference comparison can be used to find the most similar reference spectra.

This is useful for reporting:

- nearest Raman reference
- nearest CARS/BCARS reference
- distance to the reference space
- whether the spectrum is an outlier
- whether the validation result should be inspected manually

Nearest-reference matches should not be interpreted as final identification unless the reference set is designed for identification.

---

## Reference profile limitations

Reference profiles are only as strong as the data used to build them.

Current limitations may include:

- limited reference diversity
- incomplete sample classes
- different preprocessing histories
- different acquisition conditions
- unbalanced reference datasets
- missing metadata
- possible domain mismatch between reference and target spectra

A low score can mean the target spectrum is problematic, but it can also mean the reference space is incomplete.

---

## Recommended reporting

When using reference profiles, report:

- reference source
- number of reference spectra
- preprocessing settings
- axis range
- normalization method
- whether references are Raman, CARS/BCARS, simulated, or curated
- reference-profile version or file path
- nearest-reference matches, when available

This makes validation results easier to interpret and reproduce.

---

## Future improvements

Planned reference improvements include:

- class-specific reference profiles
- richer biological and chemical reference groups
- uncertainty-aware reference distributions
- better real CARS/BCARS reference support
- reference quality flags
- dataset-level reference reports
