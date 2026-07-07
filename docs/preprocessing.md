# Preprocessing and harmonization

CARSGuard uses preprocessing to make spectra comparable before feature extraction, reference comparison, and scoring.

Preprocessing is important because spectra may come from different sources, instruments, simulations, axis ranges, resolutions, and normalization conventions.

---

## Why preprocessing matters

CARS/BCARS and Raman spectra can differ in:

- wavenumber range
- spectral resolution
- number of points
- intensity scale
- baseline behavior
- noise level
- file format
- source type

Without preprocessing, validation scores may reflect formatting or scaling differences instead of real spectral quality.

---

## Typical preprocessing workflow

A typical CARSGuard preprocessing workflow is:

```text
Load spectrum
Validate axis and intensity arrays
Remove invalid values
Sort by wavenumber axis
Crop to target spectral window
Resample to common axis
Smooth or denoise if configured
Normalize intensity
Store harmonized spectrum
```

---

## Axis validation

CARSGuard should first check that each spectrum has:

- a valid wavenumber axis
- a valid intensity array
- matching axis and intensity lengths
- finite numeric values
- no unsupported missing values

Spectra with invalid axes or corrupted intensity arrays should be flagged before scoring.

---

## Cropping

Cropping restricts spectra to a shared spectral window.

Example:

```text
800–1800 cm⁻¹
```

This can be useful when comparing biological fingerprint-region spectra or when reference profiles are only available over a limited range.

Cropping helps ensure that scoring uses the same spectral region across all samples.

---

## Resampling

Resampling places spectra onto a common wavenumber axis.

Example configuration:

```text
x_min: 800
x_max: 1800
num_points: 1000
```

This produces a fixed-length representation for every spectrum.

Resampling is important for:

- feature extraction
- reference comparison
- distance metrics
- batch scoring
- machine-learning-ready tables

---

## Smoothing and denoising

Smoothing can reduce high-frequency noise before feature extraction.

Possible smoothing methods include:

- Savitzky-Golay smoothing
- moving average smoothing
- Gaussian smoothing
- no smoothing

Smoothing should be used carefully. Too much smoothing can remove narrow Raman peaks or hide artifacts.

---

## Normalization

Normalization makes spectra comparable across different intensity scales.

Common normalization options include:

| Method | Description |
|---|---|
| `max` | Divide by maximum absolute intensity |
| `area` | Normalize by total area |
| `zscore` | Subtract mean and divide by standard deviation |
| `none` | Keep original scale |

The best normalization depends on the validation task.

For many quick comparisons, `max` normalization is a practical starting point.

---

## Handling negative values

Some spectra may contain negative values after preprocessing, baseline correction, or retrieval.

Negative values are not always errors, especially for Raman-like recovered signals.

However, strong negative artifacts may indicate:

- poor baseline correction
- unstable retrieval
- noise amplification
- edge artifacts
- invalid normalization

CARSGuard should preserve negative values when they are meaningful but flag suspicious behavior when needed.

---

## Source-specific preprocessing

Different sources may need different preprocessing assumptions.

| Source | Possible preprocessing concern |
|---|---|
| Real Raman | baseline, fluorescence, axis range |
| Real CARS/BCARS | background dominance, dynamic range, detector artifacts |
| CARSBench simulation | domain shift, synthetic noise, generated metadata |
| prCARS retrieval output | negative values, phase artifacts, retrieval instability |

CARSGuard should avoid treating all sources as identical unless they have been harmonized.

---

## Recommended default settings

A practical starting configuration is:

```yaml
x_min: 800
x_max: 1800
num_points: 1000
normalization: max
smoothing: savgol
remove_invalid: true
sort_axis: true
```

These settings are suitable for small benchmark experiments, but they should be adjusted for each dataset.

---

## What preprocessing should not do

Preprocessing should not make unrealistic spectra look artificially valid.

Avoid:

- excessive smoothing
- aggressive baseline removal without reporting it
- clipping strong artifacts silently
- changing peak positions
- hiding failed retrieval behavior
- overwriting raw data without preserving metadata

CARSGuard should keep validation conservative and transparent.

---

## Recommended reporting

When reporting CARSGuard scores, include preprocessing details such as:

- spectral window
- number of resampled points
- normalization method
- smoothing method
- invalid-value handling
- source type
- whether the spectrum was raw, simulated, or retrieved

This makes validation results easier to reproduce and compare.

---

## Notes

Preprocessing is not just a technical step. It affects the meaning of the validation scores.

For this reason, preprocessing settings should be saved with reports and benchmark outputs whenever possible.
