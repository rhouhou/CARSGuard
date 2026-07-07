# Data folder

This folder documents the expected data layout for CARSGuard.

Large raw datasets should not be committed to Git. Keep only lightweight metadata, examples, or documentation files in the repository.

---

## Recommended structure

```text
data/
  raw/
    ramanbiolib/
    real_cars/
    carsbench/
    external/

  benchmark_table.csv
```

---

## Folder meanings

| Folder | Purpose |
|---|---|
| `raw/ramanbiolib/` | Raman reference spectra |
| `raw/real_cars/` | Real CARS/BCARS spectra |
| `raw/carsbench/` | Simulated spectra from CARSBench |
| `raw/external/` | Other external spectra used for validation experiments |

---

## Benchmark table

The benchmark table should store metadata for each spectrum.

Possible columns include:

| Column | Meaning |
|---|---|
| `spectrum_id` | Unique spectrum identifier |
| `source_type` | Source category such as Raman, CARS, CARSBench, or external |
| `domain` | Simulation or acquisition domain, if available |
| `file_path` | Path to the spectrum file |
| `axis_column` | Column containing wavenumbers, if applicable |
| `intensity_column` | Column containing intensity values, if applicable |
| `preprocessing_status` | Whether the spectrum has been harmonized |
| `notes` | Optional comments |

---

## Git policy

Do not commit:

- large raw datasets
- generated outputs
- intermediate processed arrays
- private or restricted data
- files with unclear licensing

Commit only:

- lightweight metadata
- small example files when licensing allows
- documentation
- reproducible scripts
