# Changelog

All notable changes to CARSGuard will be documented in this file.

This project follows a simple versioned changelog format.

---

## [0.1.0-alpha] - 2026-07-07

### Added

- Initial alpha version of CARSGuard
- CARS/BCARS spectrum loading utilities
- Raman spectrum loading utilities
- CARSBench adapter for simulated spectra
- Dataset benchmark-table workflow
- Spectrum preprocessing and harmonization utilities
- Spectral feature extraction
- Reference profile construction
- Nearest-reference comparison utilities
- BCARS/CARS realism scoring
- Raman consistency scoring
- Artifact-risk scoring
- Confidence scoring
- JSON and text validation report export
- Configuration files for preprocessing, references, and scoring
- Unit tests for adapters, config, features, loaders, preprocessing, and scoring
- GitHub Actions CI for automated testing
- Citation metadata with `CITATION.cff`
- Improved README with installation, workflow, scores, limitations, and roadmap
- Lightweight documentation pages for scoring, preprocessing, references, reports, and integration

### Fixed

- Cleaned repository hygiene
- Removed macOS system files from Git tracking
- Updated `.gitignore` for Python, virtual environments, generated outputs, and local data

### Notes

- CARSGuard is currently an alpha-stage research and portfolio project.
- Scoring rules are partly heuristic and should be expanded with stronger real-data validation.
- Large raw datasets are not included in the repository.
- This project is for research, education, benchmarking, and portfolio demonstration.
- It is not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings.
