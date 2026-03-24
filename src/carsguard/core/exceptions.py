class CARSGuardError(Exception):
    """Base exception for the CARSGuard package."""


class SpectrumValidationError(CARSGuardError):
    """Raised when a spectrum fails basic validation."""


class BenchmarkTableError(CARSGuardError):
    """Raised when benchmark table loading or validation fails."""


class ConfigurationError(CARSGuardError):
    """Raised when configuration files are invalid or missing."""


class ReferenceBuildError(CARSGuardError):
    """Raised when reference profiles cannot be built."""


class ScoringError(CARSGuardError):
    """Raised when a scoring function fails."""