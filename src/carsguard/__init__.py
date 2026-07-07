"""
CARSGuard

Validation framework for physical realism and Raman consistency
in CARS/BCARS spectra.
"""

from .core.config import load_yaml_config
from .core.dataset import SpectrumDataset, SpectrumRecord
from .core.spectrum import Spectrum

__all__ = [
    "Spectrum",
    "SpectrumRecord",
    "SpectrumDataset",
    "load_yaml_config",
]