"""
CARSGuard

Validation framework for physical realism and Raman consistency
in CARS/BCARS spectra.
"""

from .core.spectrum import Spectrum
from .core.dataset import SpectrumRecord, SpectrumDataset
from .core.config import load_yaml_config

__all__ = [
    "Spectrum",
    "SpectrumRecord",
    "SpectrumDataset",
    "load_yaml_config",
]