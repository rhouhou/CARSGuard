from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from carsguard.core.spectrum import Spectrum


def plot_spectrum(
    spectrum: Spectrum,
    title: Optional[str] = None,
    xlabel: str = "Spectral axis",
    ylabel: str = "Intensity",
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot a single spectrum.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(spectrum.x, spectrum.y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or spectrum.spectrum_id)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_multiple_spectra(
    spectra: Iterable[Spectrum],
    labels: Optional[Iterable[str]] = None,
    title: str = "Spectra comparison",
    xlabel: str = "Spectral axis",
    ylabel: str = "Intensity",
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot multiple spectra on the same figure.
    """
    plt.figure(figsize=(8, 4))

    labels_list = list(labels) if labels is not None else None

    for i, spectrum in enumerate(spectra):
        label = None
        if labels_list is not None and i < len(labels_list):
            label = labels_list[i]
        else:
            label = spectrum.spectrum_id

        plt.plot(spectrum.x, spectrum.y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if labels_list is not None:
        plt.legend()
    else:
        plt.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()