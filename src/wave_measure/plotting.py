"""Rendering helpers for waveforms and spectra.

These functions require ``matplotlib``, which is an optional dependency.
Install it with ``pip install "wave-measure[render]"``.
"""

from __future__ import annotations

from typing import Any, Optional

from .analysis import spectrum
from .waveform import Waveform

__all__ = ["plot", "plot_spectrum", "plot_dpo"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "matplotlib is required for rendering. Install it with "
            '`pip install "wave-measure[render]"`.'
        ) from exc
    return plt


def plot(
    waveform: Waveform,
    *,
    ax: Optional[Any] = None,
    label: Optional[str] = None,
    **kwargs: Any,
):
    """Plot a waveform in the time domain. Returns the matplotlib ``Axes``."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(waveform.time, waveform.samples, label=label, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_spectrum(
    waveform: Waveform,
    *,
    ax: Optional[Any] = None,
    window: str = "hann",
    logy: bool = False,
    **kwargs: Any,
):
    """Plot the single-sided amplitude spectrum. Returns the matplotlib ``Axes``."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    freqs, mag = spectrum(waveform, window=window)
    if logy:
        ax.semilogy(freqs, mag, **kwargs)
    else:
        ax.plot(freqs, mag, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    return ax


def plot_dpo(
    waveform: Waveform,
    *,
    ax: Optional[Any] = None,
    cmap: str = "inferno",
    scale: str = "sqrt",
    xlabel: Optional[str] = None,
    ylabel: str = "Amplitude",
    **kwargs: Any,
):
    """Draw a digital-phosphor render of ``waveform`` on a matplotlib ``Axes``.

    Unlike :func:`~wave_measure.render`, which returns a bare image array, this
    displays the render with **data-unit axes** — the same time/amplitude axes
    :func:`plot` (and ``plt.plot``) would produce — by mapping the accumulation
    histogram through ``imshow`` with the correct extent. Returns the ``Axes``.

    Extra keyword arguments (``x``, ``width``, ``height``, ``x_range``,
    ``y_range``, ``backend``, ``block``, ``workers``) are forwarded to
    :func:`~wave_measure.dpo_histogram`.
    """
    from .dpo import _intensity, dpo_histogram

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    hist, extent = dpo_histogram(waveform, **kwargs)
    intensity = _intensity(hist, scale)
    ax.imshow(
        intensity, origin="lower", extent=extent, aspect="auto", cmap=cmap,
        interpolation="nearest",
    )
    if xlabel is None:
        xlabel = "X" if kwargs.get("x") is not None else "Time (s)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
