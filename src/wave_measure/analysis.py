"""Frequency-domain analysis and filtering helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .waveform import Waveform

__all__ = ["spectrum", "dominant_frequency", "moving_average", "detrend"]


def spectrum(
    waveform: Waveform, *, window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of the waveform.

    Returns ``(freqs, magnitude)`` where ``freqs`` is in hertz.

    Parameters
    ----------
    window:
        ``"hann"``, ``"hamming"``, or ``"none"`` (rectangular).
    """
    v = np.asarray(waveform.samples, dtype=float)
    n = v.size
    if n == 0:
        return np.empty(0), np.empty(0)

    win = _window(window, n)
    coherent_gain = win.mean()
    spectrum_full = np.fft.rfft(v * win)
    mag = np.abs(spectrum_full) / (n * coherent_gain)
    # Account for folding the negative-frequency half onto the positive side.
    if mag.size > 1:
        mag[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(n, d=waveform.dt or 1.0)
    return freqs, mag


def dominant_frequency(waveform: Waveform, *, window: str = "hann") -> float:
    """Frequency of the largest spectral component, ignoring DC."""
    freqs, mag = spectrum(waveform, window=window)
    if freqs.size < 2:
        return float("nan")
    idx = int(np.argmax(mag[1:])) + 1  # skip the DC bin
    return float(freqs[idx])


def moving_average(waveform: Waveform, window: int) -> Waveform:
    """Smooth the waveform with a centered moving average of ``window`` samples."""
    if window < 1:
        raise ValueError("window must be >= 1")
    v = np.asarray(waveform.samples, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(v, kernel, mode="same")
    return Waveform(
        samples=smoothed,
        time=waveform.time.copy(),
        metadata=dict(waveform.metadata),
    )


def detrend(waveform: Waveform) -> Waveform:
    """Remove a best-fit linear trend (and DC offset) from the waveform."""
    v = np.asarray(waveform.samples, dtype=float)
    t = waveform.time
    if v.size < 2:
        return waveform.copy()
    coeffs = np.polyfit(t, v, 1)
    trend = np.polyval(coeffs, t)
    return Waveform(
        samples=v - trend,
        time=t.copy(),
        metadata=dict(waveform.metadata),
    )


def _window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name in ("none", "rect", "rectangular"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    raise ValueError(f"unknown window: {name!r}")
