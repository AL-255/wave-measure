"""Waveform measurements: amplitude, timing, and statistics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .accelerator import njit
from .waveform import Waveform

__all__ = [
    "vpp",
    "vmax",
    "vmin",
    "mean",
    "vrms",
    "std",
    "crossings",
    "period",
    "frequency",
    "duty_cycle",
    "rise_time",
    "fall_time",
]


def _values(waveform: Waveform) -> np.ndarray:
    return np.asarray(waveform.samples, dtype=float)


# -- amplitude -------------------------------------------------------------


def vpp(waveform: Waveform) -> float:
    """Peak-to-peak amplitude."""
    v = _values(waveform)
    return float(v.max() - v.min())


def vmax(waveform: Waveform) -> float:
    """Maximum sample value."""
    return float(_values(waveform).max())


def vmin(waveform: Waveform) -> float:
    """Minimum sample value."""
    return float(_values(waveform).min())


def mean(waveform: Waveform) -> float:
    """Mean (DC) level."""
    return float(_values(waveform).mean())


def vrms(waveform: Waveform) -> float:
    """Root-mean-square amplitude."""
    v = _values(waveform)
    return float(np.sqrt(np.mean(v * v)))


def std(waveform: Waveform) -> float:
    """Standard deviation (AC RMS)."""
    return float(_values(waveform).std())


# -- timing ----------------------------------------------------------------


_EDGE_CODES = {"rising": 1, "falling": -1, "both": 0}


@njit(cache=True)
def _crossing_times(shifted: np.ndarray, t: np.ndarray, edge_code: int) -> np.ndarray:
    """Numba kernel: interpolated crossing times where ``shifted`` changes sign.

    ``edge_code`` is ``1`` (rising), ``-1`` (falling), or ``0`` (both). This is
    the one genuinely sequential loop in the measurement code, so it is the part
    that benefits most from JIT compilation on large captures.
    """
    n = shifted.size
    out = np.empty(n, dtype=np.float64)
    count = 0
    for i in range(n - 1):
        a = shifted[i]
        b = shifted[i + 1]
        sa = 1 if a > 0 else (-1 if a < 0 else 0)
        sb = 1 if b > 0 else (-1 if b < 0 else 0)
        if sa == sb:
            continue
        going_up = b > a
        if edge_code == 1 and not going_up:
            continue
        if edge_code == -1 and going_up:
            continue
        denom = b - a
        frac = 0.0 if denom == 0 else -a / denom
        out[count] = t[i] + frac * (t[i + 1] - t[i])
        count += 1
    return out[:count]


def crossings(
    waveform: Waveform,
    level: Optional[float] = None,
    *,
    edge: str = "rising",
) -> np.ndarray:
    """Return the interpolated times at which the signal crosses ``level``.

    Parameters
    ----------
    level:
        Threshold to detect. Defaults to the signal mean.
    edge:
        ``"rising"``, ``"falling"``, or ``"both"``.
    """
    if edge not in _EDGE_CODES:
        raise ValueError("edge must be 'rising', 'falling', or 'both'")

    v = _values(waveform)
    if level is None:
        level = float(v.mean())

    shifted = np.ascontiguousarray(v - level)
    t = np.ascontiguousarray(waveform.time)
    return _crossing_times(shifted, t, _EDGE_CODES[edge])


def period(waveform: Waveform, level: Optional[float] = None) -> float:
    """Estimate the fundamental period from the mean spacing of rising edges."""
    edges = crossings(waveform, level, edge="rising")
    if edges.size < 2:
        return float("nan")
    return float(np.mean(np.diff(edges)))


def frequency(waveform: Waveform, level: Optional[float] = None) -> float:
    """Estimate the fundamental frequency in hertz."""
    p = period(waveform, level)
    return float(1.0 / p) if p and np.isfinite(p) else float("nan")


def duty_cycle(waveform: Waveform, level: Optional[float] = None) -> float:
    """Fraction of each cycle spent above ``level`` (0..1)."""
    v = _values(waveform)
    if level is None:
        level = float(v.mean())
    return float(np.count_nonzero(v > level) / v.size)


def _edge_time(
    waveform: Waveform, low_frac: float, high_frac: float, rising: bool
) -> float:
    v = _values(waveform)
    t = waveform.time
    lo, hi = v.min(), v.max()
    span = hi - lo
    if span == 0:
        return float("nan")
    low_level = lo + low_frac * span
    high_level = lo + high_frac * span

    low_edges = crossings(waveform, low_level, edge="rising" if rising else "falling")
    high_edges = crossings(
        waveform, high_level, edge="rising" if rising else "falling"
    )
    if low_edges.size == 0 or high_edges.size == 0:
        return float("nan")

    if rising:
        t_low, t_high = low_edges[0], high_edges[0]
    else:
        t_high, t_low = high_edges[0], low_edges[0]
    return float(abs(t_high - t_low))


def rise_time(
    waveform: Waveform, low_frac: float = 0.1, high_frac: float = 0.9
) -> float:
    """10%-90% rise time of the first rising edge (configurable thresholds)."""
    return _edge_time(waveform, low_frac, high_frac, rising=True)


def fall_time(
    waveform: Waveform, low_frac: float = 0.1, high_frac: float = 0.9
) -> float:
    """90%-10% fall time of the first falling edge (configurable thresholds)."""
    return _edge_time(waveform, low_frac, high_frac, rising=False)
