"""Streaming operators applied lazily to waveforms.

Each operator transforms a window of samples. The streaming engine fetches a
slightly larger window than requested -- ``left_margin`` extra samples before
the target and ``right_margin`` after -- runs :meth:`Operator.apply`, then trims
the margins off. Declaring margins is what makes random access through stateful
operators (differentiators, filters) correct: the operator always sees enough
context to produce valid output across an arbitrary slice boundary.

Operators here are *length-preserving*: output has one sample per input sample.
Reductions that change that (histogram, peak finding) live in ``reductions.py``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .accelerator import njit

__all__ = [
    "Operator",
    # amplitude (elementwise)
    "AbsOp",
    "AffineOp",
    "ClipOp",
    # math
    "DiffOp",
    "SquareOp",
    "SqrtOp",
    "LogOp",
    # filters
    "FirOp",
    "IirOp",
    "LowPassOp",
    "HighPassOp",
    "MovingAverageOp",
    "MedianOp",
    "moving_average_op",
]


class Operator(ABC):
    """Base class for length-preserving streaming operators."""

    #: Samples of context needed before / after the requested range.
    left_margin: int = 0
    right_margin: int = 0

    @abstractmethod
    def apply(self, window: np.ndarray) -> np.ndarray:
        """Transform ``window`` and return an array of the same length.

        Edge values within the margins may be invalid; the engine trims them.
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


# -- elementwise (margin 0) -------------------------------------------------


class AbsOp(Operator):
    def apply(self, window: np.ndarray) -> np.ndarray:
        return np.abs(window)


class AffineOp(Operator):
    """``value * gain + offset`` -- backs scaling and scalar arithmetic."""

    def __init__(self, gain: float = 1.0, offset: float = 0.0) -> None:
        self.gain = float(gain)
        self.offset = float(offset)

    def apply(self, window: np.ndarray) -> np.ndarray:
        return window * self.gain + self.offset

    def __repr__(self) -> str:
        return f"AffineOp(gain={self.gain:g}, offset={self.offset:g})"


class ClipOp(Operator):
    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)

    def apply(self, window: np.ndarray) -> np.ndarray:
        return np.clip(window, self.lo, self.hi)

    def __repr__(self) -> str:
        return f"ClipOp(lo={self.lo:g}, hi={self.hi:g})"


# -- stencils (need context) ------------------------------------------------


class DiffOp(Operator):
    """First difference ``y[n] = x[n] - x[n-1]``. Needs one sample of history."""

    left_margin = 1

    def apply(self, window: np.ndarray) -> np.ndarray:
        out = np.empty_like(window)
        out[0] = 0.0  # trimmed away except at the global start
        np.subtract(window[1:], window[:-1], out=out[1:])
        return out


class SquareOp(Operator):
    def apply(self, window: np.ndarray) -> np.ndarray:
        return window * window


class SqrtOp(Operator):
    def apply(self, window: np.ndarray) -> np.ndarray:
        return np.sqrt(window)


class LogOp(Operator):
    def __init__(self, base: float = None) -> None:
        self.base = base

    def apply(self, window: np.ndarray) -> np.ndarray:
        out = np.log(window)
        if self.base is not None:
            out = out / math.log(self.base)
        return out

    def __repr__(self) -> str:
        return f"LogOp(base={self.base or 'e'})"


# -- filters ----------------------------------------------------------------


class FirOp(Operator):
    """Causal FIR filter ``y[n] = sum_k h[k] x[n-k]``. Exact random access."""

    def __init__(self, coeffs) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float).ravel()
        if self.coeffs.size == 0:
            raise ValueError("FIR needs at least one coefficient")
        self.left_margin = self.coeffs.size - 1

    def apply(self, window: np.ndarray) -> np.ndarray:
        return np.convolve(window, self.coeffs, mode="full")[: window.size]

    def __repr__(self) -> str:
        return f"FirOp(taps={self.coeffs.size})"


def moving_average_op(window: int) -> FirOp:
    """A causal box-car (running mean) FIR of length ``window``."""
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    return FirOp(np.full(window, 1.0 / window))


class MovingAverageOp(FirOp):  # alias kept for discoverability
    def __init__(self, window: int) -> None:
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        super().__init__(np.full(w, 1.0 / w))
        self.window = w

    def __repr__(self) -> str:
        return f"MovingAverageOp(window={self.window})"


@njit(cache=True)
def _iir_df1(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Direct-form-I IIR filter over a window (zero initial state)."""
    n = x.size
    nb = b.size
    na = a.size
    y = np.empty_like(x)
    a0 = a[0]
    for i in range(n):
        acc = 0.0
        for j in range(nb):
            if i - j >= 0:
                acc += b[j] * x[i - j]
        for j in range(1, na):
            if i - j >= 0:
                acc -= a[j] * y[i - j]
        y[i] = acc / a0
    return y


class IirOp(Operator):
    """Generic IIR filter with numerator ``b`` and denominator ``a``.

    Being recursive, exact random access is impossible; the operator declares a
    warm-up margin sized from the slowest pole (``warmup_tau`` time constants),
    over which the filter state re-converges. Pass ``warmup`` to override.
    """

    _MAX_WARMUP = 1 << 20

    def __init__(self, b, a=(1.0,), *, warmup: int = None, warmup_tau: float = 20.0):
        self.b = np.asarray(b, dtype=float).ravel()
        self.a = np.asarray(a, dtype=float).ravel()
        if self.a.size == 0 or self.a[0] == 0:
            raise ValueError("a[0] must be non-zero")
        if warmup is None:
            warmup = self._estimate_warmup(warmup_tau)
        self.left_margin = max(self.b.size - 1, int(warmup))

    def _estimate_warmup(self, warmup_tau: float) -> int:
        if self.a.size <= 1:
            return self.b.size  # FIR-like
        rho = float(np.max(np.abs(np.roots(self.a))))
        if not np.isfinite(rho) or rho >= 1.0:
            return self._MAX_WARMUP  # marginal/unstable: warm up generously
        tau = -1.0 / math.log(rho) if rho > 0 else 1.0
        return min(self._MAX_WARMUP, max(1, int(math.ceil(warmup_tau * tau))))

    def apply(self, window: np.ndarray) -> np.ndarray:
        return _iir_df1(np.ascontiguousarray(window), self.b, self.a)

    def __repr__(self) -> str:
        return f"IirOp(nb={self.b.size}, na={self.a.size}, margin={self.left_margin})"


class MedianOp(Operator):
    """Centered sliding-median filter of odd length ``window``."""

    def __init__(self, window: int) -> None:
        w = int(window)
        if w < 1 or w % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        self.window = w
        self.left_margin = self.right_margin = w // 2

    def apply(self, window: np.ndarray) -> np.ndarray:
        w = self.window
        if window.size < w:
            fill = float(np.median(window)) if window.size else 0.0
            return np.full_like(window, fill)
        med = np.median(sliding_window_view(window, w), axis=1)
        out = np.empty_like(window)
        k = w // 2
        out[k : k + med.size] = med
        out[:k] = med[0]
        out[k + med.size :] = med[-1]
        return out

    def __repr__(self) -> str:
        return f"MedianOp(window={self.window})"


@njit(cache=True)
def _one_pole_lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole IIR low-pass: ``y[n] = y[n-1] + alpha*(x[n] - y[n-1])``."""
    out = np.empty_like(x)
    if x.size == 0:
        return out
    acc = x[0]
    for i in range(x.size):
        acc += alpha * (x[i] - acc)
        out[i] = acc
    return out


class LowPassOp(Operator):
    """One-pole low-pass filter with a configurable -3 dB cutoff.

    Being recursive (IIR), its impulse response is infinite, so exact random
    access is impossible; instead it declares a finite warm-up margin (a few
    time constants) over which the filter state re-converges. Output is then
    correct to a small, documented tolerance regardless of where a slice begins.
    """

    def __init__(self, cutoff: float, sample_rate: float, *, warmup_tau: float = 20.0):
        if cutoff <= 0 or cutoff >= sample_rate / 2:
            raise ValueError("cutoff must be in (0, sample_rate/2)")
        self.cutoff = float(cutoff)
        self.sample_rate = float(sample_rate)
        # Standard one-pole mapping from cutoff to smoothing factor.
        rc_omega = 2.0 * math.pi * cutoff / sample_rate
        self.alpha = rc_omega / (rc_omega + 1.0)
        # Warm up for several time constants so state re-converges per slice.
        tau_samples = 1.0 / self.alpha
        self.left_margin = max(1, int(math.ceil(warmup_tau * tau_samples)))

    def apply(self, window: np.ndarray) -> np.ndarray:
        return _one_pole_lowpass(np.ascontiguousarray(window), self.alpha)

    def __repr__(self) -> str:
        return f"LowPassOp(cutoff={self.cutoff:g}, margin={self.left_margin})"


@njit(cache=True)
def _one_pole_highpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole IIR high-pass: ``y[n] = alpha*(y[n-1] + x[n] - x[n-1])``."""
    out = np.empty_like(x)
    if x.size == 0:
        return out
    prev_x = x[0]
    prev_y = 0.0
    for i in range(x.size):
        y = alpha * (prev_y + x[i] - prev_x)
        out[i] = y
        prev_y = y
        prev_x = x[i]
    return out


class HighPassOp(Operator):
    """One-pole high-pass filter with a configurable -3 dB cutoff (IIR warm-up)."""

    def __init__(self, cutoff: float, sample_rate: float, *, warmup_tau: float = 20.0):
        if cutoff <= 0 or cutoff >= sample_rate / 2:
            raise ValueError("cutoff must be in (0, sample_rate/2)")
        self.cutoff = float(cutoff)
        self.sample_rate = float(sample_rate)
        rc_omega = 2.0 * math.pi * cutoff / sample_rate
        self.alpha = 1.0 / (1.0 + rc_omega)
        tau = -1.0 / math.log(self.alpha) if self.alpha > 0 else 1.0
        self.left_margin = max(1, int(math.ceil(warmup_tau * tau)))

    def apply(self, window: np.ndarray) -> np.ndarray:
        return _one_pole_highpass(np.ascontiguousarray(window), self.alpha)

    def __repr__(self) -> str:
        return f"HighPassOp(cutoff={self.cutoff:g}, margin={self.left_margin})"
