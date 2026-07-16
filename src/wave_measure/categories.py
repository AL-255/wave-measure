"""Operator categories.

Operators are grouped into domains -- ``filter``, ``amplitude``, ``math`` -- and
reached two ways:

* **Chained on a waveform** (the primary surface): each category is an accessor
  bound to the waveform, so methods build the operator, append it to the lazy
  pipeline, and return a new :class:`~wave_measure.waveform.Waveform` ::

      wave.filter.iir(b, a).math.diff().amplitude.histogram()

  Length-preserving methods return a waveform (keep chaining); terminal
  reductions (``histogram``, ``min``, ``max``, ...) return a result.

* **As a module-level catalog** (``wm.filter``, ``wm.amplitude``, ``wm.math``):
  factories that build standalone operator objects for discovery and advanced
  use, e.g. ``Waveform(source=src, ops=[wm.filter.fir(taps)])``.
"""

from __future__ import annotations

from types import SimpleNamespace

from .operators import (
    AbsOp,
    AffineOp,
    ClipOp,
    DiffOp,
    FirOp,
    HighPassOp,
    IirOp,
    LogOp,
    LowPassOp,
    MedianOp,
    SquareOp,
    SqrtOp,
    moving_average_op,
)
from .reductions import (
    stream_histogram,
    stream_levels,
    stream_peaks,
    stream_stats,
)

__all__ = ["FilterCategory", "AmplitudeCategory", "MathCategory", "filter", "amplitude", "math"]


class _Bound:
    """A category accessor bound to one waveform."""

    __slots__ = ("_wave",)

    def __init__(self, wave) -> None:
        self._wave = wave


class FilterCategory(_Bound):
    """Filtering operators (length-preserving)."""

    def fir(self, coeffs):
        """Causal FIR filter from explicit coefficients."""
        return self._wave._derive(FirOp(coeffs))

    def iir(self, b, a=(1.0,), **kwargs):
        """Generic IIR filter with numerator ``b`` and denominator ``a``."""
        return self._wave._derive(IirOp(b, a, **kwargs))

    def lowpass(self, cutoff, **kwargs):
        """One-pole low-pass at ``cutoff`` Hz."""
        return self._wave._derive(LowPassOp(cutoff, self._wave.sample_rate, **kwargs))

    def highpass(self, cutoff, **kwargs):
        """One-pole high-pass at ``cutoff`` Hz."""
        return self._wave._derive(HighPassOp(cutoff, self._wave.sample_rate, **kwargs))

    def bandpass(self, low, high, **kwargs):
        """Band-pass = high-pass at ``low`` then low-pass at ``high``."""
        return self.highpass(low, **kwargs).filter.lowpass(high, **kwargs)

    def moving_average(self, window):
        """Causal running-mean (box-car) of ``window`` samples."""
        return self._wave._derive(moving_average_op(window))

    def median(self, window):
        """Centered sliding-median of odd length ``window``."""
        return self._wave._derive(MedianOp(window))


class MathCategory(_Bound):
    """Point-wise math operators (length-preserving)."""

    def diff(self):
        """First difference (discrete derivative)."""
        return self._wave._derive(DiffOp())

    def square(self):
        """Square each sample."""
        return self._wave._derive(SquareOp())

    def sqrt(self):
        """Square root of each sample."""
        return self._wave._derive(SqrtOp())

    def log(self, base=None):
        """Logarithm (natural by default, else base ``base``)."""
        return self._wave._derive(LogOp(base))


class AmplitudeCategory(_Bound):
    """Amplitude-domain operators and reductions.

    ``abs``/``clip``/``gain``/``offset`` are length-preserving (return a
    waveform); ``histogram``/``min``/``max``/``mean``/``rms``/``peak_to_peak``/
    ``peaks`` are terminal reductions that stream the whole signal.
    """

    # -- length-preserving --
    def abs(self):
        """Absolute value, sample by sample."""
        return self._wave._derive(AbsOp())

    def clip(self, lo, hi):
        """Clamp samples to the range ``[lo, hi]``."""
        return self._wave._derive(ClipOp(lo, hi))

    def gain(self, gain):
        """Multiply samples by ``gain``."""
        return self._wave._derive(AffineOp(gain=gain))

    def offset(self, offset):
        """Add a constant ``offset`` to every sample."""
        return self._wave._derive(AffineOp(offset=offset))

    # -- terminal reductions --
    def histogram(self, bins=256, value_range=None, *, block=1 << 20):
        """Amplitude histogram over the whole stream (returns a
        :class:`~wave_measure.reductions.Histogram`)."""
        return stream_histogram(self._wave, bins=bins, value_range=value_range, block=block)

    def stats(self, *, block=1 << 20):
        """Single-pass count/min/max/mean/rms/std (a
        :class:`~wave_measure.reductions.Stats`)."""
        return stream_stats(self._wave, block=block)

    def min(self, *, block=1 << 20):
        """Minimum sample value (streamed)."""
        return self.stats(block=block).min

    def max(self, *, block=1 << 20):
        """Maximum sample value (streamed)."""
        return self.stats(block=block).max

    def mean(self, *, block=1 << 20):
        """Mean (DC) level (streamed)."""
        return self.stats(block=block).mean

    def rms(self, *, block=1 << 20):
        """Root-mean-square amplitude (streamed)."""
        return self.stats(block=block).rms

    def peak_to_peak(self, *, block=1 << 20):
        """Peak-to-peak amplitude, ``max - min`` (streamed)."""
        return self.stats(block=block).peak_to_peak

    def peaks(self, *, height=None, distance=1, block=1 << 20):
        """Find peaks across the whole stream (returns
        :class:`~wave_measure.reductions.Peaks`)."""
        return stream_peaks(self._wave, height=height, distance=distance, block=block)

    def levels(self, bins=256, value_range=None, *, block=1 << 20):
        """Fit two Gaussians to the amplitude histogram; return both logic levels."""
        return stream_levels(self._wave, bins=bins, value_range=value_range, block=block)

    def top(self, bins=256, value_range=None, *, block=1 << 20):
        """High logic level: mean of the upper Gaussian of the amplitude histogram."""
        return self.levels(bins=bins, value_range=value_range, block=block).top

    def bottom(self, bins=256, value_range=None, *, block=1 << 20):
        """Low logic level: mean of the lower Gaussian of the amplitude histogram."""
        return self.levels(bins=bins, value_range=value_range, block=block).bottom


# -- module-level catalog (standalone operator factories) -------------------
# Primary usage is the chained accessors above; these mirror the categories for
# discovery and for building reusable operators independent of any waveform.

filter = SimpleNamespace(
    fir=lambda coeffs: FirOp(coeffs),
    iir=lambda b, a=(1.0,), **kw: IirOp(b, a, **kw),
    lowpass=lambda cutoff, sample_rate, **kw: LowPassOp(cutoff, sample_rate, **kw),
    highpass=lambda cutoff, sample_rate, **kw: HighPassOp(cutoff, sample_rate, **kw),
    moving_average=lambda window: moving_average_op(window),
    median=lambda window: MedianOp(window),
)

amplitude = SimpleNamespace(
    # length-preserving operator factories
    abs=lambda: AbsOp(),
    clip=lambda lo, hi: ClipOp(lo, hi),
    gain=lambda gain: AffineOp(gain=gain),
    offset=lambda offset: AffineOp(offset=offset),
    # functional reductions over a waveform (parity with wave.amplitude.*)
    histogram=lambda wave, bins=256, value_range=None, **kw: wave.amplitude.histogram(
        bins=bins, value_range=value_range, **kw
    ),
    stats=lambda wave, **kw: wave.amplitude.stats(**kw),
    min=lambda wave, **kw: wave.amplitude.min(**kw),
    max=lambda wave, **kw: wave.amplitude.max(**kw),
    mean=lambda wave, **kw: wave.amplitude.mean(**kw),
    rms=lambda wave, **kw: wave.amplitude.rms(**kw),
    peak_to_peak=lambda wave, **kw: wave.amplitude.peak_to_peak(**kw),
    peaks=lambda wave, **kw: wave.amplitude.peaks(**kw),
    levels=lambda wave, **kw: wave.amplitude.levels(**kw),
    top=lambda wave, **kw: wave.amplitude.top(**kw),
    bottom=lambda wave, **kw: wave.amplitude.bottom(**kw),
)

math = SimpleNamespace(
    diff=lambda: DiffOp(),
    square=lambda: SquareOp(),
    sqrt=lambda: SqrtOp(),
    log=lambda base=None: LogOp(base),
)
