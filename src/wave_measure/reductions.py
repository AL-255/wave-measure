"""Terminal reductions that consume a stream and aggregate it.

Unlike the length-preserving operators, these change what a "sample" means, so
they are *terminal*: they pull the whole (chunked) stream once and return an
aggregate rather than another streamable waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "Histogram",
    "Peaks",
    "Stats",
    "Levels",
    "stream_histogram",
    "stream_peaks",
    "stream_stats",
    "stream_levels",
]


@dataclass
class Stats:
    """Single-pass amplitude statistics over a whole stream."""

    count: int
    min: float
    max: float
    mean: float
    rms: float
    std: float

    @property
    def peak_to_peak(self) -> float:
        return self.max - self.min

    def __repr__(self) -> str:
        return (
            f"Stats(count={self.count}, min={self.min:g}, max={self.max:g}, "
            f"mean={self.mean:g}, rms={self.rms:g})"
        )


def stream_stats(waveform, *, block: int = 1 << 20) -> Stats:
    """Compute count/min/max/mean/rms/std in one bounded-memory streaming pass."""
    count = 0
    vmin = np.inf
    vmax = -np.inf
    total = 0.0
    total_sq = 0.0
    for chunk in waveform.blocks(block):
        v = chunk.samples
        if v.size:
            count += int(v.size)
            vmin = min(vmin, float(v.min()))
            vmax = max(vmax, float(v.max()))
            total += float(v.sum())
            total_sq += float(np.dot(v, v))
    if count == 0:
        nan = float("nan")
        return Stats(0, nan, nan, nan, nan, nan)
    mean = total / count
    rms = (total_sq / count) ** 0.5
    var = max(0.0, total_sq / count - mean * mean)
    return Stats(count, vmin, vmax, mean, rms, var ** 0.5)


@dataclass
class Histogram:
    """Counts per amplitude bin, accumulated across the whole stream."""

    counts: np.ndarray
    edges: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def __repr__(self) -> str:
        return f"Histogram(bins={self.counts.size}, total={self.total})"


@dataclass
class Peaks:
    """Detected peaks as parallel arrays of sample index, time, and value."""

    index: np.ndarray
    time: np.ndarray
    value: np.ndarray

    def __len__(self) -> int:
        return int(self.index.size)

    def __repr__(self) -> str:
        return f"Peaks(n={len(self)})"


def stream_histogram(
    waveform,
    bins: int = 256,
    value_range: Optional[Tuple[float, float]] = None,
    *,
    block: int = 1 << 20,
) -> Histogram:
    """Histogram a waveform in bounded memory by accumulating per block.

    ``value_range`` defaults to a first streaming pass for the min/max, so a
    range-less call costs two passes over the data; pass an explicit range to
    do it in one.
    """
    if value_range is None:
        lo = np.inf
        hi = -np.inf
        for chunk in waveform.blocks(block):
            v = chunk.samples
            if v.size:
                lo = min(lo, float(v.min()))
                hi = max(hi, float(v.max()))
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if lo == hi:
            hi = lo + 1.0
        value_range = (lo, hi)

    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    for chunk in waveform.blocks(block):
        c, _ = np.histogram(chunk.samples, bins=edges)
        counts += c
    return Histogram(counts=counts, edges=edges)


def stream_peaks(
    waveform,
    *,
    height: Optional[float] = None,
    distance: int = 1,
    block: int = 1 << 20,
) -> Peaks:
    """Find local maxima across the stream, stitching across block boundaries.

    A peak is a strict local maximum at least ``height`` tall (when given). The
    ``distance`` guard suppresses peaks closer than that many samples and also
    sizes the overlap carried between blocks so boundary peaks are not missed.
    """
    distance = max(1, int(distance))
    overlap = distance + 1
    idx_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    n = len(waveform)
    pos = 0
    last_kept = -distance - 1
    while pos < n:
        stop = min(pos + block, n)
        # Pull a little context on each side so peaks at the seam are seen once.
        lo = max(0, pos - overlap)
        hi = min(n, stop + overlap)
        v = waveform.get_from_to(lo, hi).samples
        local = _local_maxima(v, height)
        for j in local:
            g = lo + int(j)  # global index
            if g < pos or g >= stop:
                continue  # owned by an adjacent block's core
            if g - last_kept < distance:
                continue
            idx_parts.append(g)
            val_parts.append(v[j])
            last_kept = g
        pos = stop

    if idx_parts:
        index = np.asarray(idx_parts, dtype=np.int64)
        value = np.asarray(val_parts, dtype=float)
    else:
        index = np.empty(0, dtype=np.int64)
        value = np.empty(0, dtype=float)
    time = waveform.source.t0 + index / waveform.sample_rate
    return Peaks(index=index, time=time, value=value)


def _local_maxima(v: np.ndarray, height: Optional[float]) -> np.ndarray:
    if v.size < 3:
        return np.empty(0, dtype=np.int64)
    middle = v[1:-1]
    is_peak = (middle > v[:-2]) & (middle >= v[2:])
    if height is not None:
        is_peak &= middle >= height
    return np.nonzero(is_peak)[0] + 1


@dataclass
class Levels:
    """Logic levels from a two-Gaussian fit of the amplitude histogram.

    The two fitted components correspond to the signal's low and high dwell
    levels; ``bottom`` and ``top`` are their means.
    """

    bottom: float
    top: float
    means: Tuple[float, float]
    sigmas: Tuple[float, float]
    weights: Tuple[float, float]

    @property
    def amplitude(self) -> float:
        return self.top - self.bottom

    def __repr__(self) -> str:
        return f"Levels(bottom={self.bottom:g}, top={self.top:g})"


def _gaussian_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _fit_two_gaussians(centers, counts, *, max_iter: int = 300, tol: float = 1e-8):
    """Fit a 2-component Gaussian mixture to a histogram via weighted EM.

    Treats each bin center as a data point weighted by its count. Returns
    ``(means, sigmas, weights)`` sorted so index 0 is the lower (bottom) mode.
    """
    c = np.asarray(centers, dtype=float)
    w = np.asarray(counts, dtype=float)
    total = w.sum()
    if total <= 0 or np.count_nonzero(w) < 2:
        m = float(c[np.argmax(w)]) if w.size and total > 0 else float("nan")
        return np.array([m, m]), np.array([0.0, 0.0]), np.array([0.5, 0.5])

    spacing = float(c[1] - c[0]) if c.size > 1 else 1.0
    span = float(c[-1] - c[0]) or 1.0
    var_floor = max(spacing * spacing, (span * 1e-6) ** 2)

    # Initialize the two means from the centroids either side of the mean level.
    grand_mean = np.average(c, weights=w)

    def _centroid(mask):
        ww = w * mask
        s = ww.sum()
        return float(np.average(c, weights=ww)) if s > 0 else grand_mean

    mu = np.array([_centroid(c <= grand_mean), _centroid(c > grand_mean)])
    if mu[0] == mu[1]:  # unimodal: nudge apart so EM can separate
        mu = np.array([grand_mean - spacing, grand_mean + spacing])
    var = np.full(2, max(np.average((c - grand_mean) ** 2, weights=w), var_floor))
    pi = np.array([0.5, 0.5])

    prev_ll = -np.inf
    for _ in range(max_iter):
        g0 = pi[0] * _gaussian_pdf(c, mu[0], var[0])
        g1 = pi[1] * _gaussian_pdf(c, mu[1], var[1])
        denom = g0 + g1
        denom = np.where(denom <= 0, 1e-300, denom)
        r0, r1 = g0 / denom, g1 / denom

        n0 = float((w * r0).sum())
        n1 = float((w * r1).sum())
        if n0 <= 0 or n1 <= 0:
            break
        mu = np.array([(w * r0 * c).sum() / n0, (w * r1 * c).sum() / n1])
        var = np.array(
            [
                (w * r0 * (c - mu[0]) ** 2).sum() / n0,
                (w * r1 * (c - mu[1]) ** 2).sum() / n1,
            ]
        )
        var = np.maximum(var, var_floor)
        pi = np.array([n0, n1]) / total

        ll = float((w * np.log(denom)).sum())
        if abs(ll - prev_ll) <= tol * (abs(ll) + 1.0):
            break
        prev_ll = ll

    order = np.argsort(mu)
    return mu[order], np.sqrt(var[order]), pi[order]


def stream_levels(
    waveform,
    bins: int = 256,
    value_range: Optional[Tuple[float, float]] = None,
    *,
    block: int = 1 << 20,
) -> Levels:
    """Estimate bottom/top logic levels by fitting two Gaussians to the
    streamed amplitude histogram."""
    hist = stream_histogram(waveform, bins=bins, value_range=value_range, block=block)
    means, sigmas, weights = _fit_two_gaussians(hist.centers, hist.counts)
    return Levels(
        bottom=float(means[0]),
        top=float(means[1]),
        means=(float(means[0]), float(means[1])),
        sigmas=(float(sigmas[0]), float(sigmas[1])),
        weights=(float(weights[0]), float(weights[1])),
    )
