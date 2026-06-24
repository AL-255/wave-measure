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
    "stream_histogram",
    "stream_peaks",
    "stream_stats",
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
