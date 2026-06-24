"""The lazy, streaming waveform model.

A :class:`Waveform` is a node in a lazy pipeline, not a buffer of samples. It is
a :class:`~wave_measure.sources.Source` (a file or an in-memory array) plus a
chain of length-preserving :class:`~wave_measure.operators.Operator` s. Building
a chain (``wave.amplitude.abs().filter.lowpass(1e6).math.diff()``) reads nothing;
data is pulled only when you ask for a range:

* ``get_from_to(start, stop)`` -- random access to a sample range.
* ``get_next(n)`` -- a sequential cursor over the whole signal.
* ``blocks(size)`` -- iterate the signal in bounded-memory chunks.

Both pull paths fetch each operator's required margin of context, so results are
correct even when a slice starts in the middle of a filter's response.

In-memory waveforms (``Waveform(samples=..., time=...)``) are just the array
wrapped in an :class:`~wave_measure.sources.ArraySource`, so the same object
serves small signals and 100 GB files alike.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np

from .operators import AffineOp, Operator
from .sources import ArraySource, Source

__all__ = ["Waveform"]

# Guard: refuse to silently materialize more than this many samples into RAM.
_MATERIALIZE_LIMIT = 64_000_000


class Waveform:
    """A lazily-evaluated, streamable signal."""

    def __init__(
        self,
        samples=None,
        time=None,
        sample_rate: Optional[float] = None,
        metadata: Optional[dict] = None,
        *,
        source: Optional[Source] = None,
        ops: Tuple[Operator, ...] = (),
    ) -> None:
        if source is None:
            # Construct an in-memory waveform from arrays (back-compat path).
            if samples is None:
                raise TypeError("Waveform requires either `samples` or a `source`")
            rate = float(sample_rate) if sample_rate else 1.0
            source = ArraySource(samples, sample_rate=rate, time=time)
        self._source = source
        self._ops: Tuple[Operator, ...] = tuple(ops)
        self.metadata: dict[str, Any] = dict(metadata or {})

        self._left_margin = sum(op.left_margin for op in self._ops)
        self._right_margin = sum(op.right_margin for op in self._ops)
        self._cursor = 0

    # -- construction helpers ---------------------------------------------

    def _derive(self, op: Operator, **meta: Any) -> "Waveform":
        """Return a new node with ``op`` appended to the pipeline."""
        return Waveform(
            source=self._source,
            ops=self._ops + (op,),
            metadata={**self.metadata, **meta},
        )

    # -- descriptors -------------------------------------------------------

    @property
    def source(self) -> Source:
        return self._source

    @property
    def ops(self) -> Tuple[Operator, ...]:
        return self._ops

    @property
    def sample_rate(self) -> float:
        return self._source.sample_rate

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate if self.sample_rate else 0.0

    def __len__(self) -> int:  # length-preserving pipeline
        return len(self._source)

    @property
    def duration(self) -> float:
        n = len(self)
        return (n - 1) / self.sample_rate if n >= 2 and self.sample_rate else 0.0

    @property
    def is_streaming(self) -> bool:
        """True when this node is not a plain in-memory array (has ops or a file)."""
        return bool(self._ops) or not isinstance(self._source, ArraySource)

    # -- pull / streaming --------------------------------------------------

    def _read_core(self, start: int, stop: int) -> np.ndarray:
        """Apply the pipeline and return values for global range ``[start, stop)``."""
        n = len(self)
        start = max(0, min(int(start), n))
        stop = max(start, min(int(stop), n))
        s = max(0, start - self._left_margin)
        e = min(n, stop + self._right_margin)

        data = self._source.read(s, e)
        for op in self._ops:
            data = op.apply(data)

        off = start - s  # local offset of the first wanted sample
        return data[off:off + (stop - start)]

    def get_from_to(self, start: int, stop: int) -> "Waveform":
        """Materialize the sample range ``[start, stop)`` into an in-memory waveform."""
        values = self._read_core(start, stop)
        time = self._source.time(start, stop)
        return Waveform(samples=values, time=time, metadata=dict(self.metadata))

    def get_next(self, n: int) -> Optional["Waveform"]:
        """Return the next ``n`` samples from the sequential cursor (``None`` at end)."""
        if self._cursor >= len(self):
            return None
        start, stop = self._cursor, min(self._cursor + int(n), len(self))
        self._cursor = stop
        return self.get_from_to(start, stop)

    def reset(self) -> "Waveform":
        """Rewind the ``get_next`` cursor to the start."""
        self._cursor = 0
        return self

    def blocks(self, size: int = 1 << 20) -> Iterator["Waveform"]:
        """Iterate the whole signal in chunks of ``size`` samples."""
        size = max(1, int(size))
        n = len(self)
        for start in range(0, n, size):
            yield self.get_from_to(start, min(start + size, n))

    def __iter__(self) -> Iterator["Waveform"]:
        return self.blocks()

    def to_array(self, *, allow_large: bool = False) -> np.ndarray:
        """Materialize the entire pipeline output to a single array."""
        if not self._ops and isinstance(self._source, ArraySource):
            return self._source.array
        n = len(self)
        if n > _MATERIALIZE_LIMIT and not allow_large:
            raise MemoryError(
                f"refusing to materialize {n:,} samples into RAM; stream with "
                "blocks()/get_from_to(), or pass allow_large=True"
            )
        return self._read_core(0, n)

    def to_file(self, path, *, dtype="float32", block: int = 1 << 20) -> None:
        """Stream the pipeline output to a flat binary file, block by block."""
        dt = np.dtype(dtype)
        with open(path, "wb") as fh:
            for chunk in self.blocks(block):
                fh.write(chunk.samples.astype(dt, copy=False).tobytes())

    # -- array-like / in-memory compatibility ------------------------------

    @property
    def samples(self) -> np.ndarray:
        return self.to_array()

    @property
    def time(self) -> np.ndarray:
        if not self._ops and isinstance(self._source, ArraySource):
            return self._source.time(0, len(self))
        if len(self) > _MATERIALIZE_LIMIT:
            raise MemoryError(
                "refusing to build a full time base for a large streaming "
                "waveform; use get_from_to(...).time instead"
            )
        return self._source.time(0, len(self))

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.to_array(), dtype=dtype)

    # -- categorized operators (chainable) --------------------------------

    @property
    def filter(self):
        """Filtering operators: ``fir``, ``iir``, ``lowpass``, ``highpass``, ..."""
        from .categories import FilterCategory

        return FilterCategory(self)

    @property
    def amplitude(self):
        """Amplitude operators & reductions: ``abs``, ``clip``, ``histogram``, ..."""
        from .categories import AmplitudeCategory

        return AmplitudeCategory(self)

    @property
    def math(self):
        """Point-wise math: ``diff``, ``square``, ``sqrt``, ``log``."""
        from .categories import MathCategory

        return MathCategory(self)

    # -- scalar arithmetic (lazy) -----------------------------------------

    def __mul__(self, other: Any) -> "Waveform":
        if np.isscalar(other):
            return self._derive(AffineOp(gain=float(other)))
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> "Waveform":
        if np.isscalar(other):
            return self._derive(AffineOp(offset=float(other)))
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Waveform":
        if np.isscalar(other):
            return self._derive(AffineOp(offset=-float(other)))
        return self._binary(other, np.subtract)

    def __truediv__(self, other: Any) -> "Waveform":
        if np.isscalar(other):
            return self._derive(AffineOp(gain=1.0 / float(other)))
        return self._binary(other, np.true_divide)

    def _binary(self, other: "Waveform", op) -> "Waveform":
        # Two-input ops fall outside the single-source streaming model; support
        # them by materializing when both sides are small.
        a = self.to_array()
        b = other.to_array() if isinstance(other, Waveform) else np.asarray(other)
        return Waveform(samples=op(a, b), time=self.time, metadata=dict(self.metadata))

    # -- misc --------------------------------------------------------------

    def copy(self) -> "Waveform":
        """Return an independent waveform with the same pipeline and metadata."""
        return Waveform(
            source=self._source, ops=self._ops, metadata=dict(self.metadata)
        )

    def with_metadata(self, **kwargs: Any) -> "Waveform":
        return Waveform(
            source=self._source,
            ops=self._ops,
            metadata={**self.metadata, **kwargs},
        )

    def __repr__(self) -> str:
        chain = "".join(f".{op.name}" for op in self._ops)
        src = type(self._source).__name__
        return (
            f"Waveform(n={len(self)}, sample_rate={self.sample_rate:g} Hz, "
            f"source={src}{chain})"
        )
