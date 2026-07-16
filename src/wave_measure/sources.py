"""Streaming data sources for waveforms.

A :class:`Source` is the file (or array) at the root of a lazy waveform. It
never loads the whole signal: it exposes a length and a ``read(start, stop)``
that returns just the requested sample range as ``float64``. This is what lets
wave-measure operate on captures far larger than memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["Source", "ArraySource", "RawBinaryReader"]

PathLike = Union[str, Path]


class Source(ABC):
    """Random-access provider of a uniformly-sampled signal.

    Implementations must define :meth:`__len__` and :meth:`read`. The time base
    is uniform by default (``t0 + index / sample_rate``); override :meth:`time`
    for non-uniform sampling.
    """

    sample_rate: float
    t0: float

    @abstractmethod
    def __len__(self) -> int:  # number of samples
        ...

    @abstractmethod
    def read(self, start: int, stop: int) -> np.ndarray:
        """Return samples ``[start, stop)`` as ``float64`` (clamped to bounds)."""

    def time(self, start: int, stop: int) -> np.ndarray:
        """Timestamps (seconds) for samples ``[start, stop)``."""
        start, stop = self._clamp(start, stop)
        idx = np.arange(start, stop, dtype=np.float64)
        return self.t0 + idx / self.sample_rate

    def _clamp(self, start: int, stop: int) -> tuple[int, int]:
        n = len(self)
        start = max(0, min(int(start), n))
        stop = max(start, min(int(stop), n))
        return start, stop


class ArraySource(Source):
    """An in-memory array dressed up as a streaming source.

    Used both for small signals built directly from arrays and for the
    materialized chunks returned by ``get_from_to``/``get_next``.
    """

    def __init__(
        self,
        samples,
        sample_rate: float = 1.0,
        t0: float = 0.0,
        time: Optional[np.ndarray] = None,
    ) -> None:
        self._samples = np.asarray(samples, dtype=float).ravel()
        self._time = None if time is None else np.asarray(time, dtype=float).ravel()
        if self._time is not None and self._time.size != self._samples.size:
            raise ValueError(
                f"time and samples must be the same length, got "
                f"{self._time.size} and {self._samples.size}"
            )
        if self._time is not None and self._time.size >= 2:
            dt = float(np.mean(np.diff(self._time)))
            sample_rate = 1.0 / dt if dt else sample_rate
            t0 = float(self._time[0])
        self.sample_rate = float(sample_rate)
        self.t0 = float(t0)

    def __len__(self) -> int:
        return int(self._samples.size)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Return samples ``[start, stop)`` as ``float64``."""
        start, stop = self._clamp(start, stop)
        return self._samples[start:stop]

    def time(self, start: int, stop: int) -> np.ndarray:
        """Timestamps for ``[start, stop)`` (explicit if given, else uniform)."""
        if self._time is not None:
            start, stop = self._clamp(start, stop)
            return self._time[start:stop]
        return super().time(start, stop)

    @property
    def array(self) -> np.ndarray:
        """The underlying sample array."""
        return self._samples


class RawBinaryReader(Source):
    """Memory-mapped reader for flat binary captures (the common scope export).

    The file is interpreted as ``count`` samples of ``dtype`` after an optional
    ``header_bytes`` prefix. Raw ADC values are converted to engineering units
    as ``value = raw * gain + offset``. Nothing is read until :meth:`read` is
    called, and only the requested slice is touched.

    Parameters
    ----------
    path:
        Binary file to map.
    dtype:
        Sample dtype on disk (e.g. ``"int16"``, ``"float32"``).
    sample_rate:
        Sampling rate in hertz.
    header_bytes:
        Bytes to skip before the first sample.
    gain, offset:
        Linear calibration applied on read (counts -> volts, say).
    count:
        Number of samples; inferred from the file size when omitted.
    t0:
        Timestamp of the first sample, in seconds.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        dtype: Union[str, np.dtype] = "int16",
        sample_rate: float,
        header_bytes: int = 0,
        gain: float = 1.0,
        offset: float = 0.0,
        count: Optional[int] = None,
        t0: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.sample_rate = float(sample_rate)
        self.header_bytes = int(header_bytes)
        self.gain = float(gain)
        self.offset = float(offset)
        self.t0 = float(t0)

        if count is None:
            file_bytes = self.path.stat().st_size - self.header_bytes
            count = file_bytes // self.dtype.itemsize
        self._count = int(count)
        # mmap is lazy: pages are only faulted in when actually sliced.
        self._mmap = np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            offset=self.header_bytes,
            shape=(self._count,),
        )

    def __len__(self) -> int:
        return self._count

    def read(self, start: int, stop: int) -> np.ndarray:
        """Return calibrated samples ``[start, stop)`` as ``float64``."""
        start, stop = self._clamp(start, stop)
        raw = np.asarray(self._mmap[start:stop], dtype=np.float64)
        if self.gain != 1.0 or self.offset != 0.0:
            raw = raw * self.gain + self.offset
        return raw
