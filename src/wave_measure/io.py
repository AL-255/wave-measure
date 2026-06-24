"""Reading and writing oscilloscope-captured waveform data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .sources import RawBinaryReader
from .waveform import Waveform

__all__ = ["read_csv", "write_csv", "read_raw", "from_array"]

PathLike = Union[str, Path]


def read_raw(
    path: PathLike,
    *,
    dtype: Union[str, np.dtype] = "int16",
    sample_rate: float,
    header_bytes: int = 0,
    gain: float = 1.0,
    offset: float = 0.0,
    count: Optional[int] = None,
    t0: float = 0.0,
    **metadata,
) -> Waveform:
    """Open a flat binary capture as a lazy, streaming :class:`Waveform`.

    Nothing is loaded up front: the file is memory-mapped and read in ranges as
    the pipeline pulls. Suitable for captures far larger than memory.

    See :class:`~wave_measure.sources.RawBinaryReader` for the parameters.
    """
    reader = RawBinaryReader(
        path,
        dtype=dtype,
        sample_rate=sample_rate,
        header_bytes=header_bytes,
        gain=gain,
        offset=offset,
        count=count,
        t0=t0,
    )
    return Waveform(source=reader, metadata=metadata)


def from_array(
    samples,
    *,
    sample_rate: float = 1.0,
    time=None,
    **metadata,
) -> Waveform:
    """Wrap an in-memory array as a :class:`Waveform` (handy, explicit alias)."""
    return Waveform(
        samples=samples, time=time, sample_rate=sample_rate, metadata=metadata
    )


def read_csv(
    path: PathLike,
    *,
    time_column: int = 0,
    value_column: int = 1,
    delimiter: str = ",",
) -> Waveform:
    """Read a waveform from a CSV file exported by an oscilloscope.

    Many benchtop scopes export a header block of ``key,value`` metadata lines
    followed by columns of numeric samples. This reader is forgiving: it skips
    any leading lines whose target columns are not numeric, collects the
    metadata it can parse along the way, and reads the rest as data.

    Parameters
    ----------
    path:
        File to read.
    time_column, value_column:
        Zero-based column indices for the time base and the sample values.
    delimiter:
        Field delimiter.
    """
    path = Path(path)
    times: list[float] = []
    values: list[float] = []
    metadata: dict[str, str] = {}

    with path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            try:
                t = float(row[time_column])
                v = float(row[value_column])
            except (ValueError, IndexError):
                # Treat a leading "key,value" line as metadata.
                if len(row) >= 2 and row[0].strip():
                    metadata.setdefault(row[0].strip(), row[1].strip())
                continue
            times.append(t)
            values.append(v)

    if not values:
        raise ValueError(f"no numeric samples found in {path}")

    return Waveform(
        samples=np.asarray(values, dtype=float),
        time=np.asarray(times, dtype=float),
        metadata=metadata,
    )


def write_csv(
    waveform: Waveform,
    path: PathLike,
    *,
    delimiter: str = ",",
    header: bool = True,
) -> None:
    """Write a waveform to a two-column ``time,value`` CSV file."""
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        if header:
            writer.writerow(["time", "value"])
        for t, v in zip(waveform.time, waveform.samples):
            writer.writerow([repr(float(t)), repr(float(v))])
