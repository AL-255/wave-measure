"""wave-measure: a toolbox for oscilloscope-captured waveform data.

Manipulate, measure, render, and analyze captured signals. The public API is
re-exported here so callers can simply ``import wave_measure as wm``.
"""

from __future__ import annotations

import logging
import os

from .accelerator import AcceleratorInfo, detect_accelerator, get_accelerator
from .analysis import detrend, dominant_frequency, moving_average, spectrum
from .categories import amplitude, filter, math
from .io import from_array, read_csv, read_raw, write_csv
from .reductions import Histogram, Peaks, Stats
from .sources import ArraySource, RawBinaryReader, Source
from .measure import (
    crossings,
    duty_cycle,
    fall_time,
    frequency,
    mean,
    period,
    rise_time,
    std,
    vmax,
    vmin,
    vpp,
    vrms,
)
from .render import plot, plot_spectrum
from .waveform import Waveform

__version__ = "0.1.0"

# Detect the host accelerator at import time so callers can dispatch on it.
# Set WAVE_MEASURE_SKIP_DETECT=1 to defer detection (e.g. for fast imports).
if os.environ.get("WAVE_MEASURE_SKIP_DETECT", "").strip().lower() in ("1", "true", "yes"):
    accelerator: AcceleratorInfo | None = None
else:
    accelerator = get_accelerator()
    logging.getLogger("wave_measure").info(accelerator.summary())

__all__ = [
    "__version__",
    "accelerator",
    "AcceleratorInfo",
    "detect_accelerator",
    "get_accelerator",
    "Waveform",
    # sources / streaming
    "Source",
    "ArraySource",
    "RawBinaryReader",
    "Histogram",
    "Peaks",
    "Stats",
    # operator categories (catalog; chain on a waveform: wave.filter.iir(...))
    "filter",
    "amplitude",
    "math",
    # io
    "read_csv",
    "write_csv",
    "read_raw",
    "from_array",
    # measure
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
    # analysis
    "spectrum",
    "dominant_frequency",
    "moving_average",
    "detrend",
    # render
    "plot",
    "plot_spectrum",
]
