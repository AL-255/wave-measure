# wave-measure

A toolbox for manipulating, measuring, rendering, and analyzing
oscilloscope-captured waveform data. It is designed as a small, importable
Python module that other projects can build on, and is built around a **lazy,
streaming** model so it handles captures far larger than memory.

```python
import wave_measure as wm

# Memory-map a 100 GB capture; nothing is loaded yet.
wave = wm.read_raw("capture.bin", dtype="int16", sample_rate=1e6, gain=1e-3)

# Build a lazy pipeline from categorized operators.
processed = wave.amplitude.abs().filter.lowpass(50_000).math.diff()

# Pull only what you need.
chunk = processed.get_from_to(10_000, 20_000)   # random access
print(wm.vpp(chunk), wm.frequency(chunk))

# Render the whole capture as a digital-phosphor image.
img = wm.render(wave, width=1000, height=500)
```

## Highlights

- **Streaming core** — a `Waveform` is a lazy node (a file/array source plus an
  operator chain); data is pulled per range via `get_from_to`, `get_next`, and
  `blocks`, with operator margins keeping random access correct through filters.
- **Categorized operators** — chain `filter`, `amplitude`, and `math` operators
  fluently on a waveform.
- **Measurements** — amplitude, timing, logic levels, and statistics.
- **Digital-phosphor rendering** — draw 10⁸+ sample captures without aliasing.
- **Acceleration** — hot loops are JIT-compiled with Numba, and the best
  available accelerator (CUDA GPU or CPU SIMD) is detected at import.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guide/streaming
guide/rendering
guide/acceleration
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api/index
```
