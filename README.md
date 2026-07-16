# wave-measure

A toolbox for manipulating, measuring, rendering, and analyzing
oscilloscope-captured waveform data. It is designed as a small, importable
Python module that other projects can build on.

## Features

- **Core data model** — a lightweight `Waveform` holding a time base and
  samples, with convenient slicing and arithmetic.
- **I/O** — read waveforms from the CSV exports produced by most benchtop
  oscilloscopes, and write them back out.
- **Measurements** — amplitude (Vpp, Vrms, mean), timing (period, frequency,
  rise/fall time, duty cycle), and statistics.
- **Analysis** — FFT/spectrum and simple filtering helpers.
- **Rendering** — quick plots of waveforms and spectra (requires
  `matplotlib`).
- **Acceleration** — hot loops are JIT-compiled with [Numba](https://numba.pydata.org/).
  On import the package detects the best available accelerator (NVIDIA GPU via
  CUDA, or a CPU with auto-vectorized SIMD) and reports it.

## Acceleration

Waveform captures can be huge, so wave-measure JIT-compiles its sequential
kernels with Numba and probes the host for an accelerator at import time:

```python
import wave_measure as wm

print(wm.accelerator)
# wave-measure accelerator → backend=cuda; NVIDIA GPU (NVIDIA GeForce RTX 5060 ...);
#   SIMD=avx512f,avx2,fma,avx,...; numba=0.61.0
```

`wm.accelerator` is an `AcceleratorInfo` with fields like `backend`
(`"cuda"`/`"cpu"`), `hardware`, `simd_features`, `cuda_available`, and
`amd_gpu_detected`. Notes:

- **NVIDIA** GPUs are used through `numba.cuda` when a driver and device exist.
- **AMD** GPUs are detected and reported, but not used for compute: mainline
  Numba removed its ROCm/HSA backend in 0.54, so wave-measure falls back to the
  CPU (and says so in `accelerator.notes`).
- **CPU** is the default; Numba's LLVM backend auto-vectorizes kernels with the
  host's SIMD ISA (AVX-512, AVX2/FMA, SSE, NEON, ...).

NVIDIA selection is verified by actually compiling and running a tiny kernel,
not just by `numba.cuda.is_available()` — this catches toolchain mismatches
(e.g. a `ptxas` too old for the PTX Numba emits) that would otherwise fail at
the first real kernel. If the probe fails, the reason is recorded in
`accelerator.notes` and the backend falls back to CPU.

Environment overrides:

- `WAVE_MEASURE_SKIP_DETECT=1` — skip detection at import (leaves
  `wm.accelerator is None`; call `wm.get_accelerator()` on demand).
- `WAVE_MEASURE_BACKEND=cpu|cuda` — force the selected backend.
- `WAVE_MEASURE_NO_CUDA_PROBE=1` — trust `is_available()` and skip the
  functional CUDA kernel probe (faster import; less safe).

## Installation

```bash
pip install wave-measure            # core (numpy only)
pip install "wave-measure[all]"     # with rendering + analysis extras
```

For local development:

```bash
pip install -e ".[dev]"
```

## Streaming huge captures (lazy pipelines)

A `Waveform` is a **lazy node**, not a buffer: it is a *source* (a file or an
array) plus a chain of operators. Building a chain reads nothing — data is
pulled only for the range you ask for, so captures far larger than memory work
the same as small ones.

```python
import wave_measure as wm

# Memory-map a flat binary capture (100 GB is fine — nothing is loaded yet).
# Raw ADC counts are calibrated to volts via gain/offset.
wave_before = wm.read_raw("capture.bin", dtype="int16", sample_rate=1e6, gain=1e-3)

# Fluent, lazy pipeline. Operators are grouped into categories. Nothing computed.
wave_after = (wave_before
    .amplitude.abs()
    .filter.lowpass(cutoff=50_000)
    .math.diff())

# Pull only what you need:
chunk = wave_after.get_from_to(10_000, 20_000)   # random access -> in-memory Waveform
block = wave_after.get_next(4096)                # sequential cursor (call until None)
for blk in wave_after.blocks(1 << 20):           # bounded-memory iteration
    ...

wave_after.to_file("processed.f32", dtype="float32")  # stream result back to disk
```

The chunk returned by `get_from_to`/`get_next` is itself an (in-memory)
`Waveform`, so every measurement and plot works on it directly:

```python
chunk = wave_after.get_from_to(10_000, 20_000)
print(wm.vpp(chunk), wm.frequency(chunk))
```

### Operator categories

Operators are organized into domains, reached as chainable accessors on the
waveform:

| Category | Length-preserving (return a waveform) | Terminal reductions (return a result) |
|---|---|---|
| `wave.filter` | `fir`, `iir`, `lowpass`, `highpass`, `bandpass`, `moving_average`, `median` | — |
| `wave.amplitude` | `abs`, `clip`, `gain`, `offset` | `histogram`, `min`, `max`, `mean`, `rms`, `peak_to_peak`, `stats`, `peaks`, `top`, `bottom`, `levels` |
| `wave.math` | `diff`, `square`, `sqrt`, `log` | — |

```python
# chain freely across categories
spikes = wave_before.amplitude.abs().filter.median(7).math.diff()

# terminal reductions stream the whole signal in bounded memory
h     = wave_before.amplitude.histogram(bins=256)     # Histogram
stats = wave_before.amplitude.stats()                 # min/max/mean/rms/p2p
pk    = wave_before.amplitude.abs().filter.lowpass(20_000).amplitude.peaks(height=2.0)

# logic levels via a two-Gaussian fit of the amplitude histogram
top    = wave_before.amplitude.top()       # mean of the upper mode
bottom = wave_before.amplitude.bottom()    # mean of the lower mode
levels = wave_before.amplitude.levels()    # both, plus sigmas/weights
```

**Two kinds of operator.** *Length-preserving* ops return a new lazy `Waveform`
and each declares a **margin** of context, so random access through stateful
operators is correct even when a slice starts mid-filter (`fir`/`diff`/`median`
are exact; the IIR `lowpass`/`highpass`/`iir` reconverge within a finite warm-up
margin — correct to a tiny, documented tolerance). *Terminal reductions* consume
the stream once and return an aggregate (`Histogram`, `Peaks`, `Stats`).

The same categories exist at module level as a catalog — `wm.filter.fir(taps)`
builds a standalone operator for reuse, and `wm.amplitude.mean(wave)` is the
functional form of `wave.amplitude.mean()`:

```python
op = wm.filter.fir([0.25, 0.5, 0.25])
wave = wm.Waveform(source=src, ops=[op])     # advanced: build a pipeline directly
```

Custom binary formats: subclass `wm.Source` (implement `__len__` and
`read(start, stop)`) and pass it as `Waveform(source=...)`.

## Digital-phosphor rendering

`plt.plot` aliases and crawls on captures with 10^8+ samples. `wm.render`
instead draws them the way a digital-phosphor oscilloscope does: it rasterizes
the line segments between consecutive samples (Bresenham) into a 2-D
accumulation histogram, so dense regions glow. It runs on **CUDA or CPU**
(multi-core), and **streams** the whole waveform in bounded memory.

```python
import matplotlib.pyplot as plt
import wave_measure as wm

wave = wm.read_raw("capture.bin", dtype="int16", sample_rate=10e6)

img = wm.render(wave, width=1000, height=500)   # 2-D intensity image (uncolored)
plt.imshow(img, cmap="inferno")                  # colorize at display time
# or bake the colormap in and save:
img = wm.render(wave, width=1000, height=500, cmap="inferno")   # RGBA uint8
plt.imsave("scope.png", img)
```

`render` always draws the **whole** waveform you give it — to render a slice,
pass one: `wm.render(wave.get_from_to(a, b))`. Pass `x=` (a second channel or
array) for X-Y mode — a Lissajous figure or 2-D random walk. Key options: `cmap`
(omit for a raw intensity image, or pass any matplotlib colormap to bake in
RGBA), `scale` (`"sqrt"`/`"log"`/`"linear"` intensity compression),
`x_range`/`y_range`, `backend` (`"cuda"`/`"cpu"`/auto), and `workers`. The
backend defaults to the detected accelerator; an unusable CUDA request falls
back to CPU with a warning.

For axes in data units, use the lower-level `wm.dpo_histogram`, which returns the
raw `(counts, extent)`:

```python
hist, extent = wm.dpo_histogram(wave, width=1000, height=500)
plt.imshow(hist ** 0.5, origin="lower", aspect="auto", extent=extent)
```

## Quick start

```python
import wave_measure as wm

# Load a capture exported from an oscilloscope
wf = wm.read_csv("capture.csv")

# Take some measurements
print("Vpp:        ", wm.vpp(wf))
print("Vrms:       ", wm.vrms(wf))
print("Frequency:  ", wm.frequency(wf))
print("Rise time:  ", wm.rise_time(wf))

# Analyze and render
freqs, mag = wm.spectrum(wf)
wm.plot(wf)            # time-domain
wm.plot_spectrum(wf)  # frequency-domain
```

You can also build a `Waveform` directly from arrays:

```python
import numpy as np
import wave_measure as wm

t = np.linspace(0, 1e-3, 1000)
v = np.sin(2 * np.pi * 5000 * t)
wf = wm.Waveform(time=t, samples=v)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
