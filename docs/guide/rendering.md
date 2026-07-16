# Digital-phosphor rendering

`plt.plot` aliases and crawls on captures with 10⁸+ samples.
{py:func}`~wave_measure.render` instead draws them the way a digital-phosphor
oscilloscope (DPO) does: it rasterizes the line segments between consecutive
samples (Bresenham) into a 2-D accumulation histogram, so dense regions glow.
It runs on **CUDA or CPU** (multi-core) and **streams** the whole waveform in
bounded memory.

```python
import matplotlib.pyplot as plt
import wave_measure as wm

wave = wm.read_raw("capture.bin", dtype="int16", sample_rate=10e6)

img = wm.render(wave, width=1000, height=500)   # RGBA uint8 image
plt.imshow(img)                                  # display
plt.imsave("scope.png", img)                     # ...or save
# or save directly:
wm.render(wave, width=1000, height=500, path="scope.png")
```

`render` always draws the **whole** waveform you give it — to render a slice,
pass one: `wm.render(wave.get_from_to(a, b))`.

## Examples

These reproduce the demonstrations from Lithcore's
[Python multi-core / GPU digital phosphor][blog] article, whose CUDA
implementation this renderer is based on.

### AM signal

A carrier modulated by a slow sine. The carrier fills the modulation envelope,
and phosphor accumulation glows brightest at the envelope nodes where the trace
density peaks.

```python
import numpy as np
import wave_measure as wm

fs, n = 1e6, 8_000_000
t = np.arange(n) / fs
carrier, modulation = 1000.0, 1.0
y = np.sin(2 * np.pi * carrier * t) * (0.5 + 0.5 * np.sin(2 * np.pi * modulation * t))

am = wm.from_array(y.astype(np.float32), sample_rate=fs)
wm.render(am, width=1100, height=440, cmap="inferno", path="am.png")
```

```{image} ../_static/dpo_am_signal.png
:alt: Digital-phosphor render of an amplitude-modulated signal
:width: 100%
```

### Random walk

A 30-million-sample Brownian walk. Because the trace is single-valued in time,
the glow comes from the many samples that fall in each pixel column — brightest
where the walk lingers or reverses.

```python
walk = np.cumsum(np.random.default_rng(1).standard_normal(30_000_000))
rw = wm.from_array(walk.astype(np.float32), sample_rate=1.0)
wm.render(rw, width=1100, height=440, cmap="magma", path="walk.png")
```

```{image} ../_static/dpo_random_walk.png
:alt: Digital-phosphor render of a random walk
:width: 100%
```

[blog]: https://www.lithcore.net/2025/02/python-multi-core-gpu-digital-phosphor.html

## Options

- `width`, `height` — output image size in pixels.
- `cmap` — any matplotlib colormap (default `"inferno"` for a phosphor look).
- `scale` — intensity compression: `"sqrt"` (default), `"log"`, or `"linear"`.
- `x_range`, `y_range` — data bounds to map; default to the full time span and
  the (streamed) amplitude min/max.
- `backend` — `"cuda"`, `"cpu"`, or auto from the detected accelerator. An
  unusable CUDA request falls back to CPU with a warning.
- `workers` — CPU worker count (defaults to all cores).
- `path` — if given, also save the image.

## Data-unit axes

`render` returns a raster with no axes. For labeled axes, use the lower-level
{py:func}`~wave_measure.dpo_histogram`, which returns the raw counts and extent:

```python
hist, extent = wm.dpo_histogram(wave, width=1000, height=500)
plt.imshow(hist ** 0.5, origin="lower", aspect="auto", extent=extent)
```
