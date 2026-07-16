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
