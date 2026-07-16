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

# By default render returns an uncolored 2-D intensity image; choose a colormap
# at display time.
img = wm.render(wave, width=1000, height=500)
plt.imshow(img, cmap="inferno")

# Or pass cmap to bake the colors in and save an RGBA image directly.
img = wm.render(wave, width=1000, height=500, cmap="inferno")
plt.imsave("scope.png", img)
wm.render(wave, width=1000, height=500, cmap="inferno", path="scope.png")
```

`render` always draws the **whole** waveform you give it — to render a slice,
pass one: `wm.render(wave.get_from_to(a, b))`.

## Plotting with axes

`render` returns a bare image. To display it with **data-unit axes** — the same
time/amplitude axes `plt.plot` produces — use {py:func}`~wave_measure.plot_dpo`,
which draws the render onto a matplotlib `Axes` and returns it:

```python
import matplotlib.pyplot as plt

ax = wm.plot_dpo(wave, cmap="inferno")   # x = time, y = amplitude
plt.show()
```

## Examples

These reproduce the demonstrations from Lithcore's
[Python multi-core / GPU digital phosphor][blog] article, whose CUDA
implementation this renderer is based on.

### `wm.render` vs `plt.plot`

The same 2-million-sample AM signal, drawn both ways with **identical axes**.
`plt.plot` overdraws into a solid smear that hides all internal structure;
`wm.render` grades every pixel by trace density, so the modulation envelope and
its bright nodes emerge.

```python
import numpy as np
import matplotlib.pyplot as plt
import wave_measure as wm

fs, n = 1e6, 2_000_000
t = np.arange(n) / fs
y = np.sin(2 * np.pi * 1000 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
wave = wm.from_array(y.astype(np.float32), sample_rate=fs)

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
left.plot(t, y, lw=0.4)
left.set(title="plt.plot()", xlabel="Time (s)", ylabel="Amplitude")

wm.plot_dpo(wave, ax=right, cmap="inferno")
right.set_title("wm.render()")
right.set_xlim(left.get_xlim())        # match plt.plot's axes exactly
right.set_ylim(left.get_ylim())
```

```{image} ../_static/dpo_vs_plot.png
:alt: plt.plot and wm.render of an AM signal, side by side with identical axes
:width: 100%
```

### 2-D random walk

Both axes are Brownian walks. Passing `x=` renders in **X-Y mode** — the y
waveform is plotted against a second channel (here another walk) instead of
against time. The phosphor accumulation reveals the fractal structure, glowing
where the walk revisits a region.

```python
rng = np.random.default_rng(7)
x = np.cumsum(rng.standard_normal(8_000_000)).astype(np.float32)
y = np.cumsum(rng.standard_normal(8_000_000)).astype(np.float32)

walk = wm.from_array(y)
ax = wm.plot_dpo(walk, x=x, width=680, height=680, cmap="magma", ylabel="Y")
```

```{image} ../_static/dpo_random_walk.png
:alt: Digital-phosphor render of a 2-D random walk with X-Y axes
:width: 70%
:align: center
```

[blog]: https://www.lithcore.net/2025/02/python-multi-core-gpu-digital-phosphor.html

## Options

- `x` — a second channel (`Waveform` or array) for X-Y mode; defaults to time.
- `width`, `height` — output image size in pixels.
- `cmap` — a matplotlib colormap (name or object). Omit it to get the raw
  intensity image and colorize yourself with `plt.imshow(img, cmap=...)`; pass
  it to bake an RGBA image.
- `scale` — intensity compression: `"sqrt"` (default), `"log"`, or `"linear"`.
- `x_range`, `y_range` — data bounds to map; default to the full x/y span.
- `backend` — `"cuda"`, `"cpu"`, or auto from the detected accelerator. An
  unusable CUDA request falls back to CPU with a warning.
- `workers` — CPU worker count (defaults to all cores).
- `path` — if given, also save the image.

The return value is oriented for a plain `plt.imshow(img)` (amplitude increases
upward): an `(height, width)` float array in `[0, 1]` without `cmap`, or an
`(height, width, 4)` `uint8` RGBA image with it.

## Data-unit axes

{py:func}`~wave_measure.plot_dpo` (above) is the easy way to get labeled axes.
Under the hood it uses {py:func}`~wave_measure.dpo_histogram`, which returns the
raw counts and extent if you want full control:

```python
hist, extent = wm.dpo_histogram(wave, width=1000, height=500)
plt.imshow(hist ** 0.5, origin="lower", aspect="auto", extent=extent)
```
