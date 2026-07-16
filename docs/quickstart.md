# Quick start

## From arrays

Build a waveform directly from NumPy arrays:

```python
import numpy as np
import wave_measure as wm

t = np.linspace(0, 1e-3, 1000)
v = np.sin(2 * np.pi * 5000 * t)
wave = wm.Waveform(samples=v, time=t)          # or wm.from_array(v, sample_rate=1e6)

print(wm.vpp(wave))        # peak-to-peak
print(wm.frequency(wave))  # fundamental frequency
print(wm.vrms(wave))       # RMS amplitude
```

Any of the {doc}`measurements <api/measurements>` (`vpp`, `vrms`, `frequency`,
`rise_time`, …) work on an in-memory waveform.

## From a file

Most benchtop scopes export either CSV or a flat binary buffer:

```python
wave = wm.read_csv("capture.csv")                              # CSV export
wave = wm.read_raw("capture.bin", dtype="int16", sample_rate=1e6)  # flat binary
```

`read_raw` memory-maps the file and returns a lazy, streaming
{py:class}`~wave_measure.Waveform` — see {doc}`guide/streaming`.

## Building a pipeline

Operators are grouped into categories and chain fluently, producing a new lazy
waveform. Reductions (`histogram`, `min`, `top`, …) end the chain with a result:

```python
processed = wave.amplitude.abs().filter.lowpass(50_000).math.diff()
levels = wave.amplitude.levels()          # bottom/top logic levels
top = wave.amplitude.top()
```

## Rendering

```python
import matplotlib.pyplot as plt

img = wm.render(wave, width=1000, height=500)  # digital-phosphor RGBA image
plt.imshow(img)
plt.imsave("scope.png", img)
```

See {doc}`guide/rendering`.
