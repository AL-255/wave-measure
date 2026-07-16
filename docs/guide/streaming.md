# Streaming & operators

A {py:class}`~wave_measure.Waveform` is a **lazy node**, not a buffer of
samples: it is a {py:class}`~wave_measure.Source` (a file or an array) plus a
chain of length-preserving operators. Building a chain reads nothing; data is
pulled only for the range you ask for, so captures far larger than memory are
handled the same as small ones.

## Pulling data

```python
wave = wm.read_raw("capture.bin", dtype="int16", sample_rate=1e6, gain=1e-3)
processed = wave.amplitude.abs().filter.lowpass(50_000).math.diff()

chunk = processed.get_from_to(10_000, 20_000)   # random access -> in-memory Waveform
block = processed.get_next(4096)                # sequential cursor (None at the end)
for blk in processed.blocks(1 << 20):           # bounded-memory iteration
    ...

processed.to_file("out.f32", dtype="float32")   # stream the result back to disk
```

The chunk returned by {py:meth}`~wave_measure.Waveform.get_from_to` and
{py:meth}`~wave_measure.Waveform.get_next` is itself an in-memory `Waveform`, so
every measurement and plot works on it directly.

## Correct random access

Each operator declares a **margin** of context it needs. To compute
`get_from_to(start, stop)`, the engine fetches
`[start − left_margin, stop + right_margin]` from upstream, applies the
operator, and trims the margins. This is what makes random access correct even
when a slice starts in the middle of a filter's response.

`fir`, `diff`, and `median` have exact finite margins; the IIR filters
(`lowpass`, `highpass`, `iir`) reconverge within a finite warm-up margin, so
results are correct to a small, documented tolerance.

## Operator categories

Operators are organized into domains reached as chainable accessors on the
waveform. Length-preserving methods return a new lazy `Waveform`; terminal
reductions return a result.

| Category | Length-preserving | Terminal reductions |
|---|---|---|
| {py:class}`~wave_measure.categories.FilterCategory` (`wave.filter`) | `fir`, `iir`, `lowpass`, `highpass`, `bandpass`, `moving_average`, `median` | — |
| {py:class}`~wave_measure.categories.AmplitudeCategory` (`wave.amplitude`) | `abs`, `clip`, `gain`, `offset` | `histogram`, `min`, `max`, `mean`, `rms`, `peak_to_peak`, `stats`, `peaks`, `top`, `bottom`, `levels` |
| {py:class}`~wave_measure.categories.MathCategory` (`wave.math`) | `diff`, `square`, `sqrt`, `log` | — |

```python
spikes = wave.amplitude.abs().filter.median(7).math.diff()

h      = wave.amplitude.histogram(bins=256)          # Histogram
stats  = wave.amplitude.stats()                      # min/max/mean/rms/p2p
top    = wave.amplitude.top()                        # logic-level top (2-Gaussian fit)
```

The same categories exist at module level as a catalog for building standalone,
reusable operators (`wm.filter.fir(taps)`) and functional reductions
(`wm.amplitude.mean(wave)`).

## Logic levels

`top` and `bottom` estimate a signal's high and low dwell levels by fitting two
Gaussians to the amplitude histogram (via EM) and taking the two means. See
{py:class}`~wave_measure.Levels`.

## Custom formats

To read a proprietary binary format, subclass {py:class}`~wave_measure.Source`
(implement `__len__` and `read(start, stop)`) and pass it as
`Waveform(source=...)`.
