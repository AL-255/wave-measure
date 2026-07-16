# Acceleration

Waveform captures can be huge, so wave-measure JIT-compiles its hot loops with
[Numba](https://numba.pydata.org/) and probes the host for the best available
accelerator at import time. The result is exposed as
{py:data}`wave_measure.accelerator`, an {py:class}`~wave_measure.AcceleratorInfo`.

```python
import wave_measure as wm

print(wm.accelerator)
# wave-measure accelerator -> backend=cpu; CPU (znver5); SIMD=avx512f,avx2,...; numba=0.65.1
print(wm.accelerator.backend)          # "cuda" or "cpu"
print(wm.accelerator.simd_features)    # detected SIMD ISAs
```

Backend selection:

- **NVIDIA** GPUs are used through `numba.cuda` when a driver and device exist —
  and only when a test kernel actually compiles and runs, so `backend == "cuda"`
  is trustworthy (a present-but-unusable GPU is reported in `notes` and the CPU
  is used instead).
- **AMD** GPUs are detected and reported but not used for compute (mainline Numba
  dropped its ROCm/HSA backend in 0.54).
- **CPU** is the default; Numba's LLVM backend auto-vectorizes kernels with the
  host's SIMD ISA (AVX-512, AVX2/FMA, SSE, NEON, …).

Environment overrides:

- `WAVE_MEASURE_SKIP_DETECT=1` — skip detection at import (`wm.accelerator` is
  then `None`; call {py:func}`~wave_measure.get_accelerator` on demand).
- `WAVE_MEASURE_BACKEND=cpu|cuda` — force the selected backend.
- `WAVE_MEASURE_NO_CUDA_PROBE=1` — trust `is_available()` and skip the functional
  CUDA kernel probe (faster import, less safe).
