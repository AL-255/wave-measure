"""Accelerator detection and Numba-backed dispatch.

Waveform captures can be very large, so wave-measure leans on `Numba
<https://numba.pydata.org/>`_ to JIT-compile its hot loops. At import time the
package probes the host for the best available accelerator:

* **NVIDIA GPU** — used through ``numba.cuda`` when a CUDA driver and device
  are present.
* **AMD GPU** — detected and reported, but *not* used: mainline Numba removed
  its ROCm/HSA backend in 0.54, so there is no supported path to it here. We
  fall back to the CPU and say so.
* **CPU with SIMD** — the default. Numba's LLVM backend auto-vectorizes the
  JIT-compiled kernels using whatever SIMD ISA the host advertises (AVX-512,
  AVX2/FMA, SSE, NEON, ...), which we detect and report.

The detected :class:`AcceleratorInfo` is exposed as ``wave_measure.accelerator``.
Detection can be skipped by setting ``WAVE_MEASURE_SKIP_DETECT=1`` and the
chosen backend forced with ``WAVE_MEASURE_BACKEND=cpu|cuda``.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger("wave_measure")

__all__ = [
    "AcceleratorInfo",
    "detect_accelerator",
    "get_accelerator",
    "njit",
]

# SIMD instruction sets we care to report, most to least capable.
_SIMD_FEATURES_X86 = (
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx2",
    "fma",
    "avx",
    "sse4.2",
    "sse4.1",
    "ssse3",
    "sse3",
    "sse2",
    "sse",
)
_SIMD_FEATURES_ARM = ("sve", "asimd", "neon")


@dataclass(frozen=True)
class AcceleratorInfo:
    """The accelerator wave-measure selected, plus what else was detected."""

    backend: str  # "cuda" or "cpu" — what Numba will actually target
    hardware: str  # human-readable description of the selected device
    cuda_available: bool = False  # an NVIDIA device Numba can compile for
    nvidia_gpu_detected: bool = False  # an NVIDIA device was enumerated
    amd_gpu_detected: bool = False
    simd_features: Tuple[str, ...] = ()
    cpu_name: str = ""
    numba_version: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_gpu(self) -> bool:
        """True when the selected backend runs on a GPU."""
        return self.backend == "cuda"

    def summary(self) -> str:
        """A one-line human-readable summary of the detected accelerator."""
        parts = [f"backend={self.backend}", self.hardware]
        if self.simd_features:
            parts.append("SIMD=" + ",".join(self.simd_features))
        parts.append(
            f"numba={self.numba_version}" if self.numba_version else "numba=unavailable"
        )
        return "wave-measure accelerator → " + "; ".join(parts)

    def __str__(self) -> str:
        return self.summary()


# -- individual probes ------------------------------------------------------


def _detect_numba_version() -> Optional[str]:
    try:
        import numba

        return numba.__version__
    except Exception as exc:  # pragma: no cover - numba is a hard dependency
        logger.debug("Numba unavailable: %s", exc)
        return None


def _decode(name) -> Optional[str]:
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return name


def _short(msg: str, limit: int = 220) -> str:
    """Condense a multi-line exception/error log to its most telling line."""
    lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
    text = lines[-1] if lines else msg
    return text[:limit]


def _cuda_functional_probe() -> Tuple[bool, Optional[str]]:
    """Compile and launch a trivial kernel to confirm CUDA actually works.

    ``numba.cuda.is_available()`` only checks that a device and the CUDA
    libraries are present; it does not catch toolchain mismatches (e.g. a
    ``ptxas`` too old for the PTX version Numba emits). Running one real kernel
    does, so ``backend == "cuda"`` means kernels genuinely run.
    """
    try:
        import warnings

        import numpy as np
        from numba import cuda

        @cuda.jit
        def _probe(arr):  # pragma: no cover - runs on device
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] += 1.0

        a = np.zeros(32, dtype=np.float64)
        with warnings.catch_warnings():
            # A 1-block probe is intentionally tiny; ignore occupancy warnings.
            warnings.simplefilter("ignore")
            d = cuda.to_device(a)
            _probe[1, 32](d)
            cuda.synchronize()
            result = float(d.copy_to_host()[0])
        if result == 1.0:
            return True, None
        return False, "a test kernel produced an unexpected result"
    except Exception as exc:
        return False, _short(str(exc))


def _detect_cuda() -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """Probe for an NVIDIA/CUDA device.

    Returns ``(usable, present, device_name, reason)`` where ``usable`` means
    Numba can actually compile and run kernels on it, while ``present`` means a
    device was merely enumerated. They differ when the driver can see a GPU but
    the toolkit is missing or its ``ptxas``/NVVM is incompatible with Numba.
    """
    usable = present = False
    name = reason = None
    try:
        from numba import cuda

        # Enumeration uses the driver API only — works without the toolkit.
        try:
            devices = list(cuda.gpus)
            if devices:
                present = True
                name = _decode(devices[0].name)
        except Exception as exc:
            reason = _short(str(exc))  # e.g. no NVIDIA driver / libcuda

        if cuda.is_available():
            probe_disabled = os.environ.get(
                "WAVE_MEASURE_NO_CUDA_PROBE", ""
            ).strip().lower() in ("1", "true", "yes")
            if name is None:
                name = _decode(cuda.get_current_device().name)
            if probe_disabled:
                usable = True
            else:
                ok, probe_reason = _cuda_functional_probe()
                if ok:
                    usable = True
                else:
                    reason = (
                        "device reported available but a test kernel failed to "
                        f"run: {probe_reason}"
                    )
        elif present and reason is None:
            reason = (
                "device found but the CUDA toolkit/NVVM could not be loaded to "
                "compile kernels (install a matching CUDA toolkit to enable it)"
            )
    except Exception as exc:
        logger.debug("CUDA detection failed: %s", exc)
    return usable, present, name, reason


def _detect_amd_gpu() -> bool:
    """Best-effort, side-effect-free check for an AMD ROCm-capable GPU."""
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return True
    # The AMDKFD kernel driver exposes these when an AMD GPU/APU is present.
    return any(os.path.exists(p) for p in ("/dev/kfd", "/sys/class/kfd"))


def _detect_cpu() -> Tuple[str, Tuple[str, ...]]:
    """Return ``(cpu_name, enabled_simd_features)`` for the host CPU."""
    try:
        from llvmlite import binding as llvm

        cpu_name = llvm.get_host_cpu_name()
        feature_map = llvm.get_host_cpu_features()  # name -> bool
        enabled = {name for name, on in feature_map.items() if on}
        features = tuple(
            f for f in (_SIMD_FEATURES_X86 + _SIMD_FEATURES_ARM) if f in enabled
        )
        return cpu_name, features
    except Exception as exc:
        logger.debug("llvmlite CPU probe failed (%s); falling back to /proc", exc)
        return "", _detect_cpu_features_fallback()


def _detect_cpu_features_fallback() -> Tuple[str, ...]:
    """Parse ``/proc/cpuinfo`` flags when llvmlite is unavailable (Linux only)."""
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                low = line.lower()
                if low.startswith("flags") or low.startswith("features"):
                    present = set(low.split(":", 1)[1].split())
                    out = []
                    for f in _SIMD_FEATURES_X86 + _SIMD_FEATURES_ARM:
                        # /proc spells e.g. "sse4.1" as "sse4_1".
                        if f in present or f.replace(".", "_") in present:
                            out.append(f)
                    return tuple(out)
    except OSError:
        pass
    return ()


# -- top-level detection ----------------------------------------------------


def detect_accelerator() -> AcceleratorInfo:
    """Probe the host and return a fresh :class:`AcceleratorInfo` (uncached)."""
    numba_version = _detect_numba_version()
    notes: list[str] = []
    if numba_version is None:
        notes.append("Numba is not importable; kernels run as plain Python/NumPy.")

    cuda_usable, cuda_present, cuda_name, cuda_reason = _detect_cuda()
    amd_detected = _detect_amd_gpu()
    cpu_name, simd = _detect_cpu()

    override = os.environ.get("WAVE_MEASURE_BACKEND", "").strip().lower()
    if override in ("cpu", "cuda"):
        backend = override
        if override == "cuda" and not cuda_usable:
            notes.append(
                "WAVE_MEASURE_BACKEND=cuda forced, but no usable CUDA device was found."
            )
    elif cuda_usable:
        backend = "cuda"
    else:
        backend = "cpu"

    if backend == "cuda" and cuda_name:
        hardware = f"NVIDIA GPU ({cuda_name})"
    else:
        hardware = f"CPU ({cpu_name})" if cpu_name else "CPU"

    # An NVIDIA GPU is present but we couldn't select it — explain why.
    if cuda_present and backend != "cuda":
        detail = f": {cuda_reason}" if cuda_reason else ""
        nm = f" ({cuda_name})" if cuda_name else ""
        notes.append(f"NVIDIA GPU{nm} detected but unusable{detail}.")

    if amd_detected and backend != "cuda":
        notes.append(
            "AMD GPU detected but unused: mainline Numba dropped its ROCm/HSA "
            "backend in 0.54, so there is no supported GPU path for it."
        )
    elif amd_detected and backend == "cuda":
        notes.append("AMD GPU also detected; NVIDIA/CUDA was preferred.")

    info = AcceleratorInfo(
        backend=backend,
        hardware=hardware,
        cuda_available=cuda_usable,
        nvidia_gpu_detected=cuda_present,
        amd_gpu_detected=amd_detected,
        simd_features=simd,
        cpu_name=cpu_name,
        numba_version=numba_version,
        notes=tuple(notes),
    )
    return info


_CACHED: Optional[AcceleratorInfo] = None


def get_accelerator(*, refresh: bool = False) -> AcceleratorInfo:
    """Return the cached accelerator info, detecting it on first use."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = detect_accelerator()
    return _CACHED


# -- Numba dispatch helper --------------------------------------------------


def _make_njit():
    """Return Numba's ``njit`` if available, else a transparent no-op decorator.

    This lets kernels be written once and still run (as plain Python) in the
    unlikely event Numba cannot be imported.
    """
    try:
        from numba import njit as _njit

        return _njit
    except Exception:  # pragma: no cover - numba is a hard dependency

        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def wrap(func):
                return func

            return wrap

        return njit


njit = _make_njit()
