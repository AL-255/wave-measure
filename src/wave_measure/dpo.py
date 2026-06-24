"""Digital-phosphor (DPO) rendering of waveforms.

Rather than plotting points, the DPO renderer rasterizes the *line segments*
between consecutive samples into a 2-D accumulation histogram (Bresenham's line
algorithm), so each pixel counts how many traces pass through it. Displayed with
a compressive intensity map, dense regions glow like an analog phosphor screen.
This is the only way to faithfully draw captures with 10^8-10^9 samples, where
``plt.plot`` both aliases and crawls.

The algorithm follows the CUDA implementation documented at
https://gist.github.com/AL-255/abb3193b3697bb1a618d725ddabfd3d6 and
https://www.lithcore.net/2025/02/python-multi-core-gpu-digital-phosphor.html ,
extended here in two ways:

* a **CPU** rasterizer (the gist is CUDA-only), multi-core via Numba ``prange``
  with per-worker buffers, and
* **streaming**: the whole waveform is consumed block by block and accumulated
  into one fixed-size histogram, connecting the last sample of each block to the
  first of the next. Memory is bounded by the histogram, not the capture size.

``render()`` returns an RGBA image (a NumPy ``uint8`` array) ready for
``plt.imshow(img)`` or ``plt.imsave(path, img)``. Select a sub-range to draw by
slicing the waveform yourself first, e.g. ``wm.render(wave.get_from_to(a, b))``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .accelerator import get_accelerator, njit

try:  # numba is a core dependency, but stay graceful if its parallel API moves
    from numba import get_num_threads, prange
except Exception:  # pragma: no cover
    prange = range

    def get_num_threads() -> int:
        return 1


__all__ = ["render", "dpo_histogram"]


# -- CPU rasterizers (Numba) ------------------------------------------------


@njit(cache=True)
def _rasterize_cpu_single(x, y, hist, Nx, Ny, x_min, x_max, y_min, y_max):
    """Rasterize all segments of (x, y) into ``hist`` on a single thread."""
    n = x.shape[0]
    scale_x = Nx / (x_max - x_min)
    scale_y = Ny / (y_max - y_min)
    for i in range(n - 1):
        ix0 = int(math.floor((x[i] - x_min) * scale_x))
        iy0 = int(math.floor((y[i] - y_min) * scale_y))
        ix1 = int(math.floor((x[i + 1] - x_min) * scale_x))
        iy1 = int(math.floor((y[i + 1] - y_min) * scale_y))
        if ix0 < 0:
            ix0 = 0
        elif ix0 >= Nx:
            ix0 = Nx - 1
        if iy0 < 0:
            iy0 = 0
        elif iy0 >= Ny:
            iy0 = Ny - 1
        if ix1 < 0:
            ix1 = 0
        elif ix1 >= Nx:
            ix1 = Nx - 1
        if iy1 < 0:
            iy1 = 0
        elif iy1 >= Ny:
            iy1 = Ny - 1

        dx = ix1 - ix0
        if dx < 0:
            dx = -dx
        sx = 1 if ix0 < ix1 else -1
        dy = iy1 - iy0
        if dy < 0:
            dy = -dy
        dy = -dy
        sy = 1 if iy0 < iy1 else -1
        err = dx + dy
        xc = ix0
        yc = iy0
        while True:
            hist[yc, xc] += 1
            if xc == ix1 and yc == iy1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                xc += sx
            if e2 <= dx:
                err += dx
                yc += sy


@njit(parallel=True, cache=True)
def _rasterize_cpu_parallel(x, y, buffers, Nx, Ny, x_min, x_max, y_min, y_max):
    """Multi-core rasterize: each worker draws a slice of segments into its own
    buffer (``buffers[p]``), avoiding write races. Caller sums the buffers."""
    P = buffers.shape[0]
    nseg = x.shape[0] - 1
    scale_x = Nx / (x_max - x_min)
    scale_y = Ny / (y_max - y_min)
    for p in prange(P):
        seg_start = p * nseg // P
        seg_end = (p + 1) * nseg // P
        for i in range(seg_start, seg_end):
            ix0 = int(math.floor((x[i] - x_min) * scale_x))
            iy0 = int(math.floor((y[i] - y_min) * scale_y))
            ix1 = int(math.floor((x[i + 1] - x_min) * scale_x))
            iy1 = int(math.floor((y[i + 1] - y_min) * scale_y))
            if ix0 < 0:
                ix0 = 0
            elif ix0 >= Nx:
                ix0 = Nx - 1
            if iy0 < 0:
                iy0 = 0
            elif iy0 >= Ny:
                iy0 = Ny - 1
            if ix1 < 0:
                ix1 = 0
            elif ix1 >= Nx:
                ix1 = Nx - 1
            if iy1 < 0:
                iy1 = 0
            elif iy1 >= Ny:
                iy1 = Ny - 1

            dx = ix1 - ix0
            if dx < 0:
                dx = -dx
            sx = 1 if ix0 < ix1 else -1
            dy = iy1 - iy0
            if dy < 0:
                dy = -dy
            dy = -dy
            sy = 1 if iy0 < iy1 else -1
            err = dx + dy
            xc = ix0
            yc = iy0
            while True:
                buffers[p, yc, xc] += 1
                if xc == ix1 and yc == iy1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    xc += sx
                if e2 <= dx:
                    err += dx
                    yc += sy


# -- CUDA rasterizer (Numba, faithful to the gist) --------------------------

_CUDA_KERNEL = None


def _get_cuda_kernel():
    """Build (and cache) the CUDA kernel; mirrors the documented gist."""
    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        from numba import cuda

        @cuda.jit(device=True)
        def _bresenham(hist, x0, y0, x1, y1):  # pragma: no cover - device code
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            xc, yc = x0, y0
            while True:
                cuda.atomic.add(hist, (yc, xc), 1)
                if xc == x1 and yc == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    xc += sx
                if e2 <= dx:
                    err += dx
                    yc += sy

        @cuda.jit
        def _kernel(x, y, hist, Nx, Ny, x_min, x_max, y_min, y_max):  # pragma: no cover
            i = cuda.grid(1)
            if i >= x.shape[0] - 1:
                return
            scale_x = Nx / (x_max - x_min)
            scale_y = Ny / (y_max - y_min)
            x0 = (x[i] - x_min) * scale_x
            y0 = (y[i] - y_min) * scale_y
            x1 = (x[i + 1] - x_min) * scale_x
            y1 = (y[i + 1] - y_min) * scale_y
            ix0 = int(math.floor(x0))
            iy0 = int(math.floor(y0))
            ix1 = int(math.floor(x1))
            iy1 = int(math.floor(y1))
            if ix0 < 0:
                ix0 = 0
            elif ix0 >= Nx:
                ix0 = Nx - 1
            if iy0 < 0:
                iy0 = 0
            elif iy0 >= Ny:
                iy0 = Ny - 1
            if ix1 < 0:
                ix1 = 0
            elif ix1 >= Nx:
                ix1 = Nx - 1
            if iy1 < 0:
                iy1 = 0
            elif iy1 >= Ny:
                iy1 = Ny - 1
            _bresenham(hist, ix0, iy0, ix1, iy1)

        _CUDA_KERNEL = _kernel
    return _CUDA_KERNEL


# -- streaming orchestration ------------------------------------------------


def _segment_arrays(chunk, prev):
    """Block coords (time, samples), prepended with the previous block's last
    point so the connecting segment is drawn exactly once."""
    cx = np.ascontiguousarray(chunk.time, dtype=np.float64)
    cy = np.ascontiguousarray(chunk.samples, dtype=np.float64)
    if prev is not None:
        cx = np.concatenate((prev[0:1], cx))
        cy = np.concatenate((prev[1:2], cy))
    return cx, cy


def _render_cpu(waveform, Nx, Ny, ranges, block, workers):
    hist = np.zeros((Ny, Nx), dtype=np.uint32)
    x_min, x_max, y_min, y_max = ranges
    if workers is None:
        workers = max(1, int(get_num_threads()))
    buffers = np.zeros((workers, Ny, Nx), dtype=np.uint32) if workers > 1 else None

    prev = None
    for chunk in waveform.blocks(block):
        cx, cy = _segment_arrays(chunk, prev)
        if cx.shape[0] >= 2:
            if workers > 1:
                buffers[:] = 0
                _rasterize_cpu_parallel(cx, cy, buffers, Nx, Ny, x_min, x_max, y_min, y_max)
                hist += buffers.sum(axis=0, dtype=np.uint64).astype(np.uint32)
            else:
                _rasterize_cpu_single(cx, cy, hist, Nx, Ny, x_min, x_max, y_min, y_max)
        prev = np.array([chunk.time[-1], chunk.samples[-1]], dtype=np.float64)
    return hist


def _render_cuda(waveform, Nx, Ny, ranges, block, threads_per_block=256):
    from numba import cuda

    kernel = _get_cuda_kernel()
    d_hist = cuda.to_device(np.zeros((Ny, Nx), dtype=np.uint32))
    x_min, x_max, y_min, y_max = ranges

    prev = None
    for chunk in waveform.blocks(block):
        cx, cy = _segment_arrays(chunk, prev)
        if cx.shape[0] >= 2:
            d_x = cuda.to_device(cx)
            d_y = cuda.to_device(cy)
            nblocks = (cx.shape[0] + threads_per_block - 1) // threads_per_block
            kernel[nblocks, threads_per_block](
                d_x, d_y, d_hist, Nx, Ny, x_min, x_max, y_min, y_max
            )
        prev = np.array([chunk.time[-1], chunk.samples[-1]], dtype=np.float64)
    cuda.synchronize()
    return d_hist.copy_to_host()


def _resolve_backend(backend: Optional[str]) -> str:
    if backend not in (None, "cpu", "cuda", "auto"):
        raise ValueError("backend must be 'cpu', 'cuda', 'auto', or None")
    if backend in ("cpu", "cuda"):
        if backend == "cuda" and not get_accelerator().cuda_available:
            import warnings

            warnings.warn(
                "backend='cuda' requested but no usable CUDA device; using CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "cpu"
        return backend
    return "cuda" if get_accelerator().cuda_available else "cpu"


def _resolve_ranges(waveform, x_range, y_range, block):
    n = len(waveform)
    if x_range is None:
        x_min = float(waveform.source.time(0, 1)[0])
        x_max = float(waveform.source.time(n - 1, n)[0])
    else:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    if y_range is None:
        stats = waveform.amplitude.stats(block=block)
        y_min, y_max = float(stats.min), float(stats.max)
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])

    if not x_max > x_min:
        x_min, x_max = x_min - 0.5, x_min + 0.5
    if not y_max > y_min:  # flat / DC signal: give it room in the middle
        y_min, y_max = y_min - 0.5, y_max + 0.5
    return x_min, x_max, y_min, y_max


def dpo_histogram(
    waveform,
    *,
    width: int = 1200,
    height: int = 800,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    backend: Optional[str] = None,
    block: int = 1 << 20,
    workers: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Rasterize the whole waveform into a DPO accumulation histogram.

    Returns ``(hist, extent)`` where ``hist`` is a ``(height, width)`` ``uint32``
    array (row 0 = ``y_min``, the bottom) and ``extent`` is
    ``(x_min, x_max, y_min, y_max)`` for ``imshow(..., origin="lower", extent=...)``.
    """
    if len(waveform) < 2:
        raise ValueError("need at least two samples to render line segments")
    Nx, Ny = int(width), int(height)
    ranges = _resolve_ranges(waveform, x_range, y_range, block)
    backend = _resolve_backend(backend)
    if backend == "cuda":
        hist = _render_cuda(waveform, Nx, Ny, ranges, block)
    else:
        hist = _render_cpu(waveform, Nx, Ny, ranges, block, workers)
    return hist, ranges


def _intensity(hist: np.ndarray, scale: str) -> np.ndarray:
    h = hist.astype(np.float64)
    if scale == "sqrt":
        h = np.sqrt(h)
    elif scale == "log":
        h = np.log1p(h)
    elif scale in ("linear", "none", None):
        pass
    else:
        raise ValueError("scale must be 'sqrt', 'log', or 'linear'")
    peak = h.max()
    if peak > 0:
        h /= peak
    return h


def render(
    waveform,
    *,
    width: int = 1200,
    height: int = 800,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    cmap: str = "inferno",
    scale: str = "sqrt",
    backend: Optional[str] = None,
    block: int = 1 << 20,
    workers: Optional[int] = None,
    path: Optional[str] = None,
) -> np.ndarray:
    """Render a waveform as a digital-phosphor image.

    The entire ``waveform`` is drawn; to render a sub-range, slice it first
    (``wm.render(wave.get_from_to(a, b))``).

    Parameters
    ----------
    width, height:
        Output image size in pixels.
    x_range, y_range:
        Data bounds to map onto the image. Default to the full time span and
        (streamed) amplitude min/max. Passing ``y_range`` avoids the extra
        min/max pass.
    cmap:
        Any matplotlib colormap name (default ``"inferno"`` for a phosphor look).
    scale:
        Intensity compression: ``"sqrt"`` (default), ``"log"``, or ``"linear"``.
    backend:
        ``"cuda"``, ``"cpu"``, or ``None``/``"auto"`` to pick from the detected
        accelerator.
    block, workers:
        Streaming block size and CPU worker count (defaults to all cores).
    path:
        If given, also save the image there via ``plt.imsave``.

    Returns
    -------
    numpy.ndarray
        An ``(height, width, 4)`` ``uint8`` RGBA image, oriented for a plain
        ``plt.imshow(img)`` (amplitude increases upward). Save it with
        ``plt.imsave(path, img)`` or pass ``path=`` above.
    """
    import matplotlib

    try:
        cmap_obj = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm

        cmap_obj = cm.get_cmap(cmap)

    hist, _ = dpo_histogram(
        waveform,
        width=width,
        height=height,
        x_range=x_range,
        y_range=y_range,
        backend=backend,
        block=block,
        workers=workers,
    )
    intensity = _intensity(hist, scale)
    # Flip so row 0 is the top (max amplitude) for a default imshow / imsave.
    rgba = (cmap_obj(np.flipud(intensity)) * 255).astype(np.uint8)

    if path is not None:
        import matplotlib.pyplot as plt

        plt.imsave(path, rgba)
    return rgba
