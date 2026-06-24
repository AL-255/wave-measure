"""Tests for the digital-phosphor renderer."""

import numpy as np
import pytest

import wave_measure as wm


def test_horizontal_line_rasterizes_one_row():
    # A single flat segment should fill exactly one histogram row, edge to edge.
    wave = wm.from_array([0.5, 0.5], sample_rate=1.0)  # time = [0, 1]
    hist, extent = wm.dpo_histogram(
        wave, width=100, height=50, x_range=(0, 1), y_range=(0, 1), workers=1
    )
    assert hist.shape == (50, 100)
    row = int(0.5 * 50)
    assert np.count_nonzero(hist[row]) == 100  # whole row touched
    assert hist.sum() == 100  # nothing drawn elsewhere
    assert extent == (0, 1, 0, 1)


@pytest.fixture
def signal():
    fs = 1e6
    n = 50_000
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    y = np.sin(2 * np.pi * 2000 * t) + 0.1 * rng.standard_normal(n)
    return wm.from_array(y, sample_rate=fs)


def test_streaming_block_size_is_invariant(signal):
    # Accumulating across blocks (connecting boundaries) must match one big pass.
    whole, _ = wm.dpo_histogram(signal, width=256, height=128, block=10**9, workers=1)
    chunked, _ = wm.dpo_histogram(signal, width=256, height=128, block=512, workers=1)
    np.testing.assert_array_equal(whole, chunked)


def test_worker_count_is_invariant(signal):
    one, _ = wm.dpo_histogram(signal, width=256, height=128, workers=1)
    many, _ = wm.dpo_histogram(signal, width=256, height=128, workers=4)
    np.testing.assert_array_equal(one, many)


def test_histogram_is_connected_trace(signal):
    # Every column in the swept range should be hit (the trace is continuous).
    hist, _ = wm.dpo_histogram(signal, width=300, height=200, workers=1)
    columns_hit = np.count_nonzero(hist.sum(axis=0))
    assert columns_hit == 300
    # Counts total at least one pixel per segment.
    assert hist.sum() >= len(signal) - 1


def test_render_returns_rgba_image(signal):
    img = wm.render(signal, width=320, height=240)
    assert img.shape == (240, 320, 4)
    assert img.dtype == np.uint8


def test_render_scales_and_cmaps(signal):
    # Different intensity scales should generally produce different images.
    a = wm.render(signal, width=128, height=96, scale="sqrt", cmap="viridis")
    b = wm.render(signal, width=128, height=96, scale="log", cmap="viridis")
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_render_saves_to_path(signal, tmp_path):
    out = tmp_path / "dpo.png"
    img = wm.render(signal, width=160, height=120, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert img.shape == (120, 160, 4)


def test_render_on_streamed_waveform(signal):
    # A lazy pipeline renders the same way (it's just another Waveform).
    img = wm.render(signal.amplitude.abs(), width=100, height=80)
    assert img.shape == (80, 100, 4)


def test_cuda_backend_falls_back_when_unavailable(signal):
    if wm.accelerator.cuda_available:
        pytest.skip("CUDA is usable here; fallback path not exercised")
    with pytest.warns(RuntimeWarning):
        hist, _ = wm.dpo_histogram(signal, width=64, height=64, backend="cuda")
    assert hist.sum() > 0


def test_too_short_raises():
    with pytest.raises(ValueError):
        wm.dpo_histogram(wm.from_array([1.0], sample_rate=1.0))
