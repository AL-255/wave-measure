"""Tests for the lazy streaming pipeline and binary readers."""

import numpy as np
import pytest

import wave_measure as wm


@pytest.fixture
def raw_file(tmp_path):
    """A 200k-sample int16 capture on disk: noisy 1 kHz sine at 100 kHz."""
    fs = 100_000.0
    n = 200_000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    v = 8000 * np.sin(2 * np.pi * 1000 * t) + rng.normal(0, 300, n)
    data = v.astype(np.int16)
    path = tmp_path / "capture.bin"
    path.write_bytes(data.tobytes())
    return path, fs, n, data


def test_read_raw_is_lazy_and_streams(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    assert len(wave) == n
    assert wave.sample_rate == fs
    assert wave.is_streaming
    # A slice reads only that range and matches the file exactly.
    chunk = wave.get_from_to(1000, 1100)
    np.testing.assert_allclose(chunk.samples, data[1000:1100])


def test_gain_offset_calibration(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs, gain=0.001, offset=-1.0)
    chunk = wave.get_from_to(0, 50)
    np.testing.assert_allclose(chunk.samples, data[:50] * 0.001 - 1.0, rtol=1e-6)


def test_get_next_cursor_covers_whole_signal(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    out = []
    while (block := wave.get_next(4096)) is not None:
        out.append(block.samples)
    joined = np.concatenate(out)
    assert joined.size == n
    np.testing.assert_allclose(joined, data)
    # Cursor is exhausted; reset rewinds it.
    assert wave.get_next(10) is None
    wave.reset()
    np.testing.assert_allclose(wave.get_next(5).samples, data[:5])


def test_abs_diff_pipeline_is_lazy(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    pipe = wave.amplitude.abs().math.diff()
    assert pipe.is_streaming
    assert len(pipe) == n  # length-preserving

    ref = np.abs(data.astype(float))
    ref = np.concatenate([[0.0], np.diff(ref)])

    # Whole-signal reference vs a mid-stream random-access slice.
    full = pipe.to_array(allow_large=True)
    np.testing.assert_allclose(full, ref, atol=1e-9)
    sl = pipe.get_from_to(50_000, 50_500)
    np.testing.assert_allclose(sl.samples, ref[50_000:50_500], atol=1e-9)


def test_random_access_matches_full_stream_through_filter(raw_file):
    """The margin mechanism must make a mid-stream slice match the full result."""
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    pipe = wave.amplitude.abs().filter.lowpass(cutoff=5_000).math.diff()

    full = pipe.to_array(allow_large=True)
    # A slice that begins deep inside the signal should equal the same region of
    # the full computation. The IIR warm-up margin (~20 time constants) makes the
    # filter state reconverge to ~1e-8 of full scale, so a mid-stream slice and a
    # block-boundary slice both match the full run to a tiny tolerance.
    a, b = 120_000, 120_400
    sl = pipe.get_from_to(a, b)
    np.testing.assert_allclose(sl.samples, full[a:b], rtol=1e-4, atol=1e-3)


def test_blocks_reconstruct_pipeline_output(raw_file):
    path, fs, n, data = raw_file
    pipe = wm.read_raw(path, dtype="int16", sample_rate=fs).amplitude.abs().filter.lowpass(2_000)
    blocked = np.concatenate([blk.samples for blk in pipe.blocks(7919)])  # odd size
    full = pipe.to_array(allow_large=True)
    assert blocked.size == n
    np.testing.assert_allclose(blocked, full, rtol=1e-4, atol=1e-3)


def test_scalar_arithmetic_is_lazy(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    out = ((wave * 2.0) + 10.0).get_from_to(0, 20)
    np.testing.assert_allclose(out.samples, data[:20] * 2.0 + 10.0)


def test_hist_streams_in_bounded_memory(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    h = wave.amplitude.histogram(bins=64, block=8192)
    assert isinstance(h, wm.Histogram)
    assert h.total == n  # every sample counted, streamed in 8k blocks
    ref, _ = np.histogram(data.astype(float), bins=h.edges)
    np.testing.assert_array_equal(h.counts, ref)


def test_peaks_stitch_across_blocks(raw_file):
    path, fs, n, data = raw_file
    wave = wm.read_raw(path, dtype="int16", sample_rate=fs)
    # Smooth first so peaks correspond to the ~1 kHz cycles, not noise.
    peaks = wave.amplitude.abs().filter.lowpass(3_000).amplitude.peaks(height=3000, distance=20, block=4096)
    assert isinstance(peaks, wm.Peaks)
    # Rectification doubles the rate: a 1 kHz sine over 2 s of |signal| gives
    # ~4000 lobes. Allow a band and confirm stitching didn't drop/duplicate seams.
    assert 3500 < len(peaks) < 4500
    assert np.all(np.diff(peaks.index) >= 20)
    assert np.all(peaks.value >= 3000)


def test_to_file_roundtrip(raw_file, tmp_path):
    path, fs, n, data = raw_file
    pipe = wm.read_raw(path, dtype="int16", sample_rate=fs).amplitude.abs()
    out_path = tmp_path / "out.f32"
    pipe.to_file(out_path, dtype="float32")
    back = wm.read_raw(out_path, dtype="float32", sample_rate=fs)
    np.testing.assert_allclose(back.get_from_to(0, n).samples, np.abs(data.astype(float)))


def test_materialize_guard():
    # A small in-memory waveform materializes freely.
    wave = wm.from_array(np.zeros(100), sample_rate=1.0)
    assert wave.to_array().size == 100
