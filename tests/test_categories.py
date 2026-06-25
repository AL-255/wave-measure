"""Tests for the categorized operator API (filter / amplitude / math)."""

import numpy as np
import pytest

import wave_measure as wm


@pytest.fixture
def noisy(tmp_path):
    fs = 100_000.0
    n = 60_000
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    v = np.sin(2 * np.pi * 1000 * t) + 0.3 * rng.standard_normal(n) + 2.0  # +DC
    path = tmp_path / "w.f64"
    v.astype(np.float64).tofile(path)
    wave = wm.read_raw(path, dtype="float64", sample_rate=fs)
    return wave, v, fs, n


# -- filter category --------------------------------------------------------


def test_fir_is_exact_under_random_access(noisy):
    wave, v, fs, n = noisy
    taps = np.array([0.2, 0.3, 0.3, 0.2])
    pipe = wave.filter.fir(taps)
    ref = np.convolve(v, taps)[:n]  # causal FIR reference
    full = pipe.to_array(allow_large=True)
    np.testing.assert_allclose(full, ref, atol=1e-9)
    # FIR has an exact finite margin -> mid-stream slice matches bit-for-bit.
    sl = pipe.get_from_to(30_000, 30_200)
    np.testing.assert_allclose(sl.samples, ref[30_000:30_200], atol=1e-9)


def test_moving_average_smooths(noisy):
    wave, v, fs, n = noisy
    ma = wave.filter.moving_average(8)
    ref = np.convolve(v, np.full(8, 1 / 8))[:n]
    np.testing.assert_allclose(ma.to_array(allow_large=True), ref, atol=1e-9)


def test_iir_matches_reference_and_random_access(noisy):
    wave, v, fs, n = noisy
    b, a = [1.0], [1.0, -0.5]  # y[n] = x[n] + 0.5 y[n-1]
    ref = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = v[i] + 0.5 * acc
        ref[i] = acc
    pipe = wave.filter.iir(b, a)
    full = pipe.to_array(allow_large=True)
    np.testing.assert_allclose(full, ref, rtol=1e-6, atol=1e-6)
    sl = pipe.get_from_to(40_000, 40_300)
    np.testing.assert_allclose(sl.samples, ref[40_000:40_300], rtol=1e-6, atol=1e-6)


def test_highpass_removes_dc(noisy):
    wave, v, fs, n = noisy
    out = wave.filter.highpass(1_000)
    # The +2.0 DC offset should be gone after a settled high-pass.
    settled = out.get_from_to(20_000, 40_000).samples
    assert abs(settled.mean()) < 0.05


def test_median_kills_impulses():
    sig = np.zeros(21)
    sig[10] = 100.0  # lone spike
    wave = wm.from_array(sig, sample_rate=1.0)
    out = wave.filter.median(3).to_array()
    assert out[10] == 0.0  # spike removed
    np.testing.assert_array_equal(out, np.zeros(21))


def test_bandpass_chains_two_filters(noisy):
    wave, v, fs, n = noisy
    bp = wave.filter.bandpass(500, 5_000)
    assert len(bp.ops) == 2  # highpass then lowpass
    assert len(bp) == n


# -- amplitude category -----------------------------------------------------


def test_amplitude_stats_match_numpy(noisy):
    wave, v, fs, n = noisy
    s = wave.amplitude.stats(block=4096)
    assert s.count == n
    assert s.min == pytest.approx(v.min())
    assert s.max == pytest.approx(v.max())
    assert s.mean == pytest.approx(v.mean(), rel=1e-9)
    assert s.rms == pytest.approx(np.sqrt(np.mean(v * v)), rel=1e-9)
    assert s.peak_to_peak == pytest.approx(v.max() - v.min())


def test_amplitude_scalar_reductions(noisy):
    wave, v, fs, n = noisy
    assert wave.amplitude.min() == pytest.approx(v.min())
    assert wave.amplitude.max() == pytest.approx(v.max())
    assert wave.amplitude.mean() == pytest.approx(v.mean(), rel=1e-9)
    assert wave.amplitude.peak_to_peak() == pytest.approx(v.max() - v.min())


def test_amplitude_length_preserving_ops(noisy):
    wave, v, fs, n = noisy
    out = wave.amplitude.gain(2.0).amplitude.offset(1.0).amplitude.clip(-3, 3)
    ref = np.clip(v * 2.0 + 1.0, -3, 3)
    np.testing.assert_allclose(out.to_array(allow_large=True), ref, atol=1e-9)


def test_top_bottom_levels(tmp_path):
    # A two-level (digital) signal: lows near 0.0, highs near 3.3, plus noise
    # and transition edges between them.
    fs = 10e6
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 4000)
    samples_per_bit = 50
    level = np.repeat(np.where(bits == 1, 3.3, 0.0), samples_per_bit)
    v = level + rng.normal(0, 0.05, level.size)
    path = tmp_path / "logic.f64"
    v.astype(np.float64).tofile(path)
    wave = wm.read_raw(path, dtype="float64", sample_rate=fs)

    lv = wave.amplitude.levels(bins=256)
    assert isinstance(lv, wm.Levels)
    assert lv.bottom == pytest.approx(0.0, abs=0.1)
    assert lv.top == pytest.approx(3.3, abs=0.1)
    assert lv.amplitude == pytest.approx(3.3, abs=0.15)
    # top()/bottom() delegate to the same fit.
    assert wave.amplitude.top() == pytest.approx(lv.top)
    assert wave.amplitude.bottom() == pytest.approx(lv.bottom)
    # ordering invariant
    assert wave.amplitude.top() > wave.amplitude.bottom()
    # module-level parity
    assert wm.amplitude.top(wave) == pytest.approx(lv.top)


def test_top_bottom_dc_signal():
    # A flat (DC) signal has no two levels: both modes collapse to one another,
    # at the occupied bin (within histogram resolution of the value).
    wave = wm.from_array(np.full(2000, 1.5), sample_rate=1.0)
    lv = wave.amplitude.levels(value_range=(1.0, 2.0), bins=256)
    assert lv.top == pytest.approx(lv.bottom)
    assert lv.top == pytest.approx(1.5, abs=0.01)


# -- math category ----------------------------------------------------------


def test_math_ops(noisy):
    wave, v, fs, n = noisy
    pos = wave.amplitude.abs().math.square()  # ensure >=0 before sqrt
    np.testing.assert_allclose(
        pos.to_array(allow_large=True), v * v, atol=1e-9
    )
    back = wave.amplitude.abs().math.square().math.sqrt()
    np.testing.assert_allclose(back.to_array(allow_large=True), np.abs(v), atol=1e-6)


# -- module-level catalog ---------------------------------------------------


def test_module_catalog_builds_operators():
    op = wm.filter.fir([0.5, 0.5])
    assert hasattr(op, "apply")  # an Operator instance
    src = wm.ArraySource([1.0, 2.0, 3.0, 4.0], sample_rate=1.0)
    wave = wm.Waveform(source=src, ops=[op])
    np.testing.assert_allclose(wave.to_array(), np.convolve([1, 2, 3, 4], [0.5, 0.5])[:4])


def test_module_catalog_functional_reductions(noisy):
    wave, v, fs, n = noisy
    # wm.amplitude.mean(wave) mirrors wave.amplitude.mean()
    assert wm.amplitude.mean(wave) == pytest.approx(wave.amplitude.mean())
    assert wm.amplitude.histogram(wave, bins=32).total == n
