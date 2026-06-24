import numpy as np
import pytest

import wave_measure as wm


@pytest.fixture
def sine():
    """A 1 kHz sine wave, 1 V amplitude, sampled at 100 kHz for 10 ms."""
    fs = 100_000.0
    t = np.arange(0, 0.01, 1 / fs)
    v = np.sin(2 * np.pi * 1000 * t)
    return wm.Waveform(samples=v, time=t)


def test_waveform_basics(sine):
    assert len(sine) == sine.samples.size
    assert sine.sample_rate == pytest.approx(100_000.0, rel=1e-3)
    assert sine.duration == pytest.approx(0.01, abs=1e-4)
    assert np.asarray(sine).shape == sine.samples.shape


def test_waveform_from_sample_rate():
    wf = wm.Waveform(samples=[0, 1, 2, 3], sample_rate=4.0)
    assert wf.dt == pytest.approx(0.25)
    np.testing.assert_allclose(wf.time, [0, 0.25, 0.5, 0.75])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        wm.Waveform(samples=[1, 2, 3], time=[0, 1])


def test_arithmetic(sine):
    doubled = sine * 2
    assert wm.vpp(doubled) == pytest.approx(2 * wm.vpp(sine))
    offset = sine + 1.0
    assert wm.mean(offset) == pytest.approx(wm.mean(sine) + 1.0, abs=1e-6)


def test_amplitude_measurements(sine):
    assert wm.vpp(sine) == pytest.approx(2.0, abs=1e-2)
    assert wm.vrms(sine) == pytest.approx(1 / np.sqrt(2), abs=1e-2)
    assert wm.mean(sine) == pytest.approx(0.0, abs=1e-2)


def test_frequency(sine):
    assert wm.frequency(sine) == pytest.approx(1000.0, rel=1e-2)
    assert wm.period(sine) == pytest.approx(1e-3, rel=1e-2)


def test_dominant_frequency(sine):
    assert wm.dominant_frequency(sine) == pytest.approx(1000.0, rel=2e-2)


def test_duty_cycle():
    # Square wave: half high, half low.
    t = np.linspace(0, 1, 1000, endpoint=False)
    v = np.where((t % 0.1) < 0.05, 1.0, -1.0)
    wf = wm.Waveform(samples=v, time=t)
    assert wm.duty_cycle(wf) == pytest.approx(0.5, abs=0.05)


def test_rise_time():
    # Linear ramp from 0 to 1 over 1 us, then hold.
    t = np.linspace(0, 2e-6, 2000)
    v = np.clip(t / 1e-6, 0, 1)
    wf = wm.Waveform(samples=v, time=t)
    # 10%-90% of a linear ramp spans 80% of the 1 us edge.
    assert wm.rise_time(wf) == pytest.approx(0.8e-6, rel=0.05)


def test_spectrum_shape(sine):
    freqs, mag = wm.spectrum(sine)
    assert freqs.shape == mag.shape
    assert freqs[0] == 0.0


def test_moving_average_and_detrend(sine):
    noisy = sine + 5.0  # add DC offset
    assert wm.mean(wm.detrend(noisy)) == pytest.approx(0.0, abs=1e-6)
    smoothed = wm.moving_average(noisy, 5)
    assert len(smoothed) == len(noisy)


def test_csv_roundtrip(tmp_path, sine):
    path = tmp_path / "capture.csv"
    wm.write_csv(sine, path)
    loaded = wm.read_csv(path)
    np.testing.assert_allclose(loaded.samples, sine.samples, rtol=1e-6)
    np.testing.assert_allclose(loaded.time, sine.time, rtol=1e-6)


def test_accelerator_detected():
    acc = wm.accelerator
    assert isinstance(acc, wm.AcceleratorInfo)
    assert acc.backend in ("cpu", "cuda")
    # Numba is a hard dependency, so a version must be reported.
    assert acc.numba_version is not None
    assert acc.is_gpu == (acc.backend == "cuda")
    assert "backend=" in acc.summary()


def test_accelerator_backend_consistency():
    acc = wm.detect_accelerator()
    # If we selected CUDA, the device must be both present and usable.
    if acc.backend == "cuda":
        assert acc.cuda_available and acc.nvidia_gpu_detected
    # A present-but-unusable NVIDIA GPU must leave an explanatory note.
    if acc.nvidia_gpu_detected and not acc.cuda_available:
        assert any("NVIDIA" in n for n in acc.notes)


def test_force_cpu_backend(monkeypatch):
    monkeypatch.setenv("WAVE_MEASURE_BACKEND", "cpu")
    acc = wm.detect_accelerator()
    assert acc.backend == "cpu"


def test_csv_with_metadata_header(tmp_path):
    path = tmp_path / "scope.csv"
    path.write_text(
        "Model,DSO-X 1102G\n"
        "Channel,CH1\n"
        "time,value\n"
        "0.0,0.0\n"
        "1e-6,0.5\n"
        "2e-6,1.0\n"
    )
    wf = wm.read_csv(path)
    assert len(wf) == 3
    assert wf.metadata.get("Model") == "DSO-X 1102G"
    assert wf.metadata.get("Channel") == "CH1"
