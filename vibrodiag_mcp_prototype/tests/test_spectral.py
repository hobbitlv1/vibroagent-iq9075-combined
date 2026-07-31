import numpy as np
import pytest

from vibroagent_mcp.spectral import (
    analyze_impact_ringdown,
    apply_frequency_filter,
    aggregate_spectrum_for_plot,
    aggregate_waveform_for_plot,
    compute_periodogram_psd,
    compute_welch_psd,
    find_distinct_peaks,
    integrated_psd_power,
)


def _tone_amplitude(values: np.ndarray, sampling_rate_hz: float, frequency_hz: float) -> float:
    times = np.arange(values.size, dtype=float) / float(sampling_rate_hz)
    sine = np.sin(2.0 * np.pi * float(frequency_hz) * times)
    cosine = np.cos(2.0 * np.pi * float(frequency_hz) * times)
    return float(2.0 * np.hypot(np.dot(values, sine), np.dot(values, cosine)) / values.size)


def test_periodogram_default_is_periodic_hann():
    signal_module = pytest.importorskip("scipy.signal")
    sampling_rate_hz = 1024.0
    rng = np.random.default_rng(31)
    values = rng.normal(size=2048)

    actual = compute_periodogram_psd(values, sampling_rate_hz)
    periodic_hann = np.hanning(values.size + 1)[:-1]
    expected_frequencies, expected_psd = signal_module.periodogram(
        values,
        fs=sampling_rate_hz,
        window=periodic_hann,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    assert actual.window_name == "hann"
    np.testing.assert_allclose(actual.frequencies_hz, expected_frequencies, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.psd_g2_per_hz, expected_psd, rtol=2e-12, atol=1e-18)


def test_periodogram_matches_scipy_density_scaling():
    signal_module = pytest.importorskip("scipy.signal")
    sampling_rate_hz = 2048.0
    rng = np.random.default_rng(7)
    values = rng.normal(size=4096) + 0.4 * np.sin(2.0 * np.pi * 128.0 * np.arange(4096) / sampling_rate_hz)

    actual = compute_periodogram_psd(values, sampling_rate_hz, window_name="hamming")
    periodic_hamming = np.hamming(values.size + 1)[:-1]
    expected_frequencies, expected_psd = signal_module.periodogram(
        values,
        fs=sampling_rate_hz,
        window=periodic_hamming,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    np.testing.assert_allclose(actual.frequencies_hz, expected_frequencies, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.psd_g2_per_hz, expected_psd, rtol=2e-12, atol=1e-18)


def test_welch_matches_scipy_for_selected_segment_configuration():
    signal_module = pytest.importorskip("scipy.signal")
    sampling_rate_hz = 4096.0
    rng = np.random.default_rng(17)
    values = rng.normal(size=32768) + 0.3 * np.sin(
        2.0 * np.pi * 321.0 * np.arange(32768) / sampling_rate_hz
    )

    actual = compute_welch_psd(
        values,
        sampling_rate_hz,
        target_resolution_hz=2.0,
        overlap_fraction=0.5,
    )
    assert actual.window_name == "hann"
    periodic_hann = np.hanning(actual.segment_length + 1)[:-1]
    expected_frequencies, expected_psd = signal_module.welch(
        values,
        fs=sampling_rate_hz,
        window=periodic_hann,
        nperseg=actual.segment_length,
        noverlap=actual.overlap_samples,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    np.testing.assert_allclose(actual.frequencies_hz, expected_frequencies, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.psd_g2_per_hz, expected_psd, rtol=3e-12, atol=1e-18)


def test_integrated_periodogram_power_matches_centered_rms_for_bin_centered_tones():
    sampling_rate_hz = 1024.0
    sample_count = 8192
    times = np.arange(sample_count, dtype=float) / sampling_rate_hz
    values = 0.6 * np.sin(2.0 * np.pi * 64.0 * times) + 0.2 * np.sin(2.0 * np.pi * 192.0 * times)

    estimate = compute_periodogram_psd(values, sampling_rate_hz)
    psd_rms = np.sqrt(integrated_psd_power(estimate.frequencies_hz, estimate.psd_g2_per_hz))
    centered_rms = np.sqrt(np.mean((values - values.mean()) ** 2))

    assert psd_rms == pytest.approx(centered_rms, rel=1e-10, abs=1e-12)


def test_welch_psd_averages_segments_and_returns_distinct_tones():
    sampling_rate_hz = 2000.0
    duration_s = 20.0
    times = np.arange(int(sampling_rate_hz * duration_s), dtype=float) / sampling_rate_hz
    rng = np.random.default_rng(9)
    values = (
        0.7 * np.sin(2.0 * np.pi * 123.0 * times)
        + 0.25 * np.sin(2.0 * np.pi * 400.0 * times)
        + 0.02 * rng.normal(size=times.size)
    )

    estimate = compute_welch_psd(values, sampling_rate_hz, target_resolution_hz=0.5)
    peaks = find_distinct_peaks(estimate.frequencies_hz, estimate.psd_g2_per_hz, limit=8)
    peak_frequencies = [float(item["frequency_hz"]) for item in peaks]

    assert estimate.segment_count > 1
    assert estimate.overlap_fraction == pytest.approx(0.5, abs=1e-3)
    assert any(abs(value - 123.0) <= estimate.resolution_hz for value in peak_frequencies)
    assert any(abs(value - 400.0) <= estimate.resolution_hz for value in peak_frequencies)
    assert len(peak_frequencies) == 2


def test_plot_aggregation_keeps_floor_and_peak_envelope_separate():
    frequencies = np.arange(100, dtype=float)
    psd = np.ones(100, dtype=float)
    psd[25] = 100.0

    plot = aggregate_spectrum_for_plot(frequencies, psd, max_bins=10, frequency_scale="linear")

    assert plot["frequency_hz"].size == 10
    assert np.max(plot["psd_g2_per_hz"]) == pytest.approx(1.0)
    assert np.max(plot["peak_psd_g2_per_hz"]) == pytest.approx(100.0)
    peak_bucket = int(np.argmax(plot["peak_psd_g2_per_hz"]))
    assert plot["peak_frequency_hz"][peak_bucket] == pytest.approx(25.0)


def test_welch_short_high_rate_capture_still_averages_multiple_segments():
    sampling_rate_hz = 26_667.0
    duration_s = 5.0
    t = np.arange(int(sampling_rate_hz * duration_s), dtype=float) / sampling_rate_hz
    signal = np.sin(2.0 * np.pi * 731.0 * t)

    estimate = compute_welch_psd(
        signal,
        sampling_rate_hz,
        target_resolution_hz=0.25,
        overlap_fraction=0.5,
    )

    assert estimate.segment_count >= 3
    assert estimate.segment_length < signal.size
    assert estimate.resolution_hz < 1.0


def test_waveform_plot_aggregation_preserves_impulse_and_bounds_payload():
    sampling_rate_hz = 1000.0
    values = np.zeros(10_000, dtype=float)
    values[4321] = 7.0
    timestamps = 12.0 + np.arange(values.size, dtype=float) / sampling_rate_hz

    plot = aggregate_waveform_for_plot(
        values,
        sampling_rate_hz,
        timestamps_s=timestamps,
        max_points=200,
        center=True,
    )

    assert plot["point_count"] == 200
    assert plot["downsampled"] is True
    assert plot["timestamp_source"] == "sdk_timestamps"
    assert plot["display_method"] == "min_max_mean_envelope"
    assert plot["time_s"][0] >= 0.0
    assert float(plot["time_s"][-1]) <= (values.size - 1) / sampling_rate_hz + 0.01
    assert np.max(plot["max_g"]) > 6.9
    assert np.min(plot["min_g"]) < 0.0


def test_psd_rejects_nonfinite_samples_instead_of_time_compressing_signal():
    with pytest.raises(ValueError, match="non-finite"):
        compute_periodogram_psd([0.0, 1.0, np.nan, -1.0], 100.0)


def test_flat_broadband_spectrum_does_not_report_arbitrary_distinct_peak():
    frequencies = np.arange(0.0, 101.0)
    psd = np.ones_like(frequencies)

    assert find_distinct_peaks(frequencies, psd, min_prominence_db=3.0) == []


def test_impact_ringdown_recovers_short_damped_structural_resonance():
    sampling_rate_hz = 1000.0
    duration_s = 20.0
    sample_count = int(sampling_rate_hz * duration_s)
    rng = np.random.default_rng(3)
    values = 0.002 * rng.normal(size=sample_count)
    event_index = int(15.0 * sampling_rate_hz)
    values[event_index] += 2.0
    ringdown_time = np.arange(sample_count - event_index, dtype=float) / sampling_rate_hz
    values[event_index:] += (
        0.01
        * np.exp(-ringdown_time / 0.2)
        * np.sin(2.0 * np.pi * 37.0 * ringdown_time)
    )

    result = analyze_impact_ringdown(values, sampling_rate_hz)

    assert result["detected"] is True
    assert result["status"] == "ringdown_peak"
    assert result["peak_reliable"] is True
    assert result["supporting_duration_count"] >= 2
    assert result["peak_frequency_hz"] == pytest.approx(37.0, abs=1.5)
    assert result["ringdown_energy_ratio"] > 1.3
    assert result["decay_ratio"] > 1.1
    assert result["frequencies_hz"].size == result["psd_g2_per_hz"].size


def test_broadband_impulse_does_not_manufacture_ringdown_frequency():
    sampling_rate_hz = 1000.0
    sample_count = 20_000
    rng = np.random.default_rng(11)
    values = 0.002 * rng.normal(size=sample_count)
    values[15_000] += 2.0

    result = analyze_impact_ringdown(values, sampling_rate_hz)

    assert result["detected"] is True
    assert result["status"] == "broadband_impact_no_ringdown_peak"
    assert result["peak_frequency_hz"] is None
    assert result["peak_reliable"] is False


def test_steady_tone_is_not_misclassified_as_an_impact():
    sampling_rate_hz = 1000.0
    times = np.arange(20_000, dtype=float) / sampling_rate_hz
    values = np.sin(2.0 * np.pi * 15.0 * times)

    result = analyze_impact_ringdown(values, sampling_rate_hz)

    assert result["detected"] is False
    assert result["status"] == "no_transient"
    assert result["peak_frequency_hz"] is None


def test_impact_near_window_end_requests_more_ringdown_data():
    sampling_rate_hz = 1000.0
    values = np.zeros(10_000, dtype=float)
    values[9_800] = 2.0

    result = analyze_impact_ringdown(values, sampling_rate_hz)

    assert result["detected"] is True
    assert result["status"] == "need_more_post_event_data"
    assert result["peak_frequency_hz"] is None
    assert result["available_post_event_s"] < 0.5


def test_optional_highpass_filter_attenuates_low_tone_and_preserves_high_tone():
    sampling_rate_hz = 1000.0
    times = np.arange(10_000, dtype=float) / sampling_rate_hz
    values = np.sin(2.0 * np.pi * 5.0 * times) + np.sin(2.0 * np.pi * 100.0 * times)

    filtered, metadata = apply_frequency_filter(
        values,
        sampling_rate_hz,
        filter_type="highpass",
        low_cutoff_hz=20.0,
        filter_order=4,
    )

    assert metadata["applied"] is True
    assert metadata["type"] == "highpass"
    assert metadata["phase_response"] == "zero phase"
    assert metadata["anti_alias_filter"] is False
    assert _tone_amplitude(filtered, sampling_rate_hz, 5.0) < 0.03
    assert _tone_amplitude(filtered, sampling_rate_hz, 100.0) > 0.97


def test_optional_lowpass_and_bandpass_filters_select_expected_tones():
    sampling_rate_hz = 1000.0
    times = np.arange(10_000, dtype=float) / sampling_rate_hz
    values = (
        np.sin(2.0 * np.pi * 10.0 * times)
        + np.sin(2.0 * np.pi * 100.0 * times)
        + np.sin(2.0 * np.pi * 300.0 * times)
    )

    lowpassed, low_metadata = apply_frequency_filter(
        values,
        sampling_rate_hz,
        filter_type="lowpass",
        high_cutoff_hz=30.0,
    )
    bandpassed, band_metadata = apply_frequency_filter(
        values,
        sampling_rate_hz,
        filter_type="bandpass",
        low_cutoff_hz=80.0,
        high_cutoff_hz=120.0,
    )

    assert low_metadata["type"] == "lowpass"
    assert _tone_amplitude(lowpassed, sampling_rate_hz, 10.0) > 0.97
    assert _tone_amplitude(lowpassed, sampling_rate_hz, 100.0) < 0.03
    assert band_metadata["type"] == "bandpass"
    assert _tone_amplitude(bandpassed, sampling_rate_hz, 10.0) < 0.03
    assert _tone_amplitude(bandpassed, sampling_rate_hz, 100.0) > 0.75
    assert _tone_amplitude(bandpassed, sampling_rate_hz, 300.0) < 0.03


def test_mains_notch_filter_removes_selected_frequency_without_moving_other_tone():
    sampling_rate_hz = 1000.0
    times = np.arange(10_000, dtype=float) / sampling_rate_hz
    values = np.sin(2.0 * np.pi * 50.0 * times) + np.sin(2.0 * np.pi * 80.0 * times)

    filtered, metadata = apply_frequency_filter(
        values,
        sampling_rate_hz,
        filter_type="notch_50",
        notch_width_hz=2.0,
        filter_order=4,
    )

    assert metadata["type"] == "notch"
    assert metadata["preset"] == "50_hz_mains"
    assert metadata["notch_frequency_hz"] == pytest.approx(50.0)
    assert _tone_amplitude(filtered, sampling_rate_hz, 50.0) < 0.03
    assert _tone_amplitude(filtered, sampling_rate_hz, 80.0) > 0.97


def test_none_filter_only_removes_mean_and_invalid_filter_edges_are_rejected():
    values = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=float)
    centered, metadata = apply_frequency_filter(values, 1000.0, filter_type="none")

    np.testing.assert_allclose(centered, values - values.mean())
    assert metadata["applied"] is False
    assert metadata["type"] == "none"

    with pytest.raises(ValueError, match="lower than"):
        apply_frequency_filter(
            np.arange(100, dtype=float),
            1000.0,
            filter_type="bandpass",
            low_cutoff_hz=200.0,
            high_cutoff_hz=100.0,
        )
    with pytest.raises(ValueError, match="below Nyquist"):
        apply_frequency_filter(
            np.arange(100, dtype=float),
            1000.0,
            filter_type="lowpass",
            high_cutoff_hz=500.0,
        )
