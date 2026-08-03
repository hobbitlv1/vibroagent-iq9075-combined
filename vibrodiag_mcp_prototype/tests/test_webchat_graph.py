import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from vibroagent_mcp import webchat_server
from vibroagent_mcp.schemas import OUT_OF_DOMAIN_COMPONENT_TERMS, find_out_of_domain_component_terms
from vibroagent_mcp.sdk_vibrometer import SdkVibrometerWindow


@pytest.fixture(autouse=True)
def _allow_tmp_sensor_config_dirs(monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", tempfile.gettempdir())


def test_request_origin_policy_rejects_cross_origin_browser_request():
    assert webchat_server._request_origin_allowed(
        host_header="127.0.0.1:7860",
        origin_header="https://evil.example",
        referer_header=None,
        sec_fetch_site="cross-site",
    ) is False


def test_request_origin_policy_allows_same_origin_browser_request():
    assert webchat_server._request_origin_allowed(
        host_header="127.0.0.1:7860",
        origin_header="http://127.0.0.1:7860",
        referer_header=None,
        sec_fetch_site="same-origin",
    ) is True


def test_request_model_base_url_rejects_remote_by_default(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOW_REMOTE_MODEL_URLS", raising=False)

    with pytest.raises(ValueError):
        webchat_server._request_model_base_url("https://model.example/v1", default=None)


def test_request_model_base_url_allows_loopback():
    assert (
        webchat_server._request_model_base_url("http://127.0.0.1:8910/v1", default=None)
        == "http://127.0.0.1:8910/v1"
    )


def test_live_sensors_payload_rejects_config_outside_allowed_roots(monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", tempfile.gettempdir())

    payload = webchat_server._live_sensors_payload({"config_path": ["/etc/passwd"]})

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"


def test_live_sensors_payload_reads_live_registry(tmp_path):
    baseline_dir = tmp_path / "live_baseline"
    target_dir = tmp_path / "live_target_1"
    baseline_dir.mkdir()
    target_dir.mkdir()
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "location": "reference",
                        "role": "baseline",
                        "acquisition_folder": str(baseline_dir),
                        "hsd_sensor_name": "iis3dwb_acc",
                        "axis": "z",
                    },
                    {
                        "sensor_id": "target_1",
                        "location": "target",
                        "role": "target",
                        "acquisition_folder": str(target_dir),
                        "hsd_sensor_name": "iis3dwb_acc",
                        "axis": "z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = webchat_server._live_sensors_payload({"config_path": [str(config)]})

    assert payload["ok"] is True
    assert payload["baseline_sensor_id"] == "baseline"
    assert [sensor["sensor_id"] for sensor in payload["sensors"]] == ["baseline", "target_1"]
    assert payload["sensors"][1]["acquisition_folder"] == str(target_dir)


def test_live_vibrometer_payload_passes_selected_folder(monkeypatch):
    captured = {}

    def fake_read_sdk_vibrometer_window(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=np.asarray([1.0, 1.2, 0.9, 1.1], dtype=float),
            timestamps_s=np.asarray([0.0, 0.01, 0.02, 0.03], dtype=float),
            sampling_rate_hz=100,
            metadata={
                "machine_id": kwargs["machine_id"],
                "acquisition_folder": kwargs["acquisition_folder"],
                "sensor_name": kwargs["sensor_name"],
                "axis": kwargs["axis"],
                "data_is_current": True,
                "data_source_mode": "current_or_live_file",
                "data_file_age_s": 0.02,
                "sdk_latest_timestamp_gap_s": 0.25,
            },
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._live_vibrometer_payload(
        {
            "machine_id": ["target_1"],
            "acquisition_folder": ["/tmp/live_target_1"],
            "sensor_name": ["iis3dwb_acc"],
            "axis": ["z"],
            "duration_s": ["0.5"],
            "max_points": ["200"],
        }
    )

    assert payload["ok"] is True
    assert captured["machine_id"] == "target_1"
    assert captured["acquisition_folder"] == "/tmp/live_target_1"
    assert captured["sensor_name"] == "iis3dwb_acc"
    assert captured["axis"] == "z"
    assert payload["machine_id"] == "target_1"
    assert payload["acquisition_folder"] == "/tmp/live_target_1"
    assert payload["diagnostics"]["data_file_age_s"] == 0.02
    assert payload["diagnostics"]["live_latency_s"] == 0.02
    assert payload["diagnostics"]["sdk_timestamp_gap_s"] == 0.25


def test_effective_live_latency_uses_sdk_gap_when_decoded_stream_lags():
    assert (
        webchat_server._effective_live_latency_s(
            {
                "data_source_mode": "sdk_decoded_stream_lagging",
                "data_file_age_s": 0.02,
                "sdk_latest_timestamp_gap_s": 12.5,
            }
        )
        == 12.5
    )


def test_live_vibrometer_payload_rejects_invalid_axis(monkeypatch):
    called = False

    def fake_read_sdk_vibrometer_window(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("read should not be called for an invalid axis")

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._live_vibrometer_payload({"axis": ["vertical"]})

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert called is False


def test_live_vibrometer_payload_handles_empty_sdk_window(monkeypatch):
    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=np.asarray([], dtype=float),
            timestamps_s=np.asarray([], dtype=float),
            sampling_rate_hz=100,
            metadata={"data_is_current": True},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._live_vibrometer_payload({})

    assert payload["ok"] is False
    assert payload["error"] == "invalid_window"
    assert "finite samples" in payload["message"]


def test_live_vibrometer_payload_filters_nonfinite_samples_and_defaults_bad_numbers(monkeypatch):
    captured = {}

    def fake_read_sdk_vibrometer_window(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=np.asarray([1.0, np.nan, 2.0, np.inf, 3.0], dtype=float),
            timestamps_s=None,
            sampling_rate_hz=10,
            metadata={"data_is_current": True},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._live_vibrometer_payload(
        {"duration_s": ["nan"], "max_points": ["not-an-int"], "require_current": ["0"]}
    )

    assert payload["ok"] is True
    assert captured["duration_s"] == 1.0
    assert captured["require_current"] is False
    assert payload["sample_count"] == 3
    assert payload["samples_g"] == [1.0, 2.0, 3.0]
    assert payload["stats"]["rms_g"] > 0


def test_live_vibrometer_payload_accepts_numeric_string_sampling_rate(monkeypatch):
    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
            timestamps_s=None,
            sampling_rate_hz=cast(int, "100.0"),
            metadata={"data_is_current": True},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._live_vibrometer_payload({})

    assert payload["ok"] is True
    assert payload["sampling_rate_hz"] == 100



def test_psd_fft_payload_defaults_to_welch_and_finds_distinct_peak(monkeypatch):
    captured = {}
    sampling_rate_hz = 1000
    duration_s = 100
    peak_hz = 123.0
    t = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    signal = np.sin(2.0 * np.pi * peak_hz * t)

    def fake_read_sdk_vibrometer_window(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=t,
            sampling_rate_hz=sampling_rate_hz,
            metadata={
                "data_is_current": False,
                "data_source_mode": "saved_replay",
                "timestamp_end_s": float(t[-1]),
                "estimated_data_duration_s": float(duration_s),
            },
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload({"max_bins": ["1000"]})

    assert payload["ok"] is True
    assert payload["estimator"] == "welch"
    assert payload["window_name"] == "hann"
    assert payload["plot_scale"] == "linear"
    assert payload["preprocessing"]["analysis_window"] == "periodic hann"
    assert payload["preprocessing"]["frequency_filter"] == "none"
    assert payload["preprocessing"]["signal_decimation_before_psd"] is False
    assert "nonlinear overview" in payload["axis_warning"]
    assert captured["duration_s"] == 100.0
    assert captured["machine_id"] == "baseline"
    assert Path(captured["acquisition_folder"]).name == "live_baseline"
    assert payload["source"]["sensor_id"] == "baseline"
    assert payload["source"]["selection_mode"] == "registry_baseline_default"
    assert payload["source"]["locked"] is True
    assert captured["require_current"] is False
    assert captured["max_duration_s"] == webchat_server.MAX_PSD_FFT_WINDOW_S
    assert payload["sample_count"] == sampling_rate_hz * duration_s
    assert payload["record_resolution_hz"] == 0.01
    assert payload["fft_resolution_hz"] > payload["record_resolution_hz"]
    assert payload["segment_count"] > 1
    assert abs(payload["peak_frequency_hz"] - peak_hz) <= payload["fft_resolution_hz"]
    assert payload["peak_phase_degrees"] is None
    assert payload["points_returned"] <= 1000
    assert payload["downsampled"] is True
    assert payload["top_peaks"]
    assert abs(payload["power_consistency_ratio"] - 1.0) < 0.01
    assert payload["variance_units"] == "g^2"
    assert "Integral of the one-sided PSD" in payload["psd_variance_definition"]
    assert payload["psd_variance_g2"] == pytest.approx(payload["integrated_psd_power_g2"])
    assert payload["psd_rms_g"] ** 2 == pytest.approx(payload["psd_variance_g2"])
    assert payload["centered_rms_g"] ** 2 == pytest.approx(payload["centered_variance_g2"])
    assert payload["analysis_stats"]["variance_g2"] == pytest.approx(payload["centered_variance_g2"])
    assert payload["stats"]["variance_g2"] == pytest.approx(float(np.var(signal)))
    assert abs(payload["variance_consistency_ratio"] - 1.0) < 0.02
    assert payload["peak_envelope_db_re_1_g2_per_hz"]
    assert payload["waveform_centered"] is True
    assert payload["waveform_downsampled"] is True
    assert payload["waveform_points"] == webchat_server.DEFAULT_PSD_WAVEFORM_MAX_POINTS
    assert len(payload["waveform_time_s"]) == payload["waveform_points"]
    assert len(payload["waveform_min_g"]) == payload["waveform_points"]
    assert len(payload["waveform_max_g"]) == payload["waveform_points"]
    assert len(payload["waveform_mean_g"]) == payload["waveform_points"]


def test_psd_fft_payload_reports_mean_removed_signal_variance(monkeypatch):
    sampling_rate_hz = 1000
    duration_s = 4
    amplitude_g = 0.3
    offset_g = 2.0
    tone_hz = 40.0
    times = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    signal = offset_g + amplitude_g * np.sin(2.0 * np.pi * tone_hz * times)

    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=times,
            sampling_rate_hz=sampling_rate_hz,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {
            "axis": ["x"],
            "estimator": ["periodogram"],
            "duration_s": [str(duration_s)],
            "max_bins": ["5000"],
        }
    )

    expected_variance_g2 = amplitude_g**2 / 2.0
    assert payload["ok"] is True
    assert payload["stats"]["rms_g"] > payload["stats"]["centered_rms_g"]
    assert payload["stats"]["variance_g2"] == pytest.approx(expected_variance_g2, rel=1e-10)
    assert payload["centered_variance_g2"] == pytest.approx(expected_variance_g2, rel=1e-10)
    assert payload["psd_variance_g2"] == pytest.approx(expected_variance_g2, rel=1e-10)
    assert payload["psd_rms_g"] == pytest.approx(amplitude_g / np.sqrt(2.0), rel=1e-10)
    assert payload["variance_consistency_ratio"] == pytest.approx(1.0, rel=1e-10)


def test_psd_fft_payload_locks_each_run_to_the_selected_registry_board(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "live_baseline"
    target_dir = tmp_path / "live_target_1"
    baseline_dir.mkdir()
    target_dir.mkdir()
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "location": "reference",
                        "role": "baseline",
                        "acquisition_folder": str(baseline_dir),
                        "hsd_sensor_name": "iis3dwb_acc",
                        "axis": "z",
                    },
                    {
                        "sensor_id": "target_1",
                        "location": "target",
                        "role": "target",
                        "acquisition_folder": str(target_dir),
                        "hsd_sensor_name": "iis3dwb_acc",
                        "axis": "z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    sampling_rate_hz = 1000
    duration_s = 8
    times = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    calls: list[dict] = []

    def fake_read_sdk_vibrometer_window(**kwargs):
        calls.append(dict(kwargs))
        frequency_hz = 15.0 if kwargs["machine_id"] == "baseline" else 100.0
        signal = np.sin(2.0 * np.pi * frequency_hz * times)
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=times,
            sampling_rate_hz=sampling_rate_hz,
            metadata={
                "machine_id": kwargs["machine_id"],
                "sensor_name": kwargs["sensor_name"],
                "acquisition_folder": kwargs["acquisition_folder"],
                "data_is_current": True,
            },
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    baseline = webchat_server._psd_fft_payload(
        {
            "config_path": [str(config)],
            "sensor_id": ["baseline"],
            "duration_s": [str(duration_s)],
        }
    )
    target = webchat_server._psd_fft_payload(
        {
            "config_path": [str(config)],
            "sensor_id": ["target_1"],
            "duration_s": [str(duration_s)],
        }
    )

    assert baseline["ok"] is True
    assert target["ok"] is True
    assert calls[0]["machine_id"] == "baseline"
    assert Path(calls[0]["acquisition_folder"]) == baseline_dir
    assert calls[1]["machine_id"] == "target_1"
    assert Path(calls[1]["acquisition_folder"]) == target_dir
    assert baseline["source"]["sensor_id"] == "baseline"
    assert target["source"]["sensor_id"] == "target_1"
    assert abs(baseline["peak_frequency_hz"] - 15.0) <= baseline["fft_resolution_hz"]
    assert abs(target["peak_frequency_hz"] - 100.0) <= target["fft_resolution_hz"]


def test_psd_fft_payload_uses_impact_ringdown_when_long_welch_has_no_peak(monkeypatch):
    sampling_rate_hz = 1000
    duration_s = 20
    sample_count = sampling_rate_hz * duration_s
    times = np.arange(sample_count, dtype=float) / sampling_rate_hz
    rng = np.random.default_rng(3)
    signal = 0.002 * rng.normal(size=sample_count)
    event_index = 15 * sampling_rate_hz
    signal[event_index] += 2.0
    ringdown_time = np.arange(sample_count - event_index, dtype=float) / sampling_rate_hz
    signal[event_index:] += (
        0.01
        * np.exp(-ringdown_time / 0.2)
        * np.sin(2.0 * np.pi * 37.0 * ringdown_time)
    )

    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=times,
            sampling_rate_hz=sampling_rate_hz,
            metadata={"data_is_current": True, "full_scale_g": 16.0},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {
            "axis": ["z"],
            "duration_s": [str(duration_s)],
            "max_bins": ["1000"],
        }
    )

    assert payload["ok"] is True
    assert payload["peak_frequency_hz"] is None
    assert payload["impact_analysis"]["detected"] is True
    assert payload["impact_analysis"]["peak_reliable"] is True
    assert payload["impact_peak_frequency_hz"] == pytest.approx(37.0, abs=1.5)
    assert payload["effective_peak_frequency_hz"] == payload["impact_peak_frequency_hz"]
    assert payload["effective_peak_kind"] == "impact_ringdown"
    assert payload["effective_peak_reliable"] is True
    assert payload["impact_spectrum_frequency_hz"]
    assert len(payload["impact_spectrum_frequency_hz"]) == len(
        payload["impact_spectrum_db_re_1_g2_per_hz"]
    )


def test_psd_fft_payload_reports_nearly_equal_distinct_peaks_as_competing(monkeypatch):
    sampling_rate_hz = 1000
    duration_s = 20
    times = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    signal = (
        np.sin(2.0 * np.pi * 15.0 * times)
        + np.sin(2.0 * np.pi * 100.0 * times)
    )

    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=times,
            sampling_rate_hz=sampling_rate_hz,
            metadata={"data_is_current": True},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {
            "axis": ["x"],
            "duration_s": [str(duration_s)],
            "max_bins": ["5000"],
        }
    )

    assert payload["ok"] is True
    assert payload["peak_is_competing"] is True
    assert payload["peak_lead_db"] is not None
    assert payload["peak_level_separation_db"] == payload["peak_lead_db"]
    assert payload["peak_lead_db"] < 3.0
    observed = sorted(
        [payload["peak_frequency_hz"], payload["competing_peak_frequency_hz"]]
    )
    assert observed[0] == pytest.approx(15.0, abs=payload["fft_resolution_hz"])
    assert observed[1] == pytest.approx(100.0, abs=payload["fft_resolution_hz"])


def test_psd_fft_payload_rejects_unknown_registry_board_before_read(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "live_baseline"
    baseline_dir.mkdir()
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "location": "reference",
                        "role": "baseline",
                        "acquisition_folder": str(baseline_dir),
                        "hsd_sensor_name": "iis3dwb_acc",
                        "axis": "z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    called = False

    def fake_read_sdk_vibrometer_window(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("reader must not run for an unknown board")

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {"config_path": [str(config)], "sensor_id": ["target_99"]}
    )

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert "Unknown PSD sensor_id" in payload["message"]
    assert called is False


def test_psd_fft_payload_supports_raw_periodogram_with_phase(monkeypatch):
    sampling_rate_hz = 1000
    duration_s = 10
    peak_hz = 125.0
    t = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    signal = np.sin(2.0 * np.pi * peak_hz * t)

    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=t,
            sampling_rate_hz=sampling_rate_hz,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {"estimator": ["periodogram"], "duration_s": [str(duration_s)], "max_bins": ["5000"]}
    )

    assert payload["ok"] is True
    assert payload["estimator"] == "periodogram"
    assert payload["window_name"] == "hann"
    assert payload["segment_count"] == 1
    assert payload["fft_resolution_hz"] == payload["record_resolution_hz"] == 0.1
    assert abs(payload["peak_frequency_hz"] - peak_hz) <= payload["fft_resolution_hz"]
    assert payload["peak_phase_degrees"] is not None


def test_psd_fft_payload_applies_filter_to_both_waveform_and_spectrum(monkeypatch):
    sampling_rate_hz = 1000
    duration_s = 10
    times = np.arange(sampling_rate_hz * duration_s, dtype=float) / sampling_rate_hz
    signal = 2.0 * np.sin(2.0 * np.pi * 5.0 * times) + 0.5 * np.sin(2.0 * np.pi * 100.0 * times)

    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=signal,
            timestamps_s=times,
            sampling_rate_hz=sampling_rate_hz,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {
            "estimator": ["periodogram"],
            "duration_s": [str(duration_s)],
            "filter_type": ["highpass"],
            "low_cutoff_hz": ["20"],
            "filter_order": ["4"],
            "max_bins": ["5000"],
        }
    )

    assert payload["ok"] is True
    assert payload["filter_requested"] == "highpass"
    assert payload["filter"]["type"] == "highpass"
    assert payload["filter"]["applied"] is True
    assert payload["filter"]["low_cutoff_hz"] == 20.0
    assert payload["filter"]["phase_response"] == "zero phase"
    assert payload["preprocessing"]["frequency_filter"].startswith("High-pass 20")
    assert payload["preprocessing"]["signal_decimation_before_psd"] is False
    assert payload["analysis_stats"]["centered_rms_g"] < payload["stats"]["centered_rms_g"] * 0.4
    assert payload["analysis_stats"]["variance_g2"] < payload["stats"]["variance_g2"] * 0.16
    assert payload["psd_variance_g2"] == pytest.approx(
        payload["analysis_stats"]["variance_g2"], rel=5e-4
    )
    assert abs(payload["peak_frequency_hz"] - 100.0) <= payload["fft_resolution_hz"]
    assert payload["diagnostics"]["filter_duration_s"] >= 0.0
    assert len(payload["waveform_time_s"]) == len(payload["waveform_mean_g"])


def test_psd_fft_payload_rejects_invalid_filter_before_read(monkeypatch):
    called = False

    def fake_read_sdk_vibrometer_window(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("read should not be called for an invalid filter")

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload({"filter_type": ["magic_filter"]})

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert called is False


def test_psd_fft_payload_rejects_filter_cutoff_at_nyquist(monkeypatch):
    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=np.arange(1000, dtype=float),
            timestamps_s=None,
            sampling_rate_hz=1000,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload(
        {"filter_type": ["lowpass"], "high_cutoff_hz": ["500"]}
    )

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert "below Nyquist" in payload["message"]


def test_psd_fft_payload_rejects_invalid_axis(monkeypatch):
    called = False

    def fake_read_sdk_vibrometer_window(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("read should not be called for an invalid axis")

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload({"axis": ["vertical"]})

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert called is False


def test_psd_fft_payload_rejects_nonfinite_samples_without_closing_time_gaps(monkeypatch):
    def fake_read_sdk_vibrometer_window(**kwargs):
        return SdkVibrometerWindow(
            signal=np.asarray([0.0, 1.0, np.nan, -1.0], dtype=float),
            timestamps_s=np.asarray([0.0, 0.01, 0.02, 0.03], dtype=float),
            sampling_rate_hz=100,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read_sdk_vibrometer_window)

    payload = webchat_server._psd_fft_payload({})

    assert payload["ok"] is False
    assert payload["error"] == "invalid_window"
    assert "distort the time axis" in payload["message"]



def test_agent_monitor_payload_builds_popup_for_network_anomaly(monkeypatch, tmp_path):
    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")
    captured = {}
    llm_calls = []

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "window_id": "window-1",
            "network_assessment": {
                "network_label": "localized_vibration_deviation",
                "main_agent_confidence": 0.82,
                "affected_sensor_ids": ["target_1"],
                "significant_sensor_ids": ["target_1"],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "Target 1 deviated from the reference sensor.",
                "model_metadata": {"model_used": False, "fallback_reason": "disabled_for_fast_monitor"},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_1",
                    "small_agent_label": "significant_local_deviation",
                    "model_metadata": {"model_used": False, "fallback_reason": "disabled_for_fast_monitor"},
                }
            ],
            "model_metadata": {
                "main_model_used": False,
                "small_model_used": False,
                "fallbacks_used": [
                    "small_sensor_agent:disabled_for_fast_monitor",
                    "main_building_agent:disabled_for_fast_monitor",
                ],
            },
        }

    def fake_apply_codes_monitor_decision(result, **kwargs):
        llm_calls.append(kwargs)
        updated = dict(result)
        updated["model_metadata"] = {
            **dict(updated.get("model_metadata") or {}),
            "monitor_llm_model_used": True,
            "monitor_llm_model": kwargs["model"],
            "fallbacks_used": [],
        }
        return updated

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)
    monkeypatch.setattr(webchat_server, "_apply_codes_monitor_decision", fake_apply_codes_monitor_decision)

    payload = webchat_server._agent_monitor_payload(
        {
            "config_path": [str(tmp_path / "sensors.yaml")],
            "axis": ["z"],
            "duration_s": ["0.5"],
            "require_current": ["1"],
        }
    )

    assert payload["ok"] is True
    assert payload["anomaly"] is True
    assert payload["popup"]["show_popup"] is True
    assert payload["popup"]["severity"] == "warning"
    assert payload["popup"]["affected_sensor_ids"] == ["target_1"]
    assert "Target 1 deviated" in payload["popup"]["message"]
    assert payload["agent"]["llm_agent_active"] is True
    assert captured["axis"] == "z"
    assert captured["duration_s"] == 0.5
    assert captured["require_current"] is True
    assert captured["use_model_agents"] is False
    assert llm_calls
    assert llm_calls[0]["model_timeout_s"] == 60.0


def test_agent_monitor_payload_registers_anomaly_window(monkeypatch, tmp_path):
    registry_dir = tmp_path / "anomaly_registry"
    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(registry_dir))

    def fake_pipeline(**kwargs):
        return {
            "schema_version": "0.4.0",
            "analysis_domain": "building_vibration_relative_monitoring",
            "window_id": "window-register-1",
            "network_assessment": {
                "network_label": "localized_vibration_deviation",
                "main_agent_confidence": 0.82,
                "affected_sensor_ids": ["target_1"],
                "significant_sensor_ids": ["target_1"],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "Target 1 deviated from the reference sensor.",
                "model_metadata": {"model_used": False},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_1",
                    "small_agent_label": "significant_local_deviation",
                    "model_metadata": {"model_used": False},
                }
            ],
            "model_metadata": {"main_model_used": False, "small_model_used": False, "fallbacks_used": []},
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

    registration = payload["anomaly_registration"]
    assert registration["registered"] is True
    assert registration["window_id"] == "window-register-1"
    jsonl_path = registry_dir / "anomaly_windows.jsonl"
    detail_path = registry_dir / "windows" / "window-register-1.json"
    assert jsonl_path.exists()
    assert detail_path.exists()
    record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert record["network_label"] == "localized_vibration_deviation"
    assert record["message_source"] == "server"
    assert record["affected_sensor_ids"] == ["target_1"]
    assert detail["pipeline_result"]["window_id"] == "window-register-1"

    listing = getattr(webchat_server, "_anomaly_windows_payload")({"limit": ["10"]})
    assert listing["ok"] is True
    assert listing["legacy_hidden_count"] == 0
    assert listing["windows"][0]["window_id"] == "window-register-1"


def test_agent_monitor_payload_does_not_register_normal_window(monkeypatch, tmp_path):
    registry_dir = tmp_path / "anomaly_registry"
    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(registry_dir))

    def fake_pipeline(**kwargs):
        return {
            "window_id": "window-normal-register",
            "network_assessment": {
                "network_label": "normal_relative_to_reference_sensor",
                "main_agent_confidence": 0.91,
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "All targets track the reference sensor.",
                "model_metadata": {"model_used": False},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_1",
                    "small_agent_label": "normal_relative_to_baseline",
                    "model_metadata": {"model_used": False},
                }
            ],
            "model_metadata": {"main_model_used": False, "small_model_used": False, "fallbacks_used": []},
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

    assert payload["anomaly"] is False
    assert payload["anomaly_registration"] == {"registered": False, "reason": "not_anomaly"}
    assert not (registry_dir / "anomaly_windows.jsonl").exists()


def test_agent_monitor_payload_does_not_popup_for_normal_network(monkeypatch):
    def fake_pipeline(**kwargs):
        return {
            "window_id": "window-2",
            "network_assessment": {
                "network_label": "normal_relative_to_reference_sensor",
                "main_agent_confidence": 0.91,
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "All targets track the reference sensor.",
                "model_metadata": {"model_used": True},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_1",
                    "small_agent_label": "normal_relative_to_baseline",
                    "model_metadata": {"model_used": True},
                }
            ],
            "model_metadata": {"main_model_used": True, "small_model_used": True, "fallbacks_used": []},
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

    assert payload["ok"] is True
    assert payload["anomaly"] is False
    assert payload["popup"]["show_popup"] is False


def test_agent_monitor_payload_keeps_mild_deviations_quiet(monkeypatch):
    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")

    cases = [
        ("mild_local_deviation", ["target_1"]),
        ("mild_multi_sensor_deviation", ["target_1", "target_2"]),
    ]

    for label, sensor_ids in cases:
        def fake_pipeline(**kwargs):
            return {
                "window_id": f"window-{label}",
                "network_assessment": {
                    "network_label": label,
                    "main_agent_confidence": 0.64,
                    "affected_sensor_ids": sensor_ids,
                    "significant_sensor_ids": [],
                    "mild_sensor_ids": sensor_ids,
                    "quality_flags": [],
                    "explanation": "Small vibration drift remained below the alert threshold.",
                    "model_metadata": {"model_used": False},
                },
                "sensor_agent_reports": [
                    {
                        "sensor_id": sensor_id,
                        "small_agent_label": "mild_deviation",
                        "model_metadata": {"model_used": False},
                    }
                    for sensor_id in sensor_ids
                ],
                "model_metadata": {"main_model_used": False, "small_model_used": False, "fallbacks_used": []},
            }

        monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

        payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

        assert payload["ok"] is True
        assert payload["anomaly"] is False
        assert payload["popup"]["show_popup"] is False
        assert payload["popup"]["severity"] == "notice"
        assert payload["popup"]["network_label"] == label
        assert payload["popup"]["mild_sensor_ids"] == sensor_ids
        assert payload["anomaly_registration"] == {"registered": False, "reason": "not_anomaly"}


def test_agent_monitor_payload_requires_llm_by_default(monkeypatch):
    webchat_server._AGENT_MONITOR_CACHE.clear()

    def fake_pipeline(**kwargs):
        return {
            "network_assessment": {
                "network_label": "normal_relative_to_reference_sensor",
                "main_agent_confidence": 0.91,
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "Fallback labels only.",
                "model_metadata": {"model_used": False, "fallback_reason": "endpoint_not_configured"},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_1",
                    "small_agent_label": "normal_relative_to_baseline",
                    "model_metadata": {"model_used": False, "fallback_reason": "endpoint_not_configured"},
                }
            ],
            "model_metadata": {
                "main_model_used": False,
                "small_model_used": False,
                "fallbacks_used": [
                    "small_sensor_agent:endpoint_not_configured",
                    "main_building_agent:endpoint_not_configured",
                ],
            },
        }

    def fake_apply_codes_monitor_decision(result, **kwargs):
        updated = dict(result)
        updated["model_metadata"] = {
            **dict(updated.get("model_metadata") or {}),
            "monitor_llm_model_used": False,
            "monitor_llm_fallback_reason": "endpoint_not_configured",
            "fallbacks_used": ["qwen_autonomous_monitor:endpoint_not_configured"],
        }
        return updated

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)
    monkeypatch.setattr(webchat_server, "_apply_codes_monitor_decision", fake_apply_codes_monitor_decision)

    payload = webchat_server._agent_monitor_payload({})

    assert payload["ok"] is False
    assert payload["error"] == "llm_agent_unavailable"
    assert payload["agent"]["llm_agent_active"] is False
    assert payload["agent"]["fallbacks_used"]


def test_agent_monitor_payload_can_skip_llm_and_still_popup(monkeypatch):
    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")

    def fake_pipeline(**kwargs):
        return {
            "network_assessment": {
                "network_label": "localized_vibration_deviation",
                "main_agent_confidence": 0.86,
                "affected_sensor_ids": ["target_2"],
                "significant_sensor_ids": ["target_2"],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "Target 2 deviated from the reference sensor.",
                "model_metadata": {"model_used": False, "fallback_reason": "endpoint_not_configured"},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_2",
                    "small_agent_label": "significant_local_deviation",
                    "model_metadata": {"model_used": False, "fallback_reason": "endpoint_not_configured"},
                }
            ],
            "model_metadata": {
                "main_model_used": False,
                "small_model_used": False,
                "fallbacks_used": [
                    "small_sensor_agent:endpoint_not_configured",
                    "main_building_agent:endpoint_not_configured",
                ],
            },
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

    assert payload["ok"] is True
    assert payload["anomaly"] is True
    assert payload["popup"]["show_popup"] is True
    assert payload["popup"]["affected_sensor_ids"] == ["target_2"]
    assert payload["agent"]["llm_agent_active"] is False
    assert payload["result"]["model_metadata"]["monitor_llm_skipped"] is True


def _history_rows(sensor_values, label="normal_relative_to_reference_sensor", q=0):
    return [
        {"ts": f"2026-07-06T00:00:{i:02d}", "label": label, "q": q, "sensors": {sid: dict(vals) for sid, vals in row.items()}}
        for i, row in enumerate(sensor_values)
    ]


def _fresh_history(monkeypatch, rows):
    from collections import deque

    monkeypatch.setattr(webchat_server, "_WINDOW_HISTORY", deque(rows, maxlen=20000))
    monkeypatch.setattr(webchat_server, "_HISTORY_STATS_CACHE", {})


def test_history_scoring_percentile_and_x_median(monkeypatch):
    monkeypatch.setenv("VIBRO_HISTORY_MIN_SAMPLES", "100")
    # 200 normal windows: target_1 rms hovers around 1.0 (0.90 … 1.10).
    rows = _history_rows(
        [{"target_1": {"rms": round(0.90 + (i % 100) * 0.002, 4), "peak": 1.0, "ppv": 1.0, "spec": 0.1}} for i in range(200)]
    )
    _fresh_history(monkeypatch, rows)

    result = {
        "network_assessment": {"quality_flags": []},
        "sensor_agent_reports": [
            {
                "sensor_id": "target_1",
                "baseline_comparison": {"rms_ratio": 2.0, "peak_ratio": 1.0, "ppv_ratio": 1.0, "spectral_distance": 0.1},
            },
            {
                "sensor_id": "target_9",  # no history at all -> learning
                "baseline_comparison": {"rms_ratio": 1.0, "peak_ratio": 1.0, "ppv_ratio": 1.0, "spectral_distance": 0.1},
            },
        ],
    }
    webchat_server._attach_window_history_context(result)

    context = result["sensor_agent_reports"][0]["history_context"]
    assert context["metric"] == "rms"
    assert context["percentile"] == 100.0
    assert 1.9 <= context["x_median"] <= 2.1  # median ~1.0
    assert context["n"] == 200
    assert result["sensor_agent_reports"][1]["history_context"]["learning"] is True
    summary = result["history_summary"]
    assert summary["sensor"] == "target_1"
    assert summary["percentile"] == 100.0


def test_history_learning_below_min_samples(monkeypatch):
    monkeypatch.setenv("VIBRO_HISTORY_MIN_SAMPLES", "100")
    rows = _history_rows([{"target_1": {"rms": 1.0, "peak": 1.0, "ppv": 1.0, "spec": 0.1}} for _ in range(30)])
    _fresh_history(monkeypatch, rows)

    result = {
        "network_assessment": {"quality_flags": []},
        "sensor_agent_reports": [
            {"sensor_id": "target_1", "baseline_comparison": {"rms_ratio": 5.0, "peak_ratio": 1.0, "ppv_ratio": 1.0, "spectral_distance": 0.1}}
        ],
    }
    webchat_server._attach_window_history_context(result)

    assert result["sensor_agent_reports"][0]["history_context"] == {"n": 30, "learning": True}
    assert result["history_summary"] == {"n": 30, "learning": True}


def test_history_baseline_excludes_anomalous_and_quality_windows(monkeypatch):
    monkeypatch.setenv("VIBRO_HISTORY_MIN_SAMPLES", "5")
    normal = _history_rows([{"target_1": {"rms": 1.0, "peak": 1.0, "ppv": 1.0, "spec": 0.1}} for _ in range(10)])
    anomalous = _history_rows(
        [{"target_1": {"rms": 9.0, "peak": 9.0, "ppv": 9.0, "spec": 3.0}} for _ in range(50)],
        label="multi_sensor_structure_wide_vibration_event",
    )
    flagged = _history_rows([{"target_1": {"rms": 7.0, "peak": 7.0, "ppv": 7.0, "spec": 2.0}} for _ in range(50)], q=1)
    _fresh_history(monkeypatch, normal + anomalous + flagged)

    values = webchat_server._history_values_locked("target_1", "rms")
    assert len(values) == 10  # only the normal, unflagged windows count
    assert max(values) == 1.0


def test_record_window_history_appends_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("VIBRO_WINDOW_HISTORY_DIR", str(tmp_path))
    monkeypatch.setattr(webchat_server, "_WINDOW_HISTORY", None)
    monkeypatch.setattr(webchat_server, "_HISTORY_STATS_CACHE", {})

    result = {
        "network_assessment": {"quality_flags": []},
        "sensor_agent_reports": [
            {
                "sensor_id": "target_1",
                "metrics": {"acceleration_rms_g": 0.02},
                "baseline_comparison": {"rms_ratio": 1.25, "peak_ratio": 1.1, "ppv_ratio": 0.9, "spectral_distance": 0.2},
                "quality_flags": [],
            }
        ],
    }
    popup = {"network_label": "normal_relative_to_reference_sensor"}
    webchat_server._record_window_history(result, popup)
    webchat_server._record_window_history(result, popup)

    lines = (tmp_path / "window_metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["label"] == "normal_relative_to_reference_sensor"
    assert row["q"] == 0
    assert row["sensors"]["target_1"]["rms"] == 1.25
    assert row["ref_rms"] == 0.016  # 0.02 / 1.25


def test_popup_message_includes_history_context():
    result = {
        "network_assessment": {
            "network_label": "localized_vibration_deviation",
            "main_agent_confidence": 0.85,
            "affected_sensor_ids": ["target_3"],
            "quality_flags": [],
            "status_text": "codes_v3: localized vibration deviation",
            "explanation": "Codes model verdict: one sensor stands out.",
        },
        "sensor_agent_reports": [],
        "history_summary": {"sensor": "target_3", "metric": "rms", "percentile": 99.94, "x_median": 2.4, "n": 4321},
    }
    popup = webchat_server._build_agent_popup(result)
    assert popup["history_x_median"] == 2.4
    assert popup["history_sensor"] == "target_3"
    assert popup["status_text"] == "localized vibration deviation"
    assert "Monitoring assessment: one sensor stands out." in popup["message"]
    assert "codes" not in json.dumps(popup).lower()
    assert "Versus its own normal history: target_3 rms at 2.4x its normal median" in popup["message"]
    assert "p99.9 of 4321 learned windows" in popup["message"]


def test_popup_uses_model_authored_message_verbatim():
    message = (
        "Localized vibration deviation detected at target_2 and target_4. "
        "Impulsive or shock event likely occurred. Normal duration observed. "
        "Immediate next step: verify sensor integrity under controlled loading."
    )
    result = {
        "network_assessment": {
            "network_label": "localized_vibration_deviation",
            "affected_sensor_ids": ["target_2", "target_4"],
            "status_text": "localized vibration deviation",
            "popup_message": message,
            "popup_message_source": "model",
        },
        "sensor_agent_reports": [],
        "history_summary": {"sensor": "target_4", "x_median": 2.8, "n": 50},
    }
    popup = webchat_server._build_agent_popup(result)
    assert popup["message"] == message
    assert popup["message_source"] == "model"
    assert popup["status_text"] == "localized vibration deviation"
    assert "codes" not in json.dumps(popup).lower()
    assert "Versus its own normal history" not in popup["message"]


def test_agent_monitor_payload_rejects_invalid_axis(monkeypatch):
    called = False

    def fake_pipeline(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("pipeline should not be called")

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"axis": ["vertical"]})

    assert payload["ok"] is False
    assert payload["error"] == "bad_request"
    assert called is False



def test_live_vibrometer_payload_uses_board_process_when_requested(monkeypatch):
    captured = {}

    def fail_direct(**_kwargs):
        raise AssertionError("direct SDK reader should not be used when process_reader=1")

    def fake_process_reader(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
            timestamps_s=np.asarray([0.0, 0.01, 0.02, 0.03], dtype=float),
            sampling_rate_hz=100,
            metadata={"data_is_current": True, "data_file_age_s": 0.01},
        )

    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fail_direct)
    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window_via_board_process", fake_process_reader)

    payload = webchat_server._live_vibrometer_payload(
        {
            "machine_id": ["target_1"],
            "acquisition_folder": ["/tmp/live_target_1"],
            "sensor_name": ["iis3dwb_acc"],
            "process_reader": ["1"],
        }
    )

    assert payload["ok"] is True
    assert payload["diagnostics"]["board_reader_process"] is True
    assert captured["machine_id"] == "target_1"
    assert captured["acquisition_folder"] == "/tmp/live_target_1"


def test_live_sensors_payload_can_prewarm_board_reader_processes(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "live_baseline"
    target_dir = tmp_path / "live_target_1"
    baseline_dir.mkdir()
    target_dir.mkdir()
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "location": "reference",
                        "role": "baseline",
                        "acquisition_folder": str(baseline_dir),
                    },
                    {
                        "sensor_id": "target_1",
                        "location": "target",
                        "role": "target",
                        "acquisition_folder": str(target_dir),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_prewarm(sensors):
        captured["sensor_ids"] = [sensor["sensor_id"] for sensor in sensors]
        return [{"sensor_id": sensor["sensor_id"], "ok": True, "pid": 1234} for sensor in sensors]

    monkeypatch.setattr(webchat_server, "prewarm_board_reader_processes", fake_prewarm)

    payload = webchat_server._live_sensors_payload(
        {"config_path": [str(config)], "process_reader": ["1"], "prewarm": ["1"]}
    )

    assert payload["ok"] is True
    assert payload["board_reader_process_enabled"] is True
    assert captured["sensor_ids"] == ["baseline", "target_1"]
    assert [item["sensor_id"] for item in payload["board_reader_processes"]] == ["baseline", "target_1"]


def test_agent_monitor_payload_passes_process_reader_flag(monkeypatch, tmp_path):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "window_id": "window-process-reader",
            "network_assessment": {
                "network_label": "normal_relative_to_reference_sensor",
                "main_agent_confidence": 0.9,
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "All targets track the reference sensor.",
                "model_metadata": {"model_used": False},
            },
            "sensor_agent_reports": [],
            "model_metadata": {"main_model_used": False, "small_model_used": False, "fallbacks_used": []},
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload(
        {
            "config_path": [str(tmp_path / "sensors.yaml")],
            "process_reader": ["1"],
            "require_llm": ["0"],
            "use_llm": ["0"],
        }
    )

    assert payload["ok"] is True
    assert captured["process_reader"] is True
    assert captured["use_model_agents"] is False


def test_graph_page_agent_script_has_labelize_helper():
    graph_script = webchat_server._GRAPH_HTML

    assert 'const labelize = (value)' in graph_script
    assert 'const status = labelize(popup.network_label || "checked")' in graph_script


def test_graph_page_agent_monitor_requires_codes_v3_with_server_timeout():
    graph_script = webchat_server._GRAPH_HTML

    assert 'require_llm: "1"' in graph_script
    assert 'use_llm: "1"' in graph_script
    assert "model_timeout_s:" not in graph_script
    assert 'skip_cache: "1"' in graph_script


def test_agent_monitor_payload_popups_for_network_quality_issue(monkeypatch):
    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")

    def fake_pipeline(**kwargs):
        return {
            "network_assessment": {
                "network_label": "sensor_network_quality_issue",
                "main_agent_confidence": 0.9,
                "affected_sensor_ids": ["target_5"],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": ["missing_data", "insufficient_samples"],
                "explanation": "Target 5 did not provide a usable current window.",
                "model_metadata": {"model_used": False, "fallback_reason": "disabled_for_fast_monitor"},
            },
            "sensor_agent_reports": [
                {
                    "sensor_id": "target_5",
                    "small_agent_label": "insufficient_data",
                    "quality_flags": ["missing_data", "insufficient_samples"],
                    "model_metadata": {"model_used": False, "fallback_reason": "disabled_for_fast_monitor"},
                }
            ],
            "model_metadata": {
                "main_model_used": False,
                "small_model_used": False,
                "fallbacks_used": ["small_sensor_agent:disabled_for_fast_monitor", "main_building_agent:disabled_for_fast_monitor"],
            },
        }

    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)

    payload = webchat_server._agent_monitor_payload({"require_llm": ["0"], "use_llm": ["0"]})

    assert payload["ok"] is True
    assert payload["anomaly"] is True
    assert payload["popup"]["show_popup"] is True
    assert payload["popup"]["severity"] == "quality"
    assert payload["popup"]["affected_sensor_ids"] == ["target_5"]


def test_direct_npu_messages_are_trimmed_by_character_budget():
    messages = [
        {"role": "system", "content": "system"},
        *[
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index}-" + ("x" * 450)}
            for index in range(20)
        ],
    ]

    normalized = webchat_server._normalize_direct_chat_messages(
        {"messages": messages},
        max_prompt_chars=1800,
    )

    assert normalized[-1]["content"].startswith("turn-19-")
    assert sum(len(item["content"]) + 32 for item in normalized) <= 1800
    assert len(normalized) < len(messages)


def test_webchat_html_exposes_welch_mode_and_bounded_chat_request():
    html = webchat_server._HTML
    diagnostics = html[html.index('<aside class="panel diagnostics"') : html.index("</aside>")]

    assert '<details id="connectionDetails" class="connection-details">' in html
    assert 'id="connectionSummary"' in html
    assert '<details class="technical-details">' in diagnostics
    assert 'id="modelActivity"' in diagnostics
    assert 'setModelActivity("ok", "Model-generated explanation")' in html
    assert 'data.model_request_sent' in html
    assert 'data.model_response_used' in html
    assert 'data.model_explanation_only' in html
    assert 'Model timeout · no fallback' in html
    assert 'No model answer · no fallback' in html
    assert 'Model fallback shown' not in html
    assert 'Model request used deterministic fallback' not in html
    assert 'Reading sensors and waiting for model explanation' in html
    assert 'id="psdEstimator"' in html
    assert 'id="psdSensor"' in html
    assert 'loadPsdSensorOptions()' in html
    assert 'params.set("sensor_id", selectedSensor.sensor_id)' in html
    assert '<option value="welch" selected>Welch</option>' in html
    assert '<option value="periodogram">Raw FFT</option>' in html
    assert 'id="psdScale"' in html
    assert '<option value="linear" selected>Linear</option>' in html
    assert 'id="psdWindow"' in html
    assert '<option value="hann" selected>Periodic Hann</option>' in html
    assert 'id="waveformCanvas"' in html
    assert 'id="psdFloatingPanel"' in html
    assert 'class="psd-floating"' in html
    assert 'id="psdOpenButton"' in diagnostics
    assert 'id="waveformCanvas"' not in diagnostics
    assert 'id="psdCanvas"' not in diagnostics
    assert 'id="psdFilter"' in html
    assert '<option value="highpass">High-pass</option>' in html
    assert '<option value="bandpass">Band-pass</option>' in html
    assert '<option value="notch_50">50 Hz notch</option>' in html
    assert '<option value="notch_60">60 Hz notch</option>' in html
    assert 'appendFilterParameters(params)' in html
    assert 'initializePsdWindowDrag()' in html
    assert 'window: els.psdWindow.value' in html
    assert 'drawWaveformChart(data)' in html
    assert 'duration_s: "100"' in html
    assert 'PSD variance is the area under the mean-removed PSD curve in g²' in html
    assert 'addPill("PSD variance"' in html
    assert 'addPill("Time variance"' in html
    assert 'addPill("Impact ring-down"' in html
    assert 'impact_spectrum_frequency_hz' in html
    assert 'impact ring-down peak' in html
    assert 'addPill("Variance check"' in html
    assert 'id="rmsMetricLabel"' in html
    assert 'Primary peak is outside this frequency view' in html
    assert 'id="psdDuration"' not in html
    assert 'psdCanvas.addEventListener("mousemove"' not in html
    assert 'max_prompt_chars: 9000' in html
    assert 'model_tool_routing: false' in html
    assert 'toolSourceLabel(call.source)' in html
    assert 'automatic building route' in html


def test_webchat_templates_use_consistent_st_shell_and_collapsed_secondary_controls():
    main = webchat_server._HTML
    graph = webchat_server._GRAPH_HTML
    npu = webchat_server._NPU_CHAT_HTML

    for html in (main, graph, npu):
        assert 'class="st-logo" src="/assets/st-logo.png"' in html
        assert 'class="app-nav"' in html
        assert 'ST-inspired' in html
        assert 'linear-gradient' not in html
        assert '#e2007a' not in html
        assert '#7a2a8f' not in html

    assert '<details class="psd-advanced">' in main
    assert '<summary><span>Advanced spectrum settings</span>' in main
    assert 'class="psd-primary-controls"' in main
    assert 'class="feature-canvas"' in main
    assert '.feature-canvas[hidden] { display: none !important; }' in main

    assert 'class="graph-toolbar"' in graph
    assert '<details class="toolbar-more">' in graph
    assert graph.count('<details class="side-details') == 2

    assert 'Direct NPU console' in npu
    assert 'href="/">Analysis chat</a>' in npu
    assert 'href="/graph">Live data</a>' in npu


def test_direct_npu_chat_enforces_building_prompt_and_output_guard(monkeypatch):
    blocked_term = sorted(OUT_OF_DOMAIN_COMPONENT_TERMS)[0]
    monkeypatch.setattr(
        webchat_server,
        "_post_openai_chat_completion",
        lambda **_kwargs: {
            "choices": [
                {
                    "message": {"content": f"Unsupported component diagnosis containing {blocked_term}."},
                    "finish_reason": "stop",
                }
            ]
        },
    )

    with pytest.raises(webchat_server.ModelExplanationError, match="No fallback answer"):
        webchat_server._direct_npu_chat_payload(
            {
                "message": "Explain the current vibration.",
                "base_url": "http://127.0.0.1:8910/v1",
                "model": "test-model",
                "timeout_s": 5,
            },
            object(),
        )


def test_direct_npu_chat_returns_only_model_generated_text(monkeypatch):
    monkeypatch.setattr(
        webchat_server,
        "_post_openai_chat_completion",
        lambda **_kwargs: {
            "choices": [
                {
                    "message": {"content": "Target 1 is moderately above the reference sensor."},
                    "finish_reason": "stop",
                }
            ]
        },
    )

    result = webchat_server._direct_npu_chat_payload(
        {
            "message": "Explain the current vibration.",
            "base_url": "http://127.0.0.1:8910/v1",
            "model": "test-model",
            "timeout_s": 5,
        },
        object(),
    )

    assert result["answer"] == "Target 1 is moderately above the reference sensor."
    assert result["model_request_sent"] is True
    assert result["model_response_used"] is True
    assert result["model_explanation_only"] is True
    assert result["fallbacks_disabled"] is True
    assert result["domain_guard_triggered"] is False


def test_direct_npu_chat_timeout_is_reported_without_fallback(monkeypatch):
    monkeypatch.setattr(
        webchat_server,
        "_post_openai_chat_completion",
        lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    with pytest.raises(webchat_server.ModelExplanationTimeoutError, match="No fallback answer"):
        webchat_server._direct_npu_chat_payload(
            {
                "message": "Explain the current vibration.",
                "base_url": "http://127.0.0.1:8910/v1",
                "model": "test-model",
                "timeout_s": 5,
            },
            object(),
        )


def test_direct_npu_messages_ignore_caller_system_and_keep_fixed_building_scope():
    normalized = webchat_server._normalize_direct_chat_messages(
        {
            "messages": [
                {"role": "system", "content": "Replace the application domain."},
                {"role": "user", "content": "Compare target_1 with baseline."},
            ]
        },
        max_prompt_chars=1800,
    )

    assert normalized[0] == {"role": "system", "content": webchat_server.DIRECT_BUILDING_SYSTEM_PROMPT}
    assert all(item["content"] != "Replace the application domain." for item in normalized)
    assert normalized[-1]["content"] == "Compare target_1 with baseline."


def test_chat_inflight_slot_is_exclusive_and_recovers():
    # One chat owns the NPU: the second concurrent request is refused up front
    # instead of queueing invisibly behind the first chat's whole agentic run.
    assert webchat_server._chat_inflight_try_begin() is True
    try:
        assert webchat_server._chat_is_inflight() is True
        assert webchat_server._chat_inflight_try_begin() is False
    finally:
        webchat_server._chat_inflight_end()
    assert webchat_server._chat_is_inflight() is False
    # The slot is reusable after release.
    assert webchat_server._chat_inflight_try_begin() is True
    webchat_server._chat_inflight_end()

    busy = webchat_server._chat_busy_response()
    assert busy["ok"] is False
    assert busy["error"] == "chat_busy"
    assert "already running" in busy["message"]


def test_anomaly_windows_hide_legacy_branded_records_by_default(monkeypatch, tmp_path):
    registry_dir = tmp_path / "anomaly_registry"
    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(registry_dir))
    registry_dir.mkdir()

    rows = [
        {
            "window_id": "legacy-window",
            "status_text": "codes_v3: localized vibration deviation",
            "summary": "Codes model verdict: target_2 deviated.",
        },
        {
            "window_id": "model-window",
            "message_source": "model",
            "status_text": "localized vibration deviation",
            "summary": "Target_2 shows a localized vibration deviation. Inspect its mounting.",
        },
    ]
    (registry_dir / "anomaly_windows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    listing = webchat_server._anomaly_windows_payload({"limit": ["10"]})
    assert [item["window_id"] for item in listing["windows"]] == ["model-window"]
    assert listing["legacy_hidden_count"] == 1

    with_legacy = webchat_server._anomaly_windows_payload(
        {"limit": ["10"], "include_legacy": ["1"]}
    )
    assert [item["window_id"] for item in with_legacy["windows"]] == [
        "model-window",
        "legacy-window",
    ]
    assert with_legacy["legacy_hidden_count"] == 0
