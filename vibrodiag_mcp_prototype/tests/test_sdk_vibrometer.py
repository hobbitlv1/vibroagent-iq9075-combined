# pyright: reportArgumentType=false, reportOperatorIssue=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np

from vibroagent_mcp import sdk_vibrometer
from vibroagent_mcp.sdk_vibrometer import (
    _frames_to_arrays,
    _normalize_axis,
    _normalize_window_request,
    _PersistentLiveBoardDecoder,
    _read_latest_sdk_arrays,
    _safe_float,
    _safe_int,
    _sampling_rate_from_sensor_status,
    _trim_latest_window,
    SdkVibrometerWindow,
    find_latest_hsd_acquisition_folder,
    list_sdk_vibrometer_sources,
    read_sdk_vibrometer_window,
    read_sdk_vibrometer_window_payload,
)

SDK_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ACQUISITION = SDK_ROOT / "stdatalog_examples" / "20260521_11_42_47"


def _has_stdatalog_sdk() -> bool:
    try:
        import stdatalog_core.HSD.HSDatalog  # noqa: F401
    except ImportError:
        return False
    return True


requires_example_acquisition = pytest.mark.skipif(
    not EXAMPLE_ACQUISITION.exists() or not _has_stdatalog_sdk(),
    reason="local STDATALOG example acquisition or SDK is not available",
)


class _FakeSeries:
    def __init__(self, values):
        self._values = values

    def to_numpy(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class _FakeFrame:
    def __init__(self, data):
        self._data = data
        self.columns = list(data)

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        if isinstance(key, list):
            return _FakeFrame({item: self._data[item] for item in key})
        return _FakeSeries(self._data[key])

    def to_numpy(self, dtype=None):
        return np.column_stack([np.asarray(self._data[column], dtype=dtype) for column in self.columns])


class _FakeHsd:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def get_dataframe(self, _hsd_instance, _component, **kwargs):
        self.calls.append(kwargs)
        return self.frames


class _FakeBoardHsd(_FakeHsd):
    def __init__(self, frames, status):
        super().__init__(frames)
        self.status = status

    def get_sensor(self, _hsd_instance, sensor):
        return {sensor: self.status}


class _PersistentCursorHsd:
    def __init__(self, status, start_cursor=0):
        self.status = dict(status)
        self.cursor = int(start_cursor)
        self.component_ids = []
        self.calls = []

    def get_sensor(self, _hsd_instance, sensor):
        return {sensor: self.status}

    def get_dataframe(self, _hsd_instance, component, **kwargs):
        # Simulate the SDK mutating its per-read component status while keeping
        # the fake data cursor in the HSD object. Real STDATALOG reads should be
        # independent bounded reads from a fresh component copy.
        status = component["iis3dwb_acc"]
        cursor = self.cursor
        self.component_ids.append(id(component))
        self.calls.append({"cursor": cursor, **kwargs})
        values = np.arange(cursor, cursor + 150, dtype=np.float64)
        status["prev_data_byte_counter"] = cursor
        status["last_index"] = cursor + 50
        self.cursor = cursor + 50
        return [
            _FakeFrame(
                {
                    "Time": values / 10.0,
                    "X": values,
                    "Y": values + 1.0,
                    "Z": values + 2.0,
                }
            )
        ]


class _FakeDataCorruptedException(RuntimeError):
    pass


class _FlakyPersistentCursorHsd(_PersistentCursorHsd):
    def __init__(self, status, fail_on_call):
        super().__init__(status)
        self.fail_on_call = int(fail_on_call)

    def get_dataframe(self, _hsd_instance, component, **kwargs):
        if len(self.calls) == self.fail_on_call:
            raise _FakeDataCorruptedException(
                "Error extracting data from /tmp/iis3dwb_acc.dat file. Missing packets revealed, data corrupted."
            )
        return super().get_dataframe(_hsd_instance, component, **kwargs)


class _NoDataFramePersistentCursorHsd(_PersistentCursorHsd):
    def __init__(self, status, none_on_call):
        super().__init__(status)
        self.none_on_call = int(none_on_call)

    def get_dataframe(self, _hsd_instance, component, **kwargs):
        if len(self.calls) == self.none_on_call:
            self.calls.append({"cursor": self.cursor, "returned_none": True, **kwargs})
            return None
        return super().get_dataframe(_hsd_instance, component, **kwargs)


def test_latest_trim_reconstructs_timestamps_when_sdk_timestamps_jump():
    samples = np.arange(10_000, dtype=np.float64).reshape(-1, 1)
    timestamps = np.concatenate(
        [
            np.arange(9_000, dtype=np.float64) / 1_000.0,
            100.0 + (np.arange(1_000, dtype=np.float64) / 1_000.0),
        ]
    )

    trimmed = _trim_latest_window(
        samples,
        timestamps,
        sampling_rate_hz=1_000,
        duration_s=5.0,
    )

    assert trimmed.samples_2d.shape == (5_000, 1)
    assert trimmed.timestamp_mode == "sample_count_tail_from_sdk"
    assert trimmed.timestamp_warning
    assert np.all(np.diff(trimmed.timestamps_s) > 0)
    assert trimmed.samples_2d[0, 0] == 5_000.0


def test_latest_trim_reconstructs_timestamps_when_sdk_timestamps_repeat_or_go_backwards():
    samples = np.arange(200, dtype=np.float64).reshape(-1, 1)
    timestamps = np.concatenate(
        [
            np.arange(100, dtype=np.float64) / 100.0,
            np.asarray([0.99, 0.98], dtype=np.float64),
            1.0 + (np.arange(98, dtype=np.float64) / 100.0),
        ]
    )

    trimmed = _trim_latest_window(
        samples,
        timestamps,
        sampling_rate_hz=100,
        duration_s=1.0,
    )

    assert trimmed.samples_2d.shape == (100, 1)
    assert trimmed.samples_2d[0, 0] == 100.0
    assert trimmed.timestamp_mode == "sample_count_tail_from_sdk"
    assert "non-monotonic" in trimmed.timestamp_warning
    assert np.all(np.diff(trimmed.timestamps_s) > 0)


def test_latest_sdk_read_accepts_sample_only_frames_and_reconstructs_time():
    hsd = _FakeHsd(
        [
            _FakeFrame(
                {
                    "X": np.arange(150, dtype=np.float64),
                    "Y": np.arange(150, dtype=np.float64) + 1.0,
                    "Z": np.arange(150, dtype=np.float64) + 2.0,
                }
            )
        ]
    )

    result = _read_latest_sdk_arrays(
        hsd=hsd,
        hsd_instance=object(),
        sensor_component={"iis3dwb_acc": {}},
        duration_s=1.0,
        duration_hint_s=None,
        chunk_size=150,
        minimum_sample_count=100,
    )
    trimmed = _trim_latest_window(
        result.samples_2d,
        result.timestamps_s,
        sampling_rate_hz=100,
        duration_s=1.0,
    )

    assert result.timestamps_s is None
    assert result.samples_2d.shape == (150, 3)
    assert result.latest_timestamp_gap_s is None
    assert hsd.calls[0]["start_time"] == 0.0
    assert trimmed.samples_2d.shape == (100, 3)
    assert trimmed.timestamp_mode == "sample_count_tail_from_sdk"
    assert trimmed.timestamp_warning
    assert np.all(np.diff(trimmed.timestamps_s) > 0)


def test_read_sdk_window_reconstructs_latest_sample_only_timestamps(tmp_path, monkeypatch):
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 1024)
    status = {
        "measodr": 100,
        "odr": 100,
        "dim": 3,
        "samples_per_ts": 50,
        "data_type": "float32",
    }
    hsd = _FakeBoardHsd(
        [
            _FakeFrame(
                {
                    "X": np.arange(150, dtype=np.float64),
                    "Y": np.arange(150, dtype=np.float64) + 1.0,
                    "Z": np.arange(150, dtype=np.float64) + 2.0,
                }
            )
        ],
        status,
    )

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (hsd, object()))

    window = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert window.signal.shape == (100,)
    assert window.signal[0] == 50.0
    assert window.timestamps_s is not None
    assert np.all(np.diff(window.timestamps_s) > 0)
    assert window.metadata["timestamp_mode"] == "sample_count_tail_from_sdk"
    assert window.metadata["timestamp_warning"]


def test_read_sdk_window_reconstructs_fixed_sample_only_timestamps(tmp_path, monkeypatch):
    folder = tmp_path / "fixed_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 1024)
    status = {
        "measodr": 100,
        "odr": 100,
        "dim": 3,
        "samples_per_ts": 50,
        "data_type": "float32",
    }
    hsd = _FakeBoardHsd(
        [
            _FakeFrame(
                {
                    "X": np.arange(150, dtype=np.float64),
                    "Y": np.arange(150, dtype=np.float64) + 1.0,
                    "Z": np.arange(150, dtype=np.float64) + 2.0,
                }
            )
        ],
        status,
    )

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (hsd, object()))

    window = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        start_time_s=2.0,
        duration_s=1.0,
        require_current=False,
    )

    assert window.signal.shape == (100,)
    assert window.signal[0] == 0.0
    assert window.timestamps_s is not None
    assert window.timestamps_s[0] == 2.0
    assert np.isclose(window.timestamps_s[-1], 2.99)
    assert window.metadata["timestamp_mode"] == "sample_count_fixed_window_from_sdk"
    assert window.metadata["timestamp_warning"]


def test_read_sdk_window_reconstructs_fixed_nonmonotonic_timestamps(tmp_path, monkeypatch):
    folder = tmp_path / "fixed_board_bad_time"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 1024)
    status = {
        "measodr": 100,
        "odr": 100,
        "dim": 3,
        "samples_per_ts": 50,
        "data_type": "float32",
    }
    timestamps = np.concatenate(
        [
            2.0 + (np.arange(60, dtype=np.float64) / 100.0),
            np.asarray([2.59, 2.58], dtype=np.float64),
            2.62 + (np.arange(88, dtype=np.float64) / 100.0),
        ]
    )
    hsd = _FakeBoardHsd(
        [
            _FakeFrame(
                {
                    "Time": timestamps,
                    "X": np.arange(150, dtype=np.float64),
                    "Y": np.arange(150, dtype=np.float64) + 1.0,
                    "Z": np.arange(150, dtype=np.float64) + 2.0,
                }
            )
        ],
        status,
    )

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (hsd, object()))

    window = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        start_time_s=2.0,
        duration_s=1.0,
        require_current=False,
    )

    assert window.signal.shape == (100,)
    assert window.signal[0] == 0.0
    assert window.timestamps_s[0] == 2.0
    assert np.isclose(window.timestamps_s[-1], 2.99)
    assert window.metadata["timestamp_mode"] == "sample_count_fixed_window_from_sdk"
    assert "non-monotonic" in window.metadata["timestamp_warning"]


def test_board_frame_conversion_filters_bad_rows_and_skips_corrupt_frames():
    good_frame = _FakeFrame(
        {
            "Time": [0.0, 0.001, 0.002, float("nan"), 0.004],
            "X": [1.0, float("nan"), 3.0, 4.0, 5.0],
            "Y": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Z": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    corrupt_frame = _FakeFrame(
        {
            "Time": [0.005],
            "X": ["not-a-number"],
            "Y": [1.0],
            "Z": [1.0],
        }
    )

    samples, timestamps, axis_columns = _frames_to_arrays([good_frame, corrupt_frame])

    assert axis_columns == ["X", "Y", "Z"]
    assert samples.tolist() == [
        [1.0, 1.5, 2.0],
        [3.0, 3.5, 4.0],
        [5.0, 5.5, 6.0],
    ]
    assert timestamps.tolist() == [0.0, 0.002, 0.004]


def test_board_frame_conversion_drops_timestamps_when_frame_alignment_breaks():
    mismatched_time_frame = _FakeFrame(
        {
            "Time": [0.0],
            "X": [1.0, 2.0],
            "Y": [1.0, 2.0],
            "Z": [1.0, 2.0],
        }
    )

    samples, timestamps, axis_columns = _frames_to_arrays([mismatched_time_frame])

    assert axis_columns == ["X", "Y", "Z"]
    assert samples.shape == (2, 3)
    assert timestamps is None


def test_board_frame_conversion_reorders_consistent_axis_columns():
    first = _FakeFrame(
        {
            "X": [1.0, 2.0],
            "Y": [10.0, 20.0],
            "Z": [100.0, 200.0],
            "Timestamp": [0.0, 0.01],
        }
    )
    reordered = _FakeFrame(
        {
            "Timestamp": [0.02, 0.03],
            "Z": [300.0, 400.0],
            "X": [3.0, 4.0],
            "Y": [30.0, 40.0],
        }
    )

    samples, timestamps, axis_columns = _frames_to_arrays([first, reordered])

    assert axis_columns == ["X", "Y", "Z"]
    assert samples.tolist() == [
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
    ]
    assert timestamps.tolist() == [0.0, 0.01, 0.02, 0.03]


def test_latest_live_reader_reuses_persistent_decoder_state_per_board(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    hsd = _PersistentCursorHsd(status)
    create_calls = []
    estimated_durations = iter([15.0, 15.0, 20.0, 20.0])

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (create_calls.append(str(_folder)) or hsd, object()))
    monkeypatch.setattr(sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: next(estimated_durations))
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    first = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )
    second = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert len(create_calls) == 1
    assert "prev_data_byte_counter" not in hsd.status
    assert "last_index" not in hsd.status
    assert second.metadata["sdk_read_strategy"] == "persistent_live_incremental"
    assert second.metadata["timestamp_end_s"] > first.metadata["timestamp_end_s"]
    sdk_vibrometer._clear_live_decoder_pool()


def test_latest_live_reader_normalizes_bounded_tail_gap(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    hsd = _PersistentCursorHsd(status)
    estimated_durations = iter([15.0, 15.0])

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (hsd, object()))
    monkeypatch.setattr(sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: next(estimated_durations))
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    window = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    # The seed catch-up anchors the duration estimate at the decoded tail, so
    # the 0.1s wobble between the scripted file-size estimate and the tail no
    # longer surfaces even as a raw gap.
    assert window.metadata["sdk_duration_estimate_anchored"] is True
    assert window.metadata["sdk_latest_timestamp_raw_gap_s"] == pytest.approx(0.0)
    assert window.metadata["sdk_latest_timestamp_gap_s"] == pytest.approx(0.0)
    sdk_vibrometer._clear_live_decoder_pool()


def test_latest_live_reader_reports_lag_even_after_gap_calibration(tmp_path):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.sensor_status = {"samples_per_ts": 100}
    decoder.configured_sampling_rate = 1_000
    decoder.latest_gap_calibration_s = 0.8

    # Packet/timestamp tolerance is 0.3 s for this synthetic board.  The 0.8 s
    # calibration anchor is the frontier-measured estimate offset, so the true
    # decode lag in a raw 1.2 s SDK gap is the 0.4 s residual; its excess over
    # tolerance (0.1 s) must be reported.  Reporting the raw gap instead would
    # dump the whole anchor into the value and (hours into an acquisition, when
    # the anchor holds tens of seconds of clock skew) trip the reset threshold
    # and the data-currentness limit on a fraction-of-a-second real hiccup.
    assert decoder._gap_tolerance_s_locked() == pytest.approx(0.3)
    assert decoder._normalized_latest_timestamp_gap_s_locked(1.2) == pytest.approx(0.1)


def test_gap_calibration_rise_is_rate_limited_so_lag_cannot_hide(tmp_path):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.sensor_status = {"samples_per_ts": 100}
    decoder.configured_sampling_rate = 1_000
    decoder.timestamps_s = np.asarray([0.0, 0.1], dtype=np.float64)
    decoder.samples_2d = np.zeros((2, 3), dtype=np.float64)
    decoder.last_sdk_read_strategy = "persistent_live_incremental"
    freshness = {"data_is_current": True}

    decoder.latest_gap_calibration_s = 0.5
    decoder.gap_calibration_updated_monotonic = 0.0

    # A progress read may absorb estimate wobble, but only at a clock-skew
    # plausible rate (one packet duration here, 0.1 s).  A raw gap that jumped
    # to 10 s because the decoder is genuinely behind must NOT be swallowed by
    # the anchor; the residual stays large enough to trip the reset threshold.
    decoder._maybe_update_gap_calibration_locked(raw_gap_s=10.0, freshness=freshness)
    assert decoder.latest_gap_calibration_s == pytest.approx(0.6)
    assert decoder._normalized_latest_timestamp_gap_s_locked(10.0) > decoder._gap_reset_threshold_s_locked()

    # Initial anchoring (fresh decoder at the seed frontier) is exempt: it may
    # take the full measured offset.
    decoder.latest_gap_calibration_s = 0.0
    decoder._maybe_update_gap_calibration_locked(raw_gap_s=2.0, freshness=freshness, initial=True)
    assert decoder.latest_gap_calibration_s == pytest.approx(2.0)


def test_latest_live_reader_resets_decoder_after_large_raw_gap(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    create_calls = []
    estimated_durations = iter([15.0, 25.0, 25.0, 20.0, 20.0])

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)

    def fake_create(_folder):
        create_calls.append(str(_folder))
        start_cursor = 0 if len(create_calls) == 1 else 50
        return _PersistentCursorHsd(status, start_cursor=start_cursor), object()

    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", fake_create)
    monkeypatch.setattr(sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: next(estimated_durations))
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )
    # ~10s of new frames land in the file while the decoder gets nothing new:
    # the anchored estimate must track the growth, expose the gap, and force a
    # reseed instead of quietly re-anchoring at the stalled tail.
    with (folder / "iis3dwb_acc.dat").open("ab") as handle:
        handle.write(b"\x00" * (20 * 68))
    second = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert len(create_calls) == 2
    assert second.metadata["sdk_read_strategy"].startswith("persistent_live_seed:")
    assert second.metadata["sdk_duration_estimate_anchored"] is True
    assert second.metadata["sdk_latest_timestamp_raw_gap_s"] == pytest.approx(0.0)
    assert second.metadata["sdk_latest_timestamp_gap_s"] == pytest.approx(0.0)
    sdk_vibrometer._clear_live_decoder_pool()


def test_estimate_data_duration_uses_measured_rate_not_rounded_odr(tmp_path):
    # The byte/frame math yields an exact decodable SAMPLE COUNT; converting it to
    # seconds must use the decoded-timestamp rate, otherwise a configured-vs-true
    # ODR mismatch integrates into a steadily growing (false) decode gap.
    folder = tmp_path / "live_board"
    folder.mkdir()
    status = {"data_type": "int16", "dim": 3, "samples_per_ts": 4}
    frame_size = 4 * 3 * 2 + 8  # samples_per_ts * dim * dtype_len + timestamp bytes
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * (100 * frame_size))  # 100 frames -> 400 samples

    nominal = sdk_vibrometer._estimate_data_duration_s(
        folder=folder,
        sensor="iis3dwb_acc",
        sensor_status=status,
        sampling_rate_hz=1_000,
        hsd_instance=object(),
    )
    measured = sdk_vibrometer._estimate_data_duration_s(
        folder=folder,
        sensor="iis3dwb_acc",
        sensor_status=status,
        sampling_rate_hz=1_000,
        hsd_instance=object(),
        effective_sampling_rate_hz=800.0,
    )

    assert nominal == pytest.approx(400 / 1_000)  # 0.4 s at the rounded configured ODR
    assert measured == pytest.approx(400 / 800)   # 0.5 s on the real decoded-timestamp clock


def test_effective_sample_rate_tracks_timestamp_drift_and_rejects_outliers(tmp_path):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.configured_sampling_rate = 1_000

    # Decoded timestamps drifting 1% slow vs the nominal ODR must be reflected, so
    # the duration estimate stays on the same clock as the timestamps (gap ~ 0).
    decoder.timestamps_s = np.arange(2_000, dtype=np.float64) / 990.0
    assert decoder._measured_sample_period_s_locked() == pytest.approx(1.0 / 990.0)
    assert decoder._effective_sample_rate_hz_locked() == pytest.approx(990.0)

    # A degenerate buffer (e.g. one spanning a discontinuity) implying a wildly
    # different rate is rejected in favour of the configured rate.
    decoder.timestamps_s = np.arange(100, dtype=np.float64) / 3_000.0
    assert decoder._effective_sample_rate_hz_locked() is None

    decoder.timestamps_s = np.empty(0, dtype=np.float64)
    assert decoder._effective_sample_rate_hz_locked() is None


def test_incremental_overlap_only_keeps_cursor_at_decoded_frontier(tmp_path):
    # When an incremental append returns only already-buffered rows, the cursor must
    # not advance past the decoded frontier. The read end tracks the file-size
    # duration estimate, which runs AHEAD of the decodable tail by the integrated
    # clock-skew error; jumping there parks the cursor in not-yet-written data, every
    # later append reads empty, and the gap check then forces a full reset+reseed on
    # every poll (decode time per request degrades from ~20ms to the full seed cost).
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.configured_sampling_rate = 1_000
    decoder.sensor_status = {"samples_per_ts": 100}  # packet 0.1s -> overlap 0.2s
    decoder.timestamps_s = 99.0 + np.arange(1_000, dtype=np.float64) / 1_000.0
    decoder.samples_2d = np.zeros((1_000, 3), dtype=np.float64)
    decoder.latest_timestamp_s = float(decoder.timestamps_s[-1])
    decoder.next_start_time_s = decoder.latest_timestamp_s - decoder._overlap_duration_s_locked()

    overlap_rows = decoder.timestamps_s[-50:]
    decoder._read_incremental_samples_locked = lambda **_kwargs: (  # type: ignore[method-assign]
        np.zeros((overlap_rows.size, 3), dtype=np.float64),
        overlap_rows.copy(),
        ["A_x [g]", "A_y [g]", "A_z [g]"],
    )

    # Duration estimate 0.5s ahead of the decodable tail -> read end = tail + 1.5s.
    decoder._append_new_data_locked(duration_s=1.0, duration_hint_s=decoder.latest_timestamp_s + 0.5)

    assert decoder.last_sdk_read_strategy == "persistent_live_incremental_overlap_only"
    assert decoder.next_start_time_s <= decoder.latest_timestamp_s


def test_incremental_read_end_uses_gap_calibration_to_stay_bounded(tmp_path):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.configured_sampling_rate = 1_000
    decoder.sensor_status = {"samples_per_ts": 100}
    decoder.timestamps_s = 99.0 + np.arange(1_000, dtype=np.float64) / 1_000.0
    decoder.samples_2d = np.zeros((1_000, 3), dtype=np.float64)
    decoder.latest_timestamp_s = float(decoder.timestamps_s[-1])
    decoder.next_start_time_s = decoder.latest_timestamp_s - decoder._overlap_duration_s_locked()
    decoder.latest_gap_calibration_s = 60.0
    captured = {}

    overlap_rows = decoder.timestamps_s[-50:]

    def fake_read_incremental(**kwargs):
        captured.update(kwargs)
        return (
            np.zeros((overlap_rows.size, 3), dtype=np.float64),
            overlap_rows.copy(),
            ["A_x [g]", "A_y [g]", "A_z [g]"],
        )

    decoder._read_incremental_samples_locked = fake_read_incremental  # type: ignore[method-assign]
    raw_duration_hint_s = decoder.latest_timestamp_s + decoder.latest_gap_calibration_s + 0.25

    decoder._append_new_data_locked(duration_s=1.0, duration_hint_s=raw_duration_hint_s)

    assert captured["end_time_s"] == pytest.approx(decoder.latest_timestamp_s + 1.25)
    assert raw_duration_hint_s + 1.0 - captured["end_time_s"] > 59.0


def test_latest_live_reader_keeps_previous_buffer_when_incremental_has_no_dataframe(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    hsd = _NoDataFramePersistentCursorHsd(status, none_on_call=1)

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", lambda _folder: (hsd, object()))
    monkeypatch.setattr(sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: 15.0)
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    first = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )
    second = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert second.metadata["sdk_read_strategy"] == "persistent_live_incremental_empty"
    assert second.metadata["timestamp_end_s"] == first.metadata["timestamp_end_s"]
    assert second.signal.tolist() == first.signal.tolist()
    sdk_vibrometer._clear_live_decoder_pool()


def test_latest_live_reader_recovers_from_incremental_data_corruption(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder = tmp_path / "live_board"
    folder.mkdir()
    (folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    create_calls = []
    estimated_durations = iter([15.0, 15.0, 20.0, 20.0, 20.0, 20.0])

    monkeypatch.setattr(sdk_vibrometer, "resolve_hsd_acquisition_folder", lambda *_args, **_kwargs: folder)

    def fake_create(_folder):
        create_calls.append(str(_folder))
        if len(create_calls) == 1:
            return _FlakyPersistentCursorHsd(status, fail_on_call=1), object()
        return _PersistentCursorHsd(status, start_cursor=50), object()

    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", fake_create)
    monkeypatch.setattr(sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: next(estimated_durations))
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )
    second = read_sdk_vibrometer_window(
        acquisition_folder=str(folder),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert len(create_calls) == 2
    assert second.metadata["sdk_read_strategy"].startswith("persistent_live_recovered:persistent_live_seed:")
    assert second.metadata["sdk_read_attempts"][0]["strategy"] == "persistent_live_incremental_recovery"
    assert "data corrupted" in second.metadata["sdk_read_attempts"][0]["error"]
    sdk_vibrometer._clear_live_decoder_pool()


def test_latest_live_reader_keeps_decoders_isolated_between_boards(tmp_path, monkeypatch):
    sdk_vibrometer._clear_live_decoder_pool()
    folder_a = tmp_path / "live_a"
    folder_b = tmp_path / "live_b"
    folder_a.mkdir()
    folder_b.mkdir()
    (folder_a / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    (folder_b / "iis3dwb_acc.dat").write_bytes(b"\x00" * 4096)
    status_a = {
        "measodr": 10,
        "odr": 10,
        "dim": 3,
        "samples_per_ts": 5,
        "data_type": "float32",
        "fs": 16.0,
        "sensitivity": 0.5,
    }
    status_b = dict(status_a)
    hsd_a = _PersistentCursorHsd(status_a)
    hsd_b = _PersistentCursorHsd(status_b)
    create_calls = []
    duration_by_folder = {
        str(folder_a): 15.0,
        str(folder_b): 12.0,
    }

    monkeypatch.setattr(
        sdk_vibrometer,
        "resolve_hsd_acquisition_folder",
        lambda acquisition_folder=None, **_kwargs: Path(acquisition_folder),
    )

    def fake_create(folder):
        create_calls.append(str(folder))
        if Path(folder) == folder_a:
            return hsd_a, object()
        return hsd_b, object()

    monkeypatch.setattr(sdk_vibrometer, "_create_hsd", fake_create)
    monkeypatch.setattr(
        sdk_vibrometer,
        "_estimate_data_duration_s",
        lambda *, folder, **_kwargs: duration_by_folder[str(folder)],
    )
    monkeypatch.setattr(sdk_vibrometer, "_acquisition_duration_s", lambda _instance: None)
    monkeypatch.setattr(sdk_vibrometer, "_live_decoder_min_refresh_interval_s", lambda: 0.0)

    first = read_sdk_vibrometer_window(
        acquisition_folder=str(folder_a),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )
    second = read_sdk_vibrometer_window(
        acquisition_folder=str(folder_b),
        sensor_name="iis3dwb_acc",
        axis="x",
        duration_s=1.0,
        require_current=False,
    )

    assert create_calls == [str(folder_a), str(folder_b)]
    assert "prev_data_byte_counter" not in hsd_a.status
    assert "prev_data_byte_counter" not in hsd_b.status
    assert first.metadata["acquisition_folder"] == str(folder_a)
    assert second.metadata["acquisition_folder"] == str(folder_b)
    sdk_vibrometer._clear_live_decoder_pool()


def test_board_frame_conversion_skips_incompatible_axis_layouts():
    first = _FakeFrame(
        {
            "Time": [0.0, 0.01],
            "X": [1.0, 2.0],
            "Y": [10.0, 20.0],
            "Z": [100.0, 200.0],
        }
    )
    incompatible = _FakeFrame(
        {
            "Time": [0.02, 0.03],
            "X": [3.0, 4.0],
            "Y": [30.0, 40.0],
            "W": [300.0, 400.0],
        }
    )

    samples, timestamps, axis_columns = _frames_to_arrays([first, incompatible])

    assert axis_columns == ["X", "Y", "Z"]
    assert samples.tolist() == [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]
    assert timestamps.tolist() == [0.0, 0.01]


def test_board_reader_request_normalization_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="duration_s must be a finite number"):
        _normalize_window_request(start_time_s=None, duration_s=float("nan"))

    with pytest.raises(ValueError, match="start_time_s must be a finite number"):
        _normalize_window_request(start_time_s=float("inf"), duration_s=1.0)

    with pytest.raises(ValueError, match="duration_s is capped"):
        _normalize_window_request(start_time_s=None, duration_s=61.0)


def test_board_reader_axis_and_metadata_numeric_normalization():
    assert _normalize_axis(" Z ") == "z"
    assert _safe_float("nan") is None
    assert _safe_float("inf") is None
    assert _safe_int("1000.0") == 1000
    assert _sampling_rate_from_sensor_status({"measodr": "0.4", "odr": "1666.7"}) == 1667

    with pytest.raises(ValueError, match="axis must be one of"):
        _normalize_axis(None)


def test_read_sdk_payload_returns_structured_error_for_bad_board_request():
    result = read_sdk_vibrometer_window_payload(
        machine_id="board_01",
        axis="not-an-axis",
        duration_s=1.0,
        require_current=False,
    )

    assert result["status"] == "error"
    assert result["error"] == "sdk_read_failed"
    assert result["machine_id"] == "board_01"
    assert result["sample_count"] == 0
    assert "axis must be one of" in result["message"]


def test_live_metadata_repair_copy_failure_becomes_structured_error(tmp_path, monkeypatch):
    live_folder = tmp_path / "20260526_12_00_00"
    template_folder = tmp_path / "template"
    live_folder.mkdir()
    template_folder.mkdir()
    (live_folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 128)
    (template_folder / "device_config.json").write_text("{}", encoding="utf-8")
    (template_folder / "acquisition_info.json").write_text("{}", encoding="utf-8")

    def fail_copy(*_args, **_kwargs):
        raise OSError("copy blocked")

    monkeypatch.setenv("VIBRO_ALLOWED_HSD_DIR", str(tmp_path))
    monkeypatch.setattr(sdk_vibrometer.shutil, "copy2", fail_copy)

    result = read_sdk_vibrometer_window_payload(
        machine_id="board_01",
        acquisition_folder=str(live_folder),
        sensor_name="iis3dwb_acc",
        require_current=False,
    )

    assert result["status"] == "error"
    assert result["error"] == "sdk_read_failed"
    assert "Missing device_config.json" in result["message"]
    assert "metadata_repair_failed" in result["message"]
    assert "copy blocked" in result["message"]


def test_live_metadata_repair_write_failure_becomes_structured_error(tmp_path, monkeypatch):
    live_folder = tmp_path / "20260526_12_00_00"
    live_folder.mkdir()
    (live_folder / "iis3dwb_acc.dat").write_bytes(b"\x00" * 128)
    (live_folder / "device_config.json").write_text("{}", encoding="utf-8")

    def fail_write(*_args, **_kwargs):
        raise OSError("write blocked")

    monkeypatch.setenv("VIBRO_ALLOWED_HSD_DIR", str(tmp_path))
    monkeypatch.setattr(sdk_vibrometer, "_write_live_acquisition_info", fail_write)

    result = read_sdk_vibrometer_window_payload(
        machine_id="board_01",
        acquisition_folder=str(live_folder),
        sensor_name="iis3dwb_acc",
        require_current=False,
    )

    assert result["status"] == "error"
    assert result["error"] == "sdk_read_failed"
    assert "Missing acquisition_info.json" in result["message"]
    assert "metadata_repair_failed" in result["message"]
    assert "write blocked" in result["message"]


def test_find_latest_board_folder_skips_transient_stat_failures(tmp_path, monkeypatch):
    bad_folder = tmp_path / "bad_live"
    good_folder = tmp_path / "good_live"
    bad_folder.mkdir()
    good_folder.mkdir()
    bad_file = bad_folder / "iis3dwb_acc.dat"
    good_file = good_folder / "iis3dwb_acc.dat"
    bad_file.write_bytes(b"\x00")
    good_file.write_bytes(b"\x00")

    original_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == bad_file:
            raise OSError("file rotated during discovery")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setenv("VIBRO_ALLOWED_HSD_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "stat", flaky_stat)

    assert find_latest_hsd_acquisition_folder(sensor_name="iis3dwb_acc") == good_folder.resolve()


def test_read_sdk_payload_tolerates_bad_preview_and_return_limits(monkeypatch):
    def fake_read_window(**_kwargs):
        return SdkVibrometerWindow(
            signal=np.arange(40, dtype=np.float64),
            timestamps_s=None,
            sampling_rate_hz=100,
            metadata={"sample_count": 40, "data_is_current": True},
        )

    monkeypatch.setattr(sdk_vibrometer, "read_sdk_vibrometer_window", fake_read_window)

    result = read_sdk_vibrometer_window_payload(
        machine_id="board_01",
        include_samples=True,
        max_preview_samples="nan",
        max_return_samples="inf",
    )

    assert result["sample_count"] == 40
    assert len(result["preview_samples"]) == 16
    assert result["returned_sample_count"] == 40
    assert result["samples_truncated"] is False


@requires_example_acquisition
def test_list_sdk_vibrometer_sources_finds_iis3dwb(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOWED_HSD_DIR", raising=False)
    monkeypatch.delenv("VIBRO_ALLOWED_DATA_DIR", raising=False)
    result = list_sdk_vibrometer_sources(acquisition_folder=str(EXAMPLE_ACQUISITION))

    names = {sensor["name"] for sensor in result["sensors"]}
    assert "iis3dwb_acc" in names
    assert result["default_sensor_name"] == "iis3dwb_acc"


@requires_example_acquisition
def test_read_sdk_vibrometer_window_payload(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOWED_HSD_DIR", raising=False)
    monkeypatch.delenv("VIBRO_ALLOWED_DATA_DIR", raising=False)
    result = read_sdk_vibrometer_window_payload(
        machine_id="test_stwinbox",
        acquisition_folder=str(EXAMPLE_ACQUISITION),
        sensor_name="iis3dwb_acc",
        axis="norm",
        start_time_s=0.02,
        duration_s=0.05,
        max_preview_samples=4,
    )

    assert result["source"] == "stdatalog_sdk_hsd"
    assert result["metadata"]["sensor_name"] == "iis3dwb_acc"
    assert result["metadata"]["sample_count"] > 32
    assert result["sampling_rate_hz"] > 0
    assert len(result["preview_samples"]) == 4


def _decoder_with_buffer(tmp_path, *, true_rate_hz=26680.0, configured_rate=26637, tail_ts_s=10000.0, span_s=10.0):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.configured_sampling_rate = configured_rate
    count = int(round(span_s * true_rate_hz))
    decoder.timestamps_s = tail_ts_s - (np.arange(count, dtype=np.float64)[::-1] / true_rate_hz)
    decoder.samples_2d = np.zeros((count, 3), dtype=np.float64)
    return decoder


def test_anchored_duration_estimate_bounds_rate_error_by_poll_span(tmp_path, monkeypatch):
    # Board truly runs at 26680 Hz while the configured ODR says 26637 (0.16%
    # off). Anchored at the frontier 2s (of board time) ago, the estimate must
    # land within milliseconds — NOT configured-vs-true error times file age.
    decoder = _decoder_with_buffer(tmp_path)
    grown = int(round(2.0 * 26680.0))
    decoder.duration_anchor_ts_s = 10000.0
    decoder.duration_anchor_samples = 266_800_000
    monkeypatch.setattr(
        sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 266_800_000 + grown
    )
    # Prove precedence over the legacy full-file path.
    monkeypatch.setattr(
        sdk_vibrometer, "_estimate_data_duration_s", lambda **_kwargs: 99999.0
    )

    estimate = decoder._estimate_data_duration_s_locked()
    assert estimate is not None and abs(estimate - 10002.0) < 0.01

    raw_gap = decoder._raw_latest_timestamp_gap_s_locked(estimate)
    assert raw_gap is not None and raw_gap < 2.1  # tail is 2s behind anchor+growth


def test_initial_anchor_only_at_seed_catch_up_then_reanchor_gated_by_frontier(tmp_path, monkeypatch):
    decoder = _decoder_with_buffer(tmp_path)
    monkeypatch.setattr(sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 500_000)

    # The re-anchor hook never CREATES an anchor.
    decoder._maybe_update_duration_anchor_locked()
    assert decoder.duration_anchor_ts_s is None

    decoder._set_initial_duration_anchor_locked()
    assert decoder.duration_anchor_ts_s == pytest.approx(10000.0)
    assert decoder.duration_anchor_samples == 500_000

    # File grew ~10s beyond the tail: the decoder is behind. The anchor must
    # hold so the estimate keeps tracking the file and the gap stays honest.
    monkeypatch.setattr(
        sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 500_000 + 266_800
    )
    decoder._maybe_update_duration_anchor_locked()
    assert decoder.duration_anchor_samples == 500_000

    # Tail within a whisker of the frontier: re-anchor to keep the span short.
    monkeypatch.setattr(
        sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 500_000 + 1_000
    )
    decoder._maybe_update_duration_anchor_locked()
    assert decoder.duration_anchor_samples == 501_000
    assert decoder.duration_anchor_ts_s == pytest.approx(10000.0)

    # Rollover rebases the clock: never anchor while an offset is active.
    decoder.duration_anchor_ts_s = None
    decoder.duration_anchor_samples = None
    decoder.rollover_timestamp_offset_s = 3.0
    decoder._set_initial_duration_anchor_locked()
    assert decoder.duration_anchor_ts_s is None


def test_anchored_estimate_invalidates_on_shrink_and_defers_on_rollover(tmp_path, monkeypatch):
    decoder = _decoder_with_buffer(tmp_path)
    decoder.duration_anchor_ts_s = 10000.0
    decoder.duration_anchor_samples = 600_000

    monkeypatch.setattr(sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 400_000)
    assert decoder._anchored_data_duration_s_locked() is None
    assert decoder.duration_anchor_ts_s is None  # stale anchor cleared

    decoder.duration_anchor_ts_s = 10000.0
    decoder.duration_anchor_samples = 600_000
    decoder.rollover_timestamp_offset_s = 5.0
    monkeypatch.setattr(sdk_vibrometer, "_estimate_file_sample_count", lambda **_kwargs: 700_000)
    assert decoder._anchored_data_duration_s_locked() is None


def test_reset_clears_duration_anchor(tmp_path, monkeypatch):
    decoder = _decoder_with_buffer(tmp_path)
    decoder.duration_anchor_ts_s = 10000.0
    decoder.duration_anchor_samples = 600_000
    monkeypatch.setattr(decoder, "_reload_sdk_handles_locked", lambda: None)
    decoder._reset_locked()
    assert decoder.duration_anchor_ts_s is None and decoder.duration_anchor_samples is None


def test_seed_anchor_recalibrates_gap_on_anchored_clock(tmp_path, monkeypatch):
    decoder = _PersistentLiveBoardDecoder(tmp_path, "iis3dwb_acc")
    decoder.configured_sampling_rate = 10
    decoder.sensor_status = {"samples_per_ts": 5}
    decoder.timestamps_s = np.arange(100, dtype=np.float64) / 10.0
    decoder.samples_2d = np.zeros((100, 3), dtype=np.float64)
    decoder.latest_timestamp_s = float(decoder.timestamps_s[-1])
    decoder.last_sdk_read_strategy = "persistent_live_seed:test"
    decoder.last_sdk_read_attempts = []

    monkeypatch.setattr(
        sdk_vibrometer,
        "_estimate_file_sample_count",
        lambda **_kwargs: 1_000,
    )
    monkeypatch.setattr(
        decoder,
        "_estimate_data_duration_s_locked",
        lambda: (
            decoder.latest_timestamp_s
            if decoder.duration_anchor_ts_s is not None
            else decoder.latest_timestamp_s + 100.0
        ),
    )
    monkeypatch.setattr(
        decoder,
        "_append_new_data_locked",
        lambda **_kwargs: None,
    )

    decoder._catch_up_after_seed_locked(
        duration_s=1.0,
        duration_hint_s=decoder.latest_timestamp_s + 100.0,
    )

    assert decoder.duration_anchor_ts_s == pytest.approx(
        decoder.latest_timestamp_s
    )
    assert decoder.latest_gap_calibration_s == pytest.approx(0.0)
    anchored_hint = decoder._estimate_data_duration_s_locked()
    assert decoder._normalized_latest_timestamp_gap_s_locked(
        decoder._raw_latest_timestamp_gap_s_locked(anchored_hint)
    ) == pytest.approx(0.0)
