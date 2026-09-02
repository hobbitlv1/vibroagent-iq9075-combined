from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vibroagent_mcp import vibrogemma_webchat, webchat_server
from vibroagent_mcp.offline_replay import (
    ClaimedInference,
    OfflineReplayController,
    ScheduledInference,
)
from vibroagent_mcp.sdk_vibrometer import SdkVibrometerWindow
from vibroagent_mcp.sensor_registry import SensorRegistry
from vibroagent_mcp.vibrogemma_live import _capture_board


class ManualClock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_manifest_claims_lumo_window_at_exact_start_without_modifying_dat(tmp_path: Path) -> None:
    recording = tmp_path / "live_target_3" / "iis3dwb_acc.dat"
    recording.parent.mkdir()
    recording.write_bytes(b"immutable")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "duration_s": 300,
                "inference_windows": [
                    {"start_time_s": 30, "duration_s": 10, "label": "lumo:target_3"}
                ],
            }
        ),
        encoding="utf-8",
    )
    clock = ManualClock()
    replay = OfflineReplayController(manifest, source_root=tmp_path, monotonic=clock)
    before = recording.stat()

    clock.value = 29.999
    assert replay.claim_due_inference() is None
    clock.value = 30.0
    claim = replay.claim_due_inference()

    assert claim is not None
    assert claim.window == ScheduledInference(30.0, 10.0, ("lumo:target_3",))
    assert recording.read_bytes() == b"immutable"
    assert recording.stat().st_mtime_ns == before.st_mtime_ns


def test_registry_and_graph_use_offline_root_and_virtual_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_root = tmp_path / "recordings"
    folder = replay_root / "live_baseline"
    folder.mkdir(parents=True)
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "role": "baseline",
                        "acquisition_folder": "/old/live_baseline",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIBRO_ACQUISITION_ROOT", str(replay_root))
    assert SensorRegistry(config).baseline().acquisition_folder == str(folder.resolve())

    captured: dict = {}

    class FakeReplay:
        def snapshot(self):
            return {"enabled": True, "immutable": True, "position_s": 30.0}

        def window_start(self, duration_s):
            assert duration_s == 10.0
            return 30.0

    def reader(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=np.ones(1000),
            timestamps_s=np.linspace(30.0, 40.0, 1000, endpoint=False),
            sampling_rate_hz=100,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "offline_replay_from_environment", lambda: FakeReplay())
    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", reader)
    payload = webchat_server._live_vibrometer_payload(
        {"duration_s": ["10"], "acquisition_folder": [str(folder)]}
    )

    assert payload["ok"] is True
    assert captured["start_time_s"] == 30.0
    assert captured["require_current"] is False
    assert payload["replay"]["immutable"] is True


def test_fixed_gemma_capture_reads_exact_ten_second_dat_window() -> None:
    calls: list[dict] = []
    sensor = type("Sensor", (), {"acquisition_folder": "/recordings/live_target_3", "hsd_sensor_name": "iis3dwb_acc"})()

    def reader(**kwargs):
        calls.append(kwargs)
        return SdkVibrometerWindow(
            signal=np.arange(1000, dtype=np.float64),
            timestamps_s=np.linspace(30.0, 40.0, 1000, endpoint=False),
            sampling_rate_hz=100,
            metadata={"data_is_current": False},
        )

    captured = _capture_board(
        "target_3",
        sensor,
        reader,
        start_time_s=30.0,
        require_current=False,
    )

    assert len(calls) == 3
    assert all(call["start_time_s"] == 30.0 for call in calls)
    assert all(call["duration_s"] == 10.25 for call in calls)
    assert all(call["require_current"] is False for call in calls)
    assert captured.values.shape == (3, 1000)


def test_monitor_passes_claim_timestamp_and_lumo_label_to_gemma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_call: dict = {}
    claim = ClaimedInference(
        cycle=0,
        schedule_index=0,
        window=ScheduledInference(30.0, 10.0, ("lumo:target_3",)),
    )

    class FakeReplay:
        def snapshot(self):
            return {"enabled": True, "immutable": True, "position_s": 30.0}

        def claim_due_inference(self):
            return claim

        def release_claim(self, _claim):
            raise AssertionError("successful replay inference was released")

    def pipeline(**kwargs):
        assert kwargs["start_time_s"] == 30.0
        assert kwargs["duration_s"] == 10.0
        assert kwargs["require_current"] is False
        return {
            "window_id": "offline-30",
            "network_assessment": {
                "network_label": "normal",
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
            },
            "sensor_agent_reports": [],
            "model_metadata": {},
        }

    def decision(result, **kwargs):
        decision_call.update(kwargs)
        result["model_metadata"] = {"monitor_llm_model_used": True, "fallbacks_used": []}
        return result

    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")
    monkeypatch.setattr(webchat_server, "offline_replay_from_environment", lambda: FakeReplay())
    monkeypatch.setattr(webchat_server, "_chat_is_inflight", lambda: False)
    monkeypatch.setattr(webchat_server, "_auto_select_agent_model", lambda **_kwargs: "vibrogemma")
    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", pipeline)
    monkeypatch.setattr(webchat_server, "_apply_monitor_llm_decision", decision)

    payload = webchat_server._agent_monitor_payload({"skip_cache": ["1"]})

    assert payload["ok"] is True
    assert decision_call["start_time_s"] == 30.0
    assert decision_call["require_current"] is False
    assert decision_call["replay_labels"] == ("lumo:target_3",)
    assert payload["replay"]["inference"]["start_time_s"] == 30.0


def test_offline_waveform_status_tracks_scheduled_lumo_event(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReplay:
        schedule = (ScheduledInference(30.0, 10.0, ("lumo:target_5",)),)

        def snapshot(self):
            return {"enabled": True, "position_s": 35.0}

    monkeypatch.setattr(vibrogemma_webchat, "offline_replay_from_environment", lambda: FakeReplay())

    status = vibrogemma_webchat._synthetic_injection_status()

    assert status["scheduled"] is True
    assert status["enabled"] is True
    assert status["target"] == "target_5"


def test_monitor_thread_sleeps_to_exact_next_replay_event(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    class FakeReplay:
        schedule = ()

        def snapshot(self):
            return {"next_inference_in_s": 5.25}

    def stop(delay: float) -> None:
        delays.append(delay)
        raise StopIteration

    monkeypatch.setattr(vibrogemma_webchat, "_pin_current_thread", lambda *_args: None)
    monkeypatch.setattr(vibrogemma_webchat, "offline_replay_from_environment", lambda: FakeReplay())
    monkeypatch.setattr(vibrogemma_webchat.time, "sleep", stop)
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR", None)
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR_AT", 0.0)

    with pytest.raises(StopIteration):
        vibrogemma_webchat._monitor_loop(
            lambda _query: {
                "ok": True,
                "status": "ok",
                "replay": {"inference": {"start_time_s": 15.0}},
                "result": {"window": {"end_utc": "2000-01-01T00:00:00+00:00"}},
            }
        )

    assert delays == [5.25]
    assert vibrogemma_webchat.time.monotonic() - vibrogemma_webchat._LATEST_MONITOR_AT < 1.0


def test_scheduled_offline_alert_is_kept_for_replay() -> None:
    result = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "unknown_anomaly",
                "global_target_consistent": True,
                "localization_verified": True,
                "localization_status": "target",
                "capture": {
                    "synthetic_injection": {
                        "active": True,
                        "kind": "lumo_offline_replay",
                    }
                },
                "targets": [
                    {
                        "sensor_id": f"target_{index}",
                        "affected": index == 3,
                        "class": "unknown_anomaly" if index == 3 else "normal",
                        "severity": "advisory" if index == 3 else "none",
                    }
                    for index in range(1, 6)
                ],
            },
        },
        "network_assessment": {},
    }

    popup = vibrogemma_webchat._apply_popup_policy({"show_popup": True}, result)

    assert popup["show_popup"] is True
    assert popup["save_replay"] is True
    assert "synthetic_test" not in popup
