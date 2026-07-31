from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vibroagent_mcp import webchat_server
from vibroagent_mcp.offline_replay import (
    ClaimedInference,
    OfflineReplayConfigurationError,
    OfflineReplayController,
    ScheduledInference,
)
from vibroagent_mcp.sensor_registry import SensorRegistry
from vibroagent_mcp.sdk_vibrometer import SdkVibrometerWindow


class ManualClock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_event_windows_are_deduplicated_and_claimed_once_per_cycle(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "recordings_manifest.json",
        {
            "seconds": 300.0,
            "events": [
                {"slot": "live_target_2", "kind": "impulse", "start_s": 45.0, "dur_s": 10.0},
                {"slot": "live_target_4", "kind": "tone", "start_s": 90.0, "dur_s": 10.0},
                {"slot": "live_target_3", "kind": "impulse", "start_s": 210.0, "dur_s": 10.0},
                {"slot": "live_target_5", "kind": "tone", "start_s": 210.0, "dur_s": 10.0},
            ],
        },
    )
    clock = ManualClock(100.0)
    replay = OfflineReplayController(manifest, monotonic=clock)

    assert [item.start_time_s for item in replay.schedule] == [45.0, 90.0, 210.0]
    assert replay.schedule[-1].labels == (
        "live_target_3:impulse",
        "live_target_5:tone",
    )

    clock.value = 144.9
    assert replay.claim_due_inference() is None

    clock.value = 145.0
    first = replay.claim_due_inference()
    assert first is not None
    assert first.cycle == 0
    assert first.window.start_time_s == 45.0
    assert replay.claim_due_inference() is None

    replay.release_claim(first)
    assert replay.claim_due_inference() == first

    clock.value = 190.0
    second = replay.claim_due_inference()
    assert second is not None
    assert second.window.start_time_s == 90.0

    clock.value = 310.0
    third = replay.claim_due_inference()
    assert third is not None
    assert third.window.start_time_s == 210.0

    clock.value = 445.0
    next_cycle = replay.claim_due_inference()
    assert next_cycle is not None
    assert next_cycle.cycle == 1
    assert next_cycle.window.start_time_s == 45.0


def test_explicit_inference_timestamps_take_precedence_and_graph_window_is_bounded(
    tmp_path: Path,
) -> None:
    manifest = write_manifest(
        tmp_path / "recordings_manifest.json",
        {
            "seconds": 30.0,
            "inference_window_s": 5.0,
            "inference_timestamps_s": [
                2.0,
                {"start_time_s": 20.0, "duration_s": 5.0, "label": "injected_tone"},
            ],
            "events": [{"start_s": 10.0, "dur_s": 5.0}],
        },
    )
    clock = ManualClock()
    replay = OfflineReplayController(manifest, speed=2.0, monotonic=clock)

    clock.value = 14.0
    snapshot = replay.snapshot()
    assert snapshot["position_s"] == pytest.approx(28.0)
    assert snapshot["next_inference_time_s"] == pytest.approx(2.0)
    assert snapshot["next_inference_cycle"] == 1
    assert snapshot["next_inference_in_s"] == pytest.approx(2.0)
    assert replay.window_start(5.0) == pytest.approx(25.0)
    assert [item.start_time_s for item in replay.schedule] == [2.0, 20.0]

    assert replay.claim_due_inference().window.start_time_s == 2.0
    assert replay.claim_due_inference().window.start_time_s == 20.0
    assert replay.claim_due_inference() is None


def test_replay_clock_never_modifies_recording_bytes(tmp_path: Path) -> None:
    recording = tmp_path / "live_baseline" / "iis3dwb_acc.dat"
    recording.parent.mkdir()
    original = b"immutable recording bytes"
    recording.write_bytes(original)
    manifest = write_manifest(
        tmp_path / "recordings_manifest.json",
        {"seconds": 60.0, "inference_timestamps_s": [10.0]},
    )
    clock = ManualClock()
    replay = OfflineReplayController(manifest, source_root=tmp_path, monotonic=clock)
    before = recording.stat()

    clock.value = 10.0
    replay.snapshot()
    replay.window_start(5.0)
    replay.claim_due_inference()

    after = recording.stat()
    assert recording.read_bytes() == original
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns


def test_schedule_rejects_a_window_past_recording_end(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "recordings_manifest.json",
        {
            "seconds": 30.0,
            "inference_windows": [{"start_time_s": 25.0, "duration_s": 10.0}],
        },
    )
    with pytest.raises(OfflineReplayConfigurationError, match="exceeds"):
        OfflineReplayController(manifest)


def test_sensor_registry_offline_root_maps_live_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    offline_root = tmp_path / "recordings"
    acquisition = offline_root / "live_baseline"
    acquisition.mkdir(parents=True)
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "role": "baseline",
                        "acquisition_folder": "/old/stdatalog_examples/live_baseline",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIBRO_ACQUISITION_ROOT", str(offline_root))

    registry = SensorRegistry(config)

    assert registry.baseline().acquisition_folder == str(acquisition.resolve())

def test_graph_reads_the_virtual_position_as_a_fixed_immutable_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeReplay:
        def snapshot(self):
            return {"enabled": True, "immutable": True, "position_s": 123.5}

        def window_start(self, duration_s):
            assert duration_s == 0.5
            return 123.5

    def fake_read(**kwargs):
        captured.update(kwargs)
        return SdkVibrometerWindow(
            signal=np.asarray([0.1, 0.2, 0.3, 0.4]),
            timestamps_s=np.asarray([123.5, 123.51, 123.52, 123.53]),
            sampling_rate_hz=100,
            metadata={"data_is_current": False},
        )

    monkeypatch.setattr(webchat_server, "offline_replay_from_environment", lambda: FakeReplay())
    monkeypatch.setattr(webchat_server, "read_sdk_vibrometer_window", fake_read)

    payload = webchat_server._live_vibrometer_payload(
        {
            "machine_id": ["target_1"],
            "acquisition_folder": ["/recordings/live_target_1"],
            "duration_s": ["0.5"],
            "require_current": ["1"],
        }
    )

    assert payload["ok"] is True
    assert captured["start_time_s"] == 123.5
    assert captured["require_current"] is False
    assert payload["replay"]["immutable"] is True


def test_scheduled_anomaly_uses_same_fixed_window_for_pipeline_codec_and_codes_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_call: dict = {}
    codec_call: dict = {}
    claim = ClaimedInference(
        cycle=0,
        schedule_index=0,
        window=ScheduledInference(
            start_time_s=45.0,
            duration_s=10.0,
            labels=("live_target_2:impulse",),
        ),
    )

    class FakeReplay:
        def snapshot(self):
            return {
                "enabled": True,
                "immutable": True,
                "position_s": 46.0,
                "next_inference_time_s": 90.0,
            }

        def claim_due_inference(self):
            return claim

        def release_claim(self, released):
            raise AssertionError(f"successful inference unexpectedly released {released}")

    def fake_pipeline(**kwargs):
        pipeline_call.update(kwargs)
        return {
            "window_id": "offline-45",
            "network_assessment": {
                "network_label": "normal_relative_to_reference_sensor",
                "affected_sensor_ids": [],
                "significant_sensor_ids": [],
                "mild_sensor_ids": [],
                "quality_flags": [],
                "explanation": "All targets track the reference sensor.",
            },
            "sensor_agent_reports": [],
            "model_metadata": {},
        }

    def fake_codes(result, **kwargs):
        codec_call.update(kwargs)
        result["model_metadata"] = {
            "monitor_llm_model_used": True,
            "monitor_llm_model": kwargs["model"],
            "fallbacks_used": [],
        }
        return result

    monkeypatch.setenv("VIBRO_REGISTER_ANOMALY_WINDOWS", "0")
    monkeypatch.setattr(webchat_server, "offline_replay_from_environment", lambda: FakeReplay())
    monkeypatch.setattr(webchat_server, "_chat_is_inflight", lambda: False)
    monkeypatch.setattr(webchat_server, "run_building_vibration_agent_pipeline_service", fake_pipeline)
    monkeypatch.setattr(webchat_server, "_apply_codes_monitor_decision", fake_codes)

    payload = webchat_server._agent_monitor_payload({"axis": ["z"], "skip_cache": ["1"]})

    assert payload["ok"] is True
    assert pipeline_call["start_time_s"] == 45.0
    assert pipeline_call["duration_s"] == 10.0
    assert pipeline_call["require_current"] is False
    assert codec_call["start_time_s"] == 45.0
    assert codec_call["require_current"] is False
    assert payload["replay"]["inference"]["start_time_s"] == 45.0
    assert payload["agent"]["llm_agent_active"] is True
