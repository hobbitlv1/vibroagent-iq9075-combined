import os
import time

import pytest

from vibroagent_mcp.vibrogemma_live import (
    _CapturedBoard,
    _capture_board,
    _capture_window_fingerprints,
    _device_identity,
    _finalize_capture,
    _lumo_training_fixture,
    _panel_network_label,
    _panel_sensor_label,
    _panel_system_label,
    _payload_sha256,
    _resample_board_window,
    _source_bindings,
    _truthy,
    _validate_model_result,
)


def test_live_board_clocks_are_aligned_to_reference_rate():
    import numpy as np

    source = np.tile(np.arange(12, dtype=np.float32), (3, 1))
    source[:, 6] = np.nan
    aligned = _resample_board_window(
        source,
        source_rate_hz=12.0,
        target_rate_hz=10.0,
        duration_s=1.0,
    )

    assert aligned.shape == (3, 10)
    assert np.isnan(aligned).any()
    assert aligned[:, -1].tolist() == [11.0, 11.0, 11.0]


def test_lumo_training_fixture_preserves_real_xy_and_missing_z(monkeypatch, tmp_path):
    import hashlib

    import numpy as np
    import scipy.io

    source = tmp_path / "lumo.mat"
    source.write_bytes(b"verified lumo fixture")
    names = [
        f"{sensor}{axis}"
        for sensor in ("accel04", "accel02", "accel05", "accel06", "accel08", "accel09")
        for axis in ("x", "y")
    ]
    values = np.arange(540 * len(names), dtype=np.float32).reshape(540, len(names))
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *_args, **_kwargs: {
            "Dat": {"ChannelNames": names, "Data": values, "Fs": 1.0}
        },
    )
    _lumo_training_fixture.cache_clear()
    arrays, rate, positions = _lumo_training_fixture(
        str(source), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    _lumo_training_fixture.cache_clear()

    assert rate == 1.0
    assert arrays["target_3"].shape == (3, 10)
    assert arrays["target_3"][0, 0] == values[520, names.index("accel06x")]
    assert not arrays["target_3"][2].any()
    assert positions["target_3"].tolist() == pytest.approx([0.0, 0.0, 4.0 / 8.95])


def test_lumo_compact_window_preserves_real_xy_and_missing_z(tmp_path):
    import hashlib

    import numpy as np

    sensors = ("accel04", "accel02", "accel05", "accel06", "accel08", "accel09")
    names = [f"{sensor}{axis}" for sensor in sensors for axis in ("x", "y")]
    values = np.arange(10 * len(names), dtype=np.float32).reshape(10, len(names))
    source = tmp_path / "lumo-window.npz"
    np.savez_compressed(
        source,
        data=values,
        channel_names=np.asarray(names),
        sample_rate_hz=np.asarray(1.0),
    )

    _lumo_training_fixture.cache_clear()
    arrays, rate, _positions = _lumo_training_fixture(
        str(source), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    _lumo_training_fixture.cache_clear()

    assert rate == 1.0
    assert arrays["target_5"][0, 0] == values[0, names.index("accel09x")]
    assert arrays["target_5"][1, -1] == values[-1, names.index("accel09y")]
    assert not arrays["target_5"][2].any()


def test_lumo_live_overlay_adds_template_without_replacing_live_capture(monkeypatch):
    import numpy as np

    from vibroagent_mcp import vibrogemma_live

    live = {
        slot: np.full((3, 10), index, dtype=np.float32)
        for index, slot in enumerate(vibrogemma_live.SLOTS)
    }
    templates = {
        slot: np.full((3, 10), index + 0.5, dtype=np.float32)
        for index, slot in enumerate(vibrogemma_live.SLOTS)
    }
    for template in templates.values():
        template[2] = 0.0
    monkeypatch.setattr(vibrogemma_live, "_lumo_overlay_gain", lambda: 1.0)
    monkeypatch.setattr(
        vibrogemma_live,
        "_lumo_training_replay",
        lambda _target: (
            {},
            dict.fromkeys(vibrogemma_live.SLOTS, 1.0),
            {},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        vibrogemma_live,
        "_lumo_overlay_templates",
        lambda *_args: (templates, {"kind": "lumo_live_overlay"}),
    )

    hybrid, metadata, rate = vibrogemma_live._apply_lumo_live_overlay(
        live,
        sample_rate_hz=1.0,
        target="target_5",
    )

    assert np.all(live["target_5"] == 5.0)
    assert np.all(hybrid["target_5"][:2] == 2.75)
    assert np.all(hybrid["target_5"][2] == 5.0)
    assert metadata["kind"] == "lumo_live_overlay"
    assert rate == 1.0


def test_lumo_overlay_template_matches_decimated_waveform_length(monkeypatch):
    import numpy as np

    from vibroagent_mcp import vibrogemma_live

    arrays = {
        slot: np.tile(np.arange(100, dtype=np.float32), (3, 1))
        for slot in vibrogemma_live.SLOTS
    }
    positions = {
        slot: np.asarray((0.0, 0.0, index / 5), dtype=np.float32)
        for index, slot in enumerate(vibrogemma_live.SLOTS)
    }
    masks = {
        slot: np.asarray((True, True, False))
        for slot in vibrogemma_live.SLOTS
    }
    monkeypatch.setattr(
        vibrogemma_live,
        "_lumo_training_replay",
        lambda _target: (
            arrays,
            dict.fromkeys(vibrogemma_live.SLOTS, 10.0),
            {},
            {
                "source_fixture": "fixture.mat@520-530s",
                "episode_provenance": {},
            },
            positions,
            masks,
        ),
    )
    vibrogemma_live._lumo_overlay_templates.cache_clear()
    templates, _metadata = vibrogemma_live._lumo_overlay_templates(
        "target_5", 71.7, 720, 1.0
    )
    vibrogemma_live._lumo_overlay_templates.cache_clear()

    assert all(template.shape == (3, 720) for template in templates.values())


def test_live_stdatalog_g_is_calibrated_by_default(monkeypatch):
    monkeypatch.delenv("VIBROGEMMA_AMPLITUDE_CALIBRATED", raising=False)
    assert _truthy("VIBROGEMMA_AMPLITUDE_CALIBRATED", default=True) is True
    monkeypatch.setenv("VIBROGEMMA_AMPLITUDE_CALIBRATED", "0")
    assert _truthy("VIBROGEMMA_AMPLITUDE_CALIBRATED", default=True) is False


def test_live_source_bindings_reject_one_folder_wired_to_two_slots(tmp_path):
    import json

    from vibroagent_mcp.sensor_registry import SensorRegistry

    folders = [tmp_path / f"board-{index}" for index in range(5)]
    for folder in folders:
        folder.mkdir()
    sensors = []
    slots = ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
    for index, slot in enumerate(slots):
        folder = folders[0] if slot == "target_1" else folders[min(index, 4)]
        sensors.append(
            {
                "sensor_id": slot,
                "role": "baseline" if slot == "baseline" else "target",
                "acquisition_folder": str(folder),
                "hsd_sensor_name": "iis3dwb_acc",
            }
        )
    config = tmp_path / "sensors.json"
    config.write_text(json.dumps({"baseline_sensor_id": "baseline", "sensors": sensors}))
    with pytest.raises(ValueError, match="duplicates"):
        _source_bindings(SensorRegistry(config))


def test_capture_fingerprints_reject_reused_board_window():
    import numpy as np

    slots = ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
    arrays = {
        slot: np.full((3, 16), index + 1, dtype=np.float32)
        for index, slot in enumerate(slots)
    }
    assert len(_capture_window_fingerprints(arrays)) == 6
    arrays["target_5"] = arrays["baseline"].copy()
    with pytest.raises(ValueError, match="byte-identical"):
        _capture_window_fingerprints(arrays)


def test_live_encoder_uses_configured_healthy_profile(monkeypatch, tmp_path):
    import json

    from scripts.live_vibrogemma_worker import _runtime_config

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "vibrogemma_config.json").write_text(
        json.dumps({"model": {"multiresolution": {"physics": {}}}})
    )
    profile = tmp_path / "healthy.json"
    profile.write_text("{}")
    monkeypatch.setenv("VIBROGEMMA_HEALTHY_PROFILE", str(profile))

    physics = _runtime_config(bundle)["model"]["multiresolution"]["physics"]
    assert physics["healthy_profile_path"] == str(profile.resolve())
    assert physics["require_healthy_profile"] is True

    monkeypatch.setenv("VIBROGEMMA_DISABLE_HEALTHY_PROFILE", "1")
    physics = _runtime_config(bundle)["model"]["multiresolution"]["physics"]
    assert physics["healthy_profile_path"] is None
    assert physics["require_healthy_profile"] is False


def test_gemma_cpu_set_parser():
    from vibroagent_mcp.vibrogemma_webchat import _cpu_set

    assert _cpu_set("0-2,4") == {0, 1, 2, 4}
    assert _cpu_set("bad") == set()


def test_board_readers_are_assigned_one_ui_core(monkeypatch):
    from vibroagent_mcp.board_reader_process import _reader_cpu_affinity

    monkeypatch.setenv("VIBRO_BOARD_READER_CPUS", "0-5")
    assert [_reader_cpu_affinity(index) for index in range(6)] == [
        {0},
        {1},
        {2},
        {3},
        {4},
        {5},
    ]


def test_gemma_webchat_enables_reduced_anomaly_popups(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat, webchat_server

    original_decision = webchat_server._apply_monitor_llm_decision
    original_popup = webchat_server._build_agent_popup
    original_monitor = webchat_server._agent_monitor_payload
    original_handler = webchat_server.WebchatHandler
    monkeypatch.setattr(webchat_server, "ANOMALOUS_NETWORK_LABELS", set())
    monkeypatch.setattr(webchat_server, "ANOMALOUS_SENSOR_LABELS", set())
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR", None)
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR_AT", 0.0)
    monkeypatch.setattr(vibrogemma_webchat, "_MONITOR_QUERY", {})
    try:
        vibrogemma_webchat._install()
        assert "unknown_anomaly" in webchat_server.ANOMALOUS_NETWORK_LABELS
        assert {"unknown_anomaly", "data_invalid"} <= webchat_server.ANOMALOUS_SENSOR_LABELS
        assert webchat_server.WebchatHandler is vibrogemma_webchat.VibroGemmaHandler
        popup = webchat_server._build_agent_popup(
            {
                "network_assessment": {
                    "network_label": "unknown_anomaly",
                    "affected_sensor_ids": ["target_1", "target_2"],
                    "mild_sensor_ids": ["target_1", "target_2"],
                },
                "sensor_agent_reports": [],
            }
        )
        assert popup["show_popup"] is True
        assert popup["severity"] == "notice"
        assert webchat_server._agent_monitor_payload({})["warming"] is True
        vibrogemma_webchat._LATEST_MONITOR = {"ok": True, "status": "ok", "result": {}}
        vibrogemma_webchat._LATEST_MONITOR_AT = time.monotonic()
        assert webchat_server._agent_monitor_payload({})["cached"] is True
        assert vibrogemma_webchat._MONITOR_QUERY == {}
    finally:
        webchat_server._apply_monitor_llm_decision = original_decision
        webchat_server._build_agent_popup = original_popup
        webchat_server._agent_monitor_payload = original_monitor
        webchat_server.WebchatHandler = original_handler


def test_gemma_ui_suppresses_legacy_fallback_popup():
    from vibroagent_mcp import vibrogemma_webchat

    popup = {"show_popup": True, "network_label": "multi_sensor_structure_wide_vibration_event"}
    result = {"model_metadata": {"agent_mode": "vibrogemma_live", "monitor_llm_model_used": False}}
    output = vibrogemma_webchat._apply_popup_policy(popup, result)
    assert output["show_popup"] is False
    assert output["network_label"] == "gemma_unavailable"


def test_gemma_static_ui_has_no_legacy_model_controls():
    from pathlib import Path
    from vibroagent_mcp import vibrogemma_webchat

    static_dir = Path(vibrogemma_webchat.__file__).parent / "static" / "vibrogemma"
    html = (static_dir / "index.html").read_text()
    javascript = (static_dir / "vibroagent.js").read_text()
    assert "VibroAgent" in html
    assert "Ask Agent" in html
    assert "Spectrum" in html
    assert "Replay" in html
    assert '<option value="10" selected>10 s</option>' in html
    assert "phosphor-icons.svg#ph-pause" in html
    assert "alertTargets" in html
    assert "alertMessage" not in html + javascript
    assert 'label: "Not localized"' not in javascript
    assert "classLabel(target.class)" in javascript
    assert "Normal not confirmed" in javascript
    assert "Global and target verdicts disagree" in javascript
    assert "No anomaly detected" in javascript
    assert "All systems operating normally" not in javascript
    assert "Shared vertical scale" in javascript
    assert "Research/replay-only checkpoint" in javascript
    assert "Anomaly candidate" in javascript
    assert "physical localization unavailable" in javascript
    assert 'id="toggleSynthetic"' in html
    assert 'id="toggleSyntheticTarget5"' in html
    assert "Run Target 3 anomaly test" in html
    assert "Stop Target 3 anomaly test" in javascript
    assert "Run Target 5 anomaly test" in html
    assert "Stop Target 5 anomaly test" in javascript
    assert "/api/vibro/synthetic-injection" in javascript
    assert "syntheticTargetLabel" not in javascript
    assert "Synthetic test missed" not in javascript
    assert "no localized anomaly confirmed" in javascript
    assert "(geometryUnavailable || inconclusive) && affected" in javascript
    assert "No alert raised; target localization unavailable" not in javascript
    assert 'item.state === "inconclusive"' in javascript
    assert "setInterval(() => loadReplays(true), 10000)" in javascript
    assert "Structural peak (0-200 Hz)" in html
    assert "legacy_wideband_only" in javascript
    assert "not be read as five independent faults" in javascript
    assert "Gemma" not in html
    assert "Gemma " not in javascript
    assert (static_dir / "phosphor-icons.svg").is_file()
    assert not set("—–") & set(html + javascript)
    assert all(term not in html for term in ("Qwen", "NPU console", "Model connection", "Raw response"))


def test_building_dominant_frequency_ignores_wideband_sensor_noise():
    import numpy as np

    from vibroagent_mcp.features import compute_fft_features

    sampling_rate_hz = 2000
    time_s = np.arange(sampling_rate_hz, dtype=np.float64) / sampling_rate_hz
    signal = np.sin(2 * np.pi * 12 * time_s) + 4 * np.sin(2 * np.pi * 800 * time_s)

    assert compute_fft_features(signal, sampling_rate_hz)["dominant_frequency_hz"] == pytest.approx(12.0)


def test_old_replay_does_not_present_wideband_peak_as_structural(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat

    detail = {
        "window_id": "old",
        "created_at_utc": "2026-08-20T00:00:00Z",
        "pipeline_result": {
            "baseline_sensor_id": "baseline",
            "all_sensor_metrics": [{"sensor_id": "baseline", "dominant_frequency_hz": 13_200.0}],
            "main_agent_explanation": "VibroGemma verdict. Gemma marked all five targets.",
            "model_metadata": {},
        },
    }
    monkeypatch.setattr(vibrogemma_webchat, "_load_replay_detail", lambda _window_id: detail)
    monkeypatch.setattr(vibrogemma_webchat, "_replay_state", lambda _detail: "normal")
    monkeypatch.setattr(vibrogemma_webchat, "_replay_targets", lambda _detail: [])

    payload = vibrogemma_webchat._gemma_replay_payload({"window_id": ["old"]})
    board = payload["boards"][0]
    assert board["dominant_frequency_hz"] is None
    assert board["wideband_peak_frequency_hz"] == 13_200.0
    assert board["dominant_frequency_status"] == "legacy_wideband_only"
    assert payload["explanation"] == "Agent verdict. Agent marked all five targets."


def test_gemma_chat_fails_closed_without_model_context():
    from vibroagent_mcp.vibrogemma_webchat import _gemma_chat_payload

    result = _gemma_chat_payload({"message": "What changed?", "context": {"available": False}})
    assert "unavailable" in result["answer"]


def test_gemma_popup_appears_on_first_generated_anomaly(monkeypatch, tmp_path):
    import json

    from vibroagent_mcp import vibrogemma_webchat, webchat_server

    anomaly = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {"targets": [{"sensor_id": "target_1", "affected": True, "class": "unknown_anomaly"}]},
        }
    }
    normal = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {"targets": [{"sensor_id": "target_1", "affected": False, "class": "normal"}]},
        }
    }
    popup = {"show_popup": True, "message": "Unknown anomaly", "network_label": "unknown_anomaly"}
    activated = vibrogemma_webchat._apply_popup_policy(popup, anomaly)
    assert activated["show_popup"] is True
    assert activated["affected_sensor_ids"] == ["target_1"]
    assert activated["localization"] == "geometry_unavailable"
    assert activated["persistence"]["trigger"] == "current_non_normal"
    assert vibrogemma_webchat._apply_popup_policy(popup, normal)["show_popup"] is False

    invalid = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {"targets": [{"sensor_id": "target_1", "affected": False, "class": "data_invalid"}]},
        }
    }
    quality = vibrogemma_webchat._apply_popup_policy(popup, invalid)
    assert quality["show_popup"] is True
    assert quality["severity"] == "quality"

    shared_result = {
        "window_id": "inconclusive-window",
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "normal",
                "global_target_consistent": False,
                "localization_status": "partial_target_set",
                "targets": [
                    {"sensor_id": f"target_{index}", "affected": True, "class": "unknown_anomaly"}
                    for index in range(1, 6)
                ],
            },
        },
    }
    shared = vibrogemma_webchat._apply_popup_policy(popup, shared_result)
    assert shared["show_popup"] is True
    assert shared["save_replay"] is True
    assert shared["network_label"] == "inconclusive"
    assert shared["localization"] == "inconclusive"
    assert shared["title"] == "Agent check inconclusive"
    assert shared["message"] == "Conflicting global and target decisions; review required. This is not a localized anomaly."

    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(tmp_path))
    registration = webchat_server._register_anomaly_window(
        result=shared_result,
        popup=shared,
        agent_status={},
        monitor_request={},
    )
    assert registration["registered"] is True
    detail = json.loads((tmp_path / "windows" / "inconclusive-window.json").read_text())
    assert vibrogemma_webchat._replay_state(detail) == "inconclusive"


def test_gemma_global_only_and_network_wide_anomalies_alert_and_replay():
    from vibroagent_mcp import vibrogemma_webchat

    normal_targets = [
        {"sensor_id": f"target_{index}", "affected": False, "class": "normal", "severity": "none"}
        for index in range(1, 6)
    ]
    global_only = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "unknown_anomaly",
                "global_target_consistent": False,
                "localization_status": "unresolved_global_anomaly",
                "targets": normal_targets,
            },
        }
    }
    popup = vibrogemma_webchat._apply_popup_policy({"show_popup": False}, global_only)
    assert popup["show_popup"] is True
    assert popup["save_replay"] is True
    assert popup["network_label"] == "unknown_anomaly"
    assert popup["localization"] == "unresolved_global_anomaly"
    assert popup["affected_sensor_ids"] == []
    assert vibrogemma_webchat._replay_state({"pipeline_result": global_only}) == "anomaly"

    all_targets = [
        {
            "sensor_id": f"target_{index}",
            "affected": True,
            "class": "unknown_anomaly",
            "severity": "advisory",
        }
        for index in range(1, 6)
    ]
    network_wide = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "unknown_anomaly",
                "global_target_consistent": True,
                "localization_verified": True,
                "targets": all_targets,
            },
        }
    }
    popup = vibrogemma_webchat._apply_popup_policy({"show_popup": False}, network_wide)
    assert popup["show_popup"] is True
    assert popup["save_replay"] is True
    assert popup["network_label"] == "unknown_anomaly"
    assert popup["localization"] == "network_wide"
    assert vibrogemma_webchat._replay_state({"pipeline_result": network_wide}) == "anomaly"


def test_gemma_geometry_unavailable_marks_targets_as_candidates():
    from vibroagent_mcp import vibrogemma_webchat

    targets = [
        {
            "sensor_id": f"target_{index}",
            "affected": index == 1,
            "class": "unknown_anomaly" if index == 1 else "normal",
            "severity": "advisory" if index == 1 else "none",
        }
        for index in range(1, 6)
    ]
    result = {
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "unknown_anomaly",
                "global_target_consistent": True,
                "localization_status": "geometry_unavailable",
                "localization_verified": False,
                "targets": targets,
            },
        }
    }
    popup = vibrogemma_webchat._apply_popup_policy({"show_popup": False}, result)
    assert popup["show_popup"] is True
    assert popup["localization"] == "geometry_unavailable"
    assert popup["localization_verified"] is False
    assert "target candidate" in popup["message"]
    assert result["network_assessment"]["localization_status"] == "geometry_unavailable"
    assert "model candidates" in result["main_agent_explanation"]


def test_synthetic_test_does_not_override_the_model_popup(monkeypatch, tmp_path):
    from vibroagent_mcp import vibrogemma_webchat, webchat_server

    result = {
        "window_id": "synthetic-target-5",
        "model_metadata": {
            "agent_mode": "vibrogemma_live",
            "vibrogemma_monitor": {
                "global_class": "normal",
                "global_target_consistent": True,
                "localization_verified": False,
                "capture": {
                    "synthetic_injection": {
                        "active": True,
                        "test_only": True,
                        "targets": {
                            f"target_{index}": {"kind": "sine"}
                            for index in range(1, 6)
                        },
                    }
                },
                "targets": [
                    {
                        "sensor_id": f"target_{index}",
                        "affected": False,
                        "class": "normal",
                        "severity": "none",
                    }
                    for index in range(1, 6)
                ],
            },
        },
    }
    popup = vibrogemma_webchat._apply_popup_policy({"show_popup": False}, result)
    assert popup["show_popup"] is False
    assert popup["save_replay"] is False
    assert popup["network_label"] == "normal"
    assert popup["synthetic_test"]["active"] is True

    result["model_metadata"]["vibrogemma_monitor"]["targets"][-1].update(
        {"class": "data_invalid", "severity": "advisory"}
    )
    invalid_popup = vibrogemma_webchat._apply_popup_policy({"show_popup": False}, result)
    assert invalid_popup["show_popup"] is True
    assert invalid_popup["network_label"] == "data_invalid"
    assert invalid_popup["title"] == "Agent data quality"
    assert invalid_popup["save_replay"] is False

    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(tmp_path))
    registration = webchat_server._register_anomaly_window(
        result={"_vibrogemma_replay_input": {"private": True}},
        popup=invalid_popup,
        agent_status={},
        monitor_request={},
    )
    assert registration == {"registered": False, "reason": "synthetic_test"}


def test_ui_synthetic_injection_control_uses_lumo_live_overlay(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat

    warmed = []
    monkeypatch.setattr(
        vibrogemma_webchat,
        "_lumo_training_replay",
        lambda target: warmed.append(target),
    )
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR", {"old": True})
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR_AT", 123.0)
    monkeypatch.delenv("VIBRO_LUMO_TRAINING_INJECTION", raising=False)
    assert vibrogemma_webchat._synthetic_injection_status()["enabled"] is False
    assert vibrogemma_webchat._set_synthetic_injection(True) == {
        "ok": True,
        "enabled": True,
        "mode": "lumo_live_overlay",
        "target": "target_3",
        "condition": "DAM4_010",
        "expected_target": "target_3",
    }
    assert os.environ["VIBRO_SYNTHETIC_TARGET5_ENABLED"] == "0"
    assert vibrogemma_webchat._LATEST_MONITOR is None
    assert vibrogemma_webchat._set_synthetic_injection(True, "target_5") == {
        "ok": True,
        "enabled": True,
        "mode": "lumo_live_overlay",
        "target": "target_5",
        "condition": "DAM6_010",
        "expected_target": "target_5",
    }
    with pytest.raises(ValueError, match="target_3 or target_5"):
        vibrogemma_webchat._set_synthetic_injection(True, "target_4")
    assert vibrogemma_webchat._set_synthetic_injection(False)["enabled"] is False
    assert warmed == ["target_3", "target_5"]


def test_monitor_cache_rejects_previous_overlay_target(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat

    payload = {
        "result": {
            "model_metadata": {
                "vibrogemma_monitor": {
                    "capture": {
                        "synthetic_injection": {
                            "active": True,
                            "selected_target": "target_5",
                        }
                    }
                }
            }
        }
    }
    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "target_3")
    assert vibrogemma_webchat._monitor_matches_current_injection(payload) is False
    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "target_5")
    assert vibrogemma_webchat._monitor_matches_current_injection(payload) is True


def test_lumo_test_waveform_overlays_the_moving_live_episode(monkeypatch):
    import numpy as np

    from vibroagent_mcp import vibrogemma_webchat
    from vibroagent_mcp.vibrogemma_live import SLOTS

    templates = {
        slot: np.vstack(
            (
                np.arange(100, dtype=np.float32) + index * 100,
                np.ones(100, dtype=np.float32) * index,
                np.zeros(100, dtype=np.float32),
            )
        )
        for index, slot in enumerate(SLOTS)
    }
    calls = 0

    def live_payload(_query):
        nonlocal calls
        calls += 1
        values = np.arange(10, dtype=float) + calls
        return {
            "ok": True,
            "status": "ok",
            "machine_id": "target_5",
            "axis": "x",
            "sampling_rate_hz": 10.0,
            "downsample_stride": 1,
            "samples_g": values.tolist(),
            "stats": {},
            "metadata": {"data_provenance": "live"},
            "diagnostics": {"board_reader_process": True},
        }

    monkeypatch.setattr(vibrogemma_webchat.webchat_server, "_live_vibrometer_payload", live_payload)
    monkeypatch.setattr(
        vibrogemma_webchat,
        "_lumo_overlay_templates",
        lambda *_args: (
            templates,
            {
                "kind": "lumo_live_overlay",
                "source_fixture": "08_DAM6_010/lumo.mat@520-530s",
                "live_capture_preserved": True,
            },
        ),
    )
    monkeypatch.setattr(vibrogemma_webchat, "_lumo_overlay_gain", lambda: 1.0)
    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "target_5")
    payload = vibrogemma_webchat._lumo_waveform_payload(
        {
            "machine_id": ["target_5"],
            "axis": ["x"],
            "duration_s": ["1"],
            "max_points": ["100"],
        }
    )

    expected_template = np.arange(590, 600, dtype=float)
    expected_template -= expected_template.mean()
    assert payload["samples_g"] == pytest.approx(expected_template)
    assert payload["metadata"]["waveform_source"] == "live_with_lumo_overlay"
    assert payload["metadata"]["source_fixture"].startswith("08_DAM6_010/")
    assert payload["metadata"]["data_provenance"] == "live"
    assert payload["metadata"]["overlay_axis_present"] is True
    assert payload["diagnostics"]["board_reader_process"] is True
    assert payload["diagnostics"]["overlay"] is True

    z_payload = vibrogemma_webchat._lumo_waveform_payload(
        {"machine_id": ["target_5"], "axis": ["z"], "duration_s": ["1"]}
    )
    assert z_payload["samples_g"] == pytest.approx(np.arange(2, 12))
    assert z_payload["metadata"]["overlay_axis_present"] is False
    assert calls == 2

    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "0")
    assert vibrogemma_webchat._lumo_waveform_payload({}) is None


def test_legacy_replay_without_geometry_proof_is_unverified():
    from vibroagent_mcp import vibrogemma_webchat

    targets = [
        {
            "sensor_id": f"target_{index}",
            "affected": True,
            "class": "unknown_anomaly",
            "severity": "advisory",
        }
        for index in range(1, 6)
    ]
    view = vibrogemma_webchat._monitor_view(
        {"global_class": "unknown_anomaly", "global_target_consistent": True},
        targets,
    )

    assert view["localization"] == "geometry_unavailable"
    assert view["localization_verified"] is False


def test_gemma_cached_monitor_expires_without_mutating_background_query(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat, webchat_server

    original_decision = webchat_server._apply_monitor_llm_decision
    original_popup = webchat_server._build_agent_popup
    original_monitor = webchat_server._agent_monitor_payload
    original_handler = webchat_server.WebchatHandler
    monkeypatch.setenv("VIBROGEMMA_MONITOR_STALE_S", "5")
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR", {
        "ok": True,
        "status": "ok",
        "anomaly": True,
        "popup": {"show_popup": True},
        "result": {},
    })
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR_AT", time.monotonic() - 30)
    monkeypatch.setattr(vibrogemma_webchat, "_MONITOR_QUERY", {"axis": ["norm"]})
    monkeypatch.setattr(vibrogemma_webchat, "_DEPLOYMENT_STATUS", {})
    try:
        vibrogemma_webchat._install()
        payload = webchat_server._agent_monitor_payload({"axis": ["x"], "skip_cache": ["1"]})
        assert payload["status"] == "stale"
        assert payload["stale"] is True
        assert payload["anomaly"] is False
        assert payload["popup"]["show_popup"] is False
        assert payload["agent"]["llm_agent_active"] is False
        assert vibrogemma_webchat._MONITOR_QUERY == {"axis": ["norm"]}
    finally:
        webchat_server._apply_monitor_llm_decision = original_decision
        webchat_server._build_agent_popup = original_popup
        webchat_server._agent_monitor_payload = original_monitor
        webchat_server.WebchatHandler = original_handler


def test_gemma_monitor_loop_recovers_and_runs_on_ten_second_cadence(monkeypatch):
    from vibroagent_mcp import vibrogemma_webchat

    calls = []
    sleeps = []

    def monitor(_query):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise RuntimeError("one failed cycle")
        return {
            "ok": True,
            "result": {"model_metadata": {"monitor_llm_model_used": True}},
        }

    def stop_after_second(delay):
        sleeps.append(delay)
        if len(sleeps) == 2:
            raise StopIteration

    monkeypatch.delenv("VIBROGEMMA_MONITOR_INTERVAL_S", raising=False)
    monkeypatch.setattr(vibrogemma_webchat, "_pin_current_thread", lambda *_args: None)
    monkeypatch.setattr(vibrogemma_webchat.time, "sleep", stop_after_second)
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR", None)
    monkeypatch.setattr(vibrogemma_webchat, "_LATEST_MONITOR_AT", 0.0)
    with pytest.raises(StopIteration):
        vibrogemma_webchat._monitor_loop(monitor)

    assert calls == [1, 2]
    assert all(9.0 <= delay <= 10.0 for delay in sleeps)
    assert vibrogemma_webchat._LATEST_MONITOR["ok"] is True


def test_monitor_freshness_is_measured_from_capture_end():
    from datetime import datetime, timedelta, timezone
    from vibroagent_mcp.vibrogemma_webchat import _capture_age_s

    payload = {
        "result": {
            "window": {
                "end_utc": (datetime.now(timezone.utc) - timedelta(seconds=7)).isoformat()
            }
        }
    }
    assert 6.5 <= _capture_age_s(payload) <= 8.5


def test_gemma_surfaces_bundle_permitted_use(monkeypatch, tmp_path):
    import json
    from vibroagent_mcp import vibrogemma_webchat

    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "promotion_status": "unpromoted_validation_gate_failed",
                "permitted_use": "local_research_and_replay_only",
            }
        )
    )
    monkeypatch.setenv("VIBROGEMMA_BUNDLE", str(tmp_path))
    monkeypatch.setattr(vibrogemma_webchat, "_DEPLOYMENT_STATUS", None)
    assert vibrogemma_webchat._deployment_status() == {
        "promotion_status": "unpromoted_validation_gate_failed",
        "permitted_use": "local_research_and_replay_only",
    }


def test_g1_normal_rows_feed_normal_history(monkeypatch):
    from collections import deque
    from vibroagent_mcp import webchat_server

    rows = deque(
        [
            {"label": "normal_relative_to_reference_sensor", "q": 0, "sensors": {"target_1": {"rms": 1.0}}},
            {"label": "normal", "q": 0, "sensors": {"target_1": {"rms": 1.2}}},
            {"label": "inconclusive", "q": 0, "sensors": {"target_1": {"rms": 9.0}}},
        ]
    )
    monkeypatch.setattr(webchat_server, "_WINDOW_HISTORY", rows)
    monkeypatch.setattr(webchat_server, "_HISTORY_STATS_CACHE", {})
    assert webchat_server._history_values_locked("target_1", "rms") == [1.0, 1.2]


def _payload():
    def target(sensor_id, affected, class_name, severity, probability=0.9):
        return {
            "sensor_id": sensor_id,
            "affected": affected,
            "class": class_name,
            "severity": severity,
            "choice_probability": None,
            "raw_choice_probability": probability,
            "confidence_operational": False,
        }

    return {
        "schema_version": "1.0",
        "window": {
            "start_utc": "2026-08-20T00:00:00Z",
            "duration_seconds": 10.0,
            "effective_duration_seconds": 10.0,
            "episode_id": "test-window",
        },
        "system_state": "data_invalid",
        "targets": [
            target("target_1", False, "normal", "none"),
            target("target_2", True, "unknown_anomaly", "advisory"),
            target("target_3", False, "data_invalid", "advisory", None),
            target("target_4", True, "unknown_anomaly", "advisory"),
            target("target_5", False, "normal", "none"),
        ],
        "explanation": "Target 2 has a modal frequency shift.",
        "quality": {},
        "model": {
            "checkpoint": "test",
            "raw_record": "CLASSIFICATION",
            "explanation_source": "deterministic_physics_evidence",
            "explanation_grounded": True,
            "decision_mode": "parallel_binary_global_plus_joint_target_set_v1",
            "label_contract": {
                "classes": ["normal", "unknown_anomaly"],
                "quality_class": "data_invalid",
                "severities": ["none", "advisory"],
            },
            "global_class": "normal",
            "global_severity": "none",
            "global_target_consistent": False,
            "global_choice_probability": None,
            "global_confidence_operational": False,
        },
        "evidence": {},
    }


def test_vibrogemma_result_validation_and_panel_mapping():
    targets = _validate_model_result(_payload())
    assert _panel_sensor_label(targets[0]) == "normal"
    assert _panel_sensor_label(targets[1]) == "unknown_anomaly"
    assert _panel_sensor_label(targets[2]) == "data_invalid"
    assert _panel_network_label(targets) == "data_invalid"
    assert _panel_network_label([targets[0], targets[1], targets[4]]) == "unknown_anomaly"
    assert _panel_network_label([targets[0], targets[4]]) == "normal"
    assert _panel_system_label({"system_state": "normal"}) == "normal"
    assert _panel_system_label({"system_state": "advisory"}) == "unknown_anomaly"
    assert _panel_system_label({"system_state": "data_invalid"}) == "data_invalid"


def test_vibrogemma_result_rejects_bad_order():
    payload = _payload()
    payload["targets"][0]["sensor_id"] = "target_2"
    with pytest.raises(ValueError, match="order"):
        _validate_model_result(payload)


def test_vibrogemma_result_rejects_legacy_or_unqualified_target_contract():
    payload = _payload()
    payload["targets"][0].pop("confidence_operational")
    with pytest.raises(ValueError, match="active G1 target contract"):
        _validate_model_result(payload)

    payload = _payload()
    payload["targets"][1]["class"] = "modal_frequency_shift"
    with pytest.raises(ValueError, match="unsupported class"):
        _validate_model_result(payload)

    payload = _payload()
    payload["targets"][0]["choice_probability"] = 0.9
    payload["targets"][0]["confidence_operational"] = True
    with pytest.raises(ValueError, match="operational confidence"):
        _validate_model_result(payload)

    payload = _payload()
    payload["model"]["decision_mode"] = "factorized_global_plus_independent_targets_v1"
    with pytest.raises(ValueError, match="active G1 decision mode"):
        _validate_model_result(payload)

    payload = _payload()
    payload["model"]["global_target_consistent"] = True
    with pytest.raises(ValueError, match="global_target_consistent"):
        _validate_model_result(payload)

    with pytest.raises(ValueError, match="checkpoint"):
        _validate_model_result(
            _payload(), expected_model_identity={"checkpoint": "different.gguf"}
        )


def test_vibrogemma_result_keeps_global_target_inconsistency_separate_from_schema_validity():
    payload = _payload()
    payload["system_state"] = "advisory"
    payload["targets"][2].update(
        {
            "affected": False,
            "class": "normal",
            "severity": "none",
            "raw_choice_probability": 0.8,
        }
    )
    assert len(_validate_model_result(payload)) == 5


def test_vibrogemma_result_accepts_bundle_declared_factorized_mode():
    payload = _payload()
    mode = "factorized_global_plus_independent_targets_v1"
    payload["model"]["decision_mode"] = mode
    assert len(
        _validate_model_result(payload, expected_model_identity={"decision_mode": mode})
    ) == 5


def test_healthy_profile_cache_key_changes_when_profile_is_replaced(monkeypatch, tmp_path):
    from scripts.live_vibrogemma_worker import _healthy_profile_cache_key

    profile = tmp_path / "healthy.json"
    profile.write_text("{}")
    monkeypatch.setenv("VIBROGEMMA_HEALTHY_PROFILE", str(profile))
    first = _healthy_profile_cache_key(tmp_path)
    profile.write_text('{"version":"new"}')
    second = _healthy_profile_cache_key(tmp_path)
    assert first != second


def test_bundle_selects_packaged_pipeline_before_legacy_fallback(monkeypatch, tmp_path):
    from scripts.live_vibrogemma_worker import _pipeline_source

    source = tmp_path / "pipeline_src"
    package = source / "vibrogemma"
    package.mkdir(parents=True)
    (package / "contracts.py").write_text("# packaged\n")
    monkeypatch.delenv("VIBROGEMMA_PIPELINE_SRC", raising=False)

    assert _pipeline_source(tmp_path) == source.resolve()


def test_bundled_healthy_profile_participates_in_cache_key(monkeypatch, tmp_path):
    import json

    from scripts.live_vibrogemma_worker import _healthy_profile_cache_key

    profile = tmp_path / "artifacts" / "healthy.json"
    profile.parent.mkdir()
    profile.write_text("{}")
    (tmp_path / "vibrogemma_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "multiresolution": {
                        "physics": {"healthy_profile_path": "artifacts/healthy.json"}
                    }
                }
            }
        )
    )
    monkeypatch.delenv("VIBROGEMMA_HEALTHY_PROFILE", raising=False)
    first = _healthy_profile_cache_key(tmp_path)
    profile.write_text('{"version":"new"}')

    assert first != _healthy_profile_cache_key(tmp_path)


def test_worker_verifies_manifest_encoder_pipeline_and_external_profile(monkeypatch, tmp_path):
    import hashlib
    import json

    from scripts.live_vibrogemma_worker import (
        _verify_runtime_artifacts,
        _verify_runtime_artifacts_cached,
    )

    package = tmp_path / "pipeline_src" / "vibrogemma"
    package.mkdir(parents=True)
    files = {
        "vibrogemma_config.json": json.dumps(
            {"model": {"multiresolution": {"physics": {}}}}
        ).encode(),
        "vibration_encoder_projector.onnx": b"onnx",
        "vibration_encoder_projector.onnx.data": b"weights",
        "pipeline_src/vibrogemma/contracts.py": b"# exact packaged source\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for relative, content in files.items()
        ]
    }
    manifest_path = tmp_path / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    profile = tmp_path.parent / f"{tmp_path.name}-healthy.json"
    profile.write_text('{"version":"site"}')
    monkeypatch.delenv("VIBROGEMMA_PIPELINE_SRC", raising=False)
    monkeypatch.setenv("VIBROGEMMA_HEALTHY_PROFILE", str(profile))
    monkeypatch.setenv(
        "VIBROGEMMA_BUNDLE_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "VIBROGEMMA_HEALTHY_PROFILE_SHA256",
        hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _verify_runtime_artifacts_cached.cache_clear()

    verified = _verify_runtime_artifacts(tmp_path)
    assert verified["pipeline_source"] == str((tmp_path / "pipeline_src").resolve())
    assert verified["pipeline_python_file_count"] == 1
    assert verified["sha256"]["vibration_encoder_projector.onnx"] == hashlib.sha256(
        b"onnx"
    ).hexdigest()
    assert verified["sha256"]["healthy_profile"] == hashlib.sha256(
        profile.read_bytes()
    ).hexdigest()

    monkeypatch.setenv("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="bundle manifest sha256 mismatch"):
        _verify_runtime_artifacts(tmp_path)
    monkeypatch.setenv(
        "VIBROGEMMA_BUNDLE_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("VIBROGEMMA_HEALTHY_PROFILE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="healthy profile sha256 mismatch"):
        _verify_runtime_artifacts(tmp_path)
    monkeypatch.setenv(
        "VIBROGEMMA_HEALTHY_PROFILE_SHA256",
        hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    (tmp_path / "vibration_encoder_projector.onnx").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact sha256 mismatch"):
        _verify_runtime_artifacts(tmp_path)


def test_worker_rejects_missing_or_zero_encoder_board_tokens():
    import numpy as np

    from scripts.live_vibrogemma_worker import _validate_encoder_output

    tokens = np.ones((1, 84, 1536), dtype=np.float32)
    mask = np.ones((1, 84), dtype=bool)
    _validate_encoder_output(tokens, mask)

    missing = mask.copy()
    missing[0, 28:42] = False
    with pytest.raises(ValueError, match="no valid soft tokens for board 2"):
        _validate_encoder_output(tokens, missing)

    zero = tokens.copy()
    zero[0, 56:70] = 0.0
    with pytest.raises(ValueError, match="zero-valued valid soft tokens for board 4"):
        _validate_encoder_output(zero, mask)


def test_capture_board_aligns_xyz_to_one_sdk_timestamp_end():
    import numpy as np
    from types import SimpleNamespace

    offsets = {"x": 0.0, "y": 0.1, "z": 0.2}

    def reader(*, axis, **_kwargs):
        timestamps = offsets[axis] + np.arange(120, dtype=np.float64) / 10.0
        return SimpleNamespace(
            signal=timestamps.astype(np.float32),
            timestamps_s=timestamps,
            sampling_rate_hz=10,
            metadata={
                "data_is_current": True,
                "data_file_modified_at_utc": "2026-08-31T00:00:20+00:00",
                "sdk_latest_timestamp_gap_s": 0.0,
                "timestamp_warning": None,
            },
        )

    sensor = SimpleNamespace(acquisition_folder="/capture", hsd_sensor_name="iis3dwb_acc")
    captured = _capture_board("target_1", sensor, reader)

    assert captured.values.shape == (3, 115)
    assert captured.metadata["alignment_mode"] == "sdk_timestamp_common_xyz_end"
    assert captured.metadata["selected_timestamp_start_s"] == pytest.approx(0.5)
    assert captured.metadata["selected_timestamp_end_s"] == pytest.approx(12.0)
    assert captured.values[:, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_capture_board_normalizes_mixed_post_power_cycle_timestamp_epochs():
    import numpy as np
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    raw_offsets = {"x": 0.0, "y": 1_000.1, "z": 2_000.2}
    wall_offsets = {"x": 0.0, "y": 0.1, "z": 0.2}

    def reader(*, axis, **_kwargs):
        timestamps = raw_offsets[axis] + np.arange(140, dtype=np.float64) / 10.0
        signal = wall_offsets[axis] + np.arange(140, dtype=np.float64) / 10.0
        return SimpleNamespace(
            signal=signal.astype(np.float32),
            timestamps_s=timestamps,
            sampling_rate_hz=10,
            metadata={
                "data_is_current": True,
                "data_file_modified_at_utc": (
                    base + timedelta(seconds=14.0 + wall_offsets[axis])
                ).isoformat(),
                "sdk_latest_timestamp_gap_s": 0.0,
                "timestamp_warning": None,
            },
        )

    sensor = SimpleNamespace(acquisition_folder="/capture", hsd_sensor_name="iis3dwb_acc")
    captured = _capture_board("target_2", sensor, reader)

    assert captured.metadata["timestamp_epoch_normalized"] is True
    assert "timestamp_epoch_normalized_from_file_mtime" in captured.metadata["quality_notes"]
    np.testing.assert_allclose(captured.values, np.tile(captured.values[0], (3, 1)), atol=1e-6)
    assert captured.approximate_end_utc == base + timedelta(seconds=14)


def test_capture_without_sdk_gap_cannot_claim_wall_clock_alignment():
    import numpy as np
    from types import SimpleNamespace

    def reader(**_kwargs):
        timestamps = np.arange(120, dtype=np.float64) / 10.0
        return SimpleNamespace(
            signal=timestamps.astype(np.float32),
            timestamps_s=timestamps,
            sampling_rate_hz=10,
            metadata={
                "data_is_current": True,
                "data_file_modified_at_utc": "2026-08-31T00:00:20+00:00",
                "timestamp_warning": None,
            },
        )

    sensor = SimpleNamespace(acquisition_folder="/capture", hsd_sensor_name="iis3dwb_acc")
    assert _capture_board("target_1", sensor, reader).approximate_end_utc is None


def test_finalize_capture_trims_every_board_to_one_wall_end():
    import numpy as np
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 8, 31, tzinfo=timezone.utc)
    captured = {}
    for index, slot in enumerate(("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")):
        end = base + timedelta(seconds=index * 0.1)
        captured[slot] = _CapturedBoard(
            values=np.tile(np.arange(115, dtype=np.float32), (3, 1)),
            sampling_rate_hz=10.0,
            metadata={
                "selected_timestamp_start_s": 0.0,
                "selected_timestamp_end_s": 11.5,
                "timestamp_monotonic": True,
                "alignment_mode": "sdk_timestamp_common_xyz_end",
            },
            approximate_end_utc=end,
        )

    arrays, rates, metadata, capture = _finalize_capture(
        captured,
        capture_started_utc=base - timedelta(seconds=1),
        capture_completed_utc=base + timedelta(seconds=1),
    )

    assert capture["cross_board_alignment_verified"] is True
    assert capture["window_end_utc"] == base.isoformat()
    assert arrays["baseline"].shape == (3, 100)
    assert arrays["target_5"].shape == (3, 100)
    assert arrays["baseline"][0, -1] == 114
    assert arrays["target_5"][0, -1] == 109
    assert metadata["target_5"]["cross_board_trim_s"] == pytest.approx(0.5)
    assert set(rates.values()) == {10.0}


def test_capture_quality_measures_clipping_packet_gaps_and_alignment_residual():
    import numpy as np
    from datetime import datetime, timezone

    base = datetime(2026, 8, 31, tzinfo=timezone.utc)
    metadata = {
        "selected_timestamp_start_s": 0.0,
        "selected_timestamp_end_s": 11.5,
        "timestamp_monotonic": True,
        "alignment_mode": "sdk_timestamp_common_xyz_end",
        "packet_loss_count": 2,
        "clipping_abs_g_by_axis": {"x": 15.99, "y": 15.99, "z": 15.99},
        "quality_notes": [],
    }
    values = np.zeros((3, 115), dtype=np.float32)
    values[0, -1] = 16.0
    captured = {
        slot: _CapturedBoard(
            values=values.copy(),
            sampling_rate_hz=10.0,
            metadata=dict(metadata),
            approximate_end_utc=base,
        )
        for slot in ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
    }
    _, _, capture_by_slot, _ = _finalize_capture(
        captured,
        capture_started_utc=base,
        capture_completed_utc=base,
    )
    quality = capture_by_slot["target_1"]
    assert quality["clipped_fraction"] == pytest.approx(1 / 300)
    assert quality["packet_loss_count"] == 2
    assert quality["clock_skew_ms"] == pytest.approx(0.0)
    assert quality["clock_skew_source"] == "common_end_rounding_residual"


def test_live_device_manifest_binds_slots_to_physical_serials(monkeypatch, tmp_path):
    import json
    from vibroagent_mcp.sensor_registry import SensorRegistry

    sensors = []
    devices = []
    for index, slot in enumerate(("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")):
        folder = tmp_path / slot
        folder.mkdir()
        sensors.append(
            {
                "sensor_id": slot,
                "role": "baseline" if slot == "baseline" else "target",
                "acquisition_folder": str(folder),
                "hsd_sensor_name": "iis3dwb_acc",
            }
        )
        devices.append(
            {
                "device_id": index,
                "sensor_id": slot,
                "serial": f"serial-{index}",
                "folder": str(folder),
                "logged_sensors": ["iis3dwb_acc"],
            }
        )
    config = tmp_path / "sensors.json"
    config.write_text(json.dumps({"baseline_sensor_id": "baseline", "sensors": sensors}))
    manifest = tmp_path / "vibroagent_live_devices.json"
    manifest.write_text(json.dumps({"created_at_unix": 1, "devices": devices}))
    monkeypatch.setenv("VIBRO_LIVE_DEVICE_MANIFEST", str(manifest))
    monkeypatch.setenv("VIBRO_REQUIRE_DEVICE_IDENTITY", "1")

    identity = _device_identity(SensorRegistry(config))
    assert identity["verified"] is True
    assert identity["mapping"]["target_1"]["serial"] == "serial-1"

    devices[1]["folder"] = str(tmp_path / "target_5")
    manifest.write_text(json.dumps({"devices": devices}))
    with pytest.raises(ValueError, match="registry folder"):
        _device_identity(SensorRegistry(config))


def test_episode_marks_unmeasured_timing_invalid_and_geometry_unavailable(monkeypatch):
    import numpy as np
    from pathlib import Path

    from scripts.live_vibrogemma_worker import SLOTS, _episode

    pipeline = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "vibroagent-gemma-g1"
        / "pipeline_src"
    )
    monkeypatch.setenv("VIBROGEMMA_PIPELINE_SRC", str(pipeline))
    payload = {slot: np.zeros((3, 100), dtype=np.float32) for slot in SLOTS}
    payload.update({f"fs_hz__{slot}": np.float64(10.0) for slot in SLOTS})
    payload.update(
        {
            f"metadata__{slot}": {
                "timestamp_monotonic": False,
                "timestamp_warning": "common end unavailable",
                "clock_skew_ms": None,
                "clipped_fraction": None,
                "packet_loss_count": None,
            }
            for slot in SLOTS
        }
    )
    payload["episode_id"] = "stable-episode"
    episode = _episode(payload, pipeline.parent, amplitude_calibrated=True)
    assert episode.episode_id == "stable-episode"
    assert episode.metadata["geometry_valid"] is False
    assert all(board.sensor_position is None for board in episode.boards)
    assert all(board.quality.timestamp_monotonic is False for board in episode.boards)
    assert all("clock_skew_unmeasured" in board.quality.notes for board in episode.boards)


def test_sensor_registry_accepts_optional_geometry_without_inventing_defaults(tmp_path):
    import json

    from vibroagent_mcp.sensor_registry import SensorRegistry

    baseline = tmp_path / "baseline"
    target = tmp_path / "target"
    baseline.mkdir()
    target.mkdir()
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {
                        "sensor_id": "baseline",
                        "role": "baseline",
                        "acquisition_folder": str(baseline),
                        "sensor_position": [0.0, 0.0, 0.0],
                        "orientation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                    {"sensor_id": "target_1", "acquisition_folder": str(target)},
                ],
            }
        )
    )

    registry = SensorRegistry(config)
    assert registry.sensors["baseline"].sensor_position == (0.0, 0.0, 0.0)
    assert registry.sensors["baseline"].orientation_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert registry.sensors["target_1"].sensor_position is None
    assert registry.sensors["target_1"].orientation_matrix is None


def test_live_decision_reuses_encoder_episode_for_metrics_and_private_replay(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import numpy as np

    from scripts import live_vibrogemma_worker
    from vibroagent_mcp import vibrogemma_live

    folders = {}
    sensors = []
    for index, slot in enumerate(vibrogemma_live.SLOTS):
        folder = tmp_path / slot
        folder.mkdir()
        folders[slot] = folder
        sensors.append(
            {
                "sensor_id": slot,
                "role": "baseline" if index == 0 else "target",
                "acquisition_folder": str(folder),
            }
        )
    config = tmp_path / "sensors.json"
    config.write_text(json.dumps({"baseline_sensor_id": "baseline", "sensors": sensors}))

    def reader(*, machine_id, axis, **_kwargs):
        axis_index = {"x": 1, "y": 2, "z": 3}[axis]
        slot_index = vibrogemma_live.SLOTS.index(machine_id)
        timestamps = np.arange(120, dtype=np.float64) / 10.0
        signal = np.sin(2 * np.pi * timestamps) * (axis_index + slot_index + 1)
        return SimpleNamespace(
            signal=signal,
            timestamps_s=timestamps,
            sampling_rate_hz=10,
            metadata={
                "data_is_current": True,
                "data_file": str(folders[machine_id] / "iis3dwb_acc.dat"),
                "data_file_modified_at_utc": "2026-08-31T00:00:20+00:00",
                "sdk_latest_timestamp_gap_s": 0.0,
                "timestamp_warning": None,
                "timestamp_mode": "sdk_timestamp_window",
            },
        )

    captured_encoder_payload = {}
    vibration = {
        "schema": "vibrogemma-live-embeddings-v1",
        "shape": [84, 1536],
        "dtype": "float32-le",
        "embeddings_b64": "AA==",
        "soft_token_mask": [True] * 84,
        "quality_text": "quality",
        "evidence_text": "evidence",
        "window": {"episode_id": "authoritative-window", "duration_seconds": 10.0},
        "quality": {},
        "evidence": {"source": "captured"},
        "metadata": {"geometry_valid": False, "geometry_source": "unconfigured"},
    }

    def fake_encode(payload, _bundle, *, amplitude_calibrated):
        captured_encoder_payload.update(payload)
        assert amplitude_calibrated is True
        output = dict(vibration)
        output["window"] = {**vibration["window"], "episode_id": payload["episode_id"]}
        return output

    model_payload = _payload()
    model_payload["system_state"] = "advisory"
    model_payload["targets"] = [
        {
            "sensor_id": "target_1",
            "affected": True,
            "class": "unknown_anomaly",
            "severity": "advisory",
            "choice_probability": None,
            "raw_choice_probability": 0.9,
            "confidence_operational": False,
        },
        *[
            {
                "sensor_id": f"target_{index}",
                "affected": False,
                "class": "normal",
                "severity": "none",
                "choice_probability": None,
                "raw_choice_probability": 0.9,
                "confidence_operational": False,
            }
            for index in range(2, 6)
        ],
    ]
    model_payload["model"].update(
        {
            "localization_status": "single_target",
            "global_target_consistent": True,
            "global_class": "unknown_anomaly",
            "global_severity": "advisory",
        }
    )
    model_payload["window"]["episode_id"] = "authoritative-window"

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def call_json_model(self, *_args, **kwargs):
            assert kwargs["extra_body"]["vibration"]["window"]["episode_id"] == "authoritative-window"
            return SimpleNamespace(ok=True, data=model_payload, metadata={"provider": "fake"}, error=None)

    monkeypatch.setattr(vibrogemma_live, "board_reader_processes_enabled", lambda default=False: False)
    monkeypatch.setattr(vibrogemma_live, "read_sdk_vibrometer_window", reader)
    monkeypatch.setattr(vibrogemma_live, "ModelClient", FakeClient)
    monkeypatch.setattr(live_vibrogemma_worker, "encode_live_window", fake_encode)
    monkeypatch.setenv("VIBROGEMMA_BUNDLE", str(tmp_path / "bundle"))

    result = {
        "schema_version": "0.5.0",
        "analysis_domain": "building_vibration_relative_monitoring",
        "window_id": "authoritative-window",
        "sensor_agent_reports": [
            {
                "sensor_id": f"target_{index}",
                "python_rule_label": "localized_vibration_deviation",
                "small_agent_confidence": 0.99,
                "evidence": ["inherited deterministic evidence"],
                "model_metadata": {"fallback_reason": "old"},
            }
            for index in range(1, 6)
        ],
        "network_assessment": {
            "main_agent_confidence": 0.99,
            "evidence": ["inherited"],
            "python_rule_label": "localized_vibration_deviation",
        },
        "model_metadata": {"main_model_used": False, "fallbacks_used": ["old"]},
        "history_summary": {"x_median": 9.0},
        "reference_history": {"x_median": 9.0},
    }
    updated = vibrogemma_live.apply_vibrogemma_monitor_decision(
        result,
        model_base_url="http://model/v1",
        model_api_key="EMPTY",
        model="test",
        model_timeout_s=5.0,
        config_path=str(config),
    )

    assert captured_encoder_payload["episode_id"] == "authoritative-window"
    assert updated["window"]["duration_s"] == 10.0
    assert updated["window"]["episode_id"] == "authoritative-window"
    assert len(updated["all_sensor_metrics"]) == 6
    assert all(
        metric["source_metadata"]["source"] == "vibrogemma_authoritative_xyz_episode"
        for metric in updated["all_sensor_metrics"]
    )
    assert all("small_agent_confidence" not in report for report in updated["sensor_agent_reports"])
    assert all("python_rule_label" not in report for report in updated["sensor_agent_reports"])
    assert all(report["evidence"] == [] for report in updated["sensor_agent_reports"])
    assert "main_agent_confidence" not in updated["network_assessment"]
    assert updated["network_assessment"]["evidence"] == []
    assert updated["network_assessment"]["localization_status"] == "geometry_unavailable"
    assert updated["network_assessment"]["localization_verified"] is False
    assert "history_summary" not in updated and "reference_history" not in updated
    assert updated["_vibrogemma_replay_input"]["evidence"] == {"source": "captured"}
    assert updated["_vibrogemma_replay_input_sha256"] == _payload_sha256(
        updated["_vibrogemma_replay_input"]
    )
    monitor = updated["model_metadata"]["vibrogemma_monitor"]
    assert monitor["episode_id"] == "authoritative-window"
    assert monitor["model_evidence"] == model_payload["evidence"]
    assert "main_model_used" not in updated["model_metadata"]


def test_anomaly_registration_persists_exact_model_input(monkeypatch, tmp_path):
    import hashlib
    import json
    import numpy as np

    from vibroagent_mcp import webchat_server

    model_input = {
        "schema": "vibrogemma-live-embeddings-v1",
        "shape": [84, 1536],
        "dtype": "float32-le",
        "embeddings_b64": "AA==",
        "window": {"episode_id": "replayable-window", "duration_seconds": 10.0},
    }
    canonical = json.dumps(model_input, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    result = {
        "window_id": "replayable-window",
        "_vibrogemma_replay_input": model_input,
        "_vibrogemma_replay_input_sha256": digest,
        "_vibrogemma_encoder_replay_input": {
            "arrays": {
                slot: np.full((3, 4), index, dtype=np.float32)
                for index, slot in enumerate(
                    ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
                )
            },
            "metadata": {
                "schema": "vibroagent-encoder-input-v1",
                "episode_arrays_sha256": "exact-arrays",
            },
        },
    }
    monkeypatch.setenv("VIBRO_ANOMALY_WINDOW_DIR", str(tmp_path))

    registration = webchat_server._register_anomaly_window(
        result=result,
        popup={"show_popup": True, "network_label": "unknown_anomaly"},
        agent_status={},
        monitor_request={},
    )

    assert registration["registered"] is True
    assert "_vibrogemma_replay_input" not in result
    assert "_vibrogemma_encoder_replay_input" not in result
    model_input_path = tmp_path / "windows" / "replayable-window.model-input.json"
    assert model_input_path.read_text() == canonical
    assert model_input_path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "windows").stat().st_mode & 0o777 == 0o700
    detail = json.loads((tmp_path / "windows" / "replayable-window.json").read_text())
    assert detail["model_input_sha256"] == digest
    assert detail["model_input_file"] == model_input_path.name
    assert "model_input_path" not in detail
    encoder_input_path = tmp_path / "windows" / "replayable-window.encoder-input.npz"
    assert encoder_input_path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(encoder_input_path.read_bytes()).hexdigest() == detail["encoder_input_sha256"]
    with np.load(encoder_input_path) as replay:
        assert replay["target_5"].tolist() == np.full((3, 4), 5, dtype=np.float32).tolist()
        assert json.loads(str(replay["metadata_json"]))["episode_arrays_sha256"] == "exact-arrays"
    assert detail["pipeline_result"] == {"window_id": "replayable-window"}


def test_live_deployment_failure_never_preserves_inherited_normal(tmp_path):
    from vibroagent_mcp import vibrogemma_live

    result = {
        "schema_version": "0.5.0",
        "analysis_domain": "building_vibration_relative_monitoring",
        "window_id": "failed-window",
        "sensor_agent_reports": [
            {
                "sensor_id": f"target_{index}",
                "small_agent_label": "normal",
                "small_agent_confidence": 0.99,
                "python_rule_label": "normal",
                "evidence": ["inherited normal evidence"],
            }
            for index in range(1, 6)
        ],
        "network_assessment": {
            "window_id": "failed-window",
            "baseline_sensor_id": "baseline",
            "network_label": "normal",
            "main_agent_label": "normal",
            "main_agent_confidence": 0.99,
            "evidence": ["inherited normal evidence"],
        },
        "model_metadata": {"fallbacks_used": ["legacy_normal"]},
    }

    updated = vibrogemma_live.apply_vibrogemma_monitor_decision(
        result,
        model_base_url="http://127.0.0.1:1/v1",
        model_api_key="EMPTY",
        model="vibrogemma",
        model_timeout_s=1.0,
        config_path=str(tmp_path / "missing-sensors.json"),
    )

    assert updated["network_assessment"]["network_label"] == "data_invalid"
    assert updated["network_assessment"]["main_agent_label"] == "data_invalid"
    assert "No normal verdict was emitted" in updated["main_agent_explanation"]
    assert all(
        report["small_agent_label"] == "data_invalid"
        for report in updated["sensor_agent_reports"]
    )
    assert all("small_agent_confidence" not in report for report in updated["sensor_agent_reports"])
    assert all("python_rule_label" not in report for report in updated["sensor_agent_reports"])
    metadata = updated["model_metadata"]
    assert metadata["monitor_llm_model_used"] is False
    assert metadata["fallbacks_used"] == []
    assert metadata["vibrogemma_monitor"]["deployment_error"].startswith(
        "sensor_registry_failed:"
    )
    assert all(
        target["class"] == "data_invalid"
        for target in metadata["vibrogemma_monitor"]["targets"]
    )
