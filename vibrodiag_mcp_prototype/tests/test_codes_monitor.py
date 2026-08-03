"""Tests for the only supported webchat monitor path (all IO mocked)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vibroagent_mcp.webchat_server as W  # noqa: E402
from vibroagent_mcp import live_codes  # noqa: E402

FS = 26667.0
CODES = "".join(chr(0x4E00 + i % 500) for i in range(250))


class _FakeRegistry:
    def __init__(self, path):
        self.config_path = path
        self.baseline_sensor_id = "baseline"
        self.sensors = {
            slot: SimpleNamespace(
                sensor_id=slot,
                acquisition_folder=f"/fake/{slot}",
                hsd_sensor_name="iis3dwb_acc",
                axis="z",
                location=slot,
                role="baseline" if slot == "baseline" else "target",
            )
            for slot in ("baseline", *live_codes.LIVE_SLOTS)
        }


def _fake_reader(**kwargs):
    n = int(10.5 * FS)
    return SimpleNamespace(
        signal=np.random.default_rng(1).standard_normal(n) * 1e-3,
        sampling_rate_hz=FS,
        timestamps_s=None,
    )


def _fake_worker_run(cmd, **kwargs):
    out_path = Path(cmd[cmd.index("--out") + 1])
    slots = {"baseline": {"codes": CODES, "n_codes": 250, "level_db": -60.0}}
    for i, slot in enumerate(live_codes.LIVE_SLOTS, start=1):
        slots[slot] = {"codes": CODES, "n_codes": 250,
                       "level_db": -60.0 + i, "level_rel_db": float(i)}
    out_path.write_text(json.dumps({
        "schema": "live-codes-worker-v1",
        "slots": slots,
        "provenance": {"checkpoint_step": 29000,
                       "checkpoint_sha256": "ab" * 32},
    }))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


class _FakeModelClient:
    reply: str = ""
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def call_json_model(self, prompt, allowed_labels=None, *,
                        extra_body=None, system=None):
        type(self).last_call = {"prompt": prompt, "extra_body": extra_body,
                                "system": system}
        return SimpleNamespace(
            ok=True, used_model=True, text=type(self).reply,
            data=json.loads(type(self).reply),
            error=None, metadata={"model_used": True})


def _base_result():
    return {
        "sensor_agent_reports": [
            {"sensor_id": slot, "small_agent_label":
             "normal_relative_to_baseline"}
            for slot in live_codes.LIVE_SLOTS
        ],
        "network_assessment": {"quality_flags": []},
        "model_metadata": {},
    }


@pytest.fixture
def codes_env(monkeypatch):
    import subprocess

    monkeypatch.setattr(W, "SensorRegistry", _FakeRegistry)
    monkeypatch.setattr(W, "_resolve_live_sensor_config_path",
                        lambda value: "/fake/config.yaml")
    monkeypatch.setattr(W, "board_reader_processes_enabled",
                        lambda default=False: False)
    monkeypatch.setattr(W, "read_sdk_vibrometer_window",
                        lambda **kwargs: _fake_reader(**kwargs))
    monkeypatch.setattr(subprocess, "run", _fake_worker_run)
    monkeypatch.setattr(W, "ModelClient", _FakeModelClient)
    return monkeypatch


def _call():
    return W._apply_codes_monitor_decision(
        _base_result(),
        model_base_url="http://127.0.0.1:18181/v1",
        model_api_key="EMPTY",
        model="codes-v3",
        model_timeout_s=60.0,
    )


def test_codes_decision_merges_verdict(codes_env):
    model_message = (
        "Mild deviation at target_3. Immediate next step: verify the sensor response."
    )
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {"sensor_id": slot,
             "local_status": ("mild_deviation" if slot == "target_3"
                              else "normal_relative_to_baseline")}
            for slot in live_codes.LIVE_SLOTS],
        "network_status": "mild_local_deviation",
        "affected_sensor_ids": ["target_3"],
        "popup_message": model_message,
    })
    result = _call()
    metadata = result["model_metadata"]
    assert metadata["monitor_llm_model_used"] is True
    assert metadata["agent_mode"] == "codes_v3_monitor"
    assert metadata["codes_monitor"]["codec_checkpoint_step"] == 29000
    assessment = result["network_assessment"]
    assert assessment["network_label"] == "mild_local_deviation"
    assert assessment["affected_sensor_ids"] == ["target_3"]
    assert assessment["mild_sensor_ids"] == ["target_3"]
    assert assessment["significant_sensor_ids"] == []
    labels = {report["sensor_id"]: report["small_agent_label"]
              for report in result["sensor_agent_reports"]}
    assert labels["target_3"] == "mild_deviation"
    assert labels["target_1"] == "normal_relative_to_baseline"
    assert assessment["explanation"] == model_message
    assert assessment["explanation_source"] == "model"
    assert assessment["popup_message_source"] == "model"
    popup = W._build_agent_popup(result)
    assert popup["show_popup"] is True
    assert popup["severity"] == "notice"
    assert popup["mild_sensor_ids"] == ["target_3"]
    assert popup["message"] == model_message
    assert popup["message_source"] == "model"
    assert "codes" not in json.dumps(popup).lower()
    # the exact SFT system message and grammar rode along
    assert _FakeModelClient.last_call["system"] == \
        live_codes.system_message()
    assert _FakeModelClient.last_kwargs["max_tokens"] == 256
    fmt = _FakeModelClient.last_call["extra_body"]["response_format"]
    assert fmt["json_schema"]["name"] == "building_vibration_all_sensor"
    assert "popup_message" in fmt["json_schema"]["schema"]["required"]
    # The trained prefix remains intact; the runtime suffix asks for popup prose.
    assert _FakeModelClient.last_call["prompt"].startswith(
        "Building-vibration check from discrete vibration codes.")
    assert "entirely in your own words" in _FakeModelClient.last_call["prompt"]


def test_codes_decision_accepts_per_sensor_native_rates(codes_env, monkeypatch):
    import subprocess

    rates = {
        "baseline": 26657.0,
        "target_1": 26780.0,
        "target_2": 26554.0,
        "target_3": 26624.0,
        "target_4": 26664.0,
        "target_5": 26684.0,
    }

    def varied_rate_reader(**kwargs):
        fs_hz = rates[kwargs["machine_id"]]
        return SimpleNamespace(
            signal=np.random.default_rng(1).standard_normal(
                int(10.5 * fs_hz)) * 1e-3,
            sampling_rate_hz=fs_hz,
            timestamps_s=None,
        )

    captured = {}

    def inspect_worker_input(cmd, **kwargs):
        windows_path = Path(cmd[cmd.index("--windows") + 1])
        payload = np.load(windows_path)
        captured["rates"] = {
            slot: float(payload[f"fs_hz__{slot}"]) for slot in rates
        }
        assert "fs_hz" not in payload.files
        return _fake_worker_run(cmd, **kwargs)

    monkeypatch.setattr(W, "read_sdk_vibrometer_window", varied_rate_reader)
    monkeypatch.setattr(subprocess, "run", inspect_worker_input)
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {"sensor_id": slot,
             "local_status": "normal_relative_to_baseline"}
            for slot in live_codes.LIVE_SLOTS],
        "network_status": "normal_relative_to_reference_sensor",
        "affected_sensor_ids": [],
        "popup_message": "All monitored sensors match the reference. Continue routine monitoring.",
    })

    result = _call()

    assert result["model_metadata"]["monitor_llm_model_used"] is True
    assert captured["rates"] == rates
    codes_metadata = result["model_metadata"]["codes_monitor"]
    assert codes_metadata["fs_native_hz"] is None
    assert codes_metadata["fs_native_hz_by_slot"] == rates


def test_fixed_replay_reads_exact_ten_second_windows(codes_env, monkeypatch):
    read_calls = []

    def capture_reader(**kwargs):
        read_calls.append(kwargs)
        return _fake_reader(**kwargs)

    monkeypatch.setattr(W, "read_sdk_vibrometer_window", capture_reader)
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {"sensor_id": slot,
             "local_status": "normal_relative_to_baseline"}
            for slot in live_codes.LIVE_SLOTS],
        "network_status": "normal_relative_to_reference_sensor",
        "affected_sensor_ids": [],
        "popup_message": "All monitored sensors match the reference. Continue routine monitoring.",
    })

    result = W._apply_codes_monitor_decision(
        _base_result(),
        start_time_s=45.0,
        require_current=False,
        model_base_url="http://127.0.0.1:18181/v1",
        model_api_key="EMPTY",
        model="codes-v3",
        model_timeout_s=60.0,
    )

    assert result["model_metadata"]["monitor_llm_model_used"] is True
    assert len(read_calls) == 6
    assert {call["start_time_s"] for call in read_calls} == {45.0}
    assert {call["duration_s"] for call in read_calls} == {10.0}
    assert {call["require_current"] for call in read_calls} == {False}


def test_codes_decision_rejects_bad_model_output(codes_env):
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {"sensor_id": slot, "local_status": "calm"}
            for slot in live_codes.LIVE_SLOTS],
        "network_status": "mild_local_deviation",
        "affected_sensor_ids": [],
        "popup_message": "No target-specific deviation was identified.",
    })
    result = _call()
    metadata = result["model_metadata"]
    assert metadata["monitor_llm_model_used"] is False
    assert metadata["monitor_llm_fallback_reason"].startswith(
        "model_output_invalid")


def test_codes_decision_worker_failure_is_model_failure(codes_env, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=3, stdout="",
                                        stderr="boom"))
    result = _call()
    metadata = result["model_metadata"]
    assert metadata["monitor_llm_model_used"] is False
    assert "codes_worker_failed" in metadata["monitor_llm_fallback_reason"]


def test_codes_decision_short_window_fails_closed(codes_env, monkeypatch):
    def short_reader(**kwargs):
        return SimpleNamespace(
            signal=np.zeros(int(5.0 * FS)), sampling_rate_hz=FS,
            timestamps_s=None)
    monkeypatch.setattr(W, "read_sdk_vibrometer_window",
                        lambda **kwargs: short_reader(**kwargs))
    result = _call()
    assert "window_too_short" in \
        result["model_metadata"]["monitor_llm_fallback_reason"]


def test_axis_sweep_combines_worst_case(codes_env, monkeypatch):
    monkeypatch.setenv("VIBRO_CODES_AXES", "x,y,z")
    seen_axes = []

    def axis_reader(**kwargs):
        seen_axes.append(kwargs["axis"])
        return _fake_reader(**kwargs)

    monkeypatch.setattr(W, "read_sdk_vibrometer_window", axis_reader)

    replies = {
        "x": {"target_2": "mild_deviation",
              "network": "mild_local_deviation", "affected": ["target_2"]},
        "y": {"target_4": "significant_local_deviation",
              "network": "localized_vibration_deviation",
              "affected": ["target_4"]},
        "z": {"network": "normal_relative_to_reference_sensor",
              "affected": []},
    }
    order = iter(["x", "y", "z"])

    def per_axis_reply(self, prompt, allowed_labels=None, *,
                       extra_body=None, system=None):
        axis = next(order)
        spec = replies[axis]
        message = (
            f"Deviation detected at {spec['affected'][0]}. Verify sensor response."
            if spec["affected"]
            else "All monitored sensors match the reference. Continue monitoring."
        )
        payload = {
            "sensor_reports": [
                {"sensor_id": slot,
                 "local_status": spec.get(slot,
                                          "normal_relative_to_baseline")}
                for slot in live_codes.LIVE_SLOTS],
            "network_status": spec["network"],
            "affected_sensor_ids": spec["affected"],
            "popup_message": message,
        }
        import json as _json
        from types import SimpleNamespace as NS
        return NS(ok=True, used_model=True, text=_json.dumps(payload),
                  data=payload, error=None, metadata={"model_used": True})

    monkeypatch.setattr(_FakeModelClient, "call_json_model", per_axis_reply)

    result = _call()

    metadata = result["model_metadata"]
    assert metadata["monitor_llm_model_used"] is True
    codes_meta = metadata["codes_monitor"]
    assert codes_meta["axes"] == ["x", "y", "z"]
    assert codes_meta["combine_rule"] == \
        "max_severity_per_sensor_union_affected"
    # 6 sensors read per axis, 3 axes
    assert seen_axes.count("x") == 6 and seen_axes.count("y") == 6 \
        and seen_axes.count("z") == 6
    labels = {r["sensor_id"]: r["small_agent_label"]
              for r in result["sensor_agent_reports"]}
    assert labels["target_2"] == "mild_deviation"
    assert labels["target_4"] == "significant_local_deviation"
    assessment = result["network_assessment"]
    assert assessment["network_label"] == "localized_vibration_deviation"
    assert assessment["affected_sensor_ids"] == ["target_2", "target_4"]
    assert codes_meta["popup_message_axis"] == "y"
    assert assessment["explanation"].startswith(
        "Deviation detected at target_4.")
    assert W._build_agent_popup(result)["message"] == assessment["explanation"]


def test_axis_sweep_rejects_norm(codes_env, monkeypatch):
    monkeypatch.setenv("VIBRO_CODES_AXES", "z,norm")
    result = _call()
    assert "bad_axis" in \
        result["model_metadata"]["monitor_llm_fallback_reason"]


def _target_2_dominant_result():
    result = _base_result()
    metrics_by_target = {
        "target_1": (0.303048, 0.01600),
        "target_2": (5.580280, 0.04857),
        "target_3": (0.110288, 0.01253),
        "target_4": (0.115656, 0.01200),
        "target_5": (0.112728, 0.01210),
    }
    for report in result["sensor_agent_reports"]:
        p2p_g, rms_g = metrics_by_target[report["sensor_id"]]
        report["metrics"] = {
            "acceleration_peak_to_peak_g": p2p_g,
            "acceleration_rms_g": rms_g,
        }
    return result


def _call_with_result(result):
    return W._apply_codes_monitor_decision(
        result,
        model_base_url="http://127.0.0.1:18181/v1",
        model_api_key="EMPTY",
        model="codes-v3",
        model_timeout_s=60.0,
    )


def test_measurement_identity_guard_pins_captured_target_2():
    guard = W._codes_measurement_identity_guard(_target_2_dominant_result())

    assert guard["active"] is True
    assert guard["reason"] == "localized_measurement_dominance"
    assert guard["required_affected_sensor_ids"] == ["target_2"]
    assert guard["measurements"]["target_2"]["p2p_g"] == pytest.approx(5.58028)
    assert guard["measurements"]["target_3"]["p2p_g"] == pytest.approx(0.110288)
    assert guard["measurements"]["target_2"]["p2p_to_target_median"] > 40.0


def test_codes_decision_applies_target_2_identity_guard(codes_env):
    model_message = (
        "A strong localized impulse was detected at target_2. "
        "Inspect its mounting and repeat the controlled vibration check."
    )
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {
                "sensor_id": slot,
                "local_status": (
                    "significant_local_deviation"
                    if slot == "target_2"
                    else "normal_relative_to_baseline"
                ),
            }
            for slot in live_codes.LIVE_SLOTS
        ],
        "network_status": "localized_vibration_deviation",
        "affected_sensor_ids": ["target_2"],
        "popup_message": model_message,
    })

    result = _call_with_result(_target_2_dominant_result())

    assert result["network_assessment"]["affected_sensor_ids"] == ["target_2"]
    assert result["network_assessment"]["popup_message"] == model_message
    assert result["network_assessment"]["popup_message_source"] == "model"
    guard = result["model_metadata"]["codes_monitor"]["identity_guard"]
    assert guard["active"] is True
    assert guard["required_affected_sensor_ids"] == ["target_2"]
    prompt = _FakeModelClient.last_call["prompt"]
    assert 'affected_sensor_ids must be exactly ["target_2"]' in prompt
    schema = _FakeModelClient.last_call["extra_body"]["response_format"] \
        ["json_schema"]["schema"]
    assert schema["properties"]["affected_sensor_ids"]["items"]["enum"] == \
        ["target_2"]
    assert schema["properties"]["affected_sensor_ids"]["minItems"] == 1
    assert schema["properties"]["affected_sensor_ids"]["maxItems"] == 1


def test_codes_decision_rejects_model_target_substitution(codes_env):
    _FakeModelClient.reply = json.dumps({
        "sensor_reports": [
            {
                "sensor_id": slot,
                "local_status": (
                    "significant_local_deviation"
                    if slot == "target_3"
                    else "normal_relative_to_baseline"
                ),
            }
            for slot in live_codes.LIVE_SLOTS
        ],
        "network_status": "localized_vibration_deviation",
        "affected_sensor_ids": ["target_3"],
        "popup_message": (
            "A strong localized impulse was detected at target_3. "
            "Inspect its mounting and repeat the controlled vibration check."
        ),
    })

    result = _call_with_result(_target_2_dominant_result())

    metadata = result["model_metadata"]
    assert metadata["monitor_llm_model_used"] is False
    assert "expected identity-guarded IDs ['target_2']" in \
        metadata["monitor_llm_fallback_reason"]
