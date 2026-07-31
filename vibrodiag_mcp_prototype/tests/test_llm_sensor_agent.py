import json

import numpy as np

from vibroagent_mcp.llm_sensor_agent import AutonomousLlmSensorAgent
from vibroagent_mcp.model_client import ModelCallResult
from vibroagent_mcp.schemas import (
    OUT_OF_DOMAIN_COMPONENT_TERMS,
    NETWORK_LABELS,
    SENSOR_LABELS,
    ensure_building_domain_only,
)
from vibroagent_mcp.service import run_autonomous_sensor_check_service


def _write_signal(path, scale=1.0, spike=False):
    t = np.arange(96, dtype=float) / 12.0
    signal = scale * 0.001 * np.sin(2 * np.pi * 1.5 * t)
    if spike:
        signal[48] += 0.007
    path.write_text(
        "time_s,acceleration_g\n"
        + "\n".join(f"{time:.6f},{value:.9f}" for time, value in zip(t, signal))
        + "\n",
        encoding="utf-8",
    )


def _write_config(tmp_path):
    baseline = tmp_path / "baseline.csv"
    target_1 = tmp_path / "target_1.csv"
    target_2 = tmp_path / "target_2.csv"
    _write_signal(baseline, scale=1.0)
    _write_signal(target_1, scale=1.4)
    _write_signal(target_2, scale=3.5, spike=True)
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {"sensor_id": "baseline", "location": "reference", "role": "baseline", "replay_signal_path": str(baseline), "axis": "z", "sampling_rate_hz": 12},
                    {"sensor_id": "target_1", "location": "north_wall", "role": "target", "replay_signal_path": str(target_1), "axis": "z", "sampling_rate_hz": 12},
                    {"sensor_id": "target_2", "location": "south_wall", "role": "target", "replay_signal_path": str(target_2), "axis": "z", "sampling_rate_hz": 12},
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def _write_normal_config(tmp_path):
    baseline = tmp_path / "baseline.csv"
    target_1 = tmp_path / "target_1.csv"
    target_2 = tmp_path / "target_2.csv"
    _write_signal(baseline, scale=1.0)
    _write_signal(target_1, scale=1.0)
    _write_signal(target_2, scale=1.0)
    config = tmp_path / "normal_sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {"sensor_id": "baseline", "location": "reference", "role": "baseline", "replay_signal_path": str(baseline), "axis": "z", "sampling_rate_hz": 12},
                    {"sensor_id": "target_1", "location": "north_wall", "role": "target", "replay_signal_path": str(target_1), "axis": "z", "sampling_rate_hz": 12},
                    {"sensor_id": "target_2", "location": "south_wall", "role": "target", "replay_signal_path": str(target_2), "axis": "z", "sampling_rate_hz": 12},
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def _clear_model_env(monkeypatch):
    for name in (
        "SMALL_AGENT_BASE_URL",
        "SMALL_AGENT_MODEL",
        "SMALL_AGENT_API_KEY",
        "MAIN_AGENT_BASE_URL",
        "MAIN_AGENT_MODEL",
        "MAIN_AGENT_API_KEY",
        "QWEN_BASE_URL",
        "QWEN_MODEL",
        "QWEN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_autonomous_sensor_check_attempts_every_configured_sensor(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = run_autonomous_sensor_check_service(
        config_path=str(config),
        start_time_s=0,
        duration_s=8,
        require_current=False,
        use_model_agents=False,
    )

    assert result["agent"] == "AutonomousLlmSensorAgent"
    assert result["all_sensors_checked"] is True
    assert result["attempted_sensor_ids"] == ["baseline", "target_1", "target_2"]
    assert set(result["checked_sensor_ids"]) == {"baseline", "target_1", "target_2"}
    assert len(result["sensor_agent_reports"]) == 2
    assert {report["small_agent_label"] for report in result["sensor_agent_reports"]} <= SENSOR_LABELS
    assert result["network_assessment"]["network_label"] in NETWORK_LABELS
    assert result["model_metadata"]["raw_samples_sent_to_models"] is False
    assert result["network_assessment"]["explanation_source"] == "deterministic_fallback"
    assert result["main_agent_explanation_source"] == "deterministic_fallback"
    ensure_building_domain_only(result)


class _FailingModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        raise AssertionError("per-sensor model client should not be called in batch mode")


class _RecordingBatchModel:
    def __init__(self, sensor_label, network_label):
        self.sensor_label = sensor_label
        self.network_label = network_label
        self.prompts = []

    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        self.prompts.append(prompt)
        assert allowed_labels == set()
        assert "Raw vibration sample arrays are not provided" in prompt
        assert "samples_g" not in prompt
        assert "target_1" in prompt
        assert "target_2" in prompt
        data = {
            "sensor_reports": [
                {
                    "sensor_id": "target_1",
                    "local_status": self.sensor_label,
                    "confidence": 0.67,
                    "evidence": ["Compact relative metrics are acceptable."],
                },
                {
                    "sensor_id": "target_2",
                    "local_status": self.sensor_label,
                    "confidence": 0.68,
                    "evidence": ["Compact relative metrics are acceptable."],
                },
            ],
            "network_status": self.network_label,
            "confidence": 0.71,
            "affected_sensor_ids": ["target_1"],
            "evidence": ["One checked sensor is mildly elevated."],
            "explanation": "Monitoring triage found one mild local deviation.",
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_uses_single_batch_model_call_for_all_targets(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)
    sensor_model = _FailingModel()
    batch_model = _RecordingBatchModel("mild_deviation", "mild_local_deviation")

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=sensor_model,
        main_model_client=batch_model,
    ).run_window()

    assert len(batch_model.prompts) == 1
    assert "Write explanation as exactly two complete sentences" in batch_model.prompts[0]
    assert "Stop immediately after the closing brace" in batch_model.prompts[0]
    assert [report["small_agent_label"] for report in result["sensor_agent_reports"]] == ["mild_deviation", "mild_deviation"]
    assert all(report["model_metadata"]["decision_scope"] == "all_sensor_batch" for report in result["sensor_agent_reports"])
    assert result["network_assessment"]["network_label"] == "mild_local_deviation"
    assert result["network_assessment"]["model_metadata"]["decision_scope"] == "all_sensor_batch"
    assert result["network_assessment"]["explanation_source"] == "model"
    assert result["main_agent_explanation_source"] == "model"
    assert result["main_agent_explanation"] == "Monitoring triage found one mild local deviation."
    assert result["model_metadata"]["model_used"] is True


def test_optin_network_derivation_overrides_label_and_records_raw(tmp_path, monkeypatch):
    # Fix 1 (FIXES_TODO_20260709) is OPT-IN: with the flag on, network_status is
    # re-derived from the model's own sensor_reports (2 mild of 2 targets ->
    # structure-wide per deterministic_network_label); the raw head output is
    # kept in metadata for audit. Per-sensor labels stay the model's own.
    import vibroagent_mcp.llm_sensor_agent as agent_module

    monkeypatch.setattr(agent_module, "_DERIVE_NETWORK_FROM_REPORTS", True)
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)
    batch_model = _RecordingBatchModel("mild_deviation", "mild_local_deviation")

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=batch_model,
    ).run_window()

    assert [report["small_agent_label"] for report in result["sensor_agent_reports"]] == ["mild_deviation", "mild_deviation"]
    network = result["network_assessment"]
    assert network["network_label"] == "multi_sensor_structure_wide_vibration_event"
    assert network["model_metadata"]["network_status_derived_from_sensor_reports"] is True
    assert network["model_metadata"]["model_network_status_raw"] == "mild_local_deviation"


class _NetworkOnlyNormalBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "network_status": "normal_relative_to_reference_sensor",
            "confidence": 0.84,
            "affected_sensor_ids": [],
            "evidence": ["The 8-second window is normal relative to the reference sensor."],
            "explanation": "The 8-second window is normal relative to the reference sensor. No affected target sensors were identified; continue monitoring comparable windows.",
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_accepts_network_only_normal_model_output(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    config = _write_normal_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_NetworkOnlyNormalBatchModel(),
    ).run_window()

    assert [report["small_agent_label"] for report in result["sensor_agent_reports"]] == [
        "normal_relative_to_baseline",
        "normal_relative_to_baseline",
    ]
    assert result["network_assessment"]["network_label"] == "normal_relative_to_reference_sensor"
    assert result["network_assessment"]["affected_sensor_ids"] == []
    assert result["network_assessment"]["explanation_source"] == "model"
    assert result["main_agent_explanation_source"] == "model"
    assert result["model_metadata"]["model_used"] is True
    assert result["model_metadata"]["fallbacks_used"] == []


class _InsufficientDataBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "sensor_reports": [
                {"sensor_id": "target_1", "local_status": "insufficient_data", "confidence": 0.87},
                {"sensor_id": "target_2", "local_status": "insufficient_data", "confidence": 0.88},
            ],
            "network_status": "insufficient_data",
            "confidence": 0.89,
            "affected_sensor_ids": ["target_1", "target_2"],
            "evidence": ["The model claimed there was not enough data."],
            "explanation": "The model claimed insufficient data.",
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_keeps_model_labels_and_records_quality_disagreement(tmp_path, monkeypatch):
    # LLM-only policy: the deterministic quality heuristic no longer vetoes the
    # model. An unsubstantiated insufficient_data verdict is the model's
    # decision; the disagreement is recorded in metadata instead.
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_InsufficientDataBatchModel(),
    ).run_window()

    assert result["model_metadata"]["model_used"] is True
    assert result["network_assessment"]["network_label"] == "insufficient_data"
    assert result["network_assessment"]["model_metadata"]["fallback_reason"] is None
    assert result["network_assessment"]["model_metadata"]["quality_gate_disagreement"] is True
    assert result["network_assessment"]["explanation_source"] == "model"
    assert result["main_agent_explanation_source"] == "model"
    assert [report["small_agent_label"] for report in result["sensor_agent_reports"]] == [
        "insufficient_data",
        "insufficient_data",
    ]
    assert all(
        report["model_metadata"].get("quality_gate_disagreement") is True
        for report in result["sensor_agent_reports"]
    )


class _ZeroConfidenceBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "sensor_reports": [
                {"sensor_id": "target_1", "local_status": "normal_relative_to_baseline", "confidence": 0.0},
                {"sensor_id": "target_2", "local_status": "significant_local_deviation", "confidence": 0.0},
            ],
            "network_status": "localized_vibration_deviation",
            "confidence": 0.0,
            "affected_sensor_ids": ["target_2"],
            "evidence": ["The model returned a valid label but zero confidence."],
            "explanation": "The model returned zero confidence.",
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_reports_model_confidence_verbatim(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_ZeroConfidenceBatchModel(),
    ).run_window()

    # LLM-only policy: the confidence is the model's own number, reported
    # as-is (even a zero), never substituted from the deterministic precheck.
    # A zero-confidence response is still an accepted model decision — the old
    # confidence-rejection cascade must not come back.
    assert result["model_metadata"]["model_used"] is True
    assert result["network_assessment"]["model_metadata"]["fallback_reason"] is None
    assert result["network_assessment"]["explanation_source"] == "model"
    assert result["main_agent_explanation_source"] == "model"
    assert result["network_assessment"]["main_agent_confidence"] == 0.0
    assert all(report["small_agent_confidence"] == 0.0 for report in result["sensor_agent_reports"])


class _UnsafeModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "local_status": "normal_relative_to_baseline",
            "confidence": 0.99,
            "evidence": [f"This unsafe explanation mentions {sorted(OUT_OF_DOMAIN_COMPONENT_TERMS)[0]}."],
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_sensor_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_errors_on_unsafe_model_content_without_fallback(tmp_path, monkeypatch):
    # LLM-only policy: out-of-domain model output is rejected as an explicit
    # error; rule labels never impersonate the agent's decision, and the
    # rejected model text never reaches the result.
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        selected_sensor_ids=["target_1"],
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_UnsafeModel(),
        main_model_client=None,
    ).run_window()

    assert result["status"] == "error"
    assert result["error"] == "model_decision_unavailable"
    assert "No deterministic fallback answer was generated" in result["message"]
    assert result["sensor_agent_reports"] == []
    assert result["network_assessment"] is None
    assert result["model_metadata"]["model_used"] is False
    assert result["model_metadata"]["fallback_reason"] == "model_output_rejected_for_out_of_domain_language"
    assert result["baseline_metrics"]["sensor_id"] == "baseline"
    ensure_building_domain_only(result)


class _InvalidNetworkLabelBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "sensor_reports": [
                {"sensor_id": "target_1", "local_status": "normal_relative_to_baseline", "confidence": 0.7},
                {"sensor_id": "target_2", "local_status": "normal_relative_to_baseline", "confidence": 0.7},
            ],
            "network_status": "all_good",
            "confidence": 0.8,
            "affected_sensor_ids": [],
            "evidence": ["Everything within expected range."],
            "explanation": "The window is unremarkable. No affected target sensors were identified.",
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_reports_specific_reason_for_invalid_network_label(tmp_path, monkeypatch):
    # Rejection reasons must name the actual defect so a webchat error like
    # "model_output_invalid_network_label" is diagnosable without digging.
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_InvalidNetworkLabelBatchModel(),
    ).run_window()

    assert result["status"] == "error"
    assert result["error"] == "model_decision_unavailable"
    assert result["model_metadata"]["fallback_reason"] == "model_output_invalid_network_label"


class _MissingExplanationBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        data = {
            "sensor_reports": [
                {
                    "sensor_id": "target_1",
                    "local_status": "mild_deviation",
                    "confidence": 0.7,
                    "evidence": ["Slightly above reference."],
                },
                {
                    "sensor_id": "target_2",
                    "local_status": "significant_local_deviation",
                    "confidence": 0.8,
                    "evidence": ["Clearly above reference."],
                },
            ],
            "network_status": "multi_sensor_structure_wide_vibration_event",
            "confidence": 0.8,
            "affected_sensor_ids": ["target_1", "target_2"],
            "evidence": ["Two targets differ from reference."],
        }
        return ModelCallResult(
            ok=True,
            used_model=True,
            data=data,
            text=json.dumps(data),
            error=None,
            metadata={
                "role": "autonomous_network_agent",
                "model_used": True,
                "provider": "test",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        )


def test_autonomous_agent_errors_on_model_output_without_explanation(tmp_path, monkeypatch):
    # LLM-only policy: a decision without a model-authored explanation is an
    # explicit error, not a deterministic-explanation substitution.
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_MissingExplanationBatchModel(),
    ).run_window()

    assert result["status"] == "error"
    assert result["error"] == "model_decision_unavailable"
    assert result["model_metadata"]["model_used"] is False
    assert result["model_metadata"]["fallback_reason"] == "model_output_rejected_for_missing_explanation"
    assert result["sensor_agent_reports"] == []
    assert result["network_assessment"] is None


class _TimeoutBatchModel:
    def call_json_model(self, prompt, allowed_labels=None, extra_body=None):
        return ModelCallResult(
            ok=False,
            used_model=False,
            data=None,
            text=None,
            error="APITimeoutError: Request timed out.",
            metadata={
                "role": "autonomous_network_agent",
                "model_used": False,
                "provider": "openai_compatible",
                "model": "fake",
                "endpoint_configured": True,
                "fallback_reason": "APITimeoutError: Request timed out.",
            },
        )


def test_autonomous_agent_errors_instead_of_rule_fallback_when_configured_model_fails(tmp_path, monkeypatch):
    # A CONFIGURED model that fails (timeout, HTTP error) must produce an
    # explicit error result: deterministic rule labels may only appear when no
    # model exists in the deployment at all (endpoint_not_configured et al).
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)

    result = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=8,
        require_current=False,
        mode="replay",
        small_model_client=_FailingModel(),
        main_model_client=_TimeoutBatchModel(),
    ).run_window()

    assert result["status"] == "error"
    assert result["error"] == "model_decision_unavailable"
    assert "timed out" in result["model_metadata"]["fallback_reason"]
    assert result["sensor_agent_reports"] == []
    # Measurements survive so the operator can still see what was read.
    assert result["baseline_metrics"]["sensor_id"] == "baseline"
    assert len(result["sensor_metrics"]) == 2
