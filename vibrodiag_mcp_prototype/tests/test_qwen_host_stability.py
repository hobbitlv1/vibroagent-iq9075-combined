import builtins
import asyncio
import importlib.util
import json
from types import SimpleNamespace

import pytest

from vibroagent_mcp import qwen_mcp_host
from vibroagent_mcp.errors import format_exception_for_response
from vibroagent_mcp.schemas import (
    OUT_OF_DOMAIN_COMPONENT_TERMS,
    find_out_of_domain_component_terms,
)


def test_qwen_host_imports_without_optional_clients():
    assert hasattr(qwen_mcp_host, "chat_with_mcp")
    if importlib.util.find_spec("mcp") is None:
        with pytest.raises(RuntimeError, match="MCP SDK is not installed"):
            qwen_mcp_host._load_mcp_client_symbols()


def test_qwen_env_numeric_helpers_default_and_clamp(monkeypatch):
    monkeypatch.setenv("QWEN_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("QWEN_MAX_RETRIES", "-4")

    assert qwen_mcp_host._env_float("QWEN_TIMEOUT_S", 90.0, minimum=0.1) == 90.0
    assert qwen_mcp_host._env_int("QWEN_MAX_RETRIES", 0, minimum=0) == 0

    monkeypatch.setenv("QWEN_TIMEOUT_S", "-2")
    assert qwen_mcp_host._env_float("QWEN_TIMEOUT_S", 90.0, minimum=0.1) == 0.1


def test_mcp_allowlist_contains_only_building_workflow_tools():
    assert qwen_mcp_host.ALLOWED_MCP_TOOL_NAMES == {
        "list_building_sensors",
        "read_building_sensor_window",
        "read_building_sensor_psd",
        "compare_building_sensor_psd",
        "run_autonomous_sensor_check",
        "run_building_sensor_agent",
        "run_building_vibration_agent_pipeline",
        "export_building_validation_dataset",
    }
    assert "analyze_sdk_vibrometer" not in qwen_mcp_host.ALLOWED_MCP_TOOL_NAMES
    assert "read_sdk_vibrometer_window" not in qwen_mcp_host.ALLOWED_MCP_TOOL_NAMES


def test_configuration_prompt_selects_sensor_registry_tool():
    name, args = qwen_mcp_host._infer_basic_tool_call("List configured sensors and their locations")

    assert name == "list_building_sensors"
    assert args == {"config_path": "config/sensors.live.yaml"}


def test_read_prompt_selects_registered_building_window_tool():
    name, args = qwen_mcp_host._infer_basic_tool_call(
        "Read the current 5-second target 2 window and summarize signal statistics."
    )

    assert name == "read_building_sensor_window"
    assert args["sensor_id"] == "target_2"
    assert args["require_current"] is True
    assert args["include_samples"] is False


def test_replay_read_prompt_disables_currentness_gate():
    name, args = qwen_mcp_host._infer_basic_tool_call(
        "Read the saved replay baseline window and summarize its RMS statistics."
    )

    assert name == "read_building_sensor_window"
    assert args["sensor_id"] == "baseline"
    assert args["require_current"] is False


def test_building_argument_normalization_allows_saved_replay_requests():
    args = qwen_mcp_host._normalize_tool_arguments(
        "read_building_sensor_window",
        {"require_current": "false", "sensor_id": "target_3", "axis": ""},
        "Analyze a saved replay recording from target 3",
    )

    assert args["require_current"] is False
    assert args["sensor_id"] == "target_3"
    assert "axis" not in args
    assert args["config_path"] == "config/sensors.live.yaml"


def test_building_argument_normalization_forces_current_for_latest_requests():
    args = qwen_mcp_host._normalize_tool_arguments(
        "read_building_sensor_window",
        {"require_current": False, "start_time_s": 0},
        "Read the latest target 4 window",
    )

    assert args["require_current"] is True
    assert args["sensor_id"] == "target_4"
    assert "start_time_s" not in args


def test_list_sensor_arguments_do_not_gain_currentness_fields():
    args = qwen_mcp_host._normalize_tool_arguments(
        "list_building_sensors",
        {},
        "List current sensors",
    )

    assert args == {}


def test_one_target_prompt_routes_to_single_building_sensor_agent():
    name, args = qwen_mcp_host._infer_basic_tool_call("Check target 5 against the reference")

    assert name == "run_building_sensor_agent"
    assert args["sensor_id"] == "target_5"
    assert args["baseline_sensor_id"] == "baseline"


def test_network_prompt_routes_to_autonomous_building_check():
    name, args = qwen_mcp_host._infer_basic_tool_call("Compare all sensors across the building")

    assert name == "run_autonomous_sensor_check"
    assert args["baseline_sensor_id"] == "baseline"


def test_network_prompt_preserves_explicit_seconds_duration():
    name, args = qwen_mcp_host._infer_basic_tool_call("Compare all sensors for 60 seconds")

    assert name == "run_autonomous_sensor_check"
    assert args["duration_s"] == 60.0


def test_network_prompt_parses_minute_duration():
    name, args = qwen_mcp_host._infer_basic_tool_call("Analyze the current building for 1 minute")

    assert name == "run_autonomous_sensor_check"
    assert args["duration_s"] == 60.0


def test_argument_normalization_overrides_default_duration_from_request_text():
    args = qwen_mcp_host._normalize_tool_arguments(
        "run_autonomous_sensor_check",
        {"duration_s": 10.0},
        "Analyze all target sensors for 60 seconds",
    )

    assert args["duration_s"] == 60.0


def test_requested_duration_is_clamped_to_the_tool_window_cap():
    # read_sdk_vibrometer_window rejects windows above the 60 s edge-safety cap
    # and no MCP service raises max_duration_s, so "for 2 minutes" must clamp
    # instead of making every sensor read in the turn fail validation.
    args = qwen_mcp_host._normalize_tool_arguments(
        "run_autonomous_sensor_check",
        {"duration_s": 10.0},
        "Analyze all target sensors for 2 minutes",
    )

    assert args["duration_s"] == qwen_mcp_host.MAX_TOOL_WINDOW_DURATION_S


def test_model_supplied_duration_is_clamped_to_the_tool_window_cap():
    args = qwen_mcp_host._normalize_tool_arguments(
        "run_autonomous_sensor_check",
        {"duration_s": 240.0},
        "Analyze all target sensors",
    )

    assert args["duration_s"] == qwen_mcp_host.MAX_TOOL_WINDOW_DURATION_S


def test_supported_fallback_uses_available_building_tool():
    name, args = qwen_mcp_host._supported_inferred_tool_call(
        "Compare building vibration across floors",
        {"run_building_vibration_agent_pipeline"},
    )

    assert name == "run_building_vibration_agent_pipeline"
    assert args["baseline_sensor_id"] == "baseline"


def test_supported_fallback_ignores_non_building_tools():
    with pytest.raises(RuntimeError, match="No supported building-monitoring MCP tools"):
        qwen_mcp_host._supported_inferred_tool_call(
            "Analyze vibration",
            {"analyze_sdk_vibrometer", "read_sdk_vibrometer_window"},
        )


def test_tool_summary_reports_registered_sensor_read_failure_directly():
    summary = qwen_mcp_host._summarize_tool_results(
        [
            {
                "name": "read_building_sensor_window",
                "result": {
                    "analysis_domain": "building_vibration_relative_monitoring",
                    "status": "error",
                    "error": "sdk_read_failed",
                    "message": "RuntimeError: board reader unavailable",
                },
            }
        ]
    )

    assert summary == "RuntimeError: board reader unavailable"


def test_tool_summary_removes_out_of_domain_agent_explanation():
    blocked_term = sorted(OUT_OF_DOMAIN_COMPONENT_TERMS)[0]
    summary = qwen_mcp_host._summarize_tool_results(
        [
            {
                "name": "run_autonomous_sensor_check",
                "result": {
                    "analysis_domain": "building_vibration_relative_monitoring",
                    "baseline_sensor_id": "baseline",
                    "sensor_agent_reports": [{"sensor_id": "target_1"}],
                    "network_assessment": {
                        "network_label": "localized_vibration_deviation",
                        "main_agent_confidence": 0.91,
                        "affected_sensor_ids": ["target_1"],
                        "explanation": f"Unsupported explanation containing {blocked_term}.",
                    },
                },
            }
        ]
    )

    assert not find_out_of_domain_component_terms(summary)
    assert "Repeat the synchronized measurement" in summary


def _fake_mcp_symbols(result_payload, *, tool_names=("read_building_sensor_window",), params_capture=None):
    tools = [
        SimpleNamespace(
            name=name,
            description=f"Test tool {name}",
            inputSchema={"type": "object", "properties": {}},
        )
        for name in tool_names
    ]

    class Session:
        def __init__(self, read, write):
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=tools)

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return SimpleNamespace(structuredContent=result_payload, content=[])

    class StdioContext:
        async def __aenter__(self):
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def stdio_client(params):
        return StdioContext()

    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            if params_capture is not None:
                params_capture.update(kwargs)

    return Session, Params, stdio_client


def _fake_openai_client(*, answer=None, error=None):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            message = SimpleNamespace(content=answer, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    return client, calls


def _read_payload():
    return {
        "analysis_domain": "building_vibration_relative_monitoring",
        "status": "ok",
        "sensor_id": "baseline",
        "location": "reference_position",
        "sample_count": 100,
        "sampling_rate_hz": 100,
        "window_duration_s": 1.0,
        "rms": 0.012,
        "metadata": {"data_is_current": False, "read_mode": "saved"},
    }


def test_chat_with_mcp_sends_read_result_to_model_once(monkeypatch):
    client, calls = _fake_openai_client(answer="The saved baseline window has an RMS acceleration of 0.012 g.")
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen_mcp_host, "_load_mcp_client_symbols", lambda: _fake_mcp_symbols(_read_payload()))

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Read the saved replay baseline window.",
            agentic=False,
            base_url="http://127.0.0.1:18181/v1",
            model="test-model",
            timeout_s=42.0,
            model_tool_routing=False,
        )
    )

    assert result["routing_mode"] == "deterministic"
    assert result["model_call_count"] == 1
    assert result["model_request_sent"] is True
    assert result["model_response_used"] is True
    assert result["used_host_building_summary"] is False
    assert len(calls) == 1
    assert result["mcp_calls"][0]["name"] == "read_building_sensor_window"
    assert "saved baseline window" in result["answer"]
    assert "0.012 g" in result["answer"]
    assert not find_out_of_domain_component_terms(result["answer"])


def test_chat_with_mcp_redacts_rejected_component_language_from_full_payload(monkeypatch):
    blocked_term = sorted(OUT_OF_DOMAIN_COMPONENT_TERMS)[0]
    unsafe_result = {
        "analysis_domain": "building_vibration_relative_monitoring",
        "status": "error",
        "error": "sdk_read_failed",
        "message": f"Unsafe generated text: {blocked_term}.",
    }
    client, calls = _fake_openai_client(answer="The selected building sensor could not be read.")
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen_mcp_host, "_load_mcp_client_symbols", lambda: _fake_mcp_symbols(unsafe_result))

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Read the saved replay baseline window.",
            agentic=False,
            model="test-model",
            model_tool_routing=False,
        )
    )

    assert len(calls) == 1
    prompt_text = "\n".join(message["content"] for message in calls[0]["messages"])
    assert blocked_term not in prompt_text.lower()
    assert "redacted_for_building_monitoring" in prompt_text
    assert result["domain_guard_triggered"] is True
    assert result["domain_guard_rejected_term_count"] == 1
    assert "domain_guard_rejected_terms" not in result
    assert blocked_term not in json.dumps(result).lower()
    assert not find_out_of_domain_component_terms(result)


def test_chat_with_mcp_filters_server_tools_to_building_allowlist(monkeypatch):
    client, calls = _fake_openai_client(answer="Building sensor read complete.")
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(
            _read_payload(),
            tool_names=("analyze_sdk_vibrometer", "read_sdk_vibrometer_window", "read_building_sensor_window"),
        ),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Read the saved replay baseline window.",
            agentic=False,
            model="test-model",
            model_tool_routing=False,
        )
    )

    assert len(calls) == 1
    assert result["tools_available"] == ["read_building_sensor_window"]
    assert result["mcp_calls"][0]["name"] == "read_building_sensor_window"


def test_building_agent_explanation_is_not_sent_through_model_twice(monkeypatch):
    result_payload = {
        "status": "ok",
        "analysis_domain": "building_vibration_relative_monitoring",
        "window_id": "window-1",
        "baseline_sensor_id": "baseline",
        "sensor_agent_reports": [{"sensor_id": "target_1"}],
        "network_assessment": {
            "network_label": "localized_vibration_deviation",
            "main_agent_confidence": 0.91,
            "affected_sensor_ids": ["target_1"],
            "explanation": "Target 1 differs from the reference while the other sensors remain consistent.",
            "explanation_source": "model",
        },
        "main_agent_explanation": "Target 1 differs from the reference while the other sensors remain consistent.",
        "main_agent_explanation_source": "model",
        "model_metadata": {
            "model_used": True,
            "network_model_metadata": {
                "model_used": True,
                "provider": "openai_compatible",
                "endpoint_configured": True,
                "fallback_reason": None,
            },
        },
    }
    client, calls = _fake_openai_client(error=AssertionError("a second model call must not be made"))
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(result_payload, tool_names=("run_autonomous_sensor_check",)),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Check the building sensor network against the reference.",
            agentic=False,
            model="test-model",
            model_tool_routing=False,
        )
    )

    assert calls == []
    assert result["model_call_count"] == 0
    assert result["tool_model_attempted"] is True
    assert result["tool_model_used"] is True
    assert result["model_request_sent"] is True
    assert result["model_response_used"] is True
    assert result["used_tool_agent_explanation"] is True
    assert result["model_explanation_only"] is True
    assert result["fallbacks_disabled"] is True
    assert result["answer_source"] == "tool_agent_model"
    assert result["answer"] == "Target 1 differs from the reference while the other sensors remain consistent."
    assert "localized_vibration_deviation" not in result["answer"]
    assert not find_out_of_domain_component_terms(result["answer"])



def test_agent_model_timeout_is_reported_without_deterministic_answer(monkeypatch):
    result_payload = {
        "status": "ok",
        "analysis_domain": "building_vibration_relative_monitoring",
        "network_assessment": {
            "network_label": "normal_building_response",
            "explanation": "Deterministic fallback text must not be shown.",
            "explanation_source": "deterministic_fallback",
            "model_metadata": {
                "model_used": False,
                "provider": "openai_compatible",
                "endpoint_configured": True,
                "fallback_reason": "TimeoutError: GenieX generation timed out",
            },
        },
        "main_agent_explanation": "Deterministic fallback text must not be shown.",
        "main_agent_explanation_source": "deterministic_fallback",
        "model_metadata": {
            "model_used": False,
            "fallbacks_used": ["TimeoutError: GenieX generation timed out"],
            "network_model_metadata": {
                "model_used": False,
                "provider": "openai_compatible",
                "endpoint_configured": True,
                "fallback_reason": "TimeoutError: GenieX generation timed out",
            },
        },
    }
    client, calls = _fake_openai_client(error=AssertionError("agent failure must not trigger a second model call"))
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(result_payload, tool_names=("run_autonomous_sensor_check",)),
    )

    with pytest.raises(qwen_mcp_host.ModelExplanationTimeoutError, match="No fallback answer"):
        asyncio.run(
            qwen_mcp_host.chat_with_mcp(
                "Check the building sensor network.",
                agentic=False,
                model="test-model",
                model_tool_routing=False,
            )
        )

    assert calls == []


def test_read_result_model_timeout_is_reported_without_host_summary(monkeypatch):
    client, calls = _fake_openai_client(error=TimeoutError("request timed out"))
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen_mcp_host, "_load_mcp_client_symbols", lambda: _fake_mcp_symbols(_read_payload()))

    with pytest.raises(qwen_mcp_host.ModelExplanationTimeoutError, match="No fallback answer"):
        asyncio.run(
            qwen_mcp_host.chat_with_mcp(
                "Read the saved replay baseline window.",
                agentic=False,
                model="test-model",
                model_tool_routing=False,
            )
        )

    assert len(calls) == 1


def test_read_result_empty_model_answer_is_rejected_without_fallback(monkeypatch):
    client, calls = _fake_openai_client(answer="   ")
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen_mcp_host, "_load_mcp_client_symbols", lambda: _fake_mcp_symbols(_read_payload()))

    with pytest.raises(qwen_mcp_host.ModelExplanationError, match="empty explanation"):
        asyncio.run(
            qwen_mcp_host.chat_with_mcp(
                "Read the saved replay baseline window.",
                agentic=False,
                model="test-model",
                model_tool_routing=False,
            )
        )

    assert len(calls) == 1

def test_chat_with_mcp_forwards_browser_model_settings_to_agent_subprocess(monkeypatch):
    captured = {}
    client, calls = _fake_openai_client(answer="Building sensor read complete.")
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_read_payload(), params_capture=captured),
    )

    asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Read the saved replay baseline window.",
            agentic=False,
            base_url="http://127.0.0.1:9999/v1",
            api_key="test-key",
            model="selected-model",
            timeout_s=37.0,
            model_tool_routing=False,
        )
    )

    env = captured["env"]
    assert env["QWEN_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["QWEN_API_KEY"] == "test-key"
    assert env["QWEN_MODEL"] == "selected-model"
    assert env["SMALL_AGENT_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["MAIN_AGENT_MODEL"] == "selected-model"
    assert env["AGENT_MODEL_TIMEOUT_S"] == "37.0"
    assert len(calls) == 1


def test_final_prompt_compacts_large_arrays_within_budget():
    client, calls = _fake_openai_client(answer="ok")
    huge_result = {
        "analysis_domain": "building_vibration_relative_monitoring",
        "status": "ok",
        "sensor_id": "baseline",
        "sample_count": 100_000,
        "samples_g": list(range(50_000)),
        "rms": 0.01,
    }

    answer, prompt_chars = qwen_mcp_host._request_final_answer(
        client=client,
        model="test-model",
        user_message="Explain this building-sensor result.",
        tool_results=[{"name": "read_building_sensor_window", "arguments": {}, "result": huge_result}],
        temperature=0.2,
        timeout_s=30.0,
        max_tokens=128,
        max_prompt_chars=3000,
    )

    assert answer == "ok"
    assert prompt_chars <= 3000
    assert len(calls) == 1
    prompt_text = "\n".join(item["content"] for item in calls[0]["messages"])
    assert "omitted_values" in prompt_text
    assert len(prompt_text) <= 3000
    assert not find_out_of_domain_component_terms(prompt_text)


def test_exception_formatter_unwraps_taskgroup_exception_group():
    exception_group = getattr(builtins, "ExceptionGroup", None)
    if exception_group is None:
        pytest.skip("ExceptionGroup is only available on Python 3.11+")

    exc = exception_group(
        "unhandled errors in a TaskGroup",
        [RuntimeError("board reader process exited before returning data")],
    )

    message = format_exception_for_response(exc)

    assert "ExceptionGroup: unhandled errors in a TaskGroup" in message
    assert "sub-exception 1: RuntimeError: board reader process exited before returning data" in message


def test_exception_formatter_includes_cause_chain():
    exc = RuntimeError("wrapper")
    exc.__cause__ = ValueError("invalid currentness gate")

    message = format_exception_for_response(exc)

    assert message == "RuntimeError: wrapper; caused by: ValueError: invalid currentness gate"



def test_extract_model_explanation_error_unwraps_nested_task_groups():
    # anyio task groups in the MCP stdio client wrap body exceptions in
    # ExceptionGroups; the webchat's typed handlers must still see the real
    # model error (and prefer the timeout subclass when present).
    inner = qwen_mcp_host.ModelExplanationError("boom")
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])
    assert qwen_mcp_host._extract_model_explanation_error(nested) is inner

    timeout = qwen_mcp_host.ModelExplanationTimeoutError("slow")
    mixed = ExceptionGroup("outer", [ExceptionGroup("inner", [inner]), timeout])
    assert qwen_mcp_host._extract_model_explanation_error(mixed) is timeout

    assert qwen_mcp_host._extract_model_explanation_error(ValueError("x")) is None


def _fake_openai_client_script(responses):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            index = min(len(calls) - 1, len(responses) - 1)
            message = SimpleNamespace(content=responses[index], tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions())), calls


def _network_check_payload():
    return {
        "analysis_domain": "building_vibration_relative_monitoring",
        "status": "ok",
        "window_id": "w1",
        "baseline_sensor_id": "baseline",
        "network_assessment": {
            "network_label": "mild_local_deviation",
            "main_agent_confidence": 0.7,
            "affected_sensor_ids": ["target_3"],
            "explanation": "One mild local deviation at target_3 relative to the reference sensor.",
            "explanation_source": "model",
            "model_metadata": {
                "model_used": True,
                "endpoint_configured": True,
                "provider": "openai_compatible",
                "fallback_reason": None,
            },
        },
        "main_agent_explanation": "One mild local deviation at target_3 relative to the reference sensor.",
        "main_agent_explanation_source": "model",
        "model_metadata": {
            "model_used": True,
            "endpoint_configured": True,
            "provider": "openai_compatible",
            "fallback_reason": None,
            "fallbacks_used": [],
        },
    }


_AGENTIC_TOOLS = (
    "list_building_sensors",
    "read_building_sensor_window",
    "read_building_sensor_psd",
    "run_autonomous_sensor_check",
    "run_building_sensor_agent",
)


def test_first_json_object_takes_first_balanced_object_and_ignores_junk():
    # Replicates the on-device behavior: correct first action object, then
    # repeated/malformed junk because stop discipline is weak.
    text = '{"action":"tool","tool":"check_all_sensors","args":{}}\n```json\n{"error":,"message":"x"}\n```'
    action = qwen_mcp_host._first_json_object(text)
    assert action == {"action": "tool", "tool": "check_all_sensors", "args": {}}

    repeated = '{"action":"tool","tool":"list_sensors","args":{}}\n{"action":"tool","tool":"list_sensors","args":{}}'
    assert qwen_mcp_host._first_json_object(repeated)["tool"] == "list_sensors"

    with pytest.raises(ValueError):
        qwen_mcp_host._first_json_object("no json here")


def test_agentic_chat_model_routes_tool_and_authors_answer(monkeypatch):
    steps = [
        '{"action":"tool","tool":"check_all_sensors","args":{"duration_s":10}}\n{"action":"tool","tool":"check_all_sensors","args":{}}',
        '{"action":"final","answer":"One target sensor shows a mild deviation relative to the reference sensor. Keep monitoring target 3."}',
    ]
    client, calls = _fake_openai_client_script(steps)
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp(
            "Compare all building sensors with the reference now.",
            agentic=True,
            model="test-model",
            timeout_s=42.0,
        )
    )

    assert result["routing_mode"] == "agentic"
    assert result["answer_source"] == "agentic_model"
    assert result["model_call_count"] == 2
    assert [step["kind"] for step in result["agent_steps"]] == ["tool", "final"]
    assert result["mcp_calls"][0]["name"] == "run_autonomous_sensor_check"
    assert result["mcp_calls"][0]["source"] == "agentic_model"
    assert result["tool_model_attempted"] is True and result["tool_model_used"] is True
    assert "mild deviation" in result["answer"]
    assert result["fallbacks_disabled"] is True
    assert len(calls) == 2
    # Every agent step must carry the decoder-level action constraint so a
    # grammar-capable backend (GenieX) cannot emit a malformed step.
    for call in calls:
        response_format = call["extra_body"]["response_format"]
        assert response_format is qwen_mcp_host.AGENT_ACTION_RESPONSE_FORMAT
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"]["properties"]["action"]["enum"] == ["tool", "final"]


def test_agentic_chat_recovers_from_one_protocol_violation(monkeypatch):
    steps = [
        "I think I should check the sensors first.",
        '{"action":"tool","tool":"check_all_sensors","args":{}}',
        '{"action":"final","answer":"One target sensor shows a mild deviation relative to the reference sensor."}',
    ]
    client, calls = _fake_openai_client_script(steps)
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp("Check the building now.", agentic=True, model="test-model")
    )

    assert [step["kind"] for step in result["agent_steps"]] == ["protocol_correction", "tool", "final"]
    assert result["answer"].startswith("One target sensor")


def test_agentic_chat_errors_after_persistent_protocol_violations(monkeypatch):
    client, _calls = _fake_openai_client_script(["junk", "more junk", "still junk"])
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    with pytest.raises(qwen_mcp_host.ModelExplanationError, match="did not follow the agent step protocol"):
        asyncio.run(qwen_mcp_host.chat_with_mcp("Check the building.", agentic=True, model="test-model"))


def test_agentic_chat_forces_final_at_step_limit(monkeypatch):
    monkeypatch.setenv("QWEN_AGENTIC_MAX_STEPS", "1")
    steps = [
        '{"action":"tool","tool":"check_all_sensors","args":{}}',
        '{"action":"final","answer":"Only one checked window was available; target 3 shows a mild deviation relative to the reference sensor."}',
    ]
    client, calls = _fake_openai_client_script(steps)
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp("Check everything thoroughly.", agentic=True, model="test-model")
    )

    assert result["max_steps"] == 1
    assert [step["kind"] for step in result["agent_steps"]] == ["tool", "final"]
    assert len(calls) == 2


def test_agentic_chat_reports_unknown_tool_to_the_model(monkeypatch):
    steps = [
        '{"action":"tool","tool":"run_fft_magic","args":{}}',
        '{"action":"final","answer":"I cannot run that tool; the available checks cover sensors relative to the reference sensor."}',
    ]
    client, _calls = _fake_openai_client_script(steps)
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp("Run an FFT magic scan.", agentic=True, model="test-model")
    )

    assert [step["kind"] for step in result["agent_steps"]] == ["unknown_tool", "final"]
    assert result["mcp_calls"] == []


def test_agentic_chat_can_use_the_psd_summary_tool(monkeypatch):
    steps = [
        '{"action":"tool","tool":"psd_summary","args":{"sensor_id":"baseline","duration_s":10}}',
        '{"action":"final","answer":"The reference sensor spectrum is dominated by a low-frequency peak; no impact ring-down was detected."}',
    ]
    client, _calls = _fake_openai_client_script(steps)
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(
        qwen_mcp_host,
        "_load_mcp_client_symbols",
        lambda: _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS),
    )

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp("What does the baseline spectrum look like?", agentic=True, model="test-model")
    )

    assert result["mcp_calls"][0]["name"] == "read_building_sensor_psd"
    assert result["agent_steps"][0]["tool"] == "psd_summary"
    assert "spectrum" in result["answer"]


def test_agentic_chat_reuses_observation_for_identical_repeated_calls(monkeypatch):
    steps = [
        '{"action":"tool","tool":"psd_summary","args":{"sensor_id":"baseline","duration_s":10}}',
        '{"action":"tool","tool":"psd_summary","args":{"sensor_id":"baseline","duration_s":10}}',
        '{"action":"final","answer":"The reference sensor spectrum is dominated by one low-frequency peak."}',
    ]
    client, _calls = _fake_openai_client_script(steps)
    call_counter = {"count": 0}

    symbols = _fake_mcp_symbols(_network_check_payload(), tool_names=_AGENTIC_TOOLS)
    Session, Params, stdio_client = symbols
    original_call_tool = Session.call_tool

    async def counting_call_tool(self, name, arguments):
        call_counter["count"] += 1
        return await original_call_tool(self, name, arguments)

    Session.call_tool = counting_call_tool
    monkeypatch.setattr(qwen_mcp_host, "_load_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(qwen_mcp_host, "_load_mcp_client_symbols", lambda: (Session, Params, stdio_client))

    result = asyncio.run(
        qwen_mcp_host.chat_with_mcp("Spectrum of the baseline?", agentic=True, model="test-model")
    )

    # Two identical tool steps, but only ONE real MCP execution.
    assert call_counter["count"] == 1
    assert [step.get("repeated") for step in result["agent_steps"] if step["kind"] == "tool"] == [False, True]
    assert result["answer"].startswith("The reference sensor spectrum")
