import builtins
import sys
import types

import pytest

from vibroagent_mcp.model_client import ModelClient, parse_json_response


def test_parse_json_response_accepts_nested_fenced_json():
    data = parse_json_response(
        """```json
{
  "local_status": "mild_deviation",
  "confidence": 0.7,
  "details": {"ratio": 1.8, "nested": {"ok": true}},
  "evidence": ["RMS changed"]
}
```"""
    )

    assert data["local_status"] == "mild_deviation"
    assert data["details"]["nested"]["ok"] is True


def test_parse_json_response_extracts_json_from_prose():
    data = parse_json_response(
        'Here is the JSON: {"network_status": "mild_local_deviation", "confidence": 0.61} Done.'
    )

    assert data["network_status"] == "mild_local_deviation"


def test_parse_json_response_skips_non_object_json_before_object():
    data = parse_json_response(
        'Allowed labels: ["a", "b"]. Result: {"local_status": "mild_deviation", "confidence": 0.61}'
    )

    assert data["local_status"] == "mild_deviation"


def test_parse_json_response_accepts_singleton_object_array():
    data = parse_json_response('[{"local_status": "normal_relative_to_baseline", "confidence": 0.8}]')

    assert data["local_status"] == "normal_relative_to_baseline"


def test_parse_json_response_rejects_truncated_top_level_object():
    text = '{"network_status":"normal_relative_to_reference_sensor","sensor_reports":[{"sensor_id":"target_1"}'

    with pytest.raises(ValueError, match="complete top-level JSON object"):
        parse_json_response(text)


def test_parse_json_response_accepts_complete_top_level_object_with_extra_text():
    data = parse_json_response('{"network_status":"normal_relative_to_reference_sensor"}\n{"ignored":true}')

    assert data["network_status"] == "normal_relative_to_reference_sensor"


def test_parse_json_response_rejects_non_object_json():
    with pytest.raises(ValueError, match="object"):
        parse_json_response("[1, 2, 3]")


def test_model_client_falls_back_to_qwen_env(monkeypatch):
    monkeypatch.delenv("SMALL_AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("SMALL_AGENT_MODEL", raising=False)
    monkeypatch.delenv("SMALL_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen3.5-4b-instruct-revised")
    monkeypatch.setenv("QWEN_API_KEY", "EMPTY")

    client = ModelClient(
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
    )

    assert client.is_configured is True
    assert client.base_url == "http://127.0.0.1:1234/v1"
    assert client.model == "qwen3.5-4b-instruct-revised"


def test_model_client_accepts_per_call_timeout_override(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_TIMEOUT_S", "120")
    client = ModelClient(
        base_url="http://127.0.0.1:1234/v1",
        api_key="EMPTY",
        model="qwen-test",
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
        timeout_s=3,
    )

    assert client._timeout_s() == 3


def test_model_client_accepts_per_call_max_tokens_override(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_MAX_TOKENS", "512")
    client = ModelClient(
        base_url="http://127.0.0.1:1234/v1",
        api_key="EMPTY",
        model="qwen-test",
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
        max_tokens=96,
    )

    assert client._max_tokens() == 96


def test_model_client_forwards_extra_body(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured["request_kwargs"] = kwargs
            message = types.SimpleNamespace(content="{\"network_status\": \"normal_relative_to_reference_sensor\", \"confidence\": 0.9}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    client = ModelClient(
        base_url="http://127.0.0.1:8910/v1",
        api_key="EMPTY",
        model="qwen-test",
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
    )

    result = client.call_json_model(
        "prompt",
        allowed_labels={"normal_relative_to_reference_sensor"},
        extra_body={"timeout_s": 5},
    )

    assert result.ok is True
    assert captured["request_kwargs"]["extra_body"] == {"timeout_s": 5}


def test_model_client_reports_exception_group_child(monkeypatch):
    exception_group = getattr(builtins, "ExceptionGroup", None)
    if exception_group is None:
        pytest.skip("ExceptionGroup is only available on Python 3.11+")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            raise exception_group(
                "unhandled errors in a TaskGroup",
                [ConnectionError("local model adapter refused connection")],
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = ModelClient(
        base_url="http://127.0.0.1:8910/v1",
        api_key="EMPTY",
        model="qwen-test",
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
    )

    result = client.call_json_model("prompt")

    assert result.ok is False
    assert "ExceptionGroup: unhandled errors in a TaskGroup" in result.error
    assert "sub-exception 1: ConnectionError: local model adapter refused connection" in result.error
    assert result.metadata["fallback_reason"] == result.error

