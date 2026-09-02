from scripts.check_vibrogemma_health import (
    EXPECTED_CONTRACT,
    EXPECTED_DECISION_MODE,
    EXPECTED_LABEL_CONTRACT,
    validate_health,
)
from scripts.check_vibrogemma_webchat_health import validate_health as validate_webchat_health


def _model_health():
    return {
        "status": "ok",
        "deployment_contract_version": EXPECTED_CONTRACT,
        "model": "model.gguf",
        "model_sha256": "a" * 64,
        "bundle_manifest_sha256": "b" * 64,
        "geniex_version": "0.4.0",
        "device_map": "llama_cpp:HTP0",
        "n_ctx": 4096,
        "external_embeddings": True,
        "decision_mode": EXPECTED_DECISION_MODE,
        "label_contract": EXPECTED_LABEL_CONTRACT,
        "runtime_probe": {
            "passed": True,
            "external_embedding_decode": True,
            "state_logits": True,
        },
    }


def test_model_health_contract_rejects_stale_or_wrong_server():
    payload = _model_health()
    assert validate_health(
        payload,
        model_name="model.gguf",
        model_sha256="a" * 64,
        manifest_sha256="b" * 64,
        device_map="llama_cpp:HTP0",
        n_ctx=4096,
    ) == []
    payload["deployment_contract_version"] = "old-server"
    errors = validate_health(
        payload,
        model_name="model.gguf",
        model_sha256="a" * 64,
        manifest_sha256="b" * 64,
        device_map="llama_cpp:HTP0",
        n_ctx=4096,
    )
    assert any("deployment_contract_version mismatch" in error for error in errors)


def test_model_health_contract_accepts_factorized_runtime_probe():
    mode = "factorized_global_plus_independent_targets_v1"
    payload = _model_health()
    payload["decision_mode"] = mode
    payload["runtime_probe"] = {
        "passed": True,
        "external_embedding_decode": True,
        "state_logits": False,
        "generate_from_state": True,
    }
    assert validate_health(
        payload,
        model_name="model.gguf",
        model_sha256="a" * 64,
        manifest_sha256="b" * 64,
        device_map="llama_cpp:HTP0",
        n_ctx=4096,
        decision_mode=mode,
    ) == []


def test_webchat_health_contract_requires_fail_closed_policy():
    payload = {
        "ok": True,
        "app": "vibroagent",
        "inference": "gemma",
        "deployment_contract_version": "vibroagent-gemma-web-live-v2",
        "failure_policy": "data_invalid_never_inherited_normal",
        "monitor_decision_mode": "vibrogemma",
        "model_base_url": "http://127.0.0.1:18181/v1",
        "model": "vibrogemma",
    }
    assert validate_webchat_health(
        payload,
        model_base_url="http://127.0.0.1:18181/v1",
        model="vibrogemma",
    ) == []
    payload["failure_policy"] = "legacy_normal_fallback"
    assert validate_webchat_health(
        payload,
        model_base_url="http://127.0.0.1:18181/v1",
        model="vibrogemma",
    )


def test_webchat_health_contract_rejects_inherited_legacy_routing():
    payload = {
        **{
            "ok": True,
            "app": "vibroagent",
            "inference": "gemma",
            "deployment_contract_version": "vibroagent-gemma-web-live-v2",
            "failure_policy": "data_invalid_never_inherited_normal",
        },
        "monitor_decision_mode": "codes",
        "model_base_url": "http://127.0.0.1:8910/v1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
    }
    errors = validate_webchat_health(
        payload,
        model_base_url="http://127.0.0.1:18181/v1",
        model="vibrogemma",
    )
    assert any("monitor_decision_mode mismatch" in error for error in errors)
    assert any("model_base_url mismatch" in error for error in errors)
    assert any("model mismatch" in error for error in errors)


def test_launcher_pins_gemma_monitor_to_verified_local_server():
    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[2] / "vibroagent.sh").read_text(
        encoding="utf-8"
    )
    assert 'export QWEN_BASE_URL="http://127.0.0.1:$GENIEX_PORT/v1"' in launcher
    assert 'export MAIN_AGENT_BASE_URL="$QWEN_BASE_URL"' in launcher
    assert 'export VIBRO_MONITOR_DECISION_MODE="vibrogemma"' in launcher
    assert 'export MAIN_AGENT_MODEL="$local_model"' in launcher
    assert '--model-base-url "http://127.0.0.1:$GENIEX_PORT/v1"' in launcher
