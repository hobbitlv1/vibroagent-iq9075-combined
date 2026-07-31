from pathlib import Path
from types import SimpleNamespace

import pytest

from vibroagent_mcp import genie_openai_server as server


def test_messages_to_prompt_formats_roles_and_tool_result():
    prompt = server.messages_to_prompt(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Analyze vibration."},
            {"role": "tool", "name": "run_building_vibration_agent_pipeline", "content": "{\"ok\": true}"},
        ]
    )

    assert "System:\nReturn JSON only." in prompt
    assert "User:\nAnalyze vibration." in prompt
    assert "Tool result (run_building_vibration_agent_pipeline):\n{\"ok\": true}" in prompt
    assert prompt.endswith("Assistant:")


def test_messages_to_prompt_supports_qwen3_chat_template():
    prompt = server.messages_to_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello."},
            {"role": "tool", "content": "tool output"},
        ],
        prompt_format="qwen3",
    )

    assert "<|im_start|>system\nYou are concise.<|im_end|>" in prompt
    assert "<|im_start|>user\nSay hello.<|im_end|>" in prompt
    assert "<|im_start|>user\ntool output<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_clean_genie_output_extracts_generated_text():
    raw = """Using libGenie.so version 1.2.3
[PROMPT]: hello

[BEGIN]: The result is normal.[END]
"""

    assert server.clean_genie_output(raw) == "The result is normal."


def test_build_runtime_env_prefers_sdk_v73_paths(tmp_path):
    sdk_root = tmp_path / "sdk"
    env = server.build_runtime_env(sdk_root, {"LD_LIBRARY_PATH": "oldld", "ADSP_LIBRARY_PATH": "oldadsp"})

    assert str(sdk_root / "lib" / server.AARCH64_TARGET) in env["LD_LIBRARY_PATH"]
    assert str(sdk_root / "lib" / "hexagon-v73" / "unsigned") in env["LD_LIBRARY_PATH"]
    assert env["LD_LIBRARY_PATH"].endswith("oldld")
    assert env["ADSP_LIBRARY_PATH"].split(":") == [
        str(sdk_root / "lib" / "hexagon-v73" / "unsigned"),
        "oldadsp",
    ]


def test_default_runner_path_uses_arm64_genie_t2t():
    sdk_root = Path("/opt/qairt")

    assert server.default_runner_path(sdk_root) == sdk_root / "bin" / server.AARCH64_TARGET / "genie-t2t-run"

def test_default_persistent_runner_path_uses_project_build_dir():
    runner = server.default_persistent_runner_path()

    assert runner.name == "genie_persistent_runner"
    assert runner.parent.name == ".build"


def test_build_metrics_uses_generation_time_for_tokens_per_second():
    metrics = server._build_metrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_s=5.0,
        generation_s=2.0,
        time_to_first_token_s=0.5,
        persistent=True,
    )

    assert metrics.total_tokens == 30
    assert metrics.tokens_per_second == 10.0
    assert metrics.to_payload()["persistent"] is True



def test_request_timeout_override_clamps_to_server_timeout():
    assert server._request_timeout_s({"timeout_s": 5}, 120.0) == 5.0
    assert server._request_timeout_s({"timeout_s": 500}, 120.0) == 120.0
    assert server._request_timeout_s({"timeout_s": "bad"}, 120.0) == 120.0


def test_prepare_bounded_genie_config_adds_output_cap_and_stops_in_same_directory(tmp_path):
    source = tmp_path / "genie_config.json"
    source.write_text(
        '{"dialog":{"version":1,"tokenizer":{"path":"tokenizer.json"}}}',
        encoding="utf-8",
    )

    runtime = server.prepare_bounded_genie_config(source, max_output_tokens=192)

    assert runtime.parent == source.parent
    assert runtime != source
    payload = __import__("json").loads(runtime.read_text(encoding="utf-8"))
    assert payload["dialog"]["max-num-tokens"] == 192
    assert payload["dialog"]["stop-sequence"] == ["<|im_end|>", "<|endoftext|>"]
    assert payload["dialog"]["tokenizer"]["path"] == "tokenizer.json"
    server._cleanup_runtime_config(runtime)


def test_prepare_bounded_genie_config_preserves_stricter_existing_limit(tmp_path):
    source = tmp_path / "genie_config.json"
    source.write_text(
        '{"dialog":{"max-num-tokens":64,"stop-sequence":["END"]}}',
        encoding="utf-8",
    )

    runtime = server.prepare_bounded_genie_config(source, max_output_tokens=256)

    assert runtime == source


def test_persistent_runner_queue_wait_is_part_of_request_timeout():
    class BusyLock:
        def __init__(self):
            self.requested_timeout = None

        def acquire(self, *, timeout):
            self.requested_timeout = timeout
            return False

        def release(self):
            raise AssertionError("unacquired lock must not be released")

    runner = server.PersistentGenieProcess.__new__(server.PersistentGenieProcess)
    runner._config = SimpleNamespace(timeout_s=120.0)
    runner._lock = BusyLock()

    with pytest.raises(TimeoutError, match="busy"):
        runner.query("hello", timeout_s=7.5)

    assert runner._lock.requested_timeout == 7.5


def _genie_config(tmp_path, *, persistent=True, timeout_s=120.0):
    config_path = tmp_path / "genie_config.json"
    config_path.write_text('{"dialog": {}}', encoding="utf-8")
    runner_path = tmp_path / "genie-t2t-run"
    runner_path.write_text("", encoding="utf-8")
    return server.GenieServerConfig(
        sdk_root=tmp_path / "sdk",
        config_path=config_path,
        runner_path=runner_path,
        model_id="test-model",
        prompt_format="qwen3",
        timeout_s=timeout_s,
        max_prompt_chars=4000,
        persistent=persistent,
        persistent_runner_path=tmp_path / "persistent-runner",
    )


def test_default_genie_output_cap_is_bounded_for_agent_json():
    assert server.DEFAULT_GENIE_MAX_OUTPUT_TOKENS == 1024


def test_run_genie_prompt_total_budget_includes_runner_startup(tmp_path, monkeypatch):
    config = _genie_config(tmp_path, persistent=True, timeout_s=120.0)
    captured = {}

    class FakeRunner:
        def query(self, prompt, *, timeout_s=None):
            captured["prompt"] = prompt
            captured["query_timeout_s"] = timeout_s
            return server.GeniePromptResult(
                content="model answer",
                metrics=server._build_metrics(
                    prompt_tokens=4,
                    completion_tokens=3,
                    total_s=1.0,
                    generation_s=0.5,
                    time_to_first_token_s=0.1,
                    persistent=True,
                ),
            )

    def fake_get_runner(received_config, *, startup_timeout_s=None):
        assert received_config is config
        captured["startup_timeout_s"] = startup_timeout_s
        return FakeRunner()

    times = iter([100.0, 102.0])
    monkeypatch.setattr(server, "_get_persistent_runner", fake_get_runner)
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = server.run_genie_prompt_result("hello", config, timeout_s=7.0)

    assert result.content == "model answer"
    assert captured["startup_timeout_s"] == pytest.approx(7.0)
    assert captured["query_timeout_s"] == pytest.approx(5.0)


def test_prewarm_loads_persistent_runner_with_full_startup_budget(tmp_path, monkeypatch):
    config = _genie_config(tmp_path, persistent=True, timeout_s=83.0)
    captured = {}

    class FakeRunner:
        def is_alive(self):
            return True

    def fake_get_runner(received_config, *, startup_timeout_s=None):
        captured["config"] = received_config
        captured["startup_timeout_s"] = startup_timeout_s
        return FakeRunner()

    monkeypatch.setattr(server, "_get_persistent_runner", fake_get_runner)

    server.prewarm_persistent_genie_runner(config)

    assert captured["config"] is config
    assert captured["startup_timeout_s"] == pytest.approx(83.0)


def test_main_prewarms_before_http_port_is_bound(tmp_path, monkeypatch):
    config = _genie_config(tmp_path, persistent=True, timeout_s=90.0)
    events = []
    args = SimpleNamespace(host="127.0.0.1", port=8910)

    class FakeHttpServer:
        def __init__(self, address, handler):
            events.append(("bind", address, handler))
            self.genie_config = None

        def serve_forever(self):
            events.append(("serve",))
            raise KeyboardInterrupt

        def server_close(self):
            events.append(("close",))

    monkeypatch.setattr(server, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(server, "build_config", lambda parsed: config)
    monkeypatch.setattr(server, "prewarm_persistent_genie_runner", lambda received: events.append(("prewarm", received)))
    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeHttpServer)
    monkeypatch.setattr(server, "shutdown_persistent_genie_runners", lambda: events.append(("shutdown",)))

    assert server.main([]) == 0

    names = [item[0] for item in events]
    assert names.index("prewarm") < names.index("bind")


def test_protocol_line_read_uses_one_decreasing_deadline(monkeypatch):
    runner = server.PersistentGenieProcess.__new__(server.PersistentGenieProcess)
    requested = []
    chunks = iter([b"R", b"\n"])

    def fake_read_exact(size, timeout_s):
        assert size == 1
        requested.append(timeout_s)
        return next(chunks)

    times = iter([10.0, 11.0, 12.0])
    runner._read_exact = fake_read_exact
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    assert runner._readline(5.0) == b"R\n"
    assert requested == pytest.approx([4.0, 3.0])


def test_persistent_query_protocol_reads_share_one_generation_budget(monkeypatch):
    writes = []

    class FakeStdin:
        def write(self, value):
            writes.append(value)

        def flush(self):
            writes.append(b"<flush>")

    class FakeProc:
        stdin = FakeStdin()

    runner = server.PersistentGenieProcess.__new__(server.PersistentGenieProcess)
    runner._proc = FakeProc()
    requested = []

    def fake_readline(timeout_s):
        requested.append(("line", timeout_s))
        return b"OK 3 2 1 1000000 500000 100000\n"

    exact_chunks = iter([b"yes", b"\n"])

    def fake_read_exact(size, timeout_s):
        requested.append((f"exact-{size}", timeout_s))
        return next(exact_chunks)

    runner._readline = fake_readline
    runner._read_exact = fake_read_exact
    times = iter([20.0, 21.0, 22.0, 23.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = runner._query_locked("hello", timeout_s=5.0)

    assert result.content == "yes"
    assert [kind for kind, _ in requested] == ["line", "exact-3", "exact-1"]
    assert [value for _, value in requested] == pytest.approx([4.0, 3.0, 2.0])
    assert writes[:3] == [b"QUERY 5\n", b"hello", b"\n"]
