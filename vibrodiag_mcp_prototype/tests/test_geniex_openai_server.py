import base64
import json
import os
import re
from types import SimpleNamespace

import pytest

from vibroagent_mcp import geniex_openai_server as server

PROBE_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "proof": {"type": "string", "enum": ["GRAMMAR-OK-7391"]},
        "action": {"type": "string", "enum": ["tool", "final"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "answer": {"type": "string"},
    },
    "required": ["proof", "action"],
    "additionalProperties": False,
}


def _parse_rules(grammar: str) -> dict[str, str]:
    rules = {}
    for line in grammar.splitlines():
        name, sep, body = line.partition(" ::= ")
        assert sep, f"malformed grammar line: {line!r}"
        rules[name] = body
    return rules


def _referenced_rules(body: str) -> set[str]:
    cleaned = re.sub(r'"(\\.|[^"\\])*"', " ", body)
    cleaned = re.sub(r"\[(\\.|[^\]\\])*\]", " ", cleaned)
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", cleaned))


def test_schema_to_gbnf_compiles_probe_style_action_schema():
    grammar = server._json_schema_to_gbnf(PROBE_STYLE_SCHEMA)
    rules = _parse_rules(grammar)

    assert "root" in rules
    # Required properties are fixed in order; the sentinel enum is a literal.
    assert '\\"proof\\"' in rules["root"]
    assert '\\"action\\"' in rules["root"]
    assert '"\\"GRAMMAR-OK-7391\\""' in grammar
    assert '"\\"tool\\"" | "\\"final\\""' in grammar
    # Optional properties become the rest-chain rules.
    assert "root-o0" in rules and "root-o1" in rules and "root-o2" in rules
    # The bare {"type": "object"} args property pulls in the generic object.
    assert "object" in rules and "value" in rules

    # Every rule referenced anywhere must be defined (structural validity).
    defined = set(rules)
    for name, body in rules.items():
        missing = _referenced_rules(body) - defined
        assert not missing, f"rule {name} references undefined rules: {missing}"


def test_schema_to_gbnf_supports_const_arrays_and_numbers():
    grammar = server._json_schema_to_gbnf(
        {
            "type": "object",
            "properties": {
                "kind": {"const": "psd"},
                "peaks": {"type": "array", "items": {"type": "number"}},
                "count": {"type": "integer"},
                "ok": {"type": "boolean"},
            },
            "required": ["kind", "peaks", "count", "ok"],
        }
    )
    rules = _parse_rules(grammar)
    assert '"\\"psd\\"" space' in grammar
    assert "number" in rules and "integer" in rules and "boolean" in rules
    defined = set(rules)
    for name, body in rules.items():
        assert not _referenced_rules(body) - defined, name


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
        {"type": "object", "properties": {"a": {"$ref": "#/defs/a"}}},
        {"type": "string", "pattern": "^x"},
        {"type": "frobnicate"},
    ],
)
def test_schema_to_gbnf_rejects_unsupported_constructs(schema):
    with pytest.raises(server._SchemaGrammarError):
        server._json_schema_to_gbnf(schema)


def test_extract_json_schema_handles_openai_and_bare_shapes():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert server._extract_json_schema({"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}) == schema
    assert server._extract_json_schema({"type": "json_schema", "json_schema": schema}) == schema
    with pytest.raises(server._SchemaGrammarError):
        server._extract_json_schema({"type": "json_schema"})


class _FakeModel:
    def __init__(
        self,
        text='{"action":"final","answer":"ok"}',
        stop_reason="stop",
        prompt_tokens=42,
        generated_tokens=7,
    ):
        self.generate_calls = []
        self.reset_count = 0
        self._text = text
        self._stop_reason = stop_reason
        self._prompt_tokens = prompt_tokens
        self._generated_tokens = generated_tokens

    def _apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tools):
        assert add_generation_prompt is True and enable_thinking is False and tools is None
        return "\n".join(message["content"] for message in messages) + "\nassistant:"

    def reset(self):
        self.reset_count += 1

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, kwargs))
        return SimpleNamespace(
            text=self._text,
            profile=SimpleNamespace(
                ttft=2500,
                prompt_tokens=self._prompt_tokens,
                generated_tokens=self._generated_tokens,
                prefill_speed=250.0,
                decode_speed=11.0,
                stop_reason=self._stop_reason,
                backend="llama_cpp",
                device="HTP0",
            ),
        )


def _runtime(fake: _FakeModel, **overrides) -> server._GenieXRuntime:
    settings = {
        "model_ref": "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0",
        "device_map": "npu",
        "n_ctx": 6144,
        "max_output_tokens": 256,
        "timeout_s": 30.0,
        "max_prompt_chars": 4000,
    }
    settings.update(overrides)
    runtime = server._GenieXRuntime(**settings)
    runtime._model = fake
    runtime.loaded_backend = "llama_cpp"
    runtime.loaded_device = "HTP0"
    # Disable the real-/proc/meminfo reload guard by default so unit tests don't
    # depend on the host's free memory; the dedicated guard test re-enables it.
    runtime.min_mem_available_bytes = 0
    return runtime


def _request(**extra):
    return {"messages": [{"role": "user", "content": "check the sensors"}], **extra}


def test_chat_compiles_json_schema_into_grammar():
    fake = _FakeModel()
    runtime = _runtime(fake)
    response = runtime.chat(
        _request(response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": PROBE_STYLE_SCHEMA}})
    )

    _prompt, kwargs = fake.generate_calls[0]
    assert kwargs["json_mode"] is False
    assert "GRAMMAR-OK-7391" in kwargs["grammar"]
    metrics = response["geniex_metrics"]
    assert metrics["grammar_constrained"] is True
    assert metrics["grammar_source"] == "json_schema"
    assert metrics["json_schema_fallback"] is None
    assert fake.reset_count == 1


def test_chat_degrades_uncompilable_schema_to_json_mode_with_flag():
    fake = _FakeModel()
    runtime = _runtime(fake)
    response = runtime.chat(
        _request(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": {"anyOf": [{"type": "string"}]}},
            }
        )
    )

    _prompt, kwargs = fake.generate_calls[0]
    assert kwargs["grammar"] is None
    assert kwargs["json_mode"] is True
    metrics = response["geniex_metrics"]
    assert metrics["grammar_constrained"] is False
    assert "anyOf" in metrics["json_schema_fallback"]


def test_chat_prefers_explicit_grammar_over_response_format():
    fake = _FakeModel()
    runtime = _runtime(fake)
    response = runtime.chat(
        _request(grammar='root ::= "x"', response_format={"type": "json_object"})
    )

    _prompt, kwargs = fake.generate_calls[0]
    assert kwargs["grammar"] == 'root ::= "x"'
    assert kwargs["json_mode"] is False
    assert response["geniex_metrics"]["grammar_source"] == "request_grammar"


def test_chat_rejects_context_length_truncation_instead_of_answering():
    runtime = _runtime(_FakeModel(stop_reason="context_length"))
    with pytest.raises(ValueError, match="n_ctx=6144"):
        runtime.chat(_request())


def test_chat_rejects_silent_context_shift_by_token_accounting():
    # llama.cpp reports success (eos) after evicting the oldest KV entries;
    # the reported token counts are the only evidence the window overflowed.
    runtime = _runtime(_FakeModel(stop_reason="eos", prompt_tokens=11895, generated_tokens=4))
    with pytest.raises(ValueError, match="overflowed n_ctx=6144"):
        runtime.chat(_request())

    # Exactly filling the window is not an overflow.
    runtime = _runtime(_FakeModel(stop_reason="eos", prompt_tokens=6100, generated_tokens=44))
    assert runtime.chat(_request())["usage"]["total_tokens"] == 6144

    # Without reported counts there is no evidence; the request stands.
    runtime = _runtime(_FakeModel(stop_reason="eos", prompt_tokens=0, generated_tokens=0))
    assert runtime.chat(_request())["choices"][0]["finish_reason"] == "stop"


def test_chat_maps_stop_reasons_and_converts_ttft_to_ms():
    response = _runtime(_FakeModel(stop_reason="length")).chat(_request(max_tokens=16))
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["geniex_metrics"]["ttft_ms"] == 2.5
    assert response["usage"] == {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}

    response = _runtime(_FakeModel(stop_reason="stop")).chat(_request())
    assert response["choices"][0]["finish_reason"] == "stop"


def test_chat_caps_max_tokens_and_rejects_oversized_prompts():
    fake = _FakeModel()
    runtime = _runtime(fake)
    runtime.chat(_request(max_tokens=100000))
    assert fake.generate_calls[0][1]["max_new_tokens"] == 256

    with pytest.raises(ValueError, match="GENIEX_MAX_PROMPT_CHARS"):
        _runtime(_FakeModel(), max_prompt_chars=10).chat(_request())


def test_chat_returns_valid_openai_shape():
    response = _runtime(_FakeModel()).chat(_request())
    assert response["object"] == "chat.completion"
    content = response["choices"][0]["message"]["content"]
    assert json.loads(content)["action"] == "final"


def test_stuck_generation_flags_only_overdue_native_sections():
    import time

    runtime = _runtime(_FakeModel())
    # No generation in flight.
    assert runtime.stuck_generation(10.0) is None
    # In flight but still young.
    runtime._generation_started_mono = time.monotonic() - 1.0
    assert runtime.stuck_generation(10.0) is None
    # Overdue: report age and label.
    runtime._generation_started_mono = time.monotonic() - 11.0
    runtime._generation_label = "max_tokens=48 grammar=True temperature=0.0"
    stuck = runtime.stuck_generation(10.0)
    assert stuck is not None
    age_s, label = stuck
    assert age_s >= 11.0
    assert "max_tokens=48" in label
    # A non-positive limit disables the watchdog.
    assert runtime.stuck_generation(0.0) is None


def test_chat_clears_generation_marker_on_success_and_failure():
    runtime = _runtime(_FakeModel())
    runtime.chat(_request())
    assert runtime._generation_started_mono is None

    failing = _runtime(_FakeModel(), max_prompt_chars=10)
    with pytest.raises(ValueError, match="GENIEX_MAX_PROMPT_CHARS"):
        failing.chat(_request())
    # The marker must not leak from the failed locked section, or the watchdog
    # would re-exec the server while it sits idle.
    assert failing._generation_started_mono is None


# --- vision bridge ----------------------------------------------------------


class _FakeVLM(_FakeModel):
    """Only the VLM handle's generate() takes images=[...]; the runtime's
    capability probe keys on exactly that."""

    def __init__(self, *args, fail_generate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_image_values = None
        self.images_existed_at_generate = None
        self._fail_generate = fail_generate
        self.closed = False

    def close(self):
        self.closed = True

    def _apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tools):
        assert add_generation_prompt is True and enable_thinking is False and tools is None
        rendered = []
        images = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
                continue
            for part in content:
                if part["type"] == "image":
                    images.append(part["image"])
                    rendered.append("<image>")
                else:
                    rendered.append(part["text"])
        self.template_image_values = images
        return "\n".join(rendered) + "\nassistant:"

    def generate(self, prompt, *, images=None, **kwargs):
        if self._fail_generate:
            raise RuntimeError("native generate exploded")
        self.images_existed_at_generate = [os.path.isfile(p) for p in (images or [])]
        kwargs["images"] = images
        return super().generate(prompt, **kwargs)


def _png_bytes(width=2, height=2):
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def _jpeg_bytes(width=3, height=4):
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03" + b"\x00" * 10
    )


def _webp_vp8x_bytes(width=5, height=6):
    return (
        b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP"
        + b"VP8X" + (10).to_bytes(4, "little")
        + b"\x00" * 4
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )


def _data_uri(data, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _image_request(url, text="what is in this image?"):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": text},
                ],
            }
        ]
    }


def _vlm_runtime(fake, tmp_path, **overrides):
    # The image reserve (1536 tok * 6 chars) must not eat the whole test
    # budget; production derives n_ctx * 6 = 36864 which has the same headroom.
    overrides.setdefault("max_prompt_chars", 50_000)
    # Vision only runs on a CPU device_map on this build (HTP + image segfaults),
    # so the realistic VLM runtime is configured for CPU. A CPU load reports
    # device=None, matching the SDK.
    overrides.setdefault("device_map", "llama_cpp:cpu")
    runtime = _runtime(fake, image_tmpdir=str(tmp_path), **overrides)
    runtime.is_vlm = True
    runtime.loaded_device = None
    return runtime


def test_vlm_capability_probe_keys_on_images_parameter():
    assert server._detect_vlm_handle(_FakeVLM()) is True
    assert server._detect_vlm_handle(_FakeModel()) is False
    assert server._detect_vlm_handle(SimpleNamespace(generate=None)) is False


def test_image_dimension_parsers_read_container_headers():
    assert server._image_dimensions("image/png", _png_bytes(640, 480)) == (640, 480)
    assert server._image_dimensions("image/jpeg", _jpeg_bytes(31, 17)) == (31, 17)
    assert server._image_dimensions("image/webp", _webp_vp8x_bytes(101, 55)) == (101, 55)
    assert server._image_dimensions("image/png", b"\x89PNG\r\n\x1a\n") is None
    assert server._sniff_image_mime(_png_bytes()) == "image/png"
    assert server._sniff_image_mime(_jpeg_bytes()) == "image/jpeg"
    assert server._sniff_image_mime(_webp_vp8x_bytes()) == "image/webp"
    assert server._sniff_image_mime(b"<svg></svg>") is None


def test_text_content_parts_flatten_for_text_requests():
    fake = _FakeModel()
    runtime = _runtime(fake)
    runtime.chat(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "line one"},
                        {"type": "text", "text": "line two"},
                    ],
                }
            ]
        }
    )
    prompt = fake.generate_calls[0][0]
    assert "line one\nline two" in prompt
    # An LLM handle's generate() must never receive an images argument.
    assert "images" not in fake.generate_calls[0][1]


def test_unknown_content_part_types_are_rejected_not_dropped():
    with pytest.raises(ValueError, match="unsupported content part type"):
        _runtime(_FakeModel()).chat(
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "data": "x"}]}]}
        )


def test_images_rejected_without_mmproj():
    with pytest.raises(ValueError, match="GENIEX_MMPROJ"):
        _runtime(_FakeModel()).chat(_image_request(_data_uri(_png_bytes())))


def test_images_rejected_outside_final_user_message(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    image_part = {"type": "image_url", "image_url": {"url": _data_uri(_png_bytes())}}
    # Image in an earlier turn.
    with pytest.raises(ValueError, match="final message"):
        runtime.chat(
            {
                "messages": [
                    {"role": "user", "content": [dict(image_part), {"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "and now?"},
                ]
            }
        )
    # Final message whose role is not user.
    with pytest.raises(ValueError, match="final message"):
        runtime.chat(
            {"messages": [{"role": "system", "content": [dict(image_part)]}]}
        )


def test_http_image_urls_are_rejected(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    with pytest.raises(ValueError, match="does not fetch remote URLs"):
        runtime.chat(_image_request("https://example.com/cat.png"))


def test_image_mime_allowlist_and_magic_validation(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    with pytest.raises(ValueError, match="unsupported image MIME"):
        runtime.chat(_image_request(_data_uri(b"<svg/>", mime="image/svg+xml")))
    # Claimed JPEG, actual PNG bytes.
    with pytest.raises(ValueError, match="do not match the declared MIME"):
        runtime.chat(_image_request(_data_uri(_png_bytes(), mime="image/jpeg")))


def test_image_base64_is_validated_and_never_echoed(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    payload = "!!!not-base64-at-all!!!" * 20
    with pytest.raises(ValueError, match="not valid base64") as excinfo:
        runtime.chat(_image_request(f"data:image/png;base64,{payload}"))
    # Data URIs carry the whole image; they must never leak into errors/logs.
    assert payload[:16] not in str(excinfo.value)


def test_image_count_size_and_pixel_limits(tmp_path):
    uri = _data_uri(_png_bytes())
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    request = _image_request(uri)
    request["messages"][0]["content"].insert(
        1, {"type": "image_url", "image_url": {"url": uri}}
    )
    with pytest.raises(server._PayloadTooLarge, match="at most 1 image"):
        runtime.chat(request)

    small = _vlm_runtime(_FakeVLM(), tmp_path, max_image_bytes=8)
    with pytest.raises(server._PayloadTooLarge, match="decoded-size limit"):
        small.chat(_image_request(uri))

    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    with pytest.raises(server._PayloadTooLarge, match="pixel limit"):
        runtime.chat(_image_request(_data_uri(_png_bytes(10000, 10000))))


def test_vlm_image_roundtrip_paths_order_and_cleanup(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    response = runtime.chat(_image_request(_data_uri(_png_bytes()), text="describe"))

    _prompt, kwargs = fake.generate_calls[0]
    # The temp file existed while generate() ran, template order matches the
    # generate() list exactly, and the file is gone after the request.
    assert fake.images_existed_at_generate == [True]
    assert kwargs["images"] == fake.template_image_values
    assert len(kwargs["images"]) == 1
    assert kwargs["images"][0].startswith(str(tmp_path))
    assert kwargs["images"][0].endswith(".png")
    assert os.listdir(tmp_path) == []
    assert response["geniex_metrics"]["images"] == 1


def test_vlm_handle_reloads_before_each_request_but_llm_never(tmp_path):
    # The VLM handle is single-shot on this build; the server must reload it
    # before each request after the first generation. The LLM handle is stable
    # and must never reload.
    reloads = {"n": 0}
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    original = runtime._construct_model

    def counting_construct():
        reloads["n"] += 1
        return fake

    runtime._construct_model = counting_construct

    # First request: not yet poisoned, no reload; poisons on the way out.
    runtime.chat(_request())
    assert reloads["n"] == 0 and runtime._needs_model_reload is True
    # Second request: reloads first, then serves.
    runtime.chat(_request())
    assert reloads["n"] == 1 and runtime.reload_count == 1
    # Third: reloads again — one fresh handle per request.
    runtime.chat(_request())
    assert reloads["n"] == 2

    # An LLM handle never reloads.
    llm_fake = _FakeModel()
    llm = _runtime(llm_fake)
    llm._construct_model = counting_construct
    before = reloads["n"]
    llm.chat(_request())
    llm.chat(_request())
    assert reloads["n"] == before and llm._needs_model_reload is False


def test_vlm_reload_happens_even_after_generate_failure(tmp_path):
    # A failed VLM generate still poisons the handle, so the next request must
    # still reload.
    fake = _FakeVLM(fail_generate=True)
    runtime = _vlm_runtime(fake, tmp_path)
    with pytest.raises(RuntimeError, match="native generate exploded"):
        runtime.chat(_request())
    assert runtime._needs_model_reload is True


def test_reload_disposes_old_handle_before_constructing_replacement(tmp_path):
    # The old handle must be close()d before the replacement is built, or its
    # native memory leaks (~4.7 GB/reload, board-measured) until OOM.
    old = _FakeVLM()
    runtime = _vlm_runtime(old, tmp_path)
    order = []
    old_close = old.close

    def tracked_close():
        order.append("close")
        old_close()

    old.close = tracked_close
    new = _FakeVLM()

    def construct():
        order.append("construct")
        return new

    runtime._construct_model = construct
    runtime._needs_model_reload = True
    runtime._reload_model()
    assert old.closed is True
    assert order == ["close", "construct"]  # dispose strictly before build
    assert runtime._model is new


def test_reload_failure_trips_breaker_and_frees_poisoned_handle(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.reload_failure_limit = 2

    def boom():
        raise RuntimeError("reload exploded")

    runtime._construct_model = boom
    runtime._needs_model_reload = True

    # First failed reload: model freed (never falls back to poisoned handle),
    # runtime still trying.
    with pytest.raises(RuntimeError, match="reload exploded"):
        runtime._reload_model()
    assert runtime._model is None
    assert runtime.reload_failure_count == 1
    assert runtime.runtime_unavailable is False
    assert runtime.handle_state() == "unavailable"  # _model is None

    # Second failure trips the breaker.
    with pytest.raises(RuntimeError, match="reload exploded"):
        runtime._reload_model()
    assert runtime.runtime_unavailable is True
    assert "reload exploded" in runtime.last_reload_error
    # Requests now fail fast without attempting the broken load again.
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        runtime.chat(_request())


def test_reload_budget_trips_breaker_before_oom(tmp_path):
    # Each VLM reload leaks native memory, so the budget must trip the breaker
    # before it can OOM the box — vision degrades to "restart required".
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.reload_budget = 2
    runtime._construct_model = lambda: _FakeVLM()

    # Two reloads are allowed (reload_count 0->1->2).
    runtime._needs_model_reload = True
    runtime._reload_model()
    runtime._needs_model_reload = True
    runtime._reload_model()
    assert runtime.reload_count == 2

    # The third reload is refused before allocating another leaky handle.
    runtime._needs_model_reload = True
    with pytest.raises(RuntimeError, match="reload budget exhausted"):
        runtime._reload_model()
    assert runtime.runtime_unavailable is True
    assert runtime.handle_state() == "unavailable"
    payload = runtime.health_payload()
    assert payload["status"] == "unavailable" and payload["reload_budget"] == 2


def test_budget_counts_attempts_not_successes(tmp_path):
    # A failed native construction leaks too, so it must consume a budget slot.
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.reload_budget = 2
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("native construct failed")
        return _FakeVLM()

    runtime._construct_model = flaky
    runtime.reload_failure_limit = 99  # isolate the budget path from the failure breaker

    runtime._needs_model_reload = True
    with pytest.raises(RuntimeError, match="native construct failed"):
        runtime._reload_model()  # attempt 1 (failed) still spends a slot
    runtime._needs_model_reload = True
    runtime._reload_model()  # attempt 2 (ok)
    assert runtime.reload_attempt_count == 2 and runtime.reload_count == 1
    # Budget (attempts) is now spent even though only one reload succeeded.
    runtime._needs_model_reload = True
    with pytest.raises(server._ServiceUnavailable, match="budget exhausted"):
        runtime._reload_model()


def test_mem_available_guard_refuses_reload(tmp_path, monkeypatch):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.reload_budget = 10  # not the limiter here
    runtime.min_mem_available_bytes = 8 * 1024**3
    runtime._construct_model = lambda: _FakeVLM()
    monkeypatch.setattr(server, "_mem_available_bytes", lambda: 2 * 1024**3)
    runtime._needs_model_reload = True
    with pytest.raises(server._ServiceUnavailable, match="memory headroom"):
        runtime._reload_model()
    assert runtime.runtime_unavailable is True


def test_budget_exhausted_is_503_not_400(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    runtime.runtime_unavailable = True  # breaker already tripped
    body = json.dumps(_request()).encode()
    status, payload = _do_post(runtime, {"Content-Length": str(len(body))}, body)
    assert status == int(server.HTTPStatus.SERVICE_UNAVAILABLE)
    assert payload["error"]["type"] == "service_unavailable"


def test_vision_require_image_rejects_text_only(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.require_image = True
    with pytest.raises(ValueError, match="requires an image"):
        runtime.chat(_request())
    assert fake.generate_calls == []
    # An image request is accepted.
    runtime.chat(_image_request(_data_uri(_png_bytes())))
    assert len(fake.generate_calls) == 1


def test_health_reports_generation_attempts_remaining(tmp_path):
    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    runtime.reload_budget = 2
    payload = runtime.health_payload()
    assert payload["generation_attempts_remaining"] == 2
    assert payload["construction_attempts"] == 0
    runtime.reload_attempt_count = 2
    assert runtime.health_payload()["generation_attempts_remaining"] == 0


def test_reload_refuses_swapped_artifact_via_file_identity(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"a" * 32)
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path, model_ref=str(model))
    runtime._record_file_identities()
    # Replace the file's content/mtime after the startup snapshot.
    import time as _t

    _t.sleep(0.01)
    model.write_bytes(b"b" * 64)
    runtime._construct_model = lambda: fake
    runtime._needs_model_reload = True
    with pytest.raises(RuntimeError, match="changed since startup"):
        runtime._reload_model()


def test_reloaded_handle_capability_is_revalidated(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    # A reload that returns a non-VLM handle (capability drift) must fail.
    runtime._construct_model = lambda: _FakeModel()
    runtime._needs_model_reload = True
    with pytest.raises(RuntimeError, match="changed VLM capability"):
        runtime._reload_model()


def test_stuck_reload_flags_only_overdue_reloads(tmp_path):
    import time as _t

    runtime = _vlm_runtime(_FakeVLM(), tmp_path)
    assert runtime.stuck_reload(10.0) is None
    runtime._reload_started_mono = _t.monotonic() - 1.0
    assert runtime.stuck_reload(10.0) is None
    runtime._reload_started_mono = _t.monotonic() - 11.0
    stuck = runtime.stuck_reload(10.0)
    assert stuck is not None and stuck[0] >= 11.0 and "reload" in stuck[1]
    assert runtime.stuck_reload(0.0) is None


def test_vlm_text_only_request_passes_explicit_empty_images(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.chat(_request())
    assert fake.generate_calls[0][1]["images"] == []


def test_vlm_cleanup_and_marker_after_generate_failure(tmp_path):
    fake = _FakeVLM(fail_generate=True)
    runtime = _vlm_runtime(fake, tmp_path)
    with pytest.raises(RuntimeError, match="native generate exploded"):
        runtime.chat(_image_request(_data_uri(_png_bytes())))
    assert os.listdir(tmp_path) == []
    assert runtime._generation_started_mono is None
    assert not runtime._lock.locked()


def test_image_requests_shrink_the_prompt_char_budget(tmp_path):
    # reserve 1536 tokens * 6 chars = 9216; with max 9300 the effective budget
    # for an image request is 84 chars, which this prompt exceeds.
    runtime = _vlm_runtime(_FakeVLM(), tmp_path, max_prompt_chars=9300)
    with pytest.raises(ValueError, match=r"1 image\(s\) reserved"):
        runtime.chat(_image_request(_data_uri(_png_bytes()), text="x" * 120))
    # The same text without an image fits comfortably.
    runtime.chat(_request())


def test_startup_smoke_reads_runtime_reported_decode_tokens():
    runtime = _runtime(_FakeModel(generated_tokens=7))
    smoke = runtime.startup_smoke()
    assert smoke["ok"] is True and smoke["decode_proven"] is True
    assert smoke["completion_tokens_reported"] == 7
    assert runtime.startup_smoke_result == smoke

    # One reported token could have come from the prefill logits alone; the
    # smoke must degrade honestly rather than claim decode proof.
    ambiguous = _runtime(_FakeModel(generated_tokens=1))
    smoke = ambiguous.startup_smoke()
    assert smoke["decode_proven"] is False
    assert "load/prefill" in smoke["note"]

    with pytest.raises(RuntimeError, match="native generate exploded"):
        _runtime(_FakeVLM(fail_generate=True)).startup_smoke()


def test_expected_device_validates_resolution():
    runtime = _runtime(_FakeModel(), expect_device="HTP0")
    runtime._verify_expected_device()  # HTP0 == loaded device: passes

    full = _runtime(_FakeModel(), expect_device="llama_cpp:HTP0")
    full._verify_expected_device()

    # A CPU load reports device=None; expect_device='cpu' must still match.
    cpu = _runtime(_FakeModel(), device_map="llama_cpp:cpu", expect_device="cpu")
    cpu.loaded_device = None
    cpu._verify_expected_device()

    wrong = _runtime(_FakeModel(), expect_device="HTP0")
    wrong.loaded_backend, wrong.loaded_device = "qairt", "NPU"
    with pytest.raises(RuntimeError, match="device resolution mismatch"):
        wrong._verify_expected_device()

    # npu->QAIRT misroute caught when HTP0 was intended but qairt resolved.
    misroute = _runtime(_FakeModel(), expect_device="llama_cpp:HTP0")
    misroute.loaded_backend, misroute.loaded_device = "qairt", "NPU"
    with pytest.raises(RuntimeError, match="device resolution mismatch"):
        misroute._verify_expected_device()


def test_artifact_hash_pin_fails_closed(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-payload")
    good = server._sha256_file(str(model))

    runtime = _runtime(_FakeModel(), model_ref=str(model), expect_model_sha256=good)
    runtime._verify_artifact_hashes()
    assert runtime.model_sha256 == good

    bad = _runtime(_FakeModel(), model_ref=str(model), expect_model_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="model sha256 mismatch"):
        bad._verify_artifact_hashes()

    missing = _runtime(_FakeModel(), model_ref="hub/repo:Q4_0", expect_model_sha256=good)
    with pytest.raises(RuntimeError, match="not a local file"):
        missing._verify_artifact_hashes()


def test_vision_device_uses_exact_allowlist_not_prefix():
    # An image request on an HTP device segfaults the daemon (CLIP tower ops
    # unsupported); anything not in the exact (backend, compute_unit) allowlist
    # must fail closed before serving. The guard keys on the CONFIGURED
    # device_map (a CPU load resolves to device=None, so the resolved device
    # cannot discriminate CPU from HTP).
    for device_map in ("llama_cpp:HTP0", "hybrid", "npu", "auto", "gpu", "llama_cpp:cpu2"):
        runtime = _runtime(_FakeVLM(), device_map=device_map)
        runtime.is_vlm = True
        assert runtime.vision_status() == "blocked_current_build"
        with pytest.raises(RuntimeError, match="not in the vision-safe allowlist"):
            runtime._verify_vision_device()

    # Only exact llama_cpp:cpu (device resolves to None) is allowed.
    for device_map in ("llama_cpp:cpu", "cpu"):
        cpu = _runtime(_FakeVLM(), device_map=device_map)
        cpu.is_vlm = True
        cpu.loaded_device = None
        assert cpu.vision_status() == "experimental_cpu"
        cpu._verify_vision_device()

    # Exact-value danger override: must name the precise device, not a boolean.
    htp = _runtime(
        _FakeVLM(), device_map="llama_cpp:HTP0", unsafe_allow_vision_device="llama_cpp:HTP0"
    )
    htp.is_vlm = True
    assert htp.vision_status() == "unsafe_override"
    htp._verify_vision_device()
    # A mismatched override does not unlock a different device.
    mismatch = _runtime(
        _FakeVLM(), device_map="hybrid", unsafe_allow_vision_device="llama_cpp:HTP0"
    )
    mismatch.is_vlm = True
    assert mismatch.vision_status() == "blocked_current_build"
    with pytest.raises(RuntimeError, match="not in the vision-safe allowlist"):
        mismatch._verify_vision_device()

    # A text-only (non-VLM) load on HTP is unaffected.
    text = _runtime(_FakeModel(), device_map="llama_cpp:HTP0")
    assert text.vision_status() == "disabled"
    text._verify_vision_device()


def test_image_request_refused_before_native_call_on_unsafe_device(tmp_path):
    # Defense in depth: even if the startup guard were bypassed, an image must
    # not reach the native generate() on a non-vision-safe device.
    fake = _FakeVLM()
    runtime = _runtime(fake, image_tmpdir=str(tmp_path), device_map="llama_cpp:HTP0", max_prompt_chars=50_000)
    runtime.is_vlm = True  # loaded but on HTP (unsafe)
    with pytest.raises(RuntimeError, match="non-vision-safe device"):
        runtime.chat(_image_request(_data_uri(_png_bytes())))
    assert fake.generate_calls == []  # never reached the native call
    assert os.listdir(tmp_path) == []


def test_health_payload_shape_without_generation(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path, model_label="ssmout-q8_0-embd-q4_0")
    payload = runtime.health_payload()
    assert payload["status"] == "ok"
    assert payload["model_label"] == "ssmout-q8_0-embd-q4_0"
    assert payload["vlm"] is True
    assert payload["vision_status"] == "experimental_cpu"  # _vlm_runtime is configured cpu
    assert payload["vision_available"] is True
    assert payload["device_configured"] == "llama_cpp:cpu"
    # A CPU load resolves device=None; /health renders it 'cpu', not 'None'.
    assert payload["device_resolved"] == "llama_cpp:cpu"
    assert payload["max_images"] == 1
    assert payload["startup_smoke"] is None
    assert payload["handle_state"] == "fresh"
    assert payload["vlm_reuse_policy"] == "reload_before_each_generate"
    assert payload["reload_count"] == 0 and payload["reload_failure_count"] == 0
    # /health must stay cheap enough to poll during a long generation.
    assert fake.generate_calls == []


class _FakeHTTPRequest:
    """Drive GenieXOpenAIHandler.do_POST without a socket: capture the status
    and JSON body it would send, with a controllable header/body pair."""

    def __init__(self, runtime, headers, body=b"", *, max_body=server.DEFAULT_MAX_BODY_BYTES, gate=None):
        import io

        self.path = "/v1/chat/completions"
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.sent_status = None
        self.server = SimpleNamespace(
            geniex_runtime=runtime, geniex_max_body_bytes=max_body, geniex_body_gate=gate
        )

    def send_response(self, status):
        self.sent_status = status

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    # The real serializer only touches send_response/send_header/end_headers/wfile.
    _send_json = server.GenieXOpenAIHandler._send_json

    def _result(self):
        self.wfile.seek(0)
        return self.sent_status, json.loads(self.wfile.read().decode())


def _do_post(runtime, headers, body=b"", **kw):
    req = _FakeHTTPRequest(runtime, headers, body, **kw)
    server.GenieXOpenAIHandler.do_POST(req)
    return req._result()


def test_post_rejects_transfer_encoding_and_bad_content_length():
    runtime = _runtime(_FakeModel())
    status, payload = _do_post(runtime, {"Transfer-Encoding": "chunked"})
    assert status == int(server.HTTPStatus.BAD_REQUEST)
    assert "Transfer-Encoding" in payload["error"]["message"]

    status, _ = _do_post(runtime, {})  # no Content-Length
    assert status == int(server.HTTPStatus.BAD_REQUEST)

    status, payload = _do_post(runtime, {"Content-Length": "not-a-number"})
    assert status == int(server.HTTPStatus.BAD_REQUEST)
    assert "invalid Content-Length" in payload["error"]["message"]

    status, _ = _do_post(runtime, {"Content-Length": "-5"})
    assert status == int(server.HTTPStatus.BAD_REQUEST)


def test_post_rejects_oversized_body_with_413():
    runtime = _runtime(_FakeModel())
    status, payload = _do_post(
        runtime, {"Content-Length": "999999"}, b"{}", max_body=10
    )
    assert status == int(server.HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


def test_post_body_gate_releases_on_success_and_error():
    import threading

    runtime = _runtime(_FakeModel())
    gate = threading.BoundedSemaphore(1)
    body = json.dumps(_request()).encode()
    status, _ = _do_post(runtime, {"Content-Length": str(len(body))}, body, gate=gate)
    assert status == int(server.HTTPStatus.OK)
    # The gate must be fully released (a leaked permit would raise here).
    assert gate.acquire(blocking=False)
    gate.release()

    # An error path must also release: bad body, gate still returns to full.
    gate2 = threading.BoundedSemaphore(1)
    status, _ = _do_post(runtime, {"Content-Length": "2"}, b"[]", gate=gate2)
    assert status == int(server.HTTPStatus.BAD_REQUEST)
    assert gate2.acquire(blocking=False)


def test_vlm_grammar_parameters_reach_generate_unchanged(tmp_path):
    fake = _FakeVLM()
    runtime = _vlm_runtime(fake, tmp_path)
    runtime.chat(
        {
            **_image_request(_data_uri(_png_bytes())),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": PROBE_STYLE_SCHEMA},
            },
        }
    )
    _prompt, kwargs = fake.generate_calls[0]
    assert "GRAMMAR-OK-7391" in kwargs["grammar"]
    assert kwargs["json_mode"] is False
    assert len(kwargs["images"]) == 1
