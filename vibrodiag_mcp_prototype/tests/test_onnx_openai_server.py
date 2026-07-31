from pathlib import Path

from vibroagent_mcp import onnx_openai_server as server


def test_normalize_messages_extracts_text_and_tool_results():
    messages = server.normalize_messages(
        [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze vibration."},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                ],
            },
            {"role": "tool", "name": "run_check", "content": "{\"ok\": true}"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Analyze vibration.\n[image_url omitted]"},
        {"role": "user", "content": "Tool result (run_check):\n{\"ok\": true}"},
    ]


def test_qwen_chat_prompt_uses_qwen_message_tokens():
    prompt = server.qwen_chat_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    assert "<|im_start|>system\nYou are concise.<|im_end|>" in prompt
    assert "<|im_start|>user\nSay hello.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_trim_at_stop_returns_text_before_first_stop_sequence():
    text, stopped = server._trim_at_stop("answer<|im_end|>trailing", ("<|im_end|>", "END"))

    assert stopped is True
    assert text == "answer"


def test_chat_completion_payload_is_openai_compatible():
    config = server.OnnxServerConfig(model_path=Path("/tmp/model"), model_id="Qwen/Qwen3.5-4B")
    result = server.OnnxCompletion(content='{"ok":true}', prompt_tokens=10, completion_tokens=3, total_s=1.5)

    payload = server._chat_completion_payload(config, result)

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "Qwen/Qwen3.5-4B"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
