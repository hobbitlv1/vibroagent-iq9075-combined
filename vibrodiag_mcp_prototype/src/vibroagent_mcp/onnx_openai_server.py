"""OpenAI-compatible HTTP adapter for ONNX Runtime GenAI model folders."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_ONNX_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_ONNX_HOST = "127.0.0.1"
DEFAULT_ONNX_PORT = 8920
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_MAX_OUTPUT_TOKENS = 384
DEFAULT_STOP_SEQUENCES = ("<|im_end|>", "<|endoftext|>")
_PROVIDER_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "NvTensorRtRtx": "NvTensorRtRtxExecutionProvider",
}


@dataclass(frozen=True)
class OnnxServerConfig:
    model_path: Path
    model_id: str = DEFAULT_ONNX_MODEL_ID
    execution_provider: str = "follow_config"
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_request_tokens: int = 2048


@dataclass(frozen=True)
class OnnxCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = _clean(item.get("type")).lower()
                if item_type == "text":
                    parts.append(_clean(item.get("text")))
                elif item_type in {"image", "image_url", "input_image", "video", "audio"}:
                    parts.append(f"[{item_type} omitted]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            else:
                parts.append(_clean(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def normalize_messages(value: Any, *, max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("request payload must include a non-empty messages array")

    messages: list[dict[str, str]] = []
    total_chars = 0
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = _clean(item.get("role")).lower() or "user"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = _message_content_to_text(item.get("content"))
        if role == "tool":
            name = _clean(item.get("name"))
            prefix = f"Tool result ({name}):\n" if name else "Tool result:\n"
            content = prefix + content
            role = "user"
        if not content:
            continue
        total_chars += len(content)
        if total_chars > max_prompt_chars:
            raise ValueError(f"messages exceed max prompt size of {max_prompt_chars} characters")
        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("messages did not contain any text content")
    return messages


def qwen_chat_prompt(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"] if message["role"] in {"system", "user", "assistant"} else "user"
        chunks.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def _payload_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _payload_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _payload_stop_sequences(payload: dict[str, Any]) -> tuple[str, ...]:
    stops: list[str] = list(DEFAULT_STOP_SEQUENCES)
    raw = payload.get("stop")
    if isinstance(raw, str):
        stops.append(raw)
    elif isinstance(raw, list):
        stops.extend(str(item) for item in raw if str(item).strip())
    return tuple(dict.fromkeys(stop for stop in stops if stop))


def _trim_at_stop(text: str, stop_sequences: tuple[str, ...]) -> tuple[str, bool]:
    stop_indexes = [text.find(stop) for stop in stop_sequences if stop and text.find(stop) >= 0]
    if not stop_indexes:
        return text, False
    return text[: min(stop_indexes)], True


def _chat_completion_payload(config: OnnxServerConfig, result: OnnxCompletion) -> dict[str, Any]:
    created = int(time.time())
    completion_id = f"chatcmpl-onnx-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": config.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        "onnx_metrics": {
            "total_s": result.total_s,
            "tokens_per_second": (
                result.completion_tokens / result.total_s if result.total_s > 0 and result.completion_tokens else None
            ),
        },
    }


class OnnxGenAIModel:
    """Small synchronous wrapper around an ONNX Runtime GenAI model."""

    def __init__(self, config: OnnxServerConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._og = self._import_onnx_genai()
        self._model = self._load_model()
        self._tokenizer = self._og.Tokenizer(self._model)

    @staticmethod
    def _import_onnx_genai() -> Any:
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai is not installed. Install the ONNX extra with "
                "`pip install -e '.[onnx]'` before starting this adapter."
            ) from exc
        return og

    def _load_model(self) -> Any:
        model_path = self.config.model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX GenAI model path not found: {model_path}")
        genai_config = model_path / "genai_config.json"
        if not genai_config.exists():
            raise FileNotFoundError(f"ONNX GenAI model folder must contain genai_config.json: {model_path}")

        og_config = self._og.Config(str(model_path))
        provider = self.config.execution_provider
        if provider != "follow_config":
            og_config.clear_providers()
            provider_name = _PROVIDER_NAMES.get(provider)
            if provider_name:
                og_config.append_provider(provider_name)
        return self._og.Model(og_config)

    def complete(self, payload: dict[str, Any]) -> OnnxCompletion:
        messages = normalize_messages(payload.get("messages"), max_prompt_chars=self.config.max_prompt_chars)
        stop_sequences = _payload_stop_sequences(payload)
        started_s = time.perf_counter()

        with self._lock:
            prompt = self._apply_chat_template(messages)
            input_tokens = self._tokenizer.encode(prompt)
            max_tokens = _payload_int(
                payload.get("max_tokens"),
                default=self.config.default_max_tokens,
                minimum=1,
                maximum=self.config.max_request_tokens,
            )
            params = self._og.GeneratorParams(self._model)
            params.set_search_options(**self._search_options(payload, len(input_tokens), max_tokens))
            generator = self._og.Generator(self._model, params)
            generator.append_tokens(input_tokens)
            stream = self._tokenizer.create_stream()
            pieces: list[str] = []
            completion_tokens = 0
            try:
                while not generator.is_done() and completion_tokens < max_tokens:
                    generator.generate_next_token()
                    token = int(generator.get_next_tokens()[0])
                    completion_tokens += 1
                    pieces.append(stream.decode(token))
                    text, stopped = _trim_at_stop("".join(pieces), stop_sequences)
                    if stopped:
                        pieces = [text]
                        break
            finally:
                del generator

        content = "".join(pieces).strip()
        return OnnxCompletion(
            content=content,
            prompt_tokens=len(input_tokens),
            completion_tokens=completion_tokens,
            total_s=max(0.0, time.perf_counter() - started_s),
        )

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        encoded_messages = json.dumps(messages, ensure_ascii=False)
        try:
            prompt = self._tokenizer.apply_chat_template(encoded_messages, add_generation_prompt=True)
            if prompt:
                return str(prompt)
        except TypeError:
            pass
        except Exception:
            pass
        try:
            prompt = self._tokenizer.apply_chat_template(
                "",
                messages=encoded_messages,
                add_generation_prompt=True,
            )
            if prompt:
                return str(prompt)
        except Exception:
            pass
        return qwen_chat_prompt(messages)

    @staticmethod
    def _search_options(payload: dict[str, Any], prompt_tokens: int, max_tokens: int) -> dict[str, Any]:
        temperature = _payload_float(payload.get("temperature"), default=0.0, minimum=0.0, maximum=2.0)
        options: dict[str, Any] = {
            "max_length": prompt_tokens + max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            options["temperature"] = temperature
        if "top_p" in payload:
            options["top_p"] = _payload_float(payload.get("top_p"), default=1.0, minimum=0.0, maximum=1.0)
        if "top_k" in payload:
            options["top_k"] = _payload_int(payload.get("top_k"), default=50, minimum=1, maximum=1000)
        if "repetition_penalty" in payload:
            options["repetition_penalty"] = _payload_float(
                payload.get("repetition_penalty"),
                default=1.0,
                minimum=0.1,
                maximum=5.0,
            )
        return options


class OnnxOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "VibroAgentOnnxOpenAI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        backend: OnnxGenAIModel = self.server.onnx_backend  # type: ignore[attr-defined]
        if path == "/health":
            self._write_json({"ok": True, "model": backend.config.model_id, "provider": "onnxruntime_genai"})
            return
        if path == "/v1/models":
            self._write_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": backend.config.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return

        try:
            payload = self._read_json_body()
            if payload.get("stream"):
                raise ValueError("streaming responses are not supported by this adapter")
            backend: OnnxGenAIModel = self.server.onnx_backend  # type: ignore[attr-defined]
            result = backend.complete(payload)
            self._write_json(_chat_completion_payload(backend.config, result))
        except ValueError as exc:
            self._write_error(str(exc), status=HTTPStatus.BAD_REQUEST, error_type="invalid_request_error")
        except Exception as exc:
            self._write_error(f"{exc.__class__.__name__}: {exc}", status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(max(0, length))
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _write_error(
        self,
        message: str,
        *,
        status: HTTPStatus,
        error_type: str = "server_error",
    ) -> None:
        self._write_json(
            {"error": {"message": message, "type": error_type, "code": None}},
            status=status,
        )

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("ONNX_OPENAI_DEBUG"):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an ONNX Runtime GenAI model through an OpenAI-compatible API")
    parser.add_argument("--host", default=os.environ.get("ONNX_OPENAI_HOST", DEFAULT_ONNX_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ONNX_OPENAI_PORT", DEFAULT_ONNX_PORT)))
    parser.add_argument("--model-path", required=True, help="ONNX GenAI model folder containing genai_config.json")
    parser.add_argument("--model-id", default=os.environ.get("ONNX_MODEL_ID", DEFAULT_ONNX_MODEL_ID))
    parser.add_argument(
        "--execution-provider",
        choices=["follow_config", "cpu", "cuda", "dml", "qnn", "NvTensorRtRtx"],
        default=os.environ.get("ONNX_EXECUTION_PROVIDER", "follow_config"),
    )
    parser.add_argument("--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS)
    parser.add_argument("--default-max-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-request-tokens", type=int, default=2048)
    args = parser.parse_args()

    config = OnnxServerConfig(
        model_path=Path(args.model_path),
        model_id=args.model_id,
        execution_provider=args.execution_provider,
        max_prompt_chars=max(1000, args.max_prompt_chars),
        default_max_tokens=max(1, args.default_max_tokens),
        max_request_tokens=max(1, args.max_request_tokens),
    )
    backend = OnnxGenAIModel(config)
    httpd = ThreadingHTTPServer((args.host, args.port), OnnxOpenAIHandler)
    httpd.onnx_backend = backend  # type: ignore[attr-defined]
    print(f"ONNX OpenAI adapter running at http://{args.host}:{args.port}/v1")
    print(f"Model: {config.model_id}")
    print(f"Model path: {config.model_path}")
    print(f"Execution provider: {config.execution_provider}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
