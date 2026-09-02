"""Optional OpenAI-compatible JSON model client for logical building agents."""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .errors import format_exception_for_response


MODEL_HTTP_TIMEOUT_GRACE_S = 10.0


def _local_genie_timeout_body(base_url: str | None, timeout_s: float) -> dict[str, Any] | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    try:
        genie_port = int(os.environ.get("GENIE_ADAPTER_PORT", "8910"))
    except ValueError:
        genie_port = 8910
    if host not in {"127.0.0.1", "localhost", "::1"} or parsed.port != genie_port:
        return None
    return {"timeout_s": max(1.0, float(timeout_s))}


@dataclass(frozen=True)
class ModelCallResult:
    ok: bool
    used_model: bool
    data: dict[str, Any] | None
    text: str | None
    error: str | None
    metadata: dict[str, Any]


class ModelClient:
    """Thin optional client for local/OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url_env: str,
        api_key_env: str,
        model_env: str,
        role: str,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
    ):
        self.base_url = base_url or os.environ.get(base_url_env) or os.environ.get("QWEN_BASE_URL")
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("QWEN_API_KEY", "EMPTY")
        self.model = model or os.environ.get(model_env) or os.environ.get("QWEN_MODEL")
        self.role = role
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    def call_json_model(
        self,
        prompt: str,
        allowed_labels: set[str] | None = None,
        *,
        extra_body: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> ModelCallResult:
        if not self.is_configured:
            return self._fallback("endpoint_not_configured")

        try:
            from openai import OpenAI
        except ImportError:
            return self._fallback("openai_client_not_installed")

        try:
            timeout_s = self._timeout_s()
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=timeout_s + MODEL_HTTP_TIMEOUT_GRACE_S,
                max_retries=0,
            )
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system",
                     "content": system or "Return strict JSON only. Do not include markdown."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": self._max_tokens(),
                "timeout": timeout_s + MODEL_HTTP_TIMEOUT_GRACE_S,
            }
            merged_extra_body = dict(extra_body or {})
            local_timeout = _local_genie_timeout_body(self.base_url, timeout_s)
            if local_timeout is not None:
                for key, value in local_timeout.items():
                    merged_extra_body.setdefault(key, value)
            if merged_extra_body:
                request_kwargs["extra_body"] = merged_extra_body
            response = client.chat.completions.create(**request_kwargs)
            text = response.choices[0].message.content or ""
            data = parse_json_response(text)
            if allowed_labels:
                label = data.get("local_status") or data.get("network_status")
                if label not in allowed_labels:
                    raise ValueError(f"model returned unsupported label: {label!r}")
            return ModelCallResult(
                ok=True,
                used_model=True,
                data=data,
                text=text,
                error=None,
                metadata=self.metadata(model_used=True),
            )
        except Exception as exc:
            error = format_exception_for_response(exc)
            return ModelCallResult(
                ok=False,
                used_model=False,
                data=None,
                text=None,
                error=error,
                metadata=self.metadata(model_used=False, fallback_reason=error),
            )

    def _timeout_s(self) -> float:
        if self.timeout_s is not None:
            try:
                timeout = float(self.timeout_s)
            except (TypeError, ValueError):
                timeout = _env_float("AGENT_MODEL_TIMEOUT_S", _env_float("QWEN_TIMEOUT_S", 120.0), minimum=1.0)
            return max(1.0, timeout) if math.isfinite(timeout) else 120.0
        return _env_float("AGENT_MODEL_TIMEOUT_S", _env_float("QWEN_TIMEOUT_S", 120.0), minimum=1.0)

    def _max_tokens(self) -> int:
        if self.max_tokens is not None:
            try:
                value = int(self.max_tokens)
            except (TypeError, ValueError):
                value = _env_int("AGENT_MODEL_MAX_TOKENS", 1024, minimum=128)
            return max(32, value)
        return _env_int("AGENT_MODEL_MAX_TOKENS", 1024, minimum=128)

    def metadata(self, *, model_used: bool, fallback_reason: str | None = None) -> dict[str, Any]:
        return {
            "role": self.role,
            "model_used": bool(model_used),
            "provider": "openai_compatible" if self.base_url else "deterministic_fallback",
            "model": self.model,
            "endpoint_configured": self.is_configured,
            "fallback_reason": fallback_reason,
        }

    def _fallback(self, reason: str) -> ModelCallResult:
        return ModelCallResult(
            ok=False,
            used_model=False,
            data=None,
            text=None,
            error=reason,
            metadata=self.metadata(model_used=False, fallback_reason=reason),
        )


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        if stripped[0] in "[{" and exc.msg != "Extra data":
            raise ValueError("model response did not contain a complete top-level JSON object") from exc
        data = _parse_embedded_json_object(stripped)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("model JSON response must be an object")
    return data


def _parse_embedded_json_object(text: str) -> Any:
    for candidate in _json_candidates(text):
        try:
            return _parse_first_json_object_like(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("model response did not contain a valid JSON object")


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    parts = text.split("```")
    if len(parts) >= 3:
        for index in range(1, len(parts), 2):
            fenced = parts[index].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            candidates.append(fenced)
    candidates.append(text)
    return candidates


def _parse_first_json_object_like(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return value
    raise json.JSONDecodeError("No JSON object or array found", text, 0)


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return float(value)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return int(value)


def call_json_model(prompt: str, allowed_labels: set[str] | None = None) -> ModelCallResult:
    """Convenience function using SMALL_AGENT_* environment variables."""
    return ModelClient(
        base_url_env="SMALL_AGENT_BASE_URL",
        api_key_env="SMALL_AGENT_API_KEY",
        model_env="SMALL_AGENT_MODEL",
        role="small_sensor_agent",
    ).call_json_model(prompt, allowed_labels=allowed_labels)
