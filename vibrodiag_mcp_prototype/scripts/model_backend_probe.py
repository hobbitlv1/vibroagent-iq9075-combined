#!/usr/bin/env python3
"""Probe an OpenAI-compatible model backend for VibroAgent suitability.

Runs the same checks against any backend (Genie adapter on :8910, GenieX on
:18181, plain llama-server, ...) so backends can be compared apples to apples:

1. /v1/models reachability
2. Generation throughput (tokens/s from usage, wall-clock fallback)
3. Agent action-protocol discipline (one clean JSON object? trailing junk?)
4. response_format json_schema support (llama.cpp grammar-constrained decoding)
5. Long-context needle recall + prefill time (does ctx > 4096 actually work?)

Usage:
  model_backend_probe.py --base-url http://127.0.0.1:18181/v1 --model 'unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0'
  model_backend_probe.py --base-url http://127.0.0.1:8910/v1 --model 'Qwen/Qwen3-4B-Instruct-2507' --long-context-chars 0
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

ACTION_SYSTEM = """You are a building-vibration operator agent.
Respond with ONE strict JSON object per turn and nothing else. Stop immediately after the closing brace.
Either call a tool:
{"action":"tool","tool":"<name>","args":{}}
or finish:
{"action":"final","answer":"<short answer>"}
Tools:
- list_sensors: {} -> configured sensors and the reference sensor
- check_all_sensors: {"duration_s":10} -> compare every target with the reference sensor
- psd_summary: {"sensor_id":"baseline","duration_s":10} -> spectrum peaks and band powers
Call at most one tool per turn."""

# The required "proof" const is deliberately something no model would emit on
# its own: it appears in the output ONLY if the server truly compiles the
# schema into a decoding grammar (llama.cpp json_schema). A backend that
# silently ignores response_format (the Genie adapter) cannot produce it.
ACTION_SCHEMA = {
    "name": "agent_action",
    "schema": {
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
    },
}


def _post(base_url: str, path: str, payload: dict, timeout_s: float) -> tuple[dict | None, str | None, float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read())
        return data, None, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return None, f"HTTP {exc.code}: {detail}", time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - report any transport failure
        return None, f"{exc.__class__.__name__}: {exc}", time.monotonic() - started


def _get(base_url: str, path: str, timeout_s: float) -> tuple[dict | None, str | None]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout_s) as response:
            return json.loads(response.read()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{exc.__class__.__name__}: {exc}"


def _chat(base_url: str, model: str, messages: list[dict], timeout_s: float, **extra) -> tuple[str | None, dict, str | None, float]:
    payload = {"model": model, "messages": messages, "temperature": 0.0, **extra}
    data, error, elapsed = _post(base_url, "/chat/completions", payload, timeout_s)
    if error or not data:
        return None, {}, error, elapsed
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return None, data.get("usage") or {}, "malformed chat response", elapsed
    return content, data.get("usage") or {}, None, elapsed


def _first_json_object(text: str) -> tuple[dict | None, bool]:
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None, False
    try:
        obj, end = json.JSONDecoder().raw_decode(stripped[start:])
    except ValueError:
        return None, False
    trailing = bool(stripped[start + end :].strip())
    return (obj if isinstance(obj, dict) else None), trailing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--throughput-tokens", type=int, default=160)
    parser.add_argument(
        "--long-context-chars",
        type=int,
        default=24000,
        help="~6k tokens; proves ctx>4096 works. 0 disables the check.",
    )
    args = parser.parse_args()

    print(f"# Backend probe: {args.base_url} model={args.model}\n")

    models, error = _get(args.base_url, "/models", timeout_s=10.0)
    if error:
        print(f"[models] FAILED: {error}")
        return 1
    ids = [item.get("id") for item in (models or {}).get("data", [])]
    print(f"[models] ok: {ids}")

    # 2. Throughput
    content, usage, error, elapsed = _chat(
        args.base_url,
        args.model,
        [{"role": "user", "content": "Describe, in plain sentences, what a vibration reference sensor is used for in building monitoring."}],
        args.timeout_s,
        max_tokens=args.throughput_tokens,
    )
    if error:
        print(f"[throughput] FAILED: {error}")
    else:
        completion_tokens = usage.get("completion_tokens") or max(1, len(content) // 4)
        prompt_tokens = usage.get("prompt_tokens")
        print(
            f"[throughput] {completion_tokens} tokens in {elapsed:.1f}s -> {completion_tokens / elapsed:.1f} tok/s "
            f"(prompt_tokens={prompt_tokens}, chars={len(content)})"
        )

    # 3. Action-protocol discipline (the agentic loop's step format)
    for label, user in (
        ("route", "Operator request: Compare all building sensors with the reference now."),
        ("list", "Operator request: what sensors do we have?"),
    ):
        content, _usage, error, elapsed = _chat(
            args.base_url,
            args.model,
            [{"role": "system", "content": ACTION_SYSTEM}, {"role": "user", "content": user}],
            args.timeout_s,
            max_tokens=192,
        )
        if error:
            print(f"[action:{label}] FAILED: {error}")
            continue
        action, trailing = _first_json_object(content)
        verdict = "clean" if action and not trailing else ("first-object-ok+trailing-junk" if action else "NO-VALID-JSON")
        print(f"[action:{label}] {verdict} in {elapsed:.1f}s: {json.dumps(action) if action else content[:120]!r}")

    # 4. Grammar-constrained decoding (llama.cpp json_schema)
    content, _usage, error, elapsed = _chat(
        args.base_url,
        args.model,
        [{"role": "system", "content": ACTION_SYSTEM}, {"role": "user", "content": "Operator request: check everything."}],
        args.timeout_s,
        max_tokens=192,
        response_format={"type": "json_schema", "json_schema": ACTION_SCHEMA},
    )
    if error:
        print(f"[json_schema] unsupported/failed: {error}")
    else:
        action, trailing = _first_json_object(content)
        enforced = bool(action) and not trailing and action.get("proof") == "GRAMMAR-OK-7391"
        verdict = "ENFORCED (grammar-constrained decoding works)" if enforced else "NOT enforced (response_format ignored)"
        print(f"[json_schema] {verdict}: {content[:140]!r}")

    # 5. Long-context needle recall + prefill cost
    if args.long_context_chars > 0:
        needle = "The maintenance access code for the pump room is 7391."
        filler_line = "Routine vibration levels on all monitored floors stayed within the expected relative band during this interval. "
        filler = (filler_line * (args.long_context_chars // len(filler_line) + 1))[: args.long_context_chars]
        prompt = (
            "Read this monitoring shift log carefully.\n"
            + filler[: len(filler) // 2]
            + "\nNOTE: "
            + needle
            + "\n"
            + filler[len(filler) // 2 :]
            + "\nQuestion: What is the maintenance access code for the pump room? Answer with the number only."
        )
        content, usage, error, elapsed = _chat(
            args.base_url,
            args.model,
            [{"role": "user", "content": prompt}],
            args.timeout_s,
            max_tokens=24,
        )
        if error:
            print(f"[long-context ~{args.long_context_chars} chars] FAILED: {error}")
        else:
            found = "7391" in (content or "")
            print(
                f"[long-context ~{args.long_context_chars} chars] {'RECALLED' if found else 'MISSED'} "
                f"in {elapsed:.1f}s (prompt_tokens={usage.get('prompt_tokens')}): {content.strip()[:80]!r}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
