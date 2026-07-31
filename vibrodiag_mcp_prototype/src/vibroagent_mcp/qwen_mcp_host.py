"""Qwen/OpenAI-compatible MCP host loop.

This script demonstrates the full idea:
1. Start/connect to the building-monitoring MCP server.
2. Route the request to a supported building tool (deterministically by default).
3. Execute the MCP tool.
4. Use exactly one model generation for the turn: either the building agent's
   structured decision or a bounded final explanation of a non-agent tool result.

Model-based tool routing remains opt-in for OpenAI-compatible runtimes that
actually support tool schemas.  The local Genie adapter consumes only chat
messages, so a deterministic first step avoids a redundant NPU generation.

Environment variables:
    QWEN_BASE_URL  default: http://127.0.0.1:1234/v1
    QWEN_API_KEY   default: EMPTY
    QWEN_MODEL     default: qwen3.5-4b-instruct-revised

Example:
    python -m vibroagent_mcp.qwen_mcp_host "Compare the current building sensors with the baseline."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from typing import Any

from .errors import format_exception_for_response
from .model_client import _local_genie_timeout_body
from .schemas import find_out_of_domain_component_terms, redact_out_of_domain_component_text

SYSTEM_PROMPT = """You are a building-vibration monitoring assistant.
This application supports only registered building sensors and relative comparison against the configured reference sensor.
For a network assessment, use run_autonomous_sensor_check when available; otherwise use run_building_vibration_agent_pipeline.
For one target sensor, use run_building_sensor_agent. For compact raw-window statistics, use read_building_sensor_window.
For sensor names or locations, use list_building_sensors.
Use only the fixed building-monitoring labels returned by the tools. Discuss relative vibration level, frequency content, transient events, sensor quality, and monitoring actions.
Do not infer component-level causes that are not present in the tool result.
Building monitoring is relative triage and is not a certified structural-safety assessment.
For latest/current requests, omit start_time_s. Set require_current=false only for saved, replay, recorded, or offline data.
Do not invent measurements. If the tool reports stale or unavailable data, state that clearly and do not present it as the current building condition.
Use sample_count as the acquisition sample count. The IIS3DWB stream is acceleration in g, not velocity.
"""

FINAL_ANSWER_PROMPT = """The building-monitoring tool has already run.
Write the final operator explanation only, in at most 180 words.
Lead with the building-level finding, then give the most important relative evidence and one practical monitoring step.
Do not output another <tool_call> block, JSON function call, or request to run a tool.
Use only the supplied result. Do not invent measurements or component-level causes.
"""

FINAL_SYSTEM_PROMPT = """You are a concise building-vibration monitoring assistant.
The host has already selected and executed a building-monitoring tool.
Explain only the supplied result in terms of the reference sensor, target sensors, vibration metrics, frequency content, transient events, and data quality.
This is relative monitoring, not a certified structural-safety assessment.
"""

DEFAULT_QWEN_TIMEOUT_S = 300.0
DEFAULT_QWEN_MAX_TOKENS = 256
DEFAULT_QWEN_MAX_PROMPT_CHARS = 9000
DEFAULT_AGENT_MODEL_MAX_TOKENS = 1024
DEFAULT_MODEL_TOOL_ROUTING = False
# Agentic chat: the model drives the whole turn (tool choice, interpretation,
# final prose) through a strict one-JSON-object-per-step protocol. Default ON
# per the LLM-decides-everything policy; QWEN_AGENTIC_CHAT=0 restores the
# legacy deterministic-routing flow.
DEFAULT_AGENTIC_CHAT = True
DEFAULT_AGENTIC_MAX_STEPS = 4
AGENTIC_MAX_PROTOCOL_CORRECTIONS = 2
HTTP_TIMEOUT_GRACE_S = 10.0
# read_sdk_vibrometer_window rejects windows above this edge-safety cap, and no
# MCP service raises max_duration_s. Durations the host injects or forwards must
# stay inside it, otherwise every sensor read in the turn fails validation.
MAX_TOOL_WINDOW_DURATION_S = 60.0


class ModelExplanationError(RuntimeError):
    """Raised when chat cannot return a verified model-generated explanation."""


class ModelExplanationTimeoutError(ModelExplanationError, TimeoutError):
    """Raised when the only allowed model explanation exceeds its time budget."""


ALLOWED_MCP_TOOL_NAMES = {
    "list_building_sensors",
    "read_building_sensor_window",
    "read_building_sensor_psd",
    "compare_building_sensor_psd",
    "run_autonomous_sensor_check",
    "run_building_sensor_agent",
    "run_building_vibration_agent_pipeline",
    "export_building_validation_dataset",
}


def _load_openai_client(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    timeout_s: float | None = None,
):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The optional OpenAI-compatible client is not installed. Run: pip install -e '.[qwen]'"
        ) from exc

    base_url = base_url or os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
    timeout = (
        _env_float("QWEN_TIMEOUT_S", DEFAULT_QWEN_TIMEOUT_S, minimum=1.0)
        if timeout_s is None
        else max(1.0, float(timeout_s))
    )
    max_retries = _env_int("QWEN_MAX_RETRIES", 0, minimum=0)
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout + HTTP_TIMEOUT_GRACE_S, max_retries=max_retries)


def _load_mcp_client_symbols():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "The MCP SDK is not installed. Run: pip install -e ."
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return float(value)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _mcp_tool_to_openai_tool(tool: Any) -> dict[str, Any]:
    """Convert an MCP tool description to an OpenAI-compatible function schema."""
    input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": input_schema,
        },
    }


def _extract_tool_result(result: Any) -> dict[str, Any] | str:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return _decode_jsonish(structured)
    if getattr(result, "content", None):
        first = result.content[0]
        text = getattr(first, "text", None)
        if text is not None:
            return _decode_jsonish(text)
    if isinstance(result, dict | list):
        return _decode_jsonish(result)
    return str(result)


def _extract_model_explanation_error(exc: BaseException) -> ModelExplanationError | None:
    """Return the ModelExplanationError hidden inside (nested) ExceptionGroups.

    The MCP stdio client runs anyio task groups, so an error raised in the
    ``async with`` body propagates wrapped in one or more ExceptionGroups.
    Without unwrapping, callers' typed handlers never match and the operator
    sees "ExceptionGroup: unhandled errors in a TaskGroup" instead of the real
    model error. The timeout subclass is preferred when present.
    """

    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    found: ModelExplanationError | None = None
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ModelExplanationTimeoutError):
            return current
        if isinstance(current, ModelExplanationError) and found is None:
            found = current
        children = getattr(current, "exceptions", None)
        if isinstance(children, tuple):
            stack.extend(child for child in children if isinstance(child, BaseException))
    return found


async def chat_with_mcp(user_message: str, **kwargs: Any) -> dict[str, Any]:
    """Run one webchat turn; typed model errors surface unwrapped.

    Dispatches to the agentic loop (the model chooses tools step by step;
    default, LLM-decides-everything policy) or the legacy single-shot flow
    (``agentic=False`` or ``QWEN_AGENTIC_CHAT=0``), and re-raises any
    ModelExplanation(Timeout)Error that anyio's task groups wrapped in an
    ExceptionGroup, so HTTP handlers can classify it.
    """

    agentic = kwargs.pop("agentic", None)
    if agentic is None:
        agentic = _env_bool("QWEN_AGENTIC_CHAT", DEFAULT_AGENTIC_CHAT)
    try:
        if agentic:
            return await _chat_with_mcp_agentic_inner(user_message, **kwargs)
        return await _chat_with_mcp_inner(user_message, **kwargs)
    except BaseException as exc:
        model_error = _extract_model_explanation_error(exc)
        if model_error is not None and model_error is not exc:
            raise model_error from exc
        raise


# Agent-facing tool names (short, model-friendly) mapped to the real MCP tools.
_AGENT_TOOL_TO_MCP = {
    "list_sensors": "list_building_sensors",
    "check_all_sensors": "run_autonomous_sensor_check",
    "check_sensor": "run_building_sensor_agent",
    "read_window_stats": "read_building_sensor_window",
    "psd_summary": "read_building_sensor_psd",
    "psd_compare": "compare_building_sensor_psd",
}

AGENTIC_SYSTEM_PROMPT = """You are the VibroAgent building-vibration operator agent running fully on the device NPU.
You decide which measurement tool to run, what the observations mean, and what to tell the operator.
Building monitoring is relative triage against the configured reference sensor, not a certified structural-safety assessment.
Never state measurements, labels, or sensor names you have not received in a tool observation this turn.
Respond with ONE strict JSON object per turn and nothing else. Stop immediately after the closing brace.
Either call a tool:
{"action":"tool","tool":"<name>","args":{}}
or finish with the operator answer:
{"action":"final","answer":"<clear operator answer, at most 120 words>"}
Tools:
- list_sensors: {} -> configured sensors and the reference sensor
- check_all_sensors: {"duration_s":10} -> compare every target sensor with the reference sensor
- check_sensor: {"sensor_id":"target_1","duration_s":10} -> compare one target sensor with the reference
- read_window_stats: {"sensor_id":"baseline","duration_s":5} -> compact raw statistics for one sensor
- psd_summary: {"sensor_id":"baseline","duration_s":10} -> spectrum peaks, frequency-band powers, impact/ring-down status
- psd_compare: {"sensor_ids":["baseline","target_1"],"duration_s":8} -> PSD of the SAME wall-clock window on several sensors (all when omitted), with band/peak deltas vs the reference
Call at most one tool per turn. Never repeat a tool call with the same arguments.
When the observations already answer the operator, respond with action:final.
Use the labels exactly as the observations report them.
"""

# Decoder-level constraint for agent steps: which action to take stays entirely
# the model's decision; this only pins the FORM of each step to one valid JSON
# object. The GenieX adapter compiles it to a llama.cpp GBNF grammar, so a step
# cannot be malformed; the local Genie adapter ignores response_format
# (verified by scripts/model_backend_probe.py), making it safe to send always.
AGENT_ACTION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_action",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["tool", "final"]},
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "answer": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def _first_json_object(text: str) -> dict[str, Any]:
    """Return the FIRST balanced top-level JSON object in a model step.

    The deployed finetune reliably emits a correct action object first but has
    weak stop discipline and may append repeated or malformed objects after it
    (verified on-device). Taking the first object verbatim and discarding the
    trailing junk is framing, not decision substitution.
    """

    stripped = str(text or "").strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model step contained no JSON object")
    obj, _end = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(obj, dict):
        raise ValueError("model step was not a JSON object")
    return obj


def _agentic_observation_text(agent_tool: str, mcp_tool: str, payload: Any, *, max_chars: int = 1600) -> str:
    if mcp_tool in {"run_autonomous_sensor_check", "run_building_sensor_agent"}:
        compact: Any = _essential_result_fields(payload)
    else:
        compact = _compact_json_value(payload, list_limit=10, string_limit=400)
    text = json.dumps({"observation": {"tool": agent_tool, "result": compact}}, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > max_chars:
        text = text[: max_chars - 22] + '"…truncated…"}}'
    return text


async def _chat_with_mcp_agentic_inner(
    user_message: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    max_prompt_chars: int | None = None,
    model_tool_routing: bool | None = None,
) -> dict[str, Any]:
    """Agentic chat turn: the model drives everything, step by step.

    Every step is one bounded NPU generation returning one strict action JSON
    object: the model chooses the tool (or answers), interprets observations,
    and authors the final prose. Deterministic code only executes the chosen
    tools, compacts observations, and validates the protocol. There is no
    deterministic routing and no fallback answer: protocol violations beyond a
    small correction budget, tool-side model failures, and out-of-domain
    output all raise explicit errors.
    """

    del model_tool_routing  # the model always routes in agentic mode

    base_url = base_url or os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:1234/v1")
    model = model or os.environ.get("QWEN_MODEL", "qwen3.5-4b-instruct-revised")
    resolved_timeout_s = (
        _env_float("QWEN_TIMEOUT_S", DEFAULT_QWEN_TIMEOUT_S, minimum=1.0)
        if timeout_s is None
        else max(1.0, float(timeout_s))
    )
    resolved_max_tokens = (
        _env_int("QWEN_MAX_TOKENS", DEFAULT_QWEN_MAX_TOKENS, minimum=32)
        if max_tokens is None
        else max(32, int(max_tokens))
    )
    resolved_max_prompt_chars = (
        _env_int("QWEN_MAX_PROMPT_CHARS", DEFAULT_QWEN_MAX_PROMPT_CHARS, minimum=2000)
        if max_prompt_chars is None
        else max(2000, int(max_prompt_chars))
    )
    max_steps = _env_int("QWEN_AGENTIC_MAX_STEPS", DEFAULT_AGENTIC_MAX_STEPS, minimum=1)
    max_steps = min(8, max_steps)
    local_request_body = _local_genie_timeout_body(base_url, resolved_timeout_s)

    resolved_api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
    client = _load_openai_client(base_url=base_url, api_key=resolved_api_key, timeout_s=resolved_timeout_s)
    ClientSession, StdioServerParameters, stdio_client = _load_mcp_client_symbols()

    mcp_env = dict(os.environ)
    mcp_env.update(
        {
            "QWEN_BASE_URL": base_url,
            "QWEN_API_KEY": resolved_api_key,
            "QWEN_MODEL": model,
            "SMALL_AGENT_BASE_URL": base_url,
            "SMALL_AGENT_API_KEY": resolved_api_key,
            "SMALL_AGENT_MODEL": model,
            "MAIN_AGENT_BASE_URL": base_url,
            "MAIN_AGENT_API_KEY": resolved_api_key,
            "MAIN_AGENT_MODEL": model,
            "AGENT_MODEL_TIMEOUT_S": str(resolved_timeout_s),
            "AGENT_MODEL_MAX_TOKENS": str(
                _env_int("AGENT_MODEL_MAX_TOKENS", DEFAULT_AGENT_MODEL_MAX_TOKENS, minimum=128)
            ),
        }
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vibroagent_mcp.mcp_server", "--transport", "stdio"],
        env=mcp_env,
    )

    model_calls: list[dict[str, Any]] = []
    agent_steps: list[dict[str, Any]] = []
    mcp_calls: list[dict[str, Any]] = []
    executed_tools: list[dict[str, Any]] = []
    observation_indexes: list[int] = []
    corrections_used = 0
    final_prompt_chars = 0
    last_call_key: str | None = None
    last_call_payload: Any = None

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_response = await session.list_tools()
            available_mcp_tools = {
                str(getattr(tool, "name", ""))
                for tool in tool_response.tools
                if str(getattr(tool, "name", "")) in ALLOWED_MCP_TOOL_NAMES
            }
            agent_tools = {
                agent_name: mcp_name
                for agent_name, mcp_name in _AGENT_TOOL_TO_MCP.items()
                if mcp_name in available_mcp_tools
            }

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user", "content": f"Operator request:\n{user_message}"},
            ]
            step_request_body = dict(local_request_body or {})
            step_request_body["response_format"] = AGENT_ACTION_RESPONSE_FORMAT

            def _prompt_chars() -> int:
                return sum(len(str(message.get("content") or "")) for message in messages)

            def _shrink_history() -> None:
                # Keep the newest observation intact; older ones collapse to a
                # stub so a long session cannot overflow the 4k context.
                for index in observation_indexes[:-1]:
                    messages[index]["content"] = '{"observation":"omitted_for_context_budget"}'

            answer: str | None = None
            for step in range(1, max_steps + 2):
                forced_final = step > max_steps
                if forced_final:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Step limit reached. Respond now with exactly one "
                                '{"action":"final","answer":"..."} object using only the observations above.'
                            ),
                        }
                    )
                if _prompt_chars() > resolved_max_prompt_chars:
                    _shrink_history()

                step_started_s = _monotonic()
                step_max_tokens = resolved_max_tokens if forced_final else min(192, resolved_max_tokens)
                try:
                    completion = _create_chat_completion(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout_s=resolved_timeout_s,
                        max_tokens=step_max_tokens,
                        extra_body=step_request_body,
                    )
                except Exception as exc:
                    error_text = format_exception_for_response(exc)
                    model_calls.append(
                        {"stage": f"agent_step_{step}", "ok": False, "duration_s": _monotonic() - step_started_s, "error": error_text}
                    )
                    if _looks_like_timeout_failure(exc):
                        raise ModelExplanationTimeoutError(
                            "The model timed out during an agent step. No fallback answer was generated."
                        ) from exc
                    raise ModelExplanationError(
                        f"The model failed during an agent step. No fallback answer was generated. ({error_text})"
                    ) from exc

                raw_text = completion.choices[0].message.content or ""
                model_calls.append(
                    {"stage": f"agent_step_{step}", "ok": True, "duration_s": _monotonic() - step_started_s}
                )

                try:
                    action = _first_json_object(raw_text)
                    action_kind = str(action.get("action") or "").strip().lower()
                    if action_kind not in {"tool", "final"}:
                        raise ValueError("action must be 'tool' or 'final'")
                except ValueError as exc:
                    corrections_used += 1
                    agent_steps.append({"step": step, "kind": "protocol_correction", "ok": False})
                    if corrections_used > AGENTIC_MAX_PROTOCOL_CORRECTIONS or forced_final:
                        raise ModelExplanationError(
                            "The model did not follow the agent step protocol. "
                            f"No fallback answer was generated. ({format_exception_for_response(exc)})"
                        ) from exc
                    messages.append({"role": "assistant", "content": raw_text[:400]})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That was not one valid action JSON object. Respond again with exactly one "
                                '{"action":"tool",...} or {"action":"final",...} object and nothing else.'
                            ),
                        }
                    )
                    continue

                if action_kind == "final":
                    answer = str(action.get("answer") or "").strip()
                    final_prompt_chars = _prompt_chars()
                    agent_steps.append({"step": step, "kind": "final", "ok": bool(answer)})
                    break

                if forced_final:
                    raise ModelExplanationError(
                        "The model kept requesting tools past the step limit and never answered. "
                        "No fallback answer was generated."
                    )

                agent_tool = str(action.get("tool") or "").strip()
                raw_args = action.get("args")
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                arguments = dict(raw_args) if isinstance(raw_args, dict) else {}

                mcp_tool = agent_tools.get(agent_tool)
                if mcp_tool is None:
                    agent_steps.append({"step": step, "kind": "unknown_tool", "tool": agent_tool, "ok": False})
                    messages.append({"role": "assistant", "content": json.dumps(action, separators=(",", ":"))})
                    observation = json.dumps(
                        {"observation": {"error": "unknown_tool", "requested": agent_tool, "known_tools": sorted(agent_tools)}},
                        separators=(",", ":"),
                    )
                    observation_indexes.append(len(messages))
                    messages.append({"role": "user", "content": observation + "\nRespond with the next action JSON object."})
                    continue

                arguments = _normalize_tool_arguments(mcp_tool, arguments, user_message)
                call_key = f"{mcp_tool}:{json.dumps(arguments, sort_keys=True, default=str)}"
                repeated_call = call_key == last_call_key
                if repeated_call:
                    # Identical call to the previous step: reuse the observation
                    # instead of re-reading the boards and re-running the tool's
                    # model decision. Execution caching only — the model still
                    # sees the data and still decides what to do next.
                    result_payload = last_call_payload
                else:
                    result = await session.call_tool(mcp_tool, arguments=arguments)
                    result_payload = _extract_tool_result(result)
                    last_call_key = call_key
                    last_call_payload = result_payload
                mcp_calls.append(
                    {
                        "source": "agentic_model",
                        "tool_call_id": f"agent_step_{step}_{mcp_tool}",
                        "name": mcp_tool,
                        "arguments": arguments,
                        "raw_arguments": json.dumps(arguments, default=str),
                        "note": (
                            f"The model repeated {agent_tool!r} with identical arguments in agent step {step}; "
                            "the previous observation was reused."
                            if repeated_call
                            else f"The model selected {agent_tool!r} in agent step {step}."
                        ),
                    }
                )
                executed_tools.append(
                    {"name": mcp_tool, "arguments": arguments, "result": result_payload, "source": "agentic_model"}
                )
                agent_steps.append(
                    {"step": step, "kind": "tool", "tool": agent_tool, "mcp_tool": mcp_tool, "ok": True, "repeated": repeated_call}
                )

                if isinstance(result_payload, dict) and result_payload.get("error") == "model_decision_unavailable":
                    # The tool's own model decision failed; per the LLM-only
                    # policy this is an explicit error, not something the loop
                    # narrates around.
                    _raise_tool_model_explanation_failure(executed_tools)

                messages.append({"role": "assistant", "content": json.dumps(action, separators=(",", ":"))})
                observation_indexes.append(len(messages))
                observation_text = _agentic_observation_text(agent_tool, mcp_tool, result_payload)
                if repeated_call:
                    observation_text = (
                        '{"note":"identical call repeated; observation unchanged"}\n' + observation_text
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": observation_text
                        + "\nRespond with the next action JSON object: call a different tool only if needed, otherwise finish with action:final.",
                    }
                )

            if answer is None:
                raise ModelExplanationError(
                    "The model never produced a final answer. No fallback answer was generated."
                )
            if not answer:
                raise ModelExplanationError(
                    "The model returned an empty final answer. No fallback answer was generated."
                )
            if _contains_text_tool_call(answer):
                raise ModelExplanationError(
                    "The model returned a tool call instead of a final answer. No fallback answer was generated."
                )

            tool_model_attempted, tool_model_used = _tool_result_model_activity(executed_tools)
            if tool_model_attempted and not tool_model_used:
                _raise_tool_model_explanation_failure(executed_tools)

            domain_guard_rejected_terms = sorted(
                set(find_out_of_domain_component_terms(answer))
                | set(find_out_of_domain_component_terms(executed_tools))
            )
            if find_out_of_domain_component_terms(answer):
                raise ModelExplanationError(
                    "The model explanation left the building-vibration domain and was rejected. "
                    "No fallback answer was generated."
                )

            response = {
                "answer": answer,
                "model": model,
                "base_url": base_url,
                "tools_available": sorted(agent_tools),
                "mcp_calls": mcp_calls,
                "tool_results": executed_tools,
                "used_fallback_tool": False,
                "model_unreachable": False,
                "tool_request_error": None,
                "final_fallback_error": None,
                "routing_mode": "agentic",
                "agent_steps": agent_steps,
                "max_steps": max_steps,
                "model_calls": model_calls,
                "model_call_count": len(model_calls),
                "timeout_s": resolved_timeout_s,
                "max_tokens": resolved_max_tokens,
                "final_prompt_chars": final_prompt_chars,
                "used_tool_agent_explanation": False,
                "used_host_building_summary": False,
                "tool_model_attempted": tool_model_attempted,
                "tool_model_used": tool_model_used,
                "model_request_sent": True,
                "model_response_used": True,
                "model_explanation_only": True,
                "fallbacks_disabled": True,
                "answer_source": "agentic_model",
                "domain_guard_triggered": bool(domain_guard_rejected_terms),
                "domain_guard_rejected_term_count": len(domain_guard_rejected_terms),
            }
            return redact_out_of_domain_component_text(response)


async def _chat_with_mcp_inner(
    user_message: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    max_prompt_chars: int | None = None,
    model_tool_routing: bool | None = None,
) -> dict[str, Any]:
    """Send one chat turn through one building-monitoring MCP tool.

    Tool routing is deterministic by default, so it does not consume a model
    generation. Agent tools make their own single structured model request.
    Read/list tools do not, so the host makes one bounded final-answer request
    after the tool returns. Only a verified model-generated explanation is ever
    returned as the chat answer. If that model request times out, fails, is
    rejected, or produces no explanation, this function raises an explicit
    error; it never substitutes a deterministic fallback answer.
    """

    base_url = base_url or os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:1234/v1")
    model = model or os.environ.get("QWEN_MODEL", "qwen3.5-4b-instruct-revised")
    resolved_timeout_s = (
        _env_float("QWEN_TIMEOUT_S", DEFAULT_QWEN_TIMEOUT_S, minimum=1.0)
        if timeout_s is None
        else max(1.0, float(timeout_s))
    )
    resolved_max_tokens = (
        _env_int("QWEN_MAX_TOKENS", DEFAULT_QWEN_MAX_TOKENS, minimum=32)
        if max_tokens is None
        else max(32, int(max_tokens))
    )
    resolved_max_prompt_chars = (
        _env_int("QWEN_MAX_PROMPT_CHARS", DEFAULT_QWEN_MAX_PROMPT_CHARS, minimum=2000)
        if max_prompt_chars is None
        else max(2000, int(max_prompt_chars))
    )
    routing_with_model = (
        _env_bool("QWEN_MODEL_TOOL_ROUTING", DEFAULT_MODEL_TOOL_ROUTING)
        if model_tool_routing is None
        else bool(model_tool_routing)
    )
    local_request_body = _local_genie_timeout_body(base_url, resolved_timeout_s)

    resolved_api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
    client = _load_openai_client(
        base_url=base_url,
        api_key=resolved_api_key,
        timeout_s=resolved_timeout_s,
    )
    ClientSession, StdioServerParameters, stdio_client = _load_mcp_client_symbols()

    # The browser can select a different endpoint/model from the process-wide
    # defaults.  The MCP server is a child process, so explicitly forward the
    # selected values to its building agents instead of silently falling back to
    # stale launch-time environment variables.
    mcp_env = dict(os.environ)
    mcp_env.update(
        {
            "QWEN_BASE_URL": base_url,
            "QWEN_API_KEY": resolved_api_key,
            "QWEN_MODEL": model,
            "SMALL_AGENT_BASE_URL": base_url,
            "SMALL_AGENT_API_KEY": resolved_api_key,
            "SMALL_AGENT_MODEL": model,
            "MAIN_AGENT_BASE_URL": base_url,
            "MAIN_AGENT_API_KEY": resolved_api_key,
            "MAIN_AGENT_MODEL": model,
            "AGENT_MODEL_TIMEOUT_S": str(resolved_timeout_s),
            "AGENT_MODEL_MAX_TOKENS": str(
                _env_int(
                    "AGENT_MODEL_MAX_TOKENS",
                    DEFAULT_AGENT_MODEL_MAX_TOKENS,
                    minimum=128,
                )
            ),
        }
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vibroagent_mcp.mcp_server", "--transport", "stdio"],
        env=mcp_env,
    )

    model_calls: list[dict[str, Any]] = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_response = await session.list_tools()
            tools = [
                _mcp_tool_to_openai_tool(tool)
                for tool in tool_response.tools
                if str(getattr(tool, "name", "")) in ALLOWED_MCP_TOOL_NAMES
            ]
            available_tool_names = {tool["function"]["name"] for tool in tools}

            assistant_msg = None
            tool_request_error: str | None = None
            model_unreachable = False
            tool_calls: list[Any] = []

            if routing_with_model:
                routing_started_s = _monotonic()
                try:
                    first = _create_chat_completion(
                        client=client,
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=temperature,
                        timeout_s=resolved_timeout_s,
                        max_tokens=min(128, resolved_max_tokens),
                        extra_body=local_request_body,
                        tools=tools,
                        tool_choice="auto",
                    )
                    assistant_msg = first.choices[0].message
                    tool_calls = list(getattr(assistant_msg, "tool_calls", None) or [])
                    model_calls.append(
                        {
                            "stage": "tool_routing",
                            "ok": True,
                            "duration_s": _monotonic() - routing_started_s,
                        }
                    )
                except Exception as exc:  # pragma: no cover - depends on local model runtime
                    tool_request_error = format_exception_for_response(exc)
                    model_unreachable = _is_model_connection_error(exc)
                    model_calls.append(
                        {
                            "stage": "tool_routing",
                            "ok": False,
                            "duration_s": _monotonic() - routing_started_s,
                            "error": tool_request_error,
                        }
                    )

            planned_calls: list[dict[str, Any]] = []
            if tool_calls:
                for call in tool_calls:
                    name = str(call.function.name)
                    raw_arguments = call.function.arguments or "{}"
                    parse_error = None
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError as exc:
                        arguments = {}
                        parse_error = f"{exc.__class__.__name__}: {exc}"
                    arguments = _normalize_tool_arguments(name, arguments, user_message)
                    source = "model_tool_call"
                    if name not in available_tool_names:
                        unsupported_tool = name
                        name, arguments = _supported_inferred_tool_call(user_message, available_tool_names)
                        arguments = _normalize_tool_arguments(name, arguments, user_message)
                        source = "unsupported_model_tool_fallback"
                        parse_error = (
                            f"UnsupportedTool: {unsupported_tool!r} is not available; "
                            f"used {name!r} instead."
                        )
                    planned_calls.append(
                        {
                            "source": source,
                            "tool_call_id": str(call.id),
                            "name": name,
                            "arguments": arguments,
                            "raw_arguments": raw_arguments,
                            "parse_error": parse_error,
                        }
                    )
            else:
                text_tool_call = (
                    _parse_text_tool_call(getattr(assistant_msg, "content", None))
                    if routing_with_model and assistant_msg is not None
                    else None
                )
                if text_tool_call is not None:
                    name, arguments = text_tool_call
                    source = "text_tool_call"
                    note = "The model returned a Qwen text tool call; the host parsed and executed it."
                else:
                    name, arguments = _supported_inferred_tool_call(user_message, available_tool_names)
                    source = "deterministic_router"
                    note = (
                        "The host selected the MCP tool deterministically, avoiding a redundant "
                        "local-model routing generation."
                    )
                arguments = _normalize_tool_arguments(name, arguments, user_message)
                if name not in available_tool_names:
                    unsupported_tool = name
                    name, arguments = _supported_inferred_tool_call(user_message, available_tool_names)
                    arguments = _normalize_tool_arguments(name, arguments, user_message)
                    source = "unsupported_tool_fallback"
                    note = (
                        f"The requested tool {unsupported_tool!r} is unavailable; "
                        "the host selected a supported building-monitoring tool."
                    )
                planned_calls.append(
                    {
                        "source": source,
                        "tool_call_id": f"{source}_{name}",
                        "name": name,
                        "arguments": arguments,
                        "raw_arguments": json.dumps(arguments),
                        "note": note,
                    }
                )

            mcp_calls: list[dict[str, Any]] = []
            executed_tools: list[dict[str, Any]] = []
            used_fallback_tool = False
            for planned in planned_calls:
                result = await session.call_tool(planned["name"], arguments=planned["arguments"])
                result_payload = _extract_tool_result(result)
                call_payload = {key: value for key, value in planned.items() if value is not None}
                mcp_calls.append(call_payload)
                executed_tools.append(
                    {
                        "name": planned["name"],
                        "arguments": planned["arguments"],
                        "result": result_payload,
                        "source": planned["source"],
                    }
                )
                if "fallback" in planned["source"]:
                    used_fallback_tool = True

            final_prompt_chars = 0
            result_has_agent_explanation = _result_has_agent_explanation(executed_tools)
            tool_model_attempted, tool_model_used = _tool_result_model_activity(executed_tools)
            safe_executed_tools = redact_out_of_domain_component_text(executed_tools)
            answer_source = "host_model"

            if tool_model_attempted:
                # The agent tool has already consumed the one model attempt for
                # this turn. Return its exact model-authored explanation, never a
                # deterministic summary. A failed/rejected agent model response is
                # surfaced as an error instead of being retried or hidden.
                if not tool_model_used:
                    _raise_tool_model_explanation_failure(executed_tools)
                answer = _model_agent_explanation(executed_tools)
                if not answer:
                    raise ModelExplanationError(
                        "The model completed the building analysis but did not provide a verified "
                        "model-generated explanation. No fallback answer was generated."
                    )
                answer_source = "tool_agent_model"
            else:
                final_started_s = _monotonic()
                try:
                    answer, final_prompt_chars = _request_final_answer(
                        client=client,
                        model=model,
                        user_message=user_message,
                        tool_results=safe_executed_tools,
                        temperature=temperature,
                        timeout_s=resolved_timeout_s,
                        max_tokens=resolved_max_tokens,
                        max_prompt_chars=resolved_max_prompt_chars,
                        extra_body=local_request_body,
                    )
                    model_calls.append(
                        {
                            "stage": "final_answer",
                            "ok": True,
                            "duration_s": _monotonic() - final_started_s,
                            "prompt_chars": final_prompt_chars,
                        }
                    )
                except Exception as exc:  # pragma: no cover - depends on local model runtime
                    error_text = format_exception_for_response(exc)
                    model_calls.append(
                        {
                            "stage": "final_answer",
                            "ok": False,
                            "duration_s": _monotonic() - final_started_s,
                            "prompt_chars": final_prompt_chars,
                            "error": error_text,
                        }
                    )
                    if _looks_like_timeout_failure(exc):
                        raise ModelExplanationTimeoutError(
                            "The model timed out while generating the webchat explanation. "
                            "No fallback answer was generated."
                        ) from exc
                    raise ModelExplanationError(
                        "The model could not generate the webchat explanation. "
                        f"No fallback answer was generated. ({error_text})"
                    ) from exc

                if _contains_text_tool_call(answer):
                    raise ModelExplanationError(
                        "The model returned a tool call instead of an explanation. "
                        "No fallback answer was generated."
                    )
                if not answer.strip():
                    raise ModelExplanationError(
                        "The model returned an empty explanation. No fallback answer was generated."
                    )

            domain_guard_rejected_terms = sorted(
                set(find_out_of_domain_component_terms(answer))
                | set(find_out_of_domain_component_terms(executed_tools))
            )
            if find_out_of_domain_component_terms(answer):
                raise ModelExplanationError(
                    "The model explanation left the building-vibration domain and was rejected. "
                    "No fallback answer was generated."
                )

            response = {
                "answer": answer.strip(),
                "model": model,
                "base_url": base_url,
                "tools_available": [tool["function"]["name"] for tool in tools],
                "mcp_calls": mcp_calls,
                "tool_results": executed_tools,
                "used_fallback_tool": used_fallback_tool,
                "model_unreachable": False,
                "tool_request_error": tool_request_error,
                "final_fallback_error": None,
                "routing_mode": "model" if routing_with_model else "deterministic",
                "model_calls": model_calls,
                "model_call_count": len(model_calls),
                "timeout_s": resolved_timeout_s,
                "max_tokens": resolved_max_tokens,
                "final_prompt_chars": final_prompt_chars,
                "used_tool_agent_explanation": answer_source == "tool_agent_model",
                "used_host_building_summary": False,
                "tool_model_attempted": tool_model_attempted,
                "tool_model_used": tool_model_used,
                "model_request_sent": True,
                "model_response_used": True,
                "model_explanation_only": True,
                "fallbacks_disabled": True,
                "answer_source": answer_source,
                "domain_guard_triggered": bool(domain_guard_rejected_terms),
                "domain_guard_rejected_term_count": len(domain_guard_rejected_terms),
            }
            return redact_out_of_domain_component_text(response)


def _monotonic() -> float:
    # Kept behind a helper so timing can be controlled in focused unit tests.
    import time

    return time.perf_counter()


def _create_chat_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    timeout_s: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Give the HTTP connection a small grace period beyond the server-side
        # generation budget so the adapter can return its timeout/fallback error
        # instead of the socket closing at the same instant.
        "timeout": timeout_s + HTTP_TIMEOUT_GRACE_S,
    }
    # ``timeout_s`` is a private field understood by this repository's local
    # Genie adapter.  Do not leak it to arbitrary OpenAI-compatible services,
    # which may reject unknown request-body fields.
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def _request_final_answer(
    *,
    client: Any,
    model: str,
    user_message: str,
    tool_results: list[dict[str, Any]],
    temperature: float,
    timeout_s: float,
    max_tokens: int,
    max_prompt_chars: int,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, int]:
    compact_payload = _compact_tool_results_for_model(
        tool_results,
        max_chars=max(1200, max_prompt_chars - len(FINAL_SYSTEM_PROMPT) - len(FINAL_ANSWER_PROMPT) - len(user_message) - 500),
    )
    tool_payload = json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"), default=str)
    user_content = (
        f"Operator request:\n{user_message}\n\n"
        f"MCP result:\n{tool_payload}\n\n"
        f"{FINAL_ANSWER_PROMPT}"
    )
    messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_chars = len(FINAL_SYSTEM_PROMPT) + len(user_content)
    if prompt_chars > max_prompt_chars:
        # The compact helper normally prevents this.  This final guard keeps a
        # very long operator request from overflowing a 4096-token model.
        available = max(200, max_prompt_chars - len(FINAL_SYSTEM_PROMPT) - len(FINAL_ANSWER_PROMPT) - 200)
        clipped_request = user_message[: min(len(user_message), 1200)]
        clipped_payload = tool_payload[:available]
        user_content = (
            f"Operator request:\n{clipped_request}\n\n"
            f"Host building-monitoring summary:\n{_summarize_tool_results(tool_results)}\n\n"
            f"Compact payload prefix:\n{clipped_payload}\n\n{FINAL_ANSWER_PROMPT}"
        )
        messages[1]["content"] = user_content
        prompt_chars = len(FINAL_SYSTEM_PROMPT) + len(user_content)

    final = _create_chat_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    final_message = final.choices[0].message
    if getattr(final_message, "tool_calls", None):
        return "<tool_call></tool_call>", prompt_chars
    return final_message.content or "", prompt_chars


_HEAVY_RESULT_ARRAY_KEYS = {
    "samples",
    "samples_g",
    "preview_samples",
    "time_s",
    "timestamps_s",
    "frequency_hz",
    "psd_g2_per_hz",
    "asd_g_per_sqrt_hz",
    "peak_envelope_frequency_hz",
    "peak_envelope_psd_g2_per_hz",
    "raw_signal",
    "waveform",
}


def _compact_json_value(
    value: Any,
    *,
    list_limit: int = 10,
    string_limit: int = 800,
    depth: int = 0,
) -> Any:
    if depth > 8:
        return "<nested value omitted>"
    value = _decode_jsonish(value)
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _HEAVY_RESULT_ARRAY_KEYS and isinstance(item, (list, tuple)):
                compact[key_text] = {"omitted_values": len(item)}
                continue
            compact[key_text] = _compact_json_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                depth=depth + 1,
            )
        return compact
    if isinstance(value, (list, tuple)):
        items = [
            _compact_json_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                depth=depth + 1,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            items.append({"omitted_items": len(value) - list_limit})
        return items
    if isinstance(value, str) and len(value) > string_limit:
        return value[:string_limit] + f"… <{len(value) - string_limit} chars omitted>"
    return value


def _essential_result_fields(result: Any) -> dict[str, Any]:
    decoded = _decode_jsonish(result)
    if not isinstance(decoded, dict):
        return {"value": _compact_json_value(decoded, list_limit=4, string_limit=400)}
    keys = (
        "status",
        "error",
        "message",
        "analysis_domain",
        "window_id",
        "baseline_sensor_id",
        "prediction",
        "features",
        "baseline_comparison",
        "network_assessment",
        "sensor_agent_reports",
        "model_metadata",
        "source_metadata",
        "sdk_vibrometer",
        "metadata",
        "data_source_warning",
        "sample_count",
        "sampling_rate_hz",
        "window_duration_s",
        "rms",
    )
    return {
        key: _compact_json_value(decoded[key], list_limit=6, string_limit=500)
        for key in keys
        if key in decoded
    }


def _compact_tool_results_for_model(tool_results: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
    compact = [
        {
            "name": item.get("name"),
            "arguments": _compact_json_value(item.get("arguments") or {}, list_limit=8, string_limit=400),
            "result": _compact_json_value(item.get("result"), list_limit=10, string_limit=800),
        }
        for item in tool_results
    ]
    if len(json.dumps(compact, ensure_ascii=False, default=str)) <= max_chars:
        return compact

    essential = [
        {
            "name": item.get("name"),
            "arguments": _compact_json_value(item.get("arguments") or {}, list_limit=4, string_limit=250),
            "result": _essential_result_fields(item.get("result")),
        }
        for item in tool_results
    ]
    if len(json.dumps(essential, ensure_ascii=False, default=str)) <= max_chars:
        return essential

    return [
        {
            "host_summary": _summarize_tool_results(tool_results),
            "tools": [str(item.get("name") or "unknown") for item in tool_results],
            "note": "Large arrays and secondary metadata were omitted to fit the model context.",
        }
    ]


def _parse_text_tool_call(content: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(content, str) or "<tool_call" not in content:
        return None
    function_match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", content, flags=re.DOTALL)
    if not function_match:
        return None
    name = function_match.group(1)
    body = function_match.group(2)
    arguments = {
        match.group(1): _parse_text_tool_value(match.group(2))
        for match in re.finditer(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>", body, flags=re.DOTALL)
    }
    return name, arguments


def _parse_text_tool_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _contains_text_tool_call(content: Any) -> bool:
    return isinstance(content, str) and "<tool_call" in content and "</tool_call>" in content


def _decode_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return _decode_jsonish(json.loads(text))
        except json.JSONDecodeError:
            return value
    if isinstance(value, list):
        if len(value) == 1:
            return _decode_jsonish(value[0])
        return [_decode_jsonish(item) for item in value]
    if isinstance(value, dict):
        if "structuredContent" in value:
            return _decode_jsonish(value["structuredContent"])
        content = value.get("content")
        if isinstance(content, list) and content:
            return _decode_jsonish(content[0])
        if value.get("type") == "text" and "text" in value:
            return _decode_jsonish(value["text"])
    return value


def _tool_result_dict(tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tool_results:
        return None
    result = _decode_jsonish(tool_results[0].get("result"))
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        for item in result:
            decoded = _decode_jsonish(item)
            if isinstance(decoded, dict):
                return decoded
    return None


_TIMEOUT_FAILURE_MARKERS = (
    "timeout",
    "timed out",
    "deadline exceeded",
    "gateway timeout",
    "timeout_error",
    "error code: 504",
    "status code: 504",
    "http 504",
)


def _looks_like_timeout_failure(value: Any) -> bool:
    """Recognize timeout failures from local Genie and OpenAI client wrappers."""

    seen: set[int] = set()
    current = value
    parts: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{current.__class__.__name__}: {current}" if isinstance(current, BaseException) else str(current))
        status_code = getattr(current, "status_code", None)
        if status_code is not None:
            parts.append(f"status code: {status_code}")
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    text = " ".join(parts).lower()
    return any(marker in text for marker in _TIMEOUT_FAILURE_MARKERS)


def _model_failure_reasons(tool_results: list[dict[str, Any]]) -> list[str]:
    result = _tool_result_dict(tool_results)
    if not isinstance(result, dict):
        return ["agent_tool_returned_no_structured_result"]

    reasons: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value.strip() not in reasons:
            reasons.append(value.strip())
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            add(value.get("fallback_reason"))
            add(value.get("fallbacks_used"))
            if value.get("model_used") is False:
                add(value.get("error"))
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(result.get("model_metadata", result))
    return reasons or ["model_generated_explanation_unavailable"]


def _raise_tool_model_explanation_failure(tool_results: list[dict[str, Any]]) -> None:
    reasons = _model_failure_reasons(tool_results)
    reason = next((item for item in reasons if _looks_like_timeout_failure(item)), reasons[0])
    if _looks_like_timeout_failure(reason):
        raise ModelExplanationTimeoutError(
            "The model timed out while generating the building-sensor explanation. "
            "No fallback answer was generated."
        )
    raise ModelExplanationError(
        "The model did not produce a valid building-sensor explanation. "
        f"No fallback answer was generated. ({reason})"
    )


def _model_agent_explanation(tool_results: list[dict[str, Any]]) -> str | None:
    """Return only a model-authored agent explanation with verified provenance."""

    result = _tool_result_dict(tool_results)
    if not isinstance(result, dict):
        return None
    if result.get("analysis_domain") != "building_vibration_relative_monitoring":
        return None

    assessment = result.get("network_assessment")
    if not isinstance(assessment, dict):
        assessment = {}
    explanation = result.get("main_agent_explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = assessment.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return None

    source = result.get("main_agent_explanation_source") or assessment.get("explanation_source")
    aggregate_metadata = result.get("model_metadata")
    aggregate_metadata = aggregate_metadata if isinstance(aggregate_metadata, dict) else {}
    network_metadata = assessment.get("model_metadata")
    network_metadata = network_metadata if isinstance(network_metadata, dict) else {}
    model_used = bool(aggregate_metadata.get("model_used") or network_metadata.get("model_used"))
    fallbacks = aggregate_metadata.get("fallbacks_used") or []
    if source != "model" or not model_used or bool(fallbacks):
        return None
    return explanation.strip()


def _result_has_agent_explanation(tool_results: list[dict[str, Any]]) -> bool:
    """Return true only for a verified model-authored agent explanation."""

    return _model_agent_explanation(tool_results) is not None


def _tool_result_model_activity(tool_results: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return ``(attempted, used)`` for model work performed inside an MCP tool.

    Building-agent results expose model metadata at several nested levels.  The
    aggregate object is convenient when present, while older/simpler results may
    only expose per-sensor or network metadata.  This recursive inspection keeps
    the host from issuing a duplicate NPU request after an agent already tried one.

    ``endpoint_not_configured`` and ``openai_client_not_installed`` are not real
    endpoint attempts, so the host is still allowed to make its own final-answer
    request using the browser-selected model configuration.
    """

    result = _tool_result_dict(tool_results)
    if not isinstance(result, dict):
        return False, False

    attempted = False
    used = False
    no_request_reasons = {
        "endpoint_not_configured",
        "openai_client_not_installed",
        "deterministic_precheck_only",
        "deterministic_rule_fallback",
    }

    def walk(value: Any) -> None:
        nonlocal attempted, used
        if isinstance(value, dict):
            if "model_used" in value or "endpoint_configured" in value or "provider" in value:
                model_used = bool(value.get("model_used"))
                used = used or model_used
                reason = str(value.get("fallback_reason") or value.get("error") or "").strip()
                endpoint_configured = bool(value.get("endpoint_configured"))
                provider = str(value.get("provider") or "").strip().lower()
                configured = endpoint_configured or provider == "openai_compatible"
                if model_used or (configured and reason not in no_request_reasons):
                    attempted = True
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(result.get("model_metadata", result))
    return attempted, used


def _summarize_tool_results(tool_results: list[dict[str, Any]]) -> str:
    """Build a deterministic building-only explanation for fallbacks."""
    result = _tool_result_dict(tool_results)
    if result is None:
        return "The building-monitoring tool ran, but no structured result was returned."

    if result.get("error") == "live_data_required":
        return str(result.get("message") or "No current building-sensor data is available.")

    if result.get("error") in {
        "sdk_read_failed",
        "building_sensor_source_unavailable",
        "sdk_sources_failed",
    }:
        return str(result.get("message") or "The selected building sensor could not be read.")

    if result.get("analysis_domain") == "building_vibration_relative_monitoring" and result.get("network_assessment"):
        assessment = result.get("network_assessment") or {}
        reports = result.get("sensor_agent_reports") or []
        label = (
            assessment.get("network_label")
            or assessment.get("main_agent_label")
            or assessment.get("network_status")
            or "unknown"
        )
        confidence = assessment.get("main_agent_confidence")
        confidence_text = f" with confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        affected = assessment.get("affected_sensor_ids") or []
        affected_text = ", ".join(str(item) for item in affected) if affected else "none"
        explanation = result.get("main_agent_explanation") or assessment.get("explanation") or ""
        summary = (
            f"The building sensor network was compared with reference {result.get('baseline_sensor_id', 'baseline')}. "
            f"The relative classification is {label}{confidence_text}. "
            f"Affected target sensors: {affected_text}. Sensors assessed: {len(reports)}. "
            f"{explanation}"
        ).strip()
        if find_out_of_domain_component_terms(summary):
            summary = (
                f"The building sensor network was compared with reference {result.get('baseline_sensor_id', 'baseline')}. "
                f"The relative classification is {label}{confidence_text}. "
                f"Affected target sensors: {affected_text}. Sensors assessed: {len(reports)}. "
                "Repeat the synchronized measurement and review the relative trend if the deviation persists."
            )
        return summary

    if result.get("analysis_domain") == "building_vibration_relative_monitoring":
        sensor_id = result.get("sensor_id") or result.get("machine_id") or "selected sensor"
        location = result.get("location")
        location_text = f" at {location}" if location else ""
        sample_count = result.get("sample_count", "unknown")
        duration = result.get("window_duration_s", "unknown")
        rms = result.get("rms", "unknown")
        sampling_rate = result.get("sampling_rate_hz", "unknown")
        status = result.get("status")
        if status == "error":
            return str(result.get("message") or f"The building sensor {sensor_id} could not be read.")
        return (
            f"Building sensor {sensor_id}{location_text} was read for {duration} seconds "
            f"({sample_count} samples at {sampling_rate} Hz). RMS acceleration is {rms} g. "
            "Compare this value with the synchronized reference and repeat the measurement if the relative level changes."
        )

    return "The tool result was outside the configured building-monitoring domain and was not presented."

def _is_model_connection_error(exc: Exception) -> bool:
    error_name = exc.__class__.__name__.lower()
    error_text = str(exc).lower()
    return (
        "connection" in error_name
        or "timeout" in error_name
        or "connect" in error_text
        or "connection" in error_text
        or "timed out" in error_text
    )


async def run_qwen_host(user_message: str) -> None:
    result = await chat_with_mcp(user_message)
    print(result["answer"])



def _normalize_tool_arguments(name: str, arguments: dict[str, Any], user_message: str) -> dict[str, Any]:
    if name not in ALLOWED_MCP_TOOL_NAMES:
        return arguments

    blank_keys = {"config_path", "sensor_id", "sensor_ids", "baseline_sensor_id", "axis", "start_time_s"}
    normalized = {
        key: value
        for key, value in arguments.items()
        if not (key in blank_keys and _is_blank_tool_value(value))
    }

    tools_with_currentness = {
        "read_building_sensor_window",
        "read_building_sensor_psd",
        "compare_building_sensor_psd",
        "run_autonomous_sensor_check",
        "run_building_sensor_agent",
        "run_building_vibration_agent_pipeline",
        "export_building_validation_dataset",
    }
    if name in tools_with_currentness:
        if "require_current" in normalized:
            normalized["require_current"] = _bool_tool_value(normalized["require_current"], default=True)
        else:
            normalized["require_current"] = _default_require_current(user_message)

        requested_duration = _requested_duration_s(user_message)
        if requested_duration is not None:
            normalized["duration_s"] = requested_duration
        elif "duration_s" in normalized:
            # A model tool call may request any number; keep it inside the
            # window the tools accept instead of letting every read fail.
            clamped_duration = _clamped_tool_duration_s(normalized["duration_s"])
            if clamped_duration is None:
                normalized.pop("duration_s", None)
            else:
                normalized["duration_s"] = clamped_duration

        if _looks_like_latest_request(user_message):
            normalized["require_current"] = True
            start_time = normalized.get("start_time_s")
            if _is_blank_tool_value(start_time) or start_time in (None, 0, 0.0, "0", "0.0"):
                normalized.pop("start_time_s", None)

    if name in {"read_building_sensor_window", "read_building_sensor_psd"}:
        normalized.setdefault("config_path", "config/sensors.live.yaml")
        normalized.setdefault("sensor_id", _requested_sensor_id(user_message) or "baseline")
    elif name == "compare_building_sensor_psd":
        normalized.setdefault("config_path", "config/sensors.live.yaml")
        # start_time_s belongs to the single-sensor tools; the aligned compare
        # always reads the same live wall instant on every sensor.
        normalized.pop("start_time_s", None)
        sensor_ids = normalized.get("sensor_ids")
        if isinstance(sensor_ids, str):
            # The model may emit "baseline,target_1" or "all".
            parts = [part.strip() for part in sensor_ids.split(",") if part.strip()]
            if not parts or any(part.lower() in {"all", "*"} for part in parts):
                normalized.pop("sensor_ids", None)
            else:
                normalized["sensor_ids"] = parts
        elif isinstance(sensor_ids, list):
            parts = [str(part).strip() for part in sensor_ids if str(part).strip()]
            if not parts or any(part.lower() in {"all", "*"} for part in parts):
                normalized.pop("sensor_ids", None)
            else:
                normalized["sensor_ids"] = parts
    elif name == "run_building_sensor_agent":
        normalized.setdefault("config_path", "config/sensors.live.yaml")
        requested = _requested_sensor_id(user_message)
        if requested and requested != "baseline":
            normalized.setdefault("sensor_id", requested)
        normalized.setdefault("baseline_sensor_id", "baseline")
    elif name in {"run_autonomous_sensor_check", "run_building_vibration_agent_pipeline"}:
        normalized.setdefault("config_path", "config/sensors.live.yaml")
        normalized.setdefault("baseline_sensor_id", "baseline")
    return normalized


def _is_blank_tool_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _bool_tool_value(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _looks_like_latest_request(user_message: str) -> bool:
    text = user_message.lower()
    return any(token in text for token in ("latest", "current", "now", "new", "recent", "live"))


def _looks_like_replay_request(user_message: str) -> bool:
    text = user_message.lower()
    return any(
        token in text
        for token in (
            "saved",
            "stale",
            "replay",
            "recorded",
            "offline",
            "historical",
            "history",
            "archive",
            "validation",
            "file",
            "csv",
        )
    )


def _looks_like_raw_read_request(user_message: str) -> bool:
    text = user_message.lower()
    if any(token in text for token in ("compare", "deviation", "network", "all sensors", "all boards")):
        return False
    explicit_read_terms = (
        "raw sample",
        "raw window",
        "preview sample",
        "sample preview",
        "read the signal",
        "rms of",
        "rms statistics",
        "signal statistics",
        "window statistics",
        "statistics for",
        "stats for",
        "window from",
    )
    if any(token in text for token in explicit_read_terms):
        return True
    return "read" in text and any(token in text for token in ("window", "signal", "samples", "statistics", "stats", "rms"))


def _looks_like_configuration_request(user_message: str) -> bool:
    text = user_message.lower()
    return any(
        token in text
        for token in (
            "list sensors",
            "configured sensors",
            "sensor configuration",
            "which sensors",
            "reference sensor id",
            "sensor locations",
        )
    )


def _default_require_current(user_message: str) -> bool:
    return not _looks_like_replay_request(user_message)


def _requested_sensor_id(user_message: str) -> str | None:
    text = user_message.lower()
    if any(token in text for token in ("baseline", "reference sensor", "reference board")):
        return "baseline"
    match = re.search(r"\b(?:target|sensor|board)[\s_-]*0*([1-9][0-9]*)\b", text)
    if match:
        return f"target_{int(match.group(1))}"
    return None


def _requested_duration_s(user_message: str) -> float | None:
    text = user_message.lower()
    patterns = (
        (r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", 1.0),
        (r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", 60.0),
    )
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            duration = float(match.group(1)) * multiplier
        except ValueError:
            continue
        if not math.isfinite(duration) or duration <= 0:
            continue
        return min(MAX_TOOL_WINDOW_DURATION_S, max(0.2, duration))
    return None


def _clamped_tool_duration_s(value: Any) -> float | None:
    """Clamp a tool-call duration to the window the MCP tools accept."""
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return min(MAX_TOOL_WINDOW_DURATION_S, max(0.2, duration))


def _tool_duration_s(user_message: str, default: float) -> float:
    return _requested_duration_s(user_message) or float(default)


def _is_building_tool(name: str) -> bool:
    return name in ALLOWED_MCP_TOOL_NAMES


def _infer_basic_tool_call(user_message: str) -> tuple[str, dict[str, Any]]:
    """Select only tools in the building-monitoring workflow."""
    if _looks_like_configuration_request(user_message):
        return "list_building_sensors", {"config_path": "config/sensors.live.yaml"}

    require_current = _default_require_current(user_message)
    requested_sensor = _requested_sensor_id(user_message)
    if _looks_like_raw_read_request(user_message):
        return (
            "read_building_sensor_window",
            {
                "config_path": "config/sensors.live.yaml",
                "sensor_id": requested_sensor or "baseline",
                "duration_s": _tool_duration_s(user_message, 5.0),
                "include_samples": False,
                "max_preview_samples": 16,
                "require_current": require_current,
            },
        )

    if requested_sensor and requested_sensor != "baseline" and not any(
        token in user_message.lower() for token in ("all sensors", "all boards", "network", "building")
    ):
        return (
            "run_building_sensor_agent",
            {
                "config_path": "config/sensors.live.yaml",
                "sensor_id": requested_sensor,
                "baseline_sensor_id": "baseline",
                "duration_s": _tool_duration_s(user_message, 10.0),
                "require_current": require_current,
            },
        )

    return (
        "run_autonomous_sensor_check",
        {
            "config_path": "config/sensors.live.yaml",
            "baseline_sensor_id": "baseline",
            "duration_s": _tool_duration_s(user_message, 10.0),
            "require_current": require_current,
        },
    )


def _supported_inferred_tool_call(user_message: str, available_tool_names: set[str]) -> tuple[str, dict[str, Any]]:
    available = set(available_tool_names) & ALLOWED_MCP_TOOL_NAMES
    preferred_name, preferred_args = _infer_basic_tool_call(user_message)
    if preferred_name in available:
        return preferred_name, preferred_args

    fallback_order = [
        "run_autonomous_sensor_check",
        "run_building_vibration_agent_pipeline",
        "run_building_sensor_agent",
        "read_building_sensor_window",
        "list_building_sensors",
    ]
    for name in fallback_order:
        if name not in available:
            continue
        if name in {"run_autonomous_sensor_check", "run_building_vibration_agent_pipeline"}:
            return (
                name,
                {
                    "config_path": "config/sensors.live.yaml",
                    "baseline_sensor_id": "baseline",
                    "duration_s": _tool_duration_s(user_message, 10.0),
                    "require_current": _default_require_current(user_message),
                },
            )
        if name == "run_building_sensor_agent":
            requested = _requested_sensor_id(user_message)
            return (
                name,
                {
                    "config_path": "config/sensors.live.yaml",
                    "sensor_id": requested if requested and requested != "baseline" else "",
                    "baseline_sensor_id": "baseline",
                    "duration_s": _tool_duration_s(user_message, 10.0),
                    "require_current": _default_require_current(user_message),
                },
            )
        if name == "read_building_sensor_window":
            return (
                name,
                {
                    "config_path": "config/sensors.live.yaml",
                    "sensor_id": _requested_sensor_id(user_message) or "baseline",
                    "duration_s": _tool_duration_s(user_message, 5.0),
                    "include_samples": False,
                    "max_preview_samples": 16,
                    "require_current": _default_require_current(user_message),
                },
            )
        return name, {"config_path": "config/sensors.live.yaml"}

    raise RuntimeError("No supported building-monitoring MCP tools are available from the server.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Building-vibration MCP host")
    parser.add_argument("message", nargs="+", help="User message")
    args = parser.parse_args()
    try:
        asyncio.run(run_qwen_host(" ".join(args.message)))
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
