"""Codes-v3 live monitor support: prompt building and reply parsing.

The codes_v3 SFT model (qwen3_4b_codes_v3_Q4_0_embq8.gguf) judges five
target sensors strictly relative to a reference sensor from 250 discrete
codec-v1 codes per 10 s / 400 Hz window plus a broadband level delta.
Its prompt is NOT reconstructed from the (since-drifted) builder script:
the byte-constant template was extracted from the immutable training
set (data/codes_sft_v3, 11 000 examples, one system message, one header,
two instruction tails, two alignment variants) into
``assets/codes_v3_prompt_template.json``; tests re-derive the asset from
the JSONL and byte-compare rebuilt prompts against stored examples.

This module is deliberately torch-free: codec extraction runs in the
codec-cpu-venv worker (scripts/live_codes_worker.py); here we only
assemble prompts and strictly validate the model's verdict and popup message.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_ASSET = Path(__file__).resolve().parent / "assets" / \
    "codes_v3_prompt_template.json"

# The fixed label vocabularies the codes_v3 assistant messages use —
# identical to the deterministic prepass vocabulary the webchat panel
# already renders.
SENSOR_LABELS = (
    "normal_relative_to_baseline",
    "mild_deviation",
    "significant_local_deviation",
    "sensor_data_quality_issue",
)
NETWORK_LABELS = (
    "normal_relative_to_reference_sensor",
    "mild_local_deviation",
    "localized_vibration_deviation",
    "mild_multi_sensor_deviation",
    "multi_sensor_building_wide_vibration_event",
)

LIVE_SLOTS = ("target_1", "target_2", "target_3", "target_4", "target_5")
CODES_PER_WINDOW = 250
POPUP_MESSAGE_MAX_CHARS = 480
POPUP_MESSAGE_MAX_WORDS = 60

_IMPLEMENTATION_TERM_RE = re.compile(
    r"\b(?:codes(?:_v3)?|qwen(?:3(?:\.\d+)?)?|geniex|codec(?:-v1)?|npu|hexagon)\b",
    re.IGNORECASE,
)
_TARGET_MENTION_RE = re.compile(r"\btarget[\s_-]?([1-5])\b", re.IGNORECASE)
_STRUCTURAL_CLAIM_RE = re.compile(
    r"\b(?:safe|unsafe|structurally sound)\b", re.IGNORECASE
)


class CodesPromptError(ValueError):
    """The requested prompt cannot be built faithfully to the template."""


def _template() -> dict:
    return json.loads(_ASSET.read_text(encoding="utf-8"))


def system_message() -> str:
    return _template()["system"]


def build_codes_user_prompt(
    reference_codes: str,
    targets: Sequence[tuple[str, float, str]],
    *,
    aligned: bool,
) -> str:
    """Render the exact codes_v3 user message.

    ``targets`` is an ordered sequence of (slot, level_rel_db, codes).
    Only the two shapes present in training are expressible: one target
    (slot ``target_1``) or the full five (``target_1``..``target_5`` in
    order) — the instruction tail embeds the expected-JSON shape and
    exists in exactly those two variants.
    """
    tpl = _template()
    slots = [slot for slot, _, _ in targets]
    if slots == list(LIVE_SLOTS):
        tail = tpl["tails"]["multi"]
    elif slots == ["target_1"]:
        tail = tpl["tails"]["single"]
    else:
        raise CodesPromptError(
            f"codes_v3 supports targets {list(LIVE_SLOTS)} or ['target_1'], "
            f"got {slots}")
    for what, codes in [("reference", reference_codes)] + \
            [(slot, c) for slot, _, c in targets]:
        if not isinstance(codes, str) or len(codes) != CODES_PER_WINDOW:
            raise CodesPromptError(
                f"{what}: expected a {CODES_PER_WINDOW}-symbol code string, "
                f"got {len(codes) if isinstance(codes, str) else type(codes)}")
        if any(ch.isspace() for ch in codes):
            raise CodesPromptError(f"{what}: code string contains whitespace")
    alignment = tpl["alignment_variants"]["aligned" if aligned else "fallback"]
    lines = [
        tpl["header"] + "Reference alignment: " + alignment,
        tpl["reference_line"].format(codes=reference_codes),
    ]
    for slot, level, codes in targets:
        lines.append(tpl["target_line"].format(
            slot=slot, level=float(level), codes=codes))
    return "\n".join(lines) + tail


def build_codes_popup_prompt(
    reference_codes: str,
    targets: Sequence[tuple[str, float, str]],
    *,
    aligned: bool,
    required_affected_sensor_ids: Sequence[str] | None = None,
) -> str:
    """Extend the trained verdict prompt with a model-authored popup field."""
    prompt = build_codes_user_prompt(reference_codes, targets, aligned=aligned)
    marker = "Return one compact strict JSON object only, exactly this shape:\n"
    try:
        prefix, shape = prompt.rsplit(marker, 1)
    except ValueError as exc:
        raise CodesPromptError("trained prompt has no response-shape marker") from exc
    if not shape.endswith("}"):
        raise CodesPromptError("trained prompt response shape is malformed")
    popup_shape = (
        shape[:-1]
        + ',"popup_message":"<complete operator-facing message>"}'
    )
    instruction = (
        "Write popup_message entirely in your own words as the finished "
        "operator alert. Use one to three concise plain-English sentences "
        "and no more than 60 words. When affected sensors exist, name every "
        "one using its target_N ID, describe the likely vibration pattern "
        "only when supported by the input, and end with the immediate next "
        "check. Do not mention models, software, implementation details, "
        "codes, or make a certified structural-safety conclusion. "
    )
    required = _validated_required_affected_sensor_ids(
        required_affected_sensor_ids,
        [slot for slot, _, _ in targets],
    )
    if required is not None:
        exact_ids = json.dumps(required, separators=(",", ":"))
        instruction += (
            "An independent raw-measurement identity check found a strong "
            "localized event. For this verdict, affected_sensor_ids must be "
            f"exactly {exact_ids}, and popup_message must name exactly those "
            "IDs. "
            "sensor_reports must mark exactly those IDs mild or significant "
            "and must mark every other target normal relative to baseline. "
            "This pins physical sensor identity only; write the complete "
            "interpretation and operator message yourself from the supplied "
            "vibration evidence. "
        )
    return prefix + instruction + marker + popup_shape


def codes_response_format(
    targets: Sequence[str],
    *,
    required_affected_sensor_ids: Sequence[str] | None = None,
) -> dict:
    """json_schema response_format pinning the reply shape (geniex GBNF)."""
    target_list = list(targets)
    required_affected = _validated_required_affected_sensor_ids(
        required_affected_sensor_ids,
        target_list,
    )
    affected_items = required_affected if required_affected is not None \
        else target_list
    affected_schema: dict[str, Any] = {
        "type": "array",
        "items": {"enum": affected_items},
    }
    network_labels = list(NETWORK_LABELS)
    if required_affected is not None:
        affected_schema.update({
            "minItems": len(required_affected),
            "maxItems": len(required_affected),
        })
        network_labels = [
            label for label in NETWORK_LABELS
            if label != "normal_relative_to_reference_sensor"
        ]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "building_vibration_all_sensor",
            "schema": {
                "type": "object",
                "properties": {
                    "sensor_reports": {
                        "type": "array",
                        "minItems": len(target_list),
                        "maxItems": len(target_list),
                        "items": {
                            "type": "object",
                            "properties": {
                                "sensor_id": {"enum": target_list},
                                "local_status": {"enum": list(SENSOR_LABELS)},
                            },
                            "required": ["sensor_id", "local_status"],
                        },
                    },
                    "network_status": {"enum": network_labels},
                    "affected_sensor_ids": affected_schema,
                    "popup_message": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": POPUP_MESSAGE_MAX_CHARS,
                    },
                },
                "required": ["sensor_reports", "network_status",
                             "affected_sensor_ids", "popup_message"],
                "additionalProperties": False,
            },
        },
    }


def validate_popup_message(
    value: Any,
    affected_sensor_ids: Sequence[str],
) -> str:
    """Validate model-authored operator text without rewriting its content."""
    if not isinstance(value, str):
        raise ValueError("popup_message must be a string")
    message = value.strip()
    if not message:
        raise ValueError("popup_message must not be empty")
    if len(message) > POPUP_MESSAGE_MAX_CHARS:
        raise ValueError("popup_message exceeds the character limit")
    if len(message.split()) > POPUP_MESSAGE_MAX_WORDS:
        raise ValueError("popup_message exceeds the word limit")
    implementation = _IMPLEMENTATION_TERM_RE.search(message)
    if implementation:
        raise ValueError(
            "popup_message mentions implementation term "
            f"{implementation.group(0)!r}")
    if _STRUCTURAL_CLAIM_RE.search(message):
        raise ValueError("popup_message makes a structural-safety claim")
    mentioned = {
        f"target_{index}" for index in _TARGET_MENTION_RE.findall(message)
    }
    expected = set(affected_sensor_ids)
    if mentioned != expected:
        raise ValueError(
            f"popup_message sensor mentions {sorted(mentioned)}, "
            f"expected {sorted(expected)}")
    return message


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_codes_reply(
    text: str,
    targets: Sequence[str],
    *,
    require_popup_message: bool = False,
    expected_affected_sensor_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Strictly parse the model's verdict; raise ValueError on any drift.

    Returns {"sensor_reports": {slot: label}, "network_status": label,
    "affected_sensor_ids": [slot, ...], "popup_message": text} with every
    field validated
    against the fixed vocabularies and the requested target set.
    """
    match = _JSON_RE.search(text or "")
    if match is None:
        raise ValueError("reply contains no JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, Mapping):
        raise ValueError("reply JSON is not an object")
    unexpected = set(data) - {"sensor_reports", "network_status",
                              "affected_sensor_ids", "popup_message"}
    if unexpected:
        raise ValueError(f"unexpected reply keys: {sorted(unexpected)}")
    reports = data.get("sensor_reports")
    if not isinstance(reports, list):
        raise ValueError("sensor_reports must be a list")
    wanted = list(targets)
    seen: dict[str, str] = {}
    for entry in reports:
        if not isinstance(entry, Mapping):
            raise ValueError("sensor_reports entries must be objects")
        slot = entry.get("sensor_id")
        label = entry.get("local_status")
        if slot not in wanted:
            raise ValueError(f"unknown sensor_id {slot!r}")
        if slot in seen:
            raise ValueError(f"duplicate sensor_id {slot!r}")
        if label not in SENSOR_LABELS:
            raise ValueError(f"{slot}: unknown local_status {label!r}")
        seen[slot] = label
    if sorted(seen) != sorted(wanted):
        raise ValueError(
            f"sensor_reports cover {sorted(seen)}, expected {sorted(wanted)}")
    network = data.get("network_status")
    if network not in NETWORK_LABELS:
        raise ValueError(f"unknown network_status {network!r}")
    affected = data.get("affected_sensor_ids")
    if not isinstance(affected, list) or \
            any(slot not in wanted for slot in affected):
        raise ValueError("affected_sensor_ids must list known targets only")
    deduped = list(dict.fromkeys(str(s) for s in affected))
    expected_affected = _validated_required_affected_sensor_ids(
        expected_affected_sensor_ids,
        wanted,
    )
    if expected_affected is not None and deduped != expected_affected:
        raise ValueError(
            f"affected_sensor_ids {deduped}, expected identity-guarded "
            f"IDs {expected_affected}"
        )
    if expected_affected is not None:
        locally_affected = [
            slot for slot in wanted
            if seen[slot] in {"mild_deviation", "significant_local_deviation"}
        ]
        if locally_affected != expected_affected:
            raise ValueError(
                f"locally affected IDs {locally_affected}, expected {expected_affected}")
    popup_message = None
    if "popup_message" in data:
        popup_message = validate_popup_message(
            data["popup_message"],
            deduped,
        )
    elif require_popup_message:
        raise ValueError("popup_message is required")
    parsed = {
        "sensor_reports": seen,
        "network_status": network,
        "affected_sensor_ids": deduped,
    }
    if popup_message is not None:
        parsed["popup_message"] = popup_message
    return parsed


def _validated_required_affected_sensor_ids(
    value: Sequence[str] | None,
    targets: Sequence[str],
) -> list[str] | None:
    if value is None:
        return None
    wanted = list(targets)
    required = list(dict.fromkeys(str(slot) for slot in value))
    if not required:
        raise CodesPromptError(
            "required_affected_sensor_ids must be non-empty when supplied"
        )
    unknown = [slot for slot in required if slot not in wanted]
    if unknown:
        raise CodesPromptError(
            f"identity guard contains unknown targets: {unknown}"
        )
    return sorted(required, key=wanted.index)
