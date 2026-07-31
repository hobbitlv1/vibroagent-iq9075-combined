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
assemble prompts and strictly parse/validate the model's JSON verdict.
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


def codes_response_format(targets: Sequence[str]) -> dict:
    """json_schema response_format pinning the reply shape (geniex GBNF)."""
    target_list = list(targets)
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
                    "network_status": {"enum": list(NETWORK_LABELS)},
                    "affected_sensor_ids": {
                        "type": "array",
                        "items": {"enum": target_list},
                    },
                },
                "required": ["sensor_reports", "network_status",
                             "affected_sensor_ids"],
            },
        },
    }


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_codes_reply(text: str,
                      targets: Sequence[str]) -> dict[str, Any]:
    """Strictly parse the model's verdict; raise ValueError on any drift.

    Returns {"sensor_reports": {slot: label}, "network_status": label,
    "affected_sensor_ids": [slot, ...]} with every field validated
    against the fixed vocabularies and the requested target set.
    """
    match = _JSON_RE.search(text or "")
    if match is None:
        raise ValueError("reply contains no JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, Mapping):
        raise ValueError("reply JSON is not an object")
    unexpected = set(data) - {"sensor_reports", "network_status",
                              "affected_sensor_ids"}
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
    return {
        "sensor_reports": seen,
        "network_status": network,
        "affected_sensor_ids": deduped,
    }
