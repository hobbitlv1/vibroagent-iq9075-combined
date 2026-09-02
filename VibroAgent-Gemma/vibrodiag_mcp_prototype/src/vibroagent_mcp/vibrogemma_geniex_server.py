"""OpenAI-compatible VibroGemma endpoint backed by patched GenieX 0.4.0."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

CLASSES = (
    "normal",
    "impact_transient",
    "periodic_excitation",
    "broadband_increase",
    "modal_frequency_shift",
    "damping_change",
    "mounting_or_sensor_fault",
    "unknown_anomaly",
    "data_invalid",
)
SEVERITIES = ("none", "advisory", "warning", "critical")
SHAPE = (84, 1536)
MAX_BODY_BYTES = 2 * 1024 * 1024
_LINE = re.compile(r"^T([1-5])\|([01])\|([a-z_]+)\|([a-z_]+)$")
_GLOBAL_LINE = re.compile(r"^G\|([a-z_]+)\|([a-z_]+)$")
STRUCTURED_DECISION_MODE = "parallel_binary_global_plus_joint_target_set_v1"
DEPLOYMENT_CONTRACT_VERSION = "vibrogemma-geniex-live-v2"
REQUIRED_BUNDLE_ARTIFACTS = (
    "vibrogemma_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "token_layout.json",
    "alert.schema.json",
)


def _installed_geniex_version() -> str | None:
    try:
        return version("geniex")
    except PackageNotFoundError:
        return None


def _choice_strings(
    classes: tuple[str, ...] = CLASSES,
    severities: tuple[str, ...] = SEVERITIES,
) -> list[str]:
    choices = ["0|normal|none"]
    if "data_invalid" in classes:
        choices.append("0|data_invalid|advisory")
    choices.extend(
        f"1|{class_name}|{severity}"
        for class_name in classes
        if class_name not in {"normal", "data_invalid"}
        for severity in severities
        if severity != "none"
    )
    return choices


def classification_grammar(
    classes: tuple[str, ...] = CLASSES,
    severities: tuple[str, ...] = SEVERITIES,
) -> str:
    alternatives = " | ".join(json.dumps(choice) for choice in _choice_strings(classes, severities))
    parts = ['"CLASSIFICATION"']
    for index in range(1, 6):
        parts.extend((json.dumps(f"\nT{index}|"), "choice"))
    parts.append(json.dumps("\nEND"))
    return f"root ::= {' '.join(parts)}\nchoice ::= {alternatives}\n"


def global_classification_grammar(
    classes: tuple[str, ...],
    severities: tuple[str, ...],
) -> str:
    choices = ["normal|none"]
    if "data_invalid" in classes:
        choices.append("data_invalid|advisory")
    choices.extend(
        f"{class_name}|{severity}"
        for class_name in classes
        if class_name not in {"normal", "data_invalid"}
        for severity in severities
        if severity != "none"
    )
    alternatives = " | ".join(json.dumps(choice) for choice in choices)
    return f'root ::= "GLOBAL_CLASSIFICATION\\nG|" choice "\\nEND"\nchoice ::= {alternatives}\n'


def target_classification_grammar(
    index: int,
    classes: tuple[str, ...],
    severities: tuple[str, ...],
) -> str:
    alternatives = " | ".join(json.dumps(choice) for choice in _choice_strings(classes, severities))
    return (
        f'root ::= "TARGET_CLASSIFICATION\\nT{index}|" choice "\\nEND"\n'
        f"choice ::= {alternatives}\n"
    )


def parse_global_record(
    text: str,
    classes: tuple[str, ...],
    severities: tuple[str, ...],
) -> tuple[str, str, str]:
    match = re.search(r"GLOBAL_CLASSIFICATION\s*\nG\|[a-z_]+\|[a-z_]+\s*\nEND", text)
    if match is None:
        raise ValueError("Agent response lacks a complete GLOBAL_CLASSIFICATION record")
    lines = [line.strip() for line in match.group(0).splitlines() if line.strip()]
    parsed = _GLOBAL_LINE.fullmatch(lines[1])
    if parsed is None:
        raise ValueError(f"invalid global line: {lines[1]!r}")
    class_name, severity = parsed.groups()
    if (
        class_name not in classes
        or severity not in severities
        or (class_name == "normal" and severity != "none")
        or (class_name != "normal" and severity == "none")
    ):
        raise ValueError(f"invalid global tuple: {lines[1]!r}")
    return match.group(0), class_name, severity


def parse_target_record(
    text: str,
    index: int,
    classes: tuple[str, ...],
    severities: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    match = re.search(
        rf"TARGET_CLASSIFICATION\s*\nT{index}\|[01]\|[a-z_]+\|[a-z_]+\s*\nEND",
        text,
    )
    if match is None:
        raise ValueError(f"Agent response lacks a complete TARGET_CLASSIFICATION record for T{index}")
    line = [value.strip() for value in match.group(0).splitlines() if value.strip()][1]
    parsed = _LINE.fullmatch(line)
    if parsed is None or int(parsed.group(1)) != index:
        raise ValueError(f"invalid target line: {line!r}")
    affected = parsed.group(2) == "1"
    class_name = parsed.group(3)
    severity = parsed.group(4)
    if (
        class_name not in classes
        or severity not in severities
        or (class_name == "normal" and (affected or severity != "none"))
        or (class_name == "data_invalid" and (affected or severity != "advisory"))
        or (class_name not in {"normal", "data_invalid"} and (not affected or severity == "none"))
    ):
        raise ValueError(f"invalid target tuple: {line!r}")
    return match.group(0), {
        "sensor_id": f"target_{index}",
        "affected": affected,
        "class": class_name,
        "severity": severity,
    }


def parse_record(
    text: str,
    classes: tuple[str, ...] = CLASSES,
    severities: tuple[str, ...] = SEVERITIES,
) -> tuple[str, list[dict[str, Any]]]:
    match = re.search(r"CLASSIFICATION\s*\n(?:T[1-5]\|.*\n){5}END", text)
    if match is None:
        raise ValueError("Agent response lacks a complete CLASSIFICATION record")
    lines = [line.strip() for line in match.group(0).splitlines() if line.strip()]
    targets = []
    for expected, line in enumerate(lines[1:6], 1):
        parsed = _LINE.fullmatch(line)
        if parsed is None or int(parsed.group(1)) != expected:
            raise ValueError(f"invalid target line: {line!r}")
        affected = parsed.group(2) == "1"
        class_name = parsed.group(3)
        severity = parsed.group(4)
        if (
            class_name not in classes
            or severity not in severities
            or (class_name == "normal" and (affected or severity != "none"))
            or (class_name == "data_invalid" and (affected or severity != "advisory"))
            or (class_name not in {"normal", "data_invalid"} and (not affected or severity == "none"))
        ):
            raise ValueError(f"invalid target tuple: {line!r}")
        targets.append(
            {
                "sensor_id": f"target_{expected}",
                "affected": affected,
                "class": class_name,
                "severity": severity,
            }
        )
    state = "normal"
    if any(target["class"] == "data_invalid" for target in targets):
        state = "data_invalid"
    for severity in ("critical", "warning", "advisory"):
        if any(target["severity"] == severity and target["affected"] for target in targets):
            state = severity
            break
    return match.group(0), targets


def _classification_explanation(targets: list[dict[str, Any]]) -> str:
    invalid = [target["sensor_id"] for target in targets if target["class"] == "data_invalid"]
    affected = [target["sensor_id"] for target in targets if target["affected"]]
    if invalid:
        return "Signal quality was insufficient for " + ", ".join(value.replace("_", " ") for value in invalid) + "."
    if len(affected) == 5:
        return "Agent marked all five targets as affected; this record does not isolate a single target."
    if affected:
        findings = ", ".join(
            f"{target['sensor_id'].replace('_', ' ')} ({target['class'].replace('_', ' ')}, {target['severity']})"
            for target in targets
            if target["affected"]
        )
        return f"Agent marked {findings}; the remaining targets match the reference."
    return "All five targets match the reference."


def _marker(index: int) -> str:
    return f"[[VIBROGEMMA_BOARD_{index}_SOFT_TOKENS]]"


def _episode_content(config: dict, quality_text: str, evidence_text: str) -> list[str]:
    markers = config["model"]["marker_text"]
    pieces = [
        "Read the following six-board vibration episode. Continuous vibration embeddings replace each marker.",
        quality_text,
        evidence_text,
        markers["baseline_start"] + _marker(0) + markers["baseline_end"],
    ]
    for index in range(1, 6):
        pieces.append(
            markers["target_start_template"].format(index=index)
            + _marker(index)
            + markers["target_end"]
        )
    return pieces


def _user_content(config: dict, quality_text: str, evidence_text: str) -> str:
    pieces = _episode_content(config, quality_text, evidence_text)
    pieces.append(
        "First generate exactly the constrained CLASSIFICATION record for T1 through T5. "
        "Each tuple must contain affected, class and severity. After END, emit EXPLANATION and one short sentence. "
        "Use the named auditable evidence for the sentence. You may quote a numeric value only when that exact value "
        "is supplied in the evidence."
    )
    return "\n".join(pieces)


def _factorized_user_content(
    config: dict,
    quality_text: str,
    evidence_text: str,
    *,
    target_index: int | None,
) -> str:
    pieces = _episode_content(config, quality_text, evidence_text)
    prior_guard = (
        "Treat normal and unknown_anomaly as equally plausible before reading the evidence. "
        "Do not infer anomaly from excitation magnitude, absolute scale, recording identity, "
        "or dataset-specific appearance alone. "
    )
    if target_index is None:
        instruction = (
            "Perform exactly one independent global classification task. "
            "Decide only whether the six-board episode is normal or unknown_anomaly. "
            + prior_guard
            + "Generate exactly GLOBAL_CLASSIFICATION, one G|class|severity row, and END."
        )
    else:
        instruction = (
            f"Perform exactly one independent target localization task for T{target_index}. "
            f"Decide whether the explicitly damaged physical location belongs to fixed monitored zone T{target_index}. "
            "Propagated vibration measured at another sensor is not itself a positive location label. "
            + prior_guard
            + "Never copy another target row and never convert a global anomaly into an all-target anomaly. "
            "Generate exactly TARGET_CLASSIFICATION, one "
            f"T{target_index}|affected|class|severity row, and END."
        )
    pieces.append(instruction)
    return "\n".join(pieces)


def _structured_decision_block(separator: str = " ") -> str:
    return "\n".join(
        (
            "DIRECT_DECISIONS",
            f"global:{separator}",
            *(f"target_{index}:{separator}" for index in range(1, 6)),
            "END DIRECT_DECISIONS",
        )
    )


def _structured_controls(config: dict) -> tuple[dict[str, Any], dict[str, Any]] | None:
    training = config.get("training")
    if not isinstance(training, dict):
        return None
    structured = training.get("structured_set_training")
    if not isinstance(structured, dict) or not structured.get("enabled"):
        return None
    language = training.get("language_loss")
    return structured, language if isinstance(language, dict) else {}


def _structured_user_content(config: dict, quality_text: str, evidence_text: str) -> str:
    pieces = _episode_content(config, quality_text, evidence_text)
    controls = _structured_controls(config)
    if controls is not None:
        _structured, language = controls
        instruction = (
            "The six binary decisions below are scored in parallel from this same synchronized episode."
            " At every slot, 0 and 1 are the only valid next-token choices. GLOBAL=0 means normal and"
            " GLOBAL=1 means unknown_anomaly. For each T1 through T5, 0 means normal/not affected and"
            " 1 means affected/unknown_anomaly. Compare every target with the reference board and with"
            " all contemporaneous targets. Do not copy the global choice into target slots, do not copy"
            " one target into another, and do not assume that a global anomaly affects every target."
            " The target decisions form one complete set and are evaluated jointly."
        )
        if language.get("target_decision_semantics") == "damage_source_zone":
            instruction += (
                " A target value of 1 means that the explicitly labelled localized source zone belongs"
                " to that monitored target. Propagated vibration at another target is not by itself a"
                " positive localization label."
            )
        if language.get("prompt_prior_guard"):
            instruction += (
                " Treat 0 and 1 as equally plausible before reading the evidence."
                " Do not infer anomaly from excitation magnitude, absolute scale, recording identity,"
                " or dataset-specific appearance alone."
            )
        pieces.append(instruction)
        return "\n".join(pieces)
    pieces.append(
        "Predict the global binary state and the complete affected-target set directly. "
        "At every decision context, 0 means normal and 1 means unknown_anomaly. "
        "Do not copy a global anomaly into every target."
    )
    pieces.append(_structured_decision_block())
    return "\n".join(pieces)


def _find_once(sequence: list[int], needle: list[int]) -> tuple[int, int]:
    starts = [
        start
        for start in range(len(sequence) - len(needle) + 1)
        if sequence[start : start + len(needle)] == needle
    ]
    if len(starts) != 1:
        raise ValueError(f"expected one soft-token marker, found {len(starts)}")
    return starts[0], starts[0] + len(needle)


def mixed_segments(tokenizer, formatted_prompt: str, embeddings) -> list[tuple[str, Any]]:
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False).ids
    token_to_id = getattr(tokenizer, "token_to_id", None)
    bos_id = token_to_id("<bos>") if callable(token_to_id) else None
    if bos_id is not None and (not input_ids or input_ids[0] != bos_id):
        input_ids.insert(0, bos_id)
    spans = []
    for board in range(6):
        marker_ids = tokenizer.encode(_marker(board), add_special_tokens=False).ids
        start, end = _find_once(input_ids, marker_ids)
        spans.append((start, end, board))
    spans.sort()
    segments: list[tuple[str, Any]] = []
    cursor = 0
    for start, end, board in spans:
        if start > cursor:
            segments.append(("tokens", input_ids[cursor:start]))
        segments.append(("embeddings", embeddings[board * 14 : (board + 1) * 14]))
        cursor = end
    if cursor < len(input_ids):
        segments.append(("tokens", input_ids[cursor:]))
    return segments


def _structured_segments(
    tokenizer,
    formatted_prompt: str,
    embeddings,
    config: dict,
    episode_id: str,
):
    controls = _structured_controls(config)
    if controls is not None:
        formatted_prompt += "\nSTRUCTURED_BINARY_DECISIONS"
    segments = mixed_segments(tokenizer, formatted_prompt, embeddings)
    if not segments or segments[-1][0] != "tokens":
        raise ValueError("structured prompt does not have the expected token suffix")
    if controls is not None:
        structured, _language = controls
        separator = str(structured.get("decision_separator", " "))
        if not separator or "\n" in separator or "\r" in separator:
            raise ValueError("structured decision separator must be non-empty and remain on one line")
        choices = tuple(str(value) for value in structured.get("choice_tokens", ("0", "1")))
        if len(choices) != 2 or choices[0] == choices[1]:
            raise ValueError("structured binary choices must contain two distinct values")
        mode = str(structured.get("target_query_order", "fixed"))
        if mode == "fixed":
            target_order = list(range(1, 6))
        elif mode == "deterministic_per_episode":
            if not episode_id:
                raise ValueError("deterministic target query order requires a vibration episode_id")
            target_order = sorted(
                range(1, 6),
                key=lambda index: hashlib.sha256(
                    f"{episode_id}\0target_{index}".encode("utf-8")
                ).digest(),
            )
        else:
            raise ValueError(f"unsupported structured target query order: {mode!r}")

        contexts = [(0, f"\nGLOBAL={separator}")]
        contexts.extend((index, f"\nT{index}={separator}") for index in target_order)
        resolved_choice_ids = []
        for _slot, context in contexts:
            prefix_ids = tokenizer.encode(context, add_special_tokens=False).ids
            context_choice_ids = []
            for choice in choices:
                combined_ids = tokenizer.encode(context + choice, add_special_tokens=False).ids
                if combined_ids[: len(prefix_ids)] != prefix_ids or len(combined_ids) != len(prefix_ids) + 1:
                    raise ValueError("structured binary choice does not append one token in context")
                context_choice_ids.append(combined_ids[-1])
            resolved_choice_ids.append(tuple(context_choice_ids))
        if len(set(resolved_choice_ids)) != 1:
            raise ValueError("structured binary choices resolve differently across decision contexts")
        placeholder_ids = tokenizer.encode("?", add_special_tokens=False).ids
        if len(placeholder_ids) != 1:
            raise ValueError("structured neutral placeholder must be one token")

        absolute = sum(len(values) for _kind, values in segments)
        positions = []
        slot_order = []
        for slot, context in contexts:
            context_ids = tokenizer.encode(context, add_special_tokens=False).ids
            segments.append(("tokens", context_ids))
            absolute += len(context_ids)
            positions.append(absolute - 1)
            slot_order.append(slot)
            segments.append(("tokens", placeholder_ids))
            absolute += 1
        segments.append(
            (
                "tokens",
                tokenizer.encode("\nEND_STRUCTURED_BINARY_DECISIONS", add_special_tokens=False).ids,
            )
        )
        return segments, positions, list(resolved_choice_ids[0]), slot_order

    block_ids = tokenizer.encode(_structured_decision_block(), add_special_tokens=False).ids
    start, _ = _find_once(list(segments[-1][1]), block_ids)
    separator_ids = tokenizer.encode(" ", add_special_tokens=False).ids
    choice_ids = [
        tokenizer.encode(value, add_special_tokens=False).ids for value in ("0", "1")
    ]
    if len(separator_ids) != 1 or any(len(ids) != 1 for ids in choice_ids):
        raise ValueError("structured binary choices are not tokenizer-safe single tokens")
    offsets = [index for index, token_id in enumerate(block_ids) if token_id == separator_ids[0]]
    if len(offsets) != 6 or [right - left for left, right in zip(offsets, offsets[1:])] != [6] * 5:
        raise ValueError("structured decision-context token layout does not match the checkpoint")
    prefix_length = sum(len(values) for _kind, values in segments[:-1]) + start
    positions = [prefix_length + offset for offset in offsets]
    return segments, positions, [ids[0] for ids in choice_ids], list(range(6))


def _canonical_structured_rows(
    rows: list[list[float]], slot_order: list[int]
) -> list[list[float]]:
    if len(rows) != 6 or sorted(slot_order) != list(range(6)):
        raise ValueError("structured rows require one unique global/target slot each")
    by_slot = dict(zip(slot_order, rows, strict=True))
    return [by_slot[slot] for slot in range(6)]


def _binary_log_probabilities(logits: list[float]) -> tuple[list[float], list[float]]:
    if len(logits) != 2 or any(not math.isfinite(value) for value in logits):
        raise ValueError("binary decoder requires two finite logits")
    maximum = max(logits)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    log_probabilities = [value - log_normalizer for value in logits]
    return log_probabilities, [math.exp(value) for value in log_probabilities]


def _structured_choices(
    rows: list[list[float]], cardinality_bias: list[float]
) -> tuple[int, int, list[float], float, list[float]]:
    if len(rows) != 6 or len(cardinality_bias) != 6:
        raise ValueError("structured decoder requires one global row, five target rows and six biases")
    global_logs, global_probabilities = _binary_log_probabilities(rows[0])
    target_rows = [_binary_log_probabilities(row) for row in rows[1:]]
    candidate_scores = []
    for mask in range(32):
        score = float(cardinality_bias[mask.bit_count()])
        for target_index, (log_probabilities, _probabilities) in enumerate(target_rows):
            score += log_probabilities[(mask >> target_index) & 1]
        candidate_scores.append(score)
    target_mask = max(range(32), key=candidate_scores.__getitem__)
    global_choice = int(global_logs[1] > global_logs[0])
    target_probabilities = [
        probabilities[(target_mask >> index) & 1]
        for index, (_logs, probabilities) in enumerate(target_rows)
    ]
    return (
        global_choice,
        target_mask,
        candidate_scores,
        global_probabilities[global_choice],
        target_probabilities,
    )


def _apply_uncalibrated_confidence(
    targets: list[dict[str, Any]],
    raw_probabilities: list[float] | None,
) -> None:
    if raw_probabilities is not None and len(raw_probabilities) != len(targets):
        raise ValueError("target probabilities must align with the five target decisions")
    for index, target in enumerate(targets):
        raw_probability = (
            None
            if raw_probabilities is None or target.get("class") == "data_invalid"
            else float(raw_probabilities[index])
        )
        if raw_probability is not None and (
            not math.isfinite(raw_probability) or not 0.0 <= raw_probability <= 1.0
        ):
            raise ValueError("raw target choice probability must be finite and within [0, 1]")
        target["choice_probability"] = None
        target["raw_choice_probability"] = raw_probability
        target["confidence_operational"] = False


def _shared_factorized_segments(
    segment_groups: list[list[tuple[str, Any]]],
) -> tuple[list[tuple[str, Any]], list[list[tuple[str, Any]]]]:
    if not segment_groups or any(not group or group[-1][0] != "tokens" for group in segment_groups):
        raise ValueError("factorized prompts do not have the expected token suffix")
    shared = segment_groups[0][:-1]
    if any([kind for kind, _ in group[:-1]] != [kind for kind, _ in shared] for group in segment_groups[1:]):
        raise ValueError("factorized prompt segment layouts differ")
    tails = [list(group[-1][1]) for group in segment_groups]
    common_count = 0
    for values in zip(*tails, strict=False):
        if len(set(values)) != 1:
            break
        common_count += 1
    if common_count:
        shared = [*shared, ("tokens", tails[0][:common_count])]
    return shared, [
        [("tokens", tail[common_count:])]
        for tail in tails
    ]


def _factorized_summary(
    global_class: str,
    global_severity: str,
    targets: list[dict[str, Any]],
) -> tuple[str, str, bool, str]:
    affected = [target for target in targets if target["affected"]]
    global_affected = global_class not in {"normal", "data_invalid"}
    consistent = global_affected == bool(affected)
    if global_class == "data_invalid" or any(target["class"] == "data_invalid" for target in targets):
        state = "data_invalid"
    else:
        ranked = {"none": 0, "advisory": 1, "warning": 2, "critical": 3}
        state = max(
            (global_severity, *(target["severity"] for target in affected)),
            key=lambda value: ranked.get(value, -1),
        )
        if state == "none":
            state = "normal"
    if affected:
        localization = "partial_target_set"
        explanation = _classification_explanation(targets)
    elif global_affected:
        localization = "unresolved_global_anomaly"
        explanation = "Agent detected a global anomaly but did not localize it to a target."
    else:
        localization = "normal"
        explanation = _classification_explanation(targets)
    return state, localization, consistent, explanation


def _factorized_kv_cache_enabled() -> bool:
    return os.environ.get("VIBROGEMMA_DISABLE_KV_CACHE") != "1"


def _apply_deterministic_quality_gate(
    targets: list[dict[str, Any]],
    quality: dict[str, Any],
    quality_class: str,
) -> None:
    if not quality_class:
        return
    reference = quality.get("reference")
    reference_invalid = isinstance(reference, dict) and reference.get("valid") is False
    for target in targets:
        target_quality = quality.get(target["sensor_id"])
        target_invalid = isinstance(target_quality, dict) and target_quality.get("valid") is False
        if reference_invalid or target_invalid:
            target.update({"affected": False, "class": quality_class, "severity": "advisory"})


def _decode_embeddings(payload: dict):
    import numpy as np

    if payload.get("schema") != "vibrogemma-live-embeddings-v1" or payload.get("shape") != list(SHAPE):
        raise ValueError("invalid VibroGemma embedding payload contract")
    raw = base64.b64decode(str(payload.get("embeddings_b64") or ""), validate=True)
    if len(raw) != math.prod(SHAPE) * 4:
        raise ValueError("invalid VibroGemma embedding byte length")
    embeddings = np.frombuffer(raw, dtype="<f4").reshape(SHAPE)
    if not np.isfinite(embeddings).all():
        raise ValueError("VibroGemma embeddings contain non-finite values")
    return embeddings


def _embedding_audit(embeddings) -> dict[str, Any]:
    """Reject collapsed/duplicated board embeddings and expose safe diagnostics."""
    import numpy as np

    values = np.asarray(embeddings, dtype="<f4")
    if values.shape != SHAPE:
        raise ValueError(f"invalid VibroGemma embedding shape: {values.shape}")
    board_hashes: dict[str, str] = {}
    board_rms: dict[str, float] = {}
    hash_owner: dict[str, str] = {}
    for index in range(6):
        slot = "baseline" if index == 0 else f"target_{index}"
        board = np.ascontiguousarray(values[index * 14 : (index + 1) * 14])
        rms = float(np.sqrt(np.mean(np.square(board, dtype=np.float64))))
        if not math.isfinite(rms) or rms <= 1e-12:
            raise ValueError(f"{slot}: encoder produced zero or invalid soft tokens")
        digest = hashlib.sha256(board.tobytes(order="C")).hexdigest()
        previous = hash_owner.get(digest)
        if previous is not None:
            raise ValueError(
                f"{slot}: encoder soft tokens are byte-identical to {previous}; "
                "deployment input routing is collapsed"
            )
        hash_owner[digest] = slot
        board_hashes[slot] = digest
        board_rms[slot] = rms
    return {
        "embedding_sha256_by_slot": board_hashes,
        "embedding_rms_by_slot": board_rms,
        "distinct_board_embeddings": True,
    }


def _verify_bundle_identity(bundle: Path, model_path: Path) -> dict[str, Any]:
    manifest_path = bundle / "bundle_manifest.json"
    expected_manifest_sha = os.environ.get("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", "").strip().lower()
    if not manifest_path.is_file():
        if expected_manifest_sha:
            raise RuntimeError(
                "VIBROGEMMA_BUNDLE_MANIFEST_SHA256 is set but bundle_manifest.json is missing"
            )
        return {
            "bundle_version": None,
            "manifest_sha256": None,
            "artifact_sha256": {},
            "model_sha256": None,
        }

    manifest_sha = _sha256(manifest_path)
    if expected_manifest_sha and manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"VibroGemma bundle manifest SHA-256 mismatch: {manifest_sha} != {expected_manifest_sha}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read VibroGemma bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise RuntimeError("VibroGemma bundle manifest has no files list")

    entries: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = item["path"]
        if path in entries:
            raise RuntimeError(f"VibroGemma bundle manifest contains duplicate entry: {path}")
        entries[path] = item
    try:
        gguf_path = model_path.resolve().relative_to(bundle.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("Manifest-backed VibroGemma GGUF must be inside the bundle directory") from exc

    verified: dict[str, str] = {}
    for relative in (*REQUIRED_BUNDLE_ARTIFACTS, gguf_path):
        entry = entries.get(relative)
        if entry is None:
            raise RuntimeError(f"VibroGemma bundle manifest is missing required artifact: {relative}")
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise RuntimeError(f"VibroGemma bundle manifest has invalid SHA-256 for {relative}")
        artifact = bundle / relative
        if not artifact.is_file():
            raise RuntimeError(f"VibroGemma bundle artifact is missing: {relative}")
        expected_size = entry.get("size_bytes")
        if expected_size is not None:
            try:
                expected_size_int = int(expected_size)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"VibroGemma bundle manifest has invalid size for {relative}") from exc
            actual_size = artifact.stat().st_size
            if actual_size != expected_size_int:
                raise RuntimeError(
                    f"VibroGemma bundle artifact size mismatch for {relative}: "
                    f"{actual_size} != {expected_size_int}"
                )
        actual_sha = _sha256(artifact)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"VibroGemma bundle artifact SHA-256 mismatch for {relative}: "
                f"{actual_sha} != {expected_sha}"
            )
        verified[relative] = actual_sha
    return {
        "bundle_version": manifest.get("bundle_version"),
        "manifest_sha256": manifest_sha,
        "artifact_sha256": verified,
        "model_sha256": verified[gguf_path],
    }


class VibroGemmaRuntime:
    def __init__(self, *, model_path: Path, bundle: Path, device_map: str, n_ctx: int, n_batch: int) -> None:
        self.model_path = model_path
        self.bundle = bundle
        self.device_map = device_map
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.model = None
        self.tokenizer = None
        self.runtime_probe: dict[str, Any] = {
            "passed": False,
            "reason": "model_not_loaded",
        }
        self.geniex_version = _installed_geniex_version()
        self.lock = threading.Lock()
        self.inference_started_mono: float | None = None
        self.inference_label = "idle"
        bundle_identity = _verify_bundle_identity(bundle, model_path)
        self.bundle_version = bundle_identity["bundle_version"]
        self.bundle_manifest_sha256 = bundle_identity["manifest_sha256"]
        self.verified_artifact_sha256 = bundle_identity["artifact_sha256"]
        self.config = json.loads((bundle / "vibrogemma_config.json").read_text(encoding="utf-8"))
        self.classes = tuple(self.config["labels"]["classes"])
        self.severities = tuple(self.config["labels"]["severities"])
        self.quality_class = str(self.config["labels"].get("quality_class") or "")
        self.decision_classes = self.classes
        self.factorized_decoding = bool(self.config.get("generation", {}).get("factorized_decoding", False))
        self.structured_set_decoding = bool(
            self.config.get("generation", {}).get("structured_set_decoding", False)
        )
        if self.factorized_decoding and self.structured_set_decoding:
            raise RuntimeError("factorized and structured-set decoding cannot both be enabled")
        if self.structured_set_decoding and (
            self.classes != ("normal", "unknown_anomaly")
            or self.severities != ("none", "advisory")
        ):
            raise RuntimeError("structured-set decoding requires the binary checkpoint label contract")
        if set(self.classes) - set(CLASSES) or "normal" not in self.classes:
            raise RuntimeError(f"unsupported VibroGemma class vocabulary: {self.classes}")
        if self.quality_class and self.quality_class not in CLASSES:
            raise RuntimeError(f"unsupported VibroGemma quality class: {self.quality_class}")
        if set(self.severities) - set(SEVERITIES) or not {"none", "advisory"} <= set(self.severities):
            raise RuntimeError(f"unsupported VibroGemma severity vocabulary: {self.severities}")
        self.system_prompt = str(self.config["model"]["system_prompt"])
        expected = os.environ.get("VIBROGEMMA_GGUF_SHA256")
        self.model_sha256 = bundle_identity["model_sha256"]
        if self.model_sha256 is None and expected:
            self.model_sha256 = _sha256(model_path)
        if expected and self.model_sha256 != expected.lower():
            raise RuntimeError(f"VibroGemma GGUF sha256 mismatch: {self.model_sha256} != {expected}")

    def load(self) -> None:
        from geniex import AutoModelForCausalLM
        from tokenizers import Tokenizer

        if self.geniex_version != "0.4.0":
            raise RuntimeError(
                f"VibroGemma requires GenieX 0.4.0, found {self.geniex_version or 'not installed'}"
            )
        self.tokenizer = Tokenizer.from_file(str(self.bundle / "tokenizer" / "tokenizer.json"))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map=self.device_map,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
        )
        self.runtime_probe = self._probe_loaded_runtime()

    def _probe_loaded_runtime(self) -> dict[str, Any]:
        """Exercise the patched token, external-embedding and logits APIs once."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("cannot probe an unloaded VibroGemma runtime")
        required = {"decode", "reset"}
        required.add("state_logits" if self.structured_set_decoding else "generate_from_state")
        missing = sorted(name for name in required if not callable(getattr(self.model, name, None)))
        if missing:
            raise RuntimeError(
                "patched GenieX runtime is missing required methods: " + ", ".join(missing)
            )

        controls = _structured_controls(self.config)
        if controls is not None:
            structured, _language = controls
            separator = str(structured.get("decision_separator", " "))
            choices = tuple(str(value) for value in structured.get("choice_tokens", ("0", "1")))
        else:
            separator = " "
            choices = ("0", "1")
        context = f"\nGLOBAL={separator}"
        prefix_ids = self.tokenizer.encode(context, add_special_tokens=False).ids
        choice_ids: list[int] = []
        for choice in choices:
            combined = self.tokenizer.encode(context + choice, add_special_tokens=False).ids
            if combined[: len(prefix_ids)] != prefix_ids or len(combined) != len(prefix_ids) + 1:
                raise RuntimeError(
                    "packaged tokenizer does not append a single structured choice token"
                )
            choice_ids.append(int(combined[-1]))
        if len(choice_ids) != 2 or choice_ids[0] == choice_ids[1]:
            raise RuntimeError("structured choice token IDs are invalid")

        probe_ids = self.tokenizer.encode(
            "VibroGemma deployment probe", add_special_tokens=True
        ).ids
        if not probe_ids:
            raise RuntimeError("packaged tokenizer returned no deployment-probe tokens")
        try:
            self.model.reset()
            for offset in range(0, len(probe_ids), self.n_batch):
                self.model.decode(input_ids=probe_ids[offset : offset + self.n_batch])
            self.model.decode(input_embd=[[0.0] * SHAPE[1]])
            selected_logits = None
            if self.structured_set_decoding:
                selected_logits = [float(value) for value in self.model.state_logits(choice_ids)]
                if len(selected_logits) != 2 or any(
                    not math.isfinite(value) for value in selected_logits
                ):
                    raise RuntimeError("patched GenieX state_logits returned invalid values")
        finally:
            self.model.reset()
        return {
            "passed": True,
            "external_embedding_decode": True,
            "state_logits": bool(self.structured_set_decoding),
            "generate_from_state": not self.structured_set_decoding,
            "choice_token_ids": choice_ids,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.model is not None else "loading",
            "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "runtime_probe": dict(self.runtime_probe),
            "model": self.model_path.name,
            "model_sha256": self.model_sha256,
            "checkpoint_sha256": self.model_sha256,
            "bundle_version": self.bundle_version,
            "manifest_sha256": self.bundle_manifest_sha256,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "verified_artifact_sha256": dict(self.verified_artifact_sha256),
            "geniex_version": self.geniex_version,
            "device_map": self.device_map,
            "n_ctx": self.n_ctx,
            "encoder": str(self.bundle / "vibration_encoder_projector.onnx"),
            "external_embeddings": True,
            "factorized_kv_cache_reuse": _factorized_kv_cache_enabled(),
            "decision_mode": (
                STRUCTURED_DECISION_MODE
                if self.structured_set_decoding
                else (
                    "factorized_global_plus_independent_targets_v1"
                    if self.factorized_decoding
                    else "joint_target_record"
                )
            ),
            "label_contract": {
                "classes": list(self.classes),
                "quality_class": self.quality_class or None,
                "severities": list(self.severities),
            },
        }

    def stuck_inference(self, limit_s: float) -> tuple[float, str] | None:
        if self.inference_started_mono is None or limit_s <= 0:
            return None
        age_s = time.monotonic() - self.inference_started_mono
        return (age_s, self.inference_label) if age_s >= limit_s else None

    def _chat(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages or len(raw_messages) > 32:
            raise ValueError("messages must be a non-empty list of at most 32 turns")
        messages = []
        prompt_chars = 0
        for item in raw_messages:
            if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("each message requires a supported role")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("each message requires non-empty text content")
            prompt_chars += len(content)
            messages.append({"role": item["role"], "content": content})
        if prompt_chars > 30_000:
            raise ValueError("messages exceed the 30000-character limit")
        try:
            max_tokens = int(request.get("max_tokens", 256))
            temperature = float(request.get("temperature", 0.2))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_tokens and temperature must be numeric") from exc
        if not 1 <= max_tokens <= 2048 or not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("max_tokens or temperature is outside the supported range")

        prompt = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        started = time.monotonic()
        with self.lock:
            self.inference_started_mono = time.monotonic()
            self.inference_label = "text generation"
            try:
                generated = self.model.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95 if temperature else 1.0,
                    top_k=64 if temperature else 1,
                    stop=["<turn|>"],
                )
            finally:
                self.inference_started_mono = None
                self.inference_label = "idle"
        text = generated.text.strip()
        if not text:
            raise RuntimeError("Agent returned an empty chat completion")
        profile = generated.profile
        now = int(time.time())
        return {
            "id": f"chatcmpl-vibrogemma-{now}",
            "object": "chat.completion",
            "created": now,
            "model": str(request.get("model") or "vibrogemma"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": profile.prompt_tokens,
                "completion_tokens": profile.generated_tokens,
                "total_tokens": profile.prompt_tokens + profile.generated_tokens,
            },
            "genie_metrics": {
                "total_s": round(time.monotonic() - started, 3),
                "generation_s": profile.decode_time / 1_000_000,
                "time_to_first_token_s": profile.ttft / 1_000_000,
                "tokens_per_second": profile.decode_speed,
                "completion_tokens": profile.generated_tokens,
                "persistent": True,
            },
        }

    def _decode_segments(self, segments: list[tuple[str, Any]]) -> None:
        for kind, values in segments:
            if not len(values):
                continue
            if kind == "embeddings":
                self.model.decode(input_embd=values)
            else:
                for offset in range(0, len(values), self.n_batch):
                    self.model.decode(input_ids=values[offset : offset + self.n_batch])

    def _state_logits_at_positions(
        self,
        segments: list[tuple[str, Any]],
        positions: list[int],
        choice_ids: list[int],
    ) -> list[list[float]]:
        if positions != sorted(positions) or len(positions) != 6:
            raise ValueError("structured decision positions must be six ordered offsets")
        self.model.reset()
        rows: list[list[float]] = []
        absolute = 0
        position_index = 0
        for kind, values in segments:
            local = 0
            length = len(values)
            while position_index < len(positions) and positions[position_index] < absolute + length:
                stop = positions[position_index] - absolute + 1
                self._decode_segments([(kind, values[local:stop])])
                rows.append(list(self.model.state_logits(choice_ids)))
                local = stop
                position_index += 1
            if position_index == len(positions):
                break
            self._decode_segments([(kind, values[local:])])
            absolute += length
        if len(rows) != len(positions):
            raise ValueError("structured decision positions fall outside the mixed prompt")
        return rows

    def _structured_record(
        self,
        embeddings,
        quality_text: str,
        evidence_text: str,
        episode_id: str,
    ) -> tuple[
        str,
        str,
        str,
        list[str],
        list[dict[str, Any]],
        list[float],
        float,
        list[float],
        dict[str, Any],
    ]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _structured_user_content(
                    self.config, quality_text, evidence_text
                ),
            },
        ]
        formatted = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        segments, positions, choice_ids, slot_order = _structured_segments(
            self.tokenizer, formatted, embeddings, self.config, episode_id
        )
        rows = _canonical_structured_rows(
            self._state_logits_at_positions(segments, positions, choice_ids),
            slot_order,
        )
        margins = [float(row[1] - row[0]) for row in rows]
        if max(margins) - min(margins) <= 1e-7:
            raise RuntimeError(
                "structured decision logits are identical across all six slots; "
                "the deployed stateful decoder is not resolving slot context"
            )
        structured_audit = {
            "gemma_binary_choice_slot_order": [
                "GLOBAL",
                *(f"T{index}" for index in range(1, 6)),
            ],
            "gemma_binary_choice_token_ids": [int(value) for value in choice_ids],
            "gemma_binary_choice_logits": [
                [float(value) for value in row] for row in rows
            ],
            "gemma_binary_choice_logit_margins": margins,
            "gemma_binary_query_slot_ids": [int(value) for value in slot_order],
            "gemma_binary_query_slot_order": [
                "GLOBAL" if slot == 0 else f"T{slot}" for slot in slot_order
            ],
            "gemma_binary_query_positions": [int(value) for value in positions],
        }
        biases = [
            float(value)
            for value in self.config["generation"].get(
                "target_set_cardinality_bias", [0.0] * 6
            )
        ]
        structured_audit["target_set_cardinality_bias"] = biases
        (
            global_choice,
            target_mask,
            candidate_scores,
            global_probability,
            target_probabilities,
        ) = _structured_choices(rows, biases)
        global_class = "unknown_anomaly" if global_choice else "normal"
        global_severity = "advisory" if global_choice else "none"
        targets = []
        raw_targets = []
        for index in range(1, 6):
            affected = bool(target_mask & (1 << (index - 1)))
            class_name = "unknown_anomaly" if affected else "normal"
            severity = "advisory" if affected else "none"
            targets.append(
                {
                    "sensor_id": f"target_{index}",
                    "affected": affected,
                    "class": class_name,
                    "severity": severity,
                }
            )
            raw_targets.append(
                f"TARGET_CLASSIFICATION\nT{index}|{int(affected)}|{class_name}|{severity}\nEND"
            )
        raw_global = (
            f"GLOBAL_CLASSIFICATION\nG|{global_class}|{global_severity}\nEND"
        )
        return (
            raw_global,
            global_class,
            global_severity,
            raw_targets,
            targets,
            candidate_scores,
            global_probability,
            target_probabilities,
            structured_audit,
        )

    def _factorized_records(
        self,
        embeddings,
        quality_text: str,
        evidence_text: str,
    ) -> tuple[str, str, str, list[str], list[dict[str, Any]]]:
        probes = [(None, global_classification_grammar(self.decision_classes, self.severities))]
        probes.extend(
            (index, target_classification_grammar(index, self.decision_classes, self.severities))
            for index in range(1, 6)
        )
        segment_groups = []
        for target_index, _grammar in probes:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": _factorized_user_content(
                        self.config,
                        quality_text,
                        evidence_text,
                        target_index=target_index,
                    ),
                },
            ]
            formatted = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            segment_groups.append(mixed_segments(self.tokenizer, formatted, embeddings))

        shared, suffixes = _shared_factorized_segments(segment_groups)
        cache_path: Path | None = None
        self.model.reset()
        self._decode_segments(shared)
        if (
            _factorized_kv_cache_enabled()
            and hasattr(self.model, "save_kv_cache")
            and hasattr(self.model, "load_kv_cache")
        ):
            descriptor, cache_name = tempfile.mkstemp(prefix="vibroagent-factorized-", suffix=".kv")
            os.close(descriptor)
            cache_path = Path(cache_name)
            cache_path.unlink()
            try:
                self.model.save_kv_cache(str(cache_path))
            except Exception:
                cache_path.unlink(missing_ok=True)
                cache_path = None

        generated_texts: list[str] = []
        try:
            for probe_index, ((_target_index, grammar), suffix) in enumerate(zip(probes, suffixes, strict=True)):
                if cache_path is not None:
                    try:
                        self.model.load_kv_cache(str(cache_path))
                        self._decode_segments(suffix)
                    except Exception:
                        cache_path.unlink(missing_ok=True)
                        cache_path = None
                if cache_path is None:
                    self.model.reset()
                    self._decode_segments(segment_groups[probe_index])
                generated = self.model.generate_from_state(
                    max_new_tokens=32,
                    temperature=0.0,
                    top_k=1,
                    grammar=grammar,
                )
                generated_texts.append(generated.text)
        finally:
            if cache_path is not None:
                cache_path.unlink(missing_ok=True)

        raw_global, global_class, global_severity = parse_global_record(
            generated_texts[0], self.decision_classes, self.severities
        )
        raw_targets: list[str] = []
        targets: list[dict[str, Any]] = []
        for index, text in enumerate(generated_texts[1:], 1):
            raw_target, target = parse_target_record(
                text, index, self.decision_classes, self.severities
            )
            raw_targets.append(raw_target)
            targets.append(target)
        return raw_global, global_class, global_severity, raw_targets, targets

    def classify(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        payload = request.get("vibration")
        if payload is None:
            return self._chat(request)
        if not isinstance(payload, dict):
            raise ValueError("request requires a vibration payload")
        embeddings = _decode_embeddings(payload)
        embedding_audit = _embedding_audit(embeddings)
        quality_text = str(payload.get("quality_text") or "").strip()
        evidence_text = str(payload.get("evidence_text") or "").strip()
        if not quality_text or not evidence_text:
            raise ValueError("vibration payload requires quality_text and evidence_text")
        if len(quality_text) > 2_000 or len(evidence_text) > 12_000:
            raise ValueError("vibration quality or evidence text is too long")
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {"summary": quality_text}
        window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        episode_id = str(window.get("episode_id") or metadata.get("episode_id") or "").strip()
        started = time.monotonic()
        decision_metadata: dict[str, Any]
        raw_global_probability: float | None = None
        raw_target_probabilities: list[float] | None = None
        raw_explanation = ""
        with self.lock:
            self.inference_started_mono = time.monotonic()
            self.inference_label = "vibration classification"
            try:
                if self.structured_set_decoding:
                    (
                        raw_global,
                        global_class,
                        global_severity,
                        raw_targets,
                        targets,
                        candidate_scores,
                        raw_global_probability,
                        target_probabilities,
                        structured_audit,
                    ) = self._structured_record(
                        embeddings, quality_text, evidence_text, episode_id
                    )
                    raw_target_probabilities = target_probabilities
                    _apply_deterministic_quality_gate(targets, quality, self.quality_class)
                    record = "CLASSIFICATION\n" + "\n".join(
                        f"T{index}|{int(target['affected'])}|{target['class']}|{target['severity']}"
                        for index, target in enumerate(targets, 1)
                    ) + "\nEND"
                    system_state, localization, consistent, explanation = _factorized_summary(
                        global_class, global_severity, targets
                    )
                    decision_metadata = {
                        "decision_mode": STRUCTURED_DECISION_MODE,
                        "raw_global_record": raw_global,
                        "raw_target_records": raw_targets,
                        "global_class": global_class,
                        "global_severity": global_severity,
                        "global_target_consistent": consistent,
                        "localization_status": localization,
                        "raw_global_choice_probability": raw_global_probability,
                        "raw_target_choice_probabilities": target_probabilities,
                        "structured_target_set_candidate_scores": candidate_scores,
                        **embedding_audit,
                        **structured_audit,
                    }
                elif self.factorized_decoding:
                    raw_global, global_class, global_severity, raw_targets, targets = self._factorized_records(
                        embeddings, quality_text, evidence_text
                    )
                    _apply_deterministic_quality_gate(targets, quality, self.quality_class)
                    record = "CLASSIFICATION\n" + "\n".join(
                        f"T{index}|{int(target['affected'])}|{target['class']}|{target['severity']}"
                        for index, target in enumerate(targets, 1)
                    ) + "\nEND"
                    system_state, localization, consistent, explanation = _factorized_summary(
                        global_class, global_severity, targets
                    )
                    decision_metadata = {
                        "decision_mode": "factorized_global_plus_independent_targets_v1",
                        "raw_global_record": raw_global,
                        "raw_target_records": raw_targets,
                        "global_class": global_class,
                        "global_severity": global_severity,
                        "global_target_consistent": consistent,
                        "localization_status": localization,
                        **embedding_audit,
                    }
                else:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": _user_content(self.config, quality_text, evidence_text)},
                    ]
                    formatted = self.model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    segments = mixed_segments(self.tokenizer, formatted, embeddings)
                    self.model.reset()
                    self._decode_segments(segments)
                    generated = self.model.generate_from_state(
                        max_new_tokens=160,
                        temperature=0.0,
                        top_k=1,
                        grammar=classification_grammar(self.decision_classes, self.severities),
                    )
                    record, targets = parse_record(
                        generated.text, self.decision_classes, self.severities
                    )
                    raw_explanation = generated.text.partition(record)[2].strip()
                    _apply_deterministic_quality_gate(targets, quality, self.quality_class)
                    explanation = _classification_explanation(targets)
                    system_state = "normal"
                    if any(target["class"] == "data_invalid" for target in targets):
                        system_state = "data_invalid"
                    for severity in ("critical", "warning", "advisory"):
                        if any(target["affected"] and target["severity"] == severity for target in targets):
                            system_state = severity
                            break
                    decision_metadata = {
                        "decision_mode": "joint_target_record",
                        **embedding_audit,
                    }
            finally:
                self.inference_started_mono = None
                self.inference_label = "idle"
        _apply_uncalibrated_confidence(targets, raw_target_probabilities)
        invalid_explanation = any(target["class"] == "data_invalid" for target in targets)
        decision_metadata.update(
            {
                "raw_explanation": raw_explanation,
                "explanation_source": (
                    "deterministic_quality_evidence"
                    if invalid_explanation
                    else "deterministic_physics_evidence"
                ),
                "explanation_grounded": True,
                "raw_global_choice_probability": raw_global_probability,
                "global_choice_probability": None,
                "global_confidence_operational": False,
                "confidence_calibration": {
                    "path": self.config.get("generation", {}).get(
                        "confidence_calibration_path"
                    ),
                    "target_operational": False,
                    "target_validation_auroc": None,
                    "global_operational": False,
                    "global_validation_auroc": None,
                },
            }
        )
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {"summary": evidence_text}
        raw_effective_duration = window.get("effective_duration_seconds")
        if raw_effective_duration is None:
            raw_effective_duration = window.get("duration_seconds")
        if raw_effective_duration is None:
            raw_effective_duration = 10.0
        try:
            effective_duration = float(raw_effective_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("effective vibration window duration must be numeric") from exc
        if not math.isfinite(effective_duration) or effective_duration <= 0:
            raise ValueError("effective vibration window duration must be positive and finite")
        window = {
            "start_utc": str(
                window.get("start_utc")
                or metadata.get("window_start_utc")
                or (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
            ),
            "duration_seconds": 10.0,
            "effective_duration_seconds": effective_duration,
            "episode_id": episode_id,
            **{
                key: value
                for key, value in window.items()
                if key
                not in {
                    "start_utc",
                    "duration_seconds",
                    "effective_duration_seconds",
                    "episode_id",
                }
            },
        }
        content = json.dumps(
            {
                "schema_version": "1.0",
                "window": window,
                "system_state": system_state,
                "targets": targets,
                "explanation": explanation,
                "quality": quality,
                "model": {
                    "checkpoint": self.model_path.name,
                    "checkpoint_sha256": self.model_sha256,
                    "bundle_version": self.bundle_version,
                    "bundle_manifest_sha256": self.bundle_manifest_sha256,
                    "base_model": str(self.config["model"]["gemma_model_id"]),
                    "raw_record": record,
                    "label_contract": {
                        "classes": list(self.classes),
                        "quality_class": self.quality_class or None,
                        "severities": list(self.severities),
                    },
                    **decision_metadata,
                },
                "evidence": evidence,
            },
            separators=(",", ":"),
        )
        now = int(time.time())
        return {
            "id": f"chatcmpl-vibrogemma-{now}",
            "object": "chat.completion",
            "created": now,
            "model": str(request.get("model") or "vibrogemma"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "vibrogemma_metrics": {"latency_s": round(time.monotonic() - started, 3)},
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        runtime: VibroGemmaRuntime = self.server.runtime  # type: ignore[attr-defined]
        if self.path.rstrip("/") == "/health":
            self._json(200, runtime.health())
        elif self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "vibrogemma", "object": "model"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            runtime: VibroGemmaRuntime = self.server.runtime  # type: ignore[attr-defined]
            self._json(200, runtime.classify(request))
        except ValueError as exc:
            self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except Exception as exc:
            self._json(500, {"error": {"message": str(exc), "type": "vibrogemma_error"}})

    def log_message(self, format: str, *args) -> None:
        if os.environ.get("VIBROGEMMA_QUIET") != "1":
            super().log_message(format, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GENIEX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GENIEX_PORT", "18181")))
    parser.add_argument("--model", type=Path, default=Path(os.environ["VIBROGEMMA_GGUF"]))
    parser.add_argument("--bundle", type=Path, default=Path(os.environ["VIBROGEMMA_BUNDLE"]))
    parser.add_argument("--device-map", default=os.environ.get("GENIEX_DEVICE_MAP", "llama_cpp:HTP0"))
    parser.add_argument("--n-ctx", type=int, default=int(os.environ.get("GENIEX_N_CTX", "4096")))
    parser.add_argument("--n-batch", type=int, default=int(os.environ.get("GENIEX_N_BATCH", "512")))
    args = parser.parse_args()
    runtime = VibroGemmaRuntime(
        model_path=args.model.resolve(),
        bundle=args.bundle.resolve(),
        device_map=args.device_map,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
    )
    runtime.load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    watchdog_s = float(os.environ.get("VIBROGEMMA_GENERATION_WATCHDOG_S", "30"))
    if watchdog_s > 0:
        def watchdog() -> None:
            while True:
                time.sleep(2.0)
                stuck = runtime.stuck_inference(watchdog_s)
                if stuck is None:
                    continue
                age_s, label = stuck
                print(f"VIBROGEMMA WATCHDOG: {label} stuck for {age_s:.0f}s; restarting", flush=True)
                os.execv(
                    sys.executable,
                    [sys.executable, "-u", "-m", "vibroagent_mcp.vibrogemma_geniex_server", *sys.argv[1:]],
                )

        threading.Thread(target=watchdog, name="vibrogemma-watchdog", daemon=True).start()
    print(f"Serving VibroAgent-Gemma/GenieX 0.4.0 at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
