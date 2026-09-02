"""Run the existing webchat with VibroGemma as its live monitor decision."""

import copy
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import webchat_server
from .features import STRUCTURAL_FREQUENCY_MAX_HZ
from .vibrogemma_live import (
    SLOTS,
    _lumo_overlay_gain,
    _lumo_overlay_templates,
    _lumo_training_replay,
    apply_vibrogemma_monitor_decision,
)


_POPUP_LOCK = threading.Lock()
_LATEST_MONITOR: dict | None = None
_LATEST_MONITOR_AT = 0.0
_DEPLOYMENT_STATUS: dict[str, str] | None = None
WEBCHAT_DEPLOYMENT_CONTRACT_VERSION = "vibroagent-gemma-web-live-v2"
_STATIC_ROOT = Path(__file__).resolve().parent / "static" / "vibrogemma"
_MONITOR_QUERY: dict[str, list[str]] = {
    "axis": ["norm"],
    "duration_s": ["10"],
    "require_current": ["1"],
    "require_llm": ["0"],
    "use_llm": ["1"],
    "model_timeout_s": ["60"],
    "vote_k": ["1"],
    "skip_cache": ["1"],
    "process_reader": ["1"],
}


def _agent_copy(value: Any) -> str:
    return str(value or "").replace("VibroGemma", "Agent").replace("Gemma", "Agent")


def _cpu_set(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        bounds = part.strip().split("-", 1)
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError:
            continue
        cpus.update(range(min(start, stop), max(start, stop) + 1))
    return {cpu for cpu in cpus if cpu >= 0}


def _pin_current_thread(env_name: str, default: str) -> None:
    count = os.cpu_count() or 1
    cpus = {cpu for cpu in _cpu_set(os.environ.get(env_name, default)) if cpu < count}
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


def _monitor_interval_s() -> float:
    try:
        return max(1.0, float(os.environ.get("VIBROGEMMA_MONITOR_INTERVAL_S", "10")))
    except ValueError:
        return 10.0


def _monitor_stale_s() -> float:
    try:
        return max(5.0, float(os.environ.get("VIBROGEMMA_MONITOR_STALE_S", "30")))
    except ValueError:
        return 30.0


def _capture_age_s(payload: dict[str, Any]) -> float:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    window = result.get("window") if isinstance(result.get("window"), dict) else {}
    value = window.get("end_utc") or window.get("window_end_utc")
    try:
        captured = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - captured.astimezone(timezone.utc)).total_seconds())


def _deployment_status() -> dict[str, str]:
    global _DEPLOYMENT_STATUS
    if _DEPLOYMENT_STATUS is not None:
        return dict(_DEPLOYMENT_STATUS)
    bundle = Path(os.environ.get("VIBROGEMMA_BUNDLE") or "")
    try:
        manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    _DEPLOYMENT_STATUS = {
        key: str(manifest[key])
        for key in ("promotion_status", "permitted_use")
        if manifest.get(key)
    }
    return dict(_DEPLOYMENT_STATUS)


def _monitor_view(monitor: dict, targets: list[dict]) -> dict[str, Any]:
    affected = [str(target.get("sensor_id")) for target in targets if target.get("affected")]
    global_class = str(monitor.get("global_class") or "").strip()
    global_known = bool(global_class)
    global_anomaly = global_class not in {"", "normal", "data_invalid"}
    target_anomaly = bool(affected)
    quality = global_class == "data_invalid" or any(target.get("class") == "data_invalid" for target in targets)
    consistent = monitor.get("global_target_consistent")
    reported_localization = str(monitor.get("localization_status") or "").strip()
    localization_verified = monitor.get("localization_verified")
    geometry_unavailable = localization_verified is not True or reported_localization == "geometry_unavailable"
    if not isinstance(consistent, bool) and global_known:
        consistent = global_anomaly == target_anomaly

    if quality:
        state = "data_invalid"
    elif global_anomaly:
        state = "anomaly"
    elif consistent is False or (len(affected) == 5 and not global_known):
        state = "inconclusive"
    elif target_anomaly:
        state = "anomaly"
    else:
        state = "normal"

    if global_anomaly and not affected:
        localization = "unresolved_global_anomaly"
    elif state == "inconclusive":
        localization = "inconclusive"
    elif geometry_unavailable and affected:
        localization = "geometry_unavailable"
    elif len(affected) == 5:
        localization = "network_wide"
    elif affected:
        localization = "target"
    else:
        localization = "normal"
    return {
        "state": state,
        "affected": affected,
        "quality": quality,
        "global_class": global_class or None,
        "global_anomaly": global_anomaly,
        "global_target_consistent": consistent,
        "localization": localization,
        "localization_verified": False if geometry_unavailable else localization_verified,
    }


def _honest_explanation(result: dict, view: dict[str, Any]) -> str:
    affected = view["affected"]
    if view["state"] == "inconclusive":
        return "The global and target decisions disagree, so this check is inconclusive."
    if view["localization"] == "unresolved_global_anomaly":
        return "The Agent detected a global anomaly but did not localize it to a target."
    if view["global_anomaly"] and len(affected) == 5:
        return (
            "The Agent detected an anomaly and marked all five targets as model candidates; physical localization is unavailable."
            if view["localization"] == "geometry_unavailable"
            else "The Agent detected a network-wide anomaly and marked all five targets; localization is not isolated."
        )
    if view["localization"] == "geometry_unavailable" and affected:
        return (
            "The Agent marked "
            + ", ".join(value.replace("_", " ") for value in affected)
            + " as model candidates, but physical localization is unavailable because sensor geometry is not configured."
        )
    return _agent_copy(result.get("main_agent_explanation"))


def _apply_popup_policy(popup: dict, result: dict) -> dict:
    metadata = result.get("model_metadata") or {}
    monitor = metadata.get("vibrogemma_monitor") or {}
    targets = monitor.get("targets")
    if metadata.get("agent_mode") == "vibrogemma_live" and not isinstance(targets, list):
        return {
            "show_popup": False,
            "severity": "quality",
            "title": "Agent check unavailable",
            "message": "The latest Agent check did not complete. Live waveforms remain available.",
            "network_label": "gemma_unavailable",
            "affected_sensor_ids": [],
            "quality_flags": [],
        }
    if metadata.get("agent_mode") != "vibrogemma_live":
        return popup
    synthetic_test = (monitor.get("capture") or {}).get("synthetic_injection")
    view = _monitor_view(monitor, targets)
    affected = view["affected"]
    quality = view["quality"]
    inconclusive = view["state"] == "inconclusive"
    active = bool(quality or view["global_anomaly"] or inconclusive or affected)
    output = dict(popup)
    output.update(
        {
            "show_popup": active,
            "save_replay": bool(quality or affected or view["global_anomaly"] or inconclusive),
            "affected_sensor_ids": affected,
            "significant_sensor_ids": [
                str(target.get("sensor_id"))
                for target in targets
                if target.get("affected") and target.get("severity") in {"warning", "critical"}
            ],
            "mild_sensor_ids": [
                str(target.get("sensor_id"))
                for target in targets
                if target.get("affected") and target.get("severity") == "advisory"
            ],
            "network_label": (
                "data_invalid" if quality else ("inconclusive" if inconclusive else ("unknown_anomaly" if active else "normal"))
            ),
            "localization": view["localization"],
            "global_class": view["global_class"],
            "global_target_consistent": view["global_target_consistent"],
            "localization_verified": view["localization_verified"],
            "confidence": None,
            "label_uncertain": inconclusive,
            "vote_agreement": None,
            "persistence": {"active": active, "trigger": "current_non_normal"},
        }
    )
    if quality:
        if monitor.get("deployment_error"):
            output.update(
                severity="quality",
                title="Agent check unavailable",
                message="The Agent deployment did not complete. No normal verdict was emitted.",
            )
        else:
            output.update(
                severity="quality",
                title="Agent data quality",
                message="The current window is not reliable enough to classify.",
            )
    elif inconclusive:
        output.update(
            severity="quality",
            title="Agent check inconclusive",
            message="Conflicting global and target decisions; review required. This is not a localized anomaly.",
        )
    elif view["localization"] == "unresolved_global_anomaly":
        output.update(
            severity="notice",
            title="Agent network advisory",
            message="Agent detected an anomaly but could not localize it to a target.",
        )
    elif view["global_anomaly"] and len(affected) == 5:
        output.update(
            severity="notice",
            title="Agent network advisory",
            message=(
                "Agent detected an anomaly and marked all five targets as model candidates; physical localization is unverified."
                if view["localization"] == "geometry_unavailable"
                else "Agent detected a network-wide anomaly affecting all five targets."
            ),
        )
    elif affected:
        output.update(
            severity="notice",
            title="Agent advisory",
            message=(
                f"Agent marked {len(affected)} target candidate{'s' if len(affected) != 1 else ''}; physical localization is unverified."
                if view["localization"] == "geometry_unavailable"
                else f"Agent marked {len(affected)} target{'s' if len(affected) != 1 else ''} for review."
            ),
        )
    if isinstance(synthetic_test, dict) and synthetic_test.get("active"):
        output["synthetic_test"] = synthetic_test
        output["save_replay"] = False
    output["dedupe_key"] = "|".join([str(output["network_label"]), ",".join(affected)])
    assessment = dict(result.get("network_assessment") or {})
    assessment.update(
        {
            "network_label": output["network_label"],
            "main_agent_label": output["network_label"],
            "affected_sensor_ids": affected,
            "significant_sensor_ids": output["significant_sensor_ids"],
            "mild_sensor_ids": output["mild_sensor_ids"],
            "main_agent_confidence": None,
            "confidence_source": "raw_choices_not_operational",
            "label_uncertain": inconclusive,
            "localization_status": view["localization"],
            "localization_verified": view["localization_verified"],
            "explanation": _honest_explanation(result, view),
            "explanation_source": "deterministic_decision_projection",
        }
    )
    result["network_assessment"] = assessment
    result["main_agent_explanation"] = assessment["explanation"]
    result["main_agent_explanation_source"] = assessment["explanation_source"]
    return output


def _gemma_context() -> dict[str, Any]:
    with _POPUP_LOCK:
        latest = copy.deepcopy(_LATEST_MONITOR)
        age_s = time.monotonic() - _LATEST_MONITOR_AT
    stale = latest is not None and age_s > _monitor_stale_s()
    result = (latest or {}).get("result") or {}
    metadata = result.get("model_metadata") or {}
    monitor = metadata.get("vibrogemma_monitor") or {}
    targets = monitor.get("targets")
    if stale or not metadata.get("monitor_llm_model_used") or not isinstance(targets, list):
        return {
            "source": "latest_gemma_check",
            "available": False,
            "stale": stale,
            "age_s": round(age_s, 1),
            "deployment": _deployment_status(),
        }
    view = _monitor_view(monitor, targets)
    return {
        "source": "latest_gemma_check",
        "available": True,
        "age_s": round(age_s, 1),
        "network_state": view["state"],
        "global_class": view["global_class"],
        "global_target_consistent": view["global_target_consistent"],
        "localization_status": view["localization"],
        "localization_verified": view["localization_verified"],
        "targets": [
            {
                "sensor_id": target.get("sensor_id"),
                "affected": bool(target.get("affected")),
                "class": target.get("class"),
                "severity": target.get("severity"),
            }
            for target in targets
        ],
        "explanation": _honest_explanation(result, view),
        "encoder_metadata": monitor.get("encoder_metadata") or {},
        "soft_tokens_valid": monitor.get("soft_tokens_valid"),
        "window_duration_s": 10.0,
        "deployment": _deployment_status(),
    }


def _replay_state(detail: dict[str, Any]) -> str:
    targets = _replay_targets(detail)
    if not targets:
        return "unavailable"
    monitor = (((detail.get("pipeline_result") or {}).get("model_metadata") or {}).get("vibrogemma_monitor") or {})
    return str(_monitor_view(monitor, targets)["state"])


def _replay_targets(detail: dict[str, Any]) -> list[dict[str, Any]]:
    result = detail.get("pipeline_result") or {}
    metadata = result.get("model_metadata") or {}
    targets = (metadata.get("vibrogemma_monitor") or {}).get("targets")
    return targets if isinstance(targets, list) else []


def _load_replay_detail(window_id: str) -> dict[str, Any] | None:
    safe_id = webchat_server._safe_anomaly_window_id(window_id)
    path = webchat_server._resolve_anomaly_window_dir() / "windows" / f"{safe_id}.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return detail if isinstance(detail, dict) else None


def _gemma_replays_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    limit = min(200, max(1, webchat_server._int_query(query, "limit", 50)))
    records = webchat_server._anomaly_windows_payload({"limit": [str(limit * 4)]}).get("windows") or []
    items = []
    for record in records:
        detail = _load_replay_detail(str(record.get("window_id") or ""))
        if detail is None:
            continue
        state = _replay_state(detail)
        if state == "unavailable":
            continue
        targets = _replay_targets(detail)
        monitor = (((detail.get("pipeline_result") or {}).get("model_metadata") or {}).get("vibrogemma_monitor") or {})
        view = _monitor_view(monitor, targets)
        items.append(
            {
                "window_id": detail.get("window_id"),
                "created_at_utc": detail.get("created_at_utc"),
                "state": state,
                "affected_sensor_ids": view["affected"],
                "global_class": view["global_class"],
                "global_target_consistent": view["global_target_consistent"],
                "localization_status": view["localization"],
                "localization_verified": view["localization_verified"],
            }
        )
        if len(items) >= limit:
            break
    return {"ok": True, "status": "ok", "windows": items}


def _gemma_replay_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    window_id = str(webchat_server._first(query, "window_id") or "").strip()
    if not window_id:
        return {"ok": False, "error": "window_id is required"}
    detail = _load_replay_detail(window_id)
    if detail is None or _replay_state(detail) == "unavailable":
        return {"ok": False, "error": "Agent replay was not found"}
    result = detail.get("pipeline_result") or {}
    metadata = result.get("model_metadata") or {}
    monitor = metadata.get("vibrogemma_monitor") or {}
    targets = _replay_targets(detail)
    view = _monitor_view(monitor, targets)
    target_by_id = {str(target.get("sensor_id")): target for target in targets}
    metrics = result.get("all_sensor_metrics") or []
    boards = []
    for metric in metrics:
        sensor_id = str(metric.get("sensor_id") or "")
        target = target_by_id.get(sensor_id)
        dominant_frequency_hz = webchat_server._metric_float(metric, "dominant_frequency_hz")
        legacy_wideband_peak = bool(
            dominant_frequency_hz and dominant_frequency_hz > STRUCTURAL_FREQUENCY_MAX_HZ
        )
        state = "reference" if sensor_id == result.get("baseline_sensor_id") else "unavailable"
        if target:
            state = "data_invalid" if target.get("class") == "data_invalid" else (
                "anomaly" if target.get("affected") else "normal"
            )
        boards.append(
            {
                "sensor_id": sensor_id,
                "state": state,
                "class": target.get("class") if target else None,
                "severity": target.get("severity") if target else None,
                "rms_g": metric.get("acceleration_rms_g"),
                "peak_g": metric.get("acceleration_peak_g"),
                "dominant_frequency_hz": None if legacy_wideband_peak else dominant_frequency_hz,
                "wideband_peak_frequency_hz": dominant_frequency_hz if legacy_wideband_peak else None,
                "dominant_frequency_status": "legacy_wideband_only" if legacy_wideband_peak else "structural_band",
            }
        )
    return {
        "ok": True,
        "status": "ok",
        "window_id": detail.get("window_id"),
        "created_at_utc": detail.get("created_at_utc"),
        "state": _replay_state(detail),
        "affected_sensor_ids": view["affected"],
        "global_class": view["global_class"],
        "global_target_consistent": view["global_target_consistent"],
        "localization_status": view["localization"],
        "localization_verified": view["localization_verified"],
        "boards": boards,
        "explanation": _honest_explanation(result, view) or "Agent completed this saved check.",
        "soft_tokens_valid": monitor.get("soft_tokens_valid"),
        "window_duration_s": 10.0,
        "deployment": _deployment_status(),
    }


_CHAT_SYSTEM_PROMPT = (
    "You are VibroAgent, the conversational assistant of a building-vibration monitor with one reference board "
    "and five target boards. Each turn you receive an evidence sheet describing the current context (a live Agent "
    "check, a spectrum, or a saved check) followed by the user's question.\n"
    "What the sheet contains: the Agent's verdict and per-board labels; a 'Measured evidence' section with each "
    "board's waveform statistics over the window (RMS level, peak, crest factor, kurtosis, dominant frequency, "
    "share of energy by band), its strongest structural mode, and its transmissibility relative to the reference "
    "board; and a 'Label vocabulary' line explaining what the labels can and cannot express.\n"
    "How to answer:\n"
    "- Answer the specific question first, in plain conversational language, the way a knowledgeable colleague "
    "would in a chat.\n"
    "- When asked to describe an anomaly, a board, a signal, or a waveform, use the measured evidence and the "
    "comparison with the reference board, quoting the relevant numbers with units. When asked what a label "
    "means, use the label vocabulary.\n"
    "- Use only the evidence sheet and the conversation so far. If the sheet does not cover the question, say so "
    "in one sentence and say what evidence would be needed.\n"
    "- Never invent measurements, thresholds, faults, causes, damage, or trends that are not in the sheet.\n"
    "- Do not recite the whole sheet, do not mention fields the user did not ask about, and do not repeat an "
    "earlier answer word for word.\n"
    "- Do not add disclaimers. Only if the user asks whether the building is safe, damaged, or needs inspection, "
    "note that this is monitoring triage rather than a certified structural-safety assessment.\n"
    "- Keep it to one to four short sentences, or a short bullet list when a list or comparison is asked for."
)
_LOCALIZATION_TEXT = {
    "normal": "no target deviates from the reference",
    "target": "the anomaly is localized to the affected targets listed below",
    "network_wide": "all five targets share the same state, so the result is network-wide rather than localized",
    "geometry_unavailable": (
        "affected targets are model candidates only; physical localization is unavailable because sensor "
        "geometry is not configured"
    ),
    "unresolved_global_anomaly": "a global anomaly was detected but was not localized to any target",
    "inconclusive": "localization is inconclusive",
}
_BAND_LABELS = {
    "0_5_hz": "0-5 Hz", "5_20_hz": "5-20 Hz", "20_50_hz": "20-50 Hz", "50_100_hz": "50-100 Hz",
    "100_200_hz": "100-200 Hz", "200_plus_hz": "above 200 Hz",
}


def _plain(value: Any) -> str:
    return str(value or "").replace("_", " ").strip() or "unknown"


def _board_name(sensor_id: Any) -> str:
    text = str(sensor_id or "")
    if text in {"", "baseline", "reference"}:
        return "Reference board"
    return text.replace("target_", "Target ").replace("_", " ")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantity(value: Any, digits: int = 4, unit: str = "") -> str:
    number = _float(value)
    if number is None:
        return "not available"
    text = f"{number:.2e}" if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4) else f"{number:.{digits}g}"
    return f"{text} {unit}".strip()


def _target_lines(targets: list[dict[str, Any]]) -> list[str]:
    lines = []
    for target in targets:
        name = _board_name(target.get("sensor_id"))
        target_class = str(target.get("class") or "")
        if target_class == "data_invalid":
            lines.append(f"- {name}: data invalid (signal quality too poor to classify)")
        elif target.get("affected"):
            candidate = " (model candidate, physical localization unverified)" if target.get("localization_candidate") else ""
            lines.append(f"- {name}: affected, class {_plain(target_class)}, severity {_plain(target.get('severity'))}{candidate}")
        else:
            lines.append(f"- {name}: normal, matches the reference")
    return lines


def _latest_monitor_result() -> dict[str, Any] | None:
    """The full pipeline result of the latest live check, or None when there is none or it is stale."""
    with _POPUP_LOCK:
        latest = copy.deepcopy(_LATEST_MONITOR)
        age_s = time.monotonic() - _LATEST_MONITOR_AT
    if latest is None or age_s > _monitor_stale_s():
        return None
    result = latest.get("result")
    return result if isinstance(result, dict) else None


def _saved_evidence_text(detail: dict[str, Any]) -> str:
    """The exact evidence text the Agent was shown for a saved window, when the model input was kept."""
    path = str(detail.get("model_input_file") or "").strip()
    if not path:
        return ""
    window_dir = webchat_server._resolve_anomaly_window_dir()
    candidates = [Path(path), window_dir / path, window_dir / "windows" / Path(path).name]
    model_input = None
    for candidate in candidates:
        try:
            model_input = json.loads(candidate.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError, TypeError):
            continue
    if model_input is None:
        return ""
    text = model_input.get("evidence_text") if isinstance(model_input, dict) else None
    return str(text or "").strip()[:1500]


def _largest_transmissibility(board: dict[str, Any]) -> tuple[str, str, float] | None:
    relation = board.get("reference_relation") if isinstance(board.get("reference_relation"), dict) else {}
    table = relation.get("transmissibility_db") if isinstance(relation.get("transmissibility_db"), dict) else {}
    best: tuple[str, str, float] | None = None
    for axis, by_band in table.items():
        if not isinstance(by_band, dict):
            continue
        for band, value in by_band.items():
            number = _float(value)
            if number is not None and (best is None or abs(number) > abs(best[2])):
                best = (str(axis), str(band), number)
    return best


def _evidence_lines(result: dict[str, Any]) -> list[str]:
    """One measured-evidence line per board from the pipeline metrics and the model's physics evidence."""
    metrics_by_id = {
        str(metric.get("sensor_id")): metric
        for metric in (result.get("all_sensor_metrics") or [])
        if isinstance(metric, dict) and metric.get("sensor_id")
    }
    if not metrics_by_id:
        return []
    baseline_id = str(result.get("baseline_sensor_id") or "baseline")
    monitor = ((result.get("model_metadata") or {}).get("vibrogemma_monitor") or {})
    evidence = monitor.get("model_evidence") if isinstance(monitor.get("model_evidence"), dict) else {}
    boards = evidence.get("boards") if isinstance(evidence.get("boards"), dict) else {}
    reference_rms = _float((metrics_by_id.get(baseline_id) or result.get("baseline_metrics") or {}).get("acceleration_rms_g"))
    order = [baseline_id] + sorted(sensor_id for sensor_id in metrics_by_id if sensor_id != baseline_id)
    lines = []
    for sensor_id in order:
        metric = metrics_by_id.get(sensor_id) or {}
        board = boards.get("reference" if sensor_id == baseline_id else sensor_id) or {}
        parts = []
        rms = _float(metric.get("acceleration_rms_g"))
        if rms is not None:
            relative = ""
            if sensor_id != baseline_id and reference_rms:
                relative = f" ({20 * math.log10(rms / reference_rms):+.1f} dB vs reference)"
            parts.append(f"RMS {_quantity(rms, 3, 'g')}{relative}")
        peak = _float(metric.get("acceleration_peak_g"))
        if peak is not None:
            parts.append(f"peak {_quantity(peak, 3, 'g')}")
        crest = _float(metric.get("crest_factor"))
        if crest is not None:
            parts.append(f"crest factor {crest:.1f}")
        kurtosis = _float(metric.get("kurtosis"))
        if kurtosis is not None:
            parts.append(f"kurtosis {kurtosis:.1f}")
        dominant = _float(metric.get("dominant_frequency_hz"))
        if dominant is not None:
            parts.append(f"dominant {_quantity(dominant, 4, 'Hz')}")
        bands = metric.get("band_powers") if isinstance(metric.get("band_powers"), dict) else {}
        shares = [(key, _float(value)) for key, value in bands.items()]
        shares = [(key, share) for key, share in shares if share is not None]
        if shares:
            key, share = max(shares, key=lambda item: item[1])
            parts.append(f"{share * 100:.0f}% of energy {_BAND_LABELS.get(key, _plain(key))}")
        modal = board.get("modal_estimates") if isinstance(board.get("modal_estimates"), list) else []
        if modal and isinstance(modal[0], dict):
            frequency = _float(modal[0].get("frequency_hz"))
            damping = _float(modal[0].get("damping_ratio"))
            if frequency is not None:
                parts.append(f"strongest mode {_quantity(frequency, 3, 'Hz')}" + (f" (damping ratio {damping:.3f})" if damping is not None else ""))
        if sensor_id != baseline_id:
            largest = _largest_transmissibility(board)
            if largest:
                parts.append(f"largest transmissibility vs reference {largest[2]:+.1f} dB at {largest[0]}/{largest[1]}")
        flags = [str(flag) for flag in (metric.get("quality_flags") or []) if flag]
        if flags:
            parts.append("quality flags: " + ", ".join(flags))
        lines.append(f"- {_board_name(sensor_id)}: " + ("; ".join(parts) if parts else "no metrics recorded"))
    return lines


def _label_vocabulary(result: dict[str, Any]) -> str:
    monitor = ((result.get("model_metadata") or {}).get("vibrogemma_monitor") or {})
    contract = monitor.get("trained_label_contract") if isinstance(monitor.get("trained_label_contract"), dict) else {}
    classes = [str(item) for item in (contract.get("classes") or ["normal", "unknown_anomaly"])]
    severities = [str(item) for item in (contract.get("severities") or ["none", "advisory"])]
    quality_class = str(contract.get("quality_class") or "data_invalid")
    return (
        "Label vocabulary of this checkpoint: classes " + " / ".join(_plain(item) for item in classes)
        + f" (plus {_plain(quality_class)} when signal quality blocks a decision); severities "
        + " / ".join(_plain(item) for item in severities)
        + ". 'Unknown anomaly' means the board's vibration pattern deviates from the reference board without a named "
        "fault type, because this checkpoint has no fault-type classes; 'advisory' is the mildest severity and the "
        "only non-normal one it can assign. 'Model candidate' means the board was flagged but physical "
        "localization is unverified."
    )


def _context_brief(context: dict[str, Any], result: dict[str, Any] | None = None, saved_evidence: str = "") -> str:
    """Render the chat context as a compact evidence sheet the model can answer from."""
    source = str(context.get("source") or "latest_gemma_check")
    lines: list[str] = []
    if source == "spectrum":
        axis = str(context.get("axis") or "norm")
        axis_text = "vector norm of X/Y/Z" if axis == "norm" else f"{axis.upper()} axis"
        lines.append(
            f"Evidence: power spectral density of {_board_name(context.get('board'))}, {axis_text}, "
            f"{_plain(context.get('estimator') or 'welch')} estimator."
        )
        lines.append(
            f"RMS acceleration: {_quantity(context.get('rms_g'), unit='g')}. "
            f"Variance: {_quantity(context.get('variance_g2'), unit='g^2')}."
        )
        lines.append(
            f"Dominant frequency: {_quantity(context.get('dominant_frequency_hz'), 4, 'Hz')}. "
            f"Sampling rate: {_quantity(context.get('sampling_rate_hz'), 5, 'Hz')}."
        )
        peaks = [peak for peak in (context.get("top_peaks") or []) if isinstance(peak, dict)][:5]
        if peaks:
            lines.append("Strongest peaks (frequency, PSD level, prominence):")
            lines.extend(
                f"- {_quantity(peak.get('frequency_hz'), 4, 'Hz')} at {_quantity(peak.get('psd_g2_per_hz'), 3, 'g^2/Hz')}"
                + (f", prominence {_quantity(peak.get('prominence_db'), 3, 'dB')}" if _float(peak.get("prominence_db")) is not None else "")
                for peak in peaks
            )
        lines.append("No reference comparison or Agent classification belongs to this spectrum; only the numbers above are known.")
        return "\n".join(lines)

    result = result if isinstance(result, dict) else {}
    targets = [target for target in (context.get("targets") or []) if isinstance(target, dict)]
    window = result.get("window") if isinstance(result.get("window"), dict) else {}
    window_id = context.get("window_id") or result.get("window_id") or ""
    if source == "replay":
        lines.append(
            f"Evidence: saved Agent check {window_id} recorded "
            f"{context.get('created_at_utc') or window.get('end_utc') or 'at an unknown time'} (10 s window)."
        )
    else:
        age = context.get("updated_age_s")
        if age is None:
            age = context.get("age_s")
        age_text = f"updated {float(age):.0f} s ago" if isinstance(age, (int, float)) else "age unknown"
        ended = f", window ended {window.get('end_utc')}" if window.get("end_utc") else ""
        lines.append(f"Evidence: latest live Agent check of the current 10 s window{' ' + str(window_id) if window_id else ''}, {age_text}{ended}.")
    state = str(context.get("network_state") or context.get("state") or "unknown")
    if state == "data_invalid":
        lines.append(
            "Network verdict: data invalid; the data-quality gate blocks any normal or anomaly verdict for this window."
        )
        lines.append("Localization: not assessed because the window's data is invalid.")
    else:
        verdict = f"Network verdict: {_plain(state)}."
        if context.get("global_class"):
            verdict += f" Global class: {_plain(context.get('global_class'))}."
        lines.append(verdict)
        if context.get("global_target_consistent") is False:
            lines.append("The global decision and the per-target decisions disagree, so the check is inconclusive.")
        localization = _LOCALIZATION_TEXT.get(str(context.get("localization_status") or ""))
        if localization:
            lines.append(f"Localization: {localization}.")
    lines.append("Board states:")
    lines.append("- Reference board: live reference for every comparison")
    lines.extend(_target_lines(targets) or ["- Target boards: no per-target decisions available"])
    measured = _evidence_lines(result)
    if measured:
        lines.append(
            "Measured evidence over the window (norm of X/Y/Z unless an axis is named; transmissibility bands are "
            "per axis; dB values compare the target with the reference board):"
        )
        lines.extend(measured)
    elif source == "replay":
        metrics = []
        for board in (context.get("boards") or []):
            if not isinstance(board, dict):
                continue
            parts = [f"RMS {_quantity(board.get('rms_g'), 3, 'g')}", f"peak {_quantity(board.get('peak_g'), 3, 'g')}"]
            if board.get("dominant_frequency_hz") is not None:
                parts.append(f"structural peak {_quantity(board.get('dominant_frequency_hz'), 4, 'Hz')}")
            metrics.append(f"- {_board_name(board.get('sensor_id'))}: " + ", ".join(parts))
        if metrics:
            lines.append("Measured levels in this window:")
            lines.extend(metrics)
    if saved_evidence:
        lines.append("Evidence text the Agent was shown for this window:")
        lines.append(saved_evidence)
    lines.append(_label_vocabulary(result))
    if context.get("soft_tokens_valid"):
        lines.append(f"Model input: {context.get('soft_tokens_valid')} valid continuous vibration tokens.")
    deployment = context.get("deployment") if isinstance(context.get("deployment"), dict) else {}
    if deployment.get("promotion_status") or deployment.get("permitted_use"):
        lines.append(
            f"Checkpoint status: {_plain(deployment.get('promotion_status'))}; "
            f"permitted use: {_plain(deployment.get('permitted_use'))}."
        )
    return "\n".join(lines)


def _chat_evidence(context: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Look up the full pipeline result behind a chat context so the sheet can carry measured evidence."""
    source = str(context.get("source") or "latest_gemma_check")
    if source == "replay":
        detail = _load_replay_detail(str(context.get("window_id") or ""))
        if not detail:
            return {}, ""
        result = detail.get("pipeline_result")
        return (result if isinstance(result, dict) else {}), _saved_evidence_text(detail)
    if source == "latest_gemma_check":
        return (_latest_monitor_result() or {}), ""
    return {}, ""


def _gemma_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    message = str(payload.get("message") or "").strip()
    if not message:
        raise ValueError("message is required")
    if len(message) > 2400:
        raise ValueError("message is too long")
    supplied_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    context = supplied_context or _gemma_context()
    if context.get("available") is False:
        answer = (
            "The latest Agent check is stale, so I can't answer from live evidence until a fresh check completes. "
            "Live waveforms are still available on the monitor."
            if context.get("stale")
            else "The latest Agent check is unavailable, so I can't answer from live model evidence yet. "
            "Open the live monitor to see when the next check completes."
        )
        return {"answer": answer, "context": context}
    history = []
    for item in payload.get("history") or []:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            history.append({"role": item["role"], "content": content[:1800]})
    history = history[-6:]
    result, saved_evidence = _chat_evidence(context)
    messages = [
        {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Evidence sheet:\n{_context_brief(context, result, saved_evidence)}\n\nQuestion: {message}"},
    ]
    base_url = os.environ.get("MAIN_AGENT_BASE_URL") or "http://127.0.0.1:18181/v1"
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": os.environ.get("MAIN_AGENT_MODEL", "vibrogemma"),
            "messages": messages,
            "max_tokens": 320,
            "temperature": 0.35,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = webchat_server.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('MAIN_AGENT_API_KEY', 'EMPTY')}",
        },
        method="POST",
    )
    with webchat_server.urlopen(request, timeout=60.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    answer = str(data["choices"][0]["message"]["content"]).strip()
    if not answer:
        raise RuntimeError("Agent returned an empty response")
    return {"answer": _agent_copy(answer), "context": context}


def _synthetic_injection_status() -> dict[str, Any]:
    configured = os.environ.get("VIBRO_LUMO_TRAINING_INJECTION", "").strip().lower()
    target = "target_3" if configured in {"1", "true", "yes", "on"} else configured
    tests = {"target_3": "DAM4_010", "target_5": "DAM6_010"}
    enabled = target in tests
    return {
        "ok": True,
        "enabled": enabled,
        "mode": "lumo_live_overlay",
        "target": target if enabled else None,
        "condition": tests.get(target),
        "expected_target": target if enabled else None,
    }


def _monitor_matches_current_injection(payload: dict[str, Any]) -> bool:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    metadata = result.get("model_metadata") if isinstance(result.get("model_metadata"), dict) else {}
    monitor = metadata.get("vibrogemma_monitor") if isinstance(metadata.get("vibrogemma_monitor"), dict) else {}
    capture = monitor.get("capture") if isinstance(monitor.get("capture"), dict) else {}
    injection = (
        capture.get("synthetic_injection")
        if isinstance(capture.get("synthetic_injection"), dict)
        else {}
    )
    actual = injection.get("selected_target") if injection.get("active") else None
    expected = _synthetic_injection_status().get("target")
    return actual == expected


def _lumo_waveform_payload(query: dict[str, list[str]]) -> dict[str, Any] | None:
    status = _synthetic_injection_status()
    if not status["enabled"]:
        return None
    machine_id = webchat_server._clean(webchat_server._first(query, "machine_id")) or "baseline"
    axis = (webchat_server._first(query, "axis") or "norm").strip().lower()
    axis_indices = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
    if machine_id not in SLOTS or axis not in {"norm", "magnitude", "vector_norm", *axis_indices}:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "LUMO waveform requires a six-board sensor id and a valid axis",
        }
    payload = webchat_server._live_vibrometer_payload(query)
    if not payload.get("ok"):
        return payload
    try:
        values = np.asarray(payload["samples_g"], dtype=np.float64)
        stride = max(1, int(payload.get("downsample_stride") or 1))
        effective_rate = float(payload["sampling_rate_hz"]) / stride
        templates, overlay = _lumo_overlay_templates(
            str(status["target"]),
            round(effective_rate, 6),
            int(values.size),
            _lumo_overlay_gain(),
        )
        template = templates[machine_id][:, -values.size:]
        injected = (
            np.linalg.norm(template, axis=0)
            if axis in {"norm", "magnitude", "vector_norm"}
            else template[axis_indices[axis]]
        )
        injected = injected - np.mean(injected)
        if axis not in {"z", "2"}:
            live_residual = values - np.mean(values)
            live_rms = float(np.sqrt(np.mean(live_residual * live_residual)))
            template_rms = float(
                np.sqrt(np.mean((injected / _lumo_overlay_gain()) ** 2))
            )
            live_scale = template_rms / live_rms if live_rms > 1e-12 else 0.0
            values = (
                live_residual * live_scale + injected
            ) / (1.0 + _lumo_overlay_gain())
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "overlay_failed",
            "message": webchat_server.format_exception_for_response(exc),
        }
    payload["samples_g"] = [float(value) for value in values]
    payload["stats"] = {
        "min_g": float(values.min()),
        "max_g": float(values.max()),
        "mean_g": float(values.mean()),
        "rms_g": float(np.sqrt(np.mean(values * values))),
    }
    payload["metadata"] = {
        **(payload.get("metadata") or {}),
        "synthetic": True,
        "test_only": True,
        "waveform_source": "live_with_lumo_overlay",
        **overlay,
        "overlay_axis_present": axis not in {"z", "2"},
    }
    payload.setdefault("diagnostics", {})["overlay"] = True
    return payload


def _set_synthetic_injection(enabled: bool, target: str = "target_3") -> dict[str, Any]:
    global _LATEST_MONITOR, _LATEST_MONITOR_AT
    if enabled and target not in {"target_3", "target_5"}:
        raise ValueError("target must be target_3 or target_5")
    if enabled:
        _lumo_training_replay(target)
    os.environ["VIBRO_SYNTHETIC_TARGET5_ENABLED"] = "0"
    os.environ["VIBRO_LUMO_TRAINING_INJECTION"] = target if enabled else "0"
    with _POPUP_LOCK:
        _LATEST_MONITOR = None
        _LATEST_MONITOR_AT = 0.0
    return _synthetic_injection_status()


class VibroGemmaHandler(webchat_server.WebchatHandler):
    server_version = "VibroAgentWeb/1.0"

    def do_GET(self) -> None:
        if not self._request_origin_allowed():
            self._send_json({"ok": False, "error": "origin_not_allowed"}, status=HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/graph", "/spectrum", "/replay", "/ask", "/chat"}:
            self._send_bytes((_STATIC_ROOT / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            return
        assets = {
            "/assets/vibroagent.css": (_STATIC_ROOT / "vibroagent.css", "text/css; charset=utf-8"),
            "/assets/vibroagent.js": (_STATIC_ROOT / "vibroagent.js", "application/javascript; charset=utf-8"),
            "/assets/phosphor-icons.svg": (_STATIC_ROOT / "phosphor-icons.svg", "image/svg+xml"),
            "/assets/st-logo.png": (
                Path(__file__).resolve().parent / "static" / "st_logo_2020.png",
                "image/png",
            ),
            **{
                f"/assets/fonts/{name}.woff2": (_STATIC_ROOT / "fonts" / f"{name}.woff2", "font/woff2")
                for name in ("NotoSans-Regular", "NotoSans-Bold", "NotoSansMono-Regular", "NotoSansMono-Bold")
            },
        }
        if parsed.path in assets:
            path, content_type = assets[parsed.path]
            self._send_bytes(path.read_bytes(), content_type=content_type)
            return
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "app": "vibroagent",
                    "inference": "gemma",
                    "deployment_contract_version": WEBCHAT_DEPLOYMENT_CONTRACT_VERSION,
                    "failure_policy": "data_invalid_never_inherited_normal",
                    "monitor_decision_mode": os.environ.get(
                        "VIBRO_MONITOR_DECISION_MODE", ""
                    ).strip().lower(),
                    "model_base_url": (
                        os.environ.get("MAIN_AGENT_BASE_URL")
                        or os.environ.get("QWEN_BASE_URL")
                        or ""
                    ).rstrip("/"),
                    "model": (
                        os.environ.get("MAIN_AGENT_MODEL")
                        or os.environ.get("QWEN_MODEL")
                        or ""
                    ),
                }
            )
            return
        if parsed.path == "/api/vibro/sensors":
            self._send_json(webchat_server._live_sensors_payload(query))
            return
        if parsed.path == "/api/vibro/waveform":
            self._send_json(
                _lumo_waveform_payload(query)
                or webchat_server._live_vibrometer_payload(query)
            )
            return
        if parsed.path == "/api/vibro/monitor":
            self._send_json(webchat_server._agent_monitor_payload(query))
            return
        if parsed.path == "/api/vibro/synthetic-injection":
            self._send_json(_synthetic_injection_status())
            return
        if parsed.path == "/api/vibro/spectrum":
            self._send_json(webchat_server._psd_fft_payload(query))
            return
        if parsed.path == "/api/vibro/replays":
            self._send_json(_gemma_replays_payload(query))
            return
        if parsed.path == "/api/vibro/replay":
            payload = _gemma_replay_payload(query)
            self._send_json(payload, status=HTTPStatus.OK if payload.get("ok") else HTTPStatus.NOT_FOUND)
            return
        self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._request_origin_allowed():
            self._send_json({"ok": False, "error": "origin_not_allowed"}, status=HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        if path == "/api/vibro/synthetic-injection":
            body = self._read_json()
            enabled = body.get("enabled") if isinstance(body, dict) else None
            target = str(body.get("target") or "target_3") if isinstance(body, dict) else "target_3"
            if not isinstance(enabled, bool):
                self._send_json(
                    {"ok": False, "error": "enabled_must_be_boolean"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                response = _set_synthetic_injection(enabled, target)
            except ValueError as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(response)
            return
        if path != "/api/vibro/chat":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        if not webchat_server._chat_inflight_try_begin():
            self._send_json(
                {"ok": False, "error": "gemma_busy", "message": "Agent is busy with another request."},
                status=HTTPStatus.CONFLICT,
            )
            return
        try:
            self._send_json({"ok": True, **_gemma_chat_payload(self._read_json())})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json(
                {"ok": False, "error": "gemma_chat_failed", "message": str(exc)},
                status=HTTPStatus.BAD_GATEWAY,
            )
        finally:
            webchat_server._chat_inflight_end()


def _monitor_loop(original_monitor) -> None:
    global _LATEST_MONITOR, _LATEST_MONITOR_AT
    _pin_current_thread("VIBROGEMMA_INFERENCE_CPUS", "6-7")
    while True:
        cycle_started = time.monotonic()
        with _POPUP_LOCK:
            query = copy.deepcopy(_MONITOR_QUERY)
        query["skip_cache"] = ["1"]
        try:
            payload = original_monitor(query)
            if not payload.get("paused_for_chat") and _monitor_matches_current_injection(payload):
                with _POPUP_LOCK:
                    _LATEST_MONITOR = copy.deepcopy(payload)
                    _LATEST_MONITOR_AT = time.monotonic() - _capture_age_s(payload)
        except Exception as exc:
            print(f"VibroAgent monitor cycle failed: {exc}", flush=True)
            with _POPUP_LOCK:
                _LATEST_MONITOR = {
                    "ok": False,
                    "status": "unavailable",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "anomaly": False,
                    "popup": {"show_popup": False},
                    "result": {
                        "model_metadata": {
                            "agent_mode": "vibrogemma_live",
                            "monitor_llm_model_used": False,
                        }
                    },
                }
                _LATEST_MONITOR_AT = time.monotonic()
        time.sleep(max(0.1, _monitor_interval_s() - (time.monotonic() - cycle_started)))


def _install():
    webchat_server.ANOMALOUS_NETWORK_LABELS.update({"unknown_anomaly", "data_invalid"})
    webchat_server.ANOMALOUS_SENSOR_LABELS.update({"unknown_anomaly", "data_invalid"})
    original = webchat_server._apply_monitor_llm_decision
    original_popup = webchat_server._build_agent_popup
    original_monitor = webchat_server._agent_monitor_payload

    def decide(result, **kwargs):
        mode = os.environ.get("VIBRO_MONITOR_DECISION_MODE", "").strip().lower()
        if mode not in {"gemma", "vibrogemma"} or webchat_server._existing_pipeline_llm_active(result):
            return original(result, **kwargs)
        return apply_vibrogemma_monitor_decision(
            result,
            model_base_url=kwargs["model_base_url"],
            model_api_key=kwargs["model_api_key"],
            model=kwargs["model"],
            model_timeout_s=kwargs.get("model_timeout_s", 8.0),
            config_path=kwargs.get("config_path"),
            process_reader=kwargs.get("process_reader", False),
        )

    webchat_server._apply_monitor_llm_decision = decide
    webchat_server._build_agent_popup = lambda result: _apply_popup_policy(original_popup(result), result)

    def latest_monitor(query):
        with _POPUP_LOCK:
            latest = copy.deepcopy(_LATEST_MONITOR)
            age_s = time.monotonic() - _LATEST_MONITOR_AT
        if latest is None:
            return {
                "ok": True,
                "status": "ok",
                "paused_for_chat": True,
                "warming": True,
                "deployment": _deployment_status(),
            }
        latest["cached"] = True
        diagnostics = dict(latest.get("diagnostics") or {})
        diagnostics["cached_age_s"] = round(age_s, 3)
        latest["diagnostics"] = diagnostics
        latest["deployment"] = _deployment_status()
        if age_s > _monitor_stale_s():
            latest["status"] = "stale"
            latest["stale"] = True
            latest["anomaly"] = False
            latest["popup"] = {**dict(latest.get("popup") or {}), "show_popup": False}
            latest["agent"] = {**dict(latest.get("agent") or {}), "llm_agent_active": False, "stale": True}
        return latest

    webchat_server._agent_monitor_payload = latest_monitor
    webchat_server.WebchatHandler = VibroGemmaHandler
    return original_monitor


def main() -> None:
    _pin_current_thread("VIBROGEMMA_UI_CPUS", "0-5")
    original_monitor = _install()
    threading.Thread(
        target=_monitor_loop,
        args=(original_monitor,),
        name="vibrogemma-monitor",
        daemon=True,
    ).start()
    webchat_server.main()


if __name__ == "__main__":
    main()
