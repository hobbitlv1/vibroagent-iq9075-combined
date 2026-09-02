from __future__ import annotations

"""Final E3+UPC localisation protocol utilities (v0.15.3).

The module is deliberately independent from repository-specific download code.  It
patches already verified strict indexes after source EDA/indexing and before episode
materialisation.  It never reads signal arrays and cannot open a frozen test waveform.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import random
import re

from .data.lumo_geometry import (
    LUMO_DAMAGE_ELEVATIONS_M,
    LUMO_DAMAGE_TO_TARGET,
    LUMO_SENSOR_ELEVATIONS_M,
    LUMO_TARGET_MEASUREMENT_LEVELS,
    LUMO_ZONE_CONTRACT,
    lumo_zone_vertical_distance_m,
)

BANNED_DATASET_TOKENS = ("hbta", "hell_bridge", "z24")
QUGS_TARGETS = (1, 2, 3, 4, 5)
SPLITS = ("train", "validation", "test")

class FinalProtocolError(RuntimeError):
    pass


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FinalProtocolError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    if not rows:
        raise FinalProtocolError(f"{path}: no rows")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _all_text(row: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                chunks.append(str(key))
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)
        elif value is not None:
            chunks.append(str(value))
    visit(row)
    return " ".join(chunks)


def _normalise_targets(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise FinalProtocolError(f"invalid affected_targets: {value!r}")
    result: set[int] = set()
    for raw in value:
        if isinstance(raw, bool):
            raise FinalProtocolError(f"invalid boolean target: {raw!r}")
        text = str(raw).strip().lower().removeprefix("target_")
        if text not in {"1", "2", "3", "4", "5"}:
            raise FinalProtocolError(f"invalid target identifier: {raw!r}")
        result.add(int(text))
    return sorted(result)


def _candidate_qugs_joint(value: Any) -> int | None:
    """Parse one QUGS damaged-joint label without inspecting sensor-panel metadata.

    QUGS strict indexes contain many nested ``source_sensor_name=joint_XX`` values.
    Those are the six selected measurement locations, not necessarily the damaged
    joint.  Only explicit state/source fields are admissible here.
    """

    if value in (None, ""):
        return None
    text = str(value).strip()
    patterns = (
        # Direct numeric identifiers, including zero-padded forms.
        r"^\s*0*([1-9]|[12]\d|30)\s*$",
        # Canonical strict-index state/condition labels.
        r"(?:loosened|damaged|damage)[_ -]?joint[_ -]?0*([1-9]|[12]\d|30)(?!\d)",
        r"(?:joint|bolt)[_ -]?(?:no\.?[_ -]?)?0*([1-9]|[12]\d|30)(?!\d)",
        r"qugs[_ -]?state[_ -]?0*([1-9]|[12]\d|30)(?!\d)",
        # Publisher filenames: zzzAD1..30 and zzzBD1..30.
        r"(?:^|[/\\])zzz[AB]D0*([1-9]|[12]\d|30)(?:\.[^.]+)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_joint(row: Mapping[str, Any]) -> int | None:
    """Recover the physical QUGS damage joint from explicit label fields only.

    The previous row-wide scan could mistake a nested board sensor such as
    ``joint_12`` for the damaged joint, especially because canonical state labels
    are zero padded (``loosened_joint_01``).  Conflicting explicit sources now fail
    closed instead of silently selecting one.
    """

    candidate_fields = (
        "damaged_joint",
        "damage_joint",
        "joint",
        "joint_id",
        "damage_location",
        "condition",
        "state",
        "state_name",
        "session_id",
        "experiment_id",
        "source_file",
        "source_path",
        "filename",
    )
    candidates: list[tuple[str, int]] = []
    for key in candidate_fields:
        if key not in row:
            continue
        joint = _candidate_qugs_joint(row.get(key))
        if joint is not None:
            candidates.append((key, joint))

    unique = sorted({joint for _, joint in candidates})
    if len(unique) > 1:
        raise FinalProtocolError(
            "QUGS explicit joint fields disagree: "
            + ", ".join(f"{key}={joint}" for key, joint in candidates)
        )
    return unique[0] if unique else None


def _candidate_qugs_campaign(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    direct = text.upper()
    if direct in {"A", "B", "DATASET A", "DATASET B"}:
        return direct[-1]
    patterns = (
        r"(?:^|[/\\_-])dataset[_ -]?([AB])(?:$|[/\\_. -])",
        r"campaign[_ -]?([AB])(?:$|[:/\\_. -])",
        r"(?:^|[/\\])zzz([AB])[UD]\d*(?:\.[^.]+)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _extract_campaign(row: Mapping[str, Any]) -> str | None:
    """Recover QUGS campaign A/B from explicit metadata or publisher filename."""

    candidate_fields = (
        "campaign",
        "dataset_part",
        "subset",
        "dataset",
        "source_campaign",
        "experiment_id",
        "source_file",
        "source_path",
        "filename",
    )
    candidates: list[tuple[str, str]] = []
    for key in candidate_fields:
        if key not in row:
            continue
        campaign = _candidate_qugs_campaign(row.get(key))
        if campaign is not None:
            candidates.append((key, campaign))
    unique = sorted({campaign for _, campaign in candidates})
    if len(unique) > 1:
        raise FinalProtocolError(
            "QUGS explicit campaign fields disagree: "
            + ", ".join(f"{key}={campaign}" for key, campaign in candidates)
        )
    return unique[0] if unique else None


def qugs_target_for_joint(joint: int) -> int:
    if joint < 1 or joint > 30:
        raise FinalProtocolError(f"QUGS joint out of range: {joint}")
    return ((joint - 1) % 5) + 1


def build_qugs_split_map(joint_to_target: Mapping[int, int] | Iterable[int], *, seed: int) -> dict[int, str]:
    """Return exact target-stratified 4/1/1 joint splits.

    When a mapping is supplied, the target slot from the verified QUGS strict index is
    authoritative.  This keeps the localisation label aligned with the selected sensor
    panel.  The iterable fallback exists only for synthetic regression fixtures.
    """
    if isinstance(joint_to_target, Mapping):
        mapping = {int(joint): int(target) for joint, target in joint_to_target.items()}
    else:
        mapping = {int(joint): qugs_target_for_joint(int(joint)) for joint in joint_to_target}
    by_target = {target: [] for target in QUGS_TARGETS}
    for joint, target in sorted(mapping.items()):
        if target not in QUGS_TARGETS:
            raise FinalProtocolError(f"QUGS joint {joint} has invalid target slot {target}")
        by_target[target].append(joint)
    result: dict[int, str] = {}
    for target, values in by_target.items():
        if len(values) != 6:
            raise FinalProtocolError(
                f"QUGS target_{target} has {len(values)} unique damaged joints; exact 6 required for 4/1/1"
            )
        rng = random.Random(int(seed) * 1009 + target)
        shuffled = list(values)
        rng.shuffle(shuffled)
        for joint in shuffled[:4]: result[joint] = "train"
        result[shuffled[4]] = "validation"
        result[shuffled[5]] = "test"
    return result



def patch_qugs_index(input_path: Path, output_path: Path, *, seed: int) -> dict[str, Any]:
    rows = _read_jsonl(input_path)
    parsed: list[tuple[dict[str, Any], int | None, str | None, int | None]] = []
    joint_to_target: dict[int, int] = {}
    for original in rows:
        row = dict(original)
        targets = _normalise_targets(row.get("affected_targets"))
        joint = _extract_joint(row)
        campaign = _extract_campaign(row)
        changed = bool(targets) or joint is not None
        if changed and joint is None:
            raise FinalProtocolError("QUGS changed row has no recoverable damaged joint")
        if campaign not in {"A", "B"}:
            raise FinalProtocolError(
                f"QUGS row has no unambiguous A/B campaign in explicit source metadata: "
                f"{row.get('source_file')!r}"
            )
        target: int | None = None
        if joint is not None:
            if len(targets) != 1:
                raise FinalProtocolError(
                    f"QUGS joint {joint} must have exactly one verified affected target; observed {targets}"
                )
            target = int(targets[0])
            expected_target = qugs_target_for_joint(joint)
            if target != expected_target:
                raise FinalProtocolError(
                    f"QUGS joint {joint} has target_{target}, but deterministic strict-index "
                    f"panel construction requires target_{expected_target}"
                )
            previous = joint_to_target.setdefault(joint, target)
            if previous != target:
                raise FinalProtocolError(
                    f"QUGS joint {joint} maps inconsistently to target_{previous} and target_{target}"
                )
        parsed.append((row, joint, campaign, target))
    split_map = build_qugs_split_map(joint_to_target, seed=seed)
    campaign_support: dict[int, set[str]] = {joint: set() for joint in split_map}
    healthy_campaign_support: set[str] = set()
    patched: list[dict[str, Any]] = []
    for row, joint, campaign, target in parsed:
        if joint is None:
            row["session_id"] = "qugs_state_00"
            row["split_group"] = "qugs_intact_A_B_paired"
            row["split"] = "train"
            row["healthy"] = True
            row["class_name"] = "normal"
            row["severity"] = "none"
            row["assigned_split"] = "train"
            row["affected_targets"] = []
            row["healthy"] = True
            row["class_name"] = "normal"
            row["severity"] = "none"
            row["target_supervision_available"] = True
            row["supervision_scope"] = "target_localized"
            row["global_class_name"] = "normal"
            healthy_campaign_support.add(campaign)
        else:
            assert target is not None
            split = split_map[joint]
            row["session_id"] = f"qugs_state_{joint:02d}"
            row["split_group"] = f"qugs_joint_{joint:02d}_A_B_paired"
            row["split"] = split
            row["healthy"] = False
            row["class_name"] = "unknown_anomaly"
            row["severity"] = "advisory"
            row["assigned_split"] = split
            row["affected_targets"] = [target]
            row["healthy"] = False
            row["class_name"] = "unknown_anomaly"
            row["severity"] = "advisory"
            row["target_supervision_available"] = True
            row["supervision_scope"] = "target_localized"
            row["global_class_name"] = "unknown_anomaly"
            row["localization_provenance"] = {
                "source": "QUGS verified strict-index damaged-joint mapping",
                "damaged_joint": joint,
                "campaign": campaign,
                "target_slot": target,
                "campaign_pair_grouped": True,
            }
            if campaign: campaign_support[joint].add(campaign)
        patched.append(row)
    missing_campaign_pairs = sorted(
        joint for joint, values in campaign_support.items() if values != {"A", "B"}
    )
    if missing_campaign_pairs:
        observed = {joint: sorted(campaign_support[joint]) for joint in missing_campaign_pairs}
        raise FinalProtocolError(
            f"QUGS A/B campaign pair incomplete for joints {missing_campaign_pairs}: {observed}"
        )
    if healthy_campaign_support != {"A", "B"}:
        raise FinalProtocolError(
            "QUGS intact A/B campaign pair is incomplete: "
            f"{sorted(healthy_campaign_support)}"
        )
    _write_jsonl(output_path, patched)
    counts = {split: {f"target_{target}": 0 for target in QUGS_TARGETS} for split in SPLITS}
    for joint, split in split_map.items():
        counts[split][f"target_{joint_to_target[joint]}"] += 1
    expected = {
        "train": {f"target_{t}": 4 for t in QUGS_TARGETS},
        "validation": {f"target_{t}": 1 for t in QUGS_TARGETS},
        "test": {f"target_{t}": 1 for t in QUGS_TARGETS},
    }
    if counts != expected: raise FinalProtocolError(f"QUGS target split mismatch: {counts}")
    return {
        "schema": "vibrogemma-qugs-localization-index-v0.15.3",
        "passed": True,
        "input": str(input_path), "output": str(output_path), "row_count": len(patched),
        "damaged_joint_count": len(split_map),
        "campaign_pairing": "A_and_B_for_same_joint_share_split_group",
        "target_mapping_source": "explicit_QUGS_state_fields_cross_checked_against_deterministic_panel",
        "counts_by_split_and_target": counts,
        "healthy_condition_group_split": "train", "seed": int(seed),
    }



def _extract_lumo_damage_level(row: Mapping[str, Any]) -> str | None:
    for key in ("damage_level", "damage_location", "damage_zone", "dam_level"):
        if key in row and row[key] not in (None, "", 0, "0"):
            match = re.search(r"DAM[_ -]?([1-6])", str(row[key]), flags=re.IGNORECASE)
            if match:
                return f"DAM{match.group(1)}"
            text = str(row[key]).strip()
            if text in {"1", "2", "3", "4", "5", "6"}:
                return f"DAM{text}"
    match = re.search(r"DAM[_ -]?([1-6])", _all_text(row), flags=re.IGNORECASE)
    return f"DAM{match.group(1)}" if match else None


def _is_changed(row: Mapping[str, Any], damage: str | None) -> bool:
    if damage is not None:
        return True
    for key in ("state", "class_label", "global_class_name", "label", "condition"):
        text = str(row.get(key, "")).strip().lower()
        if text in {"changed", "damaged", "anomaly", "unknown_anomaly", "damage"}:
            return True
        if text in {"healthy", "normal", "intact", "undamaged"}:
            return False
    return bool(_normalise_targets(row.get("affected_targets")))


def patch_lumo_index(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(input_path)
    patched: list[dict[str, Any]] = []
    damage_counts: dict[str, int] = {}
    target_counts = {f"target_{target}": 0 for target in QUGS_TARGETS}
    for original in rows:
        row = dict(original)
        damage = _extract_lumo_damage_level(row)
        changed = _is_changed(row, damage)
        if changed:
            if damage is None:
                raise FinalProtocolError("LUMO changed row lacks publisher damage-level location")
            if damage not in LUMO_DAMAGE_TO_TARGET:
                raise FinalProtocolError(f"Unsupported LUMO damage level: {damage}")
            target = LUMO_DAMAGE_TO_TARGET[damage]
            row["affected_targets"] = [target]
            row["target_supervision_available"] = True
            row["supervision_scope"] = "target_localized"
            row["global_class_name"] = "unknown_anomaly"
            target_name = f"target_{target}"
            measurement_level = LUMO_TARGET_MEASUREMENT_LEVELS[target_name]
            row["localization_provenance"] = {
                **dict(row.get("localization_provenance") or {}),
                "source": "LUMO publisher state vector and ML/DAM geometry",
                "damage_level": damage,
                "damage_elevation_m": LUMO_DAMAGE_ELEVATIONS_M[damage],
                "target_slot": target,
                "target_measurement_level": measurement_level,
                "target_sensor_elevation_m": LUMO_SENSOR_ELEVATIONS_M[
                    measurement_level
                ],
                "vertical_distance_m": lumo_zone_vertical_distance_m(
                    target_name, damage
                ),
                "mapping_status": LUMO_ZONE_CONTRACT["mapping_status"],
                "exclusive_waveform_effect_claimed": False,
            }
            damage_counts[damage] = damage_counts.get(damage, 0) + 1
            target_counts[f"target_{target}"] += 1
        else:
            row["affected_targets"] = []
            row["target_supervision_available"] = True
            row["supervision_scope"] = "target_localized"
            row["global_class_name"] = "normal"
        patched.append(row)
    observed = set(damage_counts)
    expected_released = set(LUMO_ZONE_CONTRACT["released_damage_levels_expected"])
    missing = sorted(expected_released - observed)
    if missing:
        raise FinalProtocolError(f"LUMO released damage locations absent from index: {missing}")
    _write_jsonl(output_path, patched)
    return {
        "schema": "vibrogemma-lumo-localization-index-v0.15.3",
        "passed": True,
        "input": str(input_path),
        "output": str(output_path),
        "row_count": len(patched),
        "damage_level_counts": damage_counts,
        "positive_target_row_counts": target_counts,
        "zone_contract": LUMO_ZONE_CONTRACT,
    }


def patch_upc_global_only(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(input_path)
    patched: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["affected_targets"] = []
        row["target_supervision_available"] = False
        row["supervision_scope"] = "global_only"
        row["target_loss_masked"] = True
        patched.append(row)
    _write_jsonl(output_path, patched)
    return {
        "schema": "vibrogemma-upc-partial-label-index-v0.15.3",
        "passed": True,
        "row_count": len(patched),
        "target_supervision_available_rows": 0,
        "target_loss_masked_rows": len(patched),
    }


def audit_no_excluded_datasets(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, sort_keys=True).lower()
    found = sorted(token for token in BANNED_DATASET_TOKENS if token in text)
    return {"passed": not found, "excluded_tokens_found": found}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
