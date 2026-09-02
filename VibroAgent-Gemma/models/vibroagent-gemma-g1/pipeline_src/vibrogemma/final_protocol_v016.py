from __future__ import annotations

"""Official-geometry QUGS localization recovery for VibroGemma v0.16.1.

QUGS publishes one fixed acceleration channel at each of 30 numbered physical
beam-to-girder joints.  ADi/BDi denotes bolt loosening at Joint i; the physical
channels do not move between conditions.  This module therefore removes the old
condition-dependent panel completely.

The six-board projection used here is fixed before training:

* reference: Joint 30, a project-defined opposite-corner context channel;
* target_1..target_5: Joints 1..5, the fixed five-sensor row used in published
  QUGS localization work;
* target zones: the five longitudinal geometry columns of the published
  six-row-by-five-column numbered joint layout.

The original QUGS source does not designate a calm reference.  Joint 30 is
therefore documented as a fixed context reference, not as a guaranteed healthy
baseline.  Damage at Joint 30 remains an explicit target_5 case and is flagged
in provenance.
"""

from pathlib import Path
from typing import Any, Mapping
import copy
import json

from .final_protocol_v015 import (
    FinalProtocolError,
    QUGS_TARGETS,
    _extract_campaign,
    _extract_joint,
    _read_jsonl,
    _write_jsonl,
    build_qugs_split_map,
    patch_lumo_index,
    patch_upc_global_only,
)


FIXED_PANEL_SCHEMA = "vibrogemma-qugs-official-fixed-geometry-ledger-v1"
DEFAULT_FIXED_PANEL_LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "qugs_official_fixed_geometry_v0161.json"
)
OFFICIAL_REFERENCE_JOINT = 30
OFFICIAL_TARGET_JOINTS = {
    "target_1": 1,
    "target_2": 2,
    "target_3": 3,
    "target_4": 4,
    "target_5": 5,
}
OFFICIAL_TARGET_ZONE_JOINTS = {
    f"target_{target}": tuple(target + 5 * row for row in range(6))
    for target in QUGS_TARGETS
}
OFFICIAL_DAMAGE_TO_TARGET = {
    joint: ((joint - 1) % 5) + 1 for joint in range(1, 31)
}
OFFICIAL_ROW_MAJOR_GRID = tuple(
    tuple(range(start, start + 5)) for start in range(1, 31, 5)
)


def official_fixed_geometry_ledger() -> dict[str, Any]:
    """Load the bundled source-grounded immutable QUGS geometry ledger."""

    if not DEFAULT_FIXED_PANEL_LEDGER_PATH.is_file():
        raise FileNotFoundError(DEFAULT_FIXED_PANEL_LEDGER_PATH)
    return _load_fixed_panel_ledger(DEFAULT_FIXED_PANEL_LEDGER_PATH)


def fixed_panel_ledger_template() -> dict[str, Any]:
    """Backward-compatible alias returning the completed official ledger.

    v0.16.0 emitted an empty template.  v0.16.1 bundles the completed ledger and
    no longer asks users to invent or manually approve QUGS geometry.
    """

    return official_fixed_geometry_ledger()


def _load_fixed_panel_ledger(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalProtocolError("QUGS fixed-geometry ledger must be a JSON object")
    if payload.get("schema") != FIXED_PANEL_SCHEMA:
        raise FinalProtocolError(
            f"QUGS fixed-geometry ledger schema must be {FIXED_PANEL_SCHEMA!r}"
        )
    if payload.get("official_geometry_grounded") is not True:
        raise FinalProtocolError("QUGS ledger is not marked official-geometry-grounded")
    if payload.get("six_board_projection_status") != (
        "official_geometry_grounded_project_projection"
    ):
        raise FinalProtocolError("QUGS six-board projection status is not explicit")

    evidence = list(payload.get("evidence") or [])
    evidence_types = {
        str(item.get("evidence_type") or "")
        for item in evidence
        if isinstance(item, dict)
    }
    required_evidence = {
        "primary_benchmark_description",
        "official_benchmark_dataset_description",
        "peer_reviewed_numbered_geometry",
        "peer_reviewed_reduced_sensor_localization",
    }
    if not required_evidence.issubset(evidence_types):
        raise FinalProtocolError(
            "QUGS ledger lacks required benchmark/geometry evidence: "
            f"{sorted(required_evidence - evidence_types)}"
        )
    for item in evidence:
        if not isinstance(item, dict) or not str(item.get("source_url") or "").strip():
            raise FinalProtocolError("every QUGS evidence record needs a source URL")

    source_facts = dict(payload.get("source_facts") or {})
    expected_facts = {
        "fixed_sensor_count": 30,
        "sensor_identity": "physical QUGS joint number 1..30",
        "sensors_moved_between_tests": False,
        "damage_state_identity": "ADi/BDi means bolt loosening at physical joint i",
        "file_column_identity": "column 1 timestamp; columns 2..31 fixed Joint 1..30 signals",
    }
    for key, expected in expected_facts.items():
        if source_facts.get(key) != expected:
            raise FinalProtocolError(
                f"QUGS source fact {key!r} drifted: {source_facts.get(key)!r}"
            )

    layout = dict(payload.get("numbered_layout") or {})
    rows = tuple(tuple(int(value) for value in row) for row in layout.get("row_major_joint_grid") or [])
    if rows != OFFICIAL_ROW_MAJOR_GRID:
        raise FinalProtocolError(
            f"QUGS numbered layout must be {OFFICIAL_ROW_MAJOR_GRID}; observed {rows}"
        )
    positions_raw = dict(layout.get("normalized_ordinal_positions") or {})
    if set(positions_raw) != {str(joint) for joint in range(1, 31)}:
        raise FinalProtocolError("QUGS normalized positions must cover joints 1..30")
    positions: dict[int, tuple[float, float, float]] = {}
    for raw_joint, raw_position in positions_raw.items():
        values = tuple(float(value) for value in raw_position)
        if len(values) != 3:
            raise FinalProtocolError(f"QUGS joint {raw_joint} position must contain three values")
        positions[int(raw_joint)] = values

    reference = int(payload.get("reference_joint"))
    if reference != OFFICIAL_REFERENCE_JOINT:
        raise FinalProtocolError(
            f"QUGS fixed reference must be Joint {OFFICIAL_REFERENCE_JOINT}; observed {reference}"
        )
    reference_role = dict(payload.get("reference_role") or {})
    if reference_role.get("guaranteed_undamaged") is not False:
        raise FinalProtocolError("QUGS reference must not be claimed guaranteed undamaged")
    if reference_role.get("damage_at_reference_joint_retained") is not True:
        raise FinalProtocolError("QUGS Joint 30 damage case must remain explicitly retained")

    targets_raw = dict(payload.get("target_joints") or {})
    targets = {str(key): int(value) for key, value in targets_raw.items()}
    if targets != OFFICIAL_TARGET_JOINTS:
        raise FinalProtocolError(
            f"QUGS fixed targets must be {OFFICIAL_TARGET_JOINTS}; observed {targets}"
        )
    selected = [reference, *[targets[f"target_{index}"] for index in QUGS_TARGETS]]
    if len(set(selected)) != 6:
        raise FinalProtocolError("reference and five fixed target joints must be distinct")

    zones_raw = dict(payload.get("target_zone_joints") or {})
    zones = {
        str(key): tuple(int(value) for value in values)
        for key, values in zones_raw.items()
    }
    if zones != OFFICIAL_TARGET_ZONE_JOINTS:
        raise FinalProtocolError(
            f"QUGS target zones must follow published geometry columns: {OFFICIAL_TARGET_ZONE_JOINTS}"
        )

    mapping_raw = dict(payload.get("damage_joint_to_target") or {})
    if set(mapping_raw) != {str(joint) for joint in range(1, 31)}:
        raise FinalProtocolError("damage_joint_to_target must cover all joints 1..30")
    mapping: dict[int, int] = {}
    for raw_joint, raw_target in mapping_raw.items():
        joint = int(raw_joint)
        text = str(raw_target).strip().lower().removeprefix("target_")
        if text not in {"1", "2", "3", "4", "5"}:
            raise FinalProtocolError(
                f"damage joint {joint} has invalid target mapping {raw_target!r}"
            )
        mapping[joint] = int(text)
    if mapping != OFFICIAL_DAMAGE_TO_TARGET:
        raise FinalProtocolError(
            "QUGS damage-to-target mapping must follow the five published geometry columns"
        )
    counts = {
        target: sum(value == target for value in mapping.values())
        for target in QUGS_TARGETS
    }
    if counts != {target: 6 for target in QUGS_TARGETS}:
        raise FinalProtocolError(f"QUGS target-zone counts are not six each: {counts}")

    result = copy.deepcopy(payload)
    result["reference_joint"] = reference
    result["target_joints"] = targets
    result["target_zone_joints"] = zones
    result["damage_joint_to_target"] = mapping
    result["normalized_ordinal_positions"] = positions
    result["target_counts"] = counts
    result["ledger_path"] = str(source)
    return result


def _fixed_board_panel(row: Mapping[str, Any], ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    existing = list(row.get("boards") or [])
    if len(existing) != 6:
        raise FinalProtocolError("QUGS strict-index row must contain six board descriptors")
    selected = [
        int(ledger["reference_joint"]),
        *[
            int(ledger["target_joints"][f"target_{index}"])
            for index in QUGS_TARGETS
        ],
    ]
    positions = dict(ledger["normalized_ordinal_positions"])
    boards: list[dict[str, Any]] = []
    for position, (template, joint) in enumerate(zip(existing, selected, strict=True)):
        board = dict(template)
        board["sensor_id"] = "reference" if position == 0 else f"target_{position}"
        board["signal_path"] = "__matrix__"
        # QUGS matrices have time in column 0, so Joint i is channel index i.
        board["channel_indices"] = {"x": joint}
        board["axes_available"] = ["x"]
        board["source_sensor_name"] = f"joint_{joint:02d}"
        board["physical_sensor_id"] = f"qugs_joint_{joint:02d}"
        board["sensor_position"] = list(positions[joint])
        board["position_basis"] = "normalized_ordinal_from_published_numbered_layout"
        board["fixed_physical_panel"] = True
        boards.append(board)
    return boards


def patch_qugs_index_v016(
    input_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    mode: str = "official_fixed_panel",
    fixed_panel_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rewrite QUGS to one immutable official-geometry six-channel panel."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "fixed_panel":
        normalized_mode = "official_fixed_panel"
    if normalized_mode != "official_fixed_panel":
        raise FinalProtocolError(
            "v0.16.1 accepts only official_fixed_panel; the condition-dependent panel was removed"
        )
    ledger = _load_fixed_panel_ledger(
        fixed_panel_ledger_path or DEFAULT_FIXED_PANEL_LEDGER_PATH
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = _read_jsonl(input_path)

    parsed: list[tuple[dict[str, Any], int | None, str]] = []
    joint_to_target = dict(ledger["damage_joint_to_target"])
    campaign_support: dict[int, set[str]] = {joint: set() for joint in range(1, 31)}
    healthy_campaigns: set[str] = set()
    panel_signatures: set[tuple[int, ...]] = set()
    selected_panel = (
        int(ledger["reference_joint"]),
        *[
            int(ledger["target_joints"][f"target_{index}"])
            for index in QUGS_TARGETS
        ],
    )

    for original in rows:
        row = dict(original)
        joint = _extract_joint(row)
        campaign = _extract_campaign(row)
        if campaign not in {"A", "B"}:
            raise FinalProtocolError("QUGS row lacks an unambiguous A/B campaign")
        if joint is not None:
            campaign_support[joint].add(campaign)
        else:
            healthy_campaigns.add(campaign)
        boards = _fixed_board_panel(row, ledger)
        panel_signature = tuple(
            int(board["channel_indices"]["x"]) for board in boards
        )
        panel_signatures.add(panel_signature)
        row["boards"] = boards
        parsed.append((row, joint, campaign))

    if panel_signatures != {selected_panel}:
        raise FinalProtocolError(
            f"QUGS physical panel is not immutable: {sorted(panel_signatures)}"
        )
    missing_pairs = {
        joint: sorted(values)
        for joint, values in campaign_support.items()
        if values != {"A", "B"}
    }
    if missing_pairs:
        raise FinalProtocolError(f"QUGS A/B pairs incomplete: {missing_pairs}")
    if healthy_campaigns != {"A", "B"}:
        raise FinalProtocolError("QUGS intact A/B pair is incomplete")

    split_map = build_qugs_split_map(joint_to_target, seed=int(seed))
    patched: list[dict[str, Any]] = []
    for row, joint, campaign in parsed:
        common = {
            "localization_contract_version": "0.16.1.3",
            "fixed_physical_panel": True,
            "fixed_panel_joint_order": list(selected_panel),
            "requires_target_permutation_augmentation": False,
            "localization_claim_scope": (
                "official-geometry-grounded fixed QUGS target-zone localization; "
                "reference is a project-defined fixed context channel"
            ),
            "qatar_geometry_ledger": {
                "schema": ledger["schema"],
                "ledger_path": ledger["ledger_path"],
                "reference_joint": ledger["reference_joint"],
                "target_joints": ledger["target_joints"],
                "target_zone_joints": {
                    key: list(values)
                    for key, values in ledger["target_zone_joints"].items()
                },
                "six_board_projection_status": ledger["six_board_projection_status"],
            },
        }
        if joint is None:
            row.update(
                {
                    "session_id": "qugs_state_00",
                    "split_group": "qugs_intact_A_B_paired",
                    "split": "train",
                    "assigned_split": "train",
                    "healthy": True,
                    "class_name": "normal",
                    "global_class_name": "normal",
                    "severity": "none",
                    "affected_targets": [],
                    "target_supervision_available": True,
                    "supervision_scope": "target_localized",
                    "localization_provenance": {
                        "source": "official fixed QUGS channel identities and numbered geometry",
                        "campaign": campaign,
                        "fixed_panel": True,
                        "reference_is_guaranteed_healthy": False,
                        "exclusive_waveform_effect_claimed": False,
                        "evidence": ledger["evidence"],
                    },
                    **common,
                }
            )
        else:
            target = int(joint_to_target[joint])
            split = split_map[joint]
            row.update(
                {
                    "session_id": f"qugs_state_{joint:02d}",
                    "split_group": f"qugs_joint_{joint:02d}_A_B_paired",
                    "split": split,
                    "assigned_split": split,
                    "healthy": False,
                    "class_name": "unknown_anomaly",
                    "global_class_name": "unknown_anomaly",
                    "severity": "advisory",
                    "affected_targets": [target],
                    "target_supervision_available": True,
                    "supervision_scope": "target_localized",
                    "reference_joint_damaged": joint == int(ledger["reference_joint"]),
                    "localization_provenance": {
                        "source": (
                            "official fixed QUGS channel identity plus geometry-column zone ledger"
                        ),
                        "damaged_joint": joint,
                        "campaign": campaign,
                        "target_slot": target,
                        "target_zone_joints": list(
                            ledger["target_zone_joints"][f"target_{target}"]
                        ),
                        "fixed_panel": True,
                        "reference_is_guaranteed_healthy": False,
                        "reference_joint_damaged": joint == int(ledger["reference_joint"]),
                        "exclusive_waveform_effect_claimed": False,
                        "evidence": ledger["evidence"],
                    },
                    **common,
                }
            )
        patched.append(row)

    _write_jsonl(output_path, patched)
    counts = {
        split: {f"target_{target}": 0 for target in QUGS_TARGETS}
        for split in ("train", "validation", "test")
    }
    for joint, split in split_map.items():
        counts[split][f"target_{joint_to_target[joint]}"] += 1
    expected = {
        "train": {f"target_{target}": 4 for target in QUGS_TARGETS},
        "validation": {f"target_{target}": 1 for target in QUGS_TARGETS},
        "test": {f"target_{target}": 1 for target in QUGS_TARGETS},
    }
    if counts != expected:
        raise FinalProtocolError(f"QUGS fixed-panel split mismatch: {counts}")

    return {
        "schema": "vibrogemma-qugs-official-fixed-panel-index-v0.16.1.3",
        "mode": normalized_mode,
        "passed": True,
        "input": str(input_path),
        "output": str(output_path),
        "row_count": len(patched),
        "fixed_physical_panel": True,
        "panel_joint_order": list(selected_panel),
        "official_geometry_grounded": True,
        "six_board_projection_status": ledger["six_board_projection_status"],
        "reference_joint": ledger["reference_joint"],
        "reference_is_guaranteed_healthy": False,
        "reference_damage_condition_retained": True,
        "target_joints": ledger["target_joints"],
        "target_zone_joints": {
            key: list(values) for key, values in ledger["target_zone_joints"].items()
        },
        "damage_joint_to_target": {
            str(joint): f"target_{target}"
            for joint, target in sorted(joint_to_target.items())
        },
        "counts_by_split_and_target": counts,
        "target_permutation_augmentation_required": False,
        "test_materialized": False,
        "ledger_path": ledger["ledger_path"],
        "ledger_evidence": ledger["evidence"],
    }


__all__ = [
    "FinalProtocolError",
    "DEFAULT_FIXED_PANEL_LEDGER_PATH",
    "official_fixed_geometry_ledger",
    "fixed_panel_ledger_template",
    "patch_qugs_index_v016",
    "patch_lumo_index",
    "patch_upc_global_only",
]
