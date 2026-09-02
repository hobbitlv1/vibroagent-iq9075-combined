from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

from ..exceptions import DataContractError, ProvenanceError
from ..utils import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso
from .hbta_v4_verified import auto_index_verified_hbta
from .luh_reversible_geometry import (
    LUH_ACCELEROMETER_COUNT,
    LUH_ACCELEROMETER_POSITIONS_M,
    LUH_DAMAGE_MECHANISMS,
    LUH_EXCITATION_UPPER_HZ,
    LUH_FISHPLATE_TO_TARGET,
    LUH_GEOMETRY_BASIS,
    LUH_MEASURED_AXIS,
    LUH_POSITION_BASIS,
    LUH_REFERENCE_ACCELEROMETER,
    LUH_SAMPLE_RATE_HZ,
    LUH_STRUCTURE_ID,
    LUH_TARGET_ACCELEROMETERS,
    luh_panel_accelerometers,
    luh_target_for_fishplate,
    normalized_luh_sensor_position,
)
from .lumo_geometry import (
    LUMO_DAMAGE_ELEVATIONS_M,
    LUMO_DAMAGE_TO_TARGET,
    LUMO_POSITION_BASIS,
    LUMO_REFERENCE_MEASUREMENT_LEVEL,
    LUMO_SENSOR_ELEVATIONS_M,
    LUMO_TARGET_MEASUREMENT_LEVELS,
    lumo_zone_vertical_distance_m,
    normalized_lumo_sensor_position,
)
from .manifest import DatasetManifest, load_dataset_manifests

_AUTO_INDEX_SCHEMA_VERSION = "vibrogemma.strict-index.v2"
_SUPPORTED_ADAPTERS = {
    "lumo_mat_v1": "lumo",
    "luh_reversible_beam_mat_v1": "luh_reversible_beam",
    "hell_bridge_v4_opyndata": "hell_bridge",
    "hell_bridge_v4_verified_tests_v1": "hell_bridge_verified_legacy_alias",
    "hell_bridge_v4_verified_windows_v2": "hell_bridge_verified",
    "z24_pdt_mat_v1": "z24",
    "rwth_frame_v1_csv": "rwth_frame",
    "qugs_txt_v1": "qugs",
    "designsafe_prj2447_csv_v1": "designsafe_prj2447",
}
_BOARD_IDS = ["reference", "target_1", "target_2", "target_3", "target_4", "target_5"]
_AXIS_ORDER = ("x", "y", "z")


def _normalise_token(value: Any) -> str:
    text = str(value).strip().lower().replace("²", "2")
    return re.sub(r"[^a-z0-9]+", "", text)


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
        return [_decode_scalar(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode_scalar(item) for item in value]
    return value


def _string_value(value: Any) -> str:
    value = _decode_scalar(value)
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _canonical_unit(value: Any) -> str | None:
    token = _normalise_token(_string_value(value))
    if token in {"g", "gravity", "gravities"}:
        return "g"
    if token == "mg":
        return "mg"
    if token in {
        "ms2",
        "mps2",
        "meterpersecondsquared",
        "meterspersecondsquared",
        "metrepersecondsquared",
        "metrespersecondsquared",
    }:
        return "m/s^2"
    if token in {"mms2", "mmps2", "millimeterpersecondsquared", "millimetrepersecondsquared"}:
        return "mm/s^2"
    if token in {"releasenative", "native", "uncalibrated", "arbitrary", "arbitraryunits", "au"}:
        return "release_native"
    return None


def _attr_value(objects: Iterable[h5py.AttributeManager], keys: Iterable[str]) -> Any | None:
    wanted = {_normalise_token(key) for key in keys}
    for attrs in objects:
        for key, value in attrs.items():
            if _normalise_token(key) in wanted:
                return _decode_scalar(value)
    return None


def _numeric_vector(value: Any) -> np.ndarray | None:
    """Return a finite numeric vector, or ``None`` for descriptive metadata.

    Publisher HDF5 files sometimes use attributes named ``position`` for a
    human-readable note (for example, ``"According to the MVS position"``)
    instead of Cartesian coordinates.  Such metadata is optional and must not
    abort indexing.  Numeric arrays are accepted directly.  Numeric strings
    are accepted only when the *entire* string consists of numbers and common
    delimiters, preventing accidental extraction of numbers from prose.
    """

    decoded = _decode_scalar(value)
    if isinstance(decoded, str):
        stripped = decoded.strip()
        if not stripped:
            return None
        # Reject descriptive text rather than mining incidental numbers from it.
        if re.search(r"[A-DF-Za-df-z]", stripped):
            return None
        if re.fullmatch(r"[\s,;:\[\]\(\){}+\-0-9.eE]+", stripped) is None:
            return None
        cleaned = re.sub(r"[\[\]\(\){};,]+", " ", stripped)
        try:
            array = np.fromstring(cleaned, sep=" ", dtype=np.float64)
        except ValueError:
            return None
    else:
        try:
            array = np.asarray(decoded, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return array.reshape(-1)


def _position_from_attrs(*objects: h5py.Group | h5py.Dataset) -> list[float] | None:
    attrs = [obj.attrs for obj in objects]
    value = _attr_value(attrs, ("sensor_position", "position", "coordinates", "coordinate", "xyz"))
    if value is not None:
        array = _numeric_vector(value)
        if array is not None and array.size >= 3:
            return [float(item) for item in array[:3]]
    values: list[float] = []
    for axis in _AXIS_ORDER:
        axis_value = _attr_value(attrs, (axis, f"position_{axis}", f"coordinate_{axis}"))
        if axis_value is None:
            return None
        array = _numeric_vector(axis_value)
        if array is None or array.size == 0:
            return None
        values.append(float(array[0]))
    return values if np.isfinite(values).all() else None


def _orientation_from_attrs(*objects: h5py.Group | h5py.Dataset) -> list[list[float]] | None:
    attrs = [obj.attrs for obj in objects]
    value = _attr_value(attrs, ("orientation_matrix", "rotation_matrix", "sensor_to_global"))
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(matrix).all():
        return None
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        return None
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        return None
    return matrix.astype(float).tolist()


def _axis_from_component(name: str, dataset: h5py.Dataset) -> str | None:
    candidates = [name]
    for key in ("axis", "direction", "component", "component_name", "orientation"):
        if key in dataset.attrs:
            candidates.append(_string_value(dataset.attrs[key]))
    aliases = {
        "x": {"x", "ax", "accx", "accelerationx", "longitudinal", "longitudinalx"},
        "y": {"y", "ay", "accy", "accelerationy", "lateral", "transverse", "transversaly"},
        "z": {"z", "az", "accz", "accelerationz", "vertical", "verticalz"},
    }
    for candidate in candidates:
        token = _normalise_token(candidate)
        for axis, names in aliases.items():
            if token in names:
                return axis
        for axis in _AXIS_ORDER:
            if token.endswith(axis) and any(prefix in token for prefix in ("acc", "axis", "direction")):
                return axis
    return None


def _relative_source(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ProvenanceError(f"source {path} is outside dataset root {root}") from exc


def _receipt(root: Path) -> dict[str, Any]:
    path = root / "download_receipt.json"
    if not path.exists():
        raise ProvenanceError(f"download receipt is required before automatic indexing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProvenanceError(f"invalid download receipt: {path}")
    return value


def _receipt_sha256(receipt: dict[str, Any], filename: str) -> str | None:
    download = receipt.get("download") or {}
    for item in download.get("files") or []:
        if Path(str(item.get("name", ""))).name == Path(filename).name:
            value = item.get("sha256")
            return str(value) if value else None
    return None


def _write_index_and_report(
    manifest: DatasetManifest,
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if not entries:
        raise DataContractError(f"{manifest.id}: automatic index generation produced no entries")
    destination = manifest.index_path(config)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"strict index already exists: {destination}; pass --overwrite to regenerate")

    # Import lazily to avoid a module cycle at import time.
    from .episodes import PublicEpisodeBuilder

    for entry in entries:
        PublicEpisodeBuilder._validate_index_entry(entry, manifest)
        source = manifest.raw_root(config) / str(entry["source_file"])
        if not source.exists():
            raise DataContractError(f"generated source path does not exist: {source}")

    lines = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    atomic_write_text(destination, lines)
    report_path = destination.parent / "auto_index_report.json"
    report = {
        **report,
        "schema_version": _AUTO_INDEX_SCHEMA_VERSION,
        "generated_utc": utc_now_iso(),
        "dataset_id": manifest.id,
        "adapter": manifest.adapter,
        "index_path": str(destination),
        "index_entries": len(entries),
        "index_sha256": sha256_file(destination),
    }
    atomic_write_json(report_path, report)
    return {
        "dataset_id": manifest.id,
        "index": str(destination),
        "entries": len(entries),
        "sha256": report["index_sha256"],
        "report": str(report_path),
        "rejected_records": len(report.get("rejected_records") or []),
    }


def auto_index_dataset(
    config: dict[str, Any],
    dataset_id: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    matches = [manifest for manifest in load_dataset_manifests(config) if manifest.id == dataset_id]
    if len(matches) != 1:
        raise DataContractError(f"unknown or duplicate selected dataset ID: {dataset_id}")
    manifest = matches[0]
    if manifest.adapter not in _SUPPORTED_ADAPTERS:
        raise DataContractError(
            f"{manifest.id}: adapter {manifest.adapter!r} has no automatic strict-index generator; "
            f"supported adapters are {sorted(_SUPPORTED_ADAPTERS)}"
        )
    dispatch = {
        "lumo_mat_v1": _auto_index_lumo,
        "luh_reversible_beam_mat_v1": _auto_index_luh_reversible_beam,
        "hell_bridge_v4_opyndata": _auto_index_hell_bridge,
        "hell_bridge_v4_verified_tests_v1": auto_index_verified_hbta,
        "hell_bridge_v4_verified_windows_v2": auto_index_verified_hbta,
        "z24_pdt_mat_v1": _auto_index_z24,
        "rwth_frame_v1_csv": _auto_index_rwth,
        "qugs_txt_v1": _auto_index_qugs,
        "designsafe_prj2447_csv_v1": _auto_index_designsafe,
    }
    entries, report = dispatch[manifest.adapter](manifest, config)
    return _write_index_and_report(manifest, config, entries, report, overwrite=overwrite)


def auto_index_all(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    for manifest in load_dataset_manifests(config):
        if not manifest.training_allowed:
            continue
        if manifest.adapter not in _SUPPORTED_ADAPTERS:
            unsupported.append({"dataset_id": manifest.id, "adapter": manifest.adapter})
            continue
        results.append(auto_index_dataset(config, manifest.id, overwrite=overwrite))
    if not results:
        raise DataContractError("none of the selected training datasets has an automatic index adapter")
    return {"generated": results, "unsupported": unsupported}



# ---------------------------------------------------------------------------
# EDA-locked dataset adapters
# ---------------------------------------------------------------------------


def _mat_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        if name not in value:
            raise DataContractError(f"MAT struct has no field {name!r}")
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, np.ndarray) and value.size == 1:
        return _mat_field(value.reshape(-1)[0], name)
    raise DataContractError(f"MAT struct has no field {name!r}")


def _mat_strings(value: Any) -> list[str]:
    array = np.asarray(value, dtype=object).reshape(-1)
    result: list[str] = []
    for item in array:
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.reshape(-1)[0]
        if isinstance(item, bytes):
            result.append(item.decode("utf-8", errors="replace").strip())
        else:
            result.append(str(item).strip())
    return result


def _spread(items: list[Any], count: int = 6) -> list[Any]:
    if len(items) < count:
        raise DataContractError(f"need at least {count} genuine sensor locations; found {len(items)}")
    indices = [int(round(index)) for index in np.linspace(0, len(items) - 1, num=count)]
    selected = [items[index] for index in indices]
    if len({str(item) for item in selected}) != count:
        raise DataContractError("deterministic spatial spreading did not produce distinct sensors")
    return selected


def _source_report(receipt: dict[str, Any]) -> dict[str, Any]:
    download = receipt.get("download") or {}
    return {
        "source_type": download.get("source"),
        "source_record": download.get("record_id") or download.get("package_id"),
        "source_doi": download.get("doi"),
    }


# ---------------------------------------------------------------------------
# LUMO lattice mast MATLAB-v5 release
# ---------------------------------------------------------------------------


def _lumo_session(path: Path, root: Path) -> str:
    for parent in [path.parent, *path.parents]:
        if parent == root.parent:
            break
        match = re.fullmatch(r"(\d{2})_(Healthy|DAM\d+_[01]{3})", parent.name, flags=re.IGNORECASE)
        if match:
            return f"{int(match.group(1)):02d}_{match.group(2)}"
    raise DataContractError(f"{path}: no EDA-verified LUMO session folder like 01_Healthy or 02_DAM6_111")


def _lumo_schema(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Dat" not in payload:
        raise DataContractError(f"{path}: missing Dat struct")
    dat = payload["Dat"]
    names = _mat_strings(_mat_field(dat, "ChannelNames"))
    units = _mat_strings(_mat_field(dat, "ChannelUnits"))
    data = np.asarray(_mat_field(dat, "Data"))
    fs = float(np.asarray(_mat_field(dat, "Fs")).reshape(-1)[0])
    if data.ndim != 2:
        raise DataContractError(f"{path}: Dat.Data must be 2D, got {data.shape}")
    if data.shape[1] != len(names) and data.shape[0] == len(names):
        data = data.T
    if data.shape[1] != len(names) or len(units) != len(names):
        raise DataContractError(
            f"{path}: Dat.Data/channels mismatch: data={data.shape}, names={len(names)}, units={len(units)}"
        )
    if not math.isclose(fs, 1651.612903, rel_tol=1e-5, abs_tol=1e-3):
        raise DataContractError(f"{path}: unexpected LUMO sample rate {fs}")

    sensors: dict[str, dict[str, int]] = {}
    for index, (name, unit) in enumerate(zip(names, units, strict=True)):
        match = re.fullmatch(r"accel(\d{2})([xy])", name.strip(), flags=re.IGNORECASE)
        if match is None:
            continue
        if _canonical_unit(unit) != "g":
            raise DataContractError(f"{path}: acceleration channel {name} has unexpected unit {unit!r}")
        sensor = f"accel{int(match.group(1)):02d}"
        sensors.setdefault(sensor, {})[match.group(2).lower()] = index
    expected = [f"accel{index:02d}" for index in range(1, 10)]
    missing = [sensor for sensor in expected if set(sensors.get(sensor, {})) != {"x", "y"}]
    if missing:
        raise DataContractError(f"{path}: missing EDA-verified x/y channels for {missing}")
    return {
        "sample_rate_hz": fs,
        "sample_count": int(data.shape[0]),
        "channel_count": int(data.shape[1]),
        "sensors": sensors,
    }


def _auto_index_lumo(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    paths = sorted(path for path in root.rglob("*.mat") if path.is_file())
    if not paths:
        raise DataContractError(f"{manifest.id}: no MATLAB files found under {root}")
    # Preserve the historical generic panel unless the active manifest explicitly
    # freezes a structural-zone localisation panel.  Explicit panels are checked
    # against the canonical publisher-derived geometry contract so a manifest
    # cannot silently pair a damage label with the wrong measurement waveform.
    localisation = dict(manifest.localisation or {})
    explicit_localisation = (
        str(localisation.get("mode", "")).strip().lower() == "explicit_structural_zone"
    )

    def _sensor_from_level(value: Any) -> str:
        match = re.fullmatch(r"ML([1-9])", str(value).strip(), flags=re.IGNORECASE)
        if match is None:
            raise DataContractError(
                f"{manifest.id}: invalid LUMO measurement level {value!r}; expected ML1..ML9"
            )
        return f"accel{int(match.group(1)):02d}"

    manifest_damage_to_target: dict[str, int] = {}
    if explicit_localisation:
        reference_level = str(localisation.get("reference_measurement_level", "")).strip().upper()
        target_levels = {
            str(key).strip().lower(): str(value).strip().upper()
            for key, value in dict(
                localisation.get("target_measurement_levels") or {}
            ).items()
        }
        if reference_level != LUMO_REFERENCE_MEASUREMENT_LEVEL:
            raise DataContractError(
                f"{manifest.id}: explicit LUMO panel requires geometry-aligned reference "
                f"{LUMO_REFERENCE_MEASUREMENT_LEVEL}; "
                f"observed {reference_level!r}"
            )
        expected_targets = [f"target_{index}" for index in range(1, 6)]
        if sorted(target_levels) != expected_targets:
            raise DataContractError(
                f"{manifest.id}: explicit LUMO panel must define {expected_targets}; "
                f"observed {sorted(target_levels)}"
            )
        if target_levels != LUMO_TARGET_MEASUREMENT_LEVELS:
            raise DataContractError(
                f"{manifest.id}: explicit LUMO target panel must be "
                f"{LUMO_TARGET_MEASUREMENT_LEVELS}; observed {target_levels}"
            )
        raw_damage_mapping = {
            str(key).strip().upper(): str(value).strip().lower()
            for key, value in dict(localisation.get("damage_zone_to_target") or {}).items()
        }
        expected_damage_mapping = {
            damage: f"target_{target}"
            for damage, target in LUMO_DAMAGE_TO_TARGET.items()
        }
        if raw_damage_mapping != expected_damage_mapping:
            raise DataContractError(
                f"{manifest.id}: explicit LUMO damage-zone mapping must be "
                f"{expected_damage_mapping}; observed {raw_damage_mapping}"
            )
        manifest_damage_to_target = {
            damage: int(target.rsplit("_", 1)[-1])
            for damage, target in raw_damage_mapping.items()
        }
        selected_levels = [
            reference_level,
            *[target_levels[target] for target in expected_targets],
        ]
        if len(set(selected_levels)) != 6:
            raise DataContractError("LUMO reference and five target levels must be distinct")
        selected_names = [_sensor_from_level(level) for level in selected_levels]
        sensor_selection_note = (
            "manifest-frozen geometry-aligned panel: reference accel04/ML4; "
            "targets accel02/ML2, accel05/ML5, accel06/ML6, accel08/ML8, "
            "accel09/ML9; genuine x/y only"
        )
    else:
        selected_names = _spread([f"accel{index:02d}" for index in range(1, 10)], count=6)
        selected_levels = [f"ML{int(sensor[-2:])}" for sensor in selected_names]
        sensor_selection_note = (
            "historical deterministic spatial-spread panel across accel01..accel09; "
            "genuine x/y only"
        )
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    sessions: dict[str, int] = {}
    for path in paths:
        try:
            session = _lumo_session(path, root)
            schema = _lumo_schema(path)
            healthy = session.lower().endswith("_healthy")
            condition = session.split("_", 1)[1]
            damage_level: str | None = None
            if not healthy:
                match = re.fullmatch(r"(DAM[1-6])_[01]{3}", condition.upper())
                if match is None:
                    raise DataContractError(
                        f"{manifest.id}: changed session has invalid damage condition {condition!r}"
                    )
                damage_level = match.group(1)
            affected_targets = (
                []
                if healthy or not explicit_localisation
                else [manifest_damage_to_target[damage_level]]
            )
            relative = _relative_source(root, path)
            boards = []
            for board_id, sensor_name, measurement_level in zip(
                _BOARD_IDS, selected_names, selected_levels, strict=True
            ):
                channels = schema["sensors"][sensor_name]
                boards.append(
                    {
                        "sensor_id": board_id,
                        "signal_path": "Dat/Data",
                        "channel_indices": {"x": channels["x"], "y": channels["y"]},
                        "axes_available": ["x", "y"],
                        "sensor_position": list(
                            normalized_lumo_sensor_position(measurement_level)
                        ),
                        "orientation_matrix": None,
                        "source_sensor_name": sensor_name,
                        "source_measurement_level": measurement_level,
                        "sensor_elevation_m": LUMO_SENSOR_ELEVATIONS_M[measurement_level],
                        "position_basis": LUMO_POSITION_BASIS,
                    }
                )
            entry = {
                "dataset_id": manifest.id,
                "source_file": relative,
                "source_sha256": sha256_file(path),
                "structure_id": "lumo_lattice_mast",
                "session_id": session,
                "experiment_id": f"{session}::{path.stem}",
                "sample_rate_hz": float(schema["sample_rate_hz"]),
                "unit": "g",
                "sample_axis": 0,
                "start_time_s": 0.0,
                "condition": condition,
                "training_role": "supervised",
                "healthy": healthy,
                "severity": "none" if healthy else "advisory",
                "class_name": "normal" if healthy else "unknown_anomaly",
                "global_class_name": "normal" if healthy else "unknown_anomaly",
                "affected_targets": affected_targets,
                "target_supervision_available": explicit_localisation,
                "supervision_scope": (
                    "target_localized" if explicit_localisation else "global_only"
                ),
                "phase_synchronized": True,
                "synchronization_verification_id": "lumo_shared_dat_matrix_v1",
                "sensor_useful_band_hz": float(schema["sample_rate_hz"]) / 2.0,
                "boards": boards,
                "auto_index": {
                    "adapter": manifest.adapter,
                    "schema_basis": "user-supplied executed EDA: Dat.ChannelNames/ChannelUnits/Data/Fs",
                    "label_basis": (
                        "EDA-verified session folder; Healthy maps to normal and DAM codes remain verbatim, "
                        "conservatively mapped to unknown_anomaly/advisory without inferred mechanism or severity"
                    ),
                    "sensor_selection": sensor_selection_note,
                },
            }
            if explicit_localisation:
                target_index = affected_targets[0] if affected_targets else None
                target_name = f"target_{target_index}" if target_index is not None else None
                entry["localization_provenance"] = {
                    "source": "publisher ML/DAM elevations plus manifest-locked target zones",
                    "damage_level": damage_level,
                    "damage_elevation_m": (
                        LUMO_DAMAGE_ELEVATIONS_M[damage_level]
                        if damage_level is not None
                        else None
                    ),
                    "target_slot": target_index,
                    "target_measurement_level": (
                        target_levels[target_name] if target_name is not None else None
                    ),
                    "target_sensor_elevation_m": (
                        LUMO_SENSOR_ELEVATIONS_M[target_levels[target_name]]
                        if target_name is not None
                        else None
                    ),
                    "vertical_distance_m": (
                        lumo_zone_vertical_distance_m(target_name, damage_level)
                        if target_name is not None and damage_level is not None
                        else None
                    ),
                    "exclusive_waveform_effect_claimed": False,
                }
            entries.append(entry)
            sessions[session] = sessions.get(session, 0) + 1
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not entries:
        raise DataContractError(f"{manifest.id}: every MAT file failed the EDA-locked schema: {rejected[:5]}")
    if not any(entry["healthy"] for entry in entries) or not any(not entry["healthy"] for entry in entries):
        raise DataContractError("LUMO automatic index must retain both Healthy and DAM sessions")
    return entries, {
        **_source_report(receipt),
        "selected_source_sensors": selected_names,
        "selected_measurement_levels": selected_levels,
        "session_file_counts": sessions,
        "policy": {
            "axes": "genuine x/y only; z is unavailable and remains masked",
            "labels": (
                "folder labels plus manifest-locked geometry-aligned source-zone targets"
                if explicit_localisation
                else "folder labels exactly as documented by the supplied EDA"
            ),
            "positions": LUMO_POSITION_BASIS,
            "severity": "DAM codes do not imply engineering severity; mapped only to advisory",
            "synchronization": "all selected channels share one Dat.Data matrix and sample clock",
        },
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# LUH reversible-damage laboratory cantilever MATLAB-v5 release
# ---------------------------------------------------------------------------


def _luh_mechanism_from_path(path: Path) -> str:
    for value in (path.stem, *reversed(path.parent.parts)):
        token = _normalise_token(value)
        for mechanism in LUH_DAMAGE_MECHANISMS:
            if mechanism in token:
                return mechanism
    raise DataContractError(f"{path}: cannot resolve LUH damage mechanism")


def _luh_filename_state(path: Path) -> tuple[str, bool, int | None, str | None]:
    """Decode one publisher filename without inferring state from waveforms."""

    stem = path.stem.strip().lower()
    mechanism = _luh_mechanism_from_path(path)
    healthy = "repaired" in stem
    repetition_match = re.search(r"_1200hz_(\d+)$", stem)
    repetition = repetition_match.group(1) if repetition_match else None
    if healthy:
        return mechanism, True, None, repetition

    match = re.search(r"(?:^|_)(discrete|gaussian|uniformly)(?:_|$)", stem)
    if match is None or match.group(1) != mechanism:
        raise DataContractError(
            f"{path.name}: damaged LUH filename lacks its mechanism token"
        )
    clean = re.sub(r"_1200hz(?:_\d+)?$", "", stem)
    clean = clean.removeprefix("noise_")
    clean = re.sub(rf"(?:^|_){re.escape(mechanism)}(?:_|$)", "_", clean)
    digits = [int(value) for value in re.findall(r"[1-9]", clean)]
    if (
        digits != sorted(digits)
        or len(digits) != len(set(digits))
        or not set(digits).issubset(set(range(1, 10)))
    ):
        raise DataContractError(f"{path.name}: invalid LUH fishplate filename digits {digits}")
    missing = sorted(set(range(1, 10)) - set(digits))
    if len(missing) != 1:
        raise DataContractError(
            f"{path.name}: damaged LUH filename must omit exactly one fishplate; missing={missing}"
        )
    return mechanism, False, missing[0], repetition


def _luh_accelerometer_number(name: str) -> int | None:
    match = re.fullmatch(r"sensor_?0?([1-9]|1[0-5])", str(name).strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match is not None else None


def _luh_schema(path: Path) -> dict[str, Any]:
    variables = {name: (shape, dtype) for name, shape, dtype in whosmat(path)}
    if "Dat" not in variables:
        raise DataContractError(f"{path}: missing LUH MATLAB Dat struct")
    payload = loadmat(path, variable_names=["Dat"], squeeze_me=True, struct_as_record=False)
    dat = payload["Dat"]
    names = _mat_strings(_mat_field(dat, "ChannelNames"))
    units = _mat_strings(_mat_field(dat, "ChannelUnits"))
    data = np.asarray(_mat_field(dat, "Data"))
    data_shape = tuple(int(value) for value in data.shape)
    if len(data_shape) != 2:
        raise DataContractError(f"{path}: Dat.Data must be two-dimensional, got {data_shape}")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise DataContractError(f"{path}: Dat.Data must contain only finite numeric samples")
    sample_rate_hz = float(np.asarray(_mat_field(dat, "Fs")).reshape(-1)[0])
    if not math.isfinite(sample_rate_hz) or not math.isclose(
        sample_rate_hz, LUH_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=0.01
    ):
        raise DataContractError(f"{path}: unexpected LUH sample rate {sample_rate_hz}")
    acquisition_time = _string_value(_mat_field(dat, "Time")).strip()
    if not acquisition_time:
        raise DataContractError(f"{path}: Dat.Time is empty")
    expected_names = [f"Sensor_{index}" for index in range(1, 17)]
    if names != expected_names:
        raise DataContractError(
            f"{path}: LUH channel order must be Sensor_1..Sensor_16; observed={names}"
        )
    if len(names) != len(units) or any(_canonical_unit(unit) != "g" for unit in units):
        raise DataContractError(
            f"{path}: LUH requires sixteen g-valued channels; names={len(names)} units={units}"
        )
    if data_shape[1] == len(names):
        sample_axis = 0
        sample_count = data_shape[0]
    elif data_shape[0] == len(names):
        sample_axis = 1
        sample_count = data_shape[1]
    else:
        raise DataContractError(
            f"{path}: Data shape {data_shape} does not match {len(names)} channel names"
        )

    sensors: dict[int, int] = {}
    for channel_index, (name, _unit) in enumerate(zip(names, units, strict=True)):
        sensor = _luh_accelerometer_number(name)
        if sensor is None:
            continue
        if sensor in sensors:
            raise DataContractError(f"{path}: duplicate LUH accelerometer Acc{sensor}")
        sensors[sensor] = channel_index
    expected = set(range(1, LUH_ACCELEROMETER_COUNT + 1))
    if set(sensors) != expected:
        raise DataContractError(
            f"{path}: LUH accelerometer identities must be Acc1..Acc15; "
            f"observed={sorted(sensors)} names={names}"
        )
    return {
        "sample_rate_hz": sample_rate_hz,
        "sample_count": int(sample_count),
        "sample_axis": sample_axis,
        "data_shape": data_shape,
        "channel_count": len(names),
        "sensors": sensors,
        "channel_names": names,
        "channel_units": units,
        "acquisition_time": acquisition_time,
    }


def _validate_luh_manifest_contract(manifest: DatasetManifest) -> None:
    localisation = dict(manifest.localisation or {})
    if str(localisation.get("mode", "")).strip().lower() != "explicit_structural_zone":
        raise DataContractError("LUH requires explicit_structural_zone localisation")
    if str(localisation.get("measured_axis", "")).strip().lower() != LUH_MEASURED_AXIS:
        raise DataContractError(f"LUH measured axis must remain {LUH_MEASURED_AXIS!r}")
    if str(localisation.get("fixed_reference_accelerometer", "")).strip().lower() != (
        f"acc{LUH_REFERENCE_ACCELEROMETER}"
    ).lower():
        raise DataContractError(
            f"LUH fixed context reference must remain Acc{LUH_REFERENCE_ACCELEROMETER}"
        )
    targets = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in dict(
            localisation.get("fixed_target_accelerometers") or {}
        ).items()
    }
    expected_targets = {
        key: f"acc{value}".lower() for key, value in LUH_TARGET_ACCELEROMETERS.items()
    }
    if targets != expected_targets:
        raise DataContractError(
            f"LUH fixed target panel must remain {expected_targets}; observed={targets}"
        )
    mapping = {
        str(key).strip().upper(): str(value).strip().lower()
        for key, value in dict(localisation.get("fishplate_to_target") or {}).items()
    }
    expected_mapping = {
        f"F{fishplate}": f"target_{target}"
        for fishplate, target in LUH_FISHPLATE_TO_TARGET.items()
    }
    if mapping != expected_mapping:
        raise DataContractError(
            f"LUH fishplate-zone mapping must remain {expected_mapping}; observed={mapping}"
        )
    explicit_split = str((manifest.auto_index or {}).get("explicit_split", "")).strip().lower()
    if explicit_split != "train":
        raise DataContractError(
            "the first LUH integration is a train-only supplement; auto_index.explicit_split must be train"
        )
    stride_seconds = float((manifest.auto_index or {}).get("window_stride_seconds", 0.0))
    if stride_seconds < 10.0:
        raise DataContractError(
            "LUH auto_index.window_stride_seconds must be at least one 10-second window"
        )


def _auto_index_luh_reversible_beam(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    _validate_luh_manifest_contract(manifest)
    paths = sorted(path for path in root.rglob("*.mat") if path.is_file())
    if not paths:
        raise DataContractError(f"{manifest.id}: no MATLAB files found under {root}")

    panel = luh_panel_accelerometers()
    stride_seconds = float(manifest.auto_index["window_stride_seconds"])
    known_unreadable = {
        str(item["path"]): dict(item)
        for item in (manifest.auto_index or {}).get("known_unreadable_files") or []
    }
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    counts = {
        mechanism: {"healthy": 0, "damaged": 0, "total": 0}
        for mechanism in LUH_DAMAGE_MECHANISMS
    }
    damaged_positions = {mechanism: [] for mechanism in LUH_DAMAGE_MECHANISMS}
    for path in paths:
        try:
            relative = _relative_source(root, path)
            exclusion = known_unreadable.get(relative)
            if exclusion is not None:
                if path.stat().st_size != int(exclusion["size"]):
                    raise DataContractError(
                        f"known-unreadable LUH file size drifted: {relative}"
                    )
                observed_sha256 = sha256_file(path)
                if observed_sha256.lower() != str(exclusion["sha256"]).lower():
                    raise DataContractError(
                        f"known-unreadable LUH file checksum drifted: {relative}"
                    )
                rejected.append(
                    {
                        "source_file": relative,
                        "reason": str(exclusion["reason"]),
                    }
                )
                continue
            mechanism, healthy, fishplate, repetition = _luh_filename_state(path)
            schema = _luh_schema(path)
            target_index = None if fishplate is None else luh_target_for_fishplate(fishplate)
            boards = []
            for board_id, sensor_index in zip(_BOARD_IDS, panel, strict=True):
                boards.append(
                    {
                        "sensor_id": board_id,
                        "signal_path": "Dat/Data",
                        "channel_indices": {
                            LUH_MEASURED_AXIS: schema["sensors"][sensor_index]
                        },
                        "axes_available": [LUH_MEASURED_AXIS],
                        "sensor_position": list(
                            normalized_luh_sensor_position(sensor_index)
                        ),
                        "sensor_position_m": list(
                            LUH_ACCELEROMETER_POSITIONS_M[sensor_index]
                        ),
                        "orientation_matrix": None,
                        "source_sensor_name": f"Sensor_{sensor_index}",
                        "publisher_accelerometer_label": f"Acc{sensor_index}",
                        "physical_sensor_id": f"luh_accelerometer_{sensor_index:02d}",
                        "position_basis": LUH_POSITION_BASIS,
                        "physical_position_basis": LUH_GEOMETRY_BASIS,
                        "fixed_physical_panel": True,
                    }
                )
            state_name = (
                f"healthy_repaired_{mechanism}"
                if healthy
                else f"{mechanism}_damage_f{fishplate}"
            )
            session_token = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
            entry = {
                "dataset_id": manifest.id,
                "source_file": relative,
                "source_sha256": sha256_file(path),
                "structure_id": LUH_STRUCTURE_ID,
                "session_id": f"luh_{mechanism}_{session_token}",
                "experiment_id": f"{mechanism}::{path.stem}",
                "sample_rate_hz": float(schema["sample_rate_hz"]),
                "unit": "g",
                "sample_axis": int(schema["sample_axis"]),
                "start_time_s": 0.0,
                "window_stride_seconds": stride_seconds,
                "source_acquisition_time": schema["acquisition_time"],
                "condition": state_name,
                "training_role": "supervised",
                "healthy": healthy,
                "severity": "none" if healthy else "advisory",
                "class_name": "normal" if healthy else "unknown_anomaly",
                "global_class_name": "normal" if healthy else "unknown_anomaly",
                "affected_targets": [] if target_index is None else [target_index],
                "target_supervision_available": True,
                "supervision_scope": "target_localized",
                "split": "train",
                "split_stratum": (
                    "healthy" if healthy else f"target_{target_index}"
                ),
                "phase_synchronized": True,
                "synchronization_verification_id": "luh_shared_data_matrix_v1",
                "sensor_useful_band_hz": LUH_EXCITATION_UPPER_HZ,
                "boards": boards,
                "auto_index": {
                    "adapter": manifest.adapter,
                    "schema_basis": (
                        "publisher Dat.Data/Dat.Fs/Dat.ChannelNames/Dat.ChannelUnits"
                    ),
                    "label_basis": (
                        "publisher filename: repaired is healthy; the mechanism token occupies "
                        "the independently damaged F1..F9 position"
                    ),
                    "sensor_selection": (
                        "fixed Acc15 context plus Acc2/Acc5/Acc8/Acc11/Acc14 targets"
                    ),
                    "evaluation_role": "train_only_localization_supplement",
                    "excluded_channel": (
                        "Sensor_16: publisher-undocumented auxiliary acquisition channel"
                    ),
                },
                "localization_provenance": {
                    "source": (
                        "publisher fishplate identity plus fixed contiguous project zones"
                    ),
                    "damage_mechanism": None if healthy else mechanism,
                    "damage_fishplate": fishplate,
                    "target_slot": target_index,
                    "filename_repetition": repetition,
                    "exclusive_waveform_effect_claimed": False,
                    "reference_unaffected_claimed": False,
                },
            }
            entries.append(entry)
            counts[mechanism]["healthy" if healthy else "damaged"] += 1
            counts[mechanism]["total"] += 1
            if fishplate is not None:
                damaged_positions[mechanism].append(fishplate)
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not entries:
        raise DataContractError(
            f"{manifest.id}: every LUH MAT file failed strict indexing: {rejected[:5]}"
        )
    if bool((manifest.auto_index or {}).get("require_official_inventory", False)):
        expected_counts = {
            "discrete": {"healthy": 10, "damaged": 8, "total": 18},
            "gaussian": {"healthy": 9, "damaged": 9, "total": 18},
            "uniformly": {"healthy": 8, "damaged": 9, "total": 17},
        }
        expected_positions = {
            "discrete": [1, 2, 3, 4, 5, 6, 7, 9],
            "gaussian": list(range(1, 10)),
            "uniformly": list(range(1, 10)),
        }
        position_coverage_ok = all(
            sorted(damaged_positions[mechanism]) == expected_positions[mechanism]
            for mechanism in LUH_DAMAGE_MECHANISMS
        )
        rejected_paths = {str(item["source_file"]) for item in rejected}
        if (
            counts != expected_counts
            or len(entries) != 53
            or rejected_paths != set(known_unreadable)
            or not position_coverage_ok
        ):
            raise DataContractError(
                "LUH official inventory mismatch: "
                f"counts={counts}, damaged_positions={damaged_positions}, "
                f"entries={len(entries)}, rejected={rejected[:5]}"
            )
        if sum(not entry["healthy"] for entry in entries) != 26:
            raise DataContractError("LUH index must retain all 26 readable damaged acquisitions")
    return entries, {
        **_source_report(receipt),
        "selected_source_sensors": [f"Acc{value}" for value in panel],
        "mechanism_counts": counts,
        "policy": {
            "axes": "one genuine synchronous vertical axis; x/y remain masked",
            "labels": "publisher filename state and F1..F9 identity only",
            "geometry": LUH_GEOMETRY_BASIS,
            "target_zones": dict(LUH_FISHPLATE_TO_TARGET),
            "grouping": "every complete MAT acquisition and all its windows stay in train",
            "evaluation": "LUH is train-only; existing measured validation/test remain untouched",
            "severity": "damage mechanisms are not converted into alert severity",
        },
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# Z24 progressive damage-test MATLAB release
# ---------------------------------------------------------------------------


def _z24_axis(label: str) -> str | None:
    token = label.strip().upper()
    if token.endswith("L"):
        return "x"
    if token.endswith("T"):
        return "y"
    if token.endswith("V"):
        return "z"
    return None


def _z24_state(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"0?([1-9]|1[0-7])", part)
        if match:
            return int(match.group(1))
    match = re.search(r"(?<!\d)(0?[1-9]|1[0-7])setup", path.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise DataContractError(f"{path}: cannot resolve progressive-test state 01..17")


def _z24_schema(path: Path) -> tuple[np.ndarray, list[str]]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "data" not in payload or "labelshulp" not in payload:
        raise DataContractError(f"{path}: expected data and labelshulp")
    data = np.asarray(payload["data"])
    labels = _mat_strings(payload["labelshulp"])
    if data.ndim != 2:
        raise DataContractError(f"{path}: data must be 2D")
    if data.shape[1] != len(labels) and data.shape[0] == len(labels):
        data = data.T
    if data.shape[1] != len(labels):
        raise DataContractError(f"{path}: data/labelshulp mismatch {data.shape} vs {len(labels)}")
    return data, labels


def _auto_index_z24(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit = _canonical_unit(manifest.reader.get("unit"))
    if unit is None:
        raise DataContractError(
            "z24: reader.unit must be a verified physical acceleration unit or the explicit value "
            "'release_native'. The adapter never guesses a physical unit."
        )
    amplitude_calibrated = unit != "release_native"
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    paths = sorted(path for path in root.rglob("*.mat") if path.is_file() and "avt" in path.as_posix().lower())
    if not paths:
        paths = sorted(path for path in root.rglob("*.mat") if path.is_file())
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    reference_states = {1, 2, 8}
    for path in paths:
        try:
            state = _z24_state(path)
            data, labels = _z24_schema(path)
            channel_records = [
                (index, label, _z24_axis(label)) for index, label in enumerate(labels) if _z24_axis(label) is not None
            ]
            selected = _spread(channel_records)
            boards = []
            for board_id, (column, label, axis) in zip(_BOARD_IDS, selected, strict=True):
                assert axis is not None
                boards.append(
                    {
                        "sensor_id": board_id,
                        "signal_path": "data",
                        "channel_indices": {axis: int(column)},
                        "axes_available": [axis],
                        "sensor_position": None,
                        "orientation_matrix": None,
                        "source_sensor_name": label,
                    }
                )
            healthy = state in reference_states
            relative = _relative_source(root, path)
            entries.append(
                {
                    "dataset_id": manifest.id,
                    "source_file": relative,
                    "source_sha256": sha256_file(path),
                    "structure_id": "z24_bridge",
                    "session_id": f"z24_state_{state:02d}",
                    "experiment_id": f"z24_state_{state:02d}::{path.stem}",
                    "sample_rate_hz": 100.0,
                    "unit": unit,
                    "amplitude_calibrated": amplitude_calibrated,
                    "sample_axis": 0,
                    "start_time_s": 0.0,
                    "condition": f"progressive_state_{state:02d}",
                    "training_role": "supervised",
                    "healthy": healthy,
                    "severity": "none" if healthy else "advisory",
                    "class_name": "normal" if healthy else "unknown_anomaly",
                    "affected_targets": [],
                    "phase_synchronized": True,
                    "synchronization_verification_id": "z24_shared_avt_setup_matrix_v1",
                    "sensor_useful_band_hz": 50.0,
                    "boards": boards,
                    "auto_index": {
                        "adapter": manifest.adapter,
                        "schema_basis": "user-supplied EDA: data matrix and labelshulp DOF labels",
                        "label_basis": "publisher progressive state number; states 1, 2 and 8 are distinct references",
                    },
                }
            )
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not entries:
        raise DataContractError(f"{manifest.id}: no MAT setup survived strict indexing: {rejected[:5]}")
    return entries, {
        **_source_report(receipt),
        "policy": {
            "unit": (
                f"explicit verified physical unit {unit}"
                if amplitude_calibrated
                else "release-native scale retained; absolute-amplitude physics features are masked"
            ),
            "references": "states 1, 2 and 8 are retained as separate split groups",
            "axes": "DOF suffix L/T/V maps to x/y/z; each source channel remains single-axis",
            "localisation": "global unknown; no damage location is inferred",
        },
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# Qatar University Grandstand Simulator text release
# ---------------------------------------------------------------------------


QUGS_FIXED_REFERENCE_JOINT = 30
QUGS_FIXED_TARGET_JOINTS = (1, 2, 3, 4, 5)
QUGS_FIXED_PANEL = (QUGS_FIXED_REFERENCE_JOINT, *QUGS_FIXED_TARGET_JOINTS)


def _qugs_mapping(damaged_joint: int | None) -> tuple[list[int], int | None]:
    """Return the immutable official-geometry QUGS six-channel projection.

    The publisher fixes Joint 1..30 to signal columns 1..30.  Published numbered
    geometry forms six rows of five joints, so each target zone is one geometry
    column: k, k+5, ..., k+25.  Joint 30 is the fixed context reference and
    Joints 1..5 are the fixed target sensor row.
    """

    selected = list(QUGS_FIXED_PANEL)
    if damaged_joint is None:
        return selected, None
    if not 1 <= damaged_joint <= 30:
        raise DataContractError(f"invalid QUGS joint {damaged_joint}")
    target_slot = ((int(damaged_joint) - 1) % 5) + 1
    return selected, target_slot


def _qugs_filename(path: Path) -> tuple[str, int | None]:
    match = re.fullmatch(r"zzz([AB])([UD])(\d*)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise DataContractError(f"{path.name}: expected zzzAU, zzzBU, zzzAD1..30 or zzzBD1..30")
    campaign, state_type, suffix = match.groups()
    if state_type.upper() == "U":
        if suffix:
            raise DataContractError(f"{path.name}: healthy U file must not have a joint number")
        return campaign.upper(), None
    joint = int(suffix) if suffix else 0
    if not 1 <= joint <= 30:
        raise DataContractError(f"{path.name}: damaged D file needs joint 1..30")
    return campaign.upper(), joint


def _validate_qugs_first_row(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for _ in range(11):
            if handle.readline() == "":
                raise DataContractError(f"{path}: fewer than 11 ME'scope header rows")
        row = handle.readline().strip().split("\t")
    if len(row) != 31:
        raise DataContractError(f"{path}: expected time + 30 acceleration columns, found {len(row)}")
    try:
        values = np.asarray([float(value) for value in row], dtype=np.float64)
    except ValueError as exc:
        raise DataContractError(f"{path}: first data row is not numeric") from exc
    if not np.isfinite(values).all():
        raise DataContractError(f"{path}: first data row is non-finite")


def _auto_index_qugs(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    paths = sorted(path for path in root.rglob("*.TXT") if path.is_file())
    if not paths:
        paths = sorted(path for path in root.rglob("*.txt") if path.is_file())
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in paths:
        try:
            campaign, joint = _qugs_filename(path)
            _validate_qugs_first_row(path)
            selected, affected_target = _qugs_mapping(joint)
            boards = []
            for board_id, sensor_index in zip(_BOARD_IDS, selected, strict=True):
                boards.append(
                    {
                        "sensor_id": board_id,
                        "signal_path": "__matrix__",
                        "channel_indices": {"x": sensor_index},
                        "axes_available": ["x"],
                        "sensor_position": [
                            ((sensor_index - 1) % 5) / 4.0,
                            ((sensor_index - 1) // 5) / 5.0,
                            0.0,
                        ],
                        "orientation_matrix": None,
                        "source_sensor_name": f"joint_{sensor_index:02d}",
                        "physical_sensor_id": f"qugs_joint_{sensor_index:02d}",
                        "position_basis": "normalized_ordinal_from_published_numbered_layout",
                        "fixed_physical_panel": True,
                    }
                )
            healthy = joint is None
            state_name = "undamaged" if healthy else f"loosened_joint_{joint:02d}"
            relative = _relative_source(root, path)
            entries.append(
                {
                    "dataset_id": manifest.id,
                    "source_file": relative,
                    "source_sha256": sha256_file(path),
                    "structure_id": "qugs_grandstand_simulator",
                    # A/B repetitions of the same physical state share a group.
                    "session_id": f"qugs_state_{0 if healthy else joint:02d}",
                    "experiment_id": f"campaign_{campaign}::{state_name}",
                    "sample_rate_hz": 1024.0,
                    "unit": "m/s^2",
                    "sample_axis": 0,
                    "source_options": {"delimiter": "\t", "has_header": False, "skip_rows": 11},
                    "start_time_s": 0.0,
                    "condition": state_name,
                    "training_role": "supervised",
                    "healthy": healthy,
                    "severity": "none" if healthy else "advisory",
                    "class_name": "normal" if healthy else "unknown_anomaly",
                    "affected_targets": [] if affected_target is None else [affected_target],
                    "phase_synchronized": True,
                    "synchronization_verification_id": "qugs_shared_measurement_matrix_v1",
                    "sensor_useful_band_hz": 512.0,
                    "boards": boards,
                    "auto_index": {
                        "adapter": manifest.adapter,
                        "schema_basis": "user-supplied EDA: 11 header rows, time + 30 DOFs, 1024 Hz",
                        "label_basis": "AU/BU undamaged; ADi/BDi loosened joint i",
                        "grouping": "A/B repetitions of the same joint state share session_id",
                        "localisation": (
                            "fixed Joint 30 context reference plus fixed Joints 1..5 targets; "
                            "damage joint maps to one published geometry column"
                        ),
                        "geometry_basis": (
                            "official fixed Joint 1..30 channels and published six-row by five-column numbered layout"
                        ),
                        "reference_limitation": "Joint 30 is fixed context, not a guaranteed-undamaged channel",
                    },
                }
            )
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not entries:
        raise DataContractError(f"{manifest.id}: no QUGS file survived strict indexing: {rejected[:5]}")
    return entries, {
        **_source_report(receipt),
        "policy": {
            "axes": "one genuine measured acceleration DOF per joint; y/z remain masked",
            "labels": "owner-documented AU/BU and ADi/BDi filename semantics plus fixed geometry-column target zones",
            "geometry": "immutable panel [Joint 30 reference, Joints 1..5 targets] across every condition",
            "grouping": "campaign A and B repeats of each state are indivisible across splits",
            "licence": "training remains disabled in the shipped manifest until reuse permission is acknowledged",
        },
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# DesignSafe PRJ-2447 multi-row CSV release
# ---------------------------------------------------------------------------


def _designsafe_axis(name: str) -> tuple[str, str] | None:
    clean = name.strip()
    match = re.match(r"^(A-.+?)-(NS|EW|V)$", clean, flags=re.IGNORECASE)
    if match is None:
        return None
    axis = {"NS": "x", "EW": "y", "V": "z"}[match.group(2).upper()]
    return match.group(1), axis


def _designsafe_headers(path: Path) -> tuple[str, list[str], list[str]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:131072]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
    if len(rows) < 6:
        raise DataContractError(f"{path}: fewer than six header/data rows")
    names = [item.strip() for item in rows[3]]
    units = [item.strip() for item in rows[4]]
    if len(names) != len(units):
        raise DataContractError(f"{path}: instrument-name/unit row length mismatch")
    return delimiter, names, units


def _auto_index_designsafe(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    paths = sorted(path for path in root.rglob("*.csv") if path.is_file())
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in paths:
        try:
            delimiter, names, units = _designsafe_headers(path)
            grouped: dict[str, dict[str, int]] = {}
            for index, (name, unit) in enumerate(zip(names, units, strict=True)):
                parsed = _designsafe_axis(name)
                if parsed is None or _canonical_unit(unit) != "g":
                    continue
                base, axis = parsed
                grouped.setdefault(base, {})[axis] = index
            selected_names = _spread(sorted(grouped, key=_natural_key))
            boards = []
            for board_id, sensor_name in zip(_BOARD_IDS, selected_names, strict=True):
                axis_indices = grouped[sensor_name]
                boards.append(
                    {
                        "sensor_id": board_id,
                        "signal_path": "__matrix__",
                        "channel_indices": {axis: int(index) for axis, index in axis_indices.items()},
                        "axes_available": [axis for axis in _AXIS_ORDER if axis in axis_indices],
                        "sensor_position": None,
                        "orientation_matrix": None,
                        "source_sensor_name": sensor_name,
                    }
                )
            relative = _relative_source(root, path)
            entries.append(
                {
                    "dataset_id": manifest.id,
                    "source_file": relative,
                    "source_sha256": sha256_file(path),
                    "structure_id": "designsafe_prj2447_specimen",
                    "session_id": path.stem,
                    "experiment_id": path.stem,
                    "sample_rate_hz": 256.0,
                    "unit": "g",
                    "sample_axis": 0,
                    "source_options": {"delimiter": delimiter, "has_header": False, "skip_rows": 5},
                    "invalid_abs_threshold": 1.0e12,
                    "start_time_s": 0.0,
                    "condition": path.stem,
                    "training_role": "representation_only",
                    "phase_synchronized": True,
                    "synchronization_verification_id": "designsafe_shared_daq_csv_v1",
                    "sensor_useful_band_hz": 128.0,
                    "boards": boards,
                    "auto_index": {
                        "adapter": manifest.adapter,
                        "schema_basis": "user-supplied EDA: names row 4, units row 5, numeric data row 6, 256 Hz",
                        "label_basis": "protocol/run identifier only; no damage class is inferred from test number or intensity",
                        "sentinel_policy": "absolute values above 1e12 are invalidated before window quality assessment",
                    },
                }
            )
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    if not entries:
        raise DataContractError(f"{manifest.id}: no DesignSafe CSV survived strict indexing: {rejected[:5]}")
    return entries, {
        **_source_report(receipt),
        "policy": {
            "labels": "representation-only; inspection/protocol metadata are preserved without invented damage labels",
            "axes": "NS/EW/V map to x/y/z only when explicitly present in instrument names",
            "sentinels": "mixed-channel fill values above 1e12 are masked, never interpreted as acceleration",
        },
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# RWTH two-storey frame CSV release
# ---------------------------------------------------------------------------


def _sniff_csv(path: Path) -> tuple[str, list[str]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    header = next(csv.reader([first_line], delimiter=delimiter), [])
    header = [item.strip() for item in header]
    if len(header) < 2:
        raise DataContractError(f"{path}: could not identify a CSV header and delimiter")
    return delimiter, header


def _match_csv_header(header: list[str], aliases: tuple[str, ...], *, label: str) -> str:
    normalised = [(_normalise_token(value), value) for value in header]
    matches: list[str] = []
    for token, original in normalised:
        if any(token == alias or token.startswith(alias) for alias in aliases):
            matches.append(original)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise DataContractError(f"CSV schema needs exactly one {label} column; found {matches} in {header}")
    return matches[0]


def _parse_float(value: str, delimiter: str) -> float:
    text = value.strip()
    if delimiter != "," and "," in text and "." not in text:
        text = text.replace(",", ".")
    return float(text)


def _validate_csv_columns_populated(
    path: Path, delimiter: str, headers: list[str], *, minimum_numeric: int = 3
) -> None:
    counts = {header: 0 for header in headers}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            for header in headers:
                text = str(row.get(header) or "").strip()
                if not text:
                    continue
                try:
                    value = _parse_float(text, delimiter)
                except ValueError:
                    continue
                if math.isfinite(value):
                    counts[header] += 1
            if min(counts.values(), default=0) >= minimum_numeric:
                return
    missing = [header for header, count in counts.items() if count < minimum_numeric]
    raise DataContractError(f"{path}: selected acceleration columns are empty or non-numeric: {missing}")


def _validate_rwth_timebase(path: Path, delimiter: str, time_header: str, expected_rate: float) -> float:
    samples: list[float] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or time_header not in reader.fieldnames:
            raise DataContractError(f"{path}: time column {time_header!r} is absent")
        for row in reader:
            value = row.get(time_header)
            if value is None or not value.strip():
                continue
            try:
                samples.append(_parse_float(value, delimiter))
            except ValueError as exc:
                raise DataContractError(f"{path}: non-numeric time value {value!r}") from exc
            if len(samples) >= 10000:
                break
    if len(samples) < 3:
        raise DataContractError(f"{path}: fewer than three numeric time samples")
    differences = np.diff(np.asarray(samples, dtype=np.float64))
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise DataContractError(f"{path}: time column is not finite and strictly increasing")
    median_step = float(np.median(differences))
    inferred_rate = 1.0 / median_step
    if not math.isclose(inferred_rate, expected_rate, rel_tol=0.01, abs_tol=0.1):
        raise DataContractError(
            f"{path}: inferred sample rate {inferred_rate:.6g} Hz differs from declared {expected_rate:.6g} Hz"
        )
    jitter = float(np.max(np.abs(differences - median_step)))
    if jitter > max(1e-9, median_step * 0.01):
        raise DataContractError(f"{path}: timebase jitter exceeds 1% of the nominal step")
    return inferred_rate


def _auto_index_rwth(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    expected_rate = float(manifest.reader.get("sample_rate_hz", 400.0))
    csv_paths = sorted(root.glob(str(manifest.reader.get("file_glob", "extracted/**/*.csv"))))
    csv_paths = [path for path in csv_paths if path.is_file() and path.suffix.lower() == ".csv"]
    if not csv_paths:
        csv_paths = sorted(path for path in (root / "extracted").rglob("*.csv") if path.is_file())
    if not csv_paths:
        raise DataContractError(f"{manifest.id}: no extracted CSV files found under {root}")

    required_columns = {
        "reference": ("acc0",),
        "target_1": ("acc1",),
        "target_2": ("acc2",),
        "target_3": ("accl",),
        "target_4": ("accf",),
        "target_5": ("acch",),
    }
    entries: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for path in csv_paths:
        try:
            delimiter, header = _sniff_csv(path)
            time_header = _match_csv_header(header, ("time", "times"), label="time")
            selected = {
                board_id: _match_csv_header(header, aliases, label=board_id)
                for board_id, aliases in required_columns.items()
            }
            if len(set(selected.values())) != 6:
                raise DataContractError("six RWTH board roles did not resolve to six distinct CSV columns")
            _validate_csv_columns_populated(path, delimiter, list(selected.values()))
            inferred_rate = _validate_rwth_timebase(path, delimiter, time_header, expected_rate)
            checksum = sha256_file(path)
            relative = _relative_source(root, path)
            session = path.stem
            boards = []
            for board_id in _BOARD_IDS:
                boards.append(
                    {
                        "sensor_id": board_id,
                        "axis_paths": {"x": selected[board_id]},
                        "axes_available": ["x"],
                        "sensor_position": None,
                        "orientation_matrix": None,
                        "source_sensor_name": selected[board_id],
                    }
                )
            entry = {
                "dataset_id": manifest.id,
                "source_file": relative,
                "source_sha256": checksum,
                "structure_id": "rwth_two_storey_frame",
                "session_id": session,
                "experiment_id": session,
                "sample_rate_hz": float(inferred_rate),
                "unit": "m/s^2",
                "start_time_s": 0.0,
                "condition": session,
                # The publisher release contains excitation protocols but no
                # damage-state annotation.  It is therefore admitted only for
                # representation learning and never silently converted into a
                # supervised "normal" class.
                "training_role": "representation_only",
                "phase_synchronized": True,
                "synchronization_verification_id": "rwth_shared_csv_timebase_v1",
                "sensor_useful_band_hz": min(float(inferred_rate) / 2.0, 500.0),
                "boards": boards,
                "auto_index": {
                    "adapter": manifest.adapter,
                    "time_column": time_header,
                    "delimiter": delimiter,
                    "label_basis": (
                        "no publisher damage-state labels are assigned; load-protocol filenames are retained only "
                        "as protocol identifiers and this record is representation-only"
                    ),
                },
            }
            entries.append(entry)
            file_reports.append(
                {
                    "source_file": relative,
                    "sha256": checksum,
                    "delimiter": delimiter.encode("unicode_escape").decode("ascii"),
                    "sample_rate_hz": inferred_rate,
                    "time_column": time_header,
                    "selected_columns": selected,
                }
            )
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    if not entries:
        raise DataContractError(f"{manifest.id}: every CSV file failed automatic schema validation: {rejected[:5]}")
    return entries, {
        "source_record": receipt.get("download", {}).get("record_id"),
        "source_doi": receipt.get("download", {}).get("doi"),
        "source_archive_sha256": _receipt_sha256(receipt, "Data_v1.0.0.zip"),
        "policy": {
            "sensor_mapping": "exact publisher CSV headers",
            "labels": (
                "no supervised vibration labels are created; RWTH records are representation-only because the "
                "publisher supplies excitation protocols rather than damage-state annotations"
            ),
            "synchronization": "shared validated monotonic CSV time column",
        },
        "files": file_reports,
        "rejected_records": rejected,
    }


# ---------------------------------------------------------------------------
# Hell Bridge Test Arena HDF5 release
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _HdfSensor:
    key: str
    axis_paths: dict[str, str]
    axis_lengths: dict[str, int]
    unit: str | None
    position: list[float] | None
    orientation: list[list[float]] | None


def _is_acceleration_group(relative_path: str) -> bool:
    token = _normalise_token(relative_path)
    if not ("acceleration" in token or token.endswith("acc") or "/acc/" in relative_path.lower()):
        return False
    forbidden = ("load", "shaker", "input", "excitation", "force")
    return not any(item in token for item in forbidden)


def _dataset_signal_length(dataset: h5py.Dataset) -> int | None:
    shape = tuple(int(value) for value in dataset.shape if int(value) != 1)
    if len(shape) != 1 or shape[0] < 2:
        return None
    if not np.issubdtype(dataset.dtype, np.number):
        return None
    return shape[0]


def _discover_hdf_sensors(recording: h5py.Group, default_unit: str | None) -> dict[str, _HdfSensor]:
    sensors: dict[str, _HdfSensor] = {}

    def visitor(relative_name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group) or not _is_acceleration_group(relative_name):
            return
        for _sensor_name, sensor_obj in obj.items():
            if not isinstance(sensor_obj, h5py.Group):
                continue
            axis_paths: dict[str, str] = {}
            axis_lengths: dict[str, int] = {}
            units: list[str] = []
            datasets_by_axis: dict[str, h5py.Dataset] = {}
            for component_name, component in sensor_obj.items():
                if not isinstance(component, h5py.Dataset):
                    continue
                axis = _axis_from_component(component_name, component)
                length = _dataset_signal_length(component)
                if axis is None or length is None:
                    continue
                if axis in axis_paths:
                    raise DataContractError(
                        f"{recording.name}: sensor {sensor_obj.name} has multiple datasets for axis {axis}"
                    )
                axis_paths[axis] = component.name
                axis_lengths[axis] = length
                datasets_by_axis[axis] = component
                unit_value = _attr_value(
                    (component.attrs, sensor_obj.attrs, obj.attrs, recording.attrs),
                    ("unit", "units", "physical_unit", "measurement_unit", "unit_symbol"),
                )
                unit = _canonical_unit(unit_value) if unit_value is not None else default_unit
                if unit is not None:
                    units.append(unit)
            if not axis_paths:
                continue
            if len(set(axis_lengths.values())) != 1:
                raise DataContractError(f"{recording.name}: sensor {sensor_obj.name} axes have unequal lengths")
            if len(set(units)) > 1:
                raise DataContractError(f"{recording.name}: sensor {sensor_obj.name} axes use different units")
            relative_sensor = sensor_obj.name[len(recording.name) :].strip("/")
            key = relative_sensor
            representative = datasets_by_axis[next(iter(axis_paths))]
            sensors[key] = _HdfSensor(
                key=key,
                axis_paths={axis: axis_paths[axis] for axis in _AXIS_ORDER if axis in axis_paths},
                axis_lengths={axis: axis_lengths[axis] for axis in _AXIS_ORDER if axis in axis_lengths},
                unit=(units[0] if units else default_unit),
                position=_position_from_attrs(sensor_obj, obj, recording),
                orientation=_orientation_from_attrs(sensor_obj, obj, recording, representative),
            )

    recording.visititems(visitor)
    return sensors


def _group_ancestors(handle: h5py.File, group: h5py.Group) -> list[h5py.Group]:
    parts = [part for part in group.name.strip("/").split("/") if part]
    result: list[h5py.Group] = []
    for index in range(1, len(parts) + 1):
        obj = handle["/" + "/".join(parts[:index])]
        if isinstance(obj, h5py.Group):
            result.append(obj)
    return result


def _collect_condition_metadata(handle: h5py.File, recording: h5py.Group) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for group in _group_ancestors(handle, recording):
        for key, value in group.attrs.items():
            items.append((f"{group.name}@{key}", _decode_scalar(value)))

    keywords = ("state", "damage", "condition", "health", "reference", "description")

    def visitor(relative_name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            depth = len([part for part in relative_name.split("/") if part])
            if depth <= 2:
                for key, value in obj.attrs.items():
                    if any(word in _normalise_token(key) for word in keywords):
                        items.append((f"{obj.name}@{key}", _decode_scalar(value)))
            return
        token = _normalise_token(relative_name)
        if any(word in token for word in keywords) and obj.size <= 32:
            with contextlib.suppress(OSError):
                items.append((obj.name, _decode_scalar(np.asarray(obj))))

    recording.visititems(visitor)
    return items


def _condition_field_name(key: str) -> str:
    if "@" in key:
        return key.rsplit("@", 1)[-1]
    return Path(key).name


def _condition_from_candidate(key: str, value: Any) -> tuple[bool, str] | None:
    key_token = _normalise_token(_condition_field_name(key))
    if not any(word in key_token for word in ("state", "damage", "condition", "health", "reference", "description")):
        return None
    decoded = _decode_scalar(value)
    if isinstance(decoded, (int, float)) and not isinstance(decoded, bool):
        number = float(decoded)
        if math.isfinite(number) and number.is_integer():
            state = int(number)
            if state == 0:
                return True, f"state_{state}"
            if state > 0:
                return False, f"state_{state}"
    text = _string_value(decoded).strip()
    token = text.lower()
    compact = _normalise_token(token)
    if re.fullmatch(r"uds\d*", compact):
        return True, text
    if re.fullmatch(r"ds\d+", compact):
        return False, text
    if any(word in token for word in ("undamaged", "healthy", "intact", "reference", "baseline")):
        return True, text
    state_match = re.search(r"(?:structural\s*)?(?:state|ss|ds|damage)[\s_\-:#]*(\d+)", token)
    if state_match:
        state = int(state_match.group(1))
        return (state == 0), text
    if any(word in token for word in ("damaged", "damage", "removed", "loosened", "loose", "crack", "cut")):
        return False, text
    return None


def _resolve_hbta_condition(
    handle: h5py.File,
    recording: h5py.Group,
) -> tuple[bool, str, str] | None:
    metadata = _collect_condition_metadata(handle, recording)
    priority = sorted(
        metadata,
        key=lambda item: (
            0
            if any(
                word in _normalise_token(_condition_field_name(item[0]))
                for word in ("structuralstate", "damagestate", "condition")
            )
            else 1,
            item[0],
        ),
    )
    for key, value in priority:
        resolved = _condition_from_candidate(key, value)
        if resolved is not None:
            healthy, condition = resolved
            return healthy, condition, key

    # A recording/group path is accepted only when it contains an explicit state word.
    lower_path = recording.name.lower()
    uds_match = re.search(r"(?:^|[/_\-])uds(?:[_\- ]*\d+)?(?:$|[/_\-])", lower_path)
    if uds_match:
        return True, uds_match.group(0).strip("/_- "), "recording_path_explicit_uds_token"
    ds_match = re.search(r"(?:^|[/_\-])ds[_\- ]*(\d+)(?:$|[/_\-])", lower_path)
    if ds_match:
        return False, f"DS{int(ds_match.group(1))}", "recording_path_explicit_ds_token"
    path_match = re.search(
        r"(?:^|[/_\-])(?:state|ss|damage)[_\- ]*(\d+)(?:$|[/_\-])",
        lower_path,
    )
    if path_match:
        state = int(path_match.group(1))
        return state == 0, f"state_{state}", "recording_path_explicit_state"
    if any(word in lower_path for word in ("undamaged", "healthy", "reference", "baseline")):
        return True, recording.name, "recording_path_explicit_healthy_token"
    if any(word in lower_path for word in ("damaged", "damage")):
        return False, recording.name, "recording_path_explicit_damage_token"
    return None


def _sample_rate_from_group(handle: h5py.File, recording: h5py.Group, fallback: float) -> float:
    for group in reversed(_group_ancestors(handle, recording)):
        value = _attr_value(
            (group.attrs,),
            ("samplerate", "sample_rate", "sampling_rate", "sampling_frequency", "fs", "frequency"),
        )
        if value is None:
            continue
        try:
            rate = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    return fallback


def _recording_groups(
    handle: h5py.File,
    default_unit: str | None,
) -> list[tuple[h5py.Group, dict[str, _HdfSensor]]]:
    candidates: list[tuple[h5py.Group, dict[str, _HdfSensor]]] = []

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group) or obj.name == "/" or obj.name.split("/")[-1].startswith("."):
            return
        sensors = _discover_hdf_sensors(obj, default_unit)
        if len(sensors) >= 6:
            candidates.append((obj, sensors))

    handle.visititems(visitor)
    # Keep deepest qualifying groups. A structural-state parent can otherwise aggregate child recordings.
    candidate_names = {group.name for group, _ in candidates}
    filtered = []
    for group, sensors in candidates:
        prefix = group.name.rstrip("/") + "/"
        if any(name.startswith(prefix) for name in candidate_names if name != group.name):
            continue
        filtered.append((group, sensors))
    return sorted(filtered, key=lambda item: _natural_key(item[0].name))


def _select_common_hbta_sensors(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    coverage: dict[str, int] = {}
    examples: dict[str, _HdfSensor] = {}
    for record in records:
        for key, sensor in record["sensors"].items():
            coverage[key] = coverage.get(key, 0) + 1
            examples.setdefault(key, sensor)
    total = len(records)
    ranked = sorted(
        coverage,
        key=lambda key: (
            -coverage[key],
            -(1 if "z" in examples[key].axis_paths else 0),
            -len(examples[key].axis_paths),
            _natural_key(key),
        ),
    )
    fully_common = [key for key in ranked if coverage[key] == total]

    # HBTA documents response accelerometers using ALxx (uniaxial vertical)
    # and AGxx (triaxial) identifiers.  Prefer six AL locations because they
    # are genuine, unambiguous vertical channels and are available densely
    # across the bridge.  The selection is deterministic and spatially spread
    # in identifier order; arbitrary "first six datasets" mappings are not
    # permitted.
    def published_sensor_id(key: str) -> tuple[str, int] | None:
        match = re.search(r"(?<![a-z0-9])(al|ag)[_\- ]*0*(\d{1,2})(?![a-z0-9])", key.lower())
        if match is None:
            return None
        return match.group(1).upper(), int(match.group(2))

    identified: dict[str, tuple[str, int]] = {}
    for key in fully_common:
        identity = published_sensor_id(key)
        if identity is not None:
            identified[key] = identity

    al_candidates = sorted(
        [key for key, identity in identified.items() if identity[0] == "AL" and "z" in examples[key].axis_paths],
        key=lambda key: identified[key][1],
    )
    ag_candidates = sorted(
        [key for key, identity in identified.items() if identity[0] == "AG"],
        key=lambda key: identified[key][1],
    )
    candidates = al_candidates if len(al_candidates) >= 6 else ag_candidates
    if len(candidates) < 6:
        available = sorted({identity[0] for identity in identified.values()})
        raise DataContractError(
            "Hell Bridge automatic mapping needs at least six fully common publisher-labelled AL or AG response "
            f"sensor locations; found AL={len(al_candidates)}, AG={len(ag_candidates)}, labels={available}"
        )

    indices = np.linspace(0, len(candidates) - 1, num=6)
    selected = [candidates[int(round(index))] for index in indices]
    if len(set(selected)) != 6:  # Defensive guard against any future selection change.
        raise DataContractError("Hell Bridge deterministic sensor spacing did not produce six distinct locations")
    included = [record for record in records if all(key in record["sensors"] for key in selected)]
    if not any(record["healthy"] for record in included):
        raise DataContractError("automatic Hell Bridge mapping retained no undamaged/reference recording")
    if not any(not record["healthy"] for record in included):
        raise DataContractError("automatic Hell Bridge mapping retained no known-damage recording")
    return selected, included


def _auto_index_hell_bridge(
    manifest: DatasetManifest,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = manifest.raw_root(config)
    receipt = _receipt(root)
    source_name = str(manifest.reader.get("source_filename", "data_100Hz.h5"))
    candidates = sorted(path for path in root.rglob(source_name) if path.is_file())
    if len(candidates) != 1:
        raise DataContractError(f"{manifest.id}: expected exactly one {source_name}, found {candidates}")
    source_path = candidates[0]
    relative_source = _relative_source(root, source_path)
    source_checksum = _receipt_sha256(receipt, source_name) or sha256_file(source_path)
    default_rate = float(manifest.reader.get("sample_rate_hz", 100.0))
    default_unit = _canonical_unit(manifest.reader.get("unit", "m/s^2"))
    if default_unit is None:
        raise DataContractError(f"{manifest.id}: reader.unit must be a supported physical acceleration unit")

    resolved_records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    with h5py.File(source_path, "r") as handle:
        groups = _recording_groups(handle, default_unit)
        if not groups:
            raise DataContractError(
                f"{manifest.id}: no recording group contained six acceleration sensors in the expected "
                "recording/sensor-group/sensor/component hierarchy"
            )
        for recording, sensors in groups:
            try:
                condition = _resolve_hbta_condition(handle, recording)
                if condition is None:
                    raise DataContractError("no explicit publisher structural-state metadata was found")
                healthy, condition_text, condition_source = condition
                rate = _sample_rate_from_group(handle, recording, default_rate)
                if not math.isclose(rate, default_rate, rel_tol=1e-6, abs_tol=1e-6):
                    raise DataContractError(
                        f"recording sample rate {rate} Hz does not match pinned release rate {default_rate} Hz"
                    )
                resolved_records.append(
                    {
                        "recording_name": recording.name,
                        "healthy": healthy,
                        "condition": condition_text,
                        "condition_source": condition_source,
                        "sample_rate_hz": rate,
                        "sensors": sensors,
                    }
                )
            except Exception as exc:
                rejected.append({"recording": recording.name, "reason": f"{type(exc).__name__}: {exc}"})

        if not resolved_records:
            raise DataContractError(f"{manifest.id}: no recording had an explicit usable state label: {rejected[:5]}")
        selected_keys, included_records = _select_common_hbta_sensors(resolved_records)

        entries: list[dict[str, Any]] = []
        included_names = {record["recording_name"] for record in included_records}
        for record in resolved_records:
            if record["recording_name"] not in included_names:
                rejected.append(
                    {
                        "recording": record["recording_name"],
                        "reason": "missing one or more globally selected six sensor locations",
                    }
                )
                continue
            sensors = [record["sensors"][key] for key in selected_keys]
            lengths = {next(iter(sensor.axis_lengths.values())) for sensor in sensors}
            if len(lengths) != 1:
                rejected.append(
                    {
                        "recording": record["recording_name"],
                        "reason": "selected sensors do not share a common sample count",
                    }
                )
                continue
            units = {sensor.unit or default_unit for sensor in sensors}
            if len(units) != 1:
                rejected.append(
                    {
                        "recording": record["recording_name"],
                        "reason": f"selected sensors use inconsistent units: {sorted(units)}",
                    }
                )
                continue
            unit = next(iter(units))
            boards = []
            for board_id, sensor_key, sensor in zip(_BOARD_IDS, selected_keys, sensors, strict=True):
                boards.append(
                    {
                        "sensor_id": board_id,
                        "axis_paths": sensor.axis_paths,
                        "axes_available": [axis for axis in _AXIS_ORDER if axis in sensor.axis_paths],
                        "sensor_position": sensor.position,
                        "orientation_matrix": sensor.orientation,
                        "source_sensor_name": sensor_key,
                    }
                )
            recording_id = record["recording_name"].strip("/").replace("/", "::")
            healthy = bool(record["healthy"])
            entries.append(
                {
                    "dataset_id": manifest.id,
                    "source_file": relative_source,
                    "source_sha256": source_checksum,
                    "structure_id": "hell_bridge",
                    "session_id": recording_id,
                    "experiment_id": recording_id,
                    "sample_rate_hz": float(record["sample_rate_hz"]),
                    "unit": unit,
                    "start_time_s": 0.0,
                    "condition": record["condition"],
                    "training_role": "supervised",
                    "healthy": healthy,
                    "severity": "none" if healthy else "advisory",
                    "class_name": "normal" if healthy else "unknown_anomaly",
                    "affected_targets": [],
                    # A shared HDF5 recording establishes aligned sample count,
                    # but this adapter does not claim independently verified
                    # sample-level phase synchronisation.  Phase-sensitive
                    # features therefore remain disabled by default.
                    "phase_synchronized": False,
                    "synchronization_verification_id": None,
                    "sensor_useful_band_hz": float(record["sample_rate_hz"]) / 2.0,
                    "boards": boards,
                    "auto_index": {
                        "adapter": manifest.adapter,
                        "condition_source": record["condition_source"],
                        "label_basis": (
                            "publisher structural-state metadata; known damage is conservatively mapped to "
                            "unknown_anomaly/advisory because the release does not provide an engineering alert severity"
                        ),
                        "sensor_selection": (
                            "six deterministic, spatially spread publisher-labelled AL response accelerometer "
                            "locations common across retained recordings (AG fallback only when six AL channels "
                            "are unavailable)"
                        ),
                    },
                }
            )

    if not entries:
        raise DataContractError(f"{manifest.id}: no complete six-sensor recording survived strict validation")
    if not any(entry["healthy"] for entry in entries) or not any(not entry["healthy"] for entry in entries):
        raise DataContractError("Hell Bridge automatic index must contain both reference and known-damage states")
    return entries, {
        "source_record": receipt.get("download", {}).get("record_id"),
        "source_doi": receipt.get("download", {}).get("doi"),
        "source_file": relative_source,
        "source_sha256": source_checksum,
        "selected_source_sensors": selected_keys,
        "policy": {
            "sensor_mapping": "exact HDF5 component paths discovered from the NTNU open-data hierarchy",
            "axes": "only explicit x/y/z component datasets are admitted",
            "labels": "only explicit structural-state metadata or explicit state tokens are admitted",
            "localisation": "global unknown; no damage location is inferred from filenames",
            "severity": "known damage maps to advisory, not warning/critical, because engineering severity is absent",
            "synchronization": (
                "equal sample count is validated, but sample-level phase synchronisation is not asserted; "
                "cross-phase/coherence features remain gated off"
            ),
        },
        "resolved_recordings": len(resolved_records),
        "retained_recordings": len(entries),
        "healthy_recordings": sum(bool(entry["healthy"]) for entry in entries),
        "known_damage_recordings": sum(not bool(entry["healthy"]) for entry in entries),
        "rejected_records": rejected,
    }
