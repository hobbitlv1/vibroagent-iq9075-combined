#!/usr/bin/env python3
"""Convert one six-board live window into VibroGemma's 84 soft tokens."""

from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SLOTS = ("baseline", "target_1", "target_2", "target_3", "target_4", "target_5")
RATE_PREFIX = "fs_hz__"
METADATA_PREFIX = "metadata__"
POSITION_PREFIX = "sensor_position__"
ORIENTATION_PREFIX = "orientation_matrix__"
AXIS_MASK_PREFIX = "axis_mask__"
_REQUIRED_BUNDLE_FILES = (
    "vibrogemma_config.json",
    "vibration_encoder_projector.onnx",
    "vibration_encoder_projector.onnx.data",
)


def _pipeline_source(bundle: Path | None = None) -> Path:
    project = Path(__file__).resolve().parents[1]
    candidates = []
    configured = os.environ.get("VIBROGEMMA_PIPELINE_SRC", "").strip()
    if configured:
        candidates.append(Path(configured))
    elif bundle is not None:
        candidates.append(bundle.resolve() / "pipeline_src")
    candidates.extend(
        [
            project
            / ".run"
            / "VibroGemma_Repository_v0.13.5"
            / "vibrogemma-full-pipeline-v13.5"
            / "src",
            project / ".run" / "vibrogemma-full-pipeline-v4" / "src",
            Path("/media/ubuntu/Drive/vibrogemma-full-pipeline/src"),
        ]
    )
    for candidate in candidates:
        if (candidate / "vibrogemma" / "contracts.py").is_file():
            return candidate.resolve()
    raise RuntimeError("VibroGemma pipeline source not found; run prepare_vibrogemma.sh")


def _load_contracts(bundle: Path):
    source = _pipeline_source(bundle)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from vibrogemma.contracts import (  # noqa: PLC0415
        BoardWindow,
        EpisodeProvenance,
        QualityReport,
        SixBoardEpisode,
    )
    from vibrogemma.signal.multiresolution import MultiResolutionEpisodePreprocessor  # noqa: PLC0415

    return BoardWindow, EpisodeProvenance, QualityReport, SixBoardEpisode, MultiResolutionEpisodePreprocessor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_pin(name: str) -> str:
    value = os.environ.get(name, "").strip().lower()
    if value and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise RuntimeError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value


def _resolved_healthy_profile(bundle: Path, configured_profile: str | None = None) -> Path | None:
    if os.environ.get("VIBROGEMMA_DISABLE_HEALTHY_PROFILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    external = os.environ.get("VIBROGEMMA_HEALTHY_PROFILE", "").strip()
    if external:
        return Path(external).expanduser().resolve()
    if configured_profile is None:
        payload = json.loads((bundle / "vibrogemma_config.json").read_text(encoding="utf-8"))
        configured_profile = str(
            payload.get("model", {})
            .get("multiresolution", {})
            .get("physics", {})
            .get("healthy_profile_path")
            or ""
        ).strip()
    return (bundle / configured_profile).resolve() if configured_profile else None


def _file_identity(path: Path) -> tuple[str, int | None, int | None]:
    try:
        stat = path.stat()
    except OSError:
        return str(path), None, None
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _verify_runtime_artifacts_cached(
    bundle_text: str,
    source_text: str,
    profile_text: str,
    manifest_pin: str,
    profile_pin: str,
    _file_identities: tuple[tuple[str, int | None, int | None], ...],
) -> dict:
    bundle = Path(bundle_text)
    source = Path(source_text)
    manifest_path = bundle / "bundle_manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_pin and manifest_sha256 != manifest_pin:
        raise RuntimeError(
            f"VibroGemma bundle manifest sha256 mismatch: {manifest_sha256} != {manifest_pin}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("VibroGemma bundle manifest has no files inventory")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("VibroGemma bundle manifest contains an invalid file row")
        relative = row["path"]
        digest = str(row.get("sha256") or "").lower()
        if relative in expected or len(digest) != 64:
            raise RuntimeError("VibroGemma bundle manifest contains an invalid or duplicate hash row")
        expected[relative] = digest

    observed: dict[str, str] = {}
    for relative in _REQUIRED_BUNDLE_FILES:
        path = bundle / relative
        if relative not in expected or not path.is_file():
            raise RuntimeError(f"VibroGemma bundle is missing manifest-bound artifact {relative}")
        digest = _sha256(path)
        if digest != expected[relative]:
            raise RuntimeError(f"VibroGemma artifact sha256 mismatch for {relative}")
        observed[relative] = digest

    package = source / "vibrogemma"
    python_files = sorted(package.rglob("*.py")) if package.is_dir() else []
    manifest_python = {
        relative: digest
        for relative, digest in expected.items()
        if relative.startswith("pipeline_src/vibrogemma/") and relative.endswith(".py")
    }
    observed_python = {
        str(Path("pipeline_src/vibrogemma") / path.relative_to(package)): path
        for path in python_files
    }
    if set(observed_python) != set(manifest_python):
        missing = sorted(set(manifest_python) - set(observed_python))
        extra = sorted(set(observed_python) - set(manifest_python))
        raise RuntimeError(
            f"VibroGemma pipeline source inventory mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    pipeline_digest = hashlib.sha256()
    for relative, path in sorted(observed_python.items()):
        digest = _sha256(path)
        if digest != manifest_python[relative]:
            raise RuntimeError(f"VibroGemma pipeline source sha256 mismatch for {relative}")
        pipeline_digest.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")

    profile = Path(profile_text) if profile_text else None
    profile_sha256 = None
    if profile is not None:
        if not profile.is_file():
            raise RuntimeError(f"VibroGemma healthy profile does not exist: {profile}")
        profile_sha256 = _sha256(profile)
        external = os.environ.get("VIBROGEMMA_HEALTHY_PROFILE", "").strip()
        if external:
            if profile_pin and profile_sha256 != profile_pin:
                raise RuntimeError(
                    f"VibroGemma healthy profile sha256 mismatch: {profile_sha256} != {profile_pin}"
                )
        else:
            relative = profile.relative_to(bundle).as_posix()
            if relative not in expected or profile_sha256 != expected[relative]:
                raise RuntimeError(f"VibroGemma bundled healthy profile sha256 mismatch for {relative}")
    elif profile_pin:
        raise RuntimeError(
            "VIBROGEMMA_HEALTHY_PROFILE_SHA256 requires VIBROGEMMA_HEALTHY_PROFILE"
        )

    sha256_identities = {
        "bundle_manifest.json": manifest_sha256,
        **observed,
        "pipeline_src/vibrogemma/**/*.py": pipeline_digest.hexdigest(),
        "healthy_profile": profile_sha256,
    }
    runtime_sha256 = hashlib.sha256(
        json.dumps(sha256_identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "pipeline_source": str(source),
        "pipeline_python_file_count": len(observed_python),
        "runtime_sha256": runtime_sha256,
        "sha256": sha256_identities,
    }


def _verify_runtime_artifacts(bundle: Path) -> dict:
    bundle = bundle.resolve()
    source = _pipeline_source(bundle)
    profile = _resolved_healthy_profile(bundle)
    profile_pin = _sha256_pin("VIBROGEMMA_HEALTHY_PROFILE_SHA256")
    if profile_pin and not os.environ.get("VIBROGEMMA_HEALTHY_PROFILE", "").strip():
        raise RuntimeError(
            "VIBROGEMMA_HEALTHY_PROFILE_SHA256 requires VIBROGEMMA_HEALTHY_PROFILE"
        )
    paths = [
        bundle / "bundle_manifest.json",
        *(bundle / relative for relative in _REQUIRED_BUNDLE_FILES),
        *sorted((source / "vibrogemma").rglob("*.py")),
    ]
    if profile is not None:
        paths.append(profile)
    verified = _verify_runtime_artifacts_cached(
        str(bundle),
        str(source),
        str(profile) if profile is not None else "",
        _sha256_pin("VIBROGEMMA_BUNDLE_MANIFEST_SHA256"),
        profile_pin,
        tuple(_file_identity(path) for path in paths),
    )
    return copy.deepcopy(verified)


def _runtime_config(bundle: Path) -> dict:
    payload = json.loads((bundle / "vibrogemma_config.json").read_text(encoding="utf-8"))
    config = {
        "_config_path": str((bundle / "vibrogemma_config.json").resolve()),
        "project": {},
        "model": copy.deepcopy(payload["model"]),
        "labels": copy.deepcopy(payload.get("labels", {})),
        "data": {"window_seconds": 10.0, "minimum_valid_fraction": 0.98},
    }
    physics = config["model"]["multiresolution"]["physics"]
    profile_path = os.environ.get("VIBROGEMMA_HEALTHY_PROFILE", "").strip()
    configured_profile = str(physics.get("healthy_profile_path") or "").strip()
    resolved_profile = _resolved_healthy_profile(bundle, configured_profile)
    physics["healthy_profile_path"] = str(resolved_profile) if resolved_profile else None
    physics["require_healthy_profile"] = bool(
        resolved_profile and (profile_path or physics.get("require_healthy_profile"))
    )
    return config


def _payload_value(payload, key: str, default=None):
    try:
        value = payload[key]
    except (KeyError, TypeError):
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _payload_text(payload, key: str, default: str = "") -> str:
    value = _payload_value(payload, key, default)
    return str(value or default).strip()


def _payload_mapping(payload, key: str) -> dict:
    value = _payload_value(payload, key, {})
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        try:
            decoded = json.loads(value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _optional_array(payload, key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    value = _payload_value(payload, key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{key} must contain finite values with shape {shape}")
    return array


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _healthy_profile_cache_key(bundle: Path) -> str:
    path = _resolved_healthy_profile(bundle)
    if path is None:
        return "none"
    try:
        stat = path.stat()
    except OSError:
        return f"{path}:missing"
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _episode(payload, bundle: Path, *, amplitude_calibrated: bool):
    BoardWindow, EpisodeProvenance, QualityReport, SixBoardEpisode, _ = _load_contracts(bundle)
    capture_metadata = _payload_mapping(payload, "capture_metadata")
    rates = {slot: float(np.asarray(payload[f"{RATE_PREFIX}{slot}"]).reshape(())) for slot in SLOTS}
    if max(rates.values()) - min(rates.values()) > 1e-6:
        raise ValueError(f"all live boards must share one sample rate, got {rates}")
    sample_rate = rates["baseline"]
    sample_count = int(round(10.0 * sample_rate))
    boards = []
    for index, slot in enumerate(SLOTS):
        capture = _payload_mapping(payload, f"{METADATA_PREFIX}{slot}")
        values = np.asarray(payload[slot], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < sample_count:
            raise ValueError(f"{slot}: expected [3, >= {sample_count}] values, got {values.shape}")
        values = values[:, -sample_count:].copy()
        supplied_axis_mask = _optional_array(payload, f"{AXIS_MASK_PREFIX}{slot}", (3,))
        axis_mask = (
            np.ones(3, dtype=bool)
            if supplied_axis_mask is None
            else supplied_axis_mask.astype(bool)
        )
        if not axis_mask.any():
            raise ValueError(f"{slot}: at least one measured axis is required")
        finite = np.isfinite(values)
        sample_mask = finite[axis_mask].all(axis=0)
        values[~finite] = 0.0
        values[~axis_mask] = 0.0
        valid_fraction = float(sample_mask.mean())
        timestamp_warning = capture.get("timestamp_warning")
        timestamp_monotonic = capture.get("timestamp_monotonic")
        if timestamp_monotonic is None:
            timestamp_monotonic = not bool(timestamp_warning)
        notes = [str(item) for item in capture.get("quality_notes") or [] if str(item).strip()]
        if capture.get("clock_skew_ms") is None:
            notes.append("clock_skew_unmeasured")
        if capture.get("clipped_fraction") is None:
            notes.append("clipping_unmeasured")
        if capture.get("packet_loss_count") is None:
            notes.append("packet_loss_unmeasured")
        if timestamp_warning:
            notes.append(f"sdk_timestamp_warning={timestamp_warning}")
        packet_loss_count = capture.get("packet_loss_count")
        if packet_loss_count:
            notes.append(f"timestamp_gaps={int(packet_loss_count)}")
        clipping_values = [
            _optional_float(value)
            for value in (capture.get("clipping_abs_g_by_axis") or {}).values()
        ]
        clipping_abs_g = (
            min(value for value in clipping_values if value is not None)
            if clipping_values and all(value is not None and value > 0 for value in clipping_values)
            else None
        )
        start_time_s = _optional_float(capture.get("selected_timestamp_start_s"))
        if start_time_s is None:
            start_time_s = 0.0
            notes.append("acquisition_start_time_unavailable")
        position = _optional_array(payload, f"{POSITION_PREFIX}{slot}", (3,))
        orientation = _optional_array(payload, f"{ORIENTATION_PREFIX}{slot}", (3, 3))
        boards.append(
            BoardWindow(
                board_id="reference" if index == 0 else slot,
                values=values,
                sample_rate_hz=sample_rate,
                start_time_s=start_time_s,
                axis_mask=axis_mask,
                sample_mask=sample_mask,
                role="reference" if index == 0 else "target",
                target_index=None if index == 0 else index,
                sensor_position=tuple(float(item) for item in position) if position is not None else None,
                orientation_matrix=orientation,
                quality=QualityReport(
                    valid_fraction=valid_fraction,
                    missing_fraction=1.0 - valid_fraction,
                    clipped_fraction=float(capture.get("clipped_fraction") or 0.0),
                    timestamp_monotonic=bool(timestamp_monotonic),
                    clock_skew_ms=float(capture.get("clock_skew_ms") or 0.0),
                    axes_present=tuple(bool(value) for value in axis_mask),
                    usable_band_hz=min(6000.0, sample_rate / 2.0),
                    packet_loss_count=int(packet_loss_count or 0),
                    notes=notes,
                ),
                metadata={
                    "unit": "g",
                    "sensor_useful_band_hz": min(6000.0, sample_rate / 2.0),
                    "amplitude_calibrated": amplitude_calibrated,
                    "clock_skew_measured": capture.get("clock_skew_ms") is not None,
                    "clipping_measured": capture.get("clipped_fraction") is not None,
                    "packet_loss_measured": capture.get("packet_loss_count") is not None,
                    "clipping_abs_g": clipping_abs_g,
                    "capture": capture,
                },
            )
        )
    episode_id = _payload_text(payload, "episode_id") or f"live-{time.time_ns()}"
    provenance_data = capture_metadata.get("episode_provenance")
    if isinstance(provenance_data, dict):
        provenance = EpisodeProvenance(
            dataset_id=str(provenance_data.get("dataset_id") or ""),
            provenance_type=str(provenance_data.get("provenance_type") or ""),
            source_files=[str(value) for value in provenance_data.get("source_files") or []],
            source_checksums=[str(value) for value in provenance_data.get("source_checksums") or []],
            licence=str(provenance_data.get("licence") or ""),
            licence_accepted=bool(provenance_data.get("licence_accepted")),
            structure_id=str(provenance_data.get("structure_id") or ""),
            session_id=str(provenance_data.get("session_id") or ""),
            experiment_id=str(provenance_data.get("experiment_id") or ""),
            split_group=str(provenance_data.get("split_group") or ""),
            derived_operations=[
                str(value) for value in provenance_data.get("derived_operations") or []
            ],
            local_training_data=bool(provenance_data.get("local_training_data")),
        )
    else:
        provenance = EpisodeProvenance(
            dataset_id="vibroagent_live",
            provenance_type="live_unlabelled",
            source_files=[],
            source_checksums=[],
            licence="local live stream; inference only",
            licence_accepted=True,
            structure_id="live_installation",
            session_id="live",
            experiment_id="live",
            split_group="live",
            derived_operations=["tail_window", "multi_resolution_preprocess"],
            local_training_data=True,
        )
    geometry_valid = all(board.sensor_position is not None for board in boards)
    phase_synchronized = bool(capture_metadata.get("phase_synchronized"))
    synchronization_verification_id = str(
        capture_metadata.get("synchronization_verification_id") or ""
    ).strip()
    if phase_synchronized and not synchronization_verification_id:
        raise ValueError("phase_synchronized capture requires synchronization_verification_id")
    return SixBoardEpisode(
        episode_id=episode_id,
        boards=boards,
        provenance=provenance,
        metadata={
            "phase_synchronized": phase_synchronized,
            "synchronization_verification_id": synchronization_verification_id or None,
            "amplitude_calibrated": amplitude_calibrated,
            "geometry_valid": geometry_valid,
            "geometry_source": (
                str(capture_metadata.get("geometry_source") or "sensor_registry")
                if geometry_valid
                else "unconfigured"
            ),
            "capture": capture_metadata,
        },
    )


@lru_cache(maxsize=2)
def _preprocessor_runtime(bundle_text: str, _profile_cache_key: str, _artifact_cache_key: str):
    bundle = Path(bundle_text)
    *_, MultiResolutionEpisodePreprocessor = _load_contracts(bundle)
    return MultiResolutionEpisodePreprocessor.from_config(_runtime_config(bundle))


@lru_cache(maxsize=2)
def _encoder_session(bundle_text: str, _artifact_cache_key: str):
    import onnxruntime as ort  # local worker dependency

    bundle = Path(bundle_text)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(bundle / "vibration_encoder_projector.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _encoder_runtime(bundle_text: str, profile_cache_key: str, artifact_cache_key: str):
    return (
        _preprocessor_runtime(bundle_text, profile_cache_key, artifact_cache_key),
        _encoder_session(bundle_text, artifact_cache_key),
    )


def _validate_encoder_output(tokens: np.ndarray, mask: np.ndarray) -> None:
    if tokens.shape != (1, 84, 1536) or mask.shape != (1, 84):
        raise ValueError(f"encoder output ABI mismatch: {tokens.shape}, {mask.shape}")
    if not np.isfinite(tokens).all():
        raise ValueError("encoder produced non-finite soft tokens")
    for board_index in range(6):
        start = board_index * 14
        stop = start + 14
        board_mask = mask[0, start:stop]
        if not board_mask.any():
            raise ValueError(f"encoder produced no valid soft tokens for board {board_index}")
        if not np.any(tokens[0, start:stop][board_mask] != 0.0):
            raise ValueError(f"encoder produced only zero-valued valid soft tokens for board {board_index}")


def encode_live_window(payload, bundle: Path, *, amplitude_calibrated: bool = False) -> dict:
    bundle = bundle.resolve()
    artifact_verification = _verify_runtime_artifacts(bundle)
    preprocessor, session = _encoder_runtime(
        str(bundle),
        _healthy_profile_cache_key(bundle),
        artifact_verification["runtime_sha256"],
    )
    episode = _episode(payload, bundle, amplitude_calibrated=amplitude_calibrated)
    prepared = preprocessor.prepare(episode)
    for index, source_board in enumerate(episode.boards):
        if not source_board.quality.timestamp_monotonic:
            prepared.episode.boards[index].quality.timestamp_monotonic = False
            prepared.quality_features[index, 3] = 0.0
    window_end_utc = _payload_text(payload, "window_end_utc")
    window_start_utc = _payload_text(payload, "window_start_utc")
    window_time_source = _payload_text(payload, "window_time_source", "sdk_capture_metadata")
    if not window_end_utc:
        end = datetime.now(timezone.utc)
        window_end_utc = end.isoformat()
        window_time_source = "encoder_completion_fallback"
    else:
        end = datetime.fromisoformat(window_end_utc.replace("Z", "+00:00"))
    if not window_start_utc:
        window_start_utc = (end - timedelta(seconds=10)).isoformat()
    quality = prepared.quality_features
    expected_quality = session.get_inputs()[-1].shape[-1]
    if quality.shape[-1] + 1 == expected_quality:
        calibration = np.full((6, 1), float(amplitude_calibrated), dtype=np.float32)
        quality = np.concatenate((quality, calibration), axis=1)
    if quality.shape != (6, expected_quality):
        raise ValueError(f"quality feature ABI mismatch: got {quality.shape}, expected (6, {expected_quality})")

    arrays = {
        "modal_values": prepared.modal_values[None],
        "modal_axis_mask": prepared.modal_axis_mask[None],
        "modal_sample_mask": prepared.modal_sample_mask[None],
        "wideband_features": prepared.wideband_features[None],
        "wideband_frequency_mask": prepared.wideband_frequency_mask[None],
        "wideband_cross_valid": prepared.wideband_cross_valid[None],
        "physics_features": prepared.physics_features[None],
        "physics_mask": prepared.physics_mask[None],
        "sensor_positions": prepared.sensor_positions[None],
        "orientation_features": prepared.orientation_features[None],
        "quality_features": quality[None],
    }
    tokens, mask = session.run(None, arrays)
    tokens = np.asarray(tokens, dtype="<f4")
    mask = np.asarray(mask, dtype=bool)
    _validate_encoder_output(tokens, mask)
    tokens *= mask[..., None]

    quality_parts = []
    quality_report = {}
    for index, board in enumerate(prepared.episode.boards):
        role = "reference" if index == 0 else f"target_{index}"
        axes = "".join(axis.upper() for axis, present in zip("xyz", board.axis_mask, strict=True) if present)
        quality_parts.append(
            f"{role}: axes={axes or 'none'}, quality={'valid' if board.quality.is_valid else 'invalid'}"
        )
        quality_report[role] = {
            "valid": board.quality.is_valid,
            "valid_fraction": board.quality.valid_fraction,
            "missing_fraction": board.quality.missing_fraction,
            "clipped_fraction": (
                board.quality.clipped_fraction if board.metadata.get("clipping_measured") else None
            ),
            "axes_present": list(board.quality.axes_present),
            "usable_band_hz": board.quality.usable_band_hz,
            "packet_loss_count": (
                board.quality.packet_loss_count if board.metadata.get("packet_loss_measured") else None
            ),
            "clock_skew_ms": (
                board.quality.clock_skew_ms if board.metadata.get("clock_skew_measured") else None
            ),
            "orientation_calibrated": board.metadata.get("orientation_calibrated", False),
            "orientation_applied": board.metadata.get("orientation_applied", False),
            "notes": list(board.quality.notes),
        }
    return {
        "schema": "vibrogemma-live-embeddings-v1",
        "shape": list(tokens.shape[1:]),
        "dtype": "float32-le",
        "embeddings_b64": base64.b64encode(tokens[0].tobytes()).decode("ascii"),
        "soft_token_mask": mask[0].tolist(),
        "quality_text": "Signal quality metadata: " + "; ".join(quality_parts) + ".",
        "evidence_text": prepared.evidence_text,
        "window": {
            "start_utc": window_start_utc,
            "end_utc": window_end_utc,
            "duration_seconds": 10.0,
            "effective_duration_seconds": 10.0,
            "episode_id": prepared.episode.episode_id,
            "start_time_s": prepared.episode.boards[0].start_time_s,
            "time_source": window_time_source,
        },
        "quality": quality_report,
        "evidence": prepared.evidence,
        "metadata": {
            "amplitude_calibrated": amplitude_calibrated,
            "amplitude_scale_g": getattr(
                prepared,
                "amplitude_scale",
                getattr(prepared, "amplitude_scale_g", None),
            ),
            "phase_synchronized": prepared.synchronization_valid,
            "healthy_profile_available": preprocessor.physics_extractor.healthy_profile is not None,
            "healthy_profile_sha256": artifact_verification["sha256"]["healthy_profile"],
            "healthy_profile_keys": (prepared.evidence.get("healthy_profile_keys") or {}),
            "artifact_sha256": artifact_verification["sha256"],
            "artifact_runtime_sha256": artifact_verification["runtime_sha256"],
            "pipeline_source": artifact_verification["pipeline_source"],
            "pipeline_python_file_count": artifact_verification[
                "pipeline_python_file_count"
            ],
            "geometry_valid": bool(prepared.episode.metadata.get("geometry_valid")),
            "geometry_source": prepared.episode.metadata.get("geometry_source"),
            "orientation_valid_by_board": {
                ("reference" if index == 0 else f"target_{index}"): bool(board.metadata.get("orientation_calibrated"))
                for index, board in enumerate(prepared.episode.boards)
            },
            "window_start_utc": window_start_utc,
            "window_end_utc": window_end_utc,
            "window_time_source": window_time_source,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--amplitude-calibrated", action="store_true")
    args = parser.parse_args()
    with np.load(args.windows) as payload:
        result = encode_live_window(
            payload,
            args.bundle,
            amplitude_calibrated=args.amplitude_calibrated,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"encoded {result['shape']} VibroGemma soft tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
