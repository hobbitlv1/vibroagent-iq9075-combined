from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import requests
import yaml

from .hbta_v4_names import canonical_hbta_condition_family, publisher_root_group_id

ZENODO_RECORD_ID = 14632942
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
PUBLISHER_SAMPLE_RATE_HZ = 100.0
ANALYSIS_WINDOW_SAMPLES = 1024
ANALYSIS_WINDOW_DURATION_SECONDS = ANALYSIS_WINDOW_SAMPLES / PUBLISHER_SAMPLE_RATE_HZ
PUBLISHER_ROOT_COUNT = 50
ANALYSIS_WINDOWS_PER_ROOT = 16
ANALYSIS_WINDOW_COUNT = PUBLISHER_ROOT_COUNT * ANALYSIS_WINDOWS_PER_ROOT
EXPECTED_STATES = ("UDS", "DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS7", "DS8")
EXPECTED_ROOT_COUNTS = {"UDS": 10, **{f"DS{index}": 5 for index in range(1, 9)}}
EXPECTED_WINDOW_COUNTS = {
    state: roots * ANALYSIS_WINDOWS_PER_ROOT for state, roots in EXPECTED_ROOT_COUNTS.items()
}

# Compatibility aliases for older callers. These values describe derived analysis
# windows, not publisher-defined tests.
PUBLISHER_TEST_SAMPLES = ANALYSIS_WINDOW_SAMPLES
PUBLISHER_TEST_DURATION_SECONDS = ANALYSIS_WINDOW_DURATION_SECONDS
TESTS_PER_ROOT = ANALYSIS_WINDOWS_PER_ROOT
PUBLISHER_TEST_COUNT = ANALYSIS_WINDOW_COUNT
EXPECTED_TEST_COUNTS = EXPECTED_WINDOW_COUNTS

# Checksums published by the official Zenodo v4 record. Runtime metadata must
# agree with these values before any file is accepted.
EXPECTED_FILES: dict[str, str] = {
    "data_100Hz.h5": "md5:43406591cd7f8b8bd1b0b868d3312e38",
    "functions.py": "md5:847e39a27fffd03e100b609f54e1f1c9",
    "main.py": "md5:b4bb7f5676eec6d64f91f27f89e8c0d9",
    "Sensor layout.pdf": "md5:251668e09ff33f03c9ff19aa93c0b020",
    "Drawings.pdf": "md5:63d8de6902f4fc8f0e4abe22b02f9717",
}

PANELS: dict[str, dict[str, Any]] = {
    "al_global": {
        "reference": "AL01",
        "targets": ["AL08", "AL16", "AL24", "AL32", "AL40"],
        "allowed_axes": ["z"],
        "description": "Six documented AL deck locations retaining genuine vertical response only.",
    },
    "ag_local": {
        "reference": "AG01",
        "targets": ["AG04", "AG08", "AG12", "AG15", "AG18"],
        "allowed_axes": ["y", "z"],
        "description": "Six documented AG wall locations retaining genuine native y/z response.",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_file_map(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in record.get("files", []):
        key = str(item.get("key") or item.get("filename") or "")
        if key:
            result[key] = item
    return result


def fetch_official_record(*, timeout: float = 60.0) -> dict[str, Any]:
    response = requests.get(ZENODO_API_URL, timeout=timeout)
    response.raise_for_status()
    record = response.json()
    files = _record_file_map(record)
    missing = sorted(set(EXPECTED_FILES) - set(files))
    if missing:
        raise RuntimeError(f"Official HBTA v4 record lacks required files: {missing}")
    mismatched = {}
    for name, expected in EXPECTED_FILES.items():
        observed = str(files[name].get("checksum", "")).lower()
        if observed != expected.lower():
            mismatched[name] = {"expected": expected, "observed": observed}
    if mismatched:
        raise RuntimeError(f"Official HBTA v4 checksum metadata changed: {mismatched}")
    return record


def _download_url(item: Mapping[str, Any]) -> str:
    links = item.get("links", {})
    for key in ("content", "self", "download"):
        if links.get(key):
            return str(links[key])
    raise RuntimeError(f"Zenodo file entry has no usable download URL: {item}")


def _stream_download(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None,
    timeout: float = 120.0,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
        if existing and response.status_code == 200:
            partial.unlink(missing_ok=True)
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        downloaded = existing
        last_print = time.monotonic()
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_print > 15:
                    gib = downloaded / 1024**3
                    total = f"/{expected_bytes / 1024**3:.2f}" if expected_bytes else ""
                    print(f"[HBTA] {destination.name}: {gib:.2f}{total} GiB", flush=True)
                    last_print = time.monotonic()
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"HBTA download size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )
    os.replace(partial, destination)


def download_hbta_v4(
    destination: str | Path,
    *,
    reuse_candidates: Iterable[str | Path] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    record = fetch_official_record()
    files = _record_file_map(record)
    candidate_roots = list(dict.fromkeys(Path(item).resolve() for item in reuse_candidates))
    results = []
    for name, expected_checksum in EXPECTED_FILES.items():
        algorithm, expected_hex = expected_checksum.split(":", 1)
        target = destination / name
        source = "existing_verified"
        if not (target.exists() and not overwrite and checksum(target, algorithm) == expected_hex):
            target.unlink(missing_ok=True)
            source = "zenodo"
            for root in candidate_roots:
                candidates = [root / name]
                if root.exists():
                    candidates.extend(root.rglob(name))
                matched = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.is_file() and checksum(candidate, algorithm) == expected_hex
                    ),
                    None,
                )
                if matched is not None:
                    shutil.copy2(matched, target)
                    source = f"reused:{matched}"
                    break
            if not target.exists():
                item = files[name]
                _stream_download(
                    _download_url(item),
                    target,
                    expected_bytes=int(item.get("size")) if item.get("size") is not None else None,
                )
        observed = checksum(target, algorithm)
        if observed != expected_hex:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"HBTA checksum mismatch for {name}: {observed} != {expected_hex}")
        results.append(
            {
                "name": name,
                "path": str(target),
                "size_bytes": target.stat().st_size,
                "checksum": f"{algorithm}:{observed}",
                "source": source,
            }
        )
    report = {
        "record_id": ZENODO_RECORD_ID,
        "record_url": f"https://zenodo.org/records/{ZENODO_RECORD_ID}",
        "files": results,
    }
    (destination / "verified_download_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _group_has_acceleration(group: h5py.Group) -> bool:
    return any(
        isinstance(item, h5py.Group) and name.lower() == "acceleration"
        for name, item in group.items()
    )


def _recording_roots(handle: h5py.File) -> list[h5py.Group]:
    direct = [
        handle[name]
        for name in handle
        if isinstance(handle[name], h5py.Group) and _group_has_acceleration(handle[name])
    ]
    if len(direct) == PUBLISHER_ROOT_COUNT:
        return sorted(direct, key=lambda group: _natural_key(group.name))

    found: list[h5py.Group] = []

    def visitor(_: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group) and _group_has_acceleration(obj):
            found.append(obj)

    handle.visititems(visitor)
    found_names = {group.name for group in found}
    deepest = [
        group
        for group in found
        if not any(
            name != group.name and name.startswith(group.name.rstrip("/") + "/")
            for name in found_names
        )
    ]
    return sorted(deepest, key=lambda group: _natural_key(group.name))


def _acceleration_group(recording: h5py.Group) -> h5py.Group:
    for name, item in recording.items():
        if isinstance(item, h5py.Group) and name.lower() == "acceleration":
            return item
    raise RuntimeError(f"Recording has no acceleration group: {recording.name}")


def _text(value: Any) -> str:
    return str(_jsonable(value)).strip()


def _raw_state(recording: h5py.Group) -> str:
    for key, value in recording.attrs.items():
        if any(token in key.lower() for token in ("damage", "state", "condition")):
            text = _text(value)
            if text:
                try:
                    return canonical_hbta_condition_family(text)
                except ValueError:
                    pass
    try:
        return canonical_hbta_condition_family(recording.name)
    except ValueError as exc:
        raise RuntimeError(f"Cannot determine HBTA state for {recording.name}") from exc


def _normalise_state(raw: str) -> str:
    try:
        return canonical_hbta_condition_family(raw)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported HBTA state label: {raw!r}") from exc


def _sensor_axis_map(acceleration: h5py.Group) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        upper = name.upper()
        sensor_match = re.search(r"(A[GLS]\d{2})", upper)
        if not sensor_match:
            return
        sensor = sensor_match.group(1)
        tokens = [token for token in re.split(r"[/_.\-]+", name.lower()) if token]
        axis = next((token for token in reversed(tokens) if token in {"x", "y", "z"}), None)
        if axis is None:
            if sensor.startswith("AL"):
                axis = "z"
            else:
                return
        key = (sensor, axis)
        if key in result and result[key] != obj.name:
            raise RuntimeError(
                f"Duplicate HBTA channel mapping for {sensor}/{axis}: {result[key]} and {obj.name}"
            )
        result[key] = obj.name

    acceleration.visititems(visitor)
    return result


def _dataset_sample_count(dataset: h5py.Dataset) -> int:
    shape = tuple(int(value) for value in dataset.shape if int(value) != 1)
    if len(shape) != 1:
        raise RuntimeError(
            f"Expected a continuous one-dimensional HBTA response for {dataset.name}; "
            f"observed shape {dataset.shape}"
        )
    return shape[0]


def _dataset_vector(dataset: h5py.Dataset) -> np.ndarray:
    array = np.asarray(dataset[...])
    array = np.squeeze(array)
    if array.ndim != 1:
        raise RuntimeError(
            f"Expected a continuous one-dimensional HBTA response for {dataset.name}; "
            f"observed shape {array.shape}"
        )
    vector = np.asarray(array, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise RuntimeError(f"HBTA response contains non-finite samples: {dataset.name}")
    return vector


def _analysis_window_starts(
    sample_count: int,
    *,
    windows_per_root: int = ANALYSIS_WINDOWS_PER_ROOT,
    window_samples: int = ANALYSIS_WINDOW_SAMPLES,
) -> tuple[int, ...]:
    """Select one centred, non-overlapping window from each equal-time stratum.

    The official HDF stores each recording root as one continuous response. This
    policy gives every root the same downstream weight while covering the whole
    recording and keeping all windows within a single publisher recording root.
    """

    sample_count = int(sample_count)
    if sample_count < windows_per_root * window_samples:
        raise RuntimeError(
            f"HBTA recording is too short for {windows_per_root} non-overlapping "
            f"{window_samples}-sample windows: {sample_count} samples"
        )
    edges = np.linspace(0, sample_count, windows_per_root + 1, dtype=np.int64)
    starts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        width = int(right - left)
        if width < window_samples:
            raise RuntimeError(
                f"HBTA equal-time stratum is shorter than one analysis window: {width}"
            )
        starts.append(int(left + (width - window_samples) // 2))
    for first, second in zip(starts, starts[1:]):
        if first + window_samples > second:
            raise RuntimeError(f"HBTA derived windows overlap: {starts}")
    if starts[-1] + window_samples > sample_count:
        raise RuntimeError(f"HBTA final window exceeds the recording: {starts[-1]}, {sample_count}")
    return tuple(starts)


def _analysis_window_matrix(
    vector: np.ndarray,
    starts: Iterable[int],
    *,
    source_name: str,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    starts = tuple(int(value) for value in starts)
    windows = []
    for start in starts:
        stop = start + ANALYSIS_WINDOW_SAMPLES
        if start < 0 or stop > vector.size:
            raise RuntimeError(
                f"HBTA window [{start}:{stop}] exceeds {source_name} with "
                f"{vector.size} samples"
            )
        windows.append(vector[start:stop])
    matrix = np.stack(windows, axis=0).astype(np.float32, copy=False)
    expected_shape = (len(starts), ANALYSIS_WINDOW_SAMPLES)
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"HBTA analysis-window extraction failed for {source_name}: "
            f"{matrix.shape} != {expected_shape}"
        )
    return matrix


def _state_assignment(roots: list[h5py.Group]) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[h5py.Group]] = {state: [] for state in EXPECTED_STATES}
    for root in roots:
        state = _normalise_state(_raw_state(root))
        grouped.setdefault(state, []).append(root)

    observed = {state for state, items in grouped.items() if items}
    if observed != set(EXPECTED_STATES):
        raise RuntimeError(f"HBTA state mismatch: {sorted(observed)} != {list(EXPECTED_STATES)}")

    root_counts = {state: len(grouped[state]) for state in EXPECTED_STATES}
    if root_counts != EXPECTED_ROOT_COUNTS:
        raise RuntimeError(
            f"HBTA publisher-root counts mismatch: {root_counts} != {EXPECTED_ROOT_COUNTS}"
        )

    assignment = {
        root.name: state
        for state in EXPECTED_STATES
        for root in grouped[state]
    }
    return assignment, {
        "root_counts": root_counts,
        "state_policy": "one official UDS condition plus DS1 through DS8",
        "uds_substate_inference": None,
    }


def inspect_source_hdf(source_h5: str | Path, *, panel_name: str) -> dict[str, Any]:
    source_h5 = Path(source_h5)
    panel = PANELS[panel_name]
    sensor_ids = [panel["reference"], *panel["targets"]]
    channel_sample_counts: dict[str, list[int]] = {}
    recording_rows: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as handle:
        roots = _recording_roots(handle)
        if len(roots) != PUBLISHER_ROOT_COUNT:
            raise RuntimeError(
                f"Expected {PUBLISHER_ROOT_COUNT} HBTA recording roots; found {len(roots)}"
            )
        state_assignment, state_report = _state_assignment(roots)
        first_root_channels: dict[str, list[str]] | None = None
        for root_index, root in enumerate(roots):
            acceleration = _acceleration_group(root)
            mapping = _sensor_axis_map(acceleration)
            per_sensor: dict[str, list[str]] = {}
            per_channel_lengths: dict[str, int] = {}
            for sensor in sensor_ids:
                missing = [axis for axis in panel["allowed_axes"] if (sensor, axis) not in mapping]
                if missing:
                    raise RuntimeError(
                        f"Panel {panel_name} sensor {sensor} is missing genuine axes {missing} "
                        f"in {root.name}"
                    )
                axes = list(panel["allowed_axes"])
                per_sensor[sensor] = axes
                for axis in axes:
                    dataset = handle[mapping[(sensor, axis)]]
                    count = _dataset_sample_count(dataset)
                    key = f"{sensor}/{axis}"
                    per_channel_lengths[key] = count
                    channel_sample_counts.setdefault(key, []).append(count)
            unique_lengths = sorted(set(per_channel_lengths.values()))
            if len(unique_lengths) != 1:
                raise RuntimeError(
                    f"HBTA selected channels are not sample-aligned in {root.name}: "
                    f"{per_channel_lengths}"
                )
            sample_count = unique_lengths[0]
            starts = _analysis_window_starts(sample_count)
            recording_rows.append(
                {
                    "publisher_root": root.name,
                    "state": state_assignment[root.name],
                    "sample_count": sample_count,
                    "duration_seconds": sample_count / PUBLISHER_SAMPLE_RATE_HZ,
                    "analysis_window_starts": list(starts),
                    "analysis_window_stops": [
                        start + ANALYSIS_WINDOW_SAMPLES for start in starts
                    ],
                }
            )
            if first_root_channels is None:
                first_root_channels = per_sensor
            if (root_index + 1) % 10 == 0:
                print(
                    f"[HBTA] inspected {root_index + 1}/{len(roots)} continuous roots "
                    f"for {panel_name}",
                    flush=True,
                )
    shape_summary = {
        key: {
            "recordings": len(values),
            "sample_count_min": min(values),
            "sample_count_max": max(values),
            "distinct_sample_counts": sorted(set(values)),
        }
        for key, values in sorted(channel_sample_counts.items())
    }
    return {
        "source_h5": str(source_h5),
        "panel_name": panel_name,
        "panel": panel,
        "publisher_recording_root_count": PUBLISHER_ROOT_COUNT,
        "analysis_windows_per_recording_root": ANALYSIS_WINDOWS_PER_ROOT,
        "analysis_window_count": ANALYSIS_WINDOW_COUNT,
        "analysis_window_samples": ANALYSIS_WINDOW_SAMPLES,
        "analysis_window_duration_seconds": ANALYSIS_WINDOW_DURATION_SECONDS,
        "analysis_window_policy": "one centred window from each of 16 equal-time strata",
        "states": list(EXPECTED_STATES),
        "expected_root_counts": EXPECTED_ROOT_COUNTS,
        "expected_window_counts": EXPECTED_WINDOW_COUNTS,
        **state_report,
        "first_root_channels": first_root_channels or {},
        "observed_source_sample_counts": shape_summary,
        "source_recordings": recording_rows,
        "source_layout": "one continuous response vector per recording root and channel",
    }


def normalize_panel_hdf(
    source_h5: str | Path,
    output_h5: str | Path,
    *,
    panel_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_h5 = Path(source_h5)
    output_h5 = Path(output_h5)
    report_path = output_h5.with_suffix(output_h5.suffix + ".audit.json")
    if output_h5.exists() and report_path.exists() and not overwrite:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema") == "hbta-v4-verified-normalization-v3"
            and report.get("passed") is True
            and report.get("analysis_window_count") == ANALYSIS_WINDOW_COUNT
            and report.get("state_window_counts") == EXPECTED_WINDOW_COUNTS
        ):
            return report

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".building")
    temporary.unlink(missing_ok=True)
    panel = PANELS[panel_name]
    sensor_ids = [panel["reference"], *panel["targets"]]
    source_report = inspect_source_hdf(source_h5, panel_name=panel_name)
    source_sha256 = checksum(source_h5, "sha256")
    rows: list[dict[str, Any]] = []

    try:
        with h5py.File(source_h5, "r") as source, h5py.File(temporary, "w") as target:
            roots = _recording_roots(source)
            state_assignment, state_report = _state_assignment(roots)
            target.attrs["dataset_id"] = f"hell_bridge_v4_verified_windows_{panel_name}"
            target.attrs["source_sha256"] = source_sha256
            target.attrs["publisher_record_id"] = ZENODO_RECORD_ID
            target.attrs["panel_name"] = panel_name
            target.attrs["normalization_schema"] = "hbta-v4-verified-normalization-v3"
            target.attrs["source_layout"] = "continuous_recording_vectors"
            target.attrs["analysis_window_policy"] = "equal_time_strata_centered_v1"

            for root_index, root in enumerate(roots):
                state = state_assignment[root.name]
                acceleration = _acceleration_group(root)
                mapping = _sensor_axis_map(acceleration)
                root_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", root.name.strip("/"))
                vectors: dict[tuple[str, str], np.ndarray] = {}
                lengths: dict[str, int] = {}
                for sensor in sensor_ids:
                    for axis in panel["allowed_axes"]:
                        path = mapping.get((sensor, axis))
                        if path is None:
                            raise RuntimeError(
                                f"Panel {panel_name} sensor {sensor} is missing {axis} in {root.name}"
                            )
                        vector = _dataset_vector(source[path])
                        vectors[(sensor, axis)] = vector
                        lengths[f"{sensor}/{axis}"] = int(vector.size)
                unique_lengths = sorted(set(lengths.values()))
                if len(unique_lengths) != 1:
                    raise RuntimeError(
                        f"HBTA selected channels are not sample-aligned in {root.name}: {lengths}"
                    )
                sample_count = unique_lengths[0]
                starts = _analysis_window_starts(sample_count)
                matrices = {
                    key: _analysis_window_matrix(
                        vector, starts, source_name=mapping[key]
                    )
                    for key, vector in vectors.items()
                }

                for window_index, start_sample in enumerate(starts, start=1):
                    stop_sample = start_sample + ANALYSIS_WINDOW_SAMPLES
                    group_name = f"{state}__{root_slug}__window_{window_index:02d}"
                    recording = target.create_group(group_name)
                    for key, value in root.attrs.items():
                        try:
                            recording.attrs[key] = value
                        except (TypeError, ValueError):
                            recording.attrs[key] = _text(value)
                    recording.attrs["damage_state"] = state
                    recording.attrs["publisher_state"] = state
                    recording.attrs["publisher_root"] = root.name
                    recording.attrs["analysis_window_index"] = window_index
                    recording.attrs["source_start_sample"] = start_sample
                    recording.attrs["source_stop_sample"] = stop_sample
                    recording.attrs["source_start_time_s"] = start_sample / PUBLISHER_SAMPLE_RATE_HZ
                    recording.attrs["source_stop_time_s"] = stop_sample / PUBLISHER_SAMPLE_RATE_HZ
                    recording.attrs["source_recording_sample_count"] = sample_count
                    recording.attrs["sample_rate_hz"] = PUBLISHER_SAMPLE_RATE_HZ
                    recording.attrs["duration_seconds"] = ANALYSIS_WINDOW_DURATION_SECONDS
                    recording.attrs["phase_synchronized"] = False
                    recording.attrs["verification"] = (
                        "hbta_v4_continuous_recording_equal_time_windows_v1"
                    )
                    recording.attrs["window_selection_policy"] = "equal_time_strata_centered_v1"
                    recording.attrs["split_group"] = publisher_root_group_id(panel_name, root.name)
                    out_acc = recording.create_group("acceleration")
                    axes_by_sensor: dict[str, list[str]] = {}
                    for sensor in sensor_ids:
                        sensor_group = out_acc.create_group(sensor)
                        axes: list[str] = []
                        for axis in panel["allowed_axes"]:
                            matrix = matrices[(sensor, axis)]
                            sensor_group.create_dataset(
                                axis,
                                data=matrix[window_index - 1],
                                compression="gzip",
                                compression_opts=4,
                                shuffle=True,
                            )
                            axes.append(axis)
                        axes_by_sensor[sensor] = axes
                    rows.append(
                        {
                            "group": group_name,
                            "state": state,
                            "healthy": state == "UDS",
                            "publisher_root": root.name,
                            "analysis_window_index": window_index,
                            "source_start_sample": start_sample,
                            "source_stop_sample": stop_sample,
                            "split_group": publisher_root_group_id(panel_name, root.name),
                            "sensors": axes_by_sensor,
                        }
                    )
                print(
                    f"[HBTA] normalized {root_index + 1}/{len(roots)} continuous roots "
                    f"({len(rows)}/{ANALYSIS_WINDOW_COUNT} analysis windows) for {panel_name}",
                    flush=True,
                )
            target.attrs["analysis_window_count"] = len(rows)
            target.attrs["analysis_windows_per_recording_root"] = ANALYSIS_WINDOWS_PER_ROOT
        os.replace(temporary, output_h5)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    state_window_counts = {
        state: sum(row["state"] == state for row in rows) for state in EXPECTED_STATES
    }
    if len(rows) != ANALYSIS_WINDOW_COUNT or state_window_counts != EXPECTED_WINDOW_COUNTS:
        output_h5.unlink(missing_ok=True)
        raise RuntimeError(
            f"HBTA analysis-window count failure: total={len(rows)}, "
            f"states={state_window_counts}"
        )

    report = {
        "schema": "hbta-v4-verified-normalization-v3",
        "passed": True,
        "source_h5": str(source_h5),
        "source_sha256": source_sha256,
        "normalized_h5": str(output_h5),
        "normalized_sha256": checksum(output_h5, "sha256"),
        "panel_name": panel_name,
        "panel": panel,
        "publisher_recording_root_count": PUBLISHER_ROOT_COUNT,
        "analysis_windows_per_recording_root": ANALYSIS_WINDOWS_PER_ROOT,
        "analysis_window_count": len(rows),
        "root_counts": EXPECTED_ROOT_COUNTS,
        "state_window_counts": state_window_counts,
        "sample_rate_hz": PUBLISHER_SAMPLE_RATE_HZ,
        "samples_per_analysis_window": ANALYSIS_WINDOW_SAMPLES,
        "analysis_window_duration_seconds": ANALYSIS_WINDOW_DURATION_SECONDS,
        "analysis_window_policy": "one centred window from each of 16 equal-time strata",
        "source_layout": "one continuous response vector per recording root and channel",
        "phase_synchronized": False,
        "source_recording_boundary_crossing_possible": False,
        "derived_window_overlap": False,
        "publisher_root_grouping": True,
        "state_assignment": state_report,
        "source_inspection": source_report,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_normalized_hdf(path: str | Path, *, panel_name: str) -> dict[str, Any]:
    path = Path(path)
    panel = PANELS[panel_name]
    sensors = [panel["reference"], *panel["targets"]]
    state_window_counts = {state: 0 for state in EXPECTED_STATES}
    root_counts: dict[str, int] = {}
    root_states: dict[str, set[str]] = {}
    windows_by_root: dict[str, list[tuple[int, int, int, int]]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    publisher_source_sha256 = ""
    with h5py.File(path, "r") as handle:
        expected_dataset_id = f"hell_bridge_v4_verified_windows_{panel_name}"
        required_attrs = {
            "dataset_id": expected_dataset_id,
            "panel_name": panel_name,
            "normalization_schema": "hbta-v4-verified-normalization-v3",
            "source_layout": "continuous_recording_vectors",
            "analysis_window_policy": "equal_time_strata_centered_v1",
        }
        for key, expected in required_attrs.items():
            observed = _text(handle.attrs.get(key, ""))
            if observed != expected:
                raise RuntimeError(
                    f"Normalized HBTA root attribute mismatch for {key}: "
                    f"{observed!r} != {expected!r}"
                )
        publisher_source_sha256 = _text(handle.attrs.get("source_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", publisher_source_sha256) is None:
            raise RuntimeError("Normalized HBTA file lacks a valid publisher source SHA-256")
        if int(handle.attrs.get("publisher_record_id", -1)) != ZENODO_RECORD_ID:
            raise RuntimeError("Normalized HBTA publisher record ID mismatch")
        if int(handle.attrs.get("analysis_window_count", -1)) != ANALYSIS_WINDOW_COUNT:
            raise RuntimeError("Normalized HBTA root analysis-window count mismatch")
        if (
            int(handle.attrs.get("analysis_windows_per_recording_root", -1))
            != ANALYSIS_WINDOWS_PER_ROOT
        ):
            raise RuntimeError("Normalized HBTA root per-recording window count mismatch")

        groups = [handle[name] for name in handle if isinstance(handle[name], h5py.Group)]
        if len(groups) != ANALYSIS_WINDOW_COUNT:
            raise RuntimeError(
                f"Normalized HBTA file must have {ANALYSIS_WINDOW_COUNT} analysis-window "
                f"groups; found {len(groups)}"
            )
        for group in groups:
            state = _normalise_state(_text(group.attrs.get("publisher_state", "")))
            state_window_counts[state] += 1
            publisher_root = _text(group.attrs.get("publisher_root", ""))
            if not publisher_root:
                raise RuntimeError(f"Normalized HBTA group lacks publisher_root: {group.name}")
            window_index = int(group.attrs.get("analysis_window_index", -1))
            start_sample = int(group.attrs.get("source_start_sample", -1))
            stop_sample = int(group.attrs.get("source_stop_sample", -1))
            source_sample_count = int(group.attrs.get("source_recording_sample_count", -1))
            pair = (publisher_root, window_index)
            if pair in seen_pairs:
                raise RuntimeError(f"Duplicate HBTA analysis window: {pair}")
            seen_pairs.add(pair)
            if (
                stop_sample - start_sample != ANALYSIS_WINDOW_SAMPLES
                or start_sample < 0
                or stop_sample > source_sample_count
            ):
                raise RuntimeError(
                    f"Invalid HBTA source interval in {group.name}: "
                    f"[{start_sample}:{stop_sample}] of {source_sample_count}"
                )
            if _text(group.attrs.get("window_selection_policy", "")) != "equal_time_strata_centered_v1":
                raise RuntimeError(f"HBTA window policy mismatch in {group.name}")
            windows_by_root.setdefault(publisher_root, []).append(
                (window_index, start_sample, stop_sample, source_sample_count)
            )
            root_counts[publisher_root] = root_counts.get(publisher_root, 0) + 1
            root_states.setdefault(publisher_root, set()).add(state)
            expected_group = publisher_root_group_id(panel_name, publisher_root)
            observed_group = _text(group.attrs.get("split_group", ""))
            if observed_group != expected_group:
                raise RuntimeError(
                    f"HBTA split group mismatch for {group.name}: "
                    f"{observed_group!r} != {expected_group!r}"
                )
            acceleration = _acceleration_group(group)
            mapping = _sensor_axis_map(acceleration)
            for sensor in sensors:
                missing = [axis for axis in panel["allowed_axes"] if (sensor, axis) not in mapping]
                if missing:
                    raise RuntimeError(
                        f"Normalized panel sensor {sensor} is missing {missing} in {group.name}"
                    )
                for axis in panel["allowed_axes"]:
                    data = np.asarray(handle[mapping[(sensor, axis)]][...])
                    if data.shape != (ANALYSIS_WINDOW_SAMPLES,) or not np.isfinite(data).all():
                        raise RuntimeError(
                            f"Invalid normalized response {mapping[(sensor, axis)]}: {data.shape}"
                        )

    observed_root_counts_by_state = {state: 0 for state in EXPECTED_STATES}
    for publisher_root, windows in windows_by_root.items():
        states = root_states[publisher_root]
        if len(states) != 1:
            raise RuntimeError(f"HBTA root has inconsistent state labels: {publisher_root}: {states}")
        observed_root_counts_by_state[next(iter(states))] += 1
        ordered = sorted(windows)
        indices = [item[0] for item in ordered]
        if indices != list(range(1, ANALYSIS_WINDOWS_PER_ROOT + 1)):
            raise RuntimeError(
                f"HBTA analysis-window index mismatch for {publisher_root}: {indices}"
            )
        sample_counts = {item[3] for item in ordered}
        if len(sample_counts) != 1:
            raise RuntimeError(
                f"HBTA source sample-count mismatch within {publisher_root}: {sample_counts}"
            )
        source_sample_count = next(iter(sample_counts))
        expected_starts = list(_analysis_window_starts(source_sample_count))
        observed_starts = [item[1] for item in ordered]
        if observed_starts != expected_starts:
            raise RuntimeError(
                f"HBTA deterministic-window mismatch for {publisher_root}: "
                f"{observed_starts} != {expected_starts}"
            )
        by_time = sorted(ordered, key=lambda item: item[1])
        for first, second in zip(by_time, by_time[1:]):
            if first[2] > second[1]:
                raise RuntimeError(
                    f"HBTA analysis windows overlap in {publisher_root}: {first}, {second}"
                )

    per_root_ok = (
        len(root_counts) == PUBLISHER_ROOT_COUNT
        and set(root_counts.values()) == {ANALYSIS_WINDOWS_PER_ROOT}
    )
    passed = (
        len(seen_pairs) == ANALYSIS_WINDOW_COUNT
        and state_window_counts == EXPECTED_WINDOW_COUNTS
        and observed_root_counts_by_state == EXPECTED_ROOT_COUNTS
        and per_root_ok
    )
    report = {
        "schema": "hbta-v4-verified-audit-v3",
        "passed": passed,
        "path": str(path),
        "sha256": checksum(path, "sha256"),
        "publisher_source_sha256": publisher_source_sha256,
        "panel_name": panel_name,
        "analysis_window_count": len(seen_pairs),
        "publisher_recording_root_count": len(root_counts),
        "windows_per_recording_root_values": sorted(set(root_counts.values())),
        "root_counts_by_state": observed_root_counts_by_state,
        "state_window_counts": state_window_counts,
        "source_recording_boundary_crossing_possible": False,
        "derived_window_overlap": False,
        "publisher_root_grouping": True,
        "analysis_window_policy": "equal_time_strata_centered_v1",
        "deterministic_window_starts_verified": True,
    }
    if not passed:
        raise RuntimeError(f"Normalized HBTA audit failed: {report}")
    return report


def _relative_source(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def auto_index_verified_hbta(manifest: Any, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one index entry per deterministic derived analysis window."""

    root = manifest.raw_root(config)
    source_name = str(manifest.reader.get("source_filename", ""))
    candidates = sorted(path for path in root.rglob(source_name) if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(f"{manifest.id}: expected exactly one {source_name}, found {candidates}")
    source_path = candidates[0]
    source_file = _relative_source(root, source_path)
    source_sha256 = checksum(source_path, "sha256")
    panel_name = str(manifest.reader.get("panel_name") or "")
    if panel_name not in PANELS:
        panel_name = "ag_local" if manifest.id.endswith("ag_local") else "al_global"
    normalized_audit = audit_normalized_hdf(source_path, panel_name=panel_name)
    publisher_source_sha256 = str(normalized_audit["publisher_source_sha256"])
    panel = PANELS[panel_name]
    source_sensors = [panel["reference"], *panel["targets"]]
    board_ids = ["reference", "target_1", "target_2", "target_3", "target_4", "target_5"]
    entries: list[dict[str, Any]] = []
    state_window_counts = {state: 0 for state in EXPECTED_STATES}
    root_counts: dict[str, int] = {}

    with h5py.File(source_path, "r") as handle:
        groups = sorted(
            [handle[name] for name in handle if isinstance(handle[name], h5py.Group)],
            key=lambda group: _natural_key(group.name),
        )
        if len(groups) != ANALYSIS_WINDOW_COUNT:
            raise RuntimeError(
                f"{manifest.id}: normalized HDF must contain "
                f"{ANALYSIS_WINDOW_COUNT} analysis-window groups"
            )
        for group in groups:
            state = _normalise_state(_text(group.attrs.get("publisher_state", "")))
            publisher_root = _text(group.attrs.get("publisher_root", ""))
            window_index = int(group.attrs.get("analysis_window_index", -1))
            start_sample = int(group.attrs.get("source_start_sample", -1))
            stop_sample = int(group.attrs.get("source_stop_sample", -1))
            if (
                not publisher_root
                or not 1 <= window_index <= ANALYSIS_WINDOWS_PER_ROOT
                or start_sample < 0
                or stop_sample - start_sample != ANALYSIS_WINDOW_SAMPLES
            ):
                raise RuntimeError(f"{manifest.id}: invalid analysis-window identity in {group.name}")
            split_group = publisher_root_group_id(panel_name, publisher_root)
            acceleration = _acceleration_group(group)
            mapping = _sensor_axis_map(acceleration)
            boards = []
            for board_id, source_sensor in zip(board_ids, source_sensors, strict=True):
                missing = [axis for axis in panel["allowed_axes"] if (source_sensor, axis) not in mapping]
                if missing:
                    raise RuntimeError(
                        f"{manifest.id}: sensor {source_sensor} is missing {missing} in {group.name}"
                    )
                axis_paths = {
                    axis: mapping[(source_sensor, axis)] for axis in panel["allowed_axes"]
                }
                boards.append(
                    {
                        "sensor_id": board_id,
                        "source_sensor_name": source_sensor,
                        "axis_paths": axis_paths,
                        "axes_available": [
                            axis for axis in ("x", "y", "z") if axis in axis_paths
                        ],
                        "sensor_position": None,
                        "orientation_matrix": None,
                    }
                )
            experiment_id = group.name.strip("/").replace("/", "::")
            session_id = publisher_root.strip("/").replace("/", "::")
            healthy = state == "UDS"
            entries.append(
                {
                    "dataset_id": manifest.id,
                    "source_file": source_file,
                    "source_sha256": source_sha256,
                    "publisher_source_sha256": publisher_source_sha256,
                    "structure_id": "hell_bridge",
                    "session_id": session_id,
                    "experiment_id": experiment_id,
                    "sample_rate_hz": PUBLISHER_SAMPLE_RATE_HZ,
                    "unit": "m/s^2",
                    "start_time_s": start_sample / PUBLISHER_SAMPLE_RATE_HZ,
                    "condition": state,
                    "publisher_state": state,
                    "publisher_root": publisher_root,
                    "analysis_window_index": window_index,
                    "source_start_sample": start_sample,
                    "source_stop_sample": stop_sample,
                    "panel_id": panel_name,
                    "split_group": split_group,
                    "split_stratum": (
                        "healthy" if healthy else f"changed::unknown_anomaly::{state}"
                    ),
                    "training_role": "supervised",
                    "healthy": healthy,
                    "severity": "none" if healthy else "advisory",
                    "class_name": "normal" if healthy else "unknown_anomaly",
                    "affected_targets": [],
                    "phase_synchronized": False,
                    "synchronization_verification_id": None,
                    "sensor_useful_band_hz": PUBLISHER_SAMPLE_RATE_HZ / 2.0,
                    "boards": boards,
                    "auto_index": {
                        "adapter": "hell_bridge_v4_verified_windows_v2",
                        "verification": (
                            "one normalized HDF group per deterministic derived analysis window"
                        ),
                        "window_selection_policy": "equal_time_strata_centered_v1",
                        "publisher_root_split_group": split_group,
                        "label_basis": "official UDS or DS1-DS8 recording-root condition token",
                        "localisation": "global_unknown",
                    },
                }
            )
            state_window_counts[state] += 1
            root_counts[publisher_root] = root_counts.get(publisher_root, 0) + 1

    if len(entries) != ANALYSIS_WINDOW_COUNT or state_window_counts != EXPECTED_WINDOW_COUNTS:
        raise RuntimeError(
            f"{manifest.id}: index count mismatch total={len(entries)}, "
            f"states={state_window_counts}"
        )
    if (
        len(root_counts) != PUBLISHER_ROOT_COUNT
        or set(root_counts.values()) != {ANALYSIS_WINDOWS_PER_ROOT}
    ):
        raise RuntimeError(f"{manifest.id}: publisher-root window audit failed")
    return entries, {
        "source_record": ZENODO_RECORD_ID,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "publisher_source_sha256": publisher_source_sha256,
        "normalized_audit": normalized_audit,
        "panel_name": panel_name,
        "analysis_window_count": len(entries),
        "publisher_recording_root_count": len(root_counts),
        "analysis_windows_per_recording_root": ANALYSIS_WINDOWS_PER_ROOT,
        "state_window_counts": state_window_counts,
        "root_counts_by_state": EXPECTED_ROOT_COUNTS,
        "split_group_policy": (
            "all sixteen derived windows from one publisher recording root share session_id"
        ),
        "window_selection_policy": "equal_time_strata_centered_v1",
        "source_recording_boundary_crossing_possible": False,
        "derived_window_overlap": False,
        "rejected_records": [],
    }


def write_panel_ledger(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "publisher_record_id": ZENODO_RECORD_ID,
        "source_document": "Sensor layout.pdf and official functions.py",
        "panels": PANELS,
        "axis_policy": (
            "Preserve genuine axes only. AL is retained as Z-only; AG retains genuine native y/z. "
            "No missing axis is synthesized."
        ),
        "source_layout_policy": (
            "Each official HDF recording root contains one continuous response vector per channel. "
            "The normalizer derives sixteen 1,024-sample analysis windows by taking one centred "
            "window from each equal-time stratum."
        ),
        "condition_policy": (
            "One official undamaged condition UDS (ten recording roots, 160 derived windows) and "
            "DS1-DS8 (five roots and 80 derived windows each)."
        ),
        "split_policy": (
            "All derived windows from one publisher recording root remain in one split group."
        ),
        "synchronization_policy": (
            "Cross-phase and phase-sensitive coherence remain disabled until inter-controller "
            "phase synchronization is independently verified."
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path

