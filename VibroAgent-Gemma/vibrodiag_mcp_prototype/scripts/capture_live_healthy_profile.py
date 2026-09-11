#!/usr/bin/env python3
"""Fit a VibroGemma healthy profile from two separated live time blocks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path

import numpy as np

from vibroagent_mcp.sdk_vibrometer import read_sdk_vibrometer_window

from live_vibrogemma_worker import SLOTS, _episode, _load_contracts, _runtime_config, _preserve_acquisition_quality

logging.getLogger("HSDatalogApp").setLevel(logging.WARNING)
logging.disable(logging.INFO)


def _devices(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8")).get("devices") or []
    devices = {str(record.get("sensor_id")): record for record in records}
    if set(devices) != set(SLOTS):
        raise ValueError(f"device manifest must contain exactly {list(SLOTS)}")
    return devices


def _read_axis(device: dict, axis: str, *, start_s: float | None, duration_s: float):
    return read_sdk_vibrometer_window(
        machine_id=str(device["sensor_id"]),
        acquisition_folder=str(device["folder"]),
        sensor_name="iis3dwb_acc",
        axis=axis,
        start_time_s=start_s,
        duration_s=duration_s,
        require_current=False,
        max_duration_s=60.0,
    )


def _latest_starts(devices: dict[str, dict]) -> dict[str, float]:
    def read(slot: str):
        window = _read_axis(devices[slot], "x", start_s=None, duration_s=10.0)
        return slot, float(window.metadata["start_time_s"])

    with ThreadPoolExecutor(max_workers=3) as executor:
        return dict(executor.map(read, SLOTS))


def _read_block(
    devices: dict[str, dict], starts: dict[str, float], duration_s: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    def read(slot: str):
        axes = [_read_axis(devices[slot], axis, start_s=starts[slot], duration_s=duration_s + 1.0) for axis in "xyz"]
        rates = {float(window.sampling_rate_hz) for window in axes}
        if len(rates) != 1:
            raise ValueError(f"{slot}: XYZ sampling rates differ: {sorted(rates)}")
        rate = rates.pop()
        count = int(round(duration_s * rate))
        if any(window.signal.size < count for window in axes):
            raise ValueError(f"{slot}: captured block is shorter than {duration_s:g} seconds")
        return slot, np.stack([np.asarray(window.signal[:count], dtype=np.float32) for window in axes]), rate

    arrays: dict[str, np.ndarray] = {}
    rates: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for slot, values, rate in executor.map(read, SLOTS):
            arrays[slot] = values
            rates[slot] = rate
    return arrays, rates


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    repo = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(os.environ.get("VIBROGEMMA_BUNDLE", repo / "models" / "vibroagent-gemma-g1")))
    parser.add_argument("--devices", type=Path, default=repo / "stdatalog_examples" / "vibroagent_live_devices.json")
    parser.add_argument("--output", type=Path, default=project / "configs" / "vibroagent_live_healthy_profile.json")
    parser.add_argument("--older-age-s", type=float, default=300.0)
    parser.add_argument("--recent-age-s", type=float, default=60.0)
    parser.add_argument("--windows-per-group", type=int, default=4)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--amplitude-calibrated", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()

    if args.windows_per_group < 4 or args.window_s != 10.0 or args.stride_s < args.window_s:
        parser.error("use at least four non-overlapping windows per group")
    block_s = args.window_s + (args.windows_per_group - 1) * args.stride_s
    if block_s > 60.0 or args.older_age_s <= args.recent_age_s + block_s:
        parser.error("capture blocks must be <=60 s and separated in time")

    devices = _devices(args.devices.resolve())
    latest = _latest_starts(devices)
    groups = (("older", args.older_age_s), ("recent", args.recent_age_s))
    starts_by_group = {
        name: {slot: latest[slot] - age_s - block_s for slot in SLOTS}
        for name, age_s in groups
    }
    if min(start for starts in starts_by_group.values() for start in starts.values()) < 0:
        raise RuntimeError("recordings are not yet long enough for two separated healthy blocks")

    BoardWindow, *_, MultiResolutionEpisodePreprocessor = _load_contracts(bundle)
    config = _runtime_config(bundle)
    physics = config["model"]["multiresolution"]["physics"]
    physics["healthy_profile_path"] = None
    physics["require_healthy_profile"] = False
    preprocessor = MultiResolutionEpisodePreprocessor.from_config(config)
    source = BoardWindow.__module__.split(".contracts", 1)[0]
    from vibrogemma.signal.profile import HealthyProfile, HealthyProfileBuilder

    builder = HealthyProfileBuilder(preprocessor.physics_extractor.profile_feature_names)
    capture_windows = []
    for group_name, _age_s in groups:
        arrays, rates = _read_block(devices, starts_by_group[group_name], block_s)
        for window_index in range(args.windows_per_group):
            window_payload = {}
            offset_s = window_index * args.stride_s
            for slot in SLOTS:
                rate = rates[slot]
                start = int(round(offset_s * rate))
                stop = start + int(round(args.window_s * rate))
                window_payload[slot] = arrays[slot][:, start:stop]
                window_payload[f"fs_hz__{slot}"] = np.float64(rate)
            episode = _episode(window_payload, bundle, amplitude_calibrated=args.amplitude_calibrated)
            prepared = preprocessor.prepare(episode)
            _preserve_acquisition_quality(episode, prepared)
            if not all(board.quality.is_valid for board in prepared.episode.boards):
                raise ValueError("Healthy profile capture contains invalid signal quality")
            for target_index in range(1, 6):
                builder.add(
                    prepared.profile_feature_values[target_index],
                    prepared.profile_feature_mask[target_index],
                    dataset_id="vibroagent_live",
                    structure_id="live_installation",
                    target_index=target_index,
                    split_group=f"healthy-{group_name}",
                )
            capture_windows.append({"group": group_name, "offset_s": offset_s})

    profile = builder.finalize(
        minimum_count=args.windows_per_group * 2,
        minimum_independent_groups=2,
        std_floor=float(physics.get("profile_std_floor", 1e-3)),
        allow_dataset_fallback=True,
        allow_global_fallback=False,
        metadata={
            "source": "vibroagent_live_operator_declared_healthy",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture_group_policy": "two_nonoverlapping_temporal_blocks",
            "amplitude_calibrated": args.amplitude_calibrated,
            "capture_windows": capture_windows,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.save(args.output)
    loaded = HealthyProfile.load(args.output)
    exact_keys = [
        loaded.lookup(dataset_id="vibroagent_live", structure_id="live_installation", target_index=index)
        for index in range(1, 6)
    ]
    if not all(exact_keys):
        raise RuntimeError("saved profile failed exact live-installation lookup")
    print(json.dumps({"profile": str(args.output.resolve()), "episodes": len(capture_windows), "groups": 2, "pipeline": source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
