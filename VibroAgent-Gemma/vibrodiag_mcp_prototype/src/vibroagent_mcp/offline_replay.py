"""Immutable recorded-acquisition replay clock.

Offline replay never copies, appends to, truncates, or otherwise opens a
recording for writing. Consumers ask this controller for a virtual position
and then use the normal STDatalog reader with an explicit start_time_s.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class OfflineReplayConfigurationError(ValueError):
    """Raised when an offline replay manifest is missing or invalid."""


@dataclass(frozen=True)
class ScheduledInference:
    """One fixed acquisition window that must be sent through the codec/model."""

    start_time_s: float
    duration_s: float
    labels: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class ClaimedInference:
    """A scheduled inference claimed once for one replay cycle."""

    cycle: int
    schedule_index: int
    window: ScheduledInference

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "schedule_index": self.schedule_index,
            **self.window.as_dict(),
        }


class OfflineReplayController:
    """Advance an immutable acquisition with a monotonic virtual clock."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str] | None = None,
        speed: float = 1.0,
        loop: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise OfflineReplayConfigurationError(
                f"offline replay manifest does not exist: {self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfflineReplayConfigurationError(
                f"unable to read offline replay manifest {self.manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise OfflineReplayConfigurationError("offline replay manifest must be a JSON object")

        self.source_root = Path(source_root or self.manifest_path.parent).expanduser().resolve()
        self.duration_s = _manifest_duration_s(manifest)
        self.speed = _positive_finite(speed, name="replay speed")
        self.loop = bool(loop)
        self.schedule = _manifest_schedule(manifest, recording_duration_s=self.duration_s)
        self._monotonic = monotonic
        self._started_at_s = float(monotonic())
        self._claimed: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def snapshot(self, *, now_s: float | None = None) -> dict[str, Any]:
        """Return the current virtual position without reading recording data."""

        now = float(self._monotonic() if now_s is None else now_s)
        elapsed_s = max(0.0, now - self._started_at_s) * self.speed
        if self.loop:
            cycle = int(elapsed_s // self.duration_s)
            position_s = elapsed_s % self.duration_s
            finished = False
        else:
            cycle = 0
            position_s = min(elapsed_s, self.duration_s)
            finished = elapsed_s >= self.duration_s
        next_item = next((item for item in self.schedule if item.start_time_s > position_s), None)
        next_cycle = cycle
        next_delay_virtual_s: float | None = None
        if next_item is not None:
            next_delay_virtual_s = next_item.start_time_s - position_s
        elif self.loop and self.schedule:
            next_item = self.schedule[0]
            next_cycle = cycle + 1
            next_delay_virtual_s = self.duration_s - position_s + next_item.start_time_s
        return {
            "enabled": True,
            "immutable": True,
            "source_root": str(self.source_root),
            "manifest_path": str(self.manifest_path),
            "duration_s": self.duration_s,
            "position_s": position_s,
            "cycle": cycle,
            "speed": self.speed,
            "loop": self.loop,
            "finished": finished,
            "inference_schedule": [item.as_dict() for item in self.schedule],
            "next_inference_time_s": next_item.start_time_s if next_item else None,
            "next_inference_cycle": next_cycle if next_item else None,
            "next_inference_in_s": (
                next_delay_virtual_s / self.speed
                if next_delay_virtual_s is not None else None
            ),
        }

    def window_start(self, duration_s: float, *, now_s: float | None = None) -> float:
        """Return a bounded fixed-window start at the current replay position."""

        duration = _positive_finite(duration_s, name="window duration")
        position = float(self.snapshot(now_s=now_s)["position_s"])
        return min(position, max(0.0, self.duration_s - duration))

    def claim_due_inference(self, *, now_s: float | None = None) -> ClaimedInference | None:
        """Claim the earliest due inference once in the current replay cycle."""

        snapshot = self.snapshot(now_s=now_s)
        position_s = self.duration_s if snapshot["finished"] and not self.loop else float(snapshot["position_s"])
        cycle = int(snapshot["cycle"])
        with self._lock:
            self._claimed = {key for key in self._claimed if key[0] >= cycle - 1}
            for index, item in enumerate(self.schedule):
                key = (cycle, index)
                if item.start_time_s <= position_s and key not in self._claimed:
                    self._claimed.add(key)
                    return ClaimedInference(cycle=cycle, schedule_index=index, window=item)
        return None

    def release_claim(self, claim: ClaimedInference) -> None:
        """Make a failed scheduled inference available to the next poll."""

        with self._lock:
            self._claimed.discard((claim.cycle, claim.schedule_index))


def _manifest_duration_s(manifest: dict[str, Any]) -> float:
    for key in ("seconds", "duration_s", "recording_duration_s"):
        value = _finite_float(manifest.get(key))
        if value is not None and value > 0:
            return value
    slots = manifest.get("slots")
    if isinstance(slots, dict):
        durations = [
            value
            for item in slots.values()
            if isinstance(item, dict)
            for value in [_finite_float(item.get("seconds"))]
            if value is not None and value > 0
        ]
        if durations:
            return min(durations)
    raise OfflineReplayConfigurationError(
        "offline replay manifest needs a positive seconds, duration_s, or slot seconds value"
    )


def _manifest_schedule(
    manifest: dict[str, Any],
    *,
    recording_duration_s: float,
) -> tuple[ScheduledInference, ...]:
    default_duration = _finite_float(manifest.get("inference_window_s")) or 10.0
    raw_windows = manifest.get("inference_windows")
    raw_timestamps = manifest.get("inference_timestamps_s")
    entries: list[tuple[float, float, str | None]] = []

    if isinstance(raw_windows, list):
        for index, item in enumerate(raw_windows):
            if not isinstance(item, dict):
                raise OfflineReplayConfigurationError(
                    f"inference_windows[{index}] must be an object"
                )
            entries.append(_schedule_entry(item, default_duration, label_key="label"))
    elif isinstance(raw_timestamps, list):
        for index, item in enumerate(raw_timestamps):
            if isinstance(item, dict):
                entries.append(_schedule_entry(item, default_duration, label_key="label"))
            else:
                start = _finite_float(item)
                if start is None:
                    raise OfflineReplayConfigurationError(
                        f"inference_timestamps_s[{index}] must be a finite number or object"
                    )
                entries.append((start, default_duration, None))
    else:
        events = manifest.get("events")
        if isinstance(events, list):
            for index, item in enumerate(events):
                if not isinstance(item, dict):
                    raise OfflineReplayConfigurationError(f"events[{index}] must be an object")
                start, duration, _ = _schedule_entry(item, default_duration, label_key="slot")
                slot = str(item.get("slot") or "").strip()
                kind = str(item.get("kind") or "").strip()
                label = ":".join(part for part in (slot, kind) if part) or None
                entries.append((start, duration, label))

    grouped: dict[float, dict[str, Any]] = {}
    for start, duration, label in entries:
        if start < 0 or start >= recording_duration_s:
            raise OfflineReplayConfigurationError(
                f"inference timestamp {start:g}s is outside the recording duration "
                f"0 <= t < {recording_duration_s:g}s"
            )
        duration = _positive_finite(duration, name=f"inference duration at {start:g}s")
        if start + duration > recording_duration_s:
            raise OfflineReplayConfigurationError(
                f"inference window {start:g}s + {duration:g}s exceeds the "
                f"{recording_duration_s:g}s recording"
            )
        group = grouped.setdefault(start, {"duration": duration, "labels": []})
        group["duration"] = max(float(group["duration"]), duration)
        if label and label not in group["labels"]:
            group["labels"].append(label)

    return tuple(
        ScheduledInference(
            start_time_s=start,
            duration_s=float(group["duration"]),
            labels=tuple(group["labels"]),
        )
        for start, group in sorted(grouped.items())
    )


def _schedule_entry(
    item: dict[str, Any],
    default_duration: float,
    *,
    label_key: str,
) -> tuple[float, float, str | None]:
    start = _finite_float(item.get("start_time_s"))
    if start is None:
        start = _finite_float(item.get("start_s"))
    if start is None:
        raise OfflineReplayConfigurationError("scheduled inference entry needs start_time_s or start_s")
    duration = _finite_float(item.get("duration_s"))
    if duration is None:
        duration = _finite_float(item.get("dur_s"))
    label = str(item.get(label_key) or "").strip() or None
    return start, duration or default_duration, label


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_finite(value: Any, *, name: str) -> float:
    number = _finite_float(value)
    if number is None or number <= 0:
        raise OfflineReplayConfigurationError(f"{name} must be a positive finite number")
    return number


_ENV_CONTROLLER_LOCK = threading.Lock()
_ENV_CONTROLLER_KEY: tuple[str, str, float, bool] | None = None
_ENV_CONTROLLER: OfflineReplayController | None = None


def offline_replay_from_environment() -> OfflineReplayController | None:
    """Return the process-wide controller when the launcher selected offline mode."""

    mode = str(os.environ.get("VIBRO_MODE") or "").strip().lower()
    enabled = str(os.environ.get("VIBRO_OFFLINE_REPLAY") or "").strip().lower()
    if mode != "offline" and enabled not in {"1", "true", "yes", "on"}:
        return None
    manifest_raw = str(os.environ.get("VIBRO_REPLAY_MANIFEST") or "").strip()
    if not manifest_raw:
        raise OfflineReplayConfigurationError("VIBRO_REPLAY_MANIFEST is required in offline mode")
    source_raw = str(os.environ.get("VIBRO_ACQUISITION_ROOT") or "").strip()
    speed = _positive_finite(os.environ.get("VIBRO_REPLAY_SPEED", "1"), name="VIBRO_REPLAY_SPEED")
    loop = str(os.environ.get("VIBRO_REPLAY_LOOP", "1")).strip().lower() not in {
        "0", "false", "no", "off", "",
    }
    key = (str(Path(manifest_raw).expanduser().resolve()), source_raw, speed, loop)
    global _ENV_CONTROLLER_KEY, _ENV_CONTROLLER
    with _ENV_CONTROLLER_LOCK:
        if _ENV_CONTROLLER is None or _ENV_CONTROLLER_KEY != key:
            _ENV_CONTROLLER = OfflineReplayController(
                key[0],
                source_root=source_raw or None,
                speed=speed,
                loop=loop,
            )
            _ENV_CONTROLLER_KEY = key
        return _ENV_CONTROLLER


def _reset_environment_controller_for_tests() -> None:
    global _ENV_CONTROLLER_KEY, _ENV_CONTROLLER
    with _ENV_CONTROLLER_LOCK:
        _ENV_CONTROLLER_KEY = None
        _ENV_CONTROLLER = None
