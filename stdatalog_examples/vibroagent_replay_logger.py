#!/usr/bin/env python3
"""Offline replay logger: stream recorded acquisitions as if boards were live.

Drop-in stand-in for ``vibroagent_two_vibrometer_logger.py`` when no STWIN.box
hardware is connected (``VIBRO_OFFLINE=1 ./vibroagent.sh start``).  For every
recorded example acquisition it re-creates the live acquisition folder
(metadata copied verbatim) and appends the recorded STDatalog frames to
``iis3dwb_acc.dat`` at the ORIGINAL real-time cadence, one whole frame at a
time (``samples_per_ts * dim * int16 + 8-byte timestamp``, ~37.5 ms per frame
at the IIS3DWB rate).

Everything downstream is byte- and time-faithful: the webchat's live-readings
graphs redraw the recorded waveforms as they "arrive", the codes monitor cuts
its 10 s windows at the same relative moments as during the recording, and —
greedy decoding being deterministic — verdicts and popups reproduce at the
same registered moments with the same content.

The webchat, agent, and codec worker are completely unaware of the replay:
they read the same folders through the same code paths as with live USB
boards.  Only this producer changes.

Usage (matches the launcher's invocation):

  python vibroagent_replay_logger.py --output-root stdatalog_examples \
      --source-root examples [--speed 1.0] [--no-loop] [--stats-s 5]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SLOT_FOLDERS = ("live_baseline", "live_target_1", "live_target_2",
                "live_target_3", "live_target_4", "live_target_5")
METADATA_FILES = ("device_config.json", "acquisition_info.json",
                  "source_identity.json")
SENSOR = "iis3dwb_acc"
DTYPE_LEN = 2  # int16


def component_status(config: dict, component: str) -> dict:
    for device in config.get("devices", []):
        for entry in device.get("components", []):
            if component in entry:
                return entry[component]
    raise KeyError(f"component {component!r} not found in device_config")


@dataclass
class ReplaySlot:
    name: str
    source: Path
    target: Path
    frames: bytes
    frame_bytes: int
    frame_dt: float          # seconds of signal per frame (samples_per_ts/odr)
    written_frames: int = 0
    handle: object = field(default=None, repr=False)

    @property
    def total_frames(self) -> int:
        return len(self.frames) // self.frame_bytes

    def begin_cycle(self) -> None:
        """Fresh-acquisition semantics: metadata verbatim, empty .dat."""
        self.target.mkdir(parents=True, exist_ok=True)
        for meta in METADATA_FILES:
            if (self.source / meta).is_file():
                shutil.copy(self.source / meta, self.target / meta)
        if self.handle:
            self.handle.close()
        self.handle = (self.target / f"{SENSOR}.dat").open("wb")
        self.written_frames = 0

    def advance_to(self, signal_seconds: float) -> None:
        due = min(int(signal_seconds / self.frame_dt), self.total_frames)
        if due > self.written_frames:
            start = self.written_frames * self.frame_bytes
            end = due * self.frame_bytes
            self.handle.write(self.frames[start:end])
            self.handle.flush()
            self.written_frames = due


def load_slot(source: Path, target: Path) -> ReplaySlot:
    config = json.loads((source / "device_config.json").read_text(encoding="utf-8"))
    status = component_status(config, SENSOR)
    samples_per_ts = int(status["samples_per_ts"])
    dim = int(status["dim"])
    odr = float(status.get("measodr") or status.get("odr"))
    if not (samples_per_ts > 0 and dim > 0 and odr > 0):
        raise ValueError(f"{source.name}: invalid stream parameters")
    frame_bytes = samples_per_ts * dim * DTYPE_LEN + 8
    blob = (source / f"{SENSOR}.dat").read_bytes()
    n_frames = len(blob) // frame_bytes
    if n_frames == 0:
        raise ValueError(f"{source.name}: recording holds no complete frame")
    return ReplaySlot(
        name=source.name,
        source=source,
        target=target,
        frames=blob[: n_frames * frame_bytes],
        frame_bytes=frame_bytes,
        frame_dt=samples_per_ts / odr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True,
                        help="live acquisition root the webchat reads "
                             "(stdatalog_examples)")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="recorded acquisitions root (examples)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay speed factor (1.0 = original real time)")
    parser.add_argument("--no-loop", action="store_true",
                        help="stop after one pass instead of looping")
    parser.add_argument("--stats-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    slots = []
    for folder in SLOT_FOLDERS:
        source = args.source_root / folder
        if (source / f"{SENSOR}.dat").is_file():
            slots.append(load_slot(source, args.output_root / folder))
    if not any(slot.name == "live_baseline" for slot in slots):
        raise SystemExit(f"no live_baseline recording under {args.source_root}")

    duration = min(slot.total_frames * slot.frame_dt for slot in slots)
    print(f"Replay of {len(slots)} recorded acquisitions "
          f"({duration:.1f} s per cycle, speed x{args.speed:g}, "
          f"{'loop' if not args.no_loop else 'single pass'})", flush=True)

    cycle = 0
    try:
        while True:
            cycle += 1
            for slot in slots:
                slot.begin_cycle()
                print(f"Replay started {slot.name} "
                      f"({slot.total_frames} frames, "
                      f"{slot.total_frames * slot.frame_dt:.1f} s)", flush=True)
            t0 = time.monotonic()
            last_stats = 0.0
            while True:
                elapsed_signal = (time.monotonic() - t0) * args.speed
                for slot in slots:
                    slot.advance_to(elapsed_signal)
                if elapsed_signal >= duration:
                    break
                if elapsed_signal - last_stats >= args.stats_s * args.speed:
                    last_stats = elapsed_signal
                    print(f"[cycle {cycle}] {elapsed_signal:6.1f}/"
                          f"{duration:.1f} s replayed", flush=True)
                time.sleep(min(0.05, max(slot.frame_dt for slot in slots)))
            print(f"[cycle {cycle}] complete ({duration:.1f} s)", flush=True)
            if args.no_loop:
                break
    except KeyboardInterrupt:
        print("replay interrupted", flush=True)
    finally:
        for slot in slots:
            if slot.handle:
                slot.handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
