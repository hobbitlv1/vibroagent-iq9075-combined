#!/usr/bin/env python3
"""Low-latency native-callback logger for STWIN.box IIS3DWB boards.

This replaces the old polling logger. It does not use the SDK
SensorAcquisitionThread, does not use a Python Queue, and does not serialize
`get_sensor_data(...)` calls. Each selected native HSDatalog component gets a
`hs_datalog_set_data_ready_callback(...)`; the callback writes the bytes directly
to the correct live `.dat` file as soon as libhs_datalog receives them.

Run this from a normal terminal, not from a restricted sandbox, because ST's
libusb-based HSDatalog engine needs direct USB access.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from stdatalog_core.HSD_link.HSDLink import HSDLink
from stdatalog_core.HSD_link.HSDLink_v2 import HSDLink_v2
from stdatalog_core.HSD_utils.exceptions import CommunicationEngineOpenError
from stdatalog_pnpl.DTDL.device_template_manager import DeviceCatalogManager


DEFAULT_ROLES = (
    {
        "serial": "003F003D3530500820323641",
        "sensor_id": "baseline",
        "folder": "live_baseline",
    },
    {
        "serial": "004E003A3530500820323641",
        "sensor_id": "target_1",
        "folder": "live_target_1",
    },
    {
        "serial": "004C00505752501620353339",
        "sensor_id": "target_2",
        "folder": "live_target_2",
    },
    {
        "serial": "004A00275752501820353339",
        "sensor_id": "target_3",
        "folder": "live_target_3",
    },
    {
        "serial": "004400225752501620353339",
        "sensor_id": "target_4",
        "folder": "live_target_4",
    },
    {
        "serial": "004B004D5752501620353339",
        "sensor_id": "target_5",
        "folder": "live_target_5",
    },
)
DEFAULT_EXTRA_BOARDS = 0


_FALLOC_FL_KEEP_SIZE = 0x01
_FALLOC_FL_PUNCH_HOLE = 0x02
_PUNCH_HOLE_ALIGN_BYTES = 4096
_LIBC_FALLOCATE: Any = None


def _resolve_libc_fallocate() -> Any:
    """Return libc.fallocate with 64-bit-safe argtypes, or None if unavailable."""
    global _LIBC_FALLOCATE
    if _LIBC_FALLOCATE is None:
        resolved: Any = False
        try:
            libc_name = ctypes.util.find_library("c")
            libc = ctypes.CDLL(libc_name, use_errno=True) if libc_name else None
            fallocate = getattr(libc, "fallocate", None)
            if fallocate is not None:
                fallocate.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                ]
                fallocate.restype = ctypes.c_int
                resolved = fallocate
        except Exception:
            resolved = False
        _LIBC_FALLOCATE = resolved
    return _LIBC_FALLOCATE or None


def _install_clean_stop_signal_handlers() -> Callable[[], bool]:
    """Request shutdown without interrupting native calls or repeated cleanup."""
    requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal requested
        requested = True

    try:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
    except (ValueError, OSError):
        pass
    return lambda: requested


def _native_command_succeeded(result: Any) -> bool:
    """PnPL commands return (success, response); older backends return bool."""
    return (result[0] if isinstance(result, tuple) and result else result) is True


@dataclass
class NativeCallbackStream:
    """One native USB callback writing one component into one .dat file."""

    device_id: int
    sensor_id: str
    component_name: str
    folder: Path
    hsd_link: Any
    fd: int | None = None
    max_dat_file_bytes: int = 0
    rollover_keep_bytes: int = 0
    rollover_align_bytes: int = 0
    rollover_sdk_sync_bytes: int = 0
    strict_source_identity: bool = False
    bytes_written: int = 0
    active_file_bytes: int = 0
    punched_bytes: int = 0
    punch_unsupported: bool = False
    chunks_written: int = 0
    rollovers: int = 0
    write_errors: int = 0
    rollover_errors: int = 0
    rollover_detected_align_bytes: int = 0
    source_mismatches: int = 0
    component_metadata_mismatches: int = 0
    last_callback_device_id: int | None = None
    last_callback_component_name: str | None = None
    first_data_monotonic: float | None = None
    last_data_monotonic: float | None = None
    # Shutdown quiesce flag: when True the callback returns immediately without
    # touching the native block, so EP teardown never races a slow file write.
    stopping: bool = False
    callback: Callable[[int, bytes | None, Any, int], int] = field(init=False)

    def __post_init__(self) -> None:
        self.callback = self._on_data_ready

    @property
    def dat_path(self) -> Path:
        return self.folder / f"{self.component_name}.dat"

    @property
    def resident_file_bytes(self) -> int:
        """Bytes of the live .dat actually occupying disk (logical size minus punched hole)."""
        return max(0, self.active_file_bytes - self.punched_bytes)

    def open(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fd = self._open_truncated_fd()
        self.active_file_bytes = 0
        self.punched_bytes = 0

    def _open_truncated_fd(self) -> int:
        flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        return os.open(self.dat_path, flags, 0o644)

    def register(self) -> None:
        ok = HSDLink.set_data_ready_callback(
            self.hsd_link,
            self.device_id,
            self.component_name,
            self.callback,
        )
        if not ok:
            raise RuntimeError(
                f"failed to register native callback for device {self.device_id} "
                f"component {self.component_name!r}"
            )

    def unregister(self) -> None:
        try:
            HSDLink.set_data_ready_callback(
                self.hsd_link,
                self.device_id,
                self.component_name,
                None,
            )
        except Exception as exc:
            print(
                f"Warning: failed to unregister callback for "
                f"{self.sensor_id}/{self.component_name}: {exc}"
            )

    def close(self) -> None:
        if self.fd is None:
            return
        try:
            os.close(self.fd)
        finally:
            self.fd = None

    def _on_data_ready(
        self,
        callback_device_id: int,
        callback_component_name: bytes | None,
        data_ptr: Any,
        size: int,
    ) -> int:
        """Native callback: copy the block immediately and write it to disk.

        Return 0 for the native library. The callback intentionally does not use
        a queue and does not do decoding; decoding belongs in the reader/server.
        """
        component_text = _callback_component_name_text(callback_component_name)
        self.last_callback_device_id = int(callback_device_id)
        self.last_callback_component_name = component_text
        device_mismatch = int(callback_device_id) != int(self.device_id)
        component_mismatch = (
            component_text is not None and component_text != self.component_name
        )
        if component_mismatch:
            self.component_metadata_mismatches += 1
            if self.component_metadata_mismatches == 1:
                print(
                    "Warning: native callback component metadata is unreliable for "
                    f"registered stream {self.sensor_id}/{self.component_name}: "
                    f"callback component={component_text!r}. Source routing continues "
                    "to use the registered callback and verified device ID."
                )
        if device_mismatch:
            self.source_mismatches += 1
            message = (
                "Warning: native callback device ID mismatch for "
                f"registered stream {self.sensor_id}/{self.component_name}: "
                f"registered device={self.device_id}, callback device={callback_device_id}."
            )
            if self.strict_source_identity:
                if self.source_mismatches == 1:
                    print(message + " Dropping chunks because strict source identity is enabled.")
                return 0
            if self.source_mismatches == 1:
                print(message + " Writing by registered callback identity.")

        if self.stopping or size <= 0 or not data_ptr or self.fd is None:
            return 0
        try:
            payload = ctypes.string_at(data_ptr, int(size))
            payload_size = len(payload)
            self._observe_payload_size(payload_size)
            self._rollover_if_needed(payload_size)
            offset = 0
            while offset < payload_size:
                offset += os.write(self.fd, payload[offset:])
            now = time.monotonic()
            if self.first_data_monotonic is None:
                self.first_data_monotonic = now
            self.last_data_monotonic = now
            self.bytes_written += payload_size
            self.active_file_bytes += payload_size
            self.chunks_written += 1
            return 0
        except Exception:
            self.write_errors += 1
            return -1

    def _observe_payload_size(self, size: int) -> None:
        if size <= 0:
            return
        if self.rollover_detected_align_bytes <= 0:
            self.rollover_detected_align_bytes = int(size)
            return
        current = int(self.rollover_detected_align_bytes)
        if size % current == 0:
            return
        if current % size == 0:
            self.rollover_detected_align_bytes = int(size)
            return
        common = math.gcd(current, int(size))
        if common >= 4:
            self.rollover_detected_align_bytes = common

    def _rollover_alignment_bytes(self, incoming_size: int) -> int:
        configured = int(self.rollover_align_bytes or 0)
        if configured > 0:
            return configured
        if self.rollover_sdk_sync_bytes > 0:
            return int(self.rollover_sdk_sync_bytes)
        if self.rollover_detected_align_bytes > 0:
            return int(self.rollover_detected_align_bytes)
        return max(1, int(incoming_size or 1))

    def _rollover_if_needed(self, incoming_size: int) -> None:
        if (
            self.fd is None
            or self.max_dat_file_bytes <= 0
            or incoming_size <= 0
            or self.resident_file_bytes + incoming_size <= self.max_dat_file_bytes
        ):
            return
        try:
            if not self.punch_unsupported and self._punch_head_hole():
                return
            self._rollover_dat_file(incoming_size=incoming_size)
        except Exception:
            self.rollover_errors += 1

    def _punch_head_hole(self) -> bool:
        """Bound disk usage by deallocating the file head while preserving offsets."""
        if self.fd is None:
            return False
        fallocate = _resolve_libc_fallocate()
        if fallocate is None:
            self.punch_unsupported = True
            print(
                f"Warning: libc fallocate unavailable; {self.dat_path} falls back "
                "to truncate rollover (live readers will reseed)."
            )
            return False
        keep_bytes = max(0, self.rollover_keep_bytes)
        edge = ((self.active_file_bytes - keep_bytes) // _PUNCH_HOLE_ALIGN_BYTES) * _PUNCH_HOLE_ALIGN_BYTES
        if edge <= self.punched_bytes:
            return True
        result = fallocate(
            self.fd,
            _FALLOC_FL_PUNCH_HOLE | _FALLOC_FL_KEEP_SIZE,
            self.punched_bytes,
            edge - self.punched_bytes,
        )
        if result != 0:
            errno = ctypes.get_errno()
            self.punch_unsupported = True
            print(
                f"Warning: hole-punch rollover failed for {self.dat_path} "
                f"(errno {errno}); falling back to truncate rollover."
            )
            return False
        self.punched_bytes = edge
        self.rollovers += 1
        return True

    def _rollover_dat_file(self, *, incoming_size: int) -> None:
        if self.fd is None:
            return
        tail = self._read_aligned_tail(incoming_size=incoming_size)
        os.ftruncate(self.fd, 0)
        os.lseek(self.fd, 0, os.SEEK_SET)
        self.active_file_bytes = 0
        self.punched_bytes = 0
        if tail:
            offset = 0
            while offset < len(tail):
                offset += os.write(self.fd, tail[offset:])
            self.active_file_bytes = len(tail)
        self.rollovers += 1

    def _read_aligned_tail(self, *, incoming_size: int) -> bytes:
        keep_limit = max(0, self.max_dat_file_bytes - max(0, incoming_size))
        max_keep_bytes = min(max(0, self.rollover_keep_bytes), keep_limit, max(0, self.active_file_bytes))
        align = max(1, self._rollover_alignment_bytes(incoming_size))
        if max_keep_bytes <= 0:
            return b""
        read_fd = os.open(self.dat_path, os.O_RDONLY)
        try:
            file_size = os.lseek(read_fd, 0, os.SEEK_END)
            max_keep_bytes = min(max_keep_bytes, max(0, file_size))
            if max_keep_bytes <= 0:
                return b""
            start_offset = max(0, file_size - max_keep_bytes)
            if align > 1:
                start_offset += (-start_offset) % align
            if start_offset >= file_size:
                return b""
            keep_bytes = file_size - start_offset
            os.lseek(read_fd, start_offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = keep_bytes
            while remaining > 0:
                chunk = os.read(read_fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        finally:
            os.close(read_fd)


def _callback_component_name_text(value: bytes | None) -> str | None:
    if value is None:
        return None
    try:
        return value.decode("utf-8", errors="replace")
    except AttributeError:
        return str(value)


def _ignore_data_ready(
    callback_device_id: int,
    callback_component_name: bytes | None,
    data_ptr: Any,
    size: int,
) -> int:
    """No-op native callback for components that are not being logged.

    libhs_datalog_v2's Device_COM::startLogging() calls getDataReadyCallback()
    for EVERY component of the device and that lookup is a bare std::map::at,
    so any component without a registered callback makes hs_datalog_start_log
    throw std::out_of_range and abort the process. Registering this no-op for
    each unlogged component keeps the map total, and a disabled component never
    actually invokes it.
    """
    return 0


@dataclass
class BoardSession:
    device_id: int
    sensor_id: str
    serial: str
    folder: Path
    firmware: dict[str, Any]
    active_sensors: list[str]
    logged_sensors: list[str]
    disabled_sensors: list[str]
    streams: list[NativeCallbackStream]
    noop_components: list[str] = field(default_factory=list)
    started_at_monotonic: float | None = None

    @property
    def total_bytes_written(self) -> int:
        return sum(stream.bytes_written for stream in self.streams)


def main() -> None:
    stop_requested = _install_clean_stop_signal_handlers()
    parser = argparse.ArgumentParser(
        description="Low-latency native-callback logger for STWIN.box vibrometers."
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
        help="Root folder that will contain live_baseline and live_target_N folders.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Seconds to log. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--sensor",
        default="iis3dwb_acc",
        help="Required vibrometer sensor component name.",
    )
    parser.add_argument(
        "--all-active-sensors",
        action="store_true",
        help="Register native callbacks for every active sensor stream.",
    )
    parser.add_argument(
        "--sensor-only",
        action="store_false",
        dest="all_active_sensors",
        help="Log only the requested --sensor stream. This is the default.",
    )
    parser.add_argument(
        "--disable-unlogged-sensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable active streams that are not being logged to reduce USB load.",
    )
    parser.add_argument(
        "--metadata-refresh-s",
        type=float,
        default=0.0,
        help="Accepted for compatibility; live metadata refresh is disabled in callback mode.",
    )
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=0.0,
        help="Optional delay after starting each device. Leave 0 for minimal start skew.",
    )
    parser.add_argument(
        "--extra-boards",
        type=int,
        default=DEFAULT_EXTRA_BOARDS,
        help="Number of visible boards not listed in DEFAULT_ROLES to auto-assign.",
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=0.0,
        help="Warn if a stream receives no callback data for this many seconds. 0 disables.",
    )
    parser.add_argument(
        "--stale-check-s",
        type=float,
        default=1.0,
        help="Seconds between stale-stream checks when --stale-timeout-s is enabled.",
    )
    parser.add_argument(
        "--max-stream-restarts",
        type=int,
        default=0,
        help="Accepted for compatibility; automatic restarts are disabled in callback mode.",
    )
    parser.add_argument(
        "--thread-stop-timeout-s",
        type=float,
        default=0.0,
        help="Accepted for compatibility; callback mode does not create Python reader threads.",
    )
    parser.add_argument(
        "--allow-single-board",
        action="store_true",
        help="Allow logging when only one configured HSDatalog board is visible.",
    )
    parser.add_argument(
        "--stats-s",
        type=float,
        default=0.0,
        help="Print byte counters every N seconds. Use 0 to disable periodic stats.",
    )
    parser.add_argument(
        "--max-dat-file-mb",
        type=float,
        default=_env_float("VIBRO_LOGGER_MAX_DAT_MB", 0.0, minimum=0.0),
        help=(
            "When >0, keep each active live .dat file's disk usage below this "
            "approximate size by hole-punching the file head in place (offsets "
            "and timestamps stay valid for live readers). Falls back to "
            "truncate/reseed where hole punching is unsupported. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--rollover-keep-mb",
        type=float,
        default=_env_float("VIBRO_LOGGER_ROLLOVER_KEEP_MB", 16.0, minimum=0.0),
        help="Tail data to preserve when --max-dat-file-mb triggers a live .dat rollover.",
    )
    parser.add_argument(
        "--rollover-align-bytes",
        type=int,
        default=_env_int("VIBRO_LOGGER_ROLLOVER_ALIGN_BYTES", 0, minimum=0),
        help=(
            "Byte alignment for preserved rollover tail data. Use 0 to compute the SDK sync "
            "stride from sensor metadata, falling back to native callback packet size."
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.metadata_refresh_s > 0:
        print("Note: --metadata-refresh-s is ignored in low-latency callback mode.")
    if args.max_stream_restarts > 0:
        print("Note: automatic stream restarts are disabled in low-latency callback mode.")
    max_dat_file_bytes = _mb_to_bytes(float(args.max_dat_file_mb))
    rollover_keep_bytes = (
        min(_mb_to_bytes(float(args.rollover_keep_mb)), max(0, max_dat_file_bytes))
        if max_dat_file_bytes > 0
        else 0
    )
    if max_dat_file_bytes > 0:
        align_arg = int(args.rollover_align_bytes)
        align_label = f"{align_arg}B" if align_arg > 0 else "auto sdk-sync"
        print(
            "Live .dat rollover enabled (hole-punch, truncate fallback): "
            f"max={max_dat_file_bytes}B, keep_tail={rollover_keep_bytes}B, "
            f"align={align_label}"
        )

    hsd_link = _open_native_hsd_link(output_root)

    devices = HSDLink.get_devices(hsd_link) or []
    print(f"Detected HSDatalog devices: {len(devices)}")
    visible_devices = _visible_devices(hsd_link, len(devices))
    for device in visible_devices:
        print(f"  device {device['device_id']}: serial {device['serial']}")

    roles = _resolve_roles(DEFAULT_ROLES, visible_devices, extra_boards=max(0, int(args.extra_boards)))
    if not roles:
        raise SystemExit("Need at least one configured HSDatalog device for this setup.")
    if len(roles) < 2 and not args.allow_single_board:
        raise SystemExit(
            "Need at least baseline and one target HSDatalog device for this setup. "
            "Use --allow-single-board when only one board is connected."
        )

    sessions: list[BoardSession] = []
    restore_items: list[dict[str, Any]] = []
    started_sessions: list[BoardSession] = []
    try:
        sessions, restore_items = _prepare_sessions(hsd_link, roles, args, output_root)
        _write_manifest(output_root, sessions)

        print("Starting native USB callbacks and board logs...")
        for session in sessions:
            for stream in session.streams:
                stream.open()
                stream.register()
            for comp_name in session.noop_components:
                ok = HSDLink.set_data_ready_callback(
                    hsd_link, session.device_id, comp_name, _ignore_data_ready
                )
                if not ok:
                    raise RuntimeError(
                        f"failed to register no-op callback for device "
                        f"{session.device_id} component {comp_name!r}"
                    )

        # Start boards in a tight second phase after all callbacks are registered.
        # This minimizes start skew and avoids losing early blocks.
        for session in sessions:
            if stop_requested():
                break
            # A failed command can have reached the board before its reply failed.
            started_sessions.append(session)
            result = HSDLink.start_log(
                hsd_link,
                session.device_id,
                sub_folder=False,
                interface=1,
                acq_folder=str(session.folder),
                save_files=True,
            )
            if not _native_command_succeeded(result):
                raise RuntimeError(f"failed to start device {session.device_id}: {result!r}")
            session.started_at_monotonic = time.monotonic()
            HSDLink.save_json_device_file(hsd_link, session.device_id, str(session.folder))
            _write_live_acquisition_info(session.folder, f"vibroagent_{session.sensor_id}")
            print(
                f"Started {session.sensor_id} on device {session.device_id}, "
                f"serial {session.serial}, folder {session.folder}"
            )
            if args.start_delay_s > 0:
                time.sleep(float(args.start_delay_s))

        deadline = None if args.duration_s <= 0 else time.monotonic() + float(args.duration_s)
        next_stats = time.monotonic() + float(args.stats_s) if args.stats_s > 0 else None
        next_stale_check = (
            time.monotonic() + max(0.25, float(args.stale_check_s))
            if args.stale_timeout_s > 0
            else None
        )
        print(
            "Logging via native callbacks. Press Ctrl+C to stop."
            if deadline is None
            else f"Logging via native callbacks for {args.duration_s:g}s."
        )
        while not stop_requested() and (deadline is None or time.monotonic() < deadline):
            now = time.monotonic()
            if next_stats is not None and now >= next_stats:
                _print_stats(sessions)
                next_stats = now + float(args.stats_s)
            if next_stale_check is not None and now >= next_stale_check:
                _warn_stale_streams(sessions, stale_timeout_s=float(args.stale_timeout_s))
                next_stale_check = now + max(0.25, float(args.stale_check_s))
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping logging...")
    finally:
        if stop_requested():
            print("\nStopping logging...", flush=True)
        stopped_device_ids: set[int] = set()
        shutdown_errors: list[str] = []
        # Quiesce BEFORE stopping, without touching native state: re-registering or
        # unregistering a callback while data is flowing races the endpoint thread's
        # std::function invocation (observed SIGSEGV), and stopping while a slow
        # file-writing callback holds a native block races the lib's EP buffer
        # cleanup (observed free(): invalid pointer, which skips stop_log on the
        # remaining boards and wedges them mid-stream). So flip a Python-side flag
        # that makes every callback return immediately, let in-flight writes drain,
        # and only then stop.
        for session in sessions:
            for stream in session.streams:
                stream.stopping = True
        time.sleep(0.5)

        # Stop native acquisition first, then unregister callbacks, then close fds.
        for session in reversed(started_sessions):
            try:
                result = HSDLink.stop_log(hsd_link, session.device_id)
                if not _native_command_succeeded(result):
                    raise RuntimeError(f"native stop was not acknowledged: {result!r}")
                stopped_device_ids.add(session.device_id)
            except (Exception, SystemExit) as exc:
                message = f"failed to stop device {session.device_id} ({session.sensor_id}): {exc}"
                shutdown_errors.append(message)
                print(f"Warning: {message}", flush=True)

        failed_device_ids = {
            session.device_id for session in started_sessions
        } - stopped_device_ids

        for session in sessions:
            if session.device_id in failed_device_ids:
                # Keep callbacks alive and quiesced until native engine teardown;
                # unregistering while this board may still stream can crash it.
                continue
            for stream in session.streams:
                stream.unregister()
            for comp_name in session.noop_components:
                try:
                    HSDLink.set_data_ready_callback(
                        hsd_link, session.device_id, comp_name, None
                    )
                except Exception as exc:
                    print(
                        f"Warning: failed to unregister no-op callback for "
                        f"{session.sensor_id}/{comp_name}: {exc}"
                    )
            for stream in session.streams:
                try:
                    stream.close()
                except Exception as exc:
                    message = f"failed to close {session.sensor_id}/{stream.component_name}: {exc}"
                    shutdown_errors.append(message)
                    print(f"Warning: {message}")
            try:
                HSDLink.save_json_device_file(hsd_link, session.device_id, str(session.folder))
                HSDLink.save_json_acq_info_file(hsd_link, session.device_id, str(session.folder))
            except Exception as exc:
                print(f"Warning: failed to save metadata for {session.sensor_id}: {exc}")
            active_bytes = sum(stream.resident_file_bytes for stream in session.streams)
            rollovers = sum(stream.rollovers for stream in session.streams)
            if session.device_id in stopped_device_ids:
                print(
                    f"Stopped {session.sensor_id} -> {session.folder} "
                    f"({session.total_bytes_written} bytes total, "
                    f"{active_bytes} bytes active, {rollovers} rollover(s))",
                    flush=True,
                )

        for item in restore_items:
            if int(item["device_id"]) in failed_device_ids:
                continue
            try:
                _restore_disabled_sensors(
                    hsd_link,
                    int(item["device_id"]),
                    list(item["disabled_sensors"]),
                )
            except Exception as exc:
                print(f"Warning: failed to restore sensors for {item['sensor_id']}: {exc}")

        try:
            if not _native_command_succeeded(hsd_link.close()):
                raise RuntimeError("native engine close was not acknowledged")
        except Exception as exc:
            message = f"failed to close native HSDatalog engine: {exc}"
            shutdown_errors.append(message)
            print(f"Warning: {message}")
        for session in sessions:
            if session.device_id in failed_device_ids:
                for stream in session.streams:
                    try:
                        stream.close()
                    except Exception as exc:
                        shutdown_errors.append(f"failed to close {session.sensor_id}: {exc}")

        print("Shutdown result: " + json.dumps({
            "pid": os.getpid(),
            "started": [session.sensor_id for session in started_sessions],
            "stopped": [session.sensor_id for session in started_sessions
                        if session.device_id in stopped_device_ids],
            "errors": shutdown_errors,
        }), flush=True)
        if shutdown_errors:
            raise SystemExit(1)


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _sensor_status_by_name(active_sensors: Any) -> dict[str, dict[str, Any]]:
    if isinstance(active_sensors, dict):
        result: dict[str, dict[str, Any]] = {}
        for name, status in active_sensors.items():
            result[str(name)] = status if isinstance(status, dict) else {}
        return result

    result: dict[str, dict[str, Any]] = {}
    for sensor in active_sensors or []:
        if isinstance(sensor, dict) and sensor:
            name, status = next(iter(sensor.items()))
            result[str(name)] = status if isinstance(status, dict) else {}
        elif isinstance(sensor, str):
            result[str(sensor)] = {}
    return result


def _data_type_size_bytes(data_type: Any) -> int:
    raw = str(data_type or "").strip().lower().replace("_t", "")
    if raw in {"int8", "uint8", "char", "uchar", "byte"}:
        return 1
    if raw in {"int16", "uint16", "short", "ushort"}:
        return 2
    if raw in {"int32", "uint32", "float", "float32", "single"}:
        return 4
    if raw in {"int64", "uint64", "double", "float64"}:
        return 8
    return 0


def _sdk_rollover_sync_bytes(sensor_status: dict[str, Any]) -> int:
    try:
        data_packet_size = int(sensor_status.get("usb_dps") or 0)
        samples_per_ts = int(sensor_status.get("samples_per_ts") or 0)
        dim = int(sensor_status.get("dim") or 0)
    except (TypeError, ValueError):
        return 0
    dtype_len = _data_type_size_bytes(sensor_status.get("data_type"))
    if data_packet_size <= 0 or samples_per_ts <= 0 or dim <= 0 or dtype_len <= 0:
        return 0

    data_protocol_size = 4
    timestamp_frame_bytes = (samples_per_ts * dim * dtype_len) + 8
    common = math.gcd(data_packet_size, timestamp_frame_bytes)
    if common <= 0:
        return 0
    packets_per_sync = timestamp_frame_bytes // common
    return packets_per_sync * (data_packet_size + data_protocol_size)


def _open_native_hsd_link(output_root: Path) -> HSDLink_v2:
    """Open only the native libhs_datalog_v2 backend.

    The low-latency logger depends on the HSD v2 USB engine and its native
    data-ready callback. Falling back to the serial backend is unsafe here:
    after native open failure the serial path can report placeholder devices and
    later native calls may segfault. Therefore this function either returns a
    real HSDLink_v2 instance or exits before any device-status calls are made.
    """
    try:
        hsd_link = HSDLink_v2(
            "st_hsd",
            acquisition_folder=str(output_root),
            plug_callback=None,
            unplug_callback=None,
            update_catalog=False,
        )
    except CommunicationEngineOpenError as exc:
        raise SystemExit(
            "Native libhs_datalog_v2 open failed. Serial/v1 fallback is disabled "
            "for the low-latency logger, because fallback can expose invalid "
            "placeholder devices and unsafe half-open native state. Fully stop "
            "the stack, power-cycle/replug the ST board hub/boards, then run "
            "the native probe before restarting the logger."
        ) from exc

    if not isinstance(hsd_link, HSDLink_v2):
        raise SystemExit(
            f"Unexpected communication backend {type(hsd_link).__name__}; "
            "low-latency callback logging requires HSDLink_v2/libhs_datalog_v2."
        )

    connected = int(getattr(hsd_link, "nof_connected_devices", 0) or 0)
    if connected <= 0:
        try:
            hsd_link.close()
        except Exception:
            pass
        raise SystemExit(
            "Native libhs_datalog_v2 opened but reported zero HSDatalog devices. "
            "Do not use serial fallback for this logger. Power-cycle/replug the "
            "ST board hub/boards and rerun the native probe."
        )

    print(f"Native libhs_datalog_v2 opened correctly with {connected} device(s).")
    return hsd_link


def _prepare_sessions(
    hsd_link: Any,
    roles: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[BoardSession], list[dict[str, Any]]]:
    sessions: list[BoardSession] = []
    restore_items: list[dict[str, Any]] = []
    for role in roles:
        device_id = int(role["device_id"])
        folder = output_root / str(role["folder"])
        folder.mkdir(parents=True, exist_ok=True)

        _load_device_template(hsd_link, device_id)
        status = HSDLink.get_component_status(hsd_link, device_id, "firmware_info") or {}
        firmware = status.get("firmware_info") or {}
        device_status = HSDLink.get_device(hsd_link, device_id)
        serial = _device_serial(device_status)
        active_sensors = HSDLink.get_sensor_list(hsd_link, device_id, only_active=True) or []
        active_sensor_names = _sensor_names(active_sensors)
        if args.sensor not in active_sensor_names:
            raise SystemExit(
                f"Device {device_id} ({serial}) does not expose active sensor {args.sensor!r}. "
                f"Active sensors: {active_sensor_names}"
            )

        disabled_sensors: list[str] = []
        if not args.all_active_sensors and args.disable_unlogged_sensors:
            disabled_sensors = _disable_unlogged_sensors(
                hsd_link,
                device_id,
                active_sensor_names,
                [args.sensor],
            )
            if disabled_sensors:
                restore_items.append(
                    {
                        "device_id": device_id,
                        "sensor_id": role["sensor_id"],
                        "disabled_sensors": disabled_sensors,
                    }
                )
            active_sensors = HSDLink.get_sensor_list(hsd_link, device_id, only_active=True) or []
            active_sensor_names = _sensor_names(active_sensors)

        logged_sensors = active_sensor_names if args.all_active_sensors else [args.sensor]
        active_sensor_status = _sensor_status_by_name(active_sensors)
        all_component_names = _sensor_names(
            HSDLink.get_sensor_list(hsd_link, device_id, only_active=False) or []
        )
        noop_components = [
            name for name in all_component_names if name not in set(logged_sensors)
        ]
        _prune_unlogged_dat_files(folder, logged_sensors)
        HSDLink.set_acquisition_info(hsd_link, device_id, f"vibroagent_{role['sensor_id']}", "")

        streams = [
            NativeCallbackStream(
                device_id=device_id,
                sensor_id=str(role["sensor_id"]),
                component_name=sensor_name,
                folder=folder,
                hsd_link=hsd_link,
                max_dat_file_bytes=_mb_to_bytes(float(args.max_dat_file_mb)),
                rollover_keep_bytes=(
                    min(
                        _mb_to_bytes(float(args.rollover_keep_mb)),
                        max(0, _mb_to_bytes(float(args.max_dat_file_mb))),
                    )
                    if float(args.max_dat_file_mb) > 0
                    else 0
                ),
                rollover_align_bytes=int(args.rollover_align_bytes),
                rollover_sdk_sync_bytes=_sdk_rollover_sync_bytes(active_sensor_status.get(sensor_name, {})),
            )
            for sensor_name in logged_sensors
        ]
        sessions.append(
            BoardSession(
                device_id=device_id,
                sensor_id=str(role["sensor_id"]),
                serial=serial,
                folder=folder,
                firmware=firmware,
                active_sensors=active_sensor_names,
                logged_sensors=logged_sensors,
                disabled_sensors=disabled_sensors,
                streams=streams,
                noop_components=noop_components,
            )
        )
    return sessions, restore_items


def _load_device_template(hsd_link: Any, device_id: int) -> None:
    presentation = HSDLink.get_device_presentation_string(hsd_link, device_id)
    if not presentation:
        return
    board_id = hex(presentation["board_id"])
    fw_id = hex(presentation["fw_id"])
    template = DeviceCatalogManager.query_dtdl_model(board_id, fw_id)
    if isinstance(template, dict):
        fw_status = HSDLink.get_firmware_info(hsd_link, device_id) or {}
        fw_name = (fw_status.get("firmware_info") or {}).get("fw_name")
        if fw_name:
            reformatted = "".join(
                [fw_name.lower().split("-")[0]]
                + [part.capitalize() for part in fw_name.lower().split("-")[1:]]
            )
            for _key, value in template.items():
                first = value[0] if isinstance(value, list) and value else {}
                if reformatted.lower() in str(first.get("@id", "")).lower():
                    template = value
                    break
    HSDLink.set_device_template(hsd_link, template)


def _device_serial(device_status: dict[str, Any] | None) -> str:
    try:
        return str(device_status["devices"][0]["sn"])
    except Exception:
        return "unknown"


def _visible_devices(hsd_link: Any, device_count: int) -> list[dict[str, Any]]:
    visible = []
    for device_id in range(device_count):
        try:
            device_status = HSDLink.get_device(hsd_link, device_id)
        except Exception:
            device_status = None
        visible.append({"device_id": device_id, "serial": _device_serial(device_status)})
    return visible


def _resolve_roles(
    configured_roles: tuple[dict[str, str], ...],
    visible_devices: list[dict[str, Any]],
    *,
    extra_boards: int = DEFAULT_EXTRA_BOARDS,
) -> list[dict[str, Any]]:
    devices_by_serial = {
        str(device["serial"]): device
        for device in visible_devices
        if str(device.get("serial") or "unknown") != "unknown"
    }
    resolved = []
    missing = []
    for role in configured_roles:
        device = devices_by_serial.get(str(role["serial"]))
        if device is None:
            missing.append(role)
            continue
        resolved.append({**role, "device_id": int(device["device_id"])})

    if missing:
        print("Configured boards not currently visible:")
        for role in missing:
            print(f"  {role['sensor_id']}: serial {role['serial']}")

    if extra_boards > 0:
        used_serials = {str(role["serial"]) for role in resolved}
        used_device_ids = {int(role["device_id"]) for role in resolved}
        next_target_index = _next_target_index(configured_roles)
        extra_candidates = [
            device
            for device in sorted(visible_devices, key=lambda item: int(item["device_id"]))
            if int(device["device_id"]) not in used_device_ids
            and str(device.get("serial") or "unknown") != "unknown"
            and str(device["serial"]) not in used_serials
        ]
        for offset, device in enumerate(extra_candidates[:extra_boards]):
            target_index = next_target_index + offset
            role = {
                "serial": str(device["serial"]),
                "sensor_id": f"target_{target_index}",
                "folder": f"live_target_{target_index}",
                "device_id": int(device["device_id"]),
                "auto_assigned": True,
            }
            resolved.append(role)
            print(
                f"Auto-assigned extra board {role['sensor_id']}: "
                f"device {role['device_id']}, serial {role['serial']}"
            )
        if len(extra_candidates) > extra_boards:
            ignored = extra_candidates[extra_boards:]
            print(f"Ignored {len(ignored)} additional visible board(s); --extra-boards={extra_boards}.")

    return resolved


def _next_target_index(configured_roles: tuple[dict[str, str], ...]) -> int:
    highest = 0
    for role in configured_roles:
        sensor_id = str(role.get("sensor_id") or "")
        prefix, _, suffix = sensor_id.rpartition("_")
        if prefix == "target" and suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def _sensor_names(active_sensors: Any) -> list[str]:
    if isinstance(active_sensors, dict):
        return [str(name) for name in active_sensors]

    names: list[str] = []
    for sensor in active_sensors or []:
        if isinstance(sensor, str):
            names.append(sensor)
        elif isinstance(sensor, dict) and sensor:
            names.append(str(next(iter(sensor.keys()))))
        else:
            names.append(str(sensor))
    return names


def _disable_unlogged_sensors(
    hsd_link: Any,
    device_id: int,
    active_sensor_names: list[str],
    logged_sensors: list[str],
) -> list[str]:
    logged = set(logged_sensors)
    disabled = []
    for sensor_name in active_sensor_names:
        if sensor_name in logged:
            continue
        result = HSDLink.set_sensor_enable(hsd_link, device_id, False, sensor_name=sensor_name)
        if result is False:
            raise SystemExit(f"Failed to disable non-logged sensor {sensor_name!r} on device {device_id}.")
        disabled.append(sensor_name)
    if disabled:
        print(f"Device {device_id}: disabled non-vibrometer streams: {', '.join(disabled)}")
    return disabled


def _restore_disabled_sensors(hsd_link: Any, device_id: int, disabled_sensors: list[str]) -> None:
    for sensor_name in disabled_sensors:
        result = HSDLink.set_sensor_enable(hsd_link, device_id, True, sensor_name=sensor_name)
        if result is False:
            raise RuntimeError(f"failed to re-enable {sensor_name!r}")
    if disabled_sensors:
        print(f"Device {device_id}: restored streams: {', '.join(disabled_sensors)}")


def _prune_unlogged_dat_files(folder: Path, logged_sensors: list[str]) -> None:
    keep = {f"{sensor_name}.dat" for sensor_name in logged_sensors}
    for path in folder.glob("*.dat"):
        if path.name not in keep:
            path.unlink()


def _write_manifest(output_root: Path, sessions: list[BoardSession]) -> None:
    payload = {
        "created_at_unix": time.time(),
        "mode": "native_data_ready_callback",
        "latency_path": "libhs_datalog USB endpoint thread -> Python callback -> os.write(fd)",
        "devices": [
            {
                "device_id": session.device_id,
                "sensor_id": session.sensor_id,
                "serial": session.serial,
                "folder": str(session.folder),
                "active_sensors": session.active_sensors,
                "logged_sensors": session.logged_sensors,
                "firmware": session.firmware,
                "dat_rollover": [
                    {
                        "component_name": stream.component_name,
                        "strategy": "punch_hole_keep_size,truncate_fallback",
                        "max_bytes": stream.max_dat_file_bytes,
                        "keep_bytes": stream.rollover_keep_bytes,
                        "align_bytes": stream.rollover_align_bytes,
                        "detected_align_bytes": stream.rollover_detected_align_bytes,
                        "sdk_sync_bytes": stream.rollover_sdk_sync_bytes,
                    }
                    for stream in session.streams
                ],
            }
            for session in sessions
        ],
    }
    (output_root / "vibroagent_live_devices.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_live_acquisition_info(folder: Path, name: str) -> None:
    device_config_path = folder / "device_config.json"
    acquisition_uuid = str(uuid.uuid4())
    if device_config_path.exists():
        try:
            device_config = json.loads(device_config_path.read_text(encoding="utf-8"))
            acquisition_uuid = str(device_config.get("uuid") or acquisition_uuid)
        except Exception:
            pass

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=1)
    payload = {
        "tags": [],
        "name": name,
        "description": "",
        "uuid": acquisition_uuid,
        "start_time": _format_hsd_time(start_time),
        "end_time": _format_hsd_time(end_time),
        "data_ext": ".dat",
        "data_fmt": "HSD_2.0.0",
        "interface": 1,
        "schema_version": "2.0.0",
        "c_type": 2,
    }
    (folder / "acquisition_info.json").write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def _mb_to_bytes(value: float | int | str | None) -> int:
    try:
        mb = float(value or 0.0)
    except (TypeError, ValueError):
        mb = 0.0
    if mb <= 0:
        return 0
    return int(mb * 1024 * 1024)


def _format_hsd_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _print_stats(sessions: list[BoardSession]) -> None:
    parts = []
    for session in sessions:
        stream_parts = []
        for stream in session.streams:
            detail = (
                f"{stream.component_name}={stream.bytes_written}B/{stream.chunks_written}cb"
                f" active={stream.resident_file_bytes}B"
            )
            if stream.max_dat_file_bytes > 0:
                detail += f" rollovers={stream.rollovers}"
            if stream.write_errors or stream.rollover_errors:
                detail += f" errors={stream.write_errors}/{stream.rollover_errors}"
            if stream.source_mismatches:
                detail += f" device_id_mismatches={stream.source_mismatches}"
            if stream.component_metadata_mismatches:
                detail += (
                    " component_metadata_mismatches="
                    f"{stream.component_metadata_mismatches}"
                )
            stream_parts.append(detail)
        parts.append(f"{session.sensor_id}: " + ", ".join(stream_parts))
    print("Stats: " + " | ".join(parts))


def _warn_stale_streams(sessions: list[BoardSession], *, stale_timeout_s: float) -> None:
    now = time.monotonic()
    for session in sessions:
        started_at = session.started_at_monotonic
        if started_at is None or now - started_at < stale_timeout_s:
            continue
        for stream in session.streams:
            last = stream.last_data_monotonic or started_at
            if now - last > stale_timeout_s:
                print(
                    f"Warning: no native callback data for {session.sensor_id}/"
                    f"{stream.component_name} for {now - last:.1f}s"
                )


if __name__ == "__main__":
    main()
