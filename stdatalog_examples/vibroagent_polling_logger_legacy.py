#!/usr/bin/env python3
"""Log visible STWIN.box IIS3DWB devices into stable live folders.

Run this from a normal Terminal, not from the default Codex sandbox, because
ST's libusb-based HSDatalog library needs direct USB access.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from stdatalog_core.HSD_link.HSDLink import HSDLink
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


def _install_clean_stop_signal_handlers() -> None:
    """Make SIGINT *and* SIGTERM run the clean stop_log path.

    The launcher starts this logger as a background process, which makes the shell
    set SIGINT -> SIG_IGN; we reset SIGINT to Python's default so `kill -INT` / Ctrl+C
    actually raise KeyboardInterrupt, and route SIGTERM through the same handler so
    `kill -TERM` stops cleanly too. Without this the signal is ignored, the logger
    never stops, and its watchdog eventually native-crashes (free(): invalid pointer),
    leaving boards mid-stream / dropped off USB. The KeyboardInterrupt is caught by the
    main loop's try/except and the `finally` block runs stop_log on every board.
    """
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    except (ValueError, OSError):
        pass  # signal() is main-thread only; harmless if unavailable


def main() -> None:
    _install_clean_stop_signal_handlers()
    parser = argparse.ArgumentParser(description="Log STWIN.box vibrometers into stable live folders.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
        help="Root folder that will contain live_baseline and live_target_1.",
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
        help="Log every active sensor stream. By default only the requested vibrometer stream is logged.",
    )
    parser.add_argument(
        "--sensor-only",
        action="store_false",
        dest="all_active_sensors",
        help="Log only the requested --sensor stream. This is the default.",
    )
    parser.add_argument(
        "--metadata-refresh-s",
        type=float,
        default=0.0,
        help="Seconds between live device_config.json refreshes while logging. Use 0 to disable.",
    )
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=0.0,
        help="Delay after starting each device before moving to the next one.",
    )
    parser.add_argument(
        "--extra-boards",
        type=int,
        default=DEFAULT_EXTRA_BOARDS,
        help=(
            "Number of visible boards not listed in DEFAULT_ROLES to auto-assign "
            "as target_4, target_5, ... Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=10.0,
        help=(
            "Restart a board acquisition if its logged .dat file stops advancing "
            "for this many seconds. Use 0 to disable the watchdog."
        ),
    )
    parser.add_argument(
        "--stale-check-s",
        type=float,
        default=1.0,
        help="Seconds between watchdog checks for stalled live streams.",
    )
    parser.add_argument(
        "--max-stream-restarts",
        type=int,
        default=0,
        help="Maximum watchdog restarts per board. Use 0 for no limit.",
    )
    parser.add_argument(
        "--thread-stop-timeout-s",
        type=float,
        default=5.0,
        help="Seconds to wait for SDK acquisition threads to exit before stop_log.",
    )
    parser.add_argument(
        "--allow-single-board",
        action="store_true",
        help="Allow logging when only one configured HSDatalog board is visible.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    thread_stop_timeout_s = max(0.0, float(args.thread_stop_timeout_s))

    factory = HSDLink()
    hsd_link = factory.create_hsd_link(acquisition_folder=str(output_root), update_catalog=False)
    if hsd_link is None:
        raise SystemExit("No HSDatalog link could be opened.")

    devices = HSDLink.get_devices(hsd_link) or []
    print(f"Detected HSDatalog devices: {len(devices)}")
    visible_devices = _visible_devices(hsd_link, len(devices))
    for device in visible_devices:
        print(f"  device {device['device_id']}: serial {device['serial']}")

    roles = _resolve_roles(DEFAULT_ROLES, visible_devices, extra_boards=max(0, int(args.extra_boards)))
    if not roles:
        raise SystemExit("Need at least one configured HSDatalog device for this setup.")
    if len(roles) < 2 and not args.allow_single_board:
        raise SystemExit("Need at least baseline and one target HSDatalog device for this setup. Use --allow-single-board when only one board is connected.")

    running: list[dict[str, Any]] = []
    restore_items: list[dict[str, Any]] = []
    try:
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
            if not args.all_active_sensors:
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

            HSDLink.set_acquisition_info(hsd_link, device_id, f"vibroagent_{role['sensor_id']}", "")
            HSDLink.start_log(hsd_link, device_id, sub_folder=False, acq_folder=str(folder))
            HSDLink.save_json_device_file(hsd_link, device_id, str(folder))
            _write_live_acquisition_info(folder, f"vibroagent_{role['sensor_id']}")

            stop_flags = []
            data_files = []
            threads = []
            logged_sensors = active_sensor_names if args.all_active_sensors else [args.sensor]
            _prune_unlogged_dat_files(folder, logged_sensors)
            for sensor_name in logged_sensors:
                started_threads = HSDLink.start_sensor_acquisition_thread(
                    hsd_link,
                    device_id,
                    sensor_name,
                    stop_flags,
                    data_files,
                    print_data_cnt=False,
                )
                _append_started_threads(threads, started_threads)

            running.append(
                {
                    "device_id": device_id,
                    "sensor_id": role["sensor_id"],
                    "serial": serial,
                    "folder": folder,
                    "stop_flags": stop_flags,
                    "data_files": data_files,
                    "threads": threads,
                    "thread_stop_timeout_s": thread_stop_timeout_s,
                    "active_sensors": active_sensors,
                    "logged_sensors": logged_sensors,
                    "disabled_sensors": disabled_sensors,
                    "firmware": firmware,
                    "started_at_monotonic": time.monotonic(),
                    "stream_restarts": 0,
                }
            )
            print(
                f"Started {role['sensor_id']} on device {device_id}, serial {serial}, "
                f"folder {folder}"
            )
            if args.start_delay_s > 0:
                time.sleep(float(args.start_delay_s))

        _write_manifest(output_root, running)
        if args.metadata_refresh_s > 0:
            time.sleep(2.0)
            _refresh_live_metadata(hsd_link, running)
        deadline = None if args.duration_s <= 0 else time.monotonic() + float(args.duration_s)
        next_metadata_refresh = (
            time.monotonic() + float(args.metadata_refresh_s)
            if args.metadata_refresh_s > 0
            else None
        )
        next_stale_check = (
            time.monotonic() + max(0.25, float(args.stale_check_s))
            if args.stale_timeout_s > 0
            else None
        )
        print("Logging. Press Ctrl+C to stop." if deadline is None else f"Logging for {args.duration_s:g}s.")
        while deadline is None or time.monotonic() < deadline:
            if next_metadata_refresh is not None and time.monotonic() >= next_metadata_refresh:
                _refresh_live_metadata(hsd_link, running)
                next_metadata_refresh = time.monotonic() + float(args.metadata_refresh_s)
            if next_stale_check is not None and time.monotonic() >= next_stale_check:
                _restart_stale_streams(
                    hsd_link,
                    running,
                    stale_timeout_s=float(args.stale_timeout_s),
                    max_restarts=max(0, int(args.max_stream_restarts)),
                )
                next_stale_check = time.monotonic() + max(0.25, float(args.stale_check_s))
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopping logging...")
    finally:
        for item in running:
            _request_acquisition_stop(item)
        time.sleep(0.25)
        for item in running:
            _join_acquisition_threads(item, timeout_s=thread_stop_timeout_s)
            device_id = int(item["device_id"])
            folder = Path(item["folder"])
            try:
                HSDLink.stop_log(hsd_link, device_id)
            except Exception as exc:
                print(f"Warning: failed to stop device {device_id}: {exc}")
            finally:
                _close_data_files(item)
            try:
                HSDLink.save_json_device_file(hsd_link, device_id, str(folder))
                HSDLink.save_json_acq_info_file(hsd_link, device_id, str(folder))
            except Exception as exc:
                print(f"Warning: failed to save metadata for device {device_id}: {exc}")
            print(f"Stopped {item['sensor_id']} -> {folder}")
        for item in restore_items:
            try:
                _restore_disabled_sensors(
                    hsd_link,
                    int(item["device_id"]),
                    list(item["disabled_sensors"]),
                )
            except Exception as exc:
                print(f"Warning: failed to restore sensors for {item['sensor_id']}: {exc}")


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
            for key, value in template.items():
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


def _append_started_threads(threads: list[Any], started_threads: Any) -> None:
    if started_threads is None:
        return
    if isinstance(started_threads, (list, tuple)):
        threads.extend(started_threads)
    else:
        threads.append(started_threads)


def _request_acquisition_stop(item: dict[str, Any]) -> None:
    for flag in item.get("stop_flags") or []:
        flag.set()


def _join_acquisition_threads(item: dict[str, Any], *, timeout_s: float) -> list[str]:
    threads = [thread for thread in item.get("threads") or [] if hasattr(thread, "join")]
    if not threads:
        return []

    deadline = time.monotonic() + max(0.0, timeout_s)
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            thread.join(remaining)
        except RuntimeError:
            pass

    alive = []
    for thread in threads:
        try:
            is_alive = thread.is_alive()
        except Exception:
            is_alive = False
        if is_alive:
            alive.append(str(getattr(thread, "name", "unnamed")))
    if alive:
        print(
            "Warning: {} acquisition thread(s) still active after {:g}s: {}".format(
                item["sensor_id"],
                timeout_s,
                ", ".join(alive),
            )
        )
    return alive


def _close_data_files(item: dict[str, Any]) -> None:
    for data_file in item.get("data_files") or []:
        try:
            data_file.close()
        except Exception:
            pass


def _restart_stale_streams(
    hsd_link: Any,
    running: list[dict[str, Any]],
    *,
    stale_timeout_s: float,
    max_restarts: int,
) -> None:
    for item in running:
        stale_paths = _stale_logged_paths(item, stale_timeout_s=stale_timeout_s)
        if not stale_paths:
            continue

        restart_count = int(item.get("stream_restarts") or 0)
        if max_restarts > 0 and restart_count >= max_restarts:
            names = ", ".join(path.name for path in stale_paths)
            print(
                f"Watchdog: {item['sensor_id']} stream(s) still stale after "
                f"{restart_count} restart(s): {names}"
            )
            continue

        _restart_logged_streams(hsd_link, item, stale_paths)


def _stale_logged_paths(item: dict[str, Any], *, stale_timeout_s: float) -> list[Path]:
    started_at = float(item.get("started_at_monotonic") or 0.0)
    if time.monotonic() - started_at < stale_timeout_s:
        return []

    now = time.time()
    folder = Path(item["folder"])
    stale = []
    for sensor_name in item.get("logged_sensors") or []:
        path = folder / f"{sensor_name}.dat"
        try:
            modified_at = path.stat().st_mtime
        except FileNotFoundError:
            stale.append(path)
            continue
        if now - modified_at > stale_timeout_s:
            stale.append(path)
    return stale


def _restart_logged_streams(hsd_link: Any, item: dict[str, Any], stale_paths: list[Path]) -> None:
    device_id = int(item["device_id"])
    folder = Path(item["folder"])
    names = ", ".join(path.name for path in stale_paths)
    next_restart_count = int(item.get("stream_restarts") or 0) + 1
    print(
        f"Watchdog: restarting {item['sensor_id']} on device {device_id} "
        f"after stale stream(s): {names}"
    )

    _request_acquisition_stop(item)
    time.sleep(0.25)
    _join_acquisition_threads(
        item,
        timeout_s=float(item.get("thread_stop_timeout_s") or 5.0),
    )

    try:
        HSDLink.stop_log(hsd_link, device_id)
    except Exception as exc:
        print(f"Watchdog warning: failed to stop {item['sensor_id']} before restart: {exc}")
    finally:
        _close_data_files(item)

    stop_flags: list[Any] = []
    data_files: list[Any] = []
    threads: list[Any] = []
    HSDLink.start_log(hsd_link, device_id, sub_folder=False, acq_folder=str(folder))
    HSDLink.save_json_device_file(hsd_link, device_id, str(folder))
    _write_live_acquisition_info(folder, f"vibroagent_{item['sensor_id']}")
    _prune_unlogged_dat_files(folder, list(item["logged_sensors"]))
    for sensor_name in item["logged_sensors"]:
        started_threads = HSDLink.start_sensor_acquisition_thread(
            hsd_link,
            device_id,
            sensor_name,
            stop_flags,
            data_files,
            print_data_cnt=False,
        )
        _append_started_threads(threads, started_threads)

    item["stop_flags"] = stop_flags
    item["data_files"] = data_files
    item["threads"] = threads
    item["started_at_monotonic"] = time.monotonic()
    item["stream_restarts"] = next_restart_count
    print(f"Watchdog: restarted {item['sensor_id']} (restart #{next_restart_count}).")


def _write_manifest(output_root: Path, running: list[dict[str, Any]]) -> None:
    payload = {
        "created_at_unix": time.time(),
        "devices": [
            {
                "device_id": item["device_id"],
                "sensor_id": item["sensor_id"],
                "serial": item["serial"],
                "folder": str(item["folder"]),
                "active_sensors": item["active_sensors"],
                "logged_sensors": item["logged_sensors"],
                "firmware": item["firmware"],
            }
            for item in running
        ],
    }
    (output_root / "vibroagent_live_devices.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refresh_live_metadata(hsd_link: Any, running: list[dict[str, Any]]) -> None:
    for item in running:
        device_id = int(item["device_id"])
        folder = Path(item["folder"])
        try:
            HSDLink.save_json_device_file(hsd_link, device_id, str(folder))
            _write_live_acquisition_info(folder, f"vibroagent_{item['sensor_id']}")
        except Exception as exc:
            print(f"Warning: failed to refresh metadata for {item['sensor_id']}: {exc}")


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


def _format_hsd_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


if __name__ == "__main__":
    main()
