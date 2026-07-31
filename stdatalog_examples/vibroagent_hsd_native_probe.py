#!/usr/bin/env python3
"""Strict native-HSD probe for vibroagent STWIN.box boards.

This intentionally does not use HSDLink.create_hsd_link(), because the stock SDK
can fall back to the serial/v1 backends after libhs_datalog_v2 fails. For the
low-latency callback logger that fallback is misleading and unsafe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from stdatalog_core.HSD_link.HSDLink_v2 import HSDLink_v2
from stdatalog_core.HSD_utils.exceptions import CommunicationEngineOpenError

DEFAULT_REQUIRED_SERIALS = (
    "003F003D3530500820323641",  # baseline
    "004E003A3530500820323641",  # target_1
    "004C00505752501620353339",  # target_2
    "004A00275752501820353339",  # target_3
    "004400225752501620353339",  # target_4
    "004B004D5752501620353339",  # target_5
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe only native libhs_datalog_v2; never fall back to serial."
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
        help="Temporary acquisition root passed to HSDLink_v2.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=6,
        help="Minimum expected native HSDatalog device count.",
    )
    parser.add_argument(
        "--require-default-serials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the six vibroagent serials configured in the logger.",
    )
    parser.add_argument(
        "--serial",
        action="append",
        default=[],
        help="Additional serial number that must be visible; can be repeated.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    hsd_link: HSDLink_v2 | None = None
    try:
        hsd_link = HSDLink_v2(
            "st_hsd",
            acquisition_folder=str(output_root),
            plug_callback=None,
            unplug_callback=None,
            update_catalog=False,
        )
    except CommunicationEngineOpenError:
        print(
            "FAIL: native libhs_datalog_v2 open failed. No serial fallback was attempted.\n"
            "Action: fully stop vibroagent, power-cycle/replug the ST hub/boards, "
            "then rerun this probe before starting the logger.",
            file=sys.stderr,
        )
        return 10

    try:
        native_count = int(getattr(hsd_link, "nof_connected_devices", 0) or 0)
        if native_count <= 0:
            print(
                "FAIL: native libhs_datalog_v2 opened but reported zero devices. "
                "Do not start the logger in this state.",
                file=sys.stderr,
            )
            return 11

        print(f"OK: native libhs_datalog_v2 opened with {native_count} device(s).")
        visible = []
        for device_id in range(native_count):
            serial = "unknown"
            try:
                serial = _device_serial(hsd_link.get_device(device_id))
            except Exception as exc:
                print(f"WARN: device {device_id}: failed to read status: {exc}")
            visible.append(serial)
            print(f"  device {device_id}: serial {serial}")

        required = []
        if args.require_default_serials:
            required.extend(DEFAULT_REQUIRED_SERIALS)
        required.extend(args.serial or [])

        missing = [serial for serial in required if serial not in visible]
        if native_count < int(args.expected_count):
            print(
                f"FAIL: expected at least {args.expected_count} native device(s), "
                f"got {native_count}.",
                file=sys.stderr,
            )
            return 13
        if missing:
            print("FAIL: missing required serial(s):", file=sys.stderr)
            for serial in missing:
                print(f"  {serial}", file=sys.stderr)
            return 12

        print("OK: native probe passed; safe to start the low-latency logger.")
        return 0
    finally:
        if hsd_link is not None:
            try:
                hsd_link.close()
            except Exception as exc:
                print(f"WARN: native close failed: {exc}", file=sys.stderr)


def _device_serial(device_status: dict[str, Any] | None) -> str:
    try:
        return str(device_status["devices"][0]["sn"])
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
