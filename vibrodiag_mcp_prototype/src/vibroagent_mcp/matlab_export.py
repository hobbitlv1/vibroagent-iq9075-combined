"""MATLAB validation export for building vibration monitoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import ensure_building_domain_only


class SensorWindow(Protocol):
    config: Any
    signal: Any
    timestamps_s: Any
    sampling_rate_hz: int
    metadata: dict[str, Any]

VALIDATION_COLUMNS = [
    "window_id",
    "sensor_id",
    "location",
    "role",
    "axis",
    "sampling_rate_hz",
    "time_s",
    "acceleration_g",
    "small_agent_label",
    "small_agent_confidence",
    "python_rule_label",
    "network_label",
    "quality_flags",
]


def export_validation_dataset(
    *,
    pipeline_result: dict[str, Any],
    windows: dict[str, SensorWindow],
    output_dir: str | Path,
) -> dict[str, Any]:
    ensure_building_domain_only(pipeline_result)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    report_by_sensor = {
        str(report.get("sensor_id")): report
        for report in pipeline_result.get("sensor_agent_reports", [])
    }
    network = pipeline_result.get("network_assessment", {})
    network_label = str(network.get("network_label") or network.get("main_agent_label") or "")

    csv_path = output_path / "validation_windows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_COLUMNS)
        writer.writeheader()
        for sensor_id, window in windows.items():
            report = report_by_sensor.get(sensor_id)
            if report is None and sensor_id == pipeline_result.get("baseline_sensor_id"):
                report = {
                    "small_agent_label": "normal_relative_to_baseline",
                    "small_agent_confidence": 1.0,
                    "python_rule_label": "normal_relative_to_baseline",
                    "quality_flags": window.metadata.get("quality_flags", []),
                }
            elif report is None:
                report = {
                    "small_agent_label": "insufficient_data",
                    "small_agent_confidence": 0.0,
                    "python_rule_label": "insufficient_data",
                    "quality_flags": window.metadata.get("quality_flags", []),
                }

            signal = np.asarray(window.signal, dtype=np.float64).reshape(-1)
            if window.timestamps_s is not None and window.timestamps_s.size == signal.size:
                times = window.timestamps_s
            elif window.sampling_rate_hz > 0:
                times = np.arange(signal.size, dtype=np.float64) / float(window.sampling_rate_hz)
            else:
                times = np.arange(signal.size, dtype=np.float64)

            for time_s, value in zip(times, signal):
                writer.writerow(
                    {
                        "window_id": pipeline_result.get("window_id"),
                        "sensor_id": sensor_id,
                        "location": window.config.location,
                        "role": window.config.role,
                        "axis": window.config.axis,
                        "sampling_rate_hz": window.sampling_rate_hz,
                        "time_s": float(time_s),
                        "acceleration_g": float(value),
                        "small_agent_label": report.get("small_agent_label"),
                        "small_agent_confidence": report.get("small_agent_confidence"),
                        "python_rule_label": report.get("python_rule_label"),
                        "network_label": network_label,
                        "quality_flags": ",".join(report.get("quality_flags") or []),
                    }
                )

    reports_path = output_path / "agent_reports.jsonl"
    with reports_path.open("w", encoding="utf-8") as f:
        for report in pipeline_result.get("sensor_agent_reports", []):
            ensure_building_domain_only(report)
            f.write(json.dumps(report, sort_keys=True) + "\n")

    network_path = output_path / "network_assessments.jsonl"
    with network_path.open("w", encoding="utf-8") as f:
        ensure_building_domain_only(network)
        f.write(json.dumps(network, sort_keys=True) + "\n")

    return {
        "output_dir": str(output_path),
        "validation_windows_csv": str(csv_path),
        "agent_reports_jsonl": str(reports_path),
        "network_assessments_jsonl": str(network_path),
        "csv_columns": VALIDATION_COLUMNS,
    }
