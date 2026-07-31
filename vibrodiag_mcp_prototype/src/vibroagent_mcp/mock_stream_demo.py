"""Simulate a live vibrometer stream and analyze each window.

This is the recommended hardware-free test for the prototype.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Iterable

from .mock_vibrometer import SUPPORTED_SCENARIOS, available_mock_scenarios
from .service import analyze_mock_vibrometer_service


def _scenario_sequence(selected: str, windows: int) -> Iterable[str]:
    if selected == "cycle":
        sequence = [
            "healthy",
            "imbalance_or_looseness",
            "outer_race_fault",
            "inner_race_fault",
            "unknown_abnormal",
        ]
        for i in range(windows):
            yield sequence[i % len(sequence)]
    else:
        for _ in range(windows):
            yield selected


def _compact_row(index: int, result: dict) -> dict:
    td = result["features"]["time_domain"]
    fd = result["features"]["frequency_domain"]
    mock = result.get("mock_vibrometer", {})
    return {
        "window": index,
        "ground_truth": mock.get("ground_truth_label"),
        "prediction": result["prediction"]["label"],
        "confidence": round(float(result["prediction"]["confidence"]), 3),
        "status": result["baseline_comparison"]["vibration_status"],
        "rms": round(float(td["rms"]), 4),
        "kurtosis": round(float(td["kurtosis"]), 3),
        "crest": round(float(td["crest_factor"]), 3),
        "dominant_hz": round(float(fd["dominant_frequency_hz"]), 2),
    }


def _print_table(rows: list[dict]) -> None:
    columns = ["window", "ground_truth", "prediction", "confidence", "status", "rms", "kurtosis", "crest", "dominant_hz"]
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[col]).ljust(widths[col]) for col in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock vibrometer live-stream demo")
    parser.add_argument("--machine-id", default="mock_motor_01")
    parser.add_argument(
        "--scenario",
        default="cycle",
        choices=[*sorted(SUPPORTED_SCENARIOS), "cycle"],
        help="Fault scenario to simulate. Use cycle to test all scenarios.",
    )
    parser.add_argument("--windows", type=int, default=5, help="Number of simulated acquisition windows")
    parser.add_argument("--duration-s", type=float, default=1.0, help="Duration of each simulated window")
    parser.add_argument("--sampling-rate-hz", type=int, default=2_000)
    parser.add_argument("--rpm", type=float, default=1_800.0)
    parser.add_argument("--fault-intensity", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-time-frequency", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Optional delay between windows")
    parser.add_argument("--jsonl", action="store_true", help="Print one full JSON result per line")
    parser.add_argument("--list-scenarios", action="store_true", help="Print available mock scenarios and exit")
    args = parser.parse_args()

    if args.list_scenarios:
        print(json.dumps(available_mock_scenarios(), indent=2))
        return

    if args.windows <= 0:
        raise ValueError("--windows must be positive")

    compact_rows: list[dict] = []
    for index, scenario in enumerate(_scenario_sequence(args.scenario, args.windows)):
        result = analyze_mock_vibrometer_service(
            machine_id=args.machine_id,
            scenario=scenario,
            duration_s=args.duration_s,
            sampling_rate_hz=args.sampling_rate_hz,
            rpm=args.rpm,
            fault_intensity=args.fault_intensity,
            noise_std=args.noise_std,
            seed=None if args.seed is None else args.seed + index,
            window_index=index,
            include_time_frequency=args.include_time_frequency,
        )
        if args.jsonl:
            print(json.dumps(result))
        else:
            compact_rows.append(_compact_row(index, result))
        if args.sleep_s > 0 and index < args.windows - 1:
            time.sleep(args.sleep_s)

    if not args.jsonl:
        _print_table(compact_rows)


if __name__ == "__main__":
    main()
