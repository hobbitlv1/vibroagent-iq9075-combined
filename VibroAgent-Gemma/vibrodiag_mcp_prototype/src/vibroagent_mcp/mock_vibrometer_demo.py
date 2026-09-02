"""Run hardware-free mock vibrometer simulations without MCP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import save_signal_file
from .mock_vibrometer import MockVibrometerConfig, iter_mock_vibrometer_stream, simulate_mock_vibrometer_window
from .service import analyze_mock_vibrometer_service, analyze_signal_window_service


def _compact_window(result: dict[str, Any], window_index: int) -> dict[str, Any]:
    return {
        "window_index": window_index,
        "source": result["source"],
        "scenario": result["source_metadata"].get("scenario"),
        "prediction": result["prediction"],
        "vibration_status": result["baseline_comparison"]["vibration_status"],
        "rms": result["features"]["time_domain"]["rms"],
        "kurtosis": result["features"]["time_domain"]["kurtosis"],
        "dominant_frequency_hz": result["features"]["frequency_domain"]["dominant_frequency_hz"],
        "recommended_action": result["recommended_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock vibrometer simulator + diagnostic demo")
    parser.add_argument("--machine-id", default="mock_motor_01")
    parser.add_argument(
        "--scenario",
        default="outer_race_fault",
        help="healthy | outer_race_fault | inner_race_fault | imbalance_or_looseness | unknown_abnormal",
    )
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--sampling-rate-hz", type=int, default=12_000)
    parser.add_argument("--rpm", type=float, default=1_800.0)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--fault-intensity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--windows", type=int, default=1, help="Number of simulated windows")
    parser.add_argument("--include-time-frequency", action="store_true")
    parser.add_argument("--save-csv-dir", default=None, help="Optional directory for generated CSV windows")
    parser.add_argument("--full-json", action="store_true", help="Print full JSON for every window")
    args = parser.parse_args()

    if args.windows == 1:
        result = analyze_mock_vibrometer_service(
            machine_id=args.machine_id,
            scenario=args.scenario,
            duration_s=args.duration_s,
            sampling_rate_hz=args.sampling_rate_hz,
            rpm=args.rpm,
            noise_std=args.noise_std,
            amplitude_scale=args.amplitude_scale,
            fault_intensity=args.fault_intensity,
            seed=args.seed,
            window_index=0,
            include_time_frequency=args.include_time_frequency,
        )
        if args.save_csv_dir:
            window = simulate_mock_vibrometer_window(
                MockVibrometerConfig(
                    machine_id=args.machine_id,
                    scenario=args.scenario,
                    duration_s=args.duration_s,
                    sampling_rate_hz=args.sampling_rate_hz,
                    rpm=args.rpm,
                    noise_std=args.noise_std,
                    amplitude_scale=args.amplitude_scale,
                    fault_intensity=args.fault_intensity,
                    seed=args.seed,
                    window_index=0,
                )
            )
            out = Path(args.save_csv_dir) / f"{args.machine_id}_{window.metadata['scenario']}_000.csv"
            saved = save_signal_file(out, window.signal, args.sampling_rate_hz, overwrite=True)
            result["saved_mock_signal_path"] = str(saved)
        print(json.dumps(result if args.full_json else _compact_window(result, 0), indent=2))
        return

    config = MockVibrometerConfig(
        machine_id=args.machine_id,
        scenario=args.scenario,
        duration_s=args.duration_s,
        sampling_rate_hz=args.sampling_rate_hz,
        rpm=args.rpm,
        noise_std=args.noise_std,
        amplitude_scale=args.amplitude_scale,
        fault_intensity=args.fault_intensity,
        seed=args.seed,
        window_index=0,
    )

    outputs: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for window in iter_mock_vibrometer_stream(config, num_windows=args.windows):
        if args.save_csv_dir:
            out = Path(args.save_csv_dir) / f"{args.machine_id}_{window.metadata['scenario']}_{window.metadata['window_index']:03d}.csv"
            save_signal_file(out, window.signal, args.sampling_rate_hz, overwrite=True)

        result = analyze_signal_window_service(
            signal=window.signal,
            machine_id=args.machine_id,
            sampling_rate_hz=args.sampling_rate_hz,
            source="mock_vibrometer_stream",
            source_metadata=window.metadata,
            include_time_frequency=args.include_time_frequency,
        )
        label = result["prediction"]["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        outputs.append(result if args.full_json else _compact_window(result, window.metadata["window_index"]))

    print(
        json.dumps(
            {
                "machine_id": args.machine_id,
                "scenario": args.scenario,
                "windows": outputs,
                "label_counts": label_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
