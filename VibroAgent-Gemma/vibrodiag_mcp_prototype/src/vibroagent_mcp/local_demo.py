"""Run the diagnostic pipeline without MCP."""

from __future__ import annotations

import argparse
import json

from .service import analyze_bearing_vibration_service, analyze_mock_vibrometer_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VibroAgent demo")
    parser.add_argument("--machine-id", default="demo_outer_race")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--sampling-rate-hz", type=int, default=12_000)
    parser.add_argument("--signal-path", default=None)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--include-time-frequency", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use explicit mock vibrometer data instead of file/demo input")
    parser.add_argument("--scenario", default="outer_race_fault", help="Mock scenario when --mock is used")
    parser.add_argument("--rpm", type=float, default=1800.0, help="Mock shaft speed when --mock is used")
    parser.add_argument("--noise-std", type=float, default=0.025, help="Mock noise level when --mock is used")
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--fault-intensity", type=float, default=1.0, help="Mock fault intensity when --mock is used")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window-index", type=int, default=0)
    args = parser.parse_args()

    if args.mock:
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
            window_index=args.window_index,
            include_time_frequency=args.include_time_frequency,
        )
    else:
        result = analyze_bearing_vibration_service(
            machine_id=args.machine_id,
            duration_s=args.duration_s,
            sampling_rate_hz=args.sampling_rate_hz,
            signal_path=args.signal_path,
            channel=args.channel,
            include_time_frequency=args.include_time_frequency,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
