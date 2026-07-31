"""Mock vibrometer source for hardware-free testing.

This is a lightweight simulator, not a calibrated physical bearing model. Its
purpose is to let us test the whole architecture before a real vibrometer is
available: synthetic acquisition -> preprocessing -> features -> small
classifier -> MCP JSON -> Qwen explanation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SUPPORTED_SCENARIOS: tuple[str, ...] = (
    "healthy",
    "outer_race_fault",
    "inner_race_fault",
    "imbalance_or_looseness",
    "unknown_abnormal",
)

_SCENARIO_ALIASES = {
    "auto": "outer_race_fault",
    "demo": "outer_race_fault",
    "mock": "outer_race_fault",
    "normal": "healthy",
    "ok": "healthy",
    "good": "healthy",
    "outer": "outer_race_fault",
    "outer_race": "outer_race_fault",
    "outer_fault": "outer_race_fault",
    "bpfo": "outer_race_fault",
    "inner": "inner_race_fault",
    "inner_race": "inner_race_fault",
    "inner_fault": "inner_race_fault",
    "bpfi": "inner_race_fault",
    "imbalance": "imbalance_or_looseness",
    "looseness": "imbalance_or_looseness",
    "loose": "imbalance_or_looseness",
    "unknown": "unknown_abnormal",
    "abnormal": "unknown_abnormal",
}


def stable_seed(text: str) -> int:
    """Return a reproducible seed for a machine id or demo name."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@dataclass(frozen=True)
class MockVibrometerConfig:
    """Configuration for one synthetic vibrometer acquisition window."""

    machine_id: str = "mock_motor_01"
    scenario: str = "outer_race_fault"
    duration_s: float = 5.0
    sampling_rate_hz: int = 12_000
    rpm: float = 1_800.0
    noise_std: float = 0.025
    amplitude_scale: float = 1.0
    fault_intensity: float = 1.0
    seed: int | None = None
    window_index: int = 0


@dataclass(frozen=True)
class MockVibrometerWindow:
    """One synthetic vibrometer reading."""

    signal: np.ndarray
    metadata: dict[str, Any]


def available_mock_scenarios() -> list[dict[str, str]]:
    """Return scenario descriptions for CLI/MCP discovery."""
    return [
        {
            "scenario": "healthy",
            "ground_truth_label": "healthy",
            "example_machine_id": "mock_healthy_motor",
            "description": "Low-noise rotating-machine vibration with small 1x/2x components.",
        },
        {
            "scenario": "outer_race_fault",
            "ground_truth_label": "outer_race_fault",
            "example_machine_id": "mock_outer_race_motor",
            "description": "Impulsive vibration around a BPFO-like repetition band.",
        },
        {
            "scenario": "inner_race_fault",
            "ground_truth_label": "inner_race_fault",
            "example_machine_id": "mock_inner_race_motor",
            "description": "Higher-frequency impulsive vibration around a BPFI-like repetition band.",
        },
        {
            "scenario": "imbalance_or_looseness",
            "ground_truth_label": "imbalance_or_looseness",
            "example_machine_id": "mock_imbalance_motor",
            "description": "Elevated low-frequency 1x/2x rotational vibration without strong impulsiveness.",
        },
        {
            "scenario": "unknown_abnormal",
            "ground_truth_label": "unknown_abnormal",
            "example_machine_id": "mock_unknown_abnormal_motor",
            "description": "Mixed abnormal noise/bursts that should trigger caution but not a specific class.",
        },
    ]


def normalize_scenario(scenario: str | None, machine_id: str = "") -> str:
    """Normalize a user/demo scenario into the supported scenario names."""
    raw = (scenario or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in SUPPORTED_SCENARIOS:
        return raw
    if raw in _SCENARIO_ALIASES:
        return _SCENARIO_ALIASES[raw]

    # Allow machine id to imply a scenario when the tool caller does not set one.
    name = machine_id.lower().replace("-", "_").replace(" ", "_")
    for key, value in _SCENARIO_ALIASES.items():
        if key and key in name:
            return value
    for value in SUPPORTED_SCENARIOS:
        if value in name:
            return value

    if raw == "":
        return "outer_race_fault"

    raise ValueError("Unsupported mock vibrometer scenario. Use one of: " + ", ".join(SUPPORTED_SCENARIOS))


# Backward-compatible alias used by some earlier prototype files.
normalize_mock_scenario = normalize_scenario


def read_mock_vibrometer_window(
    machine_id: str = "mock_motor_01",
    scenario: str | None = None,
    duration_s: float = 5.0,
    sampling_rate_hz: int = 12_000,
    rpm: float = 1_800.0,
    noise_std: float = 0.025,
    amplitude_scale: float = 1.0,
    fault_intensity: float = 1.0,
    seed: int | None = None,
    window_index: int = 0,
) -> MockVibrometerWindow:
    """Convenience wrapper that simulates one mock vibrometer window."""
    return simulate_mock_vibrometer_window(
        MockVibrometerConfig(
            machine_id=machine_id,
            scenario=normalize_scenario(scenario, machine_id),
            duration_s=duration_s,
            sampling_rate_hz=sampling_rate_hz,
            rpm=rpm,
            noise_std=noise_std,
            amplitude_scale=amplitude_scale,
            fault_intensity=fault_intensity,
            seed=seed,
            window_index=window_index,
        )
    )


def generate_mock_vibrometer_window(
    machine_id: str,
    scenario: str | None = None,
    duration_s: float = 5.0,
    sampling_rate_hz: int = 12_000,
    rpm: float = 1_800.0,
    severity: float = 1.0,
    noise_level: float = 0.025,
    seed: int | None = None,
    window_index: int = 0,
) -> MockVibrometerWindow:
    """Alias with intuitive argument names used by the tests/CLI."""
    return read_mock_vibrometer_window(
        machine_id=machine_id,
        scenario=scenario,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        noise_std=noise_level,
        fault_intensity=severity,
        seed=seed,
        window_index=window_index,
    )


def simulate_mock_vibrometer_window(config: MockVibrometerConfig) -> MockVibrometerWindow:
    """Generate one synthetic vibrometer window."""
    _validate_config(config)
    scenario = normalize_scenario(config.scenario, config.machine_id)

    n = max(32, int(round(config.duration_s * config.sampling_rate_hz)))
    t = np.arange(n, dtype=np.float64) / float(config.sampling_rate_hz)
    shaft_hz = max(float(config.rpm) / 60.0, 0.1)
    seed = config.seed if config.seed is not None else stable_seed(f"{config.machine_id}:{scenario}:{config.window_index}")
    rng = np.random.default_rng(int(seed))

    # Healthy rotating-machine baseline: shaft frequency + second harmonic + sensor noise.
    phase_1x = rng.uniform(0.0, 2.0 * np.pi)
    phase_2x = rng.uniform(0.0, 2.0 * np.pi)
    signal = 0.045 * np.sin(2 * np.pi * shaft_hz * t + phase_1x)
    signal += 0.014 * np.sin(2 * np.pi * 2.0 * shaft_hz * t + phase_2x)
    signal += float(config.noise_std) * rng.standard_normal(n)

    fault_frequency_hz: float | None = None

    if scenario == "healthy":
        signal = 0.040 * np.sin(2 * np.pi * shaft_hz * t + phase_1x)
        signal += 0.010 * np.sin(2 * np.pi * 2.0 * shaft_hz * t + phase_2x)
        signal += min(float(config.noise_std), 0.025) * rng.standard_normal(n)

    elif scenario == "imbalance_or_looseness":
        # Dominant low-frequency shaft component with limited impulsiveness.
        signal += 0.34 * config.fault_intensity * np.sin(2 * np.pi * shaft_hz * t + phase_1x)
        signal += 0.09 * config.fault_intensity * np.sin(2 * np.pi * 2.0 * shaft_hz * t + phase_2x)
        fault_frequency_hz = shaft_hz

    elif scenario == "outer_race_fault":
        # BPFO-like repetition frequency; multiplier is illustrative.
        fault_frequency_hz = 4.0 * shaft_hz
        signal += _gaussian_impulse_train(
            t,
            repetition_hz=fault_frequency_hz,
            width_s=0.0013,
            amplitude=0.55 * config.fault_intensity,
            rng=rng,
        )
        signal += 0.13 * config.fault_intensity * np.sin(2 * np.pi * fault_frequency_hz * t)
        signal += 0.04 * config.fault_intensity * np.sin(2 * np.pi * 2.0 * fault_frequency_hz * t)

    elif scenario == "inner_race_fault":
        # BPFI-like repetition frequency; higher than outer race for the mock setup.
        fault_frequency_hz = 8.7 * shaft_hz
        signal += _gaussian_impulse_train(
            t,
            repetition_hz=fault_frequency_hz,
            width_s=0.0009,
            amplitude=0.50 * config.fault_intensity,
            rng=rng,
        )
        signal += 0.14 * config.fault_intensity * np.sin(2 * np.pi * fault_frequency_hz * t)
        signal += 0.035 * config.fault_intensity * np.sin(2 * np.pi * 2.0 * fault_frequency_hz * t)

    elif scenario == "unknown_abnormal":
        fault_frequency_hz = 6.2 * shaft_hz
        signal += 0.12 * config.fault_intensity * np.sin(2 * np.pi * fault_frequency_hz * t)
        signal += _random_bursts(
            t,
            count=max(2, int(config.duration_s * 2)),
            width_s=0.0020,
            amplitude=0.40 * config.fault_intensity,
            rng=rng,
        )
        signal += 0.04 * rng.standard_normal(n)

    signal = (float(config.amplitude_scale) * signal).astype(np.float64)

    metadata: dict[str, Any] = {
        "source": "mock_vibrometer",
        "machine_id": config.machine_id,
        "scenario": scenario,
        "ground_truth_label": scenario,
        "window_index": int(config.window_index),
        "seed": int(seed),
        "duration_s": float(config.duration_s),
        "sampling_rate_hz": int(config.sampling_rate_hz),
        "num_samples": int(signal.size),
        "rpm": float(config.rpm),
        "shaft_frequency_hz": float(shaft_hz),
        "fault_frequency_hz": None if fault_frequency_hz is None else float(fault_frequency_hz),
        "noise_std": float(config.noise_std),
        "amplitude_scale": float(config.amplitude_scale),
        "fault_intensity": float(config.fault_intensity),
        "note": "Synthetic signal for hardware-free testing; not a validated physical bearing simulator.",
    }
    return MockVibrometerWindow(signal=signal, metadata=metadata)


def iter_mock_vibrometer_stream(config: MockVibrometerConfig, num_windows: int) -> Iterable[MockVibrometerWindow]:
    """Yield consecutive mock vibrometer windows."""
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if num_windows > 100:
        raise ValueError("num_windows is capped at 100 for edge safety")

    for idx in range(num_windows):
        yield simulate_mock_vibrometer_window(
            MockVibrometerConfig(
                machine_id=config.machine_id,
                scenario=config.scenario,
                duration_s=config.duration_s,
                sampling_rate_hz=config.sampling_rate_hz,
                rpm=config.rpm,
                noise_std=config.noise_std,
                amplitude_scale=config.amplitude_scale,
                fault_intensity=config.fault_intensity,
                seed=None if config.seed is None else int(config.seed) + idx,
                window_index=config.window_index + idx,
            )
        )


def signal_preview(signal: np.ndarray, max_samples: int = 12) -> list[float]:
    """Return a short JSON-friendly preview of a signal."""
    max_samples = max(0, min(int(max_samples), 100))
    return [float(x) for x in np.asarray(signal, dtype=np.float64).reshape(-1)[:max_samples]]


def _validate_config(config: MockVibrometerConfig) -> None:
    if not config.machine_id.strip():
        raise ValueError("machine_id must be a non-empty string")
    if config.duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if config.duration_s > 60:
        raise ValueError("duration_s is capped at 60 seconds for edge safety")
    if config.sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if config.sampling_rate_hz > 200_000:
        raise ValueError("sampling_rate_hz is capped at 200000 Hz for edge safety")
    if config.rpm <= 0:
        raise ValueError("rpm must be positive")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if config.noise_std > 5:
        raise ValueError("noise_std is capped at 5.0")
    if config.amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")
    if config.fault_intensity < 0:
        raise ValueError("fault_intensity must be non-negative")
    if config.fault_intensity > 5:
        raise ValueError("fault_intensity is capped at 5.0")
    if config.window_index < 0:
        raise ValueError("window_index must be non-negative")


def _gaussian_impulse_train(
    t: np.ndarray,
    repetition_hz: float,
    width_s: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    period = 1.0 / max(repetition_hz, 1e-9)
    centers = np.arange(0.0, float(t[-1]) + period, period)
    y = np.zeros_like(t)
    for center in centers:
        jitter = rng.normal(0.0, width_s * 0.25)
        amp = amplitude * float(rng.uniform(0.65, 1.35))
        y += amp * np.exp(-0.5 * ((t - (center + jitter)) / width_s) ** 2)
    return y


def _random_bursts(
    t: np.ndarray,
    count: int,
    width_s: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.zeros_like(t)
    if t.size == 0:
        return y
    centers = rng.uniform(float(t[0]), float(t[-1]), size=count)
    for center in centers:
        amp = amplitude * float(rng.uniform(0.6, 1.4))
        y += amp * np.exp(-0.5 * ((t - center) / width_s) ** 2)
    return y
