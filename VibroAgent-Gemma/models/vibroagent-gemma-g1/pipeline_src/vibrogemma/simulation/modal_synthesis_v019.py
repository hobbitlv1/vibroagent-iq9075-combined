from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SensorModalState:
    frequencies_hz: tuple[float, ...]
    damping_ratios: tuple[float, ...]
    mode_shapes: np.ndarray  # [mode, board, axis]
    participation_factors: np.ndarray  # [mode, MX/MY/MZ/RMX/RMY/RMZ]

    def validate(self) -> None:
        shapes = np.asarray(self.mode_shapes, dtype=np.float64)
        modes = len(self.frequencies_hz)
        if modes < 1 or shapes.shape != (modes, 6, 2):
            raise ValueError("modal state must contain [mode,6,2] sensor shapes")
        if len(self.damping_ratios) != modes:
            raise ValueError("modal frequencies and damping counts differ")
        participation = np.asarray(self.participation_factors, dtype=np.float64)
        if participation.shape != (modes, 6) or not np.isfinite(participation).all():
            raise ValueError("modal participation factors must have shape [mode,6]")
        if np.any(np.linalg.norm(participation, axis=1) <= 0.0):
            raise ValueError("each retained mode requires physical participation")
        if not np.isfinite(shapes).all() or np.any(
            ~np.isfinite(np.asarray(self.frequencies_hz + self.damping_ratios))
        ):
            raise ValueError("modal state contains non-finite values")
        if any(value <= 0.0 for value in self.frequencies_hz):
            raise ValueError("modal frequencies must be positive")
        if any(not 0.0 < value < 0.2 for value in self.damping_ratios):
            raise ValueError("modal damping must be in (0, 0.2)")
        norms = np.linalg.norm(shapes.reshape(modes, -1), axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("modal shape cannot be zero")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SensorModalState":
        state = cls(
            frequencies_hz=tuple(float(value) for value in payload["frequencies_hz"]),
            damping_ratios=tuple(float(value) for value in payload["damping_ratios"]),
            mode_shapes=np.asarray(payload["mode_shapes"], dtype=np.float64),
            participation_factors=np.asarray(
                payload["participation_factors"], dtype=np.float64
            ),
        )
        state.validate()
        return state

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "frequencies_hz": list(self.frequencies_hz),
            "damping_ratios": list(self.damping_ratios),
            "mode_shapes": np.asarray(self.mode_shapes, dtype=np.float64).tolist(),
            "participation_factors": np.asarray(
                self.participation_factors, dtype=np.float64
            ).tolist(),
        }


def synthesize_modal_family(
    *,
    states: Mapping[str, SensorModalState],
    rng: np.random.Generator,
    sample_rate_hz: float,
    duration_seconds: float,
    max_frequency_hz: float,
) -> dict[str, np.ndarray]:
    """Synthesize matched healthy/damaged sensor responses from FE modal states.

    Every state receives the same six-component generalized excitation. Its
    OpenSees participation factors project that excitation into mass-normalized
    modes; only calibrated modal properties change across counterfactual states.
    """

    if set(states) != {
        "healthy",
        "target_1",
        "target_2",
        "target_3",
        "target_4",
        "target_5",
    }:
        raise ValueError("modal family requires healthy plus target_1..target_5")
    if sample_rate_hz <= 0.0 or duration_seconds <= 0.0:
        raise ValueError("invalid modal synthesis duration/sample rate")
    for state in states.values():
        state.validate()
    mode_count = len(states["healthy"].frequencies_hz)
    if any(len(state.frequencies_hz) != mode_count for state in states.values()):
        raise ValueError("counterfactual modal states must have equal mode counts")

    samples = int(round(sample_rate_hz * duration_seconds))
    if samples < 32:
        raise ValueError("modal synthesis requires at least 32 samples")
    frequencies = np.fft.rfftfreq(samples, d=1.0 / sample_rate_hz)
    envelope = np.exp(-np.square(frequencies / max(max_frequency_hz, 0.5)))
    envelope[0] = 0.0
    generalized_forcing = np.stack(
        [np.fft.rfft(rng.normal(size=samples)) * envelope for _ in range(6)]
    )
    omega = 2.0 * np.pi * frequencies[None, :]

    outputs: dict[str, np.ndarray] = {}
    for name, state in states.items():
        natural = 2.0 * np.pi * np.asarray(state.frequencies_hz)[:, None]
        damping = np.asarray(state.damping_ratios)[:, None]
        denominator = natural**2 - omega**2 + 2j * damping * natural * omega
        acceleration = np.divide(
            -omega**2,
            denominator,
            out=np.zeros_like(denominator, dtype=np.complex128),
            where=np.abs(denominator) > 1e-18,
        )
        modal_forcing = np.asarray(state.participation_factors) @ generalized_forcing
        modal_spectrum = modal_forcing * acceleration
        sensor_spectrum = np.einsum(
            "mba,mf->baf",
            np.asarray(state.mode_shapes, dtype=np.float64),
            modal_spectrum,
        )
        outputs[name] = np.fft.irfft(sensor_spectrum, n=samples, axis=-1)

    healthy_rms = float(np.sqrt(np.mean(np.square(outputs["healthy"]))))
    if not np.isfinite(healthy_rms) or healthy_rms <= 0.0:
        raise ValueError("healthy modal synthesis is numerically degenerate")
    return {
        name: np.asarray(values / healthy_rms, dtype=np.float64)
        for name, values in outputs.items()
    }
