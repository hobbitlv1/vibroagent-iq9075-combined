from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


LUMO_INVERSE_CALIBRATION_SCHEMA = "vibrogemma-lumo-inverse-calibration-v0.19.0"
LUMO_INVERSE_METHOD = "train_group_equal_weight_output_only_fdd_bounded_subspace_modal_fit_v2"
LUMO_HEALTHY_MODE_BANK_POLICY = "sorted_union_of_healthy_train_group_assignments_v1"
LUMO_MODAL_RUNNER_CONTRACT = (
    "(brace_scale, damage_level|None, damage_residual)"
    "->(frequencies_hz, mode_shapes[modes,12])"
)
ModalRunner = Callable[[float, str | None, float], tuple[np.ndarray, np.ndarray]]


def healthy_train_mode_bank(
    assignments: Iterable[Sequence[int]], *, maximum_modes: int = 10
) -> tuple[int, ...]:
    rows = [tuple(values) for values in assignments]
    if not rows or maximum_modes < 1:
        raise ValueError("healthy mode bank requires train-group assignments")
    for row in rows:
        if (
            not row
            or len(set(row)) != len(row)
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                for index in row
            )
        ):
            raise ValueError("healthy mode bank received an invalid assignment")
    selected = tuple(sorted({index for row in rows for index in row}))
    if len(selected) > maximum_modes:
        raise ValueError("healthy mode bank exceeds the retained forward-mode pool")
    return selected


@dataclass(frozen=True)
class MeasuredModalExample:
    example_id: str
    physical_group: str
    split: str
    damage_level: str | None
    board_values: tuple[np.ndarray, ...]
    sample_rate_hz: float

    def validate(self) -> None:
        if not self.example_id or not self.physical_group:
            raise ValueError("modal examples require an id and physical group")
        if self.split != "train":
            raise ValueError(
                f"LUMO inverse calibration is train-only; {self.example_id} has split={self.split!r}"
            )
        if self.damage_level is not None and not str(self.damage_level).strip():
            raise ValueError("damage_level must be None or a non-empty identifier")
        if len(self.board_values) != 6:
            raise ValueError("LUMO inverse calibration requires exactly six boards")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        _stack_twelve_channels(self.board_values)


@dataclass(frozen=True)
class LumoInverseBounds:
    global_stiffness_scale: tuple[float, float] = (0.6, 1.6)
    brace_scale: tuple[float, float] = (0.5, 1.5)
    damage_residual: tuple[float, float] = (-0.25, 0.25)
    damping_ratio: tuple[float, float] = (0.002, 0.15)

    def validate(self) -> None:
        for name, values in self.as_dict().items():
            low, high = values
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"invalid {name} bounds: {(low, high)}")
        if self.global_stiffness_scale[0] <= 0.0 or self.brace_scale[0] <= 0.0:
            raise ValueError("stiffness and brace scales must remain positive")
        if self.damping_ratio[0] < 0.0:
            raise ValueError("damping ratio cannot be negative")

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "global_stiffness_scale": tuple(float(value) for value in self.global_stiffness_scale),
            "brace_scale": tuple(float(value) for value in self.brace_scale),
            "damage_residual": tuple(float(value) for value in self.damage_residual),
            "damping_ratio": tuple(float(value) for value in self.damping_ratio),
        }


@dataclass(frozen=True)
class OutputOnlyModes:
    frequencies_hz: tuple[float, ...]
    mode_shapes: np.ndarray
    damping_ratios: tuple[float, ...]

    def validate(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        modes = np.asarray(self.mode_shapes)
        damping = np.asarray(self.damping_ratios, dtype=np.float64)
        if not 1 <= frequencies.size <= 3:
            raise ValueError("FDD extraction must return one to three modes")
        if modes.shape != (frequencies.size, 12):
            raise ValueError("FDD mode shapes must have shape [modes, 12]")
        if damping.shape != frequencies.shape:
            raise ValueError("one damping estimate is required per extracted mode")
        if (
            not np.isfinite(frequencies).all()
            or not np.isfinite(modes).all()
            or not np.isfinite(damping).all()
            or np.any(frequencies <= 0.0)
            or np.any(damping < 0.0)
        ):
            raise ValueError("FDD result contains invalid values")


@dataclass(frozen=True)
class LumoInverseCalibration:
    global_stiffness_scale: float
    brace_scale: float
    damage_residual: float
    damping_ratio: float
    bounds: LumoInverseBounds
    fit_cost: float
    fit_optimality: float
    evaluations: int
    success: bool
    jacobian_rank: int
    jacobian_condition: float
    bound_hits: tuple[str, ...]
    damping_basis: str
    damping_relative_mad: float | None
    group_hashes: dict[str, str]
    group_summaries: dict[str, dict[str, Any]]
    matched_forward_mode_indices_by_group: dict[str, tuple[int, ...]]
    fdd_settings: dict[str, float | int]
    source_split: str = "train"

    def validate(self) -> None:
        self.bounds.validate()
        if self.source_split != "train":
            raise ValueError("LUMO inverse calibration must be fitted from train only")
        parameters = {
            "global_stiffness_scale": self.global_stiffness_scale,
            "brace_scale": self.brace_scale,
            "damage_residual": self.damage_residual,
            "damping_ratio": self.damping_ratio,
        }
        for name, value in parameters.items():
            low, high = self.bounds.as_dict()[name]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"inverse calibration {name} is outside its bounds")
        if (
            not np.isfinite(self.fit_cost)
            or self.fit_cost < 0.0
            or not np.isfinite(self.fit_optimality)
            or self.fit_optimality < 0.0
            or self.evaluations < 1
        ):
            raise ValueError("inverse calibration optimizer receipt is invalid")
        if not self.success:
            raise ValueError("inverse calibration optimizer did not converge")
        if self.jacobian_rank != 3 or not np.isfinite(self.jacobian_condition):
            raise ValueError("inverse calibration parameters are not locally identifiable")
        if self.jacobian_condition > 1.0e10:
            raise ValueError("inverse calibration Jacobian is too ill-conditioned")
        if any(name not in parameters for name in self.bound_hits):
            raise ValueError("inverse calibration bound-hit receipt is invalid")
        if self.damping_basis not in {
            "healthy_train_group_half_power_stable",
            "prior_fixed_unstable_output_only_damping",
        }:
            raise ValueError("inverse calibration damping basis is invalid")
        if self.damping_relative_mad is not None and (
            not np.isfinite(self.damping_relative_mad)
            or self.damping_relative_mad < 0.0
        ):
            raise ValueError("inverse calibration damping dispersion is invalid")
        if not self.group_hashes or set(self.group_hashes) != set(self.group_summaries):
            raise ValueError("inverse calibration group provenance is incomplete")
        if set(self.matched_forward_mode_indices_by_group) != set(self.group_summaries):
            raise ValueError("inverse calibration modal assignments are incomplete")
        for group, digest in self.group_hashes.items():
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"invalid physical-group hash for {group}")
            summary = self.group_summaries[group]
            if summary.get("source_split") != "train" or int(summary.get("example_count", 0)) < 1:
                raise ValueError(f"invalid physical-group summary for {group}")
            matched = self.matched_forward_mode_indices_by_group[group]
            if (
                len(matched) != len(summary.get("frequencies_hz") or [])
                or len(set(matched)) != len(matched)
                or any(index < 0 for index in matched)
            ):
                raise ValueError(f"invalid modal assignment for {group}")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema": LUMO_INVERSE_CALIBRATION_SCHEMA,
            "dataset_id": "lumo",
            "source_split": self.source_split,
            "method": LUMO_INVERSE_METHOD,
            "runner_contract": LUMO_MODAL_RUNNER_CONTRACT,
            "identifiability": {
                "input_force_scale": "not_fitted",
                "mass_scale": "fixed_by_forward_model",
                "global_stiffness_effect": "sqrt_scale_on_modal_frequencies",
                "damping_source": self.damping_basis,
                "damping_basis": self.damping_basis,
                "damping_relative_mad": self.damping_relative_mad,
            },
            "parameters": {
                "global_stiffness_scale": self.global_stiffness_scale,
                "brace_scale": self.brace_scale,
                "damage_residual": self.damage_residual,
                "damping_ratio": self.damping_ratio,
            },
            "bounds": {name: list(values) for name, values in self.bounds.as_dict().items()},
            "optimizer": {
                "success": self.success,
                "cost": self.fit_cost,
                "optimality": self.fit_optimality,
                "evaluations": self.evaluations,
                "jacobian_rank": self.jacobian_rank,
                "jacobian_condition": self.jacobian_condition,
                "bound_hits": list(self.bound_hits),
            },
            "physical_group_count": len(self.group_hashes),
            "group_hashes": dict(sorted(self.group_hashes.items())),
            "groups": {
                name: {
                    **self.group_summaries[name],
                    "matched_forward_mode_indices": list(
                        self.matched_forward_mode_indices_by_group[name]
                    ),
                }
                for name in sorted(self.group_summaries)
            },
            "fdd": dict(sorted(self.fdd_settings.items())),
        }
        payload["receipt_sha256"] = _payload_hash(payload)
        return payload


def _payload_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("receipt_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"LUMO board waveform must be 2D, observed {array.shape}")
    if array.shape[0] in {2, 3} and array.shape[1] > array.shape[0]:
        array = array.T
    elif array.shape[1] not in {2, 3} or array.shape[0] <= array.shape[1]:
        raise ValueError(f"cannot identify LUMO sample axis in shape {array.shape}")
    if array.shape[1] < 2:
        raise ValueError("LUMO inverse calibration requires x/y channels")
    array = array[:, :2]
    if array.shape[0] < 64 or not np.isfinite(array).all():
        raise ValueError("LUMO modal waveform must contain at least 64 finite samples")
    return array


def _stack_twelve_channels(board_values: Sequence[np.ndarray]) -> np.ndarray:
    if len(board_values) != 6:
        raise ValueError("LUMO inverse calibration requires exactly six boards")
    boards = [_canonical_channels(value) for value in board_values]
    lengths = {board.shape[0] for board in boards}
    if len(lengths) != 1:
        raise ValueError("all six LUMO boards must have equal sample counts")
    return np.concatenate(boards, axis=1)


def _example_hash(example: MeasuredModalExample) -> str:
    digest = hashlib.sha256()
    digest.update(example.example_id.encode())
    digest.update(b"\0")
    digest.update(example.physical_group.encode())
    digest.update(b"\0")
    digest.update(str(example.damage_level).encode())
    digest.update(np.float64(example.sample_rate_hz).tobytes())
    for board in example.board_values:
        array = np.ascontiguousarray(np.asarray(board))
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fdd_spectrum(
    examples: Sequence[MeasuredModalExample], *, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = examples[0].sample_rate_hz
    if any(not math.isclose(example.sample_rate_hz, sample_rate) for example in examples):
        raise ValueError("one physical group cannot mix sample rates")
    matrices = [_stack_twelve_channels(example.board_values) for example in examples]
    segment = min(int(nperseg), *(matrix.shape[0] for matrix in matrices))
    if segment < 64:
        raise ValueError("FDD segment length must be at least 64")
    step = max(segment // 2, 1)
    window = np.hanning(segment)
    scale = float(np.sum(window**2))
    spectral = np.zeros((segment // 2 + 1, 12, 12), dtype=np.complex128)
    snapshot_count = 0
    for matrix in matrices:
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        for start in range(0, centered.shape[0] - segment + 1, step):
            fft = np.fft.rfft(centered[start : start + segment] * window[:, None], axis=0)
            spectral += np.einsum("fi,fj->fij", fft, np.conj(fft), optimize=True) / scale
            snapshot_count += 1
    if snapshot_count < 1:
        raise ValueError("no complete FDD segments")
    frequencies = np.fft.rfftfreq(segment, d=1.0 / sample_rate)
    return frequencies, spectral / snapshot_count


def _half_power_damping(frequencies: np.ndarray, values: np.ndarray, peak: int) -> float:
    threshold = values[peak] * 0.5
    left = peak
    while left > 0 and values[left] > threshold:
        left -= 1
    right = peak
    while right + 1 < values.size and values[right] > threshold:
        right += 1
    if left == peak or right == peak or frequencies[peak] <= 0.0:
        return 0.0
    return max(float((frequencies[right] - frequencies[left]) / (2.0 * frequencies[peak])), 0.0)


def extract_output_only_modes(
    examples: Sequence[MeasuredModalExample],
    *,
    max_modes: int = 3,
    min_frequency_hz: float = 0.5,
    max_frequency_hz: float = 80.0,
    nperseg: int = 1024,
    minimum_peak_prominence_db: float = 3.0,
) -> OutputOnlyModes:
    if not examples:
        raise ValueError("at least one measured modal example is required")
    if not 1 <= max_modes <= 3:
        raise ValueError("max_modes must be in 1..3")
    for example in examples:
        example.validate()
    groups = {example.physical_group for example in examples}
    damage_levels = {example.damage_level for example in examples}
    if len(groups) != 1 or len(damage_levels) != 1:
        raise ValueError("FDD extraction requires one physical group and one damage state")
    frequencies, spectral = _fdd_spectrum(examples, nperseg=nperseg)
    valid = (frequencies >= min_frequency_hz) & (
        frequencies <= min(max_frequency_hz, examples[0].sample_rate_hz / 2.0)
    )
    selected = np.flatnonzero(valid)
    if selected.size < 3:
        raise ValueError("FDD frequency band contains too few bins")
    leading = np.zeros(frequencies.size, dtype=np.float64)
    vectors = np.zeros((frequencies.size, 12), dtype=np.complex128)
    for index in selected:
        eigenvalues, eigenvectors = np.linalg.eigh(spectral[index])
        leading[index] = max(float(eigenvalues[-1].real), 0.0)
        vectors[index] = eigenvectors[:, -1]
    log_leading = 10.0 * np.log10(np.maximum(leading[selected], 1e-30))
    peak_offsets, _ = find_peaks(log_leading, prominence=minimum_peak_prominence_db)
    if peak_offsets.size == 0:
        peak_offsets = np.asarray([int(np.argmax(log_leading))])
    peak_indices = selected[peak_offsets]
    strongest = sorted(peak_indices, key=lambda index: (-leading[index], frequencies[index]))[
        :max_modes
    ]
    strongest.sort(key=lambda index: frequencies[index])
    modes: list[np.ndarray] = []
    damping: list[float] = []
    for index in strongest:
        mode = vectors[index]
        pivot = int(np.argmax(np.abs(mode)))
        if abs(mode[pivot]) > 0.0:
            mode = mode * np.exp(-1j * np.angle(mode[pivot]))
        mode = mode / max(float(np.linalg.norm(mode)), 1e-12)
        modes.append(mode)
        damping.append(_half_power_damping(frequencies, leading, int(index)))
    result = OutputOnlyModes(
        frequencies_hz=tuple(float(frequencies[index]) for index in strongest),
        mode_shapes=np.stack(modes),
        damping_ratios=tuple(damping),
    )
    result.validate()
    return result


def modal_assurance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.complex128).reshape(-1)
    b = np.asarray(right, dtype=np.complex128).reshape(-1)
    if a.size != 12 or b.size != 12:
        raise ValueError("modal assurance requires two 12-channel shapes")
    denominator = float(np.vdot(a, a).real * np.vdot(b, b).real)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(abs(np.vdot(a, b)) ** 2 / denominator, 0.0, 1.0))


def modal_subspace_assurance(measured: np.ndarray, model_modes: np.ndarray) -> float:
    vector = np.asarray(measured, dtype=np.complex128).reshape(-1)
    basis = np.asarray(model_modes, dtype=np.complex128)
    if vector.size != 12 or basis.ndim != 2 or basis.shape[1] != 12 or basis.shape[0] < 1:
        raise ValueError("modal subspace assurance requires a 12-channel vector and basis")
    left_vectors, singular_values, _ = np.linalg.svd(basis.T, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(basis.shape) * singular_values[0]
    orthonormal = left_vectors[:, singular_values > tolerance]
    denominator = float(np.vdot(vector, vector).real)
    if denominator <= 0.0:
        return 0.0
    projection = orthonormal.conj().T @ vector
    return float(np.clip(np.vdot(projection, projection).real / denominator, 0.0, 1.0))


def _normalise_runner_output(value: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(value[0], dtype=np.float64).reshape(-1)
    modes = np.asarray(value[1], dtype=np.complex128)
    if modes.ndim != 2 or modes.shape != (frequencies.size, 12):
        raise ValueError("modal runner must return mode_shapes with shape [modes,12]")
    if frequencies.size < 1 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
        raise ValueError("modal runner returned invalid frequencies")
    if not np.isfinite(modes).all() or np.any(np.linalg.norm(modes, axis=1) <= 0.0):
        raise ValueError("modal runner returned invalid mode shapes")
    order = np.argsort(frequencies)
    return frequencies[order], modes[order]


def _best_modal_match(
    measured: OutputOnlyModes,
    model_frequencies: np.ndarray,
    model_modes: np.ndarray,
    *,
    mode_shape_weight: float,
) -> tuple[tuple[int, ...], np.ndarray]:
    count = len(measured.frequencies_hz)
    if model_frequencies.size < count:
        raise ValueError("modal runner returned fewer modes than measured FDD")
    measured_frequencies = np.asarray(measured.frequencies_hz, dtype=np.float64)
    measured_modes = np.asarray(measured.mode_shapes)
    best: np.ndarray | None = None
    best_indices: tuple[int, ...] | None = None
    best_cost = float("inf")
    for indices in itertools.permutations(range(model_frequencies.size), count):
        frequency_error = np.log(model_frequencies[list(indices)] / measured_frequencies)
        shape_error = np.asarray(
            [
                math.sqrt(
                    max(
                        1.0
                        - modal_subspace_assurance(
                            measured_modes[i],
                            model_modes[
                                np.abs(
                                    np.log(model_frequencies / model_frequencies[index])
                                )
                                <= 0.02
                            ],
                        ),
                        0.0,
                    )
                )
                * mode_shape_weight
                for i, index in enumerate(indices)
            ]
        )
        residual = np.concatenate((frequency_error, shape_error))
        cost = float(np.dot(residual, residual))
        if cost < best_cost:
            best_cost, best, best_indices = cost, residual, tuple(int(value) for value in indices)
    assert best is not None and best_indices is not None
    return best_indices, best


def _matched_residuals(
    measured: OutputOnlyModes,
    model_frequencies: np.ndarray,
    model_modes: np.ndarray,
    *,
    mode_shape_weight: float,
) -> np.ndarray:
    return _best_modal_match(
        measured,
        model_frequencies,
        model_modes,
        mode_shape_weight=mode_shape_weight,
    )[1]


def fit_lumo_inverse_calibration(
    *,
    examples: Iterable[MeasuredModalExample],
    modal_runner: ModalRunner,
    bounds: LumoInverseBounds = LumoInverseBounds(),
    max_modes: int = 3,
    min_frequency_hz: float = 0.5,
    max_frequency_hz: float = 80.0,
    nperseg: int = 1024,
    minimum_peak_prominence_db: float = 3.0,
    mode_shape_weight: float = 0.35,
    max_evaluations: int = 200,
) -> LumoInverseCalibration:
    bounds.validate()
    values = list(examples)
    if not values:
        raise ValueError("LUMO inverse calibration requires measured train examples")
    for example in values:
        example.validate()
    grouped: dict[str, list[MeasuredModalExample]] = {}
    for example in values:
        grouped.setdefault(example.physical_group, []).append(example)
    extracted: dict[str, OutputOnlyModes] = {}
    group_hashes: dict[str, str] = {}
    group_summaries: dict[str, dict[str, Any]] = {}
    for group, group_examples in sorted(grouped.items()):
        damage_levels = {example.damage_level for example in group_examples}
        if len(damage_levels) != 1:
            raise ValueError(f"physical group {group} spans multiple damage states")
        modes = extract_output_only_modes(
            group_examples,
            max_modes=max_modes,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
            nperseg=nperseg,
            minimum_peak_prominence_db=minimum_peak_prominence_db,
        )
        extracted[group] = modes
        hashes = sorted(_example_hash(example) for example in group_examples)
        group_hashes[group] = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
        group_summaries[group] = {
            "source_split": "train",
            "damage_level": next(iter(damage_levels)),
            "example_count": len(group_examples),
            "frequencies_hz": list(modes.frequencies_hz),
            "damping_ratios": list(modes.damping_ratios),
        }
    if not any(summary["damage_level"] is None for summary in group_summaries.values()):
        raise ValueError("LUMO inverse calibration requires a healthy train group")
    if not any(summary["damage_level"] is not None for summary in group_summaries.values()):
        raise ValueError("LUMO inverse calibration requires a damaged train group")
    if not np.isfinite(mode_shape_weight) or mode_shape_weight < 0.0 or max_evaluations < 1:
        raise ValueError("invalid inverse calibration optimizer settings")

    bound_values = bounds.as_dict()
    lower = np.asarray(
        [bound_values[name][0] for name in ("global_stiffness_scale", "brace_scale", "damage_residual")]
    )
    upper = np.asarray(
        [bound_values[name][1] for name in ("global_stiffness_scale", "brace_scale", "damage_residual")]
    )
    initial = (lower + upper) / 2.0

    def residuals(parameters: np.ndarray) -> np.ndarray:
        global_scale, brace_scale, damage_residual = (float(value) for value in parameters)
        cache: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        output: list[np.ndarray] = []
        group_scale = 1.0 / math.sqrt(len(grouped))
        for group in sorted(grouped):
            damage_level = group_summaries[group]["damage_level"]
            if damage_level not in cache:
                frequencies, modes = _normalise_runner_output(
                    modal_runner(brace_scale, damage_level, damage_residual)
                )
                cache[damage_level] = frequencies * math.sqrt(global_scale), modes
            frequencies, modes = cache[damage_level]
            one = _matched_residuals(
                extracted[group], frequencies, modes, mode_shape_weight=mode_shape_weight
            )
            output.append(one * group_scale / math.sqrt(one.size))
        return np.concatenate(output)

    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        max_nfev=int(max_evaluations),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    if not result.success:
        raise RuntimeError(f"bounded LUMO inverse calibration did not converge: {result.message}")
    damping_values = [
        float(np.median(valid))
        for group, modes in extracted.items()
        if group_summaries[group]["damage_level"] is None
        if (
            valid := [
                value
                for value in modes.damping_ratios
                if np.isfinite(value) and value > 0.0
            ]
        )
    ]
    if len(damping_values) >= 2:
        damping_median = float(np.median(damping_values))
        damping_relative_mad = float(
            np.median(np.abs(np.asarray(damping_values) - damping_median))
            / max(damping_median, 1e-12)
        )
    else:
        damping_median, damping_relative_mad = 0.02, None
    if (
        damping_relative_mad is not None
        and damping_relative_mad <= 0.5
        and bounds.damping_ratio[0] < damping_median < bounds.damping_ratio[1]
    ):
        damping = damping_median
        damping_basis = "healthy_train_group_half_power_stable"
    else:
        damping = float(np.clip(0.02, *bounds.damping_ratio))
        damping_basis = "prior_fixed_unstable_output_only_damping"
    singular_values = np.linalg.svd(np.asarray(result.jac, dtype=np.float64), compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(result.jac.shape) * singular_values[0]
    jacobian_rank = int(np.sum(singular_values > tolerance))
    jacobian_condition = float(
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 0.0
        else float("inf")
    )
    parameter_names = ("global_stiffness_scale", "brace_scale", "damage_residual")
    bound_hits = tuple(
        name
        for index, name in enumerate(parameter_names)
        if min(abs(result.x[index] - lower[index]), abs(result.x[index] - upper[index]))
        <= 1.0e-6 * max(upper[index] - lower[index], 1.0)
    )
    if min(
        abs(damping - bounds.damping_ratio[0]),
        abs(damping - bounds.damping_ratio[1]),
    ) <= 1.0e-6 * max(bounds.damping_ratio[1] - bounds.damping_ratio[0], 1.0):
        bound_hits = (*bound_hits, "damping_ratio")
    final_models: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
    matched_forward_mode_indices_by_group: dict[str, tuple[int, ...]] = {}
    for group in sorted(grouped):
        damage_level = group_summaries[group]["damage_level"]
        if damage_level not in final_models:
            frequencies, modes = _normalise_runner_output(
                modal_runner(float(result.x[1]), damage_level, float(result.x[2]))
            )
            final_models[damage_level] = (
                frequencies * math.sqrt(float(result.x[0])),
                modes,
            )
        frequencies, modes = final_models[damage_level]
        matched_forward_mode_indices_by_group[group] = _best_modal_match(
            extracted[group],
            frequencies,
            modes,
            mode_shape_weight=mode_shape_weight,
        )[0]
    calibration = LumoInverseCalibration(
        global_stiffness_scale=float(result.x[0]),
        brace_scale=float(result.x[1]),
        damage_residual=float(result.x[2]),
        damping_ratio=damping,
        bounds=bounds,
        fit_cost=float(result.cost),
        fit_optimality=float(result.optimality),
        evaluations=int(result.nfev),
        success=bool(result.success),
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        bound_hits=bound_hits,
        damping_basis=damping_basis,
        damping_relative_mad=damping_relative_mad,
        group_hashes=group_hashes,
        group_summaries=group_summaries,
        matched_forward_mode_indices_by_group=matched_forward_mode_indices_by_group,
        fdd_settings={
            "max_modes": int(max_modes),
            "min_frequency_hz": float(min_frequency_hz),
            "max_frequency_hz": float(max_frequency_hz),
            "nperseg": int(nperseg),
            "minimum_peak_prominence_db": float(minimum_peak_prominence_db),
            "mode_shape_weight": float(mode_shape_weight),
        },
    )
    calibration.validate()
    return calibration


def write_lumo_inverse_receipt(calibration: LumoInverseCalibration, path: Path) -> Path:
    payload = calibration.to_payload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def load_lumo_inverse_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema",
        "dataset_id",
        "source_split",
        "method",
        "runner_contract",
        "identifiability",
        "parameters",
        "bounds",
        "optimizer",
        "physical_group_count",
        "group_hashes",
        "groups",
        "fdd",
        "receipt_sha256",
    }
    if set(payload) != expected:
        raise ValueError("LUMO inverse receipt has an unexpected schema shape")
    if (
        payload.get("schema") != LUMO_INVERSE_CALIBRATION_SCHEMA
        or payload.get("dataset_id") != "lumo"
        or payload.get("source_split") != "train"
        or payload.get("method") != LUMO_INVERSE_METHOD
        or payload.get("runner_contract") != LUMO_MODAL_RUNNER_CONTRACT
        or payload.get("receipt_sha256") != _payload_hash(payload)
    ):
        raise ValueError("LUMO inverse receipt identity or hash is invalid")
    groups = dict(payload.get("groups") or {})
    hashes = dict(payload.get("group_hashes") or {})
    if (
        not groups
        or set(groups) != set(hashes)
        or int(payload.get("physical_group_count", 0)) != len(groups)
        or any(summary.get("source_split") != "train" for summary in groups.values())
        or any(
            len(summary.get("matched_forward_mode_indices") or [])
            != len(summary.get("frequencies_hz") or [])
            or len(set(summary.get("matched_forward_mode_indices") or []))
            != len(summary.get("matched_forward_mode_indices") or [])
            for summary in groups.values()
        )
        or any(
            len(str(digest)) != 64
            or any(character not in "0123456789abcdef" for character in str(digest))
            for digest in hashes.values()
        )
    ):
        raise ValueError("LUMO inverse receipt group provenance is invalid")
    bounds = dict(payload.get("bounds") or {})
    parameters = dict(payload.get("parameters") or {})
    expected_parameters = {
        "global_stiffness_scale",
        "brace_scale",
        "damage_residual",
        "damping_ratio",
    }
    if set(bounds) != expected_parameters or set(parameters) != expected_parameters:
        raise ValueError("LUMO inverse receipt parameter schema is invalid")
    for name in expected_parameters:
        low, high = (float(value) for value in bounds[name])
        value = float(parameters[name])
        if not np.isfinite([low, high, value]).all() or low >= high or not low <= value <= high:
            raise ValueError(f"LUMO inverse receipt contains invalid {name}")
    optimizer = dict(payload.get("optimizer") or {})
    if (
        optimizer.get("success") is not True
        or int(optimizer.get("jacobian_rank", 0)) != 3
        or not np.isfinite(float(optimizer.get("jacobian_condition", float("nan"))))
        or float(optimizer["jacobian_condition"]) > 1.0e10
        or any(name not in expected_parameters for name in optimizer.get("bound_hits", []))
    ):
        raise ValueError("LUMO inverse receipt identifiability gate is invalid")
    return payload


__all__ = [
    "LUMO_INVERSE_CALIBRATION_SCHEMA",
    "LUMO_INVERSE_METHOD",
    "LUMO_MODAL_RUNNER_CONTRACT",
    "LumoInverseBounds",
    "LumoInverseCalibration",
    "MeasuredModalExample",
    "ModalRunner",
    "OutputOnlyModes",
    "extract_output_only_modes",
    "fit_lumo_inverse_calibration",
    "load_lumo_inverse_receipt",
    "modal_assurance",
    "modal_subspace_assurance",
    "write_lumo_inverse_receipt",
]
