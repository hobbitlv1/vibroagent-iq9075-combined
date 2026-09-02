from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .exceptions import DataContractError

AXES = ("x", "y", "z")
ALLOWED_CLASSES = (
    "normal",
    "impact_transient",
    "periodic_excitation",
    "broadband_increase",
    "modal_frequency_shift",
    "damping_change",
    "mounting_or_sensor_fault",
    "unknown_anomaly",
    "data_invalid",
)
ALLOWED_SEVERITIES = ("none", "advisory", "warning", "critical")
ALLOWED_SYSTEM_STATES = ("normal", "advisory", "warning", "critical", "data_invalid")
ALLOWED_GLOBAL_CLASSES = ("normal", "unknown_anomaly")
ALLOWED_SUPERVISION_SCOPES = ("joint", "global_only", "target_only")


@dataclass(slots=True)
class QualityReport:
    valid_fraction: float = 1.0
    missing_fraction: float = 0.0
    clipped_fraction: float = 0.0
    timestamp_monotonic: bool = True
    clock_skew_ms: float = 0.0
    axes_present: tuple[bool, bool, bool] = (True, True, True)
    usable_band_hz: float | None = None
    packet_loss_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.valid_fraction >= 0.98
            and self.timestamp_monotonic
            and any(self.axes_present)
            and self.clipped_fraction < 0.01
        )


@dataclass(slots=True)
class BoardWindow:
    board_id: str
    values: np.ndarray
    sample_rate_hz: float
    start_time_s: float
    axis_mask: np.ndarray
    sample_mask: np.ndarray
    role: Literal["reference", "target"]
    target_index: int | None = None
    sensor_position: tuple[float, float, float] | None = None
    orientation_matrix: np.ndarray | None = None
    quality: QualityReport = field(default_factory=QualityReport)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] != 3:
            raise DataContractError(f"{self.board_id}: values must have shape [3, samples], got {self.values.shape}")
        if self.axis_mask.shape != (3,):
            raise DataContractError(f"{self.board_id}: axis_mask must have shape [3]")
        if self.sample_mask.shape != (self.values.shape[1],):
            raise DataContractError(f"{self.board_id}: sample_mask length must equal signal length")
        if self.sample_rate_hz <= 0:
            raise DataContractError(f"{self.board_id}: sample_rate_hz must be positive")
        if self.role == "reference" and self.target_index is not None:
            raise DataContractError("Reference board cannot have target_index")
        if self.role == "target" and self.target_index not in {1, 2, 3, 4, 5}:
            raise DataContractError("Target board target_index must be 1..5")
        if not np.isfinite(self.values[self.axis_mask.astype(bool)]).all():
            raise DataContractError(f"{self.board_id}: finite values are required on present axes")
        if self.sensor_position is not None:
            position = np.asarray(self.sensor_position, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise DataContractError(f"{self.board_id}: sensor_position must contain three finite coordinates")
        if self.orientation_matrix is not None:
            from .signal.orientation import validate_orientation_matrix

            validate_orientation_matrix(self.orientation_matrix)


@dataclass(slots=True)
class TargetLabel:
    sensor_id: str
    affected: bool
    class_name: str
    severity: str
    source: str

    def validate(self) -> None:
        if self.class_name not in ALLOWED_CLASSES:
            raise DataContractError(f"Unknown class: {self.class_name}")
        if self.severity not in ALLOWED_SEVERITIES:
            raise DataContractError(f"Unknown severity: {self.severity}")
        if self.class_name == "normal" and (self.affected or self.severity != "none"):
            raise DataContractError("normal requires affected=false and severity=none")
        if self.class_name == "data_invalid" and (self.affected or self.severity != "advisory"):
            raise DataContractError("data_invalid requires affected=false and severity=advisory")
        if self.class_name not in {"normal", "data_invalid"} and not self.affected:
            raise DataContractError("An anomaly class requires affected=true")


@dataclass(slots=True)
class EpisodeLabel:
    """Supervision available for one six-board episode.

    ``global_class_name`` is available for every supervised source. Target rows are
    optional because a source-level change label does not establish which sensors
    are affected. ``supervision_scope`` is a training-time view used to factorize
    Gemma supervision into one global decision and independent target decisions.
    """

    targets: list[TargetLabel]
    explanation_target: str = ""
    global_class_name: str | None = None
    global_severity: str | None = None
    target_supervision_available: bool = True
    supervision_scope: str = "joint"
    focus_target_index: int | None = None

    def __post_init__(self) -> None:
        if self.global_class_name is None:
            self.global_class_name = (
                "normal"
                if self.targets and all(target.class_name == "normal" for target in self.targets)
                else "unknown_anomaly"
            )
        if self.global_severity is None:
            order = {"none": 0, "advisory": 1, "warning": 2, "critical": 3}
            self.global_severity = max(
                (target.severity for target in self.targets),
                key=lambda value: order.get(value, -1),
                default=("none" if self.global_class_name == "normal" else "advisory"),
            )

    @property
    def is_globally_healthy(self) -> bool:
        return self.global_class_name == "normal"

    @property
    def affected_target_indices(self) -> tuple[int, ...]:
        if not self.target_supervision_available:
            return ()
        return tuple(
            index
            for index, target in enumerate(self.targets, start=1)
            if target.affected
        )

    def validate(self) -> None:
        if self.global_class_name not in ALLOWED_GLOBAL_CLASSES:
            raise DataContractError(
                f"Global class must be one of {ALLOWED_GLOBAL_CLASSES}, got {self.global_class_name!r}"
            )
        if self.global_severity not in ALLOWED_SEVERITIES:
            raise DataContractError(f"Unknown global severity: {self.global_severity}")
        if self.global_class_name == "normal" and self.global_severity != "none":
            raise DataContractError("global normal requires severity=none")
        if self.global_class_name != "normal" and self.global_severity == "none":
            raise DataContractError("global anomaly requires non-none severity")
        if self.supervision_scope not in ALLOWED_SUPERVISION_SCOPES:
            raise DataContractError(
                f"supervision_scope must be one of {ALLOWED_SUPERVISION_SCOPES}"
            )
        if self.supervision_scope == "global_only":
            if self.target_supervision_available:
                raise DataContractError(
                    "global_only supervision must not claim target labels are available"
                )
            if self.targets:
                raise DataContractError(
                    "global_only supervision must carry no target labels; missing is not normal"
                )
            if self.focus_target_index is not None:
                raise DataContractError("global_only supervision cannot focus one target")
            return
        if not self.target_supervision_available:
            raise DataContractError(
                "joint/target_only supervision requires genuine target labels"
            )
        if len(self.targets) != 5:
            raise DataContractError(
                "Target-supervised EpisodeLabel must contain exactly five target labels"
            )
        expected = [f"target_{index}" for index in range(1, 6)]
        if [target.sensor_id for target in self.targets] != expected:
            raise DataContractError(f"Target order must be {expected}")
        for target in self.targets:
            target.validate()
        if self.supervision_scope == "target_only":
            if self.focus_target_index not in {1, 2, 3, 4, 5}:
                raise DataContractError("target_only supervision requires focus_target_index=1..5")
        elif self.focus_target_index is not None:
            raise DataContractError("joint supervision cannot set focus_target_index")

        derived_global = (
            "unknown_anomaly"
            if any(target.class_name not in {"normal", "data_invalid"} for target in self.targets)
            else "normal"
        )
        if derived_global != self.global_class_name:
            raise DataContractError(
                "Global and target labels disagree in a target-supervised episode"
            )


@dataclass(slots=True)
class EpisodeProvenance:
    dataset_id: str
    provenance_type: str
    source_files: list[str]
    source_checksums: list[str]
    licence: str
    licence_accepted: bool
    structure_id: str
    session_id: str
    experiment_id: str
    split_group: str
    derived_operations: list[str] = field(default_factory=list)
    local_training_data: bool = False


@dataclass(slots=True)
class SixBoardEpisode:
    episode_id: str
    boards: list[BoardWindow]
    provenance: EpisodeProvenance
    label: EpisodeLabel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_label: bool = False) -> None:
        if len(self.boards) != 6:
            raise DataContractError(f"Expected six boards, got {len(self.boards)}")
        if self.boards[0].role != "reference":
            raise DataContractError("Board 0 must be the reference")
        for index, board in enumerate(self.boards):
            board.validate()
            if index > 0 and (board.role != "target" or board.target_index != index):
                raise DataContractError(f"Board {index} must be target_{index}")
        lengths = {board.values.shape[1] for board in self.boards}
        rates = {round(board.sample_rate_hz, 9) for board in self.boards}
        if len(lengths) != 1 or len(rates) != 1:
            raise DataContractError("All boards in an episode must share length and sample rate")
        if self.provenance.local_training_data and require_label:
            raise DataContractError("Local/live recordings are forbidden in labelled training episodes")
        if require_label and self.label is None:
            raise DataContractError("Label required")
        if self.label is not None:
            self.label.validate()

    def stacked_values(self) -> np.ndarray:
        self.validate()
        return np.stack([board.values for board in self.boards], axis=0).astype(np.float32, copy=False)

    def stacked_axis_mask(self) -> np.ndarray:
        return np.stack([board.axis_mask for board in self.boards], axis=0).astype(np.bool_, copy=False)

    def stacked_sample_mask(self) -> np.ndarray:
        return np.stack([board.sample_mask for board in self.boards], axis=0).astype(np.bool_, copy=False)


@dataclass(slots=True)
class TargetPrediction:
    sensor_id: str
    affected: bool
    class_name: str
    severity: str
    choice_probability: float | None = None
    raw_choice_probability: float | None = None
    confidence_operational: bool = False

    def validate(self) -> None:
        if self.sensor_id not in {f"target_{index}" for index in range(1, 6)}:
            raise DataContractError(
                f"Prediction sensor_id must be target_1..target_5, got {self.sensor_id!r}"
            )
        TargetLabel(
            sensor_id=self.sensor_id,
            affected=self.affected,
            class_name=self.class_name,
            severity=self.severity,
            source="prediction",
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "affected": self.affected,
            "class": self.class_name,
            "severity": self.severity,
            "choice_probability": self.choice_probability,
            "raw_choice_probability": self.raw_choice_probability,
            "confidence_operational": self.confidence_operational,
        }


@dataclass(slots=True)
class ClassificationResult:
    schema_version: str
    window: dict[str, Any]
    system_state: str
    targets: list[TargetPrediction]
    explanation: str
    quality: dict[str, Any]
    model: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise DataContractError("Unsupported schema version")
        if self.system_state not in ALLOWED_SYSTEM_STATES:
            raise DataContractError(f"Invalid system_state: {self.system_state}")
        if len(self.targets) != 5:
            raise DataContractError("Exactly five target predictions are required")
        expected = [f"target_{index}" for index in range(1, 6)]
        observed = [target.sensor_id for target in self.targets]
        if observed != expected:
            raise DataContractError(
                f"Target prediction order must be {expected}, got {observed}"
            )
        for target in self.targets:
            target.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "window": self.window,
            "system_state": self.system_state,
            "targets": [target.to_dict() for target in self.targets],
            "explanation": self.explanation,
            "quality": self.quality,
            "model": self.model,
            "evidence": self.evidence,
        }


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
