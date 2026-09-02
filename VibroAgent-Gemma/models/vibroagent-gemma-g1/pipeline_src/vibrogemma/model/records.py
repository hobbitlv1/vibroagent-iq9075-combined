from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..contracts import (
    ALLOWED_CLASSES,
    ALLOWED_GLOBAL_CLASSES,
    ALLOWED_SEVERITIES,
    EpisodeLabel,
    TargetPrediction,
)
from ..exceptions import DataContractError


@dataclass(frozen=True, slots=True)
class TargetChoice:
    affected: bool
    class_name: str
    severity: str

    def encode(self, target_index: int) -> str:
        return f"\nT{target_index}|{1 if self.affected else 0}|{self.class_name}|{self.severity}"


@dataclass(frozen=True, slots=True)
class GlobalChoice:
    class_name: str
    severity: str

    def encode(self) -> str:
        return f"\nG|{self.class_name}|{self.severity}"


@dataclass(frozen=True, slots=True)
class GlobalPrediction:
    class_name: str
    severity: str
    choice_probability: float | None = None

    @property
    def anomalous(self) -> bool:
        return self.class_name != "normal"


class GlobalRecordCodec:
    """Constrained episode-level record used by partially labelled sources."""

    header = "GLOBAL_CLASSIFICATION"
    footer = "\nEND"
    line_pattern = re.compile(r"^G\|(?P<class>[a-z_]+)\|(?P<severity>[a-z_]+)$")

    def __init__(
        self,
        classes: Iterable[str] = ALLOWED_GLOBAL_CLASSES,
        severities: Iterable[str] = ("none", "advisory"),
    ) -> None:
        self.classes = tuple(dict.fromkeys(str(value) for value in classes))
        self.severities = tuple(dict.fromkeys(str(value) for value in severities))
        if set(self.classes) - set(ALLOWED_GLOBAL_CLASSES):
            raise DataContractError("Global record supports only normal and unknown_anomaly")
        if "normal" not in self.classes or "unknown_anomaly" not in self.classes:
            raise DataContractError("Global record requires normal and unknown_anomaly")
        if "none" not in self.severities:
            raise DataContractError("Global record requires severity none")
        anomaly_severities = [value for value in self.severities if value != "none"]
        if not anomaly_severities:
            raise DataContractError("Global anomaly requires at least one non-none severity")
        self.choices = (
            GlobalChoice("normal", "none"),
            *(GlobalChoice("unknown_anomaly", value) for value in anomaly_severities),
        )

    def choice_strings(self) -> list[str]:
        return [choice.encode() for choice in self.choices]

    def encode_label(self, label: EpisodeLabel) -> str:
        label.validate()
        choice = GlobalChoice(str(label.global_class_name), str(label.global_severity))
        if choice not in self.choices:
            raise DataContractError(f"Global label is outside the configured vocabulary: {choice}")
        return f"{self.header}{choice.encode()}\nEND"

    def parse(self, text: str, probability: float | None = None) -> GlobalPrediction:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) != 3 or lines[0] != self.header or lines[-1] != "END":
            raise DataContractError("Global record must have exactly three non-empty lines")
        match = self.line_pattern.fullmatch(lines[1])
        if match is None:
            raise DataContractError(f"Invalid global line: {lines[1]!r}")
        choice = GlobalChoice(match.group("class"), match.group("severity"))
        if choice not in self.choices:
            raise DataContractError(f"Semantically invalid global tuple: {lines[1]}")
        return GlobalPrediction(choice.class_name, choice.severity, probability)


class TargetRecordCodec:
    """One-target record decoded independently to break row-to-row dependence."""

    header = "TARGET_CLASSIFICATION"
    footer = "\nEND"
    line_pattern = re.compile(
        r"^T(?P<index>[1-5])\|(?P<affected>[01])\|(?P<class>[a-z_]+)\|(?P<severity>[a-z_]+)$"
    )

    def __init__(
        self,
        classes: Iterable[str] = ALLOWED_CLASSES,
        severities: Iterable[str] = ALLOWED_SEVERITIES,
    ) -> None:
        self.classes = tuple(dict.fromkeys(str(value) for value in classes))
        self.severities = tuple(dict.fromkeys(str(value) for value in severities))
        unknown = set(self.classes) - set(ALLOWED_CLASSES)
        if unknown:
            raise DataContractError(f"Unsupported classes: {sorted(unknown)}")
        unknown_severities = set(self.severities) - set(ALLOWED_SEVERITIES)
        if unknown_severities:
            raise DataContractError(f"Unsupported severities: {sorted(unknown_severities)}")
        self.choices = ClassificationRecordCodec._build_choice_set(
            self.classes, self.severities
        )

    def choice_strings(self, target_index: int) -> list[str]:
        if target_index not in {1, 2, 3, 4, 5}:
            raise ValueError("target_index must be 1..5")
        return [choice.encode(target_index) for choice in self.choices]

    def encode_label(self, label: EpisodeLabel) -> str:
        label.validate()
        index = label.focus_target_index
        if label.supervision_scope != "target_only" or index not in {1, 2, 3, 4, 5}:
            raise DataContractError("Target record requires target_only supervision")
        target = label.targets[index - 1]
        choice = TargetChoice(target.affected, target.class_name, target.severity)
        if choice not in self.choices:
            raise DataContractError(f"Target label is outside the configured vocabulary: {choice}")
        return f"{self.header}{choice.encode(index)}\nEND"

    def parse(
        self,
        text: str,
        *,
        target_index: int,
        probability: float | None = None,
    ) -> TargetPrediction:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) != 3 or lines[0] != self.header or lines[-1] != "END":
            raise DataContractError("Target record must have exactly three non-empty lines")
        match = self.line_pattern.fullmatch(lines[1])
        if match is None:
            raise DataContractError(f"Invalid target line: {lines[1]!r}")
        observed_index = int(match.group("index"))
        if observed_index != int(target_index):
            raise DataContractError(
                f"Expected independently decoded target {target_index}, got {observed_index}"
            )
        choice = TargetChoice(
            match.group("affected") == "1",
            match.group("class"),
            match.group("severity"),
        )
        if choice not in self.choices:
            raise DataContractError(f"Semantically invalid target tuple: {lines[1]}")
        prediction = TargetPrediction(
            sensor_id=f"target_{target_index}",
            affected=choice.affected,
            class_name=choice.class_name,
            severity=choice.severity,
            choice_probability=probability,
            raw_choice_probability=probability,
            confidence_operational=False,
        )
        prediction.validate()
        return prediction


class ClassificationRecordCodec:
    """Backward-compatible assembled five-target record for the public schema."""

    header = "CLASSIFICATION"
    footer = "\nEND"
    line_pattern = re.compile(
        r"^T(?P<index>[1-5])\|(?P<affected>[01])\|(?P<class>[a-z_]+)\|(?P<severity>[a-z_]+)$"
    )

    def __init__(
        self,
        classes: Iterable[str] = ALLOWED_CLASSES,
        severities: Iterable[str] = ALLOWED_SEVERITIES,
    ) -> None:
        self.classes = tuple(dict.fromkeys(str(value) for value in classes))
        self.severities = tuple(dict.fromkeys(str(value) for value in severities))
        unknown = set(self.classes) - set(ALLOWED_CLASSES)
        if unknown:
            raise DataContractError(f"Unsupported classes: {sorted(unknown)}")
        unknown_severities = set(self.severities) - set(ALLOWED_SEVERITIES)
        if unknown_severities:
            raise DataContractError(f"Unsupported severities: {sorted(unknown_severities)}")
        if "normal" not in self.classes:
            raise DataContractError("The constrained vocabulary must include normal")
        if "none" not in self.severities:
            raise DataContractError("The constrained severity vocabulary must include none")
        self.choices = self._build_choice_set(self.classes, self.severities)

    @staticmethod
    def _build_choice_set(
        classes: Iterable[str], severities: Iterable[str]
    ) -> tuple[TargetChoice, ...]:
        classes = tuple(classes)
        severities = tuple(severities)
        choices = [TargetChoice(False, "normal", "none")]
        if "data_invalid" in classes:
            if "advisory" not in severities:
                raise DataContractError("data_invalid requires advisory severity support")
            choices.append(TargetChoice(False, "data_invalid", "advisory"))
        anomaly_severities = [value for value in severities if value != "none"]
        for class_name in classes:
            if class_name in {"normal", "data_invalid"}:
                continue
            if not anomaly_severities:
                raise DataContractError(f"Anomaly class {class_name!r} needs a non-none severity")
            for severity in anomaly_severities:
                choices.append(TargetChoice(True, class_name, severity))
        return tuple(choices)

    def _build_choices(self) -> tuple[TargetChoice, ...]:
        return self._build_choice_set(self.classes, self.severities)

    def target_choice_strings(self, target_index: int) -> list[str]:
        return [choice.encode(target_index) for choice in self.choices]

    def encode_label(self, label: EpisodeLabel) -> str:
        label.validate()
        if not label.target_supervision_available:
            raise DataContractError("Cannot encode unavailable target labels as a joint record")
        return self.assemble(
            [
                TargetPrediction(
                    sensor_id=target.sensor_id,
                    affected=target.affected,
                    class_name=target.class_name,
                    severity=target.severity,
                )
                for target in label.targets
            ]
        )

    def assemble(self, predictions: Iterable[TargetPrediction]) -> str:
        values = list(predictions)
        if len(values) != 5:
            raise DataContractError("Exactly five target decisions are required")
        lines = [self.header]
        for index, prediction in enumerate(values, start=1):
            prediction.validate()
            if prediction.sensor_id != f"target_{index}":
                raise DataContractError("Target decisions must be ordered target_1..target_5")
            choice = TargetChoice(
                prediction.affected, prediction.class_name, prediction.severity
            )
            if choice not in self.choices:
                raise DataContractError(f"Decision outside configured vocabulary: {choice}")
            lines.append(choice.encode(index).lstrip("\n"))
        lines.append("END")
        return "\n".join(lines)

    def parse(
        self, text: str, probabilities: list[float | None] | None = None
    ) -> list[TargetPrediction]:
        normalized = text.strip()
        if not normalized.startswith(self.header):
            raise DataContractError("Classification record lacks header")
        if not normalized.endswith("END"):
            raise DataContractError("Classification record lacks footer")
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if len(lines) != 7 or lines[0] != self.header or lines[-1] != "END":
            raise DataContractError(
                f"Classification record must have seven non-empty lines, got {len(lines)}"
            )
        predictions: list[TargetPrediction] = []
        for expected_index, line in enumerate(lines[1:6], start=1):
            match = self.line_pattern.fullmatch(line)
            if match is None:
                raise DataContractError(f"Invalid target line: {line!r}")
            index = int(match.group("index"))
            if index != expected_index:
                raise DataContractError(f"Expected target {expected_index}, got {index}")
            class_name = match.group("class")
            severity = match.group("severity")
            affected = match.group("affected") == "1"
            probability = probabilities[expected_index - 1] if probabilities else None
            prediction = TargetPrediction(
                sensor_id=f"target_{index}",
                affected=affected,
                class_name=class_name,
                severity=severity,
                choice_probability=probability,
                raw_choice_probability=probability,
                confidence_operational=False,
            )
            prediction.validate()
            if TargetChoice(affected, class_name, severity) not in self.choices:
                raise DataContractError(f"Semantically invalid target tuple: {line}")
            predictions.append(prediction)
        return predictions

    @staticmethod
    def system_state(
        predictions: list[TargetPrediction],
        *,
        global_class_name: str | None = None,
        global_severity: str | None = None,
    ) -> str:
        if any(prediction.class_name == "data_invalid" for prediction in predictions):
            return "data_invalid"
        severities = [prediction.severity for prediction in predictions]
        if global_class_name == "unknown_anomaly" and global_severity is not None:
            severities.append(global_severity)
        for severity in ("critical", "warning", "advisory"):
            if severity in severities:
                return severity
        return "normal"
