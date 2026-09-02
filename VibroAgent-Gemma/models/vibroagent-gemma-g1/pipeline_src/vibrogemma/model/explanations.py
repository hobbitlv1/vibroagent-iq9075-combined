from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..contracts import EpisodeLabel, TargetPrediction


_SIGNAL_TERMS = {
    "rms",
    "modal",
    "frequency",
    "hz",
    "transmissibility",
    "db",
    "coherence",
    "spectral",
    "band",
    "profile",
    "deviation",
    "kurtosis",
    "crest",
    "entropy",
    "reference",
}
_GENERIC_PATTERNS = (
    "based on the provided evidence",
    "no anomalies were detected",
    "the evidence indicates",
    "classified as an unknown anomaly",
    "classified all target sensors as normal",
)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True, slots=True)
class ExplanationDecision:
    text: str
    source: str
    grounded: bool


def _prediction_like(label: EpisodeLabel) -> list[TargetPrediction]:
    return [
        TargetPrediction(
            sensor_id=target.sensor_id,
            affected=target.affected,
            class_name=target.class_name,
            severity=target.severity,
        )
        for target in label.targets
    ]


def _iter_numeric_values(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            yield numeric
        return
    if isinstance(value, str):
        numeric_text = re.sub(r"(?<=\d)[_-](?=\d)", " ", value)
        for token in _NUMBER_PATTERN.findall(numeric_text):
            yield float(token)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if not re.fullmatch(r"(?:target_|T)[1-5]", key_text, flags=re.IGNORECASE):
                yield from _iter_numeric_values(key_text)
            yield from _iter_numeric_values(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_numeric_values(item)


def _feature_text(value: str) -> str:
    return value.replace("::", " ").replace("_", " ")


class EvidenceExplanationRenderer:
    """Create concise explanations that can be traced to emitted evidence.

    Gemma is still trained to emit a sentence. At inference the generated prose
    is accepted only when it passes a grounding check; otherwise this renderer
    produces a conservative deterministic sentence from the same evidence.
    """

    def training_target(self, label: EpisodeLabel, evidence: dict[str, Any]) -> str:
        return self.render(_prediction_like(label), evidence).text

    def render(
        self,
        predictions: Sequence[TargetPrediction],
        evidence: dict[str, Any],
        *,
        invalid_targets: Sequence[str] = (),
    ) -> ExplanationDecision:
        invalid = list(dict.fromkeys(str(value) for value in invalid_targets))
        if invalid:
            names = ", ".join(invalid)
            return ExplanationDecision(
                text=(
                    f"Signal quality is insufficient for {names}; no vibration diagnosis is issued "
                    "for those sensors."
                ),
                source="deterministic_quality_evidence",
                grounded=True,
            )

        affected = [item.sensor_id for item in predictions if item.affected]
        if not affected:
            strongest = self._strongest_profile_change(evidence, [f"target_{index}" for index in range(1, 6)])
            if strongest is not None:
                sensor, feature, z_value = strongest
                return ExplanationDecision(
                    text=(
                        "All five target responses are classified normal. "
                        f"The largest available healthy-profile deviation is {sensor} {_feature_text(feature)} "
                        f"(z={z_value:.2f}); no specific fault mechanism is inferred."
                    ),
                    source="deterministic_physics_evidence",
                    grounded=True,
                )
            return ExplanationDecision(
                text=(
                    "All five target responses are classified normal from the reference-conditioned "
                    "multi-resolution evidence; no specific fault mechanism is inferred."
                ),
                source="deterministic_physics_evidence",
                grounded=True,
            )

        prefix = (
            "All five targets are classified as unexplained anomalies."
            if len(affected) == 5
            else f"{', '.join(affected)} are classified as unexplained anomalies."
        )
        details = [detail for sensor in affected[:2] if (detail := self._target_detail(evidence, sensor))]
        detail_text = " " + " ".join(details) if details else ""
        return ExplanationDecision(
            text=(
                prefix
                + detail_text
                + " The evidence does not identify a specific damage mechanism."
            ),
            source="deterministic_physics_evidence",
            grounded=True,
        )

    def choose(
        self,
        generated_text: str,
        predictions: Sequence[TargetPrediction],
        evidence: dict[str, Any],
        *,
        invalid_targets: Sequence[str] = (),
    ) -> ExplanationDecision:
        fallback = self.render(predictions, evidence, invalid_targets=invalid_targets)
        compact = " ".join(str(generated_text).strip().split())[:500]
        if not compact:
            return fallback
        if not self.is_grounded(compact, predictions, evidence):
            return fallback
        return ExplanationDecision(text=compact, source="gemma_grounded", grounded=True)

    def is_grounded(
        self,
        text: str,
        predictions: Sequence[TargetPrediction],
        evidence: dict[str, Any],
    ) -> bool:
        lower = text.lower()
        if any(pattern in lower for pattern in _GENERIC_PATTERNS):
            return False
        affected = [item.sensor_id for item in predictions if item.affected]
        if affected and not any(sensor in lower for sensor in affected) and "all five" not in lower:
            return False
        if affected and not any(term in lower for term in _SIGNAL_TERMS):
            return False
        evidence_numbers = list(_iter_numeric_values(evidence))
        measurement_text = re.sub(r"(?:target_|T)[1-5]", "TARGET", text, flags=re.IGNORECASE)
        measurement_text = re.sub(r"(?<=\d)-(?=\d)", " ", measurement_text)
        for token in _NUMBER_PATTERN.findall(measurement_text):
            value = float(token)
            if not any(math.isclose(value, candidate, rel_tol=0.015, abs_tol=0.015) for candidate in evidence_numbers):
                return False
        return True

    @staticmethod
    def _boards(evidence: dict[str, Any]) -> dict[str, Any]:
        boards = evidence.get("boards")
        return boards if isinstance(boards, dict) else {}

    def _strongest_profile_change(
        self,
        evidence: dict[str, Any],
        sensors: Sequence[str],
    ) -> tuple[str, str, float] | None:
        candidate: tuple[str, str, float] | None = None
        boards = self._boards(evidence)
        for sensor in sensors:
            board = boards.get(sensor, {})
            changes = (
                board.get("normal_distribution_change", {})
                .get("top_absolute_z", [])
            )
            if not changes:
                continue
            first = changes[0]
            value = float(first.get("z", 0.0))
            item = (sensor, str(first.get("feature", "feature")), value)
            if candidate is None or abs(value) > abs(candidate[2]):
                candidate = item
        return candidate

    def _target_detail(self, evidence: dict[str, Any], sensor: str) -> str | None:
        boards = self._boards(evidence)
        board = boards.get(sensor)
        if not isinstance(board, dict):
            return None

        profile = self._strongest_profile_change(evidence, [sensor])
        if profile is not None:
            _, feature, z_value = profile
            return f"{sensor}'s largest healthy-profile deviation is {_feature_text(feature)} (z={z_value:.2f})."

        relation = board.get("reference_relation", {})
        transmissibility = relation.get("transmissibility_db", {}) if isinstance(relation, dict) else {}
        strongest_ratio: tuple[str, float] | None = None
        if isinstance(transmissibility, dict):
            for axis, values in transmissibility.items():
                if not isinstance(values, dict):
                    continue
                for band, raw in values.items():
                    value = float(raw)
                    item = (f"{axis}/{band}", value)
                    if strongest_ratio is None or abs(value) > abs(strongest_ratio[1]):
                        strongest_ratio = item
        if strongest_ratio is not None:
            location, value = strongest_ratio
            return f"{sensor} differs from the reference by {value:+.2f} dB at {location}."

        reference = boards.get("reference", {})
        target_modal = board.get("modal_estimates", [])
        reference_modal = reference.get("modal_estimates", []) if isinstance(reference, dict) else []
        if target_modal and reference_modal:
            delta = float(target_modal[0]["frequency_hz"]) - float(reference_modal[0]["frequency_hz"])
            return f"{sensor}'s strongest modal estimate differs from the reference by {delta:+.2f} Hz."

        coherences = []
        coherence = relation.get("coherence", {}) if isinstance(relation, dict) else {}
        if isinstance(coherence, dict):
            for values in coherence.values():
                if isinstance(values, dict):
                    coherences.extend(float(value) for value in values.values())
        if coherences:
            mean_value = sum(coherences) / len(coherences)
            return f"{sensor}'s mean reference coherence is {mean_value:.2f}."

        return f"{sensor}'s multi-resolution response differs from the reference."
