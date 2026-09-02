from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from ..contracts import ClassificationResult


@dataclass(slots=True)
class AlertDecision:
    emit: bool
    active: bool
    reason: str


class GeneratedOutputAlertPolicy:
    """Rolling persistence and hysteresis applied only to Gemma outputs.

    No separate anomaly score is introduced. The policy merges overlapping
    non-normal windows into one operational event and emits only on activation.
    """

    def __init__(
        self,
        *,
        trigger_count: int = 3,
        trigger_window: int = 5,
        clear_after_consecutive_normal: int = 3,
        cooldown_seconds: float = 30.0,
        alert_on_data_invalid: bool = True,
    ) -> None:
        self.trigger_count = int(trigger_count)
        self.trigger_window = int(trigger_window)
        if self.trigger_count < 1 or self.trigger_window < self.trigger_count:
            raise ValueError("trigger_window must be >= trigger_count >= 1")
        self.clear_normal = int(clear_after_consecutive_normal)
        if self.clear_normal < 1:
            raise ValueError("clear_after_consecutive_normal must be >= 1")
        self.cooldown_seconds = float(cooldown_seconds)
        self.alert_on_data_invalid = bool(alert_on_data_invalid)
        self.history: deque[bool] = deque(maxlen=self.trigger_window)
        self.normal_count = 0
        self.active = False
        self.last_emit_monotonic = float("-inf")

    def reset(self) -> None:
        self.history.clear()
        self.normal_count = 0
        self.active = False
        self.last_emit_monotonic = float("-inf")

    def evaluate(self, result: ClassificationResult, *, now_monotonic: float | None = None) -> AlertDecision:
        non_normal = any(target.affected for target in result.targets)
        if self.alert_on_data_invalid:
            non_normal = non_normal or any(target.class_name == "data_invalid" for target in result.targets)
        self.history.append(non_normal)

        if self.active:
            if non_normal:
                self.normal_count = 0
                return AlertDecision(False, True, "active_event_merged")
            self.normal_count += 1
            if self.normal_count >= self.clear_normal:
                self.active = False
                self.normal_count = 0
                self.history.clear()
                return AlertDecision(False, False, "cleared_after_generated_normal_outputs")
            return AlertDecision(False, True, "waiting_for_clear_hysteresis")

        self.normal_count = 0
        votes = sum(self.history)
        if len(self.history) < self.trigger_count or votes < self.trigger_count:
            return AlertDecision(False, False, "waiting_for_generated_output_persistence")

        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if now - self.last_emit_monotonic < self.cooldown_seconds:
            return AlertDecision(False, False, "popup_cooldown")
        self.last_emit_monotonic = now
        self.active = True
        return AlertDecision(True, True, "gemma_generated_persistent_non_normal_event")
