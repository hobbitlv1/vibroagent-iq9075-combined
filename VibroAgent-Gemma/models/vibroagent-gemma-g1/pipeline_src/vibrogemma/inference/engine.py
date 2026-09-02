from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ..config import resolve_from_config
from ..contracts import ClassificationResult, SixBoardEpisode
from ..data.serialization import save_episode
from ..model.pipeline import VibroGemmaForClassification
from .notifier import DesktopNotifier
from .policy import GeneratedOutputAlertPolicy
from .store import EventStore


class InferenceEngine:
    def __init__(
        self,
        *,
        model: VibroGemmaForClassification,
        checkpoint_name: str,
        store: EventStore,
        policy: GeneratedOutputAlertPolicy,
        notifier: DesktopNotifier,
        schema_path: str | Path,
        alert_window_dir: str | Path,
        persist_alert_windows: bool,
        persist_all_windows: bool,
    ) -> None:
        self.model = model
        self.checkpoint_name = checkpoint_name
        self.store = store
        self.policy = policy
        self.notifier = notifier
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.validator = Draft202012Validator(schema)
        self.alert_window_dir = Path(alert_window_dir)
        self.persist_alert_windows = persist_alert_windows
        self.persist_all_windows = persist_all_windows

    def process(self, episode: SixBoardEpisode) -> tuple[ClassificationResult, int, bool]:
        result = self.model.classify(episode, checkpoint_name=self.checkpoint_name)
        self.validator.validate(result.to_dict())
        decision = self.policy.evaluate(result)
        window_path = None
        if self.persist_all_windows or (self.persist_alert_windows and decision.emit):
            array_path, _ = save_episode(episode, self.alert_window_dir)
            window_path = str(array_path)
        event_id = self.store.insert(
            result,
            alert_emitted=decision.emit,
            alert_reason=decision.reason,
            window_path=window_path,
        )
        if decision.emit:
            self.notifier.send(result)
        return result, event_id, decision.emit
