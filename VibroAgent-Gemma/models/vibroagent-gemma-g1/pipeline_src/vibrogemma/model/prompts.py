from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..contracts import EpisodeLabel, SixBoardEpisode
from ..exceptions import ModelCompatibilityError
from .explanations import EvidenceExplanationRenderer
from .gemma_bridge import MixedEmbeddingBatch, build_mixed_embedding_batch
from .records import (
    ClassificationRecordCodec,
    GlobalRecordCodec,
    TargetRecordCodec,
)


@dataclass(slots=True)
class PromptEncoding:
    input_ids: list[int]
    slot_positions: list[int]
    prompt_length: int
    labels: list[int] | None
    loss_weights: list[float] | None


@dataclass(slots=True)
class StructuredDecisionPromptBatch:
    mixed_batch: MixedEmbeddingBatch
    decision_positions: torch.Tensor


class SoftTokenPromptBuilder:
    """Build mixed soft-token prompts for joint or factorized supervision.

    The final localization protocol trains an episode-level global task on every
    labelled source and independent target tasks only when genuine target labels
    exist. This prevents globally labelled datasets from being translated into
    all-five pseudo-labels and removes teacher-forced row-to-row dependence from
    the target task.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        tokens_per_board: int,
        system_prompt: str,
        marker_text: dict[str, str],
        codec: ClassificationRecordCodec,
        global_codec: GlobalRecordCodec | None = None,
        target_codec: TargetRecordCodec | None = None,
        classification_loss_weight: float = 1.0,
        explanation_loss_weight: float = 1.0,
        eos_loss_weight: float = 1.0,
        global_task_weight: float = 1.0,
        target_task_weight: float = 1.0,
        joint_task_weight: float = 1.0,
        target_row_balance: bool = False,
        prompt_prior_guard: bool = False,
        factorized_training: bool = False,
        target_decision_semantics: str = "response_anomaly",
        target_focus_layout: str = "fixed",
        target_evidence_scope: str = "focus_only",
        classification_structure_weight: float = 1.0,
        global_decision_weight: float = 1.0,
        target_decision_weight: float = 1.0,
        joint_decision_weight: float = 1.0,
        structured_decision_separator: str = " ",
        structured_target_query_order: str = "fixed",
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens_per_board = int(tokens_per_board)
        self.system_prompt = system_prompt
        self.marker_text = marker_text
        self.codec = codec
        self.global_codec = global_codec or GlobalRecordCodec(
            ("normal", "unknown_anomaly"),
            tuple(value for value in codec.severities if value in {"none", "advisory", "warning", "critical"}),
        )
        self.target_codec = target_codec or TargetRecordCodec(codec.classes, codec.severities)
        self.classification_loss_weight = float(classification_loss_weight)
        self.explanation_loss_weight = float(explanation_loss_weight)
        self.eos_loss_weight = float(eos_loss_weight)
        self.global_task_weight = float(global_task_weight)
        self.target_task_weight = float(target_task_weight)
        self.joint_task_weight = float(joint_task_weight)
        self.target_row_balance = bool(target_row_balance)
        self.prompt_prior_guard = bool(prompt_prior_guard)
        self.factorized_training = bool(factorized_training)
        self.target_decision_semantics = str(target_decision_semantics)
        self.target_focus_layout = str(target_focus_layout)
        self.target_evidence_scope = str(target_evidence_scope)
        self.classification_structure_weight = float(classification_structure_weight)
        self.global_decision_weight = float(global_decision_weight)
        self.target_decision_weight = float(target_decision_weight)
        self.joint_decision_weight = float(joint_decision_weight)
        self.structured_decision_separator = str(structured_decision_separator)
        self.structured_target_query_order = str(structured_target_query_order)
        if not self.structured_decision_separator:
            raise ValueError("structured_decision_separator must be non-empty")
        if "\n" in self.structured_decision_separator or "\r" in self.structured_decision_separator:
            raise ValueError("structured_decision_separator must remain on one line")
        if self.structured_target_query_order not in {
            "fixed",
            "deterministic_per_episode",
        }:
            raise ValueError(
                "structured_target_query_order must be fixed or deterministic_per_episode"
            )
        if self.target_decision_semantics not in {"response_anomaly", "damage_source_zone"}:
            raise ValueError(
                "target_decision_semantics must be response_anomaly or damage_source_zone"
            )
        if self.target_focus_layout not in {"fixed", "focus_last"}:
            raise ValueError("target_focus_layout must be fixed or focus_last")
        if self.target_evidence_scope not in {"focus_only", "comparative_all_targets"}:
            raise ValueError(
                "target_evidence_scope must be focus_only or comparative_all_targets"
            )
        strictly_positive = {
            "classification_loss_weight": self.classification_loss_weight,
            "eos_loss_weight": self.eos_loss_weight,
            "global_task_weight": self.global_task_weight,
            "target_task_weight": self.target_task_weight,
            "global_decision_weight": self.global_decision_weight,
            "target_decision_weight": self.target_decision_weight,
            "joint_decision_weight": self.joint_decision_weight,
        }
        nonnegative = {
            "explanation_loss_weight": self.explanation_loss_weight,
            "joint_task_weight": self.joint_task_weight,
            "classification_structure_weight": self.classification_structure_weight,
        }
        invalid_positive = {name: value for name, value in strictly_positive.items() if value <= 0}
        invalid_nonnegative = {name: value for name, value in nonnegative.items() if value < 0}
        if invalid_positive or invalid_nonnegative:
            raise ValueError(
                "invalid language-model loss weights: "
                f"strictly_positive={invalid_positive}, nonnegative={invalid_nonnegative}"
            )
        if not self.factorized_training and self.joint_task_weight <= 0:
            raise ValueError("joint_task_weight must be positive when factorized training is disabled")
        self.explanation_renderer = EvidenceExplanationRenderer()
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ModelCompatibilityError("Tokenizer needs pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = int(tokenizer.pad_token_id)
        placeholder_ids = self._tokenize("?")
        self.decision_placeholder_token_id = (
            int(placeholder_ids[0]) if len(placeholder_ids) == 1 else self.pad_token_id
        )

    def _marker(self, board_index: int) -> str:
        return f"[[VIBROGEMMA_BOARD_{board_index}_SOFT_TOKENS]]"

    @staticmethod
    def _global_decision_marker() -> str:
        return "[[VIBROGEMMA_GLOBAL_BINARY_DECISION_SLOT]]"

    @staticmethod
    def _target_decision_marker(target_index: int) -> str:
        if target_index not in {1, 2, 3, 4, 5}:
            raise ValueError("target_index must be 1..5")
        return f"[[VIBROGEMMA_TARGET_{target_index}_BINARY_DECISION_SLOT]]"

    def _task_instruction(self, task_mode: str, focus_target_index: int | None) -> str:
        prior = ""
        if self.prompt_prior_guard:
            prior = (
                " Treat normal and unknown_anomaly as equally plausible before reading the evidence."
                " Do not infer anomaly from excitation magnitude, absolute scale, recording identity,"
                " or dataset-specific appearance alone."
            )
        if task_mode == "global":
            return (
                "Generate exactly one constrained GLOBAL_CLASSIFICATION record. Decide only whether"
                " the six-board episode contains a structural-response anomaly somewhere. Do not"
                " infer or emit target-localization labels in this task." + prior
            )
        if task_mode == "target":
            if focus_target_index not in {1, 2, 3, 4, 5}:
                raise ValueError("target inference requires focus_target_index=1..5")
            semantics = ""
            if self.target_decision_semantics == "damage_source_zone":
                semantics = (
                    " Interpret affected as the most plausible localized damage source or monitored"
                    " structural zone, not merely a sensor whose response changed because vibration"
                    " propagated through the structure. Do not mark this target affected solely because"
                    " it has a large response when another target is the more plausible source zone."
                )
            comparison = ""
            if self.target_evidence_scope == "comparative_all_targets":
                comparison = (
                    " Use the other targets only as contemporaneous comparison context; emit no"
                    " decision for them in this record."
                )
            focus_layout = ""
            if self.target_focus_layout == "focus_last":
                focus_layout = (
                    " The final continuous target block in the prompt is the named focus target."
                )
            return (
                f"Generate exactly one constrained TARGET_CLASSIFICATION record for T{focus_target_index}."
                " Compare that target with the reference and the other contemporaneous targets, but"
                " decide this target independently. Do not copy a previous target decision and do not"
                " convert an episode-level anomaly into an all-target result."
                + comparison
                + focus_layout
                + semantics
                + prior
            )
        if task_mode == "joint":
            return (
                "First generate exactly the constrained CLASSIFICATION record for T1 through T5."
                " Each tuple must contain affected, class and severity. After END, emit EXPLANATION"
                " and one short sentence. Use only named auditable evidence. You may quote a numeric"
                " value only when that exact value is supplied in the evidence." + prior
            )
        if task_mode == "structured":
            return self._structured_decision_instruction()
        raise ValueError(f"unknown prompt task mode: {task_mode!r}")

    def _structured_decision_instruction(self) -> str:
        prior = ""
        if self.prompt_prior_guard:
            prior = (
                " Treat 0 and 1 as equally plausible before reading the evidence."
                " Do not infer anomaly from excitation magnitude, absolute scale, recording identity,"
                " or dataset-specific appearance alone."
            )
        semantics = ""
        if self.target_decision_semantics == "damage_source_zone":
            semantics = (
                " A target value of 1 means that the explicitly labelled localized source zone belongs"
                " to that monitored target. Propagated vibration at another target is not by itself a"
                " positive localization label."
            )
        return (
            "The six binary decisions below are scored in parallel from this same synchronized episode."
            " At every slot, 0 and 1 are the only valid next-token choices. GLOBAL=0 means normal and"
            " GLOBAL=1 means unknown_anomaly. For each T1 through T5, 0 means normal/not affected and"
            " 1 means affected/unknown_anomaly. Compare every target with the reference board and with"
            " all contemporaneous targets. Do not copy the global choice into target slots, do not copy"
            " one target into another, and do not assume that a global anomaly affects every target."
            " The target decisions form one complete set and are evaluated jointly."
            + semantics
            + prior
        )

    def _evidence_for_task(
        self,
        evidence_text: str,
        *,
        task_mode: str,
        focus_target_index: int | None,
    ) -> str:
        """Remove cross-row text shortcuts from an independent target task.

        All six continuous token blocks remain in the prompt. The textual evidence
        is narrowed to the common header, reference and named target so another
        target's explicit evidence line cannot become an autoregressive shortcut.
        """

        if task_mode != "target":
            return evidence_text
        if focus_target_index not in {1, 2, 3, 4, 5}:
            raise ValueError("target evidence filtering requires focus_target_index=1..5")
        lines = [line.strip() for line in str(evidence_text).splitlines() if line.strip()]
        prefix = f"target_{focus_target_index}:"
        if self.target_evidence_scope == "comparative_all_targets":
            selected = list(lines)
            if not any(line.lower().startswith(prefix) for line in selected):
                selected.append(
                    f"target_{focus_target_index}: target-specific textual physics evidence unavailable; "
                    "use the continuous target/reference tokens."
                )
            return "\n".join(selected)
        selected: list[str] = []
        for line in lines:
            lowered = line.lower()
            if (
                lowered.startswith("auditable vibration evidence")
                or lowered.startswith("reference:")
                or lowered.startswith(prefix)
            ):
                selected.append(line)
        if not any(line.lower().startswith(prefix) for line in selected):
            selected.append(
                f"target_{focus_target_index}: target-specific textual physics evidence unavailable; "
                "use the continuous target/reference tokens."
            )
        return "\n".join(selected)

    def _target_marker_order(
        self,
        *,
        task_mode: str,
        focus_target_index: int | None,
    ) -> list[int]:
        order = list(range(1, 6))
        if task_mode == "target" and self.target_focus_layout == "focus_last":
            if focus_target_index not in {1, 2, 3, 4, 5}:
                raise ValueError("focus-last target layout requires focus_target_index=1..5")
            order.remove(int(focus_target_index))
            order.append(int(focus_target_index))
        return order


    def _user_content(
        self,
        quality_text: str,
        evidence_text: str,
        extra_instruction: str = "",
        *,
        task_mode: str = "joint",
        focus_target_index: int | None = None,
    ) -> str:
        task_evidence_text = self._evidence_for_task(
            evidence_text,
            task_mode=task_mode,
            focus_target_index=focus_target_index,
        )
        pieces = [
            "Read the following six-board vibration episode. Continuous vibration embeddings replace each marker.",
            quality_text,
            task_evidence_text,
        ]
        pieces.append(
            self.marker_text["baseline_start"]
            + self._marker(0)
            + self.marker_text["baseline_end"]
        )
        target_order = self._target_marker_order(
            task_mode=task_mode,
            focus_target_index=focus_target_index,
        )
        if task_mode == "target" and self.target_focus_layout == "focus_last":
            pieces.append(
                f"TARGET BLOCK ORDER: comparison targets first; focus target T{focus_target_index} is last."
            )
        for index in target_order:
            pieces.append(
                self.marker_text["target_start_template"].format(index=index)
                + self._marker(index)
                + self.marker_text["target_end"]
            )
        pieces.append(self._task_instruction(task_mode, focus_target_index))
        if extra_instruction:
            pieces.append(extra_instruction)
        return "\n".join(piece for piece in pieces if piece)

    def _format_chat_prefix(
        self,
        quality_text: str,
        evidence_text: str,
        extra_instruction: str = "",
        *,
        task_mode: str = "joint",
        focus_target_index: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._user_content(
                    quality_text,
                    evidence_text,
                    extra_instruction,
                    task_mode=task_mode,
                    focus_target_index=focus_target_index,
                ),
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"System: {self.system_prompt}\nUser: {messages[1]['content']}\nAssistant:"

    def _structured_decision_prefix(
        self,
        quality_text: str,
        evidence_text: str,
        extra_instruction: str = "",
    ) -> str:
        """Return the chat prefix and structured-block header only.

        Decision queries are appended directly at token level by
        :meth:`encode_structured_decisions`. They are intentionally not embedded
        as searchable text sentinels because SentencePiece/BPE tokenization can
        change a sentinel's first token when it follows ``GLOBAL= `` or ``Tn= ``.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._user_content(
                    quality_text,
                    evidence_text,
                    extra_instruction,
                    task_mode="structured",
                    focus_target_index=None,
                ),
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prefix = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prefix = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prefix = f"System: {self.system_prompt}\nUser: {messages[1]['content']}\nAssistant:"
        return prefix + "\nSTRUCTURED_BINARY_DECISIONS"

    def _structured_decision_contexts(
        self,
        target_query_order: Sequence[int] | None = None,
    ) -> tuple[tuple[int, str], ...]:
        """Exact causal query strings used by prompt construction and token audit."""

        order = tuple(
            int(value) for value in (target_query_order or tuple(range(1, 6)))
        )
        if sorted(order) != list(range(1, 6)):
            raise ValueError("target query order must be a permutation of 1..5")
        return (
            (0, f"\nGLOBAL={self.structured_decision_separator}"),
            *(
                (index, f"\nT{index}={self.structured_decision_separator}")
                for index in order
            ),
        )

    def _episode_target_query_order(self, episode_id: str) -> tuple[int, ...]:
        if self.structured_target_query_order == "fixed":
            return tuple(range(1, 6))
        identity = str(episode_id).strip()
        if not identity:
            raise ValueError("deterministic target query order requires an episode_id")
        return tuple(
            sorted(
                range(1, 6),
                key=lambda index: hashlib.sha256(
                    f"{identity}\0target_{index}".encode("utf-8")
                ).digest(),
            )
        )

    def _tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]

    @staticmethod
    def _find_subsequence(sequence: Sequence[int], needle: Sequence[int]) -> int:
        matches = [
            start
            for start in range(0, len(sequence) - len(needle) + 1)
            if list(sequence[start : start + len(needle)]) == list(needle)
        ]
        if len(matches) != 1:
            raise ModelCompatibilityError(f"Expected one marker occurrence, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _balanced_target_line_weights(label: EpisodeLabel) -> list[float]:
        anomaly_flags = [target.class_name != "normal" for target in label.targets]
        anomaly_count = sum(anomaly_flags)
        normal_count = len(anomaly_flags) - anomaly_count
        if anomaly_count == 0 or normal_count == 0:
            return [1.0] * len(anomaly_flags)
        half_mass = len(anomaly_flags) / 2.0
        return [
            half_mass / anomaly_count if is_anomaly else half_mass / normal_count
            for is_anomaly in anomaly_flags
        ]

    def _target_line_token_spans(self, record_text: str) -> list[tuple[int, int]]:
        lines = record_text.splitlines()
        if len(lines) != 7 or lines[0] != self.codec.header or lines[-1] != "END":
            raise ModelCompatibilityError("unexpected classification-record layout")
        spans: list[tuple[int, int]] = []
        for line_index in range(1, 6):
            before = "\n".join(lines[:line_index]) + "\n"
            through = "\n".join(lines[: line_index + 1])
            if line_index < len(lines) - 1:
                through += "\n"
            start = len(self._tokenize(before))
            end = len(self._tokenize(through))
            if not 0 <= start < end:
                raise ModelCompatibilityError("could not isolate target-row token span")
            spans.append((start, end))
        return spans

    def _decision_line_token_spans(
        self,
        record_text: str,
        *,
        task_mode: str,
    ) -> list[tuple[int, int]]:
        lines = record_text.splitlines()
        if task_mode == "global":
            expected_header = self.global_codec.header
            decision_indices = (1,)
            expected_line_count = 3
        elif task_mode == "target":
            expected_header = self.target_codec.header
            decision_indices = (1,)
            expected_line_count = 3
        elif task_mode == "joint":
            expected_header = self.codec.header
            decision_indices = tuple(range(1, 6))
            expected_line_count = 7
        else:
            raise ValueError(f"unknown prompt task mode: {task_mode!r}")
        if (
            len(lines) != expected_line_count
            or lines[0] != expected_header
            or lines[-1] != "END"
        ):
            raise ModelCompatibilityError(
                f"unexpected {task_mode} classification-record layout"
            )
        spans: list[tuple[int, int]] = []
        for line_index in decision_indices:
            before = "\n".join(lines[:line_index]) + "\n"
            through = "\n".join(lines[: line_index + 1])
            if line_index < len(lines) - 1:
                through += "\n"
            start = len(self._tokenize(before))
            end = len(self._tokenize(through))
            if not 0 <= start < end:
                raise ModelCompatibilityError(
                    f"could not isolate {task_mode} decision token span"
                )
            spans.append((start, end))
        return spans

    def _decision_weight(self, task_mode: str) -> float:
        if task_mode == "global":
            return self.global_decision_weight
        if task_mode == "target":
            return self.target_decision_weight
        if task_mode == "joint":
            return self.joint_decision_weight
        raise ValueError(f"unknown prompt task mode: {task_mode!r}")

    def encode(
        self,
        *,
        quality_text: str,
        evidence_text: str,
        target_text: str | None = None,
        classification_prefix_text: str | None = None,
        target_row_weights: Sequence[float] | None = None,
        extra_instruction: str = "",
        task_mode: str = "joint",
        focus_target_index: int | None = None,
        task_weight: float = 1.0,
    ) -> PromptEncoding:
        formatted = self._format_chat_prefix(
            quality_text,
            evidence_text,
            extra_instruction,
            task_mode=task_mode,
            focus_target_index=focus_target_index,
        )
        ids = self._tokenize(formatted)
        marker_spans: list[tuple[int, int, int]] = []
        for board_index in range(6):
            marker_ids = self._tokenize(self._marker(board_index))
            start = self._find_subsequence(ids, marker_ids)
            marker_spans.append((start, start + len(marker_ids), board_index))
        marker_spans.sort()
        output_ids: list[int] = []
        slot_positions_by_board: dict[int, list[int]] = {
            board_index: [] for board_index in range(6)
        }
        cursor = 0
        for start, end, board_index in marker_spans:
            if start < cursor:
                raise ModelCompatibilityError("Overlapping soft-token markers")
            output_ids.extend(ids[cursor:start])
            for _ in range(self.tokens_per_board):
                slot_positions_by_board[board_index].append(len(output_ids))
                output_ids.append(self.pad_token_id)
            cursor = end
        output_ids.extend(ids[cursor:])
        slot_positions = [
            position
            for board_index in range(6)
            for position in slot_positions_by_board[board_index]
        ]
        prompt_length = len(output_ids)
        labels = None
        loss_weights = None
        if target_text is not None:
            target_ids_without_eos = self._tokenize(target_text)
            classification_token_count = len(target_ids_without_eos)
            if classification_prefix_text is not None:
                prefix_ids = self._tokenize(classification_prefix_text)
                classification_token_count = 0
                for expected, actual in zip(
                    prefix_ids, target_ids_without_eos, strict=False
                ):
                    if expected != actual:
                        break
                    classification_token_count += 1
                if classification_token_count == 0:
                    raise ModelCompatibilityError(
                        "classification record is not a token prefix of the full target"
                    )
            target_ids = list(target_ids_without_eos)
            base_classification_weight = (
                self.classification_loss_weight * float(task_weight)
            )
            target_weights = [
                base_classification_weight * self.classification_structure_weight
                if index < classification_token_count
                else self.explanation_loss_weight * float(task_weight)
                for index in range(len(target_ids))
            ]
            if classification_prefix_text is not None:
                decision_weight = base_classification_weight * self._decision_weight(
                    task_mode
                )
                for start, end in self._decision_line_token_spans(
                    classification_prefix_text,
                    task_mode=task_mode,
                ):
                    bounded_end = min(end, classification_token_count)
                    for index in range(start, bounded_end):
                        target_weights[index] = decision_weight
            if target_row_weights is not None:
                if classification_prefix_text is None or task_mode != "joint":
                    raise ValueError("target-row weights require a joint classification prefix")
                weights = [float(value) for value in target_row_weights]
                if len(weights) != 5 or any(value <= 0 for value in weights):
                    raise ValueError("target-row weights must contain five positive values")
                for (start, end), row_weight in zip(
                    self._target_line_token_spans(classification_prefix_text),
                    weights,
                    strict=True,
                ):
                    bounded_end = min(end, classification_token_count)
                    for index in range(start, bounded_end):
                        target_weights[index] = (
                            base_classification_weight
                            * self.joint_decision_weight
                            * row_weight
                        )
            if self.tokenizer.eos_token_id is not None:
                target_ids.append(int(self.tokenizer.eos_token_id))
                target_weights.append(self.eos_loss_weight * float(task_weight))
            output_ids.extend(target_ids)
            labels = [-100] * prompt_length + target_ids
            loss_weights = [0.0] * prompt_length + target_weights
        return PromptEncoding(
            input_ids=output_ids,
            slot_positions=slot_positions,
            prompt_length=prompt_length,
            labels=labels,
            loss_weights=loss_weights,
        )

    def encode_structured_decisions(
        self,
        *,
        quality_text: str,
        evidence_text: str,
        extra_instruction: str = "",
        target_query_order: Sequence[int] | None = None,
    ) -> tuple[PromptEncoding, list[int]]:
        """Encode one prompt with six parallel binary decision-query positions.

        The neutral placeholders contain no answer. Consequently later target
        queries cannot condition on earlier target decisions or on the global
        verdict, while all six queries still see the same reference and target
        signal embeddings.
        """

        formatted = self._structured_decision_prefix(
            quality_text,
            evidence_text,
            extra_instruction,
        )
        ids = self._tokenize(formatted)
        replacements: list[tuple[int, int, int]] = []
        for board_index in range(6):
            marker_ids = self._tokenize(self._marker(board_index))
            start = self._find_subsequence(ids, marker_ids)
            replacements.append((start, start + len(marker_ids), board_index))
        replacements.sort()

        output_ids: list[int] = []
        slot_positions_by_board: dict[int, list[int]] = {
            board_index: [] for board_index in range(6)
        }
        cursor = 0
        for start, end, board_index in replacements:
            if start < cursor:
                raise ModelCompatibilityError("Overlapping structured board markers")
            output_ids.extend(ids[cursor:start])
            for _ in range(self.tokens_per_board):
                slot_positions_by_board[board_index].append(len(output_ids))
                output_ids.append(self.pad_token_id)
            cursor = end
        output_ids.extend(ids[cursor:])

        # Append the six query contexts directly as token sequences. This avoids
        # searching for a marker tokenized in isolation inside a prompt where its
        # first token may have merged with the preceding separator.
        decision_position_by_slot: dict[int, int] = {}
        for slot, context in self._structured_decision_contexts(target_query_order):
            context_ids = self._tokenize(context)
            if not context_ids:
                raise ModelCompatibilityError(
                    f"Structured decision context tokenized empty: {context!r}"
                )
            output_ids.extend(context_ids)
            decision_position_by_slot[slot] = len(output_ids) - 1
            output_ids.append(self.decision_placeholder_token_id)
        output_ids.extend(self._tokenize("\nEND_STRUCTURED_BINARY_DECISIONS"))

        decision_positions = [
            decision_position_by_slot[slot] for slot in range(6)
        ]
        if len(decision_positions) != 6 or len(set(decision_positions)) != 6:
            raise ModelCompatibilityError(
                "Structured prompt did not produce six unique decision positions"
            )
        slot_positions = [
            position
            for board_index in range(6)
            for position in slot_positions_by_board[board_index]
        ]
        return (
            PromptEncoding(
                input_ids=output_ids,
                slot_positions=slot_positions,
                prompt_length=len(output_ids),
                labels=None,
                loss_weights=None,
            ),
            decision_positions,
        )

    def quality_text(self, episode: SixBoardEpisode) -> str:
        parts: list[str] = []
        for board_index, board in enumerate(episode.boards):
            axes = "".join(
                axis.upper()
                for axis, present in zip(
                    ("x", "y", "z"), board.axis_mask, strict=True
                )
                if present
            )
            valid = "valid" if board.quality.is_valid else "invalid"
            role = "reference" if board_index == 0 else f"target_{board_index}"
            parts.append(f"{role}: axes={axes or 'none'}, quality={valid}")
        return "Signal quality metadata: " + "; ".join(parts) + "."

    def _training_record(
        self,
        label: EpisodeLabel,
        evidence: dict[str, Any] | None,
    ) -> tuple[str, str, str, int | None, float, list[float] | None]:
        label.validate()
        if label.supervision_scope == "global_only":
            record = self.global_codec.encode_label(label)
            return record, record, "global", None, self.global_task_weight, None
        if label.supervision_scope == "target_only":
            record = self.target_codec.encode_label(label)
            return (
                record,
                record,
                "target",
                label.focus_target_index,
                self.target_task_weight,
                None,
            )
        if self.factorized_training:
            raise ValueError(
                "factorized training requires the loader to expand joint labels into global/target tasks"
            )
        record = self.codec.encode_label(label)
        explanation = label.explanation_target.strip()
        if not explanation and evidence is not None:
            explanation = self.explanation_renderer.training_target(label, evidence)
        target = f"{record}\nEXPLANATION\n{explanation}" if explanation else record
        row_weights = (
            self._balanced_target_line_weights(label) if self.target_row_balance else None
        )
        return record, target, "joint", None, self.joint_task_weight, row_weights

    def build_batch(
        self,
        *,
        model: Any,
        soft_tokens: torch.Tensor,
        episodes: Sequence[SixBoardEpisode],
        evidence_texts: Sequence[str] | None,
        evidences: Sequence[dict[str, Any]] | None = None,
        training: bool,
        extra_instruction: str = "",
        inference_task: str = "joint",
        focus_target_index: int | None = None,
    ) -> MixedEmbeddingBatch:
        encodings: list[PromptEncoding] = []
        if evidence_texts is None:
            evidence_texts = ["Auditable physics evidence unavailable."] * len(episodes)
        if len(evidence_texts) != len(episodes):
            raise ValueError("evidence_texts length must match episodes")
        if evidences is None:
            evidences = [{} for _ in episodes]
        if len(evidences) != len(episodes):
            raise ValueError("evidences length must match episodes")
        for episode, evidence_text, evidence in zip(
            episodes, evidence_texts, evidences, strict=True
        ):
            target = None
            classification_prefix = None
            row_weights = None
            task_mode = inference_task
            task_focus = focus_target_index
            task_weight = 1.0
            if training:
                if episode.label is None:
                    raise ValueError("Training episode lacks label")
                (
                    classification_prefix,
                    target,
                    task_mode,
                    task_focus,
                    task_weight,
                    row_weights,
                ) = self._training_record(episode.label, evidence)
            encodings.append(
                self.encode(
                    quality_text=self.quality_text(episode),
                    evidence_text=evidence_text,
                    target_text=target,
                    classification_prefix_text=classification_prefix,
                    target_row_weights=row_weights,
                    extra_instruction=extra_instruction,
                    task_mode=task_mode,
                    focus_target_index=task_focus,
                    task_weight=task_weight,
                )
            )
        return build_mixed_embedding_batch(
            model=model,
            input_id_sequences=[encoding.input_ids for encoding in encodings],
            slot_positions=[encoding.slot_positions for encoding in encodings],
            soft_tokens=soft_tokens,
            label_sequences=[encoding.labels for encoding in encodings] if training else None,
            loss_weight_sequences=[encoding.loss_weights for encoding in encodings]
            if training
            else None,
            pad_token_id=self.pad_token_id,
        )

    def build_structured_decision_batch(
        self,
        *,
        model: Any,
        soft_tokens: torch.Tensor,
        episodes: Sequence[SixBoardEpisode],
        evidence_texts: Sequence[str] | None,
        extra_instruction: str = "",
    ) -> StructuredDecisionPromptBatch:
        if evidence_texts is None:
            evidence_texts = ["Auditable physics evidence unavailable."] * len(episodes)
        if len(evidence_texts) != len(episodes):
            raise ValueError("evidence_texts length must match episodes")
        encodings: list[PromptEncoding] = []
        decision_positions: list[list[int]] = []
        for episode, evidence_text in zip(episodes, evidence_texts, strict=True):
            encoding, positions = self.encode_structured_decisions(
                quality_text=self.quality_text(episode),
                evidence_text=evidence_text,
                extra_instruction=extra_instruction,
                target_query_order=self._episode_target_query_order(
                    episode.episode_id
                ),
            )
            encodings.append(encoding)
            decision_positions.append(positions)
        mixed = build_mixed_embedding_batch(
            model=model,
            input_id_sequences=[encoding.input_ids for encoding in encodings],
            slot_positions=[encoding.slot_positions for encoding in encodings],
            soft_tokens=soft_tokens,
            pad_token_id=self.pad_token_id,
        )
        return StructuredDecisionPromptBatch(
            mixed_batch=mixed,
            decision_positions=torch.tensor(
                decision_positions,
                device=soft_tokens.device,
                dtype=torch.long,
            ),
        )
