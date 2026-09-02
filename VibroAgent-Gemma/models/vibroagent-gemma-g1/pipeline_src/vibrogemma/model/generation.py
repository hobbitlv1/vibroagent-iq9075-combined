from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..exceptions import DataContractError
from .gemma_bridge import MixedEmbeddingBatch, forward_with_mixed_embeddings
from .grammar import SegmentedTokenGrammar, tokenize_without_specials
from ..contracts import TargetPrediction
from .records import (
    ClassificationRecordCodec,
    GlobalPrediction,
    GlobalRecordCodec,
    TargetRecordCodec,
)


@dataclass(slots=True)
class GenerationOutput:
    record_text: str
    explanation: str
    target_probabilities: list[float | None]
    generated_token_ids: list[int]
    choice_probability: float | None = None
    task: str = "joint"


@dataclass(slots=True)
class _DecodeState:
    logits: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor
    shared_kv_states: Any = None


class GemmaStructuredGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        codec: ClassificationRecordCodec,
        *,
        global_codec: GlobalRecordCodec | None = None,
        target_codec: TargetRecordCodec | None = None,
        max_record_tokens: int = 160,
        max_explanation_tokens: int = 64,
        repetition_penalty: float = 1.05,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self.global_codec = global_codec or GlobalRecordCodec(
            ("normal", "unknown_anomaly"), codec.severities
        )
        self.target_codec = target_codec or TargetRecordCodec(codec.classes, codec.severities)
        self.max_record_tokens = int(max_record_tokens)
        self.max_explanation_tokens = int(max_explanation_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.grammar = self._build_joint_grammar()
        self.global_grammar = self._build_global_grammar()
        self.target_grammars = {
            index: self._build_target_grammar(index) for index in range(1, 6)
        }

    def _build_joint_grammar(self) -> SegmentedTokenGrammar:
        segments: list[list[list[int]]] = [
            [tokenize_without_specials(self.tokenizer, self.codec.header)]
        ]
        for index in range(1, 6):
            segments.append(
                [
                    tokenize_without_specials(self.tokenizer, value)
                    for value in self.codec.target_choice_strings(index)
                ]
            )
        segments.append(
            [[*tokenize_without_specials(self.tokenizer, self.codec.footer)]]
        )
        return SegmentedTokenGrammar(segments)

    def _build_global_grammar(self) -> SegmentedTokenGrammar:
        return SegmentedTokenGrammar(
            [
                [[*tokenize_without_specials(self.tokenizer, self.global_codec.header)]],
                [
                    tokenize_without_specials(self.tokenizer, value)
                    for value in self.global_codec.choice_strings()
                ],
                [[*tokenize_without_specials(self.tokenizer, self.global_codec.footer)]],
            ]
        )

    def _build_target_grammar(self, target_index: int) -> SegmentedTokenGrammar:
        return SegmentedTokenGrammar(
            [
                [[*tokenize_without_specials(self.tokenizer, self.target_codec.header)]],
                [
                    tokenize_without_specials(self.tokenizer, value)
                    for value in self.target_codec.choice_strings(target_index)
                ],
                [[*tokenize_without_specials(self.tokenizer, self.target_codec.footer)]],
            ]
        )

    @torch.inference_mode()
    def _generate_constrained(
        self,
        batch: MixedEmbeddingBatch,
        *,
        grammar: SegmentedTokenGrammar,
        choice_segments: tuple[int, ...],
    ) -> tuple[str, list[int], dict[int, float | None], _DecodeState]:
        if batch.inputs_embeds.shape[0] != 1:
            raise ValueError("Structured generation currently requires batch size 1")
        self.model.eval()
        state = self._first_forward(batch)
        generated: list[int] = []
        segment_log_probabilities: dict[int, list[float]] = {
            index: [] for index in choice_segments
        }
        for _ in range(self.max_record_tokens):
            states = grammar.states_after(generated)
            if generated and grammar.is_complete(generated):
                break
            if not states:
                raise DataContractError("Generated prefix left the constrained grammar")
            segment_indices = {state_item.segment for state_item in states}
            segment_index = next(iter(segment_indices)) if len(segment_indices) == 1 else -1
            allowed_next = grammar.allowed_next(generated)
            branching = len(allowed_next) > 1
            masked = grammar.mask_logits(state.logits, generated)
            probabilities = torch.softmax(masked.float(), dim=-1)
            token = int(torch.argmax(masked).item())
            probability = max(float(probabilities[token].item()), 1e-12)
            if segment_index in segment_log_probabilities and branching:
                segment_log_probabilities[segment_index].append(math.log(probability))
            generated.append(token)
            state = self._advance(state, token)
            if grammar.is_complete(generated):
                break
        else:
            raise DataContractError("Constrained record exceeded max_record_tokens")
        record_text = self.tokenizer.decode(generated, skip_special_tokens=False).strip()
        probabilities = {
            index: (
                math.exp(sum(segment_log_probabilities[index]))
                if segment_log_probabilities[index]
                else None
            )
            for index in choice_segments
        }
        return record_text, generated, probabilities, state

    @torch.inference_mode()
    def generate_global(
        self, batch: MixedEmbeddingBatch
    ) -> tuple[GlobalPrediction, GenerationOutput]:
        record_text, generated, probabilities, _state = self._generate_constrained(
            batch, grammar=self.global_grammar, choice_segments=(1,)
        )
        probability = probabilities[1]
        prediction = self.global_codec.parse(record_text, probability)
        return prediction, GenerationOutput(
            record_text=record_text,
            explanation="",
            target_probabilities=[],
            generated_token_ids=generated,
            choice_probability=probability,
            task="global",
        )

    @torch.inference_mode()
    def generate_target(
        self,
        batch: MixedEmbeddingBatch,
        *,
        target_index: int,
    ) -> tuple[TargetPrediction, GenerationOutput]:
        record_text, generated, probabilities, _state = self._generate_constrained(
            batch,
            grammar=self.target_grammars[target_index],
            choice_segments=(1,),
        )
        probability = probabilities[1]
        prediction = self.target_codec.parse(
            record_text, target_index=target_index, probability=probability
        )
        return prediction, GenerationOutput(
            record_text=record_text,
            explanation="",
            target_probabilities=[probability],
            generated_token_ids=generated,
            choice_probability=probability,
            task=f"target_{target_index}",
        )

    @torch.inference_mode()
    def generate(
        self,
        batch: MixedEmbeddingBatch,
        *,
        allowed_numeric_text: str | None = None,
        generate_explanation: bool = True,
    ) -> GenerationOutput:
        """Backward-compatible joint generation used only when configured."""

        record_text, generated, probabilities, state = self._generate_constrained(
            batch,
            grammar=self.grammar,
            choice_segments=(1, 2, 3, 4, 5),
        )
        target_probabilities = [probabilities[index] for index in range(1, 6)]
        self.codec.parse(record_text, target_probabilities)
        if not generate_explanation:
            return GenerationOutput(
                record_text=record_text,
                explanation="",
                target_probabilities=target_probabilities,
                generated_token_ids=list(generated),
            )
        marker_tokens = tokenize_without_specials(self.tokenizer, "\nEXPLANATION\n")
        for token in marker_tokens:
            state = self._advance(state, token)
        explanation_ids: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for _ in range(self.max_explanation_tokens):
            logits = self._apply_repetition_penalty(state.logits, explanation_ids)
            token = int(torch.argmax(logits).item())
            if eos_token_id is not None and token == int(eos_token_id):
                break
            explanation_ids.append(token)
            state = self._advance(state, token)
            decoded = self.tokenizer.decode(explanation_ids, skip_special_tokens=True)
            if "\n\n" in decoded:
                break
        explanation = self._sanitize_explanation(
            self.tokenizer.decode(explanation_ids, skip_special_tokens=True),
            record_text,
            allowed_numeric_text,
        )
        return GenerationOutput(
            record_text=record_text,
            explanation=explanation,
            target_probabilities=target_probabilities,
            generated_token_ids=[*generated, *marker_tokens, *explanation_ids],
        )

    def _first_forward(self, batch: MixedEmbeddingBatch) -> _DecodeState:
        output = forward_with_mixed_embeddings(self.model, batch, use_cache=True)
        logits = output.logits[0, -1]
        return _DecodeState(
            logits=logits,
            past_key_values=output.past_key_values,
            attention_mask=batch.attention_mask,
            shared_kv_states=getattr(output, "shared_kv_states", None),
        )

    def _advance(self, state: _DecodeState, token: int) -> _DecodeState:
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=state.logits.device)
        attention_mask = torch.cat(
            (
                state.attention_mask,
                torch.ones(
                    (state.attention_mask.shape[0], 1),
                    device=state.attention_mask.device,
                    dtype=state.attention_mask.dtype,
                ),
            ),
            dim=1,
        )
        kwargs: dict[str, Any] = {
            "input_ids": token_tensor,
            "attention_mask": attention_mask,
            "past_key_values": state.past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        if state.shared_kv_states is not None:
            kwargs["shared_kv_states"] = state.shared_kv_states
            kwargs["return_shared_kv_states"] = True
        try:
            output = self.model(**kwargs)
        except TypeError as exc:
            if "shared_kv_states" not in str(exc) and "return_shared_kv_states" not in str(exc):
                raise
            kwargs.pop("shared_kv_states", None)
            kwargs.pop("return_shared_kv_states", None)
            output = self.model(**kwargs)
        return _DecodeState(
            logits=output.logits[0, -1],
            past_key_values=output.past_key_values,
            attention_mask=attention_mask,
            shared_kv_states=getattr(output, "shared_kv_states", state.shared_kv_states),
        )

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: list[int]) -> torch.Tensor:
        if self.repetition_penalty == 1.0 or not generated:
            return logits
        adjusted = logits.clone()
        for token in set(generated):
            value = adjusted[token]
            adjusted[token] = (
                value * self.repetition_penalty
                if value < 0
                else value / self.repetition_penalty
            )
        return adjusted

    @staticmethod
    def _sanitize_explanation(
        text: str,
        record_text: str,
        allowed_numeric_text: str | None = None,
    ) -> str:
        compact = " ".join(text.strip().split())
        if not compact:
            return "Gemma emitted the classification record without an explanation."
        sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", compact)
        if sentence_match:
            compact = sentence_match.group(1)
        compact = compact[:400].strip()
        evidence_numbers = set()
        number_pattern = r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        if allowed_numeric_text:
            evidence_numbers = set(re.findall(number_pattern, allowed_numeric_text))
        measurement_text = re.sub(
            r"(?:target_|T)[1-5]", "TARGET", compact, flags=re.IGNORECASE
        )
        generated_numbers = set(re.findall(number_pattern, measurement_text))
        unsupported = generated_numbers - evidence_numbers
        if unsupported:
            predictions = ClassificationRecordCodec().parse(record_text)
            affected = [prediction.sensor_id for prediction in predictions if prediction.affected]
            if not affected:
                return "Gemma classified all target sensors as normal."
            return (
                "Gemma classified "
                + ", ".join(affected)
                + " as affected from the multi-resolution vibration and supplied physics evidence."
            )
        return compact
