from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True, slots=True)
class _State:
    segment: int
    alternative: int
    offset: int


class SegmentedTokenGrammar:
    """Finite grammar built from ordered token-sequence alternatives.

    Each segment is a list of alternatives. A static segment is represented by one
    alternative. The grammar computes valid next token IDs without enumerating the
    Cartesian product of all target choices.
    """

    def __init__(self, segments: Sequence[Sequence[Sequence[int]]]) -> None:
        normalized: list[tuple[tuple[int, ...], ...]] = []
        for segment_index, alternatives in enumerate(segments):
            segment = tuple(tuple(int(token) for token in alternative) for alternative in alternatives)
            if not segment or any(len(alternative) == 0 for alternative in segment):
                raise ValueError(f"Grammar segment {segment_index} contains an empty alternative")
            normalized.append(segment)
        if not normalized:
            raise ValueError("Grammar requires at least one segment")
        self.segments = tuple(normalized)

    def _closure(self, states: set[_State]) -> set[_State]:
        result = set(states)
        changed = True
        while changed:
            changed = False
            for state in list(result):
                alternative = self.segments[state.segment][state.alternative]
                if state.offset != len(alternative):
                    continue
                result.remove(state)
                if state.segment + 1 < len(self.segments):
                    for alternative_index in range(len(self.segments[state.segment + 1])):
                        result.add(_State(state.segment + 1, alternative_index, 0))
                changed = True
                break
        return result

    def states_after(self, prefix: Sequence[int]) -> set[_State]:
        states = self._closure({_State(0, alternative, 0) for alternative in range(len(self.segments[0]))})
        for token in prefix:
            next_states: set[_State] = set()
            for state in states:
                alternative = self.segments[state.segment][state.alternative]
                if state.offset < len(alternative) and alternative[state.offset] == int(token):
                    next_states.add(_State(state.segment, state.alternative, state.offset + 1))
            states = self._closure(next_states)
            if not states:
                return set()
        return states

    def allowed_next(self, prefix: Sequence[int]) -> set[int]:
        states = self.states_after(prefix)
        allowed: set[int] = set()
        for state in states:
            alternative = self.segments[state.segment][state.alternative]
            if state.offset < len(alternative):
                allowed.add(alternative[state.offset])
        return allowed

    def is_complete(self, prefix: Sequence[int]) -> bool:
        states: set[_State] = {_State(0, alternative, 0) for alternative in range(len(self.segments[0]))}
        for token in prefix:
            states = self._closure(states)
            next_states: set[_State] = set()
            for state in states:
                alternative = self.segments[state.segment][state.alternative]
                if state.offset < len(alternative) and alternative[state.offset] == int(token):
                    next_states.add(_State(state.segment, state.alternative, state.offset + 1))
            states = next_states
            if not states:
                return False
        states = self._closure(states)
        return not states

    def mask_logits(self, logits: torch.Tensor, prefix: Sequence[int]) -> torch.Tensor:
        allowed = self.allowed_next(prefix)
        if not allowed:
            raise RuntimeError("Grammar has no allowed continuation")
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        indices = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        masked.index_copy_(0, indices, logits.index_select(0, indices))
        return masked


def tokenize_without_specials(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]
