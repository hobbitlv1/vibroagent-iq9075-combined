from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..exceptions import ModelCompatibilityError


@dataclass(slots=True)
class MixedEmbeddingBatch:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    per_layer_inputs: torch.Tensor | None
    labels: torch.Tensor | None
    prompt_lengths: list[int]
    slot_positions: list[list[int]]
    loss_weights: torch.Tensor | None = None


def resolve_text_model(model: Any) -> Any:
    queue = [model]
    visited: set[int] = set()
    attributes = ("model", "base_model", "language_model", "module")
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if hasattr(candidate, "embed_tokens") and hasattr(candidate, "forward"):
            return candidate
        for attribute in attributes:
            child = getattr(candidate, attribute, None)
            if child is not None and id(child) not in visited:
                queue.append(child)
    raise ModelCompatibilityError("Could not locate Gemma text model / embed_tokens module")


def resolve_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ModelCompatibilityError("Model has no config")
    text_config = getattr(config, "text_config", None) or config
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ModelCompatibilityError("Cannot resolve Gemma hidden_size")
    return int(hidden_size)


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[normalized]


def make_per_layer_inputs(text_model: Any, input_ids: torch.Tensor, base_embeddings: torch.Tensor) -> torch.Tensor | None:
    hidden = int(getattr(text_model, "hidden_size_per_layer_input", 0) or 0)
    if hidden == 0:
        return None
    if not hasattr(text_model, "get_per_layer_inputs"):
        raise ModelCompatibilityError(
            "Gemma config requires per-layer embeddings but model lacks get_per_layer_inputs"
        )
    # `base_embeddings` must be the exact scaled lookup for `input_ids`. Custom
    # vibration embeddings are inserted only after this identity component is built.
    return text_model.get_per_layer_inputs(input_ids, base_embeddings)


def build_mixed_embedding_batch(
    *,
    model: Any,
    input_id_sequences: Sequence[Sequence[int]],
    slot_positions: Sequence[Sequence[int]],
    soft_tokens: torch.Tensor,
    attention_sequences: Sequence[Sequence[int]] | None = None,
    label_sequences: Sequence[Sequence[int]] | None = None,
    loss_weight_sequences: Sequence[Sequence[float]] | None = None,
    pad_token_id: int,
) -> MixedEmbeddingBatch:
    if soft_tokens.ndim != 3:
        raise ValueError("soft_tokens must have shape [batch,soft_tokens,hidden]")
    batch_size = len(input_id_sequences)
    if batch_size != soft_tokens.shape[0] or batch_size != len(slot_positions):
        raise ValueError("Batch sizes do not match")
    max_length = max(len(sequence) for sequence in input_id_sequences)
    device = soft_tokens.device
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    labels = None
    loss_weights = None
    if label_sequences is not None:
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long, device=device)
    if loss_weight_sequences is not None:
        if label_sequences is None:
            raise ValueError("loss weights require label sequences")
        loss_weights = torch.zeros((batch_size, max_length), dtype=torch.float32, device=device)
    for batch_index, sequence in enumerate(input_id_sequences):
        length = len(sequence)
        input_ids[batch_index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        if attention_sequences is None:
            attention_mask[batch_index, :length] = 1
        else:
            attention_mask[batch_index, :length] = torch.tensor(
                attention_sequences[batch_index], dtype=torch.long, device=device
            )
        if labels is not None:
            label_values = label_sequences[batch_index]
            if len(label_values) != length:
                raise ValueError("label sequence length mismatch")
            labels[batch_index, :length] = torch.tensor(label_values, dtype=torch.long, device=device)
        if loss_weights is not None:
            weight_values = loss_weight_sequences[batch_index]
            if len(weight_values) != length:
                raise ValueError("loss-weight sequence length mismatch")
            loss_weights[batch_index, :length] = torch.tensor(
                weight_values, dtype=torch.float32, device=device
            )

    text_model = resolve_text_model(model)
    embedding_module = text_model.embed_tokens
    base_embeddings = embedding_module(input_ids)
    mixed = base_embeddings.clone()
    for batch_index, positions in enumerate(slot_positions):
        if len(positions) != soft_tokens.shape[1]:
            raise ValueError(
                f"Example {batch_index} has {len(positions)} slots but {soft_tokens.shape[1]} soft tokens"
            )
        indices = torch.tensor(positions, device=device, dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= len(input_id_sequences[batch_index])):
            raise ValueError("Soft-token slot index outside prompt")
        mixed[batch_index].index_copy_(0, indices, soft_tokens[batch_index].to(mixed.dtype))

    per_layer_inputs = make_per_layer_inputs(text_model, input_ids, base_embeddings)
    return MixedEmbeddingBatch(
        input_ids=input_ids,
        inputs_embeds=mixed,
        attention_mask=attention_mask,
        per_layer_inputs=per_layer_inputs,
        labels=labels,
        loss_weights=loss_weights,
        prompt_lengths=[len(sequence) for sequence in input_id_sequences],
        slot_positions=[list(positions) for positions in slot_positions],
    )


def forward_with_mixed_embeddings(model: Any, batch: MixedEmbeddingBatch, *, use_cache: bool) -> Any:
    kwargs: dict[str, Any] = {
        "inputs_embeds": batch.inputs_embeds,
        "attention_mask": batch.attention_mask,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if batch.per_layer_inputs is not None:
        kwargs["per_layer_inputs"] = batch.per_layer_inputs
    if batch.labels is not None:
        kwargs["labels"] = batch.labels
    try:
        return model(return_shared_kv_states=True, **kwargs)
    except TypeError as exc:
        if "return_shared_kv_states" not in str(exc):
            raise
        return model(**kwargs)
