from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

TARGET_COUNT = 5
GLOBAL_SLOT = 0
TARGET_SLOT_OFFSET = 1


@dataclass(slots=True)
class StructuredSetLossResult:
    loss: torch.Tensor
    set_loss: torch.Tensor
    ranking_loss: torch.Tensor
    positive_margin_loss: torch.Tensor
    hard_negative_margin_loss: torch.Tensor
    normal_margin_loss: torch.Tensor
    class_balanced_binary_loss: torch.Tensor
    candidate_scores: torch.Tensor
    predicted_mask: torch.Tensor
    affected_probabilities: torch.Tensor


@dataclass(slots=True)
class SingleSourceZoneLossResult:
    loss: torch.Tensor
    candidate_scores: torch.Tensor
    predicted_mask: torch.Tensor
    affected_probabilities: torch.Tensor


def _candidate_masks(device: torch.device) -> torch.Tensor:
    values = torch.arange(1 << TARGET_COUNT, device=device, dtype=torch.long)
    shifts = torch.arange(TARGET_COUNT, device=device, dtype=torch.long)
    return ((values[:, None] >> shifts[None, :]) & 1).to(torch.bool)


def resolve_binary_choice_token_ids(
    tokenizer: Any,
    choices: Sequence[str] = ("0", "1"),
    *,
    contexts: Sequence[str] | None = None,
) -> tuple[int, int]:
    """Resolve two one-token binary continuations in their real query context.

    The structured prompt contains an explicit separator after ``GLOBAL=`` and
    ``Tn=``.  The scored alternatives are therefore the bare digit strings
    ``0`` and ``1``.  When contexts are supplied, every ``context + choice`` must
    preserve the context tokenization and append exactly one identical token.
    This fails before GPU training instead of silently scoring a shared separator
    token or the wrong vocabulary entry.
    """

    if len(choices) != 2 or choices[0] == choices[1]:
        raise ValueError("binary choices must contain two distinct strings")

    def tokenise(text: str) -> list[int]:
        encoded = tokenizer(
            str(text),
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(value) for value in ids]

    query_contexts = tuple(str(value) for value in (contexts or ()))
    resolved_pairs: list[tuple[int, int]] = []
    if query_contexts:
        for context in query_contexts:
            prefix_ids = tokenise(context)
            if not prefix_ids:
                raise ValueError(f"structured choice context tokenized empty: {context!r}")
            context_ids: list[int] = []
            for choice in choices:
                combined_ids = tokenise(context + str(choice))
                if (
                    len(combined_ids) != len(prefix_ids) + 1
                    or combined_ids[: len(prefix_ids)] != prefix_ids
                ):
                    raise ValueError(
                        "structured binary choice must append exactly one token in "
                        f"context {context!r}; choice={choice!r}, "
                        f"prefix_ids={prefix_ids}, combined_ids={combined_ids}"
                    )
                context_ids.append(combined_ids[-1])
            resolved_pairs.append((context_ids[0], context_ids[1]))
        if len(set(resolved_pairs)) != 1:
            raise ValueError(
                "structured binary choices resolved to different token ids across "
                f"decision contexts: {resolved_pairs}"
            )
        token_ids = list(resolved_pairs[0])
    else:
        token_ids = []
        for choice in choices:
            values = tokenise(str(choice))
            if len(values) != 1:
                raise ValueError(
                    f"structured binary choice {choice!r} must tokenize to exactly "
                    f"one token; observed token ids={values}"
                )
            token_ids.append(values[0])

    if token_ids[0] == token_ids[1]:
        raise ValueError("binary choice strings resolved to the same token id")
    return token_ids[0], token_ids[1]


def gather_binary_choice_logits(
    lm_logits: torch.Tensor,
    decision_positions: torch.Tensor,
    token_ids: tuple[int, int],
) -> torch.Tensor:
    """Gather 0/1 logits at parallel decision-query positions.

    ``decision_positions`` has shape ``[batch, 6]``: global first, followed by
    target_1 through target_5. The LM logit at each position predicts the neutral
    placeholder token that follows the corresponding ``=`` query. We ignore that
    placeholder and compare the two configured one-token decisions instead.
    """

    if lm_logits.ndim != 3:
        raise ValueError("lm_logits must have shape [batch, sequence, vocabulary]")
    if decision_positions.ndim != 2 or decision_positions.shape[1] != 6:
        raise ValueError("decision_positions must have shape [batch, 6]")
    if decision_positions.shape[0] != lm_logits.shape[0]:
        raise ValueError("decision position batch does not match LM logits")
    if torch.any(decision_positions < 0) or torch.any(
        decision_positions >= lm_logits.shape[1]
    ):
        raise ValueError("decision position is outside the LM sequence")

    batch_indices = torch.arange(lm_logits.shape[0], device=lm_logits.device)[:, None]
    selected = lm_logits[batch_indices, decision_positions]
    choice_ids = torch.tensor(token_ids, device=lm_logits.device, dtype=torch.long)
    return selected.index_select(-1, choice_ids).float()


def global_choice_loss(
    global_logits: torch.Tensor,
    truth_is_anomaly: torch.Tensor,
) -> torch.Tensor:
    if global_logits.ndim != 2 or global_logits.shape[-1] != 2:
        raise ValueError("global_logits must have shape [batch, 2]")
    truth = truth_is_anomaly.to(device=global_logits.device, dtype=torch.long)
    if truth.shape != (global_logits.shape[0],):
        raise ValueError("global truth shape does not match logits")
    return F.cross_entropy(global_logits.float(), truth)


def _score_candidate_sets(target_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if target_logits.ndim != 3 or target_logits.shape[1:] != (TARGET_COUNT, 2):
        raise ValueError("target_logits must have shape [batch, 5, 2]")
    masks = _candidate_masks(target_logits.device)
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    expanded = log_probs[:, None, :, :].expand(-1, masks.shape[0], -1, -1)
    bits = masks.to(torch.long)[None, :, :, None].expand(
        target_logits.shape[0], -1, -1, 1
    )
    candidate_scores = expanded.gather(-1, bits).squeeze(-1).sum(dim=-1)
    return candidate_scores, masks


def _smooth_masked_max(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
    temperature: float,
) -> torch.Tensor:
    """Permutation-equivariant smooth maximum over the selected values.

    A normalized log-sum-exp preserves the value when all selected margins are
    tied and distributes the gradient equally across those ties.  In contrast,
    ``Tensor.max`` sends the complete gradient through the first tied target,
    which creates a target-index prior in an otherwise symmetric objective.
    """

    if temperature <= 0:
        raise ValueError("margin aggregation temperature must be positive")
    selected = mask.to(device=values.device, dtype=torch.bool)
    if selected.shape != values.shape:
        raise ValueError("smooth maximum mask must match the value shape")
    counts = selected.sum(dim=dim)
    if torch.any(counts == 0):
        raise ValueError("smooth maximum requires at least one selected value")
    scaled = (values / float(temperature)).masked_fill(~selected, -torch.inf)
    return float(temperature) * (
        torch.logsumexp(scaled, dim=dim)
        - counts.to(values.dtype).log()
    )


def _class_balanced_per_bit_binary_loss(
    target_logits: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """Average binary CE equally across available classes and target bits.

    The structured sampler balances complete target sets, so a target bit still
    has many more negative than positive decisions.  This term first averages
    within each target bit, then gives the available positive and negative
    classes equal mass.  A class absent from a small batch is skipped rather than
    assigned a fabricated loss.
    """

    per_bit = F.cross_entropy(
        target_logits.float().reshape(-1, 2),
        truth.long().reshape(-1),
        reduction="none",
    ).reshape_as(truth)
    class_losses: list[torch.Tensor] = []
    for class_mask in (truth, ~truth):
        support = class_mask.sum(dim=0)
        supported = support > 0
        if torch.any(supported):
            means = (per_bit * class_mask.to(per_bit.dtype)).sum(dim=0) / support.clamp_min(1)
            class_losses.append(means[supported].mean())
    if not class_losses:
        return target_logits.sum() * 0.0
    return torch.stack(class_losses).mean()


def structured_target_set_loss(
    target_logits: torch.Tensor,
    true_mask: torch.Tensor,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    cardinality_cost: float = 0.5,
    ranking_weight: float = 0.25,
    positive_margin_weight: float = 0.0,
    positive_margin: float = 1.0,
    hard_negative_margin_weight: float = 0.0,
    hard_negative_margin: float = 1.0,
    normal_margin_weight: float = 0.0,
    normal_margin: float = 0.5,
    class_balanced_binary_weight: float = 0.0,
    margin_aggregation_temperature: float = 0.25,
) -> StructuredSetLossResult:
    """Cost-augmented set loss over every subset of the five monitored targets.

    There is no unconditional diversity penalty. Therefore all-zero is correct for
    a genuinely normal target-supervised event, while all-one is allowed only when
    it is the genuine labelled set. For a single-target event, the same loss directly
    penalizes all-zero, all-five, extra targets and the wrong target identity.
    """

    costs = {
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost": float(false_negative_cost),
        "cardinality_cost": float(cardinality_cost),
        "ranking_weight": float(ranking_weight),
        "positive_margin_weight": float(positive_margin_weight),
        "hard_negative_margin_weight": float(hard_negative_margin_weight),
        "normal_margin_weight": float(normal_margin_weight),
        "class_balanced_binary_weight": float(class_balanced_binary_weight),
    }
    if any(value < 0 for value in costs.values()):
        raise ValueError(f"structured set costs must be non-negative: {costs}")
    if float(margin_aggregation_temperature) <= 0:
        raise ValueError("margin_aggregation_temperature must be positive")
    if target_logits.ndim != 3 or target_logits.shape[1:] != (TARGET_COUNT, 2):
        raise ValueError("target_logits must have shape [batch, 5, 2]")
    truth = true_mask.to(device=target_logits.device, dtype=torch.bool)
    if truth.shape != target_logits.shape[:2]:
        raise ValueError("true target mask must have shape [batch, 5]")
    if target_logits.shape[0] == 0:
        zero = target_logits.sum() * 0.0
        return StructuredSetLossResult(
            loss=zero,
            set_loss=zero,
            ranking_loss=zero,
            positive_margin_loss=zero,
            hard_negative_margin_loss=zero,
            normal_margin_loss=zero,
            class_balanced_binary_loss=zero,
            candidate_scores=target_logits.new_zeros((0, 1 << TARGET_COUNT)),
            predicted_mask=truth,
            affected_probabilities=target_logits.new_zeros((0, TARGET_COUNT)),
        )

    candidate_scores, masks = _score_candidate_sets(target_logits)
    candidate = masks[None, :, :]
    truth_expanded = truth[:, None, :]
    false_positives = (candidate & ~truth_expanded).sum(dim=-1).float()
    false_negatives = (~candidate & truth_expanded).sum(dim=-1).float()
    cardinality_error = (
        candidate.sum(dim=-1).float() - truth_expanded.sum(dim=-1).float()
    ).abs()
    cost = (
        costs["false_positive_cost"] * false_positives
        + costs["false_negative_cost"] * false_negatives
        + costs["cardinality_cost"] * cardinality_error
    )

    powers = (2 ** torch.arange(TARGET_COUNT, device=target_logits.device)).long()
    true_indices = (truth.long() * powers[None, :]).sum(dim=-1)
    true_scores = candidate_scores.gather(1, true_indices[:, None]).squeeze(1)
    set_losses = torch.logsumexp(candidate_scores + cost, dim=1) - true_scores
    set_loss = set_losses.mean()

    anomaly_margins = target_logits[..., 1].float() - target_logits[..., 0].float()
    single_target = truth.sum(dim=-1) == 1
    if torch.any(single_target):
        single_margins = anomaly_margins[single_target]
        target_indices = truth[single_target].long().argmax(dim=-1)
        ranking_loss = F.cross_entropy(single_margins, target_indices)

        true_margins = single_margins.gather(
            1, target_indices[:, None]
        ).squeeze(1)
        positive_margin_loss = F.softplus(
            float(positive_margin) - true_margins
        ).mean()

        wrong_mask = torch.ones_like(single_margins, dtype=torch.bool)
        wrong_mask.scatter_(1, target_indices[:, None], False)
        hardest_wrong = _smooth_masked_max(
            single_margins,
            wrong_mask,
            dim=1,
            temperature=float(margin_aggregation_temperature),
        )
        hard_negative_margin_loss = F.softplus(
            float(hard_negative_margin) - (true_margins - hardest_wrong)
        ).mean()
    else:
        ranking_loss = anomaly_margins.sum() * 0.0
        positive_margin_loss = anomaly_margins.sum() * 0.0
        hard_negative_margin_loss = anomaly_margins.sum() * 0.0

    normal_target_set = truth.sum(dim=-1) == 0
    if torch.any(normal_target_set):
        normal_margins = anomaly_margins[normal_target_set]
        maximum_normal_margin = _smooth_masked_max(
            normal_margins,
            torch.ones_like(normal_margins, dtype=torch.bool),
            dim=1,
            temperature=float(margin_aggregation_temperature),
        )
        normal_margin_loss = F.softplus(
            float(normal_margin) + maximum_normal_margin
        ).mean()
    else:
        normal_margin_loss = anomaly_margins.sum() * 0.0

    class_balanced_binary_loss = _class_balanced_per_bit_binary_loss(
        target_logits,
        truth,
    )

    predicted_indices = candidate_scores.argmax(dim=-1)
    predicted_mask = masks.index_select(0, predicted_indices)
    affected_probabilities = torch.softmax(target_logits.float(), dim=-1)[..., 1]
    total = (
        set_loss
        + costs["ranking_weight"] * ranking_loss
        + costs["positive_margin_weight"] * positive_margin_loss
        + costs["hard_negative_margin_weight"] * hard_negative_margin_loss
        + costs["normal_margin_weight"] * normal_margin_loss
        + costs["class_balanced_binary_weight"] * class_balanced_binary_loss
    )
    return StructuredSetLossResult(
        loss=total,
        set_loss=set_loss,
        ranking_loss=ranking_loss,
        positive_margin_loss=positive_margin_loss,
        hard_negative_margin_loss=hard_negative_margin_loss,
        normal_margin_loss=normal_margin_loss,
        class_balanced_binary_loss=class_balanced_binary_loss,
        candidate_scores=candidate_scores,
        predicted_mask=predicted_mask,
        affected_probabilities=affected_probabilities,
    )


def decode_target_set(
    target_logits: torch.Tensor,
    *,
    cardinality_bias: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the highest-scoring complete target set without using global output."""

    candidate_scores, masks = _score_candidate_sets(target_logits)
    if cardinality_bias is not None:
        values = [float(value) for value in cardinality_bias]
        if len(values) != TARGET_COUNT + 1:
            raise ValueError("cardinality_bias must contain values for cardinalities 0..5")
        bias = torch.tensor(values, device=target_logits.device, dtype=candidate_scores.dtype)
        candidate_scores = candidate_scores + bias[masks.sum(dim=-1).long()][None, :]
    indices = candidate_scores.argmax(dim=-1)
    selected = masks.index_select(0, indices)
    probabilities = torch.softmax(target_logits.float(), dim=-1)[..., 1]
    return selected, probabilities, candidate_scores


def _single_source_zone_scores(target_logits: torch.Tensor) -> torch.Tensor:
    """Return categorical scores for ``NONE, T1, ..., T5``.

    Restricting the factorized target-set likelihood to the empty set and the
    five singleton sets yields these scores up to a shared additive constant.
    The empty-set score can therefore be fixed at zero and each singleton score
    is the corresponding affected-versus-normal logit margin.
    """

    if target_logits.ndim != 3 or target_logits.shape[1:] != (TARGET_COUNT, 2):
        raise ValueError("target_logits must have shape [batch, 5, 2]")
    margins = target_logits[..., 1].float() - target_logits[..., 0].float()
    none = margins.new_zeros((margins.shape[0], 1))
    return torch.cat((none, margins), dim=-1)


def _decode_single_source_zone_scores(
    candidate_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = candidate_scores.argmax(dim=-1)
    selected = torch.zeros(
        (candidate_scores.shape[0], TARGET_COUNT),
        device=candidate_scores.device,
        dtype=torch.bool,
    )
    affected = indices > 0
    if torch.any(affected):
        selected[affected, indices[affected] - 1] = True
    probabilities = torch.softmax(candidate_scores, dim=-1)[:, 1:]
    return selected, probabilities


def single_source_zone_loss(
    target_logits: torch.Tensor,
    true_mask: torch.Tensor,
) -> SingleSourceZoneLossResult:
    """Categorical localization loss for normal-or-one-source-zone datasets.

    This objective is deliberately strict. A target-supervised row containing
    more than one affected target violates the data contract instead of being
    silently projected onto an arbitrary singleton class.
    """

    candidate_scores = _single_source_zone_scores(target_logits)
    truth = true_mask.to(device=target_logits.device, dtype=torch.bool)
    if truth.shape != target_logits.shape[:2]:
        raise ValueError("true target mask must have shape [batch, 5]")
    cardinality = truth.sum(dim=-1)
    if torch.any(cardinality > 1):
        observed = cardinality[cardinality > 1].detach().cpu().tolist()
        raise ValueError(
            "single-source-zone supervision requires at most one affected target "
            f"per row; observed cardinalities={observed}"
        )
    if target_logits.shape[0] == 0:
        zero = target_logits.sum() * 0.0
        return SingleSourceZoneLossResult(
            loss=zero,
            candidate_scores=candidate_scores,
            predicted_mask=truth,
            affected_probabilities=target_logits.new_zeros((0, TARGET_COUNT)),
        )

    singleton_classes = truth.long().argmax(dim=-1) + 1
    true_classes = torch.where(
        cardinality == 0,
        torch.zeros_like(singleton_classes),
        singleton_classes,
    )
    loss = F.cross_entropy(candidate_scores, true_classes)
    predicted_mask, affected_probabilities = _decode_single_source_zone_scores(
        candidate_scores
    )
    return SingleSourceZoneLossResult(
        loss=loss,
        candidate_scores=candidate_scores,
        predicted_mask=predicted_mask,
        affected_probabilities=affected_probabilities,
    )


def decode_single_source_zone(
    target_logits: torch.Tensor,
    *,
    cardinality_bias: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode exactly one of ``NONE, T1, ..., T5`` without using global output."""

    candidate_scores = _single_source_zone_scores(target_logits)
    if cardinality_bias is not None:
        values = [float(value) for value in cardinality_bias]
        if len(values) != TARGET_COUNT + 1:
            raise ValueError("cardinality_bias must contain values for cardinalities 0..5")
        bias = torch.tensor(
            values,
            device=target_logits.device,
            dtype=candidate_scores.dtype,
        )
        candidate_scores = candidate_scores.clone()
        candidate_scores[:, 0] += bias[0]
        candidate_scores[:, 1:] += bias[1]
    selected, probabilities = _decode_single_source_zone_scores(candidate_scores)
    return selected, probabilities, candidate_scores


def decision_statistics(
    global_logits: torch.Tensor,
    global_truth: torch.Tensor,
    target_logits: torch.Tensor | None,
    target_truth: torch.Tensor | None,
    *,
    target_predicted_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return additive counts for logging and collapse guards."""

    global_prediction = global_logits.argmax(dim=-1).to(torch.bool)
    truth_global = global_truth.to(device=global_logits.device, dtype=torch.bool)
    normal_truth = ~truth_global
    anomaly_truth = truth_global
    stats = {
        "example_count": torch.tensor(float(global_logits.shape[0]), device=global_logits.device),
        "global_predicted_anomaly": global_prediction.float().sum(),
        "global_normal_support": normal_truth.float().sum(),
        "global_normal_correct": (normal_truth & ~global_prediction).float().sum(),
        "global_anomaly_support": anomaly_truth.float().sum(),
        "global_anomaly_correct": (anomaly_truth & global_prediction).float().sum(),
        "single_target_support": torch.tensor(0.0, device=global_logits.device),
        "single_target_all_zero": torch.tensor(0.0, device=global_logits.device),
        "single_target_all_five": torch.tensor(0.0, device=global_logits.device),
        "single_target_predicted_cardinality": torch.tensor(
            0.0, device=global_logits.device
        ),
    }
    if target_logits is None or target_truth is None or target_logits.shape[0] == 0:
        return stats
    if target_predicted_mask is None:
        decoded, _, _ = decode_target_set(target_logits)
    else:
        decoded = target_predicted_mask.to(
            device=target_logits.device,
            dtype=torch.bool,
        )
        if decoded.shape != target_logits.shape[:2]:
            raise ValueError("target_predicted_mask must have shape [batch, 5]")
    truth = target_truth.to(device=target_logits.device, dtype=torch.bool)
    single = truth.sum(dim=-1) == 1
    if torch.any(single):
        cardinality = decoded[single].sum(dim=-1).float()
        stats["single_target_support"] = single.float().sum()
        stats["single_target_all_zero"] = (cardinality == 0).float().sum()
        stats["single_target_all_five"] = (cardinality == TARGET_COUNT).float().sum()
        stats["single_target_predicted_cardinality"] = cardinality.sum()
    return stats
