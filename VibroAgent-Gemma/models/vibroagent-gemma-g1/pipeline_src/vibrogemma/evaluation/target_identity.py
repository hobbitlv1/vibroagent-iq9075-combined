from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from ..contracts import BoardWindow, EpisodeLabel, SixBoardEpisode
from ..model.structured_set import TARGET_COUNT

TARGET_NAMES: tuple[str, ...] = tuple(f"T{index}" for index in range(1, TARGET_COUNT + 1))
ROUTING_LABELS: tuple[str, ...] = ("NONE", *TARGET_NAMES, "MULTI")


def target_set_name(mask: Sequence[bool] | np.ndarray | torch.Tensor) -> str:
    """Return a compact routing label for a five-target boolean mask."""

    if isinstance(mask, torch.Tensor):
        values = [bool(value) for value in mask.detach().cpu().tolist()]
    else:
        values = [bool(value) for value in np.asarray(mask, dtype=bool).tolist()]
    if len(values) != TARGET_COUNT:
        raise ValueError(f"target mask must contain {TARGET_COUNT} values")
    selected = [index for index, affected in enumerate(values, start=1) if affected]
    if not selected:
        return "NONE"
    if len(selected) == 1:
        return f"T{selected[0]}"
    return "MULTI"


def detailed_target_set_name(mask: Sequence[bool] | np.ndarray | torch.Tensor) -> str:
    """Return NONE, Tn, or an explicit multi-target set such as T1+T4."""

    if isinstance(mask, torch.Tensor):
        values = [bool(value) for value in mask.detach().cpu().tolist()]
    else:
        values = [bool(value) for value in np.asarray(mask, dtype=bool).tolist()]
    if len(values) != TARGET_COUNT:
        raise ValueError(f"target mask must contain {TARGET_COUNT} values")
    selected = [f"T{index}" for index, affected in enumerate(values, start=1) if affected]
    return "+".join(selected) if selected else "NONE"


def label_target_mask(label: EpisodeLabel) -> np.ndarray:
    label.validate()
    if not label.target_supervision_available:
        raise ValueError("target identity requires a target-supervised label")
    return np.asarray([target.affected for target in label.targets], dtype=bool)


def single_target_margin_rank(margins: Sequence[float], truth_mask: Sequence[bool]) -> int | None:
    """Return the 1-based rank of the true target anomaly margin."""

    values = np.asarray(margins, dtype=np.float64)
    truth = np.asarray(truth_mask, dtype=bool)
    if values.shape != (TARGET_COUNT,) or truth.shape != (TARGET_COUNT,):
        raise ValueError("margins and truth mask must both have shape [5]")
    if int(truth.sum()) != 1:
        return None
    truth_index = int(np.flatnonzero(truth)[0])
    order = np.argsort(-values, kind="stable")
    return int(np.flatnonzero(order == truth_index)[0]) + 1


def candidate_index(mask: Sequence[bool] | np.ndarray | torch.Tensor) -> int:
    if isinstance(mask, torch.Tensor):
        values = [bool(value) for value in mask.detach().cpu().tolist()]
    else:
        values = [bool(value) for value in np.asarray(mask, dtype=bool).tolist()]
    if len(values) != TARGET_COUNT:
        raise ValueError(f"target mask must contain {TARGET_COUNT} values")
    return int(sum((1 << index) for index, affected in enumerate(values) if affected))


def descending_rank(values: Sequence[float], index: int) -> int:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not 0 <= int(index) < len(array):
        raise ValueError("rank index is outside the score vector")
    order = np.argsort(-array, kind="stable")
    return int(np.flatnonzero(order == int(index))[0]) + 1


def swap_flat_board_blocks(
    tensor: torch.Tensor,
    first_board: int,
    second_board: int,
    *,
    tokens_per_board: int,
) -> torch.Tensor:
    """Swap two board blocks in a flat [batch, 6*tokens, width] tensor."""

    if tensor.ndim != 3:
        raise ValueError("flat board tensor must have shape [batch, sequence, width]")
    if tensor.shape[1] != 6 * int(tokens_per_board):
        raise ValueError("flat board tensor sequence length does not match six board blocks")
    if first_board == second_board or first_board not in range(6) or second_board not in range(6):
        raise ValueError("board indices must be two distinct values in 0..5")
    output = tensor.clone()
    a0 = first_board * tokens_per_board
    a1 = a0 + tokens_per_board
    b0 = second_board * tokens_per_board
    b1 = b0 + tokens_per_board
    first = tensor[:, a0:a1, :].clone()
    second = tensor[:, b0:b1, :].clone()
    output[:, a0:a1, :] = second
    output[:, b0:b1, :] = first
    return output


def permute_margin_pair(values: Sequence[float], first_target: int, second_target: int) -> np.ndarray:
    margins = np.asarray(values, dtype=np.float64).copy()
    if margins.shape != (TARGET_COUNT,):
        raise ValueError("target margins must have shape [5]")
    first = int(first_target) - 1
    second = int(second_target) - 1
    if first == second or first not in range(TARGET_COUNT) or second not in range(TARGET_COUNT):
        raise ValueError("target indices must be two distinct values in 1..5")
    margins[[first, second]] = margins[[second, first]]
    return margins


def swap_follow_diagnostics(
    baseline_margins: Sequence[float],
    swapped_margins: Sequence[float],
    first_target: int,
    second_target: int,
) -> dict[str, float | bool]:
    """Compare a swap result with content-follow and slot-stay hypotheses."""

    baseline = np.asarray(baseline_margins, dtype=np.float64)
    swapped = np.asarray(swapped_margins, dtype=np.float64)
    if baseline.shape != (TARGET_COUNT,) or swapped.shape != (TARGET_COUNT,):
        raise ValueError("target margins must have shape [5]")
    expected_follow = permute_margin_pair(baseline, first_target, second_target)
    distance_follow = float(np.linalg.norm(swapped - expected_follow))
    distance_stay = float(np.linalg.norm(swapped - baseline))
    denominator = max(distance_follow + distance_stay, 1e-12)
    follow_score = float((distance_stay - distance_follow) / denominator)
    return {
        "distance_to_content_follow": distance_follow,
        "distance_to_slot_stay": distance_stay,
        "content_follow_score": follow_score,
        "follows_content_more_than_slot": bool(distance_follow < distance_stay),
    }


def _copy_payload(board: BoardWindow) -> dict[str, Any]:
    return {
        "values": board.values.copy(),
        "sample_rate_hz": float(board.sample_rate_hz),
        "start_time_s": float(board.start_time_s),
        "axis_mask": board.axis_mask.copy(),
        "sample_mask": board.sample_mask.copy(),
        "orientation_matrix": (
            None if board.orientation_matrix is None else board.orientation_matrix.copy()
        ),
        "quality": copy.deepcopy(board.quality),
        "metadata": copy.deepcopy(board.metadata),
    }


def _apply_payload(board: BoardWindow, payload: dict[str, Any]) -> None:
    board.values = payload["values"].copy()
    board.sample_rate_hz = float(payload["sample_rate_hz"])
    board.start_time_s = float(payload["start_time_s"])
    board.axis_mask = payload["axis_mask"].copy()
    board.sample_mask = payload["sample_mask"].copy()
    board.orientation_matrix = (
        None
        if payload["orientation_matrix"] is None
        else payload["orientation_matrix"].copy()
    )
    board.quality = copy.deepcopy(payload["quality"])
    board.metadata = copy.deepcopy(payload["metadata"])


def swap_episode_measurement_payload(
    episode: SixBoardEpisode,
    first_target: int,
    second_target: int,
) -> SixBoardEpisode:
    """Swap complete measurement payloads while retaining target slots/positions.

    The receiving board keeps its ``board_id``, ``target_index``, role and
    ``sensor_position``. Waveform samples, masks, timing, orientation calibration,
    quality and acquisition metadata move together. This is a diagnostic
    counterfactual, not a physically valid training example.
    """

    first = int(first_target)
    second = int(second_target)
    if first == second or first not in range(1, 6) or second not in range(1, 6):
        raise ValueError("target indices must be two distinct values in 1..5")
    changed = copy.deepcopy(episode)
    first_payload = _copy_payload(changed.boards[first])
    second_payload = _copy_payload(changed.boards[second])
    _apply_payload(changed.boards[first], second_payload)
    _apply_payload(changed.boards[second], first_payload)
    changed.episode_id = f"{episode.episode_id}::payload-swap-T{first}-T{second}"
    changed.metadata = {
        **changed.metadata,
        "target_identity_audit_intervention": {
            "kind": "measurement_payload_swap",
            "first_target": first,
            "second_target": second,
            "label_preserved_for_diagnostic_only": True,
        },
    }
    changed.validate(require_label=episode.label is not None)
    return changed


@contextmanager
def swapped_board_identity_embeddings(
    embedding: torch.nn.Embedding,
    first_target: int,
    second_target: int,
) -> Iterator[None]:
    """Temporarily swap learned target-slot identity embeddings and restore them."""

    first = int(first_target)
    second = int(second_target)
    if first == second or first not in range(1, 6) or second not in range(1, 6):
        raise ValueError("target indices must be two distinct values in 1..5")
    with torch.no_grad():
        original_first = embedding.weight[first].detach().clone()
        original_second = embedding.weight[second].detach().clone()
        embedding.weight[first].copy_(original_second)
        embedding.weight[second].copy_(original_first)
    try:
        yield
    finally:
        with torch.no_grad():
            embedding.weight[first].copy_(original_first)
            embedding.weight[second].copy_(original_second)


def board_mapping_receipt(
    episode: SixBoardEpisode,
    *,
    tokens_per_board: int,
    decision_positions: Sequence[int],
) -> dict[str, Any]:
    if len(decision_positions) != 6:
        raise ValueError("six decision positions are required")
    boards = []
    for board_index, board in enumerate(episode.boards):
        token_start = board_index * int(tokens_per_board)
        boards.append(
            {
                "board_index": board_index,
                "board_id": board.board_id,
                "role": board.role,
                "target_index": board.target_index,
                "sensor_position": (
                    list(board.sensor_position) if board.sensor_position is not None else None
                ),
                "axis_mask": [bool(value) for value in board.axis_mask.tolist()],
                "soft_token_start": token_start,
                "soft_token_stop_exclusive": token_start + int(tokens_per_board),
                "decision_slot": "GLOBAL" if board_index == 0 else f"T{board_index}",
                "decision_position": int(decision_positions[board_index]),
            }
        )
    return {
        "episode_id": episode.episode_id,
        "dataset_id": episode.provenance.dataset_id,
        "split_group": episode.provenance.split_group,
        "tokens_per_board": int(tokens_per_board),
        "boards": boards,
    }


def routing_matrix(
    rows: Iterable[dict[str, Any]],
    *,
    truth_key: str = "truth_route",
    prediction_key: str = "predicted_route",
) -> dict[str, Any]:
    labels = list(ROUTING_LABELS)
    index = {name: position for position, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    ignored = 0
    for row in rows:
        truth = str(row.get(truth_key, ""))
        prediction = str(row.get(prediction_key, ""))
        if truth not in index or prediction not in index:
            ignored += 1
            continue
        matrix[index[truth], index[prediction]] += 1
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "ignored_rows": ignored,
        "total_rows": int(matrix.sum()),
    }


def summarize_margin_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        truth = str(row.get("truth_route", ""))
        dataset = str(row.get("dataset_id", ""))
        if truth not in TARGET_NAMES:
            continue
        grouped.setdefault((dataset, truth), []).append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, truth), values in sorted(grouped.items()):
        true_index = int(truth[1:]) - 1
        true_margins = np.asarray(
            [float(item["target_margins"][true_index]) for item in values],
            dtype=np.float64,
        )
        ranks = [
            int(item["true_target_margin_rank"])
            for item in values
            if item.get("true_target_margin_rank") is not None
        ]
        summary.append(
            {
                "dataset_id": dataset,
                "truth_route": truth,
                "episodes": len(values),
                "true_margin_mean": float(true_margins.mean()),
                "true_margin_median": float(np.median(true_margins)),
                "true_margin_min": float(true_margins.min()),
                "true_margin_max": float(true_margins.max()),
                "true_margin_positive_rate": float(np.mean(true_margins > 0.0)),
                "true_margin_rank_mean": float(np.mean(ranks)) if ranks else None,
                "true_margin_top1_rate": (
                    float(np.mean(np.asarray(ranks) == 1)) if ranks else None
                ),
            }
        )
    return summary
