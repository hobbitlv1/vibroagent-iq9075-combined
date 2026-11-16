"""The model server computes logits only for the runtime calls whose logits are read."""

import pytest

from vibroagent_mcp.vibrogemma_geniex_server import VibroGemmaRuntime, _decode_can_skip_logits


class RecordingModel:
    """Stands in for the patched runtime: logits can be read only if the latest call computed them."""

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def decode(self, *, input_ids=None, input_embd=None, logits=True):
        kind, values = ("tokens", input_ids) if input_embd is None else ("embeddings", input_embd)
        self.calls.append((kind, len(values), logits))

    def state_logits(self, token_ids):
        if not self.calls[-1][2]:
            raise RuntimeError("the latest decode computed no logits")
        decoded_positions = sum(count for _kind, count, _logits in self.calls)
        return [float(decoded_positions), 0.0]


class ThreePatchModel:
    """A runtime built before patch 0004: decode() has no logits switch."""

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def decode(self, *, input_ids=None, input_embd=None):
        kind, values = ("tokens", input_ids) if input_embd is None else ("embeddings", input_embd)
        self.calls.append((kind, len(values)))

    def state_logits(self, token_ids):
        return [float(sum(count for _kind, count in self.calls)), 0.0]


def _runtime(model, n_batch=512):
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = model
    runtime.n_batch = n_batch
    return runtime


def _window_like_segments():
    """Text, two boards of embeddings, instructions, then six answer contexts with placeholders."""
    segments = [
        ("tokens", list(range(20))),
        ("embeddings", [[0.0]] * 14),
        ("tokens", list(range(9))),
        ("embeddings", [[0.0]] * 14),
        ("tokens", list(range(30))),
    ]
    positions = []
    decoded = sum(len(values) for _kind, values in segments)
    for _answer in range(6):
        segments.append(("tokens", [7, 8, 9, 10]))
        decoded += 4
        positions.append(decoded - 1)
        segments.append(("tokens", [11]))
        decoded += 1
    return segments, positions


def test_only_the_last_runtime_call_computes_logits():
    model = RecordingModel()

    _runtime(model, n_batch=4)._decode_segments(
        [("tokens", list(range(10))), ("tokens", []), ("embeddings", [[0.0]] * 3)]
    )

    assert model.calls == [
        ("tokens", 4, False),
        ("tokens", 4, False),
        ("tokens", 2, False),
        ("embeddings", 3, True),
    ]


def test_a_token_segment_split_by_the_batch_size_computes_logits_once():
    model = RecordingModel()

    _runtime(model, n_batch=4)._decode_segments([("tokens", list(range(9)))])

    assert model.calls == [("tokens", 4, False), ("tokens", 4, False), ("tokens", 1, True)]


def test_input_followed_by_more_input_computes_no_logits():
    model = RecordingModel()

    _runtime(model, n_batch=4)._decode_segments(
        [("tokens", list(range(6))), ("embeddings", [[0.0]] * 3)], logits=False
    )

    assert [logits for _kind, _count, logits in model.calls] == [False, False, False]


def test_six_answers_cost_six_logit_computations():
    model = RecordingModel()
    segments, positions = _window_like_segments()

    rows = _runtime(model)._state_logits_at_positions(segments, positions, [1, 2])

    # Every row was read right after the call that ends at its answer position.
    assert [row[0] for row in rows] == [position + 1 for position in positions]
    assert sum(logits for _kind, _count, logits in model.calls) == 6
    assert len(model.calls) == 16
    # The trailing placeholder of the last answer is never decoded.
    assert sum(count for _kind, count, _logits in model.calls) == positions[-1] + 1


def test_a_runtime_without_the_switch_is_called_as_before():
    model = ThreePatchModel()
    segments, positions = _window_like_segments()

    rows = _runtime(model)._state_logits_at_positions(segments, positions, [1, 2])

    assert not _decode_can_skip_logits(model)
    assert [row[0] for row in rows] == [position + 1 for position in positions]
    assert len(model.calls) == 16


@pytest.mark.parametrize("model, expected", [(RecordingModel(), True), (ThreePatchModel(), False), (object(), False)])
def test_the_switch_is_detected_from_the_runtime(model, expected):
    assert _decode_can_skip_logits(model) is expected
