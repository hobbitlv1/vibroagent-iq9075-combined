"""The vibration encoder sizes its thread pool from the cores its thread may use."""

import sys
from types import SimpleNamespace

import pytest

from scripts import live_vibrogemma_worker as worker


@pytest.fixture
def created_sessions(monkeypatch):
    """Replace onnxruntime with a recorder of the options each session is created with."""
    created = []

    def inference_session(path, sess_options, providers):
        created.append({"path": path, "options": sess_options, "providers": providers})
        return object()

    fake = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=inference_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return created


def test_encoder_uses_every_core_its_thread_may_use(monkeypatch, created_sessions, tmp_path):
    monkeypatch.delenv("VIBROGEMMA_ENCODER_THREADS", raising=False)
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5}, raising=False)

    worker._encoder_session.__wrapped__(str(tmp_path), "artifact-key")

    options = created_sessions[0]["options"]
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (6, 1)
    assert created_sessions[0]["providers"] == ["CPUExecutionProvider"]


def test_explicit_encoder_thread_count_wins(monkeypatch, created_sessions, tmp_path):
    monkeypatch.setenv("VIBROGEMMA_ENCODER_THREADS", "2")
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5}, raising=False)

    worker._encoder_session.__wrapped__(str(tmp_path), "artifact-key")

    assert created_sessions[0]["options"].intra_op_num_threads == 2


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_encoder_thread_count_is_rejected_before_a_session_exists(monkeypatch, created_sessions, tmp_path, value):
    monkeypatch.setenv("VIBROGEMMA_ENCODER_THREADS", value)

    with pytest.raises(RuntimeError, match="VIBROGEMMA_ENCODER_THREADS"):
        worker._encoder_session.__wrapped__(str(tmp_path), "artifact-key")

    assert created_sessions == []
