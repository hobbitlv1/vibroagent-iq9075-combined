"""The model server sizes the GenieX thread pool itself instead of leaving it to the runtime."""

import pytest

import vibroagent_mcp.vibrogemma_geniex_server as geniex_server
from vibroagent_mcp.vibrogemma_geniex_server import _inference_thread_count


def test_defaults_to_the_cores_the_process_may_use(monkeypatch):
    monkeypatch.delenv("VIBROGEMMA_N_THREADS", raising=False)
    monkeypatch.setattr(geniex_server.os, "sched_getaffinity", lambda pid: {4, 5, 6, 7}, raising=False)
    monkeypatch.setattr(geniex_server.os, "cpu_count", lambda: 8)

    assert _inference_thread_count() == 4


def test_falls_back_to_the_machine_core_count_without_affinity_support(monkeypatch):
    def unsupported(pid):
        raise OSError("affinity is not available here")

    monkeypatch.delenv("VIBROGEMMA_N_THREADS", raising=False)
    monkeypatch.setattr(geniex_server.os, "sched_getaffinity", unsupported, raising=False)
    monkeypatch.setattr(geniex_server.os, "cpu_count", lambda: 8)

    assert _inference_thread_count() == 8


def test_explicit_setting_wins_over_the_pinned_cores(monkeypatch):
    monkeypatch.setenv("VIBROGEMMA_N_THREADS", " 8 ")
    monkeypatch.setattr(geniex_server.os, "sched_getaffinity", lambda pid: {6, 7}, raising=False)

    assert _inference_thread_count() == 8


@pytest.mark.parametrize("value", ["0", "-2", "four", "2.5"])
def test_rejects_a_setting_that_is_not_a_positive_integer(monkeypatch, value):
    monkeypatch.setenv("VIBROGEMMA_N_THREADS", value)

    with pytest.raises(RuntimeError, match="VIBROGEMMA_N_THREADS"):
        _inference_thread_count()
