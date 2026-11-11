"""Stopping the model server destroys the model first, so the runtime can end its device session."""

import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import vibroagent_mcp.vibrogemma_geniex_server as geniex_server
from vibroagent_mcp.vibrogemma_geniex_server import VibroGemmaRuntime


class _Model:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _runtime(model):
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.lock = threading.Lock()
    runtime.model = model
    return runtime


def test_close_destroys_the_model_once():
    model = _Model()
    runtime = _runtime(model)

    assert runtime.close() is True
    assert runtime.close() is True

    assert model.closed == 1
    assert runtime.model is None


def test_close_leaves_a_model_that_a_request_still_uses():
    model = _Model()
    runtime = _runtime(model)

    with runtime.lock:
        released = runtime.close(wait_s=0.05)

    assert released is False
    assert model.closed == 0
    assert runtime.model is model


_SERVER_WITH_STUB_MODEL = r"""
import sys
import threading

import vibroagent_mcp.vibrogemma_geniex_server as server


class StubModel:
    def close(self):
        print("MODEL CLOSED", flush=True)


class StubRuntime:
    n_threads = 1
    close = server.VibroGemmaRuntime.close

    def __init__(self, **_kwargs):
        self.lock = threading.Lock()
        self.model = None

    def load(self):
        self.model = StubModel()

    def stuck_inference(self, _limit_s):
        return None


server.VibroGemmaRuntime = StubRuntime
sys.argv = ["server", "--port", sys.argv[1]]
raise SystemExit(server.main())
"""


def test_sigterm_closes_the_model_and_exits_cleanly(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(geniex_server.__file__).resolve().parents[1])
    env["VIBROGEMMA_GGUF"] = str(tmp_path / "model.gguf")
    env["VIBROGEMMA_BUNDLE"] = str(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", _SERVER_WITH_STUB_MODEL, str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    try:
        first_line = process.stdout.readline()
        assert "Serving VibroAgent-Gemma" in first_line, first_line + process.stdout.read()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert "MODEL CLOSED" in output
    assert process.returncode == 0
