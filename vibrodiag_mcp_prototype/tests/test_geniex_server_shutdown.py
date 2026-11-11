"""Stopping the GenieX adapter destroys the model first, so the runtime can end its NPU session."""

import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

from vibroagent_mcp import geniex_openai_server as server


class _Model:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _runtime(model):
    runtime = server._GenieXRuntime.__new__(server._GenieXRuntime)
    runtime._lock = threading.Lock()
    runtime._model = model
    return runtime


def test_close_destroys_the_model_once():
    model = _Model()
    runtime = _runtime(model)

    assert runtime.close() is True
    assert runtime.close() is True

    assert model.closed == 1
    assert runtime._model is None


def test_close_leaves_a_model_that_a_request_still_uses():
    model = _Model()
    runtime = _runtime(model)

    with runtime._lock:
        released = runtime.close(wait_s=0.05)

    assert released is False
    assert model.closed == 0
    assert runtime._model is model


_ADAPTER_WITH_STUB_MODEL = r"""
import sys
import threading

from vibroagent_mcp import geniex_openai_server as server


class StubModel:
    def close(self):
        print("MODEL CLOSED", flush=True)


class StubRuntime:
    is_vlm = False
    _needs_model_reload = False
    close = server._GenieXRuntime.close

    def __init__(self, **_settings):
        self._lock = threading.Lock()
        self._model = None

    def load(self):
        self._model = StubModel()

    def startup_smoke(self):
        if sys.argv[2] == "smoke-fails":
            raise RuntimeError("stub smoke failure")
        return {"ok": True}

    def stuck_generation(self, _limit_s):
        return None

    def stuck_reload(self, _limit_s):
        return None


server._GenieXRuntime = StubRuntime
raise SystemExit(server.main(["--port", sys.argv[1], "--model", "stub.gguf", "--generation-watchdog-s", "0"]))
"""


def _start_adapter(mode):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(server.__file__).resolve().parents[1])
    env.pop("GENIEX_STARTUP_SMOKE", None)
    return subprocess.Popen(
        [sys.executable, "-u", "-c", _ADAPTER_WITH_STUB_MODEL, str(port), mode],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )


def test_sigterm_closes_the_model_and_exits_cleanly():
    process = _start_adapter("serve")
    try:
        seen = ""
        while "Serving GenieX" not in seen:
            line = process.stdout.readline()
            assert line, seen
            seen += line
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert "MODEL CLOSED" in output
    assert "model released" in output
    assert process.returncode == 0


def test_a_failed_startup_smoke_still_closes_the_model():
    process = _start_adapter("smoke-fails")
    try:
        output, _ = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert "STARTUP SMOKE FAILED" in output
    assert "MODEL CLOSED" in output
    assert process.returncode == 1
