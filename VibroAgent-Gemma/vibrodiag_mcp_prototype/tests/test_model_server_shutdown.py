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


_SERVER_WITH_A_REQUEST_HOLDING_THE_MODEL = r"""
import os
import sys
import threading
import time

import vibroagent_mcp.vibrogemma_geniex_server as server

HOLD_S = float(sys.argv[2])


class StubModel:
    def close(self):
        print("MODEL CLOSED", flush=True)


class StubRuntime:
    n_threads = 1
    close = server.VibroGemmaRuntime.close

    def __init__(self, **_kwargs):
        self.lock = threading.Lock()
        self.model = None
        self.request_started = None

    def load(self):
        self.model = StubModel()

        def request():
            # Holds the model for HOLD_S seconds, as a slow native call does.
            with self.lock:
                self.request_started = time.monotonic()
                time.sleep(HOLD_S)
                print("REQUEST FINISHED", flush=True)

        threading.Thread(target=request, daemon=True).start()

    def stuck_inference(self, limit_s):
        if self.request_started is None:
            return None
        age_s = time.monotonic() - self.request_started
        return (age_s, "stub request") if age_s >= limit_s else None


def restart_instead_of_exec(path, argv):
    print("RESTART " + " ".join(argv[1:4]), flush=True)
    os._exit(0)


server.os.execv = restart_instead_of_exec
server.VibroGemmaRuntime = StubRuntime
sys.argv = ["server", "--port", sys.argv[1]]
raise SystemExit(server.main())
"""


def _start_server_with_a_held_model(tmp_path, hold_s, stop_wait_s):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(geniex_server.__file__).resolve().parents[1])
    env["VIBROGEMMA_GGUF"] = str(tmp_path / "model.gguf")
    env["VIBROGEMMA_BUNDLE"] = str(tmp_path)
    env["VIBROGEMMA_GENERATION_WATCHDOG_S"] = "0.5"
    env["VIBROGEMMA_STOP_WAIT_S"] = str(stop_wait_s)
    return subprocess.Popen(
        [sys.executable, "-u", "-c", _SERVER_WITH_A_REQUEST_HOLDING_THE_MODEL, str(port), str(hold_s)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )


def _finish(process, timeout=30):
    try:
        output, _ = process.communicate(timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return output


def test_watchdog_restart_waits_for_the_slow_request_and_destroys_the_model_first(tmp_path):
    output = _finish(_start_server_with_a_held_model(tmp_path, hold_s=3, stop_wait_s=10))

    events = ["VIBROGEMMA WATCHDOG: stub request stuck", "REQUEST FINISHED", "MODEL CLOSED",
              "model released, restarting", "RESTART -u -m vibroagent_mcp.vibrogemma_geniex_server"]
    positions = [output.find(event) for event in events]
    assert -1 not in positions and positions == sorted(positions), output


def test_watchdog_restart_says_so_when_the_request_never_releases_the_model(tmp_path):
    output = _finish(_start_server_with_a_held_model(tmp_path, hold_s=60, stop_wait_s=1))

    assert "MODEL CLOSED" not in output
    assert "restarting without releasing it" in output
    assert output.rstrip().endswith("RESTART -u -m vibroagent_mcp.vibrogemma_geniex_server"), output


def test_a_stop_during_a_watchdog_restart_cancels_the_restart(tmp_path):
    process = _start_server_with_a_held_model(tmp_path, hold_s=60, stop_wait_s=4)
    seen = ""
    while "VIBROGEMMA WATCHDOG" not in seen:
        line = process.stdout.readline()
        assert line, seen
        seen += line
    process.send_signal(signal.SIGTERM)
    output = seen + _finish(process)

    assert "RESTART" not in output
    assert "exiting without releasing it" in output
    assert process.returncode == 0
