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


_ADAPTER_WITH_A_CALL_HOLDING_THE_MODEL = r"""
import os
import sys
import threading
import time

from vibroagent_mcp import geniex_openai_server as server

HOLD_S = float(sys.argv[2])
server.WATCHDOG_POLL_S = 0.2


class StubModel:
    def close(self):
        print("MODEL CLOSED", flush=True)


class StubRuntime:
    is_vlm = False
    _needs_model_reload = False
    close = server._GenieXRuntime.close
    loads = 0

    def __init__(self, **_settings):
        self._lock = threading.Lock()
        self._model = None
        self.call_started = None

    def load(self):
        self._model = StubModel()
        StubRuntime.loads += 1
        if StubRuntime.loads > 1:
            return

        def call():
            # Holds the single-flight lock for HOLD_S seconds, as a slow native call does.
            with self._lock:
                self.call_started = time.monotonic()
                time.sleep(HOLD_S)
                print("CALL FINISHED", flush=True)

        threading.Thread(target=call, daemon=True).start()

    def startup_smoke(self):
        return {"ok": True}

    def stuck_generation(self, limit_s):
        if self.call_started is None:
            return None
        age_s = time.monotonic() - self.call_started
        return (age_s, "stub generation") if age_s >= limit_s else None

    def stuck_reload(self, _limit_s):
        return None


def serve_again_instead_of_exec(path, argv):
    # Stands in for the new process: the relaunch command has to be one this server accepts.
    print("RESTART " + " ".join(argv[1:]), flush=True)
    arguments = list(argv[4:])
    arguments[arguments.index("--generation-watchdog-s") + 1] = "0"
    os._exit(server.main(arguments))


server.os.execv = serve_again_instead_of_exec
server._GenieXRuntime = StubRuntime
raise SystemExit(server.main(["--port", sys.argv[1], "--model", "stub.gguf", "--generation-watchdog-s", "0.3"]))
"""


def _start_adapter_with_a_held_model(hold_s, stop_wait_s):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(server.__file__).resolve().parents[1])
    env.pop("GENIEX_STARTUP_SMOKE", None)
    env["GENIEX_STOP_WAIT_S"] = str(stop_wait_s)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", _ADAPTER_WITH_A_CALL_HOLDING_THE_MODEL, str(port), str(hold_s)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    return process, port


def _read_until(process, marker, occurrences=1):
    seen = ""
    while seen.count(marker) < occurrences:
        line = process.stdout.readline()
        assert line, seen
        seen += line
    return seen


def _stop(process, seen):
    try:
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return seen + output


def test_watchdog_restart_destroys_the_model_before_the_server_starts_again():
    process, port = _start_adapter_with_a_held_model(hold_s=1.5, stop_wait_s=10)
    output = _stop(process, _read_until(process, "Serving GenieX", occurrences=2))

    events = ["GENIEX WATCHDOG: stub generation stuck", "CALL FINISHED", "MODEL CLOSED",
              "model released, restarting", "RESTART -u -m vibroagent_mcp.geniex_openai_server"]
    positions = [output.find(event) for event in events]
    assert -1 not in positions and positions == sorted(positions), output
    restart_line = next(line for line in output.splitlines() if line.startswith("RESTART "))
    assert f"--port {port} --model stub.gguf" in restart_line
    assert "--generation-watchdog-s 0.3" in restart_line
    # The restarted server loaded a model of its own and released it on the stop.
    assert output.count("MODEL CLOSED") == 2
    assert process.returncode == 0


def test_watchdog_restart_says_so_when_the_call_never_releases_the_model():
    process, _port = _start_adapter_with_a_held_model(hold_s=60, stop_wait_s=0.5)
    output = _stop(process, _read_until(process, "Serving GenieX", occurrences=2))

    before_restart = output[: output.index("RESTART ")]
    assert "MODEL CLOSED" not in before_restart
    assert "restarting without releasing it" in before_restart


def test_a_stop_during_a_watchdog_restart_cancels_the_restart():
    process, _port = _start_adapter_with_a_held_model(hold_s=60, stop_wait_s=3)
    output = _stop(process, _read_until(process, "GENIEX WATCHDOG"))

    assert "RESTART" not in output
    assert "exiting without releasing it" in output
    assert process.returncode == 0
