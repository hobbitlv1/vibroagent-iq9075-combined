"""Launcher regressions with no SDK, pipeline processes, HTTP, or USB calls."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "vibroagent.sh"


def shell(body):
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(LAUNCHER))}\n{body}"],
        capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("host, expected", [
    ("192.168.1.5", "http://192.168.1.5:7860"),
    ("127.0.0.1", "http://127.0.0.1:7860"),
    ("0.0.0.0", "http://127.0.0.1:7860"),
    ("::", "http://127.0.0.1:7860"),
    ("::1", "http://[::1]:7860"),
    ("[::1]", "http://[::1]:7860"),
])
def test_local_web_probe_host(host, expected):
    result = shell(f"WEB_HOST={shlex.quote(host)}; WEB_PORT=7860; webchat_probe_url")
    assert result.returncode == 0
    assert result.stdout == expected


def logger_mocks(tmp_path, *, alive="return 0", curl_status=0):
    (tmp_path / "vibroagent.sh").touch()
    (tmp_path / ".run").mkdir(exist_ok=True)
    return f'''
REPO={shlex.quote(str(tmp_path))}
RUN_DIR="$REPO/.run"
VENV=/bin/true
MODE=live
VIBRO_MODE=live
logger_pid() {{ :; }}
board_count() {{ echo 6; }}
baseline_present() {{ return 0; }}
logger_is_alive() {{ {alive}; }}
sleep() {{ :; }}
nohup() {{ for i in {{1..6}}; do echo "Started mock_$i on device $i"; done; }}
curl() {{ printf 'CURL_ARG:%s\\n' "$@"; return {curl_status}; }}
# Mock the ready log independently of the asynchronous no-op shell child.
grep() {{ [[ "$1" = -cE ]] && {{ echo 6; return 0; }}; return 1; }}
'''


@pytest.mark.parametrize("stack, route", [
    ("gemma", "/api/vibro/sensors"),
    ("codes", "/api/live-sensors"),
    ("genie", "/api/live-sensors"),
])
@pytest.mark.parametrize("curl_status", [0, 7, 22, 28])
def test_optional_prewarm_uses_bound_host_and_never_fails_live_logger(tmp_path, stack, route, curl_status):
    result = shell(logger_mocks(tmp_path, curl_status=curl_status) + f'''
STACK={stack}; WEB_HOST=192.168.1.5; WEB_PORT=7860
start_logger
''')
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Logger up — 6 board(s) acquiring" in result.stdout
    assert f"CURL_ARG:http://192.168.1.5:7860{route}?process_reader=1&prewarm=1" in result.stdout
    assert "CURL_ARG:--noproxy\nCURL_ARG:*" in result.stdout
    assert "CURL_ARG:--fail" in result.stdout
    assert ("prewarm unavailable" in result.stdout) == bool(curl_status)


def test_started_lines_do_not_claim_an_exited_logger_is_alive(tmp_path):
    result = shell(logger_mocks(tmp_path, alive="return 1") + "start_logger")
    assert result.returncode == 1
    assert "Logger up" not in result.stdout
    assert "CURL_ARG:" not in result.stdout


def test_logger_exit_during_optional_prewarm_still_fails_startup(tmp_path):
    result = shell(logger_mocks(tmp_path, alive='[[ "${CHECKED:-0}" = 0 ]] && { CHECKED=1; return 0; }; return 1') + "start_logger")
    assert result.returncode == 1
    assert "Logger exited during reader prewarm" in result.stdout


def test_liveness_rejects_an_unrelated_live_python_pid():
    # The test runner exists, but is not this checkout's USB logger.
    import os
    result = shell(f"VENV={shlex.quote(sys.executable)}; logger_is_alive {os.getpid()}")
    assert result.returncode == 1


@pytest.mark.parametrize("mode", ["live", "offline"])
@pytest.mark.parametrize("keep_pid", [False, True])
def test_direct_start_preserves_unconfirmed_shutdown(tmp_path, mode, keep_pid):
    mocks = logger_mocks(tmp_path)
    run_dir = tmp_path / ".run"
    pid = int(subprocess.check_output([sys.executable, "-c", "import os; print(os.getpid())"], text=True))
    log = run_dir / "logger.log"
    prefix = "Started baseline on device 0, serial mock\n"
    log.write_text(prefix)
    intent = {"pid": pid, "started": "99", "inode": log.stat().st_ino, "offset": len(prefix.encode())}
    intent_path = run_dir / "logger-stop-request.json"
    intent_path.write_text(json.dumps(intent))
    with log.open("a") as stream:
        stream.write("Shutdown result: " + json.dumps({
            "pid": pid, "started": ["baseline"], "stopped": [], "errors": ["native failure"],
        }))
    if keep_pid:
        (run_dir / "logger.pid").write_text(str(pid))
    before_log, before_intent = log.read_bytes(), intent_path.read_bytes()
    # A direct repeated start must not skip the gate just because a PID was found.
    result = shell(mocks + f'''
MODE={mode}; VIBRO_MODE={mode}
logger_pid() {{ echo 777; }}
board_count() {{ touch "$RUN_DIR/board_probe"; echo 6; }}
stop_logger
stop_rc=$?
start_logger
start_rc=$?
stop_logger
retry_rc=$?
printf 'RESULT:%s:%s:%s\\n' "$stop_rc" "$start_rc" "$retry_rc"
''')
    assert "RESULT:2:2:2" in result.stdout, result.stderr + result.stdout
    assert not (run_dir / "board_probe").exists()
    assert not (run_dir / "native_probe.log").exists()
    assert log.read_bytes() == before_log
    assert intent_path.read_bytes() == before_intent
    assert (run_dir / "logger.pid").exists() == keep_pid


@pytest.mark.parametrize("mode", ["live", "offline"])
def test_acknowledged_shutdown_allows_start_without_erasing_log(tmp_path, mode):
    mocks = logger_mocks(tmp_path)
    run_dir = tmp_path / ".run"
    pid = int(subprocess.check_output([sys.executable, "-c", "import os; print(os.getpid())"], text=True))
    log = run_dir / "logger.log"
    log.write_text("Started baseline on device 0, serial mock\n")
    intent = {"pid": pid, "started": "99", "inode": log.stat().st_ino, "offset": log.stat().st_size}
    (run_dir / "logger-stop-request.json").write_text(json.dumps(intent))
    (run_dir / "logger.pid").write_text(str(pid))
    with log.open("a") as stream:
        stream.write("Shutdown result: " + json.dumps({
            "pid": pid, "started": ["baseline"], "stopped": ["baseline"], "errors": [],
        }))
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "manifest.json").write_text("{}")
    for role in ("baseline", *(f"target_{i}" for i in range(1, 6))):
        folder = recordings / f"live_{role}"
        folder.mkdir()
        (folder / "iis3dwb_acc.dat").write_bytes(b"immutable")
    result = shell(mocks + f'''
MODE={mode}; VIBRO_MODE={mode}
REPLAY_SOURCE_ROOT="$REPO/recordings"
REPLAY_MANIFEST="$REPLAY_SOURCE_ROOT/manifest.json"
board_count() {{ touch "$RUN_DIR/board_probe"; echo 0; }}
start_logger
first_rc=$?
start_logger
printf 'RESULT:%s:%s\\n' "$first_rc" "$?"
''')
    assert f"RESULT:{1 if mode == 'live' else 0}:{1 if mode == 'live' else 0}" in result.stdout, result.stderr + result.stdout
    assert (run_dir / "board_probe").exists() == (mode == "live")
    assert not (run_dir / "native_probe.log").exists()
    assert not (run_dir / "logger.pid").exists()
    assert not (run_dir / "logger-stop-request.json").exists()
    assert "Shutdown result:" in log.read_text()
    assert all(path.read_bytes() == b"immutable" for path in recordings.glob("*/iis3dwb_acc.dat"))
