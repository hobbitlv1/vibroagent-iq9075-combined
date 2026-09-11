import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("safe_pipeline_stop", ROOT / "vibrodiag_mcp_prototype/scripts/safe_pipeline_stop.py")
stop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stop
SPEC.loader.exec_module(stop)


def receipt(pid=123, started=None, stopped=None, errors=None):
    return "Shutdown result: " + json.dumps({
        "pid": pid, "started": ["baseline"] if started is None else started,
        "stopped": ["baseline"] if stopped is None else stopped,
        "errors": [] if errors is None else errors,
    })


def test_native_receipt_required():
    assert "acknowledged" in stop.verify_shutdown(123, {"baseline"}, receipt(), False)
    with pytest.raises(stop.UnsafeStop):
        stop.verify_shutdown(123, {"baseline"}, "Stopped baseline -> folder\n", False)


@pytest.mark.parametrize("text", [
    receipt(pid=456), receipt(stopped=[]), receipt(errors=["native stop failed"]),
    receipt(started=["baseline", "baseline"]), receipt(stopped=["not_a_board"]),
    "Shutdown result: broken-json", "Shutdown result: []", receipt(started=[], stopped=[]),
    receipt() + "\n" + receipt(errors=["native failure"]),
    receipt(errors=["native failure"]) + "\n" + receipt(),
    receipt() + "\n" + receipt(pid=456),
    receipt() + "\nStarted baseline on device 0, serial reused",
])
def test_incomplete_or_unrelated_receipt_rejected(text):
    with pytest.raises(stop.UnsafeStop):
        stop.verify_shutdown(123, {"baseline"}, text, False)


def test_legacy_requires_explicit_option_and_warns():
    text = "Stopped baseline -> folder\n"
    assert "WARNING" in stop.verify_shutdown(123, {"baseline"}, text, True)
    with pytest.raises(stop.UnsafeStop):
        stop.verify_shutdown(123, {"baseline"}, text + "failed to close engine", True)
    with pytest.raises(stop.UnsafeStop):
        stop.verify_shutdown(123, {"baseline"}, text + receipt(errors=["native failure"]), True)


def test_partial_start_is_covered_by_receipt():
    text = receipt(started=["baseline", "target_1"], stopped=["baseline", "target_1"])
    assert "2/2" in stop.verify_shutdown(123, {"baseline"}, text, False)


def test_process_ownership_requires_exact_tokens_and_checkout(tmp_path):
    logger = str(tmp_path / "stdatalog_examples/vibroagent_two_vibrometer_logger.py")
    args = ("python3.12", logger, "--output-root", str(tmp_path / "stdatalog_examples"))
    proc = stop.Process(123, 1, "99", tmp_path, args)
    assert stop.process_role(proc, tmp_path) == "logger"
    assert stop.process_role(stop.Process(123, 1, "99", Path("/tmp/another-checkout"), args), tmp_path) is None
    assert stop.process_role(stop.Process(123, 1, "99", tmp_path, ("bash", "-c", " ".join(args))), tmp_path) is None
    assert stop.process_role(stop.Process(123, 1, "99", tmp_path, ("python", "-c", " ".join(args))), tmp_path) is None
    assert stop.process_role(stop.Process(123, 1, "99", tmp_path, ("python", "-c", "pass", *args[1:])), tmp_path) is None
    assert stop.process_role(stop.Process(123, 1, "99", tmp_path, ("python", "other.py", "-m", "vibroagent_mcp.webchat_server")), tmp_path) is None
    assert stop.process_role(stop.Process(123, 1, "99", tmp_path, ("python", "-u", "-m", "vibroagent_mcp.webchat_server")), tmp_path) == "webchat"


def test_reused_pid_handle_is_closed_without_signal(monkeypatch, tmp_path):
    proc = stop.Process(123, 1, "99", tmp_path, ("python", "test.py"))
    closed = []
    monkeypatch.setattr(stop.os, "pidfd_open", lambda pid: 88)
    monkeypatch.setattr(stop.os, "close", closed.append)
    monkeypatch.setattr(stop, "read_process", lambda pid: stop.Process(123, 1, "100", tmp_path, proc.argv))
    assert stop.open_handle(proc) is None
    assert closed == [88]


def test_shared_sdk_symlink_keeps_logger_ownership_scoped_to_gemma(tmp_path):
    shared = tmp_path / "stdatalog_examples"
    shared.mkdir()
    gemma = tmp_path / "VibroAgent-Gemma"
    gemma.mkdir()
    (gemma / "stdatalog_examples").symlink_to(shared, target_is_directory=True)
    args = ("python3.12", str(gemma / "stdatalog_examples/vibroagent_two_vibrometer_logger.py"),
            "--output-root", str(gemma / "stdatalog_examples"))
    proc = stop.Process(123, 1, "99", gemma, args)
    assert stop.process_role(proc, gemma) == "logger"
    assert stop.process_role(proc, tmp_path) is None
    wrong_output = stop.Process(123, 1, "99", gemma, (*args[:-1], str(tmp_path / "unrelated")))
    assert stop.process_role(wrong_output, gemma) is None


def test_descendants_do_not_include_siblings_or_process_group_peers(tmp_path):
    def p(pid, parent):
        return stop.Process(pid, parent, "99", tmp_path, ("python", "test.py"))
    root = p(10, 1)
    assert [p.pid for p in stop.descendants(root, [root, p(11, 10), p(12, 11), p(20, 1)])] == [11, 12]


def test_reused_service_root_cannot_target_replacement_children(tmp_path, monkeypatch):
    root = stop.Process(123, 1, "99", tmp_path, ("python", "-m", "vibroagent_mcp.webchat_server"))
    monkeypatch.setattr(stop, "open_handle", lambda proc: None)
    monkeypatch.setattr(stop, "snapshot", lambda: pytest.fail("must not follow reused root PID"))
    with pytest.raises(stop.UnsafeStop, match="reused PID"):
        stop.stop_service("webchat", [root], tmp_path, 1)


def test_service_root_exit_during_discovery_sends_no_child_signal(tmp_path, monkeypatch):
    root = stop.Process(123, 1, "99", tmp_path, ("python", "-m", "vibroagent_mcp.webchat_server"))
    child = stop.Process(124, 123, "99", tmp_path, ("python", "-c", "pass"))
    monkeypatch.setattr(stop, "open_handle", lambda proc: proc.pid)
    monkeypatch.setattr(stop, "snapshot", lambda: [root, child])
    monkeypatch.setattr(stop.select, "select", lambda *args: ([123], [], []))
    monkeypatch.setattr(stop, "signal_handle", lambda *args: pytest.fail("must not signal"))
    closed = []
    monkeypatch.setattr(stop.os, "close", closed.append)
    with pytest.raises(stop.UnsafeStop, match="no child was signalled"):
        stop.stop_service("webchat", [root], tmp_path, 1)
    assert closed == [123, 124]


def test_late_orphan_in_dedicated_unit_prevents_false_completion(tmp_path, monkeypatch):
    group = tmp_path / "vibroagent-live.service"
    group.mkdir()
    (group / "cgroup.procs").write_text("98765\n")
    monkeypatch.setattr(stop, "snapshot", lambda: [])
    monkeypatch.setattr(stop.os, "getppid", lambda: 1)
    monkeypatch.setattr(stop, "read_process", lambda pid: stop.Process(pid, 1, "99", tmp_path, ("python", "-c", "worker")))
    with pytest.raises(stop.UnsafeStop, match="98765"):
        stop.verify_finished(tmp_path, ["logger", "webchat", "geniex", "npu"], {group})


@pytest.mark.parametrize("mode", ["ok", "stall", "failed_ack"])
def test_mock_logger_shutdown_never_force_kills(tmp_path, mode, monkeypatch):
    """Only this test's temporary mock process receives real signals; no SDK imports."""
    (tmp_path / "stdatalog_examples").mkdir()
    (tmp_path / ".run").mkdir()
    (tmp_path / "vibroagent.sh").touch()
    logger = tmp_path / "stdatalog_examples/vibroagent_two_vibrometer_logger.py"
    logger.write_text("""import json, os, signal, time, sys
stopping = False
def request(sig, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGINT, signal.SIG_IGN if sys.argv[-1] == 'stall' else request)
print('Started baseline on device 0, serial mock', flush=True)
while not stopping: time.sleep(0.01)
failed = sys.argv[-1] == 'failed_ack'
print('Shutdown result: ' + json.dumps({'pid': os.getpid(), 'started': ['baseline'],
    'stopped': [] if failed else ['baseline'], 'errors': ['no acknowledgement'] if failed else []}), flush=True)
""")
    with (tmp_path / ".run/logger.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(logger), "--output-root", str(tmp_path / "stdatalog_examples"), mode], cwd=tmp_path, stdout=log)
    try:
        deadline = time.monotonic() + 5
        while "Started" not in (tmp_path / ".run/logger.log").read_text():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        (tmp_path / ".run/logger.pid").write_text(str(child.pid))
        proc = stop.read_process(child.pid)
        assert stop.process_role(proc, tmp_path) == "logger"
        stop.check_logger_start(tmp_path, [proc])
        stop.check_logger_start(tmp_path, [proc])
        if mode == "ok":
            stop.stop_logger(proc, tmp_path, 2, False)
            child.wait(timeout=2)
            assert not (tmp_path / ".run/logger.pid").exists()
        else:
            with pytest.raises(stop.UnsafeStop):
                stop.stop_logger(proc, tmp_path, 0.15 if mode == "stall" else 2, False)
            assert (tmp_path / ".run/logger.pid").exists()
            if mode == "stall":
                assert child.poll() is None
                with pytest.raises(stop.UnsafeStop):
                    stop.check_logger_start(tmp_path, [proc])
                monkeypatch.setattr(stop, "signal_handle", lambda *args: pytest.fail("retry must not send another signal"))
                with pytest.raises(stop.UnsafeStop):
                    stop.stop_logger(proc, tmp_path, 0.05, False)
                assert child.poll() is None
    finally:
        if child.poll() is None:
            child.kill()  # Test-created mock only; never a real pipeline process.
        child.wait(timeout=3)


def test_invalid_stale_pid_is_not_a_signal_target(tmp_path, monkeypatch):
    (tmp_path / "vibroagent.sh").touch()
    (tmp_path / ".run").mkdir()
    (tmp_path / ".run/logger.pid").write_text("-1")
    monkeypatch.setattr(stop, "snapshot", lambda: [])
    args = argparse.Namespace(repo=tmp_path, timeout=1, service_timeout=1, service="all", dry_run=False, allow_legacy=False)
    with pytest.raises(stop.UnsafeStop, match="Invalid logger PID"):
        stop.run(args)


@pytest.fixture
def retained_logger(tmp_path, monkeypatch):
    (tmp_path / "vibroagent.sh").touch()
    run_dir = tmp_path / ".run"
    run_dir.mkdir()
    monkeypatch.setattr(stop, "snapshot", lambda: [])
    monkeypatch.setattr(stop, "read_process", lambda pid: None)
    monkeypatch.setattr(stop, "signal_handle", lambda *args: pytest.fail("retained evidence must not signal"))
    log = run_dir / "logger.log"
    prefix = "Started baseline on device 0, serial mock\n"
    log.write_text(prefix)
    intent = {"pid": 123, "started": "99", "inode": log.stat().st_ino, "offset": len(prefix.encode())}
    (run_dir / "logger-stop-request.json").write_text(json.dumps(intent))
    args = argparse.Namespace(repo=tmp_path, timeout=1, service_timeout=1,
                              service="logger", dry_run=False, allow_legacy=False, check_start=False)
    return run_dir, log, intent, args


@pytest.mark.parametrize("check_start", [False, True])
@pytest.mark.parametrize("keep_intent", [False, True])
def test_missing_pid_preserves_failed_shutdown(retained_logger, check_start, keep_intent):
    run_dir, log, intent, args = retained_logger
    intent_path = run_dir / "logger-stop-request.json"
    if not keep_intent:
        intent_path.unlink()
    with log.open("a") as stream:
        stream.write(receipt(errors=["native failure"]))
    args.check_start = check_start
    for _ in range(2):
        with pytest.raises(stop.UnsafeStop):
            stop.run(args)
        assert "native failure" in log.read_text()
        assert intent_path.exists() == keep_intent
        if keep_intent:
            assert json.loads(intent_path.read_text()) == intent


@pytest.mark.parametrize("check_start", [False, True])
@pytest.mark.parametrize("keep_pid", [False, True])
def test_valid_receipt_resolves_retained_shutdown(retained_logger, check_start, keep_pid):
    run_dir, log, intent, args = retained_logger
    if keep_pid:
        (run_dir / "logger.pid").write_text(str(intent["pid"]))
    with log.open("a") as stream:
        stream.write(receipt())
    args.check_start = check_start
    assert stop.run(args) == 0
    assert not (run_dir / "logger-stop-request.json").exists()
    assert not (run_dir / "logger.pid").exists()
    args.check_start = True
    assert stop.run(args) == 0
    assert stop.run(args) == 0


@pytest.mark.parametrize("damage", [
    "no_receipt", "malformed_intent", "invalid_offset", "truncated_log", "replaced_log",
    "wrong_pid", "conflicting_pidfile", "old_receipt", "reused_pid",
])
def test_unverifiable_shutdown_cannot_be_cleared(retained_logger, monkeypatch, damage):
    run_dir, log, intent, args = retained_logger
    intent_path = run_dir / "logger-stop-request.json"
    if damage != "no_receipt":
        with log.open("a") as stream:
            stream.write(receipt(pid=456 if damage == "wrong_pid" else 123))
    if damage == "malformed_intent":
        intent_path.write_text("[]")
    elif damage == "invalid_offset":
        intent["offset"] = -1
        intent_path.write_text(json.dumps(intent))
    elif damage == "truncated_log":
        log.write_text("")
    elif damage == "replaced_log":
        replacement = run_dir / "replacement.log"
        replacement.write_text(log.read_text())
        replacement.replace(log)
    elif damage == "conflicting_pidfile":
        (run_dir / "logger.pid").write_text("456")
    elif damage == "old_receipt":
        intent["offset"] = log.stat().st_size
        intent_path.write_text(json.dumps(intent))
    elif damage == "reused_pid":
        monkeypatch.setattr(stop, "read_process", lambda pid: stop.Process(pid, 1, "100", args.repo, ("python", "other.py")))
    original_log, original_intent = log.read_bytes(), intent_path.read_bytes()
    for args.check_start in (False, True):
        with pytest.raises(stop.UnsafeStop):
            stop.run(args)
        assert log.read_bytes() == original_log
        assert intent_path.read_bytes() == original_intent


def test_start_check_uses_shutdown_lock_without_signalling(retained_logger):
    run_dir, log, intent, args = retained_logger
    args.check_start = True
    with (run_dir / "safe-stop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(stop.UnsafeStop, match="Another shutdown"):
            stop.run(args)


def test_dry_run_preserves_pending_shutdown_without_signalling(retained_logger):
    run_dir, log, intent, args = retained_logger
    args.dry_run = True
    assert stop.run(args) == 0
    assert json.loads((run_dir / "logger-stop-request.json").read_text()) == intent
    assert "Started baseline" in log.read_text()


def test_clean_start_check_is_repeatable(retained_logger):
    run_dir, log, intent, args = retained_logger
    log.unlink()
    (run_dir / "logger-stop-request.json").unlink()
    args.check_start = True
    assert stop.run(args) == 0
    assert stop.run(args) == 0


def test_restart_does_not_start_after_failed_stop():
    script = f'''source {ROOT / "vibroagent.sh"}
do_stop() {{ return 7; }}
do_start() {{ echo BAD_RESTART; }}
sleep() {{ :; }}
do_restart
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 7
    assert "BAD_RESTART" not in result.stdout


@pytest.fixture
def combined_checkout(tmp_path):
    """Real stop entrypoints, no application venv and no acquisition SDK."""
    repo = tmp_path / "combined checkout"
    for relative in (Path("."), Path("VibroAgent-Gemma")):
        stack = repo / relative
        helper = Path("vibrodiag_mcp_prototype/scripts/safe_pipeline_stop.py")
        (stack / helper.parent).mkdir(parents=True)
        (stack / ".run").mkdir()
        shutil.copy2(ROOT / relative / "vibroagent.sh", stack / "vibroagent.sh")
        shutil.copy2(ROOT / relative / helper, stack / helper)
    (repo / "linux_setup").mkdir()
    shutil.copy2(ROOT / "linux_setup/vibroagent-safe-stop.sh", repo / "linux_setup/vibroagent-safe-stop.sh")
    return repo


@pytest.fixture
def owned_webchat():
    children = []

    def launch(stack):
        package = stack / "vibrodiag_mcp_prototype/vibroagent_mcp"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
        (package / "webchat_server.py").write_text(
            "import time\nprint('ready', flush=True)\ntime.sleep(120)\n"
        )
        child = subprocess.Popen(
            [sys.executable, "-m", "vibroagent_mcp.webchat_server"],
            cwd=package.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        children.append(child)
        assert child.stdout.readline().strip() == "ready"
        return child

    yield launch
    for child in children:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize("relative", [".", "VibroAgent-Gemma"])
def test_safe_stop_dry_run_without_application_runtime(combined_checkout, relative):
    stack = combined_checkout / relative
    result = subprocess.run(
        [str(stack / "vibroagent.sh"), "stop", "--dry-run"],
        env={**os.environ, "VIBROAGENT_PYTHON": "/missing/app/python", "VIBROGEMMA_PYTHON": "/missing/model/python"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout[result.stdout.index("{"):]) == {
        "logger": [], "webchat": [], "geniex": [], "npu": [],
    }


@pytest.mark.parametrize("relative", [".", "VibroAgent-Gemma"])
@pytest.mark.parametrize("command", ["stop", "restart"])
def test_default_commands_preserve_services_after_unconfirmed_board_stop(
    combined_checkout, owned_webchat, relative, command,
):
    stack = combined_checkout / relative
    webchat = owned_webchat(stack)
    pidfile = stack / ".run/logger.pid"
    logfile = stack / ".run/logger.log"
    pidfile.write_text("2000000000")
    failed_receipt = receipt(pid=2000000000, stopped=[], errors=["native stop failed"])
    logfile.write_text(failed_receipt)

    result = subprocess.run(
        [str(stack / "vibroagent.sh"), command],
        capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert webchat.poll() is None
    assert pidfile.read_text() == "2000000000"
    assert logfile.read_text() == failed_receipt


def test_combined_stop_dry_run_then_only_owned_stacks(combined_checkout, owned_webchat, tmp_path):
    root = owned_webchat(combined_checkout)
    gemma = owned_webchat(combined_checkout / "VibroAgent-Gemma")
    unrelated = owned_webchat(tmp_path / "other-checkout")
    command = ["bash", str(combined_checkout / "linux_setup/vibroagent-safe-stop.sh")]
    preview = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, timeout=10)
    assert preview.returncode == 0, preview.stderr
    snapshots = [json.loads(block) for block in re.findall(r"(?ms)^\{.*?^\}", preview.stdout)]
    assert [snapshot["webchat"] for snapshot in snapshots] == [[root.pid], [gemma.pid]]
    assert root.poll() is None and gemma.poll() is None and unrelated.poll() is None
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert root.wait(timeout=3) == -signal.SIGTERM
    assert gemma.wait(timeout=3) == -signal.SIGTERM
    assert unrelated.poll() is None


@pytest.mark.parametrize("busy_stack", [".", "VibroAgent-Gemma"])
def test_combined_stop_acquires_both_control_locks_before_signalling(combined_checkout, owned_webchat, busy_stack):
    root = owned_webchat(combined_checkout)
    gemma = owned_webchat(combined_checkout / "VibroAgent-Gemma")
    with (combined_checkout / busy_stack / ".run/control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(combined_checkout / "linux_setup/vibroagent-safe-stop.sh")],
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 75
    assert root.poll() is None and gemma.poll() is None
    for relative in (".", "VibroAgent-Gemma"):
        with (combined_checkout / relative / ".run/control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_combined_incomplete_stop_preserves_other_stack_on_retry(combined_checkout, owned_webchat):
    gemma = owned_webchat(combined_checkout / "VibroAgent-Gemma")
    pidfile = combined_checkout / ".run/logger.pid"
    logfile = combined_checkout / ".run/logger.log"
    pidfile.write_text("123")
    logfile.write_text(receipt(errors=["native failure"]))
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(combined_checkout / "linux_setup/vibroagent-safe-stop.sh")],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2
        assert gemma.poll() is None
        assert pidfile.read_text() == "123"
        assert logfile.read_text() == receipt(errors=["native failure"])
