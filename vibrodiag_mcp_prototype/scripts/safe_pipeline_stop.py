#!/usr/bin/env python3
"""Stop only this checkout's processes; never escalate a stuck USB logger."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import sys
import time
from dataclasses import dataclass

ROLES = {"baseline", *(f"target_{i}" for i in range(1, 6))}
MODULES = {
    "webchat": {"vibroagent_mcp.vibrogemma_webchat", "vibroagent_mcp.webchat_server"},
    "geniex": {"vibroagent_mcp.vibrogemma_geniex_server", "vibroagent_mcp.geniex_openai_server"},
    "npu": {"vibroagent_mcp.genie_openai_server"},
}
STARTED = re.compile(r"^Started (baseline|target_[1-5]) on device ", re.M)
STOPPED = re.compile(r"^Stopped (baseline|target_[1-5]) -> ", re.M)
BAD_STOP = re.compile(r"Traceback|double free|corruption|invalid pointer|failed to (stop|close)|not acknowledged", re.I)


class UnsafeStop(RuntimeError):
    pass


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    started: str
    cwd: Path
    argv: tuple[str, ...]


def read_process(pid: int) -> Process | None:
    try:
        folder = Path("/proc") / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        argv = tuple(os.fsdecode(v) for v in (folder / "cmdline").read_bytes().split(b"\0") if v)
        if not argv:
            return None
        return Process(pid, int(fields[1]), fields[19], (folder / "cwd").resolve(strict=True), argv)
    except (OSError, ValueError, IndexError):
        return None


def process_role(proc: Process, repo: Path) -> str | None:
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(proc.argv[0]).name):
        return None
    if proc.cwd not in {repo, repo / "vibrodiag_mcp_prototype"}:
        return None
    args = proc.argv
    entry = 1
    # Only interpreter flags used by our launchers (plus harmless standard
    # single-token flags). Never mistake arguments to -c/another script for an entrypoint.
    while entry < len(args) and args[entry] in {"-u", "-B", "-s", "-S", "-E", "-I", "-O", "-OO", "-q", "-P"}:
        entry += 1
    if entry < len(args) and args[entry] == "--":
        entry += 1
    if entry >= len(args):
        return None
    logger = str(repo / "stdatalog_examples/vibroagent_two_vibrometer_logger.py")
    if args[entry] == logger and "--output-root" in args[entry + 1:]:
        index = args.index("--output-root", entry + 1) + 1
        if index < len(args) and Path(args[index]).resolve() == (repo / "stdatalog_examples").resolve():
            return "logger"
    if args[entry] == "-m":
        index = entry + 1
        if index < len(args):
            return next((role for role, modules in MODULES.items() if args[index] in modules), None)
    return None


def snapshot() -> list[Process]:
    return [p for item in Path("/proc").iterdir() if item.name.isdecimal()
            if (p := read_process(int(item.name))) is not None]


def open_handle(proc: Process) -> int | None:
    """A pidfd pins process identity, including between validation and signalling."""
    try:
        fd = os.pidfd_open(proc.pid)
    except ProcessLookupError:
        return None
    current = read_process(proc.pid)
    if current is None or current.started != proc.started or current.argv != proc.argv:
        os.close(fd)
        return None
    return fd


def signal_handle(fd: int, sig: int) -> None:
    try:
        signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass


def wait_exit(fd: int, timeout: float, progress=None) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if progress:
            progress()
        remaining = max(0.0, deadline - time.monotonic())
        if select.select([fd], [], [], min(1.0, remaining))[0]:
            if progress:
                progress()
            return True
        if remaining <= 0:
            return False


def shutdown_receipts(log: str) -> list[dict]:
    receipts = []
    for line in log.splitlines():
        if line.startswith("Shutdown result:"):
            try:
                value = json.loads(line.removeprefix("Shutdown result:").strip())
            except json.JSONDecodeError as exc:
                raise UnsafeStop("Malformed logger shutdown receipt") from exc
            if (not isinstance(value, dict) or type(value.get("pid")) is not int
                    or value["pid"] <= 1):
                raise UnsafeStop("Malformed logger shutdown receipt")
            receipts.append(value)
        elif receipts and STARTED.match(line):
            raise UnsafeStop("Logger restarted after the retained shutdown receipt")
    if len(receipts) > 1:
        raise UnsafeStop("Multiple logger shutdown receipts; evidence is ambiguous")
    return receipts


def verify_shutdown(pid: int, started: set[str], new_log: str, allow_legacy: bool) -> str:
    started |= set(STARTED.findall(new_log))
    receipts = shutdown_receipts(new_log)
    if receipts:
        receipt = receipts[0]
        if receipt["pid"] != pid:
            raise UnsafeStop("Logger shutdown receipt belongs to another PID")
        began, stopped, errors = (receipt.get(k) for k in ("started", "stopped", "errors"))
        if not all(isinstance(v, list) for v in (began, stopped, errors)):
            raise UnsafeStop("Incomplete logger shutdown receipt")
        if (any(not isinstance(v, str) or v not in ROLES for v in began + stopped)
                or len(set(began)) != len(began) or len(set(stopped)) != len(stopped)
                or not started.issubset(set(began)) or set(began) != set(stopped) or errors
                or BAD_STOP.search(new_log)):
            raise UnsafeStop(f"Board shutdown not confirmed: {receipt}")
        return f"Native stop acknowledged for {len(stopped)}/{len(began)} board(s); logger exited."
    stopped = set(STOPPED.findall(new_log))
    if allow_legacy and started and stopped == started and not BAD_STOP.search(new_log):
        return (f"Legacy logger exited and reported {len(stopped)}/{len(started)} stops. "
                "WARNING: this older process has no native-acknowledgement receipt.")
    raise UnsafeStop("Logger exited without a complete shutdown receipt; board state is unconfirmed. "
                     "Do not automatically restart or unplug it.")


def logger_pidfile(repo: Path) -> int | None:
    path = repo / ".run/logger.pid"
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text.isdecimal() or int(text) <= 1:
        raise UnsafeStop("Invalid logger PID file; no processes signalled")
    return int(text)


def logger_intent(repo: Path) -> dict | None:
    path = repo / ".run/logger-stop-request.json"
    if not path.exists():
        return None
    try:
        intent = json.loads(path.read_text())
    except (ValueError, UnicodeError) as exc:
        raise UnsafeStop("Malformed logger stop request") from exc
    if (not isinstance(intent, dict)
            or type(intent.get("pid")) is not int or intent["pid"] <= 1
            or not isinstance(intent.get("started"), str) or not intent["started"].isdecimal()
            or type(intent.get("inode")) is not int or intent["inode"] <= 0
            or type(intent.get("offset")) is not int or intent["offset"] < 0):
        raise UnsafeStop("Malformed logger stop request")
    return intent


def validate_intent_log(intent: dict, stat: os.stat_result) -> None:
    if intent["inode"] != stat.st_ino or intent["offset"] > stat.st_size:
        raise UnsafeStop("Log changed since stop request; shutdown evidence is unconfirmed")


def resolve_exited_logger(repo: Path, allow_legacy: bool) -> None:
    """Retained intent and log remain authoritative even when logger.pid is missing."""
    pid = logger_pidfile(repo)
    intent = logger_intent(repo)
    log_path = repo / ".run/logger.log"
    if not log_path.exists():
        if pid is not None or intent is not None:
            raise UnsafeStop("Missing logger log; shutdown evidence is unconfirmed")
        print("No owned logger is running.", flush=True)
        return
    with log_path.open("rb") as log:
        stat = os.fstat(log.fileno())
        data = log.read()
    text = data.decode("utf-8", errors="replace")
    receipts = shutdown_receipts(text)
    if pid is None and intent is None and not text.strip():
        print("No owned logger is running.", flush=True)
        return
    if intent is not None:
        if pid is not None and pid != intent["pid"]:
            raise UnsafeStop("Logger PID conflicts with retained stop request")
        pid = intent["pid"]
        validate_intent_log(intent, stat)
        if shutdown_receipts(data[:intent["offset"]].decode("utf-8", errors="replace")):
            raise UnsafeStop("Shutdown receipt predates the retained stop request")
    if pid is None and receipts:
        pid = receipts[0]["pid"]
    if pid is None:
        raise UnsafeStop("Retained logger log has no PID-bound shutdown receipt")
    if read_process(pid) is not None:
        raise UnsafeStop("Retained logger PID is still running or reused; shutdown evidence is ambiguous")
    current_log = text if intent is None else data[intent["offset"]:].decode("utf-8", errors="replace")
    message = verify_shutdown(pid, set(STARTED.findall(text)), current_log, allow_legacy)
    if log_path.stat().st_ino != stat.st_ino or log_path.stat().st_size != len(data):
        raise UnsafeStop("Logger log changed while verifying shutdown")
    print(message, flush=True)
    remove_pidfile(repo / ".run/logger.pid", pid)
    (repo / ".run/logger-stop-request.json").unlink(missing_ok=True)


def check_logger_start(repo: Path, loggers: list[Process]) -> None:
    if len(loggers) > 1:
        raise UnsafeStop("Multiple owned loggers found; startup is ambiguous")
    if not loggers:
        resolve_exited_logger(repo, False)
        return
    proc = loggers[0]
    pid = logger_pidfile(repo)
    if logger_intent(repo) is not None:
        raise UnsafeStop("Logger shutdown is pending; retry safe stop before starting")
    if pid is not None and pid != proc.pid:
        raise UnsafeStop("Logger PID conflicts with running logger")
    text = (repo / ".run/logger.log").read_text(errors="replace")
    if (shutdown_receipts(text) or BAD_STOP.search(text)
            or "Stopping logging..." in text or STOPPED.search(text)):
        raise UnsafeStop("Running logger has shutdown evidence; not a healthy startup")
    current = read_process(proc.pid)
    if current != proc:
        raise UnsafeStop("Logger changed/exited during startup check")
    print(f"Owned logger {proc.pid} is running without a pending shutdown.", flush=True)


def remove_pidfile(path: Path, pid: int) -> None:
    if path.exists() and path.read_text().strip() == str(pid):
        path.unlink()


def stop_logger(proc: Process, repo: Path, timeout: float, allow_legacy: bool) -> None:
    log_path = repo / ".run/logger.log"
    with log_path.open(errors="replace") as log:
        previous_log = log.read()
        started = set(STARTED.findall(previous_log))
        chunks = []

        def progress():
            chunk = log.read()
            chunks.append(chunk)
            for line in chunk.splitlines():
                if line.startswith(("Stopped ", "Shutdown result: ", "Warning: failed")):
                    print(line, flush=True)

        fd = open_handle(proc)
        if fd is None:
            raise UnsafeStop("Logger changed/exited before shutdown; not signalling another process.")
        try:
            intent_path = repo / ".run/logger-stop-request.json"
            intent = logger_intent(repo)
            pid = logger_pidfile(repo)
            if pid is not None and pid != proc.pid:
                raise UnsafeStop("Logger PID conflicts with running logger; no signal sent")
            if intent is not None:
                if intent["pid"] != proc.pid or intent["started"] != proc.started:
                    raise UnsafeStop("Retained stop request belongs to another logger; no signal sent")
                validate_intent_log(intent, os.fstat(log.fileno()))
                log.seek(0)
                if shutdown_receipts(log.buffer.read(intent["offset"]).decode("utf-8", errors="replace")):
                    raise UnsafeStop("Shutdown receipt predates the retained stop request")
                log.seek(intent["offset"])
                print(f"Waiting for previously requested shutdown of logger {proc.pid}; no new signal.", flush=True)
            else:
                if shutdown_receipts(previous_log):
                    raise UnsafeStop("Running logger has retained shutdown receipt; no new signal sent")
                intent = {"pid": proc.pid, "started": proc.started,
                          "inode": os.fstat(log.fileno()).st_ino, "offset": log.tell()}
                temporary = intent_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(intent))
                temporary.replace(intent_path)
                print(f"Requesting graceful stop from logger {proc.pid} ({len(started)} started boards).", flush=True)
                # Persist before signalling: retries after timeout/interruption must
                # never send a second SIGINT into an older logger's cleanup.
                signal_handle(fd, signal.SIGINT)
            if not wait_exit(fd, timeout, progress):
                raise UnsafeStop(f"Logger still running after {timeout:g}s. Left it alive; "
                                 "no SIGTERM/SIGKILL or hub reset was attempted. Check logger.log and retry later.")
        finally:
            os.close(fd)
        if log_path.stat().st_ino != os.fstat(log.fileno()).st_ino:
            raise UnsafeStop("Logger log was replaced during shutdown; refusing to accept unrelated evidence.")
        print(verify_shutdown(proc.pid, started, "".join(chunks), allow_legacy), flush=True)
        remove_pidfile(repo / ".run/logger.pid", proc.pid)
        intent_path.unlink(missing_ok=True)


def descendants(root: Process, processes: list[Process]) -> list[Process]:
    ids = {root.pid}
    while True:
        children = {p.pid for p in processes if p.ppid in ids}
        if children.issubset(ids):
            return [p for p in processes if p.pid in ids and p.pid != root.pid]
        ids |= children


def stop_service(role: str, roots: list[Process], repo: Path, timeout: float) -> None:
    for root in roots:
        handles = []
        try:
            root_fd = open_handle(root)
            if root_fd is None:
                raise UnsafeStop(f"{role} root {root.pid} changed/exited; refusing to follow a possibly reused PID. Retry status.")
            handles.append((root.pid, root_fd))
            current = snapshot()
            for proc in descendants(root, current):
                if process_role(proc, repo) == "logger":
                    raise UnsafeStop("Refusing to include a USB logger in another service's cleanup.")
                fd = open_handle(proc)
                if fd is not None:
                    handles.append((proc.pid, fd))
            if select.select([root_fd], [], [], 0)[0]:
                raise UnsafeStop(f"{role} root exited during child discovery; no child was signalled.")
            print(f"Stopping owned {role} process {root.pid} and {max(0, len(handles)-1)} child process(es).", flush=True)
            for _, fd in handles:
                signal_handle(fd, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            remaining = [pid for pid, fd in handles if not wait_exit(fd, max(0.0, deadline-time.monotonic()))]
            if remaining:
                raise UnsafeStop(f"{role} still running: {remaining}. No forced termination; restart blocked.")
        finally:
            for _, fd in handles:
                os.close(fd)
        remove_pidfile(repo / f".run/{role}.pid", root.pid)


def pipeline_cgroups(processes: list[Process]) -> set[Path]:
    """Only the dedicated live unit, never the caller's/shared desktop cgroup."""
    groups = set()
    for proc in processes:
        try:
            lines = (Path("/proc") / str(proc.pid) / "cgroup").read_text().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if line.startswith("0::"):
                path = Path(line[3:])
                if path.is_absolute() and ".." not in path.parts and path.name == "vibroagent-live.service":
                    groups.add(Path("/sys/fs/cgroup") / str(path).lstrip("/"))
    return groups


def verify_finished(repo: Path, selected: list[str], groups: set[Path]) -> None:
    # During ExecStop our helper, shell, and flock supervisor are also in the
    # unit. Exclude only the caller's actual ancestor chain, not other children.
    controls = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in controls:
        controls.add(parent)
        proc = read_process(parent)
        if proc is None:
            break
        parent = proc.ppid
    remaining = {p.pid for p in snapshot() if process_role(p, repo) in selected}
    for group in groups:
        try:
            members = (group / "cgroup.procs").read_text().split()
        except FileNotFoundError:
            continue
        remaining |= {int(pid) for pid in members if int(pid) not in controls and read_process(int(pid)) is not None}
    if remaining:
        raise UnsafeStop(f"Pipeline processes remain: {sorted(remaining)}. No completion claimed or forced termination.")


def run(args) -> int:
    repo = args.repo.resolve(strict=True)
    if not (repo / "vibroagent.sh").is_file():
        raise UnsafeStop("Not a VibroAgent checkout")
    for timeout in (args.timeout, args.service_timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise UnsafeStop("Timeouts must be finite and positive")
    check_start = getattr(args, "check_start", False)
    if check_start and (args.service != "logger" or args.dry_run or args.allow_legacy):
        raise UnsafeStop("--check-start requires --service logger without --dry-run or --allow-legacy")
    if not check_start and (not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
        raise UnsafeStop("Safe shutdown requires Linux pidfd support; refusing PID-only signals")
    run_dir = repo / ".run"
    with (run_dir / "safe-stop.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsafeStop("Another shutdown is in progress; no second interrupt sent") from exc
        groups = {role: [] for role in ("logger", *MODULES)}
        for proc in snapshot():
            role = process_role(proc, repo)
            if role:
                groups[role].append(proc)
        selected = list(groups) if args.service == "all" else [args.service]
        if check_start:
            check_logger_start(repo, groups["logger"])
            print("Logger startup check complete; no signals sent.", flush=True)
            return 0
        if args.dry_run:
            print(json.dumps({role: [p.pid for p in groups[role]] for role in selected}, indent=2))
            return 0
        dedicated_groups = pipeline_cgroups([p for values in groups.values() for p in values]) if args.service == "all" else set()
        if "logger" in selected:
            if len(groups["logger"]) > 1:
                raise UnsafeStop("Multiple owned loggers found; stop evidence is ambiguous. No signals sent.")
            if groups["logger"]:
                stop_logger(groups["logger"][0], repo, args.timeout, args.allow_legacy)
            else:
                resolve_exited_logger(repo, args.allow_legacy)
        for role in selected:
            if role != "logger":
                stop_service(role, groups[role], repo, args.service_timeout)
        verify_finished(repo, selected, dedicated_groups)
        print("Safe stop complete. No force-kill or USB power/reset operation used.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--service", choices=["all", "logger", *MODULES], default="all")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--service-timeout", type=float, default=25)
    parser.add_argument("--allow-legacy", action="store_true", help="one-time migration: accept legacy stop reports, with warning")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-start", action="store_true",
                        help="signal-free logger startup gate; verify retained shutdown evidence")
    args = parser.parse_args()
    try:
        return run(args)
    except (UnsafeStop, OSError, ValueError) as exc:
        print(f"STOP INCOMPLETE: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("Stop checker interrupted. No extra signal sent to logger; verify cleanup before restarting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
