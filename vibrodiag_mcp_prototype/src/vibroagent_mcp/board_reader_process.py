"""Per-board live SDK reader processes.

STDATALOG-PYSDK/HSDatalog keeps enough mutable process-local decoder state that
reading several boards from one Python process tends to serialize or cross-talk.
This module runs one long-lived worker process per physical/logical board.  Each
worker owns exactly one persistent SDK decoder and serves latest-window requests
through multiprocessing queues.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sdk_vibrometer import LiveSdkVibrometerDataRequiredError, SdkVibrometerWindow

DEFAULT_BOARD_READER_TIMEOUT_S = 30.0
DEFAULT_BOARD_READER_STARTUP_GRACE_S = 45.0
DEFAULT_BOARD_READER_START_METHOD = "spawn"
BOARD_READER_CHILD_ENV = "VIBRO_BOARD_READER_CHILD"
BOARD_READER_PROCESSES_ENV = "VIBRO_BOARD_READER_PROCESSES"
BOARD_READER_TIMEOUT_ENV = "VIBRO_BOARD_READER_TIMEOUT_S"
BOARD_READER_STARTUP_GRACE_ENV = "VIBRO_BOARD_READER_STARTUP_GRACE_S"
BOARD_READER_START_METHOD_ENV = "VIBRO_BOARD_READER_START_METHOD"


class BoardReaderProcessError(RuntimeError):
    """Raised when a per-board reader process fails or times out."""


@dataclass(frozen=True)
class BoardReaderStatus:
    key: tuple[str, str]
    pid: int | None
    alive: bool
    started_at_s: float | None
    request_count: int
    last_response_at_s: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "pid": self.pid,
            "alive": self.alive,
            "started_at_s": self.started_at_s,
            "request_count": self.request_count,
            "last_response_at_s": self.last_response_at_s,
        }


class _BoardReaderClient:
    def __init__(self, *, key: tuple[str, str], ctx: mp.context.BaseContext):
        self.key = key
        self.ctx = ctx
        self.request_q: Any | None = None
        self.response_q: Any | None = None
        self.process: mp.Process | None = None
        self.started_at_s: float | None = None
        self.started_mono_s: float | None = None
        self.request_count = 0
        self.last_response_at_s: float | None = None
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()

    def ensure_started(self) -> None:
        with self._lock:
            if self.process is not None and self.process.is_alive():
                return
            self._close_locked(terminate=False)
            self.request_q = self.ctx.Queue(maxsize=8)
            self.response_q = self.ctx.Queue(maxsize=8)
            self.process = self.ctx.Process(
                target=_board_reader_worker_main,
                args=(self.key, self.request_q, self.response_q),
                daemon=True,
            )
            self.process.start()
            self.started_at_s = time.time()
            self.started_mono_s = time.monotonic()

    def status(self) -> BoardReaderStatus:
        process = self.process
        return BoardReaderStatus(
            key=self.key,
            pid=process.pid if process is not None else None,
            alive=bool(process and process.is_alive()),
            started_at_s=self.started_at_s,
            request_count=self.request_count,
            last_response_at_s=self.last_response_at_s,
        )

    def read_window(self, kwargs: dict[str, Any], *, timeout_s: float | None = None) -> SdkVibrometerWindow:
        timeout = _reader_timeout_s(timeout_s, duration_s=_safe_float(kwargs.get("duration_s")))
        with self._request_lock:
            # ensure_started/restart both run under this lock, so the queues can
            # no longer be nulled between the liveness check and the put below.
            self.ensure_started()
            if self.request_q is None or self.response_q is None:
                raise BoardReaderProcessError("board reader process queues are not initialized")
            grace = _startup_grace_s()
            started_mono = self.started_mono_s

            def _worker_still_starting() -> bool:
                # A fresh worker pays a one-time cold open of the whole
                # acquisition before it can answer. Killing it mid-open
                # restarts that open from zero, so under steady polling the
                # board can never recover; leave young workers alone.
                return started_mono is not None and (time.monotonic() - started_mono) < grace

            request_id = uuid.uuid4().hex
            request = {"command": "read_window", "request_id": request_id, "kwargs": dict(kwargs)}
            try:
                self.request_q.put(request, timeout=timeout)
            except queue.Full as exc:
                if not _worker_still_starting():
                    self.restart()
                raise BoardReaderProcessError(
                    f"board reader {self.key[0]} request queue is full"
                ) from exc

            wait_started = time.monotonic()
            deadline = wait_started + timeout
            if started_mono is not None:
                deadline = max(deadline, started_mono + grace)
            deferred: list[dict[str, Any]] = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not _worker_still_starting():
                        self.restart()
                    raise BoardReaderProcessError(
                        f"board reader {self.key[0]} timed out after {time.monotonic() - wait_started:.1f}s"
                    )
                try:
                    response = self.response_q.get(timeout=remaining)
                except queue.Empty:
                    if not _worker_still_starting():
                        self.restart()
                    raise BoardReaderProcessError(
                        f"board reader {self.key[0]} timed out after {time.monotonic() - wait_started:.1f}s"
                    ) from None

                if not isinstance(response, dict):
                    continue
                if response.get("request_id") != request_id:
                    # This should not happen because requests per client are
                    # serialized, but keep it robust if an old response is left
                    # in the queue after a restart race.
                    deferred.append(response)
                    continue

                self.request_count += 1
                self.last_response_at_s = time.time()
                if response.get("ok") is True:
                    window = response.get("window")
                    if isinstance(window, SdkVibrometerWindow):
                        return window
                    raise BoardReaderProcessError(
                        f"board reader {self.key[0]} returned an invalid window payload"
                    )

                exc_type = str(response.get("exc_type") or "")
                metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
                if exc_type == "LiveSdkVibrometerDataRequiredError":
                    raise LiveSdkVibrometerDataRequiredError(metadata)
                message = str(response.get("message") or "board reader failed")
                raise BoardReaderProcessError(message)

    def restart(self) -> None:
        with self._lock:
            self._close_locked(terminate=True)

    def shutdown(self) -> None:
        with self._lock:
            if self.process is not None and self.process.is_alive() and self.request_q is not None:
                try:
                    self.request_q.put_nowait({"command": "shutdown", "request_id": uuid.uuid4().hex})
                except Exception:
                    pass
            self._close_locked(terminate=True)

    def _close_locked(self, *, terminate: bool) -> None:
        process = self.process
        if process is not None:
            if terminate and process.is_alive():
                process.terminate()
            if process.is_alive():
                process.join(timeout=1.0)
            else:
                process.join(timeout=0.1)
            if terminate and process.is_alive():
                process.kill()
                process.join(timeout=0.5)
        for q in (self.request_q, self.response_q):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass
        self.process = None
        self.request_q = None
        self.response_q = None
        self.started_at_s = None
        self.started_mono_s = None


_POOL_LOCK = threading.Lock()
_POOL: dict[tuple[str, str], _BoardReaderClient] = {}
_CONTEXT: mp.context.BaseContext | None = None


def board_reader_processes_enabled(default: bool = False) -> bool:
    """Return whether per-board reader processes should be used."""
    if os.environ.get(BOARD_READER_CHILD_ENV) == "1":
        return False
    raw = os.environ.get(BOARD_READER_PROCESSES_ENV)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def read_sdk_vibrometer_window_via_board_process(
    *,
    machine_id: str = "stwinbox_iis3dwb",
    acquisition_folder: str | os.PathLike[str] | None = None,
    sensor_name: str | None = None,
    axis: str = "norm",
    start_time_s: float | None = None,
    duration_s: float = 5.0,
    require_current: bool = True,
    timeout_s: float | None = None,
    max_duration_s: float = 60.0,
) -> SdkVibrometerWindow:
    """Read a vibrometer window through the long-lived process for this board."""
    key = _board_key(acquisition_folder=acquisition_folder, sensor_name=sensor_name, machine_id=machine_id)
    client = _get_client(key)
    return client.read_window(
        {
            "machine_id": machine_id,
            "acquisition_folder": str(acquisition_folder) if acquisition_folder is not None else None,
            "sensor_name": sensor_name,
            "axis": axis,
            "start_time_s": start_time_s,
            "duration_s": duration_s,
            "require_current": require_current,
            "max_duration_s": max_duration_s,
        },
        timeout_s=timeout_s,
    )


def prewarm_board_reader_processes(
    sensors: list[dict[str, Any]],
    *,
    default_sensor_name: str = "iis3dwb_acc",
) -> list[dict[str, Any]]:
    """Start one worker process for every configured board.

    This intentionally does not block on an SDK read.  Starting all processes up
    front removes the launch-order penalty; the first real reads can then run in
    parallel from the graph or agent monitor.
    """
    statuses: list[dict[str, Any]] = []
    for sensor in sensors:
        machine_id = str(sensor.get("sensor_id") or sensor.get("machine_id") or "sensor")
        acquisition_folder = sensor.get("acquisition_folder")
        sensor_name = str(sensor.get("hsd_sensor_name") or sensor.get("sensor_name") or default_sensor_name)
        key = _board_key(acquisition_folder=acquisition_folder, sensor_name=sensor_name, machine_id=machine_id)
        client = _get_client(key)
        try:
            client.ensure_started()
            status = client.status().as_dict()
            status.update({"sensor_id": machine_id, "ok": True})
        except Exception as exc:
            status = {
                "sensor_id": machine_id,
                "key": list(key),
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        statuses.append(status)
    return statuses


def board_reader_process_statuses() -> list[dict[str, Any]]:
    with _POOL_LOCK:
        clients = list(_POOL.values())
    return [client.status().as_dict() for client in clients]


def shutdown_board_reader_processes() -> None:
    with _POOL_LOCK:
        clients = list(_POOL.values())
        _POOL.clear()
    for client in clients:
        client.shutdown()


def _get_client(key: tuple[str, str]) -> _BoardReaderClient:
    with _POOL_LOCK:
        client = _POOL.get(key)
        if client is None:
            client = _BoardReaderClient(key=key, ctx=_get_context())
            _POOL[key] = client
        return client


def _get_context() -> mp.context.BaseContext:
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    method = os.environ.get(BOARD_READER_START_METHOD_ENV, DEFAULT_BOARD_READER_START_METHOD).strip().lower()
    supported = set(mp.get_all_start_methods())
    if method not in supported:
        method = mp.get_start_method(allow_none=True) or ("fork" if "fork" in supported else "spawn")
    _CONTEXT = mp.get_context(method)
    return _CONTEXT


def _board_key(
    *,
    acquisition_folder: str | os.PathLike[str] | None,
    sensor_name: str | None,
    machine_id: str,
) -> tuple[str, str]:
    if acquisition_folder:
        try:
            folder_key = str(Path(acquisition_folder).expanduser().resolve())
        except Exception:
            folder_key = str(acquisition_folder)
    else:
        folder_key = f"__default__:{machine_id}"
    return folder_key, str(sensor_name or "iis3dwb_acc")


def _startup_grace_s() -> float:
    value = _safe_float(os.environ.get(BOARD_READER_STARTUP_GRACE_ENV))
    if value is None:
        value = DEFAULT_BOARD_READER_STARTUP_GRACE_S
    return max(0.0, value)


def _reader_timeout_s(timeout_s: float | None, *, duration_s: float | None) -> float:
    if timeout_s is not None:
        return max(1.0, float(timeout_s))
    raw = os.environ.get(BOARD_READER_TIMEOUT_ENV)
    value = _safe_float(raw)
    if value is None:
        value = DEFAULT_BOARD_READER_TIMEOUT_S
    if duration_s is not None:
        value = max(value, float(duration_s) + 10.0)
    return max(1.0, value)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _board_reader_worker_main(
    key: tuple[str, str],
    request_q: Any,
    response_q: Any,
) -> None:
    os.environ[BOARD_READER_CHILD_ENV] = "1"
    while True:
        request = request_q.get()
        if not isinstance(request, dict):
            continue
        command = request.get("command")
        request_id = request.get("request_id")
        if command == "shutdown":
            break
        if command != "read_window":
            response_q.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "exc_type": "BoardReaderProcessError",
                    "message": f"Unsupported board reader command: {command!r}",
                }
            )
            continue

        try:
            from .sdk_vibrometer import read_sdk_vibrometer_window

            kwargs = dict(request.get("kwargs") or {})
            window = read_sdk_vibrometer_window(**kwargs)
            response_q.put({"request_id": request_id, "ok": True, "window": window})
        except LiveSdkVibrometerDataRequiredError as exc:
            response_q.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "exc_type": "LiveSdkVibrometerDataRequiredError",
                    "message": str(exc),
                    "metadata": exc.metadata,
                }
            )
        except BaseException as exc:  # noqa: BLE001 - worker must report all failures.
            response_q.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "exc_type": exc.__class__.__name__,
                    "message": f"{exc.__class__.__name__}: {exc}",
                }
            )


atexit.register(shutdown_board_reader_processes)
