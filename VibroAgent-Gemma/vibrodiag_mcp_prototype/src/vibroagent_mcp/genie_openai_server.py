"""OpenAI-compatible HTTP adapter for Qualcomm Genie CLI runtimes."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

QAIRT_VERSION = "2.46.0.260424"
SDK_RELATIVE_PATH = Path(f"v{QAIRT_VERSION}") / "qairt" / QAIRT_VERSION
AARCH64_TARGET = "aarch64-oe-linux-gcc11.2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSISTENT_RUNNER_SOURCE = Path(__file__).with_name("genie_persistent_runner.cpp")
# The all-sensor agent returns one compact JSON object for five targets plus
# the fused network decision. The prompt constrains evidence and explanation
# lengths. Keep the cap large enough for a complete six-sensor JSON object
# while still bounding malformed generations before the webchat timeout.
DEFAULT_GENIE_MAX_OUTPUT_TOKENS = 1024
DEFAULT_GENIE_STOP_SEQUENCES = ("<|im_end|>", "<|endoftext|>")
_RUNTIME_CONFIG_PATHS: set[Path] = set()


@dataclass(frozen=True)
class GenieServerConfig:
    sdk_root: Path
    config_path: Path
    runner_path: Path
    model_id: str
    prompt_format: str
    timeout_s: float
    max_prompt_chars: int
    persistent: bool = True
    persistent_runner_path: Path | None = None


@dataclass(frozen=True)
class GeniePromptMetrics:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    total_s: float
    generation_s: float | None
    time_to_first_token_s: float | None
    tokens_per_second: float | None
    persistent: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "persistent": self.persistent,
            "total_s": self.total_s,
            "generation_s": self.generation_s,
            "time_to_first_token_s": self.time_to_first_token_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class GeniePromptResult:
    content: str
    metrics: GeniePromptMetrics


def _cleanup_runtime_config(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Best effort at interpreter shutdown.
        pass
    _RUNTIME_CONFIG_PATHS.discard(path)


def prepare_bounded_genie_config(
    config_path: Path,
    *,
    max_output_tokens: int = DEFAULT_GENIE_MAX_OUTPUT_TOKENS,
    stop_sequences: tuple[str, ...] = DEFAULT_GENIE_STOP_SEQUENCES,
) -> Path:
    """Create a same-directory runtime config with bounded generation.

    Qualcomm Genie reads the output-token ceiling and stop sequences from the
    dialog JSON.  OpenAI ``max_tokens`` is not otherwise consumed by this thin
    adapter, so leaving those fields absent can let a malformed answer run until
    the full context is exhausted.  The copy stays beside the source config so
    relative tokenizer/context-binary paths continue to resolve correctly.
    """

    source = Path(config_path).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dialog"), dict):
        raise ValueError(f"Genie config has no dialog object: {source}")
    dialog = data["dialog"]

    requested_limit = max(16, int(max_output_tokens))
    existing_limit = dialog.get("max-num-tokens")
    try:
        parsed_existing = int(existing_limit)
    except (TypeError, ValueError):
        parsed_existing = 0
    effective_limit = min(parsed_existing, requested_limit) if parsed_existing > 0 else requested_limit

    changed = parsed_existing != effective_limit
    dialog["max-num-tokens"] = effective_limit

    raw_stops = dialog.get("stop-sequence")
    if isinstance(raw_stops, str):
        existing_stops = [raw_stops] if raw_stops.strip() else []
    elif isinstance(raw_stops, list):
        existing_stops = [str(item) for item in raw_stops if str(item).strip()]
    else:
        existing_stops = []
    if not existing_stops:
        dialog["stop-sequence"] = list(stop_sequences)
        changed = True

    if not changed:
        return source

    runtime_path = source.with_name(f".{source.stem}.vibroagent-{os.getpid()}.json")
    temporary_path = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(runtime_path)
    _RUNTIME_CONFIG_PATHS.add(runtime_path)
    atexit.register(_cleanup_runtime_config, runtime_path)
    return runtime_path


def discover_sdk_root() -> Path | None:
    env_root = os.environ.get("QAIRT_SDK_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists():
            return root

    candidates: list[Path] = []
    for base in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        candidates.append((base / SDK_RELATIVE_PATH).resolve())
        candidates.append((base.parent / SDK_RELATIVE_PATH).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def default_runner_path(sdk_root: Path) -> Path:
    return sdk_root / "bin" / AARCH64_TARGET / "genie-t2t-run"


def default_persistent_runner_path() -> Path:
    configured = os.environ.get("GENIE_PERSISTENT_RUNNER")
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / ".build" / "genie_persistent_runner").resolve()


def _runner_binary_needs_rebuild(binary_path: Path) -> bool:
    if not binary_path.exists():
        return True
    if not PERSISTENT_RUNNER_SOURCE.exists():
        return False
    return PERSISTENT_RUNNER_SOURCE.stat().st_mtime > binary_path.stat().st_mtime


def ensure_persistent_runner(sdk_root: Path, runner_path: Path | None = None) -> Path:
    binary_path = (runner_path or default_persistent_runner_path()).expanduser().resolve()
    if not _runner_binary_needs_rebuild(binary_path):
        return binary_path
    if not PERSISTENT_RUNNER_SOURCE.exists():
        raise FileNotFoundError(f"Persistent Genie runner source not found: {PERSISTENT_RUNNER_SOURCE}")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to build the persistent Genie runner")

    host_lib = sdk_root / "lib" / AARCH64_TARGET
    include_root = sdk_root / "include"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        str(PERSISTENT_RUNNER_SOURCE),
        "-I",
        str(include_root),
        "-L",
        str(host_lib),
        "-Wl,-rpath," + str(host_lib),
        "-lGenie",
        "-o",
        str(binary_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"g++ exited with {proc.returncode}").strip()
        raise RuntimeError(f"Failed to build persistent Genie runner: {detail}")
    return binary_path


def build_runtime_env(sdk_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    host_lib = sdk_root / "lib" / AARCH64_TARGET
    htp_lib = sdk_root / "lib" / "hexagon-v73" / "unsigned"
    ld_parts = [str(host_lib), str(htp_lib)]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # On the QCS9075 Ubuntu image, adding broad system ADSP paths can make
    # QnnHtp device creation fail with error 1008. Prefer the SDK v73 skels
    # and preserve an explicit caller override only after that.
    adsp_parts = [str(htp_lib)]
    if env.get("ADSP_LIBRARY_PATH"):
        adsp_parts.append(env["ADSP_LIBRARY_PATH"])
    env["ADSP_LIBRARY_PATH"] = ":".join(adsp_parts)
    return env


def messages_to_prompt(messages: list[dict[str, Any]], prompt_format: str = "plain") -> str:
    if prompt_format == "qwen3":
        return _messages_to_qwen_prompt(messages)

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = _message_content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        elif role == "tool":
            name = str(message.get("name") or "tool")
            parts.append(f"Tool result ({name}):\n{content}")
        else:
            parts.append(f"User:\n{content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _messages_to_qwen_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = _message_content_to_text(message.get("content"))
        if not content:
            continue
        if role not in {"system", "assistant", "user"}:
            role = "user"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(part.strip() for part in text_parts if part and part.strip())
    return str(content).strip()


_PERSISTENT_RUNNERS: dict[tuple[str, str, str], "PersistentGenieProcess"] = {}
_PERSISTENT_RUNNERS_LOCK = threading.Lock()
_PERSISTENT_SHUTDOWN_REGISTERED = False


def _nonnegative_int(value: int) -> int | None:
    return value if value >= 0 else None


def _seconds_from_us(value: int) -> float | None:
    if value < 0:
        return None
    return max(0.0, value / 1_000_000.0)


def _coerced_timeout_s(timeout_s: Any, *, default: float) -> float:
    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(1.0, value)


def _request_timeout_s(request: dict[str, Any], default_timeout_s: float) -> float:
    ceiling_s = max(1.0, float(default_timeout_s))
    requested_s = _coerced_timeout_s(request.get("timeout_s"), default=ceiling_s)
    return min(ceiling_s, requested_s)


def _build_metrics(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_s: float,
    generation_s: float | None,
    time_to_first_token_s: float | None,
    persistent: bool,
) -> GeniePromptMetrics:
    if generation_s is not None and generation_s <= 0:
        generation_s = total_s
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = None
    if completion_tokens is not None and completion_tokens > 0:
        denominator = generation_s if generation_s is not None and generation_s > 0 else total_s
        if denominator > 0:
            tokens_per_second = completion_tokens / denominator
    return GeniePromptMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        total_s=total_s,
        generation_s=generation_s,
        time_to_first_token_s=time_to_first_token_s,
        tokens_per_second=tokens_per_second,
        persistent=persistent,
    )


def _remaining_timeout_s(deadline_s: float, message: str) -> float:
    remaining_s = float(deadline_s) - time.monotonic()
    if remaining_s <= 0:
        raise TimeoutError(message)
    return remaining_s


class PersistentGenieProcess:
    def __init__(self, config: GenieServerConfig, *, startup_timeout_s: float | None = None) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._protocol_fd: int | None = None
        self._log_handle: Any = None
        self._runner_path = ensure_persistent_runner(config.sdk_root, config.persistent_runner_path)
        self._start(timeout_s=startup_timeout_s)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def query(self, prompt: str, *, timeout_s: float | None = None) -> GeniePromptResult:
        budget_s = _coerced_timeout_s(timeout_s, default=self._config.timeout_s)
        queued_at = time.monotonic()
        # The timeout is a total request budget, not merely a generation budget.
        # Without a timed acquire, requests whose HTTP clients had already given
        # up remained queued on the single NPU and ran later as orphan work,
        # causing every following webchat request to time out in turn.
        acquired = self._lock.acquire(timeout=budget_s)
        if not acquired:
            raise TimeoutError("persistent Genie runner was busy until the request timeout expired")
        try:
            remaining_s = budget_s - (time.monotonic() - queued_at)
            if remaining_s <= 0:
                raise TimeoutError("persistent Genie runner request timeout expired while queued")
            if not self.is_alive():
                self._close_handles()
                self._start(timeout_s=remaining_s)
                remaining_s = budget_s - (time.monotonic() - queued_at)
            if remaining_s <= 0:
                raise TimeoutError("persistent Genie runner request timeout expired while restarting")
            try:
                return self._query_locked(prompt, timeout_s=remaining_s)
            except (TimeoutError, RuntimeError):
                # A timed-out or desynced generation leaves the runner mid-stream: its
                # stale output would corrupt the next query's protocol framing, and the
                # abandoned generation keeps the single serial NPU pegged so every later
                # check queues behind dead work. Discard the runner for a clean restart on
                # the next query. This is recovery, not a fallback — the caller still gets
                # the error and no deterministic result is substituted.
                self._discard_runner()
                raise
        finally:
            self._lock.release()

    def _discard_runner(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        self._close_handles()

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(b"QUIT\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
            self._close_handles()

    def _start(self, *, timeout_s: float | None = None) -> None:
        startup_timeout_s = _coerced_timeout_s(timeout_s, default=self._config.timeout_s)
        startup_deadline_s = time.monotonic() + startup_timeout_s
        read_fd, write_fd = os.pipe()

        log_path = self._runner_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab", buffering=0)
        env = build_runtime_env(self._config.sdk_root)
        env["GENIE_PERSISTENT_PROTOCOL_FD"] = str(write_fd)
        try:
            self._proc = subprocess.Popen(
                [str(self._runner_path), str(self._config.config_path)],
                cwd=str(self._config.config_path.parent),
                env=env,
                stdin=subprocess.PIPE,
                stdout=self._log_handle,
                stderr=self._log_handle,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
        self._protocol_fd = read_fd
        try:
            line = self._readline(
                _remaining_timeout_s(
                    startup_deadline_s,
                    "persistent Genie runner startup timed out",
                )
            )
            if line == b"READY\n":
                return
            if line.startswith(b"ERR "):
                raise RuntimeError(
                    self._read_error_body(
                        line,
                        _remaining_timeout_s(
                            startup_deadline_s,
                            "persistent Genie runner startup timed out while reading the error",
                        ),
                    )
                )
            raise RuntimeError(f"Unexpected persistent Genie runner startup response: {line!r}")
        except Exception:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
            self._close_handles()
            raise

    def _query_locked(self, prompt: str, *, timeout_s: float) -> GeniePromptResult:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("persistent Genie runner is not running")
        deadline_s = time.monotonic() + timeout_s
        payload = prompt.encode("utf-8")
        proc.stdin.write(f"QUERY {len(payload)}\n".encode("ascii"))
        proc.stdin.write(payload)
        proc.stdin.write(b"\n")
        proc.stdin.flush()

        line = self._readline(
            _remaining_timeout_s(deadline_s, "persistent Genie runner timed out")
        )
        if line.startswith(b"ERR "):
            raise RuntimeError(
                self._read_error_body(
                    line,
                    _remaining_timeout_s(
                        deadline_s,
                        "persistent Genie runner timed out while reading the error",
                    ),
                )
            )
        if not line.startswith(b"OK "):
            raise RuntimeError(f"Unexpected persistent Genie runner response: {line!r}")
        parts = line.decode("ascii", errors="replace").strip().split()
        if len(parts) != 7:
            raise RuntimeError(f"Malformed persistent Genie runner response header: {line!r}")
        try:
            content_size = int(parts[1])
            prompt_tokens = _nonnegative_int(int(parts[2]))
            completion_tokens = _nonnegative_int(int(parts[3]))
            total_us = int(parts[4])
            generation_us = int(parts[5])
            first_token_us = int(parts[6])
        except ValueError as exc:
            raise RuntimeError(f"Malformed persistent Genie runner numeric header: {line!r}") from exc
        content = self._read_exact(
            content_size,
            _remaining_timeout_s(
                deadline_s,
                "persistent Genie runner timed out while reading the response",
            ),
        ).decode("utf-8", errors="replace")
        delimiter = self._read_exact(
            1,
            _remaining_timeout_s(
                deadline_s,
                "persistent Genie runner timed out while finishing the response",
            ),
        )
        if delimiter != b"\n":
            raise RuntimeError("Malformed persistent Genie runner body delimiter")
        content = clean_genie_output(content)
        total_s = max(0.0, total_us / 1_000_000.0)
        metrics = _build_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_s=total_s,
            generation_s=_seconds_from_us(generation_us),
            time_to_first_token_s=_seconds_from_us(first_token_us),
            persistent=True,
        )
        return GeniePromptResult(content=content, metrics=metrics)

    def _read_error_body(self, header: bytes, timeout_s: float) -> str:
        try:
            size = int(header.decode("ascii", errors="replace").strip().split()[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Malformed persistent Genie runner error header: {header!r}") from exc
        deadline_s = time.monotonic() + timeout_s
        body = self._read_exact(
            size,
            _remaining_timeout_s(
                deadline_s,
                "persistent Genie runner timed out while reading the error body",
            ),
        ).decode("utf-8", errors="replace")
        delimiter = self._read_exact(
            1,
            _remaining_timeout_s(
                deadline_s,
                "persistent Genie runner timed out while finishing the error body",
            ),
        )
        if delimiter != b"\n":
            raise RuntimeError("Malformed persistent Genie runner error delimiter")
        return body

    def _readline(self, timeout_s: float) -> bytes:
        deadline_s = time.monotonic() + timeout_s
        data = bytearray()
        while True:
            chunk = self._read_exact(
                1,
                _remaining_timeout_s(
                    deadline_s,
                    "persistent Genie runner timed out while reading a protocol line",
                ),
            )
            data.extend(chunk)
            if chunk == b"\n":
                return bytes(data)

    def _read_exact(self, size: int, timeout_s: float) -> bytes:
        if size < 0:
            raise RuntimeError("negative protocol read size")
        fd = self._protocol_fd
        if fd is None:
            raise RuntimeError("persistent Genie protocol is closed")
        deadline = time.monotonic() + timeout_s
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("persistent Genie runner timed out")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise TimeoutError("persistent Genie runner timed out")
            chunk = os.read(fd, size - len(data))
            if not chunk:
                raise RuntimeError("persistent Genie runner exited")
            data.extend(chunk)
        return bytes(data)

    def _close_handles(self) -> None:
        if self._protocol_fd is not None:
            try:
                os.close(self._protocol_fd)
            except OSError:
                pass
            self._protocol_fd = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        self._proc = None


def _persistent_runner_key(config: GenieServerConfig) -> tuple[str, str, str]:
    return (
        str(config.sdk_root),
        str(config.config_path),
        str(config.persistent_runner_path or default_persistent_runner_path()),
    )


def _get_persistent_runner(
    config: GenieServerConfig,
    *,
    startup_timeout_s: float | None = None,
) -> PersistentGenieProcess:
    global _PERSISTENT_SHUTDOWN_REGISTERED
    key = _persistent_runner_key(config)
    with _PERSISTENT_RUNNERS_LOCK:
        runner = _PERSISTENT_RUNNERS.get(key)
        if runner is None:
            runner = PersistentGenieProcess(config, startup_timeout_s=startup_timeout_s)
            _PERSISTENT_RUNNERS[key] = runner
        if not _PERSISTENT_SHUTDOWN_REGISTERED:
            atexit.register(shutdown_persistent_genie_runners)
            _PERSISTENT_SHUTDOWN_REGISTERED = True
        return runner


def prewarm_persistent_genie_runner(config: GenieServerConfig) -> None:
    """Load the persistent model before the HTTP port advertises readiness."""

    if not config.persistent:
        return
    runner = _get_persistent_runner(config, startup_timeout_s=config.timeout_s)
    if not runner.is_alive():
        # A prior timed-out request may have discarded the process while keeping
        # the reusable runner object. Query performs a bounded restart; an empty
        # warm-up generation is deliberately avoided.
        with runner._lock:  # noqa: SLF001 - lifecycle helper in the same module
            if not runner.is_alive():
                runner._close_handles()  # noqa: SLF001
                runner._start(timeout_s=config.timeout_s)  # noqa: SLF001


def shutdown_persistent_genie_runners() -> None:
    with _PERSISTENT_RUNNERS_LOCK:
        runners = list(_PERSISTENT_RUNNERS.values())
        _PERSISTENT_RUNNERS.clear()
    for runner in runners:
        runner.stop()


def _validate_prompt_request(prompt: str, config: GenieServerConfig) -> None:
    if not config.config_path.exists():
        raise FileNotFoundError(f"Genie config not found: {config.config_path}")
    if not config.persistent and not config.runner_path.exists():
        raise FileNotFoundError(f"Genie runner not found: {config.runner_path}")
    if len(prompt) > config.max_prompt_chars:
        raise ValueError(
            f"prompt is {len(prompt)} characters, above GENIE_ADAPTER_MAX_PROMPT_CHARS={config.max_prompt_chars}"
        )


def run_genie_prompt_result(
    prompt: str,
    config: GenieServerConfig,
    *,
    timeout_s: float | None = None,
) -> GeniePromptResult:
    _validate_prompt_request(prompt, config)
    if config.persistent:
        budget_s = _coerced_timeout_s(timeout_s, default=config.timeout_s)
        started_s = time.monotonic()
        runner = _get_persistent_runner(config, startup_timeout_s=budget_s)
        remaining_s = budget_s - (time.monotonic() - started_s)
        if remaining_s <= 0:
            raise TimeoutError("persistent Genie runner startup exhausted the request timeout")
        return runner.query(prompt, timeout_s=remaining_s)
    effective_config = replace(config, timeout_s=_coerced_timeout_s(timeout_s, default=config.timeout_s))
    return _run_genie_prompt_once(prompt, effective_config)


def run_genie_prompt(prompt: str, config: GenieServerConfig) -> str:
    return run_genie_prompt_result(prompt, config).content


def _run_genie_prompt_once(prompt: str, config: GenieServerConfig) -> GeniePromptResult:
    cwd = config.config_path.parent
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".prompt.txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    started_s = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                str(config.runner_path),
                "--config",
                str(config.config_path),
                "--prompt_file",
                str(prompt_path),
            ],
            cwd=str(cwd),
            env=build_runtime_env(config.sdk_root),
            text=True,
            capture_output=True,
            timeout=config.timeout_s,
            check=False,
        )
    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass
    duration_s = max(0.0, time.perf_counter() - started_s)

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or f"genie-t2t-run exited with {proc.returncode}"
        raise RuntimeError(detail)
    content = clean_genie_output(proc.stdout)
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(0, len(content) // 4)
    metrics = _build_metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_s=duration_s,
        generation_s=duration_s,
        time_to_first_token_s=None,
        persistent=False,
    )
    return GeniePromptResult(content=content, metrics=metrics)

def clean_genie_output(output: str) -> str:
    text = output.replace("\r\n", "\n").replace("\r", "\n")
    if "[BEGIN]:" in text:
        text = text.rsplit("[BEGIN]:", 1)[1]
    if "[END]" in text:
        text = text.split("[END]", 1)[0]
    for marker in ("[COMPLETE]:", "[RESUME]:", "[ABORT]:", "[UNKNOWN]:"):
        text = text.replace(marker, "")
    for stop_sequence in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(stop_sequence, "")

    cleaned_lines: list[str] = []
    skip_prompt_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Using libGenie.so version"):
            continue
        if stripped.startswith("genie-t2t-run pid:"):
            continue
        if stripped.startswith("[PROMPT]:"):
            skip_prompt_block = True
            continue
        if skip_prompt_block and not stripped:
            skip_prompt_block = False
            continue
        if skip_prompt_block:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _usage_payload(result: GeniePromptResult, prompt: str, content: str) -> dict[str, int]:
    prompt_tokens = result.metrics.prompt_tokens
    if prompt_tokens is None:
        prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = result.metrics.completion_tokens
    if completion_tokens is None:
        completion_tokens = max(0, len(content) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class GenieOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "VibroAgentGenieOpenAI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path.rstrip("/")
        if path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self._config.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local-qualcomm-genie",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})
            return

        try:
            request = self._read_json_request()
            if request.get("stream"):
                raise ValueError("stream=true is not supported by this adapter")
            messages = request.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            prompt = messages_to_prompt(messages, self._config.prompt_format)
            timeout_s = _request_timeout_s(request, self._config.timeout_s)
            result = run_genie_prompt_result(prompt, self._config, timeout_s=timeout_s)
            content = result.content
            now = int(time.time())
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-genie-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": str(request.get("model") or self._config.model_id),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage_payload(result, prompt, content),
                    "genie_metrics": result.metrics.to_payload(),
                },
            )
        except Exception as exc:
            # subprocess.TimeoutExpired (one-shot runner mode) is a timeout too,
            # even though it is not a TimeoutError subclass.
            timed_out = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
            self._send_json(
                HTTPStatus.GATEWAY_TIMEOUT if timed_out else HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "message": f"{exc.__class__.__name__}: {exc}",
                        "type": "timeout_error" if timed_out else "genie_adapter_error",
                    }
                },
            )

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("GENIE_ADAPTER_QUIET", "0") not in {"1", "true", "TRUE"}:
            super().log_message(format, *args)

    @property
    def _config(self) -> GenieServerConfig:
        return self.server.genie_config  # type: ignore[attr-defined]

    def _read_json_request(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("missing request body")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    discovered_sdk = discover_sdk_root()
    parser = argparse.ArgumentParser(description="Serve Qualcomm Genie through an OpenAI-compatible local API.")
    parser.add_argument("--host", default=os.environ.get("GENIE_ADAPTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("GENIE_ADAPTER_PORT", 8910, minimum=1))
    parser.add_argument("--sdk-root", default=str(discovered_sdk) if discovered_sdk else os.environ.get("QAIRT_SDK_ROOT"))
    parser.add_argument("--config", default=os.environ.get("GENIE_CONFIG"), help="Path to model genie_config.json")
    parser.add_argument("--runner", default=os.environ.get("GENIE_RUNNER"), help="Path to genie-t2t-run fallback")
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("GENIE_PERSISTENT", True),
        help="Keep one loaded Genie dialog process alive between requests.",
    )
    parser.add_argument(
        "--persistent-runner",
        default=os.environ.get("GENIE_PERSISTENT_RUNNER"),
        help="Path to the persistent Genie helper binary. Built automatically when missing.",
    )
    parser.add_argument("--model-id", default=os.environ.get("GENIE_MODEL", "qcs9075-genie-local"))
    parser.add_argument(
        "--prompt-format",
        choices=["plain", "qwen3"],
        default=os.environ.get("GENIE_PROMPT_FORMAT", "plain"),
        help="Prompt template used to convert OpenAI chat messages for Genie.",
    )
    parser.add_argument("--timeout-s", type=float, default=_env_float("GENIE_TIMEOUT_S", 300.0, minimum=1.0))
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=_env_int("GENIE_ADAPTER_MAX_PROMPT_CHARS", 12000, minimum=1000),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=_env_int("GENIE_MAX_OUTPUT_TOKENS", DEFAULT_GENIE_MAX_OUTPUT_TOKENS, minimum=16),
        help="Hard generation ceiling injected into the Genie dialog config.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GenieServerConfig:
    if not args.sdk_root:
        raise SystemExit("QAIRT SDK root not found. Set QAIRT_SDK_ROOT or pass --sdk-root.")
    if not args.config:
        raise SystemExit("Genie model config is required. Set GENIE_CONFIG or pass --config.")
    sdk_root = Path(args.sdk_root).expanduser().resolve()
    source_config_path = Path(args.config).expanduser().resolve()
    config_path = prepare_bounded_genie_config(
        source_config_path,
        max_output_tokens=int(args.max_output_tokens),
    )
    runner_path = Path(args.runner).expanduser().resolve() if args.runner else default_runner_path(sdk_root)
    persistent_runner_path = (
        Path(args.persistent_runner).expanduser().resolve()
        if args.persistent_runner
        else default_persistent_runner_path()
    )
    return GenieServerConfig(
        sdk_root=sdk_root,
        config_path=config_path,
        runner_path=runner_path,
        model_id=str(args.model_id),
        prompt_format=str(args.prompt_format),
        timeout_s=float(args.timeout_s),
        max_prompt_chars=int(args.max_prompt_chars),
        persistent=bool(args.persistent),
        persistent_runner_path=persistent_runner_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    print(f"SDK root: {config.sdk_root}")
    print(f"Runner fallback: {config.runner_path}")
    print(f"Config: {config.config_path}")
    print(f"Prompt format: {config.prompt_format}")
    print(f"Persistent mode: {config.persistent}")
    if config.persistent:
        print(f"Persistent runner: {config.persistent_runner_path}")
        print("Prewarming persistent Genie model before opening the API port ...")
        prewarm_persistent_genie_runner(config)
        print("Persistent Genie model is ready.")
    server = ThreadingHTTPServer((args.host, args.port), GenieOpenAIHandler)
    server.genie_config = config  # type: ignore[attr-defined]
    print(f"Serving Qualcomm Genie OpenAI-compatible API at http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Genie adapter")
    finally:
        server.server_close()
        shutdown_persistent_genie_runners()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
