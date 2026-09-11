"""Browser webchat for testing a loaded OpenAI-compatible model with MCP tools.

Run:
    python -m vibroagent_mcp.webchat_server --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import copy
import hashlib
import json
import math
import os
import random
import ipaddress
import stat
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .errors import format_exception_for_response
from .genie_openai_server import (
    DEFAULT_GENIE_MAX_OUTPUT_TOKENS,
    GenieOpenAIHandler,
    GenieServerConfig,
    default_persistent_runner_path,
    default_runner_path,
    discover_sdk_root,
    prepare_bounded_genie_config,
    prewarm_persistent_genie_runner,
    shutdown_persistent_genie_runners,
)
from .model_client import ModelClient
from .offline_replay import (
    OfflineReplayConfigurationError,
    offline_replay_from_environment,
)
from .board_reader_process import (
    board_reader_processes_enabled,
    board_reader_process_statuses,
    prewarm_board_reader_processes,
    read_sdk_vibrometer_window_via_board_process,
)
from .qwen_mcp_host import (
    DEFAULT_QWEN_MAX_PROMPT_CHARS,
    DEFAULT_QWEN_MAX_TOKENS,
    DEFAULT_QWEN_TIMEOUT_S,
    HTTP_TIMEOUT_GRACE_S,
    ModelExplanationError,
    ModelExplanationTimeoutError,
    _load_openai_client,
    chat_with_mcp,
)
from .sdk_vibrometer import LiveSdkVibrometerDataRequiredError, read_sdk_vibrometer_window, _current_file_max_age_s
from .schemas import NETWORK_LABELS, find_out_of_domain_component_terms
from .sensor_registry import SensorRegistry
from .service import run_building_vibration_agent_pipeline_service
from .spectral import (
    analyze_impact_ringdown,
    apply_frequency_filter,
    aggregate_spectrum_for_plot,
    aggregate_waveform_for_plot,
    compute_periodogram_psd,
    compute_welch_psd,
    find_distinct_peaks,
    frequency_band_summary,
    integrated_psd_power,
    normalize_frequency_filter,
    periodic_window,
)


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen3.5-4b-instruct-revised"
DEFAULT_LIVE_SENSOR_CONFIG = "config/sensors.live.yaml"
DEFAULT_ANOMALY_WINDOW_DIR = "validation_exports/anomaly_windows"
DEFAULT_GENIE_HOST = "127.0.0.1"
DEFAULT_GENIE_PORT = 8910
DEFAULT_GENIE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_GENIE_PROMPT_FORMAT = "qwen3"
DEFAULT_AGENT_MODEL_TIMEOUT_S = 120.0
DEFAULT_MONITOR_MODEL_TIMEOUT_S = 5.0
DEFAULT_MONITOR_MAX_PROMPT_CHARS = 2600
DEFAULT_MONITOR_MAX_PROMPT_ESTIMATED_TOKENS = 880
DEFAULT_MONITOR_MAX_TOKENS = 128
DEFAULT_MONITOR_FOCUS_SENSOR_LIMIT = 5
MAX_AGENT_MODEL_TIMEOUT_S = 300.0
DEFAULT_DIRECT_NPU_TIMEOUT_S = 300.0
DEFAULT_DIRECT_NPU_MAX_TOKENS = 256
DEFAULT_DIRECT_NPU_MAX_PROMPT_CHARS = 9000
DIRECT_BUILDING_SYSTEM_PROMPT = (
    "You are a building-vibration monitoring assistant. "
    "Answer only in terms of registered building sensors, reference comparisons, vibration level, "
    "frequency content, transient events, data quality, and practical monitoring actions. "
    "Use supplied measurements only, state uncertainty clearly, and do not infer unmeasured causes. "
    "This is monitoring and triage, not a certified structural-safety assessment."
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QCS9075_GENIE_BUNDLE = (
    WORKSPACE_ROOT
    / "models"
    / "merged_4k_6000s"
    / "merged_6000_ctx4096_calib1-genie-w4a16-qualcomm_qcs9075"
)
DEFAULT_GENIE_CONFIG = DEFAULT_QCS9075_GENIE_BUNDLE / "genie_config.json"
VALID_LIVE_AXES = {"norm", "magnitude", "vector_norm", "x", "y", "z", "0", "1", "2"}
DEFAULT_PSD_FFT_WINDOW_S = 100.0
MAX_PSD_FFT_WINDOW_S = 120.0
DEFAULT_PSD_FFT_MAX_BINS = 1800
MAX_PSD_FFT_BINS = 5000
PSD_FFT_TOP_PEAKS = 8
DEFAULT_PSD_ESTIMATOR = "welch"
DEFAULT_PSD_WINDOW = "hann"
DEFAULT_PSD_PLOT_SCALE = "linear"
DEFAULT_WELCH_RESOLUTION_HZ = 0.25
DEFAULT_WELCH_OVERLAP = 0.5
DEFAULT_PSD_WAVEFORM_MAX_POINTS = 900
MAX_PSD_WAVEFORM_POINTS = 2000
ANOMALOUS_NETWORK_LABELS = {
    "localized_vibration_deviation",
    "multi_sensor_structure_wide_vibration_event",
    "sensor_network_quality_issue",
}
ANOMALOUS_SENSOR_LABELS = {
    "significant_local_deviation",
    "impulsive_or_shock_event",
    "possible_sensor_issue",
}
MONITOR_STATUS_CODE_TO_LABEL = {
    "normal": "normal_relative_to_reference_sensor",
    "local": "localized_vibration_deviation",
    "mild_local": "mild_local_deviation",
    "mild_multi": "mild_multi_sensor_deviation",
    "multi": "multi_sensor_structure_wide_vibration_event",
    "quality": "sensor_network_quality_issue",
    "insufficient": "insufficient_data",
}
MONITOR_STATUS_LABEL_TO_CODE = {label: code for code, label in MONITOR_STATUS_CODE_TO_LABEL.items()}


@dataclass
class EmbeddedGenieRuntime:
    enabled: bool
    serving: bool
    base_url: str
    model: str
    config_path: str | None = None
    runner_path: str | None = None
    persistent: bool = False
    persistent_runner_path: str | None = None
    sdk_root: str | None = None
    backend_types: list[str] = field(default_factory=list)
    htp_context_bins: list[str] = field(default_factory=list)
    htp_context_bins_exist: bool = False
    npu_offload: bool = False
    warning: str | None = None
    startup_error: str | None = None
    server: ThreadingHTTPServer | None = field(default=None, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "serving": self.serving,
            "base_url": self.base_url,
            "model": self.model,
            "config_path": self.config_path,
            "runner_path": self.runner_path,
            "persistent": self.persistent,
            "persistent_runner_path": self.persistent_runner_path,
            "sdk_root": self.sdk_root,
            "backend_types": self.backend_types,
            "htp_context_bins": self.htp_context_bins,
            "htp_context_bins_exist": self.htp_context_bins_exist,
            "npu_offload": self.npu_offload,
            "warning": self.warning,
            "startup_error": self.startup_error,
        }

    def shutdown(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        shutdown_persistent_genie_runners()


class WebchatHandler(BaseHTTPRequestHandler):
    server_version = "VibroAgentWebchat/0.1"

    def do_OPTIONS(self) -> None:
        if not self._request_origin_allowed():
            self._send_json({"ok": False, "error": "origin_not_allowed"}, status=HTTPStatus.FORBIDDEN)
            return
        self._send_bytes(b"", content_type="text/plain")

    def do_GET(self) -> None:
        if not self._request_origin_allowed():
            self._send_json({"ok": False, "error": "origin_not_allowed"}, status=HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path in {"/npu-chat", "/simple-chat"}:
            self._send_bytes(_NPU_CHAT_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path in {"/graph", "/debug"}:
            self._send_bytes(_GRAPH_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path in {"/assets/st-logo.png", "/assets/st_logo_2020.png"}:
            logo_candidates = (
                Path(__file__).resolve().parent / "static" / "st_logo_2020.png",
                Path(__file__).resolve().parents[2] / "static" / "st_logo_2020.png",
            )
            for logo_path in logo_candidates:
                try:
                    self._send_bytes(logo_path.read_bytes(), content_type="image/png")
                    return
                except OSError:
                    continue
            self._send_bytes(b"", status=HTTPStatus.NOT_FOUND, content_type="text/plain")
            return

        if parsed.path == "/favicon.ico":
            self._send_bytes(b"", status=HTTPStatus.NO_CONTENT, content_type="image/x-icon")
            return

        if parsed.path == "/health":
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/config":
            runtime_status = _embedded_genie_status(self.server)
            if runtime_status.get("serving"):
                base_url = str(runtime_status["base_url"])
                api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
                model = str(runtime_status["model"])
            else:
                base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
                api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
                requested_model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
                model = _auto_select_agent_model(
                    base_url=base_url,
                    api_key=api_key,
                    requested_model=requested_model,
                )
            self._send_json(
                {
                    "base_url": base_url,
                    "model": model,
                    "has_api_key": bool(os.environ.get("QWEN_API_KEY")),
                    "npu": runtime_status,
                }
            )
            return

        if parsed.path == "/api/npu":
            self._send_json({"ok": True, "npu": _embedded_genie_status(self.server)})
            return

        if parsed.path == "/api/board-readers":
            self._send_json({"ok": True, "status": "ok", "board_reader_processes": board_reader_process_statuses()})
            return

        if parsed.path == "/api/live-vibrometer":
            self._send_json(_live_vibrometer_payload(parse_qs(parsed.query)))
            return

        if parsed.path == "/api/psd-fft":
            self._send_json(_psd_fft_payload(parse_qs(parsed.query)))
            return

        if parsed.path == "/api/live-sensors":
            self._send_json(_live_sensors_payload(parse_qs(parsed.query)))
            return

        if parsed.path == "/api/agent-monitor":
            self._send_json(_agent_monitor_payload(parse_qs(parsed.query)))
            return

        if parsed.path == "/api/anomaly-windows":
            self._send_json(_anomaly_windows_payload(parse_qs(parsed.query)))
            return

        if parsed.path == "/api/models":
            query = parse_qs(parsed.query)
            try:
                base_url = _request_model_base_url(
                    _first(query, "base_url"),
                    default=os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL),
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            api_key = _first(query, "api_key") or os.environ.get("QWEN_API_KEY", "EMPTY")
            try:
                client = _load_openai_client(base_url=base_url, api_key=api_key)
                models = client.models.list()
                self._send_json({"ok": True, "models": [item.id for item in models.data]})
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"},
                    status=HTTPStatus.BAD_GATEWAY,
                )
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._request_origin_allowed():
            self._send_json({"ok": False, "error": "origin_not_allowed"}, status=HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/npu-chat":
            if not _chat_inflight_try_begin():
                self._send_json(_chat_busy_response(), status=HTTPStatus.CONFLICT)
                return
            try:
                payload = self._read_json()
                result = _direct_npu_chat_payload(payload, self.server)
                self._send_json({"ok": True, **result})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except ModelExplanationTimeoutError as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "model_timeout",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.GATEWAY_TIMEOUT,
                )
            except ModelExplanationError as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "model_explanation_failed",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            except Exception as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "chat_request_failed",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            finally:
                _chat_inflight_end()
            return

        if parsed.path == "/api/chat":
            if not _chat_inflight_try_begin():
                self._send_json(_chat_busy_response(), status=HTTPStatus.CONFLICT)
                return
            try:
                payload = self._read_json()
                message = str(payload.get("message", "")).strip()
                if not message:
                    self._send_json({"error": "message is required"}, status=HTTPStatus.BAD_REQUEST)
                    return

                timeout_s = _payload_float(
                    payload.get("timeout_s"),
                    default=_env_float_value("QWEN_TIMEOUT_S", DEFAULT_QWEN_TIMEOUT_S, minimum=1.0),
                    minimum=1.0,
                    maximum=600.0,
                )
                max_tokens = _payload_int(
                    payload.get("max_tokens"),
                    default=_env_int_value("QWEN_MAX_TOKENS", DEFAULT_QWEN_MAX_TOKENS, minimum=32),
                    minimum=32,
                    maximum=2048,
                )
                max_prompt_chars = _payload_int(
                    payload.get("max_prompt_chars"),
                    default=_env_int_value(
                        "QWEN_MAX_PROMPT_CHARS",
                        DEFAULT_QWEN_MAX_PROMPT_CHARS,
                        minimum=2000,
                    ),
                    minimum=2000,
                    maximum=30000,
                )
                result = asyncio.run(
                    chat_with_mcp(
                        message,
                        base_url=_request_model_base_url(_clean(payload.get("base_url")), default=None),
                        api_key=_clean(payload.get("api_key")),
                        model=_clean(payload.get("model")),
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                        max_prompt_chars=max_prompt_chars,
                        model_tool_routing=_payload_bool(
                            payload.get("model_tool_routing"),
                            default=_env_bool("QWEN_MODEL_TOOL_ROUTING", False),
                        ),
                        agentic=_payload_bool(
                            payload.get("agentic"),
                            default=_env_bool("QWEN_AGENTIC_CHAT", True),
                        ),
                    )
                )
                self._send_json({"ok": True, **result})
            except ModelExplanationTimeoutError as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "model_timeout",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.GATEWAY_TIMEOUT,
                )
            except ModelExplanationError as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "model_explanation_failed",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            except Exception as exc:
                self._send_json(
                    {
                        "ok": False,
                        "error": "chat_request_failed",
                        "message": format_exception_for_response(exc),
                        "fallback_generated": False,
                        "model_explanation_only": True,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            finally:
                _chat_inflight_end()
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        cors_origin = self._cors_origin()
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _request_origin_allowed(self) -> bool:
        return _request_origin_allowed(
            host_header=self.headers.get("Host"),
            origin_header=self.headers.get("Origin"),
            referer_header=self.headers.get("Referer"),
            sec_fetch_site=self.headers.get("Sec-Fetch-Site"),
        )

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if origin and self._request_origin_allowed():
            return origin
        return None


def _embedded_genie_status(server: Any) -> dict[str, Any]:
    runtime = getattr(server, "embedded_genie_runtime", None)
    if isinstance(runtime, EmbeddedGenieRuntime):
        return runtime.status()
    return {
        "enabled": False,
        "serving": False,
        "base_url": os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
        "config_path": None,
        "runner_path": None,
        "sdk_root": None,
        "backend_types": [],
        "htp_context_bins": [],
        "htp_context_bins_exist": False,
        "npu_offload": False,
        "warning": None,
        "startup_error": None,
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _allowed_web_origins() -> set[str]:
    raw = os.environ.get("VIBRO_ALLOWED_WEB_ORIGINS", "")
    return {origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()}


def _origin_matches_host(origin: str, host_header: str | None) -> bool:
    if not origin or not host_header:
        return False
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return parsed.netloc.lower() == host_header.lower()


def _request_origin_allowed(
    *,
    host_header: str | None,
    origin_header: str | None,
    referer_header: str | None,
    sec_fetch_site: str | None,
) -> bool:
    allowed = _allowed_web_origins()
    origin = (origin_header or "").strip().rstrip("/")
    if origin:
        return origin in allowed or _origin_matches_host(origin, host_header)

    fetch_site = (sec_fetch_site or "").strip().lower()
    if fetch_site in {"cross-site", "same-site"}:
        return False

    referer = (referer_header or "").strip()
    if referer:
        parsed = urlparse(referer)
        referer_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/") if parsed.scheme and parsed.netloc else ""
        if referer_origin and not (referer_origin in allowed or _origin_matches_host(referer_origin, host_header)):
            return False
    return True


def _request_model_base_url(value: Any, *, default: str | None) -> str | None:
    raw = _clean(value)
    if raw is None:
        return default
    return _validate_request_model_base_url(raw)


def _validate_request_model_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include credentials")
    if _is_loopback_hostname(parsed.hostname):
        return value.rstrip("/")
    if not _env_bool("VIBRO_ALLOW_REMOTE_MODEL_URLS", False):
        raise ValueError("request-supplied base_url must point to localhost unless VIBRO_ALLOW_REMOTE_MODEL_URLS=1")
    if parsed.scheme != "https" and not _env_bool("VIBRO_ALLOW_INSECURE_REMOTE_MODEL_URLS", False):
        raise ValueError("remote request-supplied base_url must use https")
    return value.rstrip("/")


def _is_loopback_hostname(hostname: str) -> bool:
    host = hostname.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _allowed_sensor_config_roots() -> list[Path]:
    roots: list[Path] = [Path(__file__).resolve().parents[2]]
    env_config = os.environ.get("VIBRO_BUILDING_SENSOR_CONFIG")
    if env_config:
        roots.append(Path(env_config).expanduser().resolve().parent)
    raw = os.environ.get("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", "")
    for item in raw.split(os.pathsep):
        if item.strip():
            roots.append(Path(item.strip()).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _validate_live_sensor_config_path(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    for root in _allowed_sensor_config_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in _allowed_sensor_config_roots())
    raise ValueError(f"config_path must be inside an allowed sensor config directory: {allowed}. Got: {resolved}")


def _env_int_value(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float_value(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _start_embedded_genie_runtime(
    *,
    host: str,
    port: int,
    sdk_root: str | None,
    config_path: str | None,
    runner_path: str | None,
    persistent: bool,
    persistent_runner_path: str | None,
    model: str,
    prompt_format: str,
    timeout_s: float,
    max_prompt_chars: int,
    max_output_tokens: int = DEFAULT_GENIE_MAX_OUTPUT_TOKENS,
    require_qnn_htp: bool = True,
) -> EmbeddedGenieRuntime:
    base_url = f"http://{host}:{port}/v1"
    resolved_config: Path | None = None
    resolved_runner: Path | None = None
    resolved_persistent_runner: Path | None = None
    resolved_sdk: Path | None = None
    summary = _summarize_genie_config_for_npu(DEFAULT_GENIE_CONFIG)
    try:
        resolved_sdk = Path(sdk_root).expanduser().resolve() if sdk_root else discover_sdk_root()
        if resolved_sdk is None:
            raise RuntimeError("QAIRT SDK root not found. Set QAIRT_SDK_ROOT or pass --qairt-sdk-root.")
        source_config = Path(config_path).expanduser().resolve() if config_path else DEFAULT_GENIE_CONFIG.resolve()
        resolved_config = prepare_bounded_genie_config(
            source_config,
            max_output_tokens=max_output_tokens,
        )
        resolved_runner = Path(runner_path).expanduser().resolve() if runner_path else default_runner_path(resolved_sdk)
        resolved_persistent_runner = (
            Path(persistent_runner_path).expanduser().resolve()
            if persistent_runner_path
            else default_persistent_runner_path()
        )
        if not resolved_config.exists():
            raise FileNotFoundError(f"Genie config not found: {resolved_config}")
        if not resolved_runner.exists():
            raise FileNotFoundError(f"Genie runner not found: {resolved_runner}")

        summary = _summarize_genie_config_for_npu(resolved_config)
        if require_qnn_htp and not summary["npu_offload"]:
            raise RuntimeError(
                summary["warning"]
                or "Genie config must use QnnHtp and all referenced context binaries must exist."
            )
        genie_config = GenieServerConfig(
            sdk_root=resolved_sdk,
            config_path=resolved_config,
            runner_path=resolved_runner,
            model_id=model,
            prompt_format=prompt_format,
            timeout_s=timeout_s,
            max_prompt_chars=max_prompt_chars,
            persistent=persistent,
            persistent_runner_path=resolved_persistent_runner,
        )
        # Do not advertise an OpenAI-compatible port until the persistent model
        # has finished loading. This prevents the first webchat turn from spending
        # its entire request budget on model startup.
        prewarm_persistent_genie_runner(genie_config)
        adapter = ThreadingHTTPServer((host, port), GenieOpenAIHandler)
        adapter.genie_config = genie_config  # type: ignore[attr-defined]
        thread = threading.Thread(target=adapter.serve_forever, name="embedded-genie-openai", daemon=True)
        runtime = EmbeddedGenieRuntime(
            enabled=True,
            serving=True,
            base_url=base_url,
            model=model,
            config_path=str(resolved_config),
            runner_path=str(resolved_runner),
            persistent=persistent,
            persistent_runner_path=str(resolved_persistent_runner),
            sdk_root=str(resolved_sdk),
            backend_types=summary["backend_types"],
            htp_context_bins=summary["htp_context_bins"],
            htp_context_bins_exist=summary["htp_context_bins_exist"],
            npu_offload=summary["npu_offload"],
            warning=summary["warning"],
            server=adapter,
            thread=thread,
        )
        thread.start()
        atexit.register(runtime.shutdown)
        return runtime
    except Exception as exc:
        return EmbeddedGenieRuntime(
            enabled=True,
            serving=False,
            base_url=base_url,
            model=model,
            config_path=str(resolved_config) if resolved_config else str(config_path or DEFAULT_GENIE_CONFIG),
            runner_path=str(resolved_runner) if resolved_runner else runner_path,
            persistent=persistent,
            persistent_runner_path=str(resolved_persistent_runner) if resolved_persistent_runner else persistent_runner_path,
            sdk_root=str(resolved_sdk) if resolved_sdk else sdk_root,
            backend_types=summary["backend_types"],
            htp_context_bins=summary["htp_context_bins"],
            htp_context_bins_exist=summary["htp_context_bins_exist"],
            npu_offload=False,
            warning=summary["warning"],
            startup_error=f"{exc.__class__.__name__}: {exc}",
        )


def _summarize_genie_config_for_npu(config_path: Path) -> dict[str, Any]:
    backend_types: list[str] = []
    htp_context_bins: list[str] = []
    htp_context_bins_exist = False
    warning: str | None = None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "backend_types": [],
            "htp_context_bins": [],
            "htp_context_bins_exist": False,
            "npu_offload": False,
            "warning": f"Could not inspect Genie config for NPU status: {exc.__class__.__name__}: {exc}",
        }

    dialog = data.get("dialog") if isinstance(data, dict) else None
    engine = dialog.get("engine") if isinstance(dialog, dict) else None
    engines = engine if isinstance(engine, list) else [engine] if isinstance(engine, dict) else []
    for item in engines:
        if not isinstance(item, dict):
            continue
        backend = item.get("backend")
        if isinstance(backend, dict):
            backend_type = str(backend.get("type") or "").strip()
            if backend_type and backend_type not in backend_types:
                backend_types.append(backend_type)
        model = item.get("model")
        if isinstance(model, dict):
            binary = model.get("binary")
            if isinstance(binary, dict):
                for raw_path in _as_string_list(binary.get("ctx-bins")):
                    htp_context_bins.append(str(_resolve_config_relative_path(config_path, raw_path)))

    if htp_context_bins:
        htp_context_bins_exist = all(Path(path).exists() for path in htp_context_bins)

    npu_offload = "QnnHtp" in backend_types and bool(htp_context_bins) and htp_context_bins_exist
    if "QnnHtp" in backend_types and not htp_context_bins_exist:
        warning = "Genie config selects QnnHtp, but one or more context binaries are missing."
    elif "QnnHtp" not in backend_types:
        backends = ", ".join(backend_types) if backend_types else "unknown"
        warning = (
            f"Embedded Genie is using backend {backends}, not QnnHtp. "
            "This validates the webchat path; true Hexagon NPU offload needs a QnnHtp config with existing context binaries."
        )

    return {
        "backend_types": backend_types,
        "htp_context_bins": htp_context_bins,
        "htp_context_bins_exist": htp_context_bins_exist,
        "npu_offload": npu_offload,
        "warning": warning,
    }


def _resolve_config_relative_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _openai_chat_completion_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _post_openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    request_payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    raw = json.dumps(request_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(_openai_chat_completion_url(base_url), data=raw, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            response_raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        if int(exc.code) == int(HTTPStatus.GATEWAY_TIMEOUT):
            raise TimeoutError(f"OpenAI-compatible endpoint timed out: {detail}") from exc
        raise RuntimeError(f"OpenAI-compatible endpoint returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower() or "timeout" in str(reason).lower():
            raise TimeoutError(f"OpenAI-compatible endpoint timed out: {reason}") from exc
        raise RuntimeError(f"OpenAI-compatible endpoint request failed: {reason}") from exc
    parsed = json.loads(response_raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI-compatible endpoint returned a non-object JSON payload")
    return parsed


def _metric_float(metrics: Any, key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _first_choice_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenAI-compatible endpoint response did not include choices[0]")
    return choices[0]


def _direct_npu_chat_payload(payload: dict[str, Any], server: Any) -> dict[str, Any]:
    runtime_status = _embedded_genie_status(server)
    if runtime_status.get("serving"):
        base_url = str(runtime_status["base_url"])
        model = str(runtime_status["model"])
        api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
    else:
        base_url = _request_model_base_url(
            _clean(payload.get("base_url")),
            default=os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL),
        )
        api_key = _clean(payload.get("api_key")) or os.environ.get("QWEN_API_KEY", "EMPTY")
        model = _clean(payload.get("model")) or os.environ.get("QWEN_MODEL", DEFAULT_MODEL)

    max_prompt_chars = _payload_int(
        payload.get("max_prompt_chars"),
        default=_env_int_value(
            "DIRECT_NPU_MAX_PROMPT_CHARS",
            DEFAULT_DIRECT_NPU_MAX_PROMPT_CHARS,
            minimum=1000,
        ),
        minimum=1000,
        maximum=30000,
    )
    messages = _normalize_direct_chat_messages(payload, max_prompt_chars=max_prompt_chars)
    temperature = _payload_float(payload.get("temperature"), default=0.2, minimum=0.0, maximum=2.0)
    timeout_s = _payload_float(
        payload.get("timeout_s"),
        default=_env_float_value("QWEN_TIMEOUT_S", DEFAULT_DIRECT_NPU_TIMEOUT_S, minimum=1.0),
        minimum=1.0,
        maximum=600.0,
    )
    max_tokens = _payload_int(
        payload.get("max_tokens"),
        default=_env_int_value("QWEN_MAX_TOKENS", DEFAULT_DIRECT_NPU_MAX_TOKENS, minimum=16),
        minimum=16,
        maximum=2048,
    )
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Consumed by the local Genie adapter.  The HTTP client below adds a
        # small grace period so it can receive the adapter's timeout response.
        "timeout_s": timeout_s,
    }
    request_started_s = time.perf_counter()
    try:
        response_payload = _post_openai_chat_completion(
            base_url=base_url,
            api_key=api_key,
            request_payload=request_payload,
            timeout_s=timeout_s + HTTP_TIMEOUT_GRACE_S,
        )
    except TimeoutError as exc:
        raise ModelExplanationTimeoutError(
            "The model timed out while generating the direct chat explanation. "
            "No fallback answer was generated."
        ) from exc
    except Exception as exc:
        raise ModelExplanationError(
            "The model could not generate the direct chat explanation. "
            f"No fallback answer was generated. ({exc.__class__.__name__}: {exc})"
        ) from exc
    request_duration_s = max(0.0, time.perf_counter() - request_started_s)

    choice = _first_choice_payload(response_payload)
    message_value = choice.get("message")
    message_payload: dict[str, Any] = message_value if isinstance(message_value, dict) else {}
    usage_value = response_payload.get("usage")
    metrics_value = response_payload.get("genie_metrics")
    usage_payload = usage_value if isinstance(usage_value, dict) else None
    adapter_metrics = metrics_value if isinstance(metrics_value, dict) else None

    completion_tokens = _usage_int(usage_payload, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _usage_int(adapter_metrics, "completion_tokens")

    total_s = _metric_float(adapter_metrics, "total_s") or request_duration_s
    generation_s = _metric_float(adapter_metrics, "generation_s")
    time_to_first_token_s = _metric_float(adapter_metrics, "time_to_first_token_s")
    tokens_per_second = _metric_float(adapter_metrics, "tokens_per_second")
    if tokens_per_second is None and completion_tokens is not None:
        denominator = generation_s if generation_s is not None and generation_s > 0 else total_s
        tokens_per_second = (completion_tokens / denominator) if denominator > 0 else None

    answer = str(message_payload.get("content") or "").strip()
    if not answer:
        raise ModelExplanationError(
            "The model returned an empty direct chat explanation. No fallback answer was generated."
        )
    rejected_terms = find_out_of_domain_component_terms(answer)
    if rejected_terms:
        raise ModelExplanationError(
            "The model explanation left the building-vibration domain and was rejected. "
            "No fallback answer was generated."
        )

    return {
        "answer": answer,
        "domain_guard_triggered": False,
        "domain_guard_rejected_term_count": 0,
        "model": model,
        "model_request_sent": True,
        "model_response_used": True,
        "model_explanation_only": True,
        "fallbacks_disabled": True,
        "base_url": base_url,
        "messages": messages,
        "finish_reason": choice.get("finish_reason"),
        "timeout_s": timeout_s,
        "max_tokens": max_tokens,
        "prompt_chars": sum(len(item.get("content", "")) for item in messages),
        "usage": usage_payload,
        "metrics": {
            "duration_s": total_s,
            "request_duration_s": request_duration_s,
            "generation_s": generation_s,
            "time_to_first_token_s": time_to_first_token_s,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
            "persistent": bool(adapter_metrics.get("persistent")) if isinstance(adapter_metrics, dict) else False,
        },
        "npu": runtime_status,
    }

def _usage_int(usage: Any, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _normalize_direct_chat_messages(
    payload: dict[str, Any],
    *,
    max_prompt_chars: int = DEFAULT_DIRECT_NPU_MAX_PROMPT_CHARS,
) -> list[dict[str, str]]:
    raw_messages = payload.get("messages")
    if raw_messages is None:
        message = _clean(payload.get("message"))
        if not message:
            raise ValueError("message or messages is required")
        raw_messages = [{"role": "user", "content": message}]
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list")

    normalized: list[dict[str, str]] = [{"role": "system", "content": DIRECT_BUILDING_SYSTEM_PROMPT}]
    user_turns = 0
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role == "system":
            # The direct test page is still part of the building application, so
            # callers cannot replace the application-domain instruction.
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        content = _clean(item.get("content"))
        if content:
            normalized.append({"role": role, "content": content})
            user_turns += 1
    if user_turns == 0:
        raise ValueError("messages must include at least one non-empty user or assistant message")

    # Keep the newest conversation turns plus the fixed application instruction.
    budget = max(1000, int(max_prompt_chars))
    system_message = normalized[0]
    system_cost = len(system_message["content"]) + 32
    turn_budget = max(200, budget - system_cost)
    selected_reversed: list[dict[str, str]] = []
    used = 0
    for item in reversed(normalized[1:]):
        cost = len(item["content"]) + 32
        if selected_reversed and used + cost > turn_budget:
            break
        if not selected_reversed and cost > turn_budget:
            clipped = dict(item)
            clipped["content"] = clipped["content"][-max(120, turn_budget - 32) :]
            selected_reversed.append(clipped)
            break
        selected_reversed.append(item)
        used += cost
        if len(selected_reversed) >= 24:
            break
    return [system_message, *reversed(selected_reversed)]


def _payload_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return min(max(number, int(minimum)), int(maximum))


def _payload_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _payload_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, minimum), maximum)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    return values[0] if values else None


def _float_query(query: dict[str, list[str]], key: str, default: float) -> float:
    raw = _first(query, key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    raw = _first(query, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_query(query: dict[str, list[str]], key: str, default: bool) -> bool:
    raw = _first(query, key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _live_vibrometer_payload(query: dict[str, list[str]], *, reader=None) -> dict[str, Any]:
    axis = (_first(query, "axis") or "norm").strip().lower()
    if axis not in VALID_LIVE_AXES:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2",
        }
    sensor_name = _clean(_first(query, "sensor_name"))
    machine_id = _clean(_first(query, "machine_id")) or "stwinbox_iis3dwb"
    acquisition_folder = _clean(_first(query, "acquisition_folder"))
    duration_s = min(10.0, max(0.1, _float_query(query, "duration_s", 1.0)))
    max_points = min(5000, max(64, _int_query(query, "max_points", 1800)))
    require_current = _bool_query(query, "require_current", True)
    start_time_raw = _clean(_first(query, "start_time_s"))
    try:
        replay_controller = offline_replay_from_environment()
        replay_snapshot = replay_controller.snapshot() if replay_controller else None
        start_time_s = (
            float(start_time_raw)
            if start_time_raw is not None
            else replay_controller.window_start(duration_s) if replay_controller else None
        )
    except (OfflineReplayConfigurationError, ValueError) as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "replay_configuration_failed",
            "message": format_exception_for_response(exc),
        }
    if replay_controller:
        require_current = False
    request_started_s = time.perf_counter()
    process_reader = _bool_query(query, "process_reader", False)
    reader_fn = reader or (
        read_sdk_vibrometer_window_via_board_process
        if board_reader_processes_enabled(default=process_reader)
        else read_sdk_vibrometer_window
    )

    try:
        decode_started_s = time.perf_counter()
        window = reader_fn(
            machine_id=machine_id,
            acquisition_folder=acquisition_folder,
            sensor_name=sensor_name,
            axis=axis,
            start_time_s=start_time_s,
            duration_s=duration_s,
            require_current=require_current,
        )
        decode_duration_s = time.perf_counter() - decode_started_s
    except LiveSdkVibrometerDataRequiredError as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "live_data_required",
            "message": format_exception_for_response(exc),
            "metadata": exc.metadata,
            "diagnostics": {
                "decode_duration_s": elapsed_s,
                "server_duration_s": elapsed_s,
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
                "decoded_latest_gap_s": exc.metadata.get("sdk_latest_timestamp_gap_s"),
                "sdk_timestamp_gap_s": exc.metadata.get("sdk_latest_timestamp_gap_s"),
                "data_file_age_s": exc.metadata.get("data_file_age_s"),
                "live_latency_s": _effective_live_latency_s(exc.metadata),
            },
        }
    except Exception as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "read_failed",
            "message": format_exception_for_response(exc),
            "diagnostics": {
                "decode_duration_s": elapsed_s,
                "server_duration_s": elapsed_s,
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            },
        }

    try:
        signal = _validated_live_signal(window.signal)
        sampling_rate_hz = _validated_sampling_rate(window.sampling_rate_hz)
        sample_count = int(signal.size)
        stride = max(1, math.ceil(sample_count / max_points))
        values = signal[::stride][:max_points]
        if window.timestamps_s is not None and window.timestamps_s.size == sample_count:
            times = window.timestamps_s[::stride][:max_points]
            time_values = [float(value) for value in times if math.isfinite(float(value))]
            if len(time_values) != int(values.size):
                time_values = [float((index * stride) / sampling_rate_hz) for index in range(values.size)]
        else:
            time_values = [float((index * stride) / sampling_rate_hz) for index in range(values.size)]
        stats = {
            "min_g": float(signal.min()),
            "max_g": float(signal.max()),
            "mean_g": float(signal.mean()),
            "rms_g": float((signal * signal).mean() ** 0.5),
        }
    except ValueError as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "invalid_window",
            "message": format_exception_for_response(exc),
            "metadata": window.metadata,
            "diagnostics": {
                "decode_duration_s": float(decode_duration_s),
                "server_duration_s": float(elapsed_s),
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            },
        }

    server_duration_s = time.perf_counter() - request_started_s
    decoded_latest_gap_s = window.metadata.get("sdk_latest_timestamp_gap_s")
    data_file_age_s = window.metadata.get("data_file_age_s")
    return {
        "ok": True,
        "status": "ok",
        "machine_id": machine_id,
        "sensor_name": sensor_name,
        "acquisition_folder": acquisition_folder,
        "axis": axis,
        "replay": replay_snapshot,
        "sample_count": sample_count,
        "points_returned": int(values.size),
        "downsample_stride": int(stride),
        "sampling_rate_hz": int(sampling_rate_hz),
        "window_duration_s": float(sample_count / float(sampling_rate_hz)),
        "time_s": time_values,
        "samples_g": [float(value) for value in values],
        "stats": stats,
        "metadata": window.metadata,
        "diagnostics": {
            "decode_duration_s": float(decode_duration_s),
            "server_duration_s": float(server_duration_s),
            "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            "decoded_latest_gap_s": decoded_latest_gap_s,
            "sdk_timestamp_gap_s": decoded_latest_gap_s,
            "data_file_age_s": data_file_age_s,
            "live_latency_s": _effective_live_latency_s(window.metadata),
            "timestamp_end_s": window.metadata.get("timestamp_end_s"),
            "estimated_data_duration_s": window.metadata.get("estimated_data_duration_s"),
        },
    }


def _effective_live_latency_s(metadata: dict[str, Any]) -> float | None:
    sdk_gap_s = _safe_float(metadata.get("sdk_latest_timestamp_gap_s"))
    if metadata.get("data_source_mode") == "sdk_decoded_stream_lagging":
        return sdk_gap_s

    file_age_s = _safe_float(metadata.get("data_file_age_s"))
    return file_age_s if file_age_s is not None else sdk_gap_s


def _validated_live_signal(signal: Any) -> Any:
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if not np.all(finite):
        arr = arr[finite]
    if arr.size == 0:
        raise ValueError("SDK vibrometer window did not contain finite samples")
    return arr


def _validated_spectral_signal(signal: Any) -> Any:
    """Validate a uniformly sampled record without changing its time base."""

    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("SDK vibrometer window did not contain samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "SDK vibrometer window contains non-finite samples; PSD was not computed because dropping them would distort the time axis"
        )
    return arr


def _validated_sampling_rate(value: Any) -> int:
    try:
        sampling_rate = float(value)
    except (TypeError, ValueError):
        raise ValueError("SDK vibrometer window reported an invalid sampling rate") from None
    if not math.isfinite(sampling_rate):
        raise ValueError("SDK vibrometer window reported an invalid sampling rate")
    sampling_rate_hz = int(round(sampling_rate))
    if sampling_rate_hz <= 0:
        raise ValueError("SDK vibrometer window reported a non-positive sampling rate")
    return sampling_rate_hz


def _resolve_psd_source(query: dict[str, list[str]]) -> dict[str, Any]:
    """Resolve one deterministic board for a PSD request.

    A PSD request must never use ``find_latest_hsd_acquisition_folder`` as its
    implicit source when several boards are logging in parallel.  The newest
    ``*.dat`` mtime moves between boards as USB writes complete, so repeated
    requests can otherwise alternate between physically different sensors.

    Source precedence is:

    1. an explicit registry ``sensor_id`` (or a registry-recognised
       ``machine_id``),
    2. an explicit ``acquisition_folder``,
    3. ``VIBRO_HSD_ACQUISITION_DIR``,
    4. the configured baseline sensor.
    """

    requested_sensor_id = _clean(_first(query, "sensor_id"))
    requested_machine_id = _clean(_first(query, "machine_id"))
    requested_folder = _clean(_first(query, "acquisition_folder"))
    requested_sensor_name = _clean(_first(query, "sensor_name"))
    config_path = _resolve_live_sensor_config_path(_first(query, "config_path"))

    registry: SensorRegistry | None = None

    def load_registry() -> SensorRegistry:
        nonlocal registry
        if registry is None:
            registry = SensorRegistry(config_path)
        return registry

    registry_sensor_id = requested_sensor_id
    if registry_sensor_id is None and requested_machine_id and requested_folder is None:
        try:
            candidate_registry = load_registry()
            canonical = candidate_registry.resolve_id(requested_machine_id)
            if canonical in candidate_registry.sensors:
                registry_sensor_id = canonical
        except Exception:
            # An explicit folder or environment folder can still provide a
            # deterministic source even when the registry is unavailable.
            pass

    if registry_sensor_id:
        try:
            selected_registry = load_registry()
        except Exception as exc:
            raise ValueError(
                f"Unable to load the live sensor registry {config_path}: {exc.__class__.__name__}: {exc}"
            ) from exc
        canonical = selected_registry.resolve_id(registry_sensor_id)
        sensor = selected_registry.sensors.get(canonical)
        if sensor is None:
            available = ", ".join(selected_registry.sensors)
            raise ValueError(
                f"Unknown PSD sensor_id {registry_sensor_id!r}. Available boards: {available}."
            )
        if not sensor.acquisition_folder:
            raise ValueError(f"PSD board {canonical!r} has no acquisition_folder in {config_path}")
        return {
            "sensor_id": sensor.sensor_id,
            "machine_id": sensor.sensor_id,
            "sensor_name": sensor.hsd_sensor_name,
            "acquisition_folder": sensor.acquisition_folder,
            "location": sensor.location,
            "role": sensor.role,
            "selection_mode": "registry_sensor",
            "config_path": str(selected_registry.config_path),
            "locked": True,
        }

    if requested_folder:
        return {
            "sensor_id": requested_machine_id,
            "machine_id": requested_machine_id or "stwinbox_iis3dwb",
            "sensor_name": requested_sensor_name,
            "acquisition_folder": requested_folder,
            "location": None,
            "role": None,
            "selection_mode": "explicit_folder",
            "config_path": None,
            "locked": True,
        }

    environment_folder = _clean(os.environ.get("VIBRO_HSD_ACQUISITION_DIR"))
    if environment_folder:
        return {
            "sensor_id": requested_machine_id,
            "machine_id": requested_machine_id or "stwinbox_iis3dwb",
            "sensor_name": requested_sensor_name,
            "acquisition_folder": environment_folder,
            "location": None,
            "role": None,
            "selection_mode": "environment_folder",
            "config_path": None,
            "locked": True,
        }

    try:
        selected_registry = load_registry()
        sensor = selected_registry.baseline()
    except Exception as exc:
        raise ValueError(
            "PSD source is ambiguous: no board/acquisition_folder was supplied, "
            f"VIBRO_HSD_ACQUISITION_DIR is unset, and the baseline registry could not be loaded: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc
    if not sensor.acquisition_folder:
        raise ValueError(
            f"PSD baseline board {sensor.sensor_id!r} has no acquisition_folder in {selected_registry.config_path}"
        )
    return {
        "sensor_id": sensor.sensor_id,
        "machine_id": sensor.sensor_id,
        "sensor_name": sensor.hsd_sensor_name,
        "acquisition_folder": sensor.acquisition_folder,
        "location": sensor.location,
        "role": sensor.role,
        "selection_mode": "registry_baseline_default",
        "config_path": str(selected_registry.config_path),
        "locked": True,
    }


def _psd_fft_payload(query: dict[str, list[str]], *, reader=None) -> dict[str, Any]:
    """Read an SDK window and return a display-ready one-sided PSD.

    ``estimator=welch`` is the webchat default because averaging overlapping
    modified periodograms gives a much more stable operator display.  The
    original full-record FFT estimate remains available as
    ``estimator=periodogram`` for maximum frequency resolution and phase.
    Optional ``filter_type`` values (highpass, lowpass, bandpass, notch,
    notch_50 and notch_60) are applied to the shared waveform/PSD signal only;
    raw acquisition data is never modified.
    """

    axis = (_first(query, "axis") or "norm").strip().lower()
    if axis not in VALID_LIVE_AXES:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2",
        }

    estimator_raw = (_first(query, "estimator") or DEFAULT_PSD_ESTIMATOR).strip().lower()
    estimator_aliases = {
        "welch": "welch",
        "averaged": "welch",
        "average": "welch",
        "periodogram": "periodogram",
        "fft": "periodogram",
        "raw": "periodogram",
        "raw_fft": "periodogram",
    }
    estimator = estimator_aliases.get(estimator_raw)
    if estimator is None:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "estimator must be welch or periodogram",
        }

    window_name = (_first(query, "window") or DEFAULT_PSD_WINDOW).strip().lower()
    try:
        # Validate the name before starting a potentially expensive SDK read.
        periodic_window(window_name, 2)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": format_exception_for_response(exc),
        }

    plot_scale = (_first(query, "plot_scale") or DEFAULT_PSD_PLOT_SCALE).strip().lower()
    if plot_scale not in {"log", "linear"}:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "plot_scale must be log or linear",
        }

    filter_raw = _first(query, "filter_type") or _first(query, "filter") or "none"
    try:
        filter_type = normalize_frequency_filter(filter_raw)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": format_exception_for_response(exc),
        }
    low_cutoff_hz = _clean(_first(query, "low_cutoff_hz"))
    high_cutoff_hz = _clean(_first(query, "high_cutoff_hz"))
    notch_frequency_hz = _clean(_first(query, "notch_frequency_hz"))
    notch_width_hz = _clean(_first(query, "notch_width_hz"))
    filter_order = _clean(_first(query, "filter_order")) or "4"

    try:
        source = _resolve_psd_source(query)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": format_exception_for_response(exc),
        }
    sensor_name = source.get("sensor_name")
    machine_id = str(source.get("machine_id") or "stwinbox_iis3dwb")
    acquisition_folder = source.get("acquisition_folder")
    duration_s = min(
        MAX_PSD_FFT_WINDOW_S,
        max(0.1, _float_query(query, "duration_s", DEFAULT_PSD_FFT_WINDOW_S)),
    )
    start_time_raw = _clean(_first(query, "start_time_s"))
    max_bins = min(MAX_PSD_FFT_BINS, max(64, _int_query(query, "max_bins", DEFAULT_PSD_FFT_MAX_BINS)))
    waveform_max_points = min(
        MAX_PSD_WAVEFORM_POINTS,
        max(64, _int_query(query, "waveform_max_points", DEFAULT_PSD_WAVEFORM_MAX_POINTS)),
    )
    welch_resolution_hz = max(
        0.001,
        _float_query(query, "welch_resolution_hz", DEFAULT_WELCH_RESOLUTION_HZ),
    )
    overlap_fraction = min(
        0.9,
        max(0.0, _float_query(query, "overlap_fraction", DEFAULT_WELCH_OVERLAP)),
    )
    require_current = _bool_query(query, "require_current", False)
    try:
        replay_controller = offline_replay_from_environment()
        replay_snapshot = replay_controller.snapshot() if replay_controller else None
        start_time_s = (
            float(start_time_raw)
            if start_time_raw is not None
            else replay_controller.window_start(duration_s) if replay_controller else None
        )
    except (OfflineReplayConfigurationError, ValueError) as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "replay_configuration_failed",
            "message": format_exception_for_response(exc),
        }
    if replay_controller:
        require_current = False
    process_reader = _bool_query(query, "process_reader", False)
    request_started_s = time.perf_counter()
    reader_fn = reader or (
        read_sdk_vibrometer_window_via_board_process
        if board_reader_processes_enabled(default=process_reader)
        else read_sdk_vibrometer_window
    )

    try:
        decode_started_s = time.perf_counter()
        window = reader_fn(
            machine_id=machine_id,
            acquisition_folder=acquisition_folder,
            sensor_name=sensor_name,
            axis=axis,
            start_time_s=start_time_s,
            duration_s=duration_s,
            require_current=require_current,
            max_duration_s=MAX_PSD_FFT_WINDOW_S,
        )
        decode_duration_s = time.perf_counter() - decode_started_s
    except LiveSdkVibrometerDataRequiredError as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "live_data_required",
            "message": format_exception_for_response(exc),
            "metadata": exc.metadata,
            "diagnostics": {
                "decode_duration_s": elapsed_s,
                "server_duration_s": elapsed_s,
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
                "decoded_latest_gap_s": exc.metadata.get("sdk_latest_timestamp_gap_s"),
                "sdk_timestamp_gap_s": exc.metadata.get("sdk_latest_timestamp_gap_s"),
                "data_file_age_s": exc.metadata.get("data_file_age_s"),
                "live_latency_s": _effective_live_latency_s(exc.metadata),
            },
        }
    except Exception as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "read_failed",
            "message": format_exception_for_response(exc),
            "diagnostics": {
                "decode_duration_s": elapsed_s,
                "server_duration_s": elapsed_s,
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            },
        }

    try:
        import numpy as np

        signal = _validated_spectral_signal(window.signal)
        sampling_rate_hz = _validated_sampling_rate(window.sampling_rate_hz)
        sample_count = int(signal.size)
        if sample_count < 2:
            raise ValueError("SDK vibrometer window needs at least two finite samples for PSD")

        filter_started_s = time.perf_counter()
        try:
            analysis_signal, filter_metadata = apply_frequency_filter(
                signal,
                sampling_rate_hz,
                filter_type=filter_type,
                low_cutoff_hz=low_cutoff_hz,
                high_cutoff_hz=high_cutoff_hz,
                notch_frequency_hz=notch_frequency_hz,
                notch_width_hz=notch_width_hz,
                filter_order=filter_order,
            )
        except ValueError as exc:
            elapsed_s = time.perf_counter() - request_started_s
            return {
                "ok": False,
                "status": "error",
                "error": "bad_request",
                "message": format_exception_for_response(exc),
                "metadata": window.metadata,
                "diagnostics": {
                    "decode_duration_s": float(decode_duration_s),
                    "filter_duration_s": float(time.perf_counter() - filter_started_s),
                    "server_duration_s": float(elapsed_s),
                    "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
                },
            }
        filter_duration_s = time.perf_counter() - filter_started_s

        spectral_started_s = time.perf_counter()
        if estimator == "welch":
            estimate = compute_welch_psd(
                analysis_signal,
                sampling_rate_hz,
                window_name=window_name,
                target_resolution_hz=welch_resolution_hz,
                overlap_fraction=overlap_fraction,
            )
        else:
            estimate = compute_periodogram_psd(analysis_signal, sampling_rate_hz, window_name=window_name)
        spectral_duration_s = time.perf_counter() - spectral_started_s

        freqs = estimate.frequencies_hz
        psd = estimate.psd_g2_per_hz
        plot = aggregate_spectrum_for_plot(
            freqs,
            psd,
            max_bins=max_bins,
            frequency_scale=plot_scale,
        )
        top_peaks = find_distinct_peaks(
            freqs,
            psd,
            limit=PSD_FFT_TOP_PEAKS,
            reference_fft=estimate.reference_fft,
            reference_fft_frequencies_hz=estimate.reference_fft_frequencies_hz,
        )

        impact_started_s = time.perf_counter()
        full_scale_g = _safe_float(window.metadata.get("full_scale_g"))
        clipping_suspected = bool(
            axis not in {"norm", "magnitude", "vector_norm"}
            and full_scale_g is not None
            and full_scale_g > 0.0
            and float(np.max(np.abs(signal))) >= 0.98 * full_scale_g
        )
        impact_analysis = analyze_impact_ringdown(
            analysis_signal,
            sampling_rate_hz,
            window_name=window_name,
            minimum_frequency_hz=0.5,
            maximum_frequency_hz=min(200.0, 0.5 * sampling_rate_hz),
            clipping_suspected=clipping_suspected,
        )
        impact_frequencies = np.asarray(
            impact_analysis.pop("frequencies_hz", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        )
        impact_psd = np.asarray(
            impact_analysis.pop("psd_g2_per_hz", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        )
        if impact_frequencies.size and impact_psd.size == impact_frequencies.size:
            impact_plot = aggregate_spectrum_for_plot(
                impact_frequencies,
                impact_psd,
                max_bins=max_bins,
                frequency_scale=plot_scale,
            )
        else:
            impact_plot = {
                "peak_frequency_hz": np.asarray([], dtype=np.float64),
                "peak_psd_g2_per_hz": np.asarray([], dtype=np.float64),
            }
        event_index = impact_analysis.get("event_index")
        if event_index is not None and window.timestamps_s is not None:
            timestamp_values = np.asarray(window.timestamps_s, dtype=np.float64).reshape(-1)
            event_index_int = int(event_index)
            if (
                timestamp_values.size == sample_count
                and 0 <= event_index_int < timestamp_values.size
                and np.all(np.isfinite(timestamp_values[[0, event_index_int]]))
            ):
                impact_analysis["event_time_s"] = float(
                    timestamp_values[event_index_int] - timestamp_values[0]
                )
        impact_analysis["full_scale_g"] = full_scale_g
        impact_analysis["axis"] = axis
        impact_analysis_duration_s = time.perf_counter() - impact_started_s

        centered = analysis_signal - float(np.mean(analysis_signal))
        centered_variance_g2 = float(np.mean(centered * centered))
        centered_rms_g = float(np.sqrt(max(centered_variance_g2, 0.0)))
        # For the mean-removed acceleration signal used by the spectrum, the
        # integral of the one-sided PSD over frequency is the signal variance
        # in g^2.  Keep ``integrated_psd_power_g2`` in the response for older
        # clients and expose the clearer ``psd_variance_g2`` name below.
        psd_variance_g2 = integrated_psd_power(freqs, psd)
        psd_rms_g = float(np.sqrt(max(psd_variance_g2, 0.0)))
        consistency_ratio = psd_rms_g / centered_rms_g if centered_rms_g > 0 else None
        variance_consistency_ratio = (
            psd_variance_g2 / centered_variance_g2 if centered_variance_g2 > 0 else None
        )
        waveform = aggregate_waveform_for_plot(
            analysis_signal,
            sampling_rate_hz,
            timestamps_s=window.timestamps_s,
            max_points=waveform_max_points,
            center=True,
        )
        plot_floor = plot["psd_g2_per_hz"]
        plot_peaks = plot["peak_psd_g2_per_hz"]
        tiny = np.finfo(np.float64).tiny
        raw_centered = signal - float(np.mean(signal))
        raw_centered_variance_g2 = float(np.mean(raw_centered * raw_centered))
        stats = {
            "min_g": float(signal.min()),
            "max_g": float(signal.max()),
            "mean_g": float(signal.mean()),
            "rms_g": float(np.sqrt(np.mean(signal * signal))),
            "centered_rms_g": float(np.sqrt(max(raw_centered_variance_g2, 0.0))),
            "variance_g2": raw_centered_variance_g2,
            "standard_deviation_g": float(np.std(signal)),
        }
        analysis_stats = {
            "min_g": float(analysis_signal.min()),
            "max_g": float(analysis_signal.max()),
            "mean_g": float(analysis_signal.mean()),
            "rms_g": float(np.sqrt(np.mean(analysis_signal * analysis_signal))),
            "centered_rms_g": centered_rms_g,
            "variance_g2": centered_variance_g2,
            "standard_deviation_g": float(np.std(analysis_signal)),
        }
    except ValueError as exc:
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "invalid_window",
            "message": format_exception_for_response(exc),
            "metadata": window.metadata,
            "diagnostics": {
                "decode_duration_s": float(decode_duration_s),
                "server_duration_s": float(elapsed_s),
                "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            },
        }

    server_duration_s = time.perf_counter() - request_started_s
    peak = top_peaks[0] if top_peaks else {}
    peak_lead_db: float | None = None
    competing_peak_frequency_hz: float | None = None
    if len(top_peaks) >= 2:
        first_db = _safe_float(top_peaks[0].get("psd_db_re_1_g2_per_hz"))
        second_db = _safe_float(top_peaks[1].get("psd_db_re_1_g2_per_hz"))
        if first_db is not None and second_db is not None:
            # Refined peak powers can reverse by a few hundredths of a dB
            # even though the source bins were ranked correctly.  Report the
            # level separation rather than claiming the second peak is lower.
            peak_lead_db = abs(first_db - second_db)
            competing_peak_frequency_hz = _safe_float(top_peaks[1].get("frequency_hz"))

    impact_peak_frequency_hz = _safe_float(impact_analysis.get("peak_frequency_hz"))
    impact_peak_psd_g2_per_hz = _safe_float(impact_analysis.get("peak_psd_g2_per_hz"))
    impact_peak_psd_db = _safe_float(impact_analysis.get("peak_psd_db_re_1_g2_per_hz"))
    steady_peak_frequency_hz = _safe_float(peak.get("frequency_hz"))
    if steady_peak_frequency_hz is not None:
        effective_peak_frequency_hz = steady_peak_frequency_hz
        effective_peak_kind = "averaged_psd"
        effective_peak_reliable = True
    elif impact_peak_frequency_hz is not None:
        effective_peak_frequency_hz = impact_peak_frequency_hz
        effective_peak_kind = "impact_ringdown"
        effective_peak_reliable = bool(impact_analysis.get("peak_reliable"))
    else:
        effective_peak_frequency_hz = None
        effective_peak_kind = None
        effective_peak_reliable = False

    decoded_latest_gap_s = window.metadata.get("sdk_latest_timestamp_gap_s")
    data_file_age_s = window.metadata.get("data_file_age_s")
    estimator_label = "Welch averaged PSD" if estimator == "welch" else "full-record FFT periodogram"
    axis_warning = (
        "Magnitude/norm is a nonlinear overview and can create or shift harmonics; "
        "use X, Y, or Z for diagnostic frequency interpretation."
        if axis in {"norm", "magnitude", "vector_norm"}
        else None
    )
    actual_machine_id = str(window.metadata.get("machine_id") or machine_id)
    actual_sensor_name = window.metadata.get("sensor_name") or sensor_name
    actual_acquisition_folder = window.metadata.get("acquisition_folder") or acquisition_folder
    source_payload = {
        **source,
        "machine_id": actual_machine_id,
        "sensor_name": actual_sensor_name,
        "acquisition_folder": actual_acquisition_folder,
    }
    return {
        "ok": True,
        "status": "ok",
        "machine_id": actual_machine_id,
        "sensor_name": actual_sensor_name,
        "acquisition_folder": actual_acquisition_folder,
        "replay": replay_snapshot,
        "source": source_payload,
        "axis": axis,
        "axis_warning": axis_warning,
        "estimator": estimator,
        "estimator_label": estimator_label,
        "window_name": window_name,
        "plot_scale": plot_scale,
        "filter_requested": filter_type,
        "filter": filter_metadata,
        "window_duration_s": float(sample_count / float(sampling_rate_hz)),
        "requested_duration_s": float(duration_s),
        "sample_count": sample_count,
        "sampling_rate_hz": int(sampling_rate_hz),
        "record_length": sample_count,
        "record_resolution_hz": float(estimate.record_resolution_hz),
        # Legacy names retained for older web clients. For Welch, fft_length and
        # fft_resolution_hz describe each averaged segment, not the full record.
        "fft_length": int(estimate.segment_length),
        "fft_resolution_hz": float(estimate.resolution_hz),
        "segment_length": int(estimate.segment_length),
        "segment_duration_s": float(estimate.segment_duration_s),
        "segment_count": int(estimate.segment_count),
        "overlap_samples": int(estimate.overlap_samples),
        "overlap_fraction": float(estimate.overlap_fraction),
        "nyquist_hz": float(sampling_rate_hz / 2.0),
        "psd_units": "g^2/Hz",
        "asd_units": "g/sqrt(Hz)",
        "variance_units": "g^2",
        "psd_variance_definition": (
            "Integral of the one-sided PSD from 0 Hz to Nyquist after the selected "
            "analysis filter and mean removal"
        ),
        "preprocessing": {
            "analysis_window": f"periodic {window_name}",
            "detrend": "constant per Welch segment" if estimator == "welch" else "constant over full record",
            "frequency_filter": "none" if not filter_metadata["applied"] else filter_metadata["label"],
            "frequency_filter_type": filter_metadata["type"],
            "filter_phase_response": filter_metadata["phase_response"],
            "filter_scope": filter_metadata["scope"],
            "signal_decimation_before_psd": False,
            "waveform_display": "centered min/max/mean envelope",
        },
        "waveform_time_s": [float(value) for value in waveform["time_s"]],
        "waveform_min_g": [float(value) for value in waveform["min_g"]],
        "waveform_max_g": [float(value) for value in waveform["max_g"]],
        "waveform_mean_g": [float(value) for value in waveform["mean_g"]],
        "waveform_points": int(waveform["point_count"]),
        "waveform_source_samples": int(waveform["source_sample_count"]),
        "waveform_downsampled": bool(waveform["downsampled"]),
        "waveform_centered": bool(waveform["centered"]),
        "waveform_timestamp_source": str(waveform["timestamp_source"]),
        "waveform_display_method": str(waveform["display_method"]),
        "frequency_hz": [float(value) for value in plot["frequency_hz"]],
        "psd_g2_per_hz": [float(value) for value in plot_floor],
        "psd_db_re_1_g2_per_hz": [float(10.0 * np.log10(max(float(value), tiny))) for value in plot_floor],
        "asd_g_per_sqrt_hz": [float(np.sqrt(max(float(value), 0.0))) for value in plot_floor],
        "peak_envelope_frequency_hz": [float(value) for value in plot["peak_frequency_hz"]],
        "peak_envelope_psd_g2_per_hz": [float(value) for value in plot_peaks],
        "peak_envelope_db_re_1_g2_per_hz": [
            float(10.0 * np.log10(max(float(value), tiny))) for value in plot_peaks
        ],
        "points_returned": int(plot["frequency_hz"].size),
        "spectrum_points": int(freqs.size),
        "downsampled": bool(plot["frequency_hz"].size != freqs.size),
        "peak_frequency_hz": peak.get("frequency_hz"),
        "peak_psd_g2_per_hz": peak.get("psd_g2_per_hz"),
        "peak_psd_db_re_1_g2_per_hz": peak.get("psd_db_re_1_g2_per_hz"),
        "peak_asd_g_per_sqrt_hz": peak.get("asd_g_per_sqrt_hz"),
        "peak_phase_degrees": peak.get("phase_degrees"),
        "peak_lead_db": peak_lead_db,
        "peak_level_separation_db": peak_lead_db,
        "competing_peak_frequency_hz": competing_peak_frequency_hz,
        "peak_is_competing": bool(peak_lead_db is not None and peak_lead_db < 3.0),
        "top_peaks": top_peaks,
        "effective_peak_frequency_hz": effective_peak_frequency_hz,
        "effective_peak_kind": effective_peak_kind,
        "effective_peak_reliable": effective_peak_reliable,
        "impact_peak_frequency_hz": impact_peak_frequency_hz,
        "impact_peak_psd_g2_per_hz": impact_peak_psd_g2_per_hz,
        "impact_peak_psd_db_re_1_g2_per_hz": impact_peak_psd_db,
        "impact_analysis": impact_analysis,
        "impact_spectrum_frequency_hz": [
            float(value) for value in impact_plot["peak_frequency_hz"]
        ],
        "impact_spectrum_psd_g2_per_hz": [
            float(value) for value in impact_plot["peak_psd_g2_per_hz"]
        ],
        "impact_spectrum_db_re_1_g2_per_hz": [
            float(10.0 * np.log10(max(float(value), tiny)))
            for value in impact_plot["peak_psd_g2_per_hz"]
        ],
        "integrated_psd_power_g2": float(psd_variance_g2),
        "psd_variance_g2": float(psd_variance_g2),
        "psd_rms_g": psd_rms_g,
        "centered_rms_g": centered_rms_g,
        "centered_variance_g2": centered_variance_g2,
        "power_consistency_ratio": float(consistency_ratio) if consistency_ratio is not None else None,
        "variance_consistency_ratio": (
            float(variance_consistency_ratio) if variance_consistency_ratio is not None else None
        ),
        "frequency_bands": frequency_band_summary(freqs, psd),
        "stats": stats,
        "analysis_stats": analysis_stats,
        "metadata": window.metadata,
        "diagnostics": {
            "decode_duration_s": float(decode_duration_s),
            "filter_duration_s": float(filter_duration_s),
            "fft_duration_s": float(spectral_duration_s),
            "spectral_duration_s": float(spectral_duration_s),
            "impact_analysis_duration_s": float(impact_analysis_duration_s),
            "server_duration_s": float(server_duration_s),
            "board_reader_process": bool(reader_fn is read_sdk_vibrometer_window_via_board_process),
            "decoded_latest_gap_s": decoded_latest_gap_s,
            "sdk_timestamp_gap_s": decoded_latest_gap_s,
            "data_file_age_s": data_file_age_s,
            "live_latency_s": _effective_live_latency_s(window.metadata),
            "timestamp_end_s": window.metadata.get("timestamp_end_s"),
            "estimated_data_duration_s": window.metadata.get("estimated_data_duration_s"),
        },
    }


def _compute_one_sided_fft_psd(
    signal: Any,
    sampling_rate_hz: int,
    *,
    window_name: str = "hann",
) -> tuple[Any, Any, Any]:
    """Backward-compatible wrapper for the corrected raw periodogram path."""

    estimate = compute_periodogram_psd(signal, sampling_rate_hz, window_name=window_name)
    return estimate.frequencies_hz, estimate.psd_g2_per_hz, estimate.reference_fft


def _fft_analysis_window(window_name: str, n: int) -> Any:
    """Backward-compatible window helper used by existing tests/callers."""

    return periodic_window(window_name, n)


def _downsample_spectrum_for_plot(freqs: Any, psd: Any, *, max_bins: int) -> tuple[Any, Any]:
    """Return the median spectral floor per bucket rather than connected maxima."""

    plot = aggregate_spectrum_for_plot(freqs, psd, max_bins=max_bins, frequency_scale="linear")
    return plot["frequency_hz"], plot["psd_g2_per_hz"]


def _spectrum_top_peaks(freqs: Any, psd: Any, fft_positive: Any, *, limit: int) -> list[dict[str, Any]]:
    """Return separated local maxima instead of adjacent bins from one lobe."""

    return find_distinct_peaks(
        freqs,
        psd,
        limit=limit,
        reference_fft=fft_positive,
        reference_fft_frequencies_hz=freqs,
    )


def _live_sensors_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    try:
        config_path = _resolve_live_sensor_config_path(_first(query, "config_path"))
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": str(exc),
        }
    try:
        registry = SensorRegistry(config_path)
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "sensor_registry_failed",
            "message": format_exception_for_response(exc),
            "config_path": str(config_path),
        }

    sensors = [
        {
            "sensor_id": sensor.sensor_id,
            "location": sensor.location,
            "role": sensor.role,
            "acquisition_folder": sensor.acquisition_folder,
            "hsd_sensor_name": sensor.hsd_sensor_name,
            "axis": sensor.axis,
        }
        for sensor in registry.sensors.values()
    ]
    try:
        replay_controller = offline_replay_from_environment()
        replay_snapshot = replay_controller.snapshot() if replay_controller else None
    except OfflineReplayConfigurationError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "replay_configuration_failed",
            "message": format_exception_for_response(exc),
        }
    # Lightweight acquisition status for views that do not decode waveforms.
    # This describes file freshness, not model verdict or decoded-signal quality.
    max_age_s = _current_file_max_age_s()
    for sensor in sensors:
        stream = {"data_is_current": False, "data_file_age_s": None,
                  "data_currentness_max_age_s": max_age_s}
        folder, name = sensor["acquisition_folder"], sensor["hsd_sensor_name"]
        if folder and name and Path(name).name == name:
            try:
                info = (Path(folder) / f"{name}.dat").stat()
                age_s = time.time() - info.st_mtime
                stream["data_file_age_s"] = age_s
                stream["data_is_current"] = bool(
                    stat.S_ISREG(info.st_mode) and info.st_size > 0
                    and 0 <= age_s <= max_age_s
                )
            except OSError:
                pass
        sensor["stream"] = stream
    process_reader = _bool_query(query, "process_reader", False)
    prewarm = _bool_query(query, "prewarm", process_reader)
    process_enabled = board_reader_processes_enabled(default=process_reader)
    process_statuses = prewarm_board_reader_processes(sensors) if process_enabled and prewarm else board_reader_process_statuses()

    return {
        "ok": True,
        "status": "ok",
        "config_path": str(registry.config_path),
        "baseline_sensor_id": registry.baseline_sensor_id,
        "sensors": sensors,
        "replay": replay_snapshot,
        "board_reader_process_enabled": bool(process_enabled),
        "board_reader_processes": process_statuses,
    }


def _agent_monitor_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    axis = (_first(query, "axis") or "norm").strip().lower()
    if axis not in VALID_LIVE_AXES:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": "axis must be one of: norm, magnitude, vector_norm, x, y, z, 0, 1, 2",
        }

    try:
        config_path = _resolve_live_sensor_config_path(_first(query, "config_path"))
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": str(exc),
        }
    baseline_sensor_id = _clean(_first(query, "baseline_sensor_id")) or "baseline"
    sensor_ids = _clean(_first(query, "sensor_ids"))
    duration_s = min(10.0, max(0.2, _float_query(query, "duration_s", 1.0)))
    require_current = _bool_query(query, "require_current", True)
    require_llm = _bool_query(query, "require_llm", True)
    use_llm = _bool_query(query, "use_llm", require_llm)
    process_reader = _bool_query(query, "process_reader", False)
    skip_cache = _bool_query(query, "skip_cache", False)
    raw_vote_k = _int_query(query, "vote_k", 0)
    vote_k = max(1, min(10, raw_vote_k)) if raw_vote_k > 0 else None
    default_model_timeout_s = _env_float_value(
        "AGENT_MONITOR_MODEL_TIMEOUT_S",
        DEFAULT_MONITOR_MODEL_TIMEOUT_S,
        minimum=1.0,
    )
    model_timeout_s = min(
        MAX_AGENT_MODEL_TIMEOUT_S,
        max(1.0, _float_query(query, "model_timeout_s", default_model_timeout_s)),
    )
    if not use_llm:
        require_llm = False
    try:
        model_base_url = _request_model_base_url(
            _clean(_first(query, "base_url")),
            default=os.environ.get("MAIN_AGENT_BASE_URL") or os.environ.get("QWEN_BASE_URL") or DEFAULT_BASE_URL,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "bad_request",
            "message": str(exc),
        }
    model_api_key = (
        _clean(_first(query, "api_key"))
        or os.environ.get("MAIN_AGENT_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or "EMPTY"
    )
    requested_model = _clean(_first(query, "model")) or os.environ.get("MAIN_AGENT_MODEL") or os.environ.get("QWEN_MODEL") or DEFAULT_MODEL
    model = requested_model
    request_started_s = time.perf_counter()
    analysis_start_time_s: float | None = None
    replay_controller = None
    replay_claim = None
    replay_snapshot = None
    try:
        replay_controller = offline_replay_from_environment()
        if replay_controller:
            use_llm = True
            require_llm = True
            require_current = False
            replay_snapshot = replay_controller.snapshot()
            if _chat_is_inflight():
                paused = _agent_monitor_paused_payload(request_started_s)
                paused["replay"] = replay_snapshot
                return paused
            replay_claim = replay_controller.claim_due_inference()
            if replay_claim is None:
                return {
                    "ok": True,
                    "status": "waiting",
                    "waiting_for_replay": True,
                    "anomaly": False,
                    "agent": {
                        "llm_agent_active": False,
                        "waiting_for_replay": True,
                    },
                    "replay": replay_snapshot,
                    "diagnostics": {
                        "server_duration_s": time.perf_counter() - request_started_s,
                    },
                }
            analysis_start_time_s = replay_claim.window.start_time_s
            duration_s = min(10.0, max(0.2, replay_claim.window.duration_s))
    except OfflineReplayConfigurationError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": "replay_configuration_failed",
            "message": format_exception_for_response(exc),
        }
    process_enabled = board_reader_processes_enabled(default=process_reader)
    cache_key = _agent_monitor_cache_key(
        config_path=config_path,
        baseline_sensor_id=baseline_sensor_id,
        sensor_ids=sensor_ids,
        axis=axis,
        duration_s=duration_s,
        start_time_s=analysis_start_time_s,
        require_current=require_current,
        use_llm=use_llm,
        model_base_url=model_base_url,
        model=requested_model,
        process_reader=process_enabled,
        model_timeout_s=model_timeout_s,
    )

    if use_llm and not skip_cache:
        cached = _cached_agent_monitor_payload(cache_key, request_started_s=request_started_s)
        if cached is not None:
            return cached

    # Yield the NPU to an in-flight chat (any tab/client): starting an LLM check now would queue on
    # the single serialized NPU and starve the chat. Serve the most recent cached result if any,
    # else a benign "paused" payload that the client renders as a no-op.
    if use_llm and _chat_is_inflight():
        cached = _cached_agent_monitor_payload(cache_key, request_started_s=request_started_s)
        if cached is not None:
            return cached
        if replay_controller and replay_claim:
            replay_controller.release_claim(replay_claim)
        return _agent_monitor_paused_payload(request_started_s)

    try:
        result = run_building_vibration_agent_pipeline_service(
            config_path=str(config_path),
            baseline_sensor_id=baseline_sensor_id,
            sensor_ids=sensor_ids,
            start_time_s=analysis_start_time_s,
            duration_s=duration_s,
            axis=axis,
            require_current=require_current,
            # The monitor path uses the deterministic feature prepass so popup
            # checks stay fast and reliable. A single compact monitor LLM call
            # may be layered on top below when requested.
            use_model_agents=False,
            process_reader=process_enabled,
            model_base_url=None,
            model_api_key=None,
            model=None,
            model_timeout_s=None,
        )
    except LiveSdkVibrometerDataRequiredError as exc:
        if replay_controller and replay_claim:
            replay_controller.release_claim(replay_claim)
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "live_data_required",
            "message": format_exception_for_response(exc),
            "metadata": exc.metadata,
            "diagnostics": {"server_duration_s": elapsed_s},
        }
    except Exception as exc:
        if replay_controller and replay_claim:
            replay_controller.release_claim(replay_claim)
        elapsed_s = time.perf_counter() - request_started_s
        return {
            "ok": False,
            "status": "error",
            "error": "agent_monitor_failed",
            "message": format_exception_for_response(exc),
            "diagnostics": {"server_duration_s": elapsed_s},
        }

    # Score this window against the building's own recorded normal BEFORE the
    # decision (the prompt and popup consume the scores) and before the window
    # itself is appended to the history.
    _attach_window_history_context(result)

    lock_acquired = False
    try:
        if use_llm:
            lock_acquired = _AGENT_MONITOR_LOCK.acquire(blocking=False)
            if not lock_acquired:
                if not skip_cache:
                    cached = _cached_agent_monitor_payload(cache_key, request_started_s=request_started_s)
                    if cached is not None:
                        return cached
                _AGENT_MONITOR_LOCK.acquire()
                lock_acquired = True
                if not skip_cache:
                    cached = _cached_agent_monitor_payload(cache_key, request_started_s=request_started_s)
                    if cached is not None:
                        return cached

            model = _auto_select_agent_model(
                base_url=model_base_url,
                api_key=model_api_key,
                requested_model=requested_model,
            )
            result = _apply_monitor_llm_decision(
                result,
                model_base_url=model_base_url,
                model_api_key=model_api_key,
                model=model,
                model_timeout_s=model_timeout_s,
                vote_k=vote_k,
                config_path=str(config_path),
                process_reader=process_enabled,
                start_time_s=analysis_start_time_s,
                require_current=require_current,
                replay_labels=(replay_claim.window.labels if replay_claim else ()),
            )
        else:
            result = _mark_monitor_llm_skipped(result)

        agent_status = _agent_llm_status(
            result,
            model_base_url=model_base_url,
            model=model,
            require_llm=require_llm,
        )
        if require_llm and not agent_status["llm_agent_active"]:
            if replay_controller and replay_claim:
                replay_controller.release_claim(replay_claim)
            return {
                "ok": False,
                "status": "error",
                "error": "llm_agent_unavailable",
                "message": "Qwen monitor LLM is required for autonomous monitoring, but the compact monitor decision fell back.",
                "agent": agent_status,
                "result": result,
                "diagnostics": {
                    "server_duration_s": time.perf_counter() - request_started_s,
                    "model_timeout_s": model_timeout_s,
                },
            }

        popup = _build_agent_popup(result)
        _record_window_history(result, popup)
        anomaly_registration = _register_anomaly_window(
            result=result,
            popup=popup,
            agent_status=agent_status,
            monitor_request={
                "config_path": str(config_path),
                "baseline_sensor_id": baseline_sensor_id,
                "sensor_ids": sensor_ids,
                "axis": axis,
                "start_time_s": analysis_start_time_s,
                "duration_s": duration_s,
                "require_current": require_current,
                "use_llm": use_llm,
                "require_llm": require_llm,
                "model_base_url": model_base_url,
                "model": model,
                "process_reader": process_enabled,
                "model_timeout_s": model_timeout_s,
                "skip_cache": skip_cache,
            },
        )
        payload = {
            "ok": True,
            "status": "ok",
            "replay": (
                {**replay_snapshot, "inference": replay_claim.as_dict()}
                if replay_snapshot is not None and replay_claim is not None else None
            ),
            "agent": agent_status,
            "anomaly": bool(popup["show_popup"]),
            "popup": popup,
            "anomaly_registration": anomaly_registration,
            "result": result,
            "diagnostics": {
                "server_duration_s": time.perf_counter() - request_started_s,
                "model_timeout_s": model_timeout_s,
            },
        }
        if use_llm and not skip_cache:
            _cache_agent_monitor_payload(cache_key, payload)
        return payload
    finally:
        if lock_acquired:
            _AGENT_MONITOR_LOCK.release()


_AGENT_MODEL_CACHE: dict[tuple[str, str], str] = {}
_AGENT_MONITOR_LOCK = threading.Lock()
_AGENT_MONITOR_CACHE_MAX_AGE_S = 90.0
_AGENT_MONITOR_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_ANOMALY_WINDOW_REGISTRY_LOCK = threading.Lock()

# Server-side "a chat is using the NPU" guard. One chat request (npu-chat or mcp chat) owns the
# NPU at a time: LLM agent-monitor checks yield to it, and a second concurrent chat is rejected
# up front with an explicit "busy" answer — on the ~11 tok/s single-flight NPU it would otherwise
# queue invisibly behind the first chat's entire multi-step run and time out minutes later.
_CHAT_INFLIGHT_LOCK = threading.Lock()
_CHAT_INFLIGHT_COUNT = 0


def _chat_inflight_try_begin() -> bool:
    global _CHAT_INFLIGHT_COUNT
    with _CHAT_INFLIGHT_LOCK:
        if _CHAT_INFLIGHT_COUNT > 0:
            return False
        _CHAT_INFLIGHT_COUNT += 1
        return True


def _chat_busy_response() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "chat_busy",
        "message": (
            "Another chat is already running on the NPU. "
            "It answers one request at a time — wait for the current answer, then retry."
        ),
        "fallback_generated": False,
        "model_explanation_only": True,
    }


def _chat_inflight_end() -> None:
    global _CHAT_INFLIGHT_COUNT
    with _CHAT_INFLIGHT_LOCK:
        if _CHAT_INFLIGHT_COUNT > 0:
            _CHAT_INFLIGHT_COUNT -= 1


def _chat_is_inflight() -> bool:
    with _CHAT_INFLIGHT_LOCK:
        return _CHAT_INFLIGHT_COUNT > 0


def _agent_monitor_paused_payload(request_started_s: float) -> dict[str, Any]:
    # Returned when an LLM agent check is skipped because a chat is using the NPU. ok:True with a
    # paused marker so the client skips rendering — it neither errors nor clobbers the agent panel.
    return {
        "ok": True,
        "status": "paused",
        "paused_for_chat": True,
        "anomaly": False,
        "agent": {"llm_agent_active": False, "paused_for_chat": True},
        "diagnostics": {"server_duration_s": time.perf_counter() - request_started_s},
    }


def _agent_monitor_cache_key(
    *,
    config_path: Path,
    baseline_sensor_id: str,
    sensor_ids: str | None,
    axis: str,
    duration_s: float,
    start_time_s: float | None = None,
    require_current: bool,
    use_llm: bool,
    model_base_url: str,
    model: str,
    process_reader: bool = False,
    model_timeout_s: float = 8.0,
) -> tuple[Any, ...]:
    return (
        str(config_path),
        baseline_sensor_id,
        sensor_ids or "",
        axis,
        round(float(duration_s), 3),
        round(float(start_time_s), 3) if start_time_s is not None else None,
        bool(require_current),
        bool(use_llm),
        bool(process_reader),
        round(float(model_timeout_s), 3),
        model_base_url,
        model,
    )


def _cached_agent_monitor_payload(
    key: tuple[Any, ...],
    *,
    request_started_s: float,
) -> dict[str, Any] | None:
    cached = _AGENT_MONITOR_CACHE.get(key)
    if cached is None:
        return None
    cached_at_s, payload = cached
    age_s = time.monotonic() - cached_at_s
    if age_s > _AGENT_MONITOR_CACHE_MAX_AGE_S:
        return None
    result = copy.deepcopy(payload)
    result["cached"] = True
    agent = result.setdefault("agent", {})
    if isinstance(agent, dict):
        agent["served_from_cache"] = True
        agent["monitor_busy"] = True
    diagnostics = result.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics["server_duration_s"] = time.perf_counter() - request_started_s
        diagnostics["cached_age_s"] = float(age_s)
    return result


def _cache_agent_monitor_payload(key: tuple[Any, ...], payload: dict[str, Any]) -> None:
    if payload.get("ok") is True:
        _AGENT_MONITOR_CACHE[key] = (time.monotonic(), copy.deepcopy(payload))


def _register_anomaly_window(
    *,
    result: dict[str, Any],
    popup: dict[str, Any],
    agent_status: dict[str, Any],
    monitor_request: dict[str, Any],
) -> dict[str, Any]:
    model_input = result.pop("_vibrogemma_replay_input", None)
    model_input_sha256 = result.pop("_vibrogemma_replay_input_sha256", None)
    encoder_input = result.pop("_vibrogemma_encoder_replay_input", None)
    if popup.get("synthetic_test"):
        return {"registered": False, "reason": "synthetic_test"}
    if not popup.get("save_replay", popup.get("show_popup")):
        return {"registered": False, "reason": "not_anomaly"}
    if not _env_bool("VIBRO_REGISTER_ANOMALY_WINDOWS", True):
        return {"registered": False, "reason": "disabled"}

    try:
        registry_dir = _resolve_anomaly_window_dir()
        registry_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        registry_dir.chmod(0o700)
        details_dir = registry_dir / "windows"
        details_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        details_dir.chmod(0o700)

        assessment = result.get("network_assessment") or {}
        raw_window_id = str(result.get("window_id") or assessment.get("window_id") or "").strip()
        window_id = raw_window_id or f"anomaly-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{os.urandom(4).hex()}"
        safe_window_id = _safe_anomaly_window_id(window_id)
        detail_path = details_dir / f"{safe_window_id}.json"
        model_input_path = details_dir / f"{safe_window_id}.model-input.json"
        encoder_input_path = details_dir / f"{safe_window_id}.encoder-input.npz"
        jsonl_path = registry_dir / "anomaly_windows.jsonl"
        created_at = datetime.now(timezone.utc).isoformat()

        record = {
            "registration_schema_version": "0.1.0",
            "created_at_utc": created_at,
            "window_id": window_id,
            "analysis_domain": result.get("analysis_domain") or assessment.get("analysis_domain"),
            "network_label": popup.get("network_label") or assessment.get("network_label"),
            "severity": popup.get("severity"),
            "confidence": popup.get("confidence"),
            "label_uncertain": bool(popup.get("label_uncertain")),
            "vote_agreement": popup.get("vote_agreement"),
            "descriptor": popup.get("descriptor"),
            "descriptor_text": popup.get("descriptor_text"),
            "reference_x_median": popup.get("reference_x_median"),
            "history_n": popup.get("history_n"),
            "history_percentile": popup.get("history_percentile"),
            "history_x_median": popup.get("history_x_median"),
            "history_sensor": popup.get("history_sensor"),
            "history_metric": popup.get("history_metric"),
            "affected_sensor_ids": list(popup.get("affected_sensor_ids") or []),
            "significant_sensor_ids": list(popup.get("significant_sensor_ids") or []),
            "mild_sensor_ids": list(popup.get("mild_sensor_ids") or []),
            "quality_flags": list(popup.get("quality_flags") or []),
            "dedupe_key": popup.get("dedupe_key"),
            "summary": popup.get("message"),
            "detail_path": str(detail_path),
            "model_input_file": model_input_path.name if isinstance(model_input, dict) else None,
            "model_input_sha256": model_input_sha256,
            "encoder_input_file": encoder_input_path.name if isinstance(encoder_input, dict) else None,
            "encoder_input_sha256": None,
            "episode_arrays_sha256": (
                (encoder_input.get("metadata") or {}).get("episode_arrays_sha256")
                if isinstance(encoder_input, dict)
                else None
            ),
        }
        detail = {
            **record,
            "popup": popup,
            "agent": agent_status,
            "monitor_request": monitor_request,
            "pipeline_result": result,
        }

        with _ANOMALY_WINDOW_REGISTRY_LOCK:
            if isinstance(model_input, dict):
                canonical_model_input = json.dumps(
                    model_input,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                observed_model_input_sha256 = hashlib.sha256(
                    canonical_model_input.encode("utf-8")
                ).hexdigest()
                if model_input_sha256 and observed_model_input_sha256 != model_input_sha256:
                    raise ValueError("model replay input SHA-256 does not match the captured payload")
                record["model_input_sha256"] = observed_model_input_sha256
                detail["model_input_sha256"] = observed_model_input_sha256
                model_input_path.write_text(
                    canonical_model_input,
                    encoding="utf-8",
                )
                model_input_path.chmod(0o600)
            if isinstance(encoder_input, dict):
                import numpy as np

                arrays = encoder_input.get("arrays")
                metadata = encoder_input.get("metadata")
                if not isinstance(arrays, dict) or not isinstance(metadata, dict):
                    raise ValueError("encoder replay input is incomplete")
                expected_slots = {"baseline", "target_1", "target_2", "target_3", "target_4", "target_5"}
                if set(arrays) != expected_slots:
                    raise ValueError("encoder replay input does not contain the six canonical board slots")
                np.savez_compressed(
                    encoder_input_path,
                    **{slot: np.asarray(arrays[slot], dtype="<f4") for slot in sorted(expected_slots)},
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    ),
                )
                encoder_input_path.chmod(0o600)
                encoder_input_sha256 = hashlib.sha256(encoder_input_path.read_bytes()).hexdigest()
                record["encoder_input_sha256"] = encoder_input_sha256
                detail["encoder_input_sha256"] = encoder_input_sha256
            detail_path.write_text(json.dumps(detail, indent=2, default=str) + "\n", encoding="utf-8")
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")

        return {
            "registered": True,
            "window_id": window_id,
            "registry_dir": str(registry_dir),
            "jsonl_path": str(jsonl_path),
            "detail_path": str(detail_path),
        }
    except Exception as exc:
        return {
            "registered": False,
            "reason": "registration_failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _anomaly_window_records():
    """Read the archive newest first; consumers limit after selecting usable checks."""
    jsonl_path = _resolve_anomaly_window_dir() / "anomaly_windows.jsonl"
    if not jsonl_path.exists():
        return

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    yield from reversed(records)


def _anomaly_windows_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    limit = min(1000, max(1, _int_query(query, "limit", 50)))
    registry_dir = _resolve_anomaly_window_dir()
    jsonl_path = registry_dir / "anomaly_windows.jsonl"
    records = []
    for record in _anomaly_window_records():
        records.append(record)
        if len(records) >= limit:
            break
    return {
        "ok": True,
        "status": "ok",
        "registry_dir": str(registry_dir),
        "jsonl_path": str(jsonl_path),
        "windows": records,
    }


def _resolve_anomaly_window_dir() -> Path:
    raw = (
        os.environ.get("VIBRO_ANOMALY_WINDOW_DIR")
        or os.environ.get("VIBRO_ANOMALY_REGISTRY_DIR")
        or DEFAULT_ANOMALY_WINDOW_DIR
    )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _safe_anomaly_window_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return safe[:160] or f"anomaly-{os.urandom(4).hex()}"


def _auto_select_agent_model(*, base_url: str, api_key: str, requested_model: str) -> str:
    cache_key = (base_url, requested_model)
    cached = _AGENT_MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    selected = requested_model
    try:
        client = _load_openai_client(base_url=base_url, api_key=api_key)
        model_ids = [str(item.id) for item in client.models.list().data]
    except Exception:
        model_ids = []

    if model_ids and requested_model not in model_ids:
        qwen_instruct = [
            model_id
            for model_id in model_ids
            if "qwen" in model_id.lower() and "instruct" in model_id.lower()
        ]
        selected = qwen_instruct[0] if qwen_instruct else model_ids[0]
    _AGENT_MODEL_CACHE[cache_key] = selected
    return selected


def _mark_monitor_llm_skipped(result: dict[str, Any], *, reason: str = "disabled_by_request") -> dict[str, Any]:
    result_metadata = dict(result.get("model_metadata") or {})
    fallbacks = list(result_metadata.get("fallbacks_used") or [])
    fallback = f"qwen_autonomous_monitor:{reason}"
    if fallback not in fallbacks:
        fallbacks.append(fallback)
    result_metadata.update(
        {
            "agent_mode": "deterministic_feature_prepass",
            "monitor_llm_model_used": False,
            "monitor_llm_skipped": True,
            "monitor_llm_fallback_reason": reason,
            "fallbacks_used": fallbacks,
        }
    )
    result["model_metadata"] = result_metadata
    return result


# ---------------------------------------------------------------------------
# Rolling per-window metrics history — the building's own learned "normal".
#
# Every monitor cycle appends one row (all sensors' target-to-reference ratios)
# to a JSONL file and an in-memory deque. New windows are scored against the
# distribution of past NORMAL windows (label normal, no quality flags): the
# empirical percentile and the multiple-of-median answer "is this evidence
# anomalous for THIS building?" from data rather than from fixed thresholds.
# Best-effort by design: any failure here must never break the monitor.
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_HISTORY_DIR = "validation_exports/window_metrics"
_NORMAL_HISTORY_LABEL = "normal_relative_to_reference_sensor"
_NORMAL_HISTORY_LABELS = {_NORMAL_HISTORY_LABEL, "normal"}
# Magnitude metrics only: dominant-frequency delta is signed and has no
# meaningful median-multiple, so it stays out of the history statistics.
_HISTORY_METRIC_SHORT = {
    "rms_ratio": "rms",
    "peak_ratio": "peak",
    "ppv_ratio": "ppv",
    "spectral_distance": "spec",
}
_WINDOW_HISTORY_LOCK = threading.Lock()
_WINDOW_HISTORY: deque | None = None
_WINDOW_HISTORY_APPENDS = 0
_HISTORY_STATS_CACHE: dict[tuple[str, str], list[float]] = {}


def _resolve_window_history_dir() -> Path:
    raw = os.environ.get("VIBRO_WINDOW_HISTORY_DIR") or DEFAULT_WINDOW_HISTORY_DIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _window_history_path() -> Path:
    return _resolve_window_history_dir() / "window_metrics.jsonl"


def _window_history_max_rows() -> int:
    return _env_int_value("VIBRO_WINDOW_HISTORY_MAX_ROWS", 20000, minimum=500)


def _history_min_samples() -> int:
    return _env_int_value("VIBRO_HISTORY_MIN_SAMPLES", 150, minimum=5)


def _load_window_history_locked() -> deque:
    global _WINDOW_HISTORY
    if _WINDOW_HISTORY is None:
        rows: deque = deque(maxlen=_window_history_max_rows())
        path = _window_history_path()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            rows.append(row)
            except OSError:
                pass
        _WINDOW_HISTORY = rows
    return _WINDOW_HISTORY


def _record_window_history(result: dict[str, Any], popup: dict[str, Any]) -> None:
    """Append this window's per-sensor comparison metrics to the rolling history."""
    if not _env_bool("VIBRO_RECORD_WINDOW_HISTORY", True):
        return
    reports = result.get("sensor_agent_reports") or []
    sensors: dict[str, dict[str, float]] = {}
    any_quality = bool((result.get("network_assessment") or {}).get("quality_flags"))
    for report in reports:
        comparison = report.get("baseline_comparison") or {}
        row: dict[str, float] = {}
        for metric, short in _HISTORY_METRIC_SHORT.items():
            value = _safe_float(comparison.get(metric))
            if value is not None and value >= 0:
                row[short] = round(value, 4)
        if row:
            sensors[str(report.get("sensor_id") or "")] = row
        if report.get("quality_flags"):
            any_quality = True
    if not sensors:
        return
    # The reference sensor's own absolute level, recovered from any target's
    # absolute RMS and its ratio. The pipeline's anomaly definition assumes the
    # reference stays calm; recording its level lets that assumption be audited
    # (a drifting reference compresses every ratio toward 1 and hides anomalies).
    ref_rms = None
    for report in reports:
        target_rms = _safe_float((report.get("metrics") or {}).get("acceleration_rms_g"))
        rms_ratio = _safe_float((report.get("baseline_comparison") or {}).get("rms_ratio"))
        if target_rms is not None and rms_ratio is not None and rms_ratio > 1e-9:
            ref_rms = round(target_rms / rms_ratio, 6)
            break
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": str(popup.get("network_label") or ""),
        "q": 1 if any_quality else 0,
        "sensors": sensors,
    }
    if popup.get("label_uncertain"):
        entry["u"] = 1
    if ref_rms is not None:
        entry["ref_rms"] = ref_rms
    global _WINDOW_HISTORY_APPENDS
    try:
        with _WINDOW_HISTORY_LOCK:
            rows = _load_window_history_locked()
            rows.append(entry)
            _HISTORY_STATS_CACHE.clear()
            path = _window_history_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            _WINDOW_HISTORY_APPENDS += 1
            if _WINDOW_HISTORY_APPENDS % 2000 == 0:
                # Periodic compaction: the deque holds the retention window, the
                # file is rewritten from it so it cannot grow without bound.
                path.write_text(
                    "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8",
                )
            else:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _history_values_locked(sensor_id: str, metric_short: str) -> list[float]:
    """Sorted values of one sensor+metric across recorded NORMAL windows (cached).

    The pseudo-sensor id "__ref__" selects the reference sensor's own absolute
    RMS (the row-level ref_rms field) — used to audit the pipeline's assumption
    that the reference stays calm.
    """
    key = (sensor_id, metric_short)
    cached = _HISTORY_STATS_CACHE.get(key)
    if cached is not None:
        return cached
    rows = _load_window_history_locked()
    values: list[float] = []
    for row in rows:
        if row.get("label") not in _NORMAL_HISTORY_LABELS or row.get("q") or row.get("u"):
            continue
        if sensor_id == "__ref__":
            value = _safe_float(row.get("ref_rms"))
        else:
            value = _safe_float((row.get("sensors") or {}).get(sensor_id, {}).get(metric_short))
        if value is not None:
            values.append(value)
    values.sort()
    _HISTORY_STATS_CACHE[key] = values
    return values


def _attach_window_history_context(result: dict[str, Any]) -> None:
    """Score each sensor's window against the building's own recorded normal.

    Attaches per-report `history_context` (worst metric's percentile and
    multiple-of-median) and a network-level `history_summary` (the worst sensor
    overall, or a learning marker while the history is still too small).
    """
    try:
        min_samples = _history_min_samples()
        worst_overall: dict[str, Any] | None = None
        worst_overall_key: tuple[float, float] = (-1.0, -1.0)
        max_samples_seen = 0

        def _rarity_key(candidate: dict[str, Any]) -> tuple[float, float]:
            # Two-sided: far BELOW the normal range is as unusual as far above
            # (a decoupled/quiet sensor is an anomaly too), so rank by distance
            # from the middle of the distribution, not by high percentile alone.
            percentile = candidate["percentile"]
            return (max(percentile, 100.0 - percentile), abs((candidate["x_median"] or 1.0) - 1.0))

        with _WINDOW_HISTORY_LOCK:
            for report in result.get("sensor_agent_reports") or []:
                sensor_id = str(report.get("sensor_id") or "")
                comparison = report.get("baseline_comparison") or {}
                best: dict[str, Any] | None = None
                best_key: tuple[float, float] = (-1.0, -1.0)
                sensor_samples = 0
                for metric, short in _HISTORY_METRIC_SHORT.items():
                    value = _safe_float(comparison.get(metric))
                    if value is None or value < 0:
                        continue
                    values = _history_values_locked(sensor_id, short)
                    sensor_samples = max(sensor_samples, len(values))
                    if len(values) < min_samples:
                        continue
                    percentile = 100.0 * bisect_right(values, value) / len(values)
                    median = values[len(values) // 2]
                    x_median = round(value / median, 2) if median > 1e-9 else None
                    candidate = {
                        "metric": short,
                        "percentile": round(percentile, 2),
                        "x_median": x_median,
                        "n": len(values),
                    }
                    candidate_key = _rarity_key(candidate)
                    if best is None or candidate_key > best_key:
                        best, best_key = candidate, candidate_key
                max_samples_seen = max(max_samples_seen, sensor_samples)
                if best is not None:
                    report["history_context"] = best
                    if worst_overall is None or best_key > worst_overall_key:
                        worst_overall, worst_overall_key = {**best, "sensor": sensor_id}, best_key
                else:
                    report["history_context"] = {"n": sensor_samples, "learning": True}
        result["history_summary"] = (
            worst_overall if worst_overall is not None else {"n": max_samples_seen, "learning": True}
        )
        # Reference-calmness audit: the pipeline's anomaly definition assumes the
        # reference sensor stays calm. Score its own absolute level against its
        # normal history so a loud reference (which compresses every ratio toward
        # 1 and hides anomalies) is visible to the model and the operator.
        ref_rms = None
        for report in result.get("sensor_agent_reports") or []:
            target_rms = _safe_float((report.get("metrics") or {}).get("acceleration_rms_g"))
            rms_ratio = _safe_float((report.get("baseline_comparison") or {}).get("rms_ratio"))
            if target_rms is not None and rms_ratio is not None and rms_ratio > 1e-9:
                ref_rms = target_rms / rms_ratio
                break
        if ref_rms is not None:
            with _WINDOW_HISTORY_LOCK:
                values = _history_values_locked("__ref__", "rms")
            if len(values) >= min_samples:
                median = values[len(values) // 2]
                if median > 1e-12:
                    result["reference_history"] = {
                        "x_median": round(ref_rms / median, 2),
                        "percentile": round(100.0 * bisect_right(values, ref_rms) / len(values), 2),
                        "n": len(values),
                    }
    except Exception:
        result["history_summary"] = None


# ---------------------------------------------------------------------------
# Structured monitor descriptor — the model's decision as orthogonal fields
# instead of one fused label. extent × character × persistence describe the
# vibration situation; data carries the trust state. The legacy 7-label
# vocabulary is kept as a deterministic projection (renaming, not judgment) so
# severities, dedupe keys, the registry, and the Anomalies panel stay stable.
# ---------------------------------------------------------------------------
DESCRIPTOR_EXTENTS = ("none", "one", "several", "all")
DESCRIPTOR_CHARACTERS = ("amplitude", "spectral", "impulsive", "mixed")
DESCRIPTOR_PERSISTENCE = ("transient", "sustained", "growing")
DESCRIPTOR_DATA = ("ok", "quality", "insufficient")


def _extent_from_affected(affected_count: int, target_count: int) -> str:
    """Extent is COUNTED from the model's own affected list, never generated
    separately — a generated extent could contradict the list it describes
    ("one sensor" next to two affected ids, observed live 2026-07-06)."""
    if affected_count <= 0:
        return "none"
    if affected_count == 1:
        return "one"
    if target_count > 0 and affected_count >= target_count:
        return "all"
    return "several"


def _parse_monitor_descriptor(
    data: dict[str, Any],
    *,
    known_sensor_ids: list[str],
) -> tuple[dict[str, str] | None, list[str], str | None]:
    """Validate the model's decision fields and derive extent from its affected list.

    Returns (descriptor, affected_ids, error). The affected list is deduplicated
    and filtered to real sensor ids (a hallucinated id cannot enter the report).
    """
    raw_affected = data.get("affected")
    if not isinstance(raw_affected, list):
        return None, [], "model_output_missing_affected"
    known = set(known_sensor_ids)
    affected = [item for item in dict.fromkeys(str(value) for value in raw_affected) if item in known]
    descriptor = {
        "extent": _extent_from_affected(len(affected), len(known_sensor_ids)),
        "character": str(data.get("character") or "").strip().lower(),
        "persistence": str(data.get("persistence") or "").strip().lower(),
        "data": str(data.get("data") or "").strip().lower(),
    }
    for key, allowed in (
        ("character", DESCRIPTOR_CHARACTERS),
        ("persistence", DESCRIPTOR_PERSISTENCE),
        ("data", DESCRIPTOR_DATA),
    ):
        if descriptor[key] not in allowed:
            return None, [], f"model_output_invalid_descriptor_{key}"
    return descriptor, affected, None


def _legacy_label_from_descriptor(descriptor: dict[str, str], assessment: dict[str, Any]) -> str:
    """Mechanical projection of the descriptor onto the legacy 7-label vocabulary.

    Pure renaming for compatibility (severity mapping, dedupe keys, registry,
    panel episodes) — the judgment itself is the model's descriptor. The
    mild/significant split reuses the deterministic prepass sensor sets, which
    are measurements.
    """
    if descriptor["data"] == "insufficient":
        return "insufficient_data"
    if descriptor["data"] == "quality":
        return "sensor_network_quality_issue"
    extent = descriptor["extent"]
    if extent == "none":
        return "normal_relative_to_reference_sensor"
    mild_only = not _string_list(assessment.get("significant_sensor_ids"))
    if extent == "one":
        return "mild_local_deviation" if mild_only else "localized_vibration_deviation"
    return "mild_multi_sensor_deviation" if mild_only else "multi_sensor_structure_wide_vibration_event"


def _descriptor_text(descriptor: dict[str, str]) -> str:
    if descriptor["data"] == "insufficient":
        return "insufficient data to judge"
    if descriptor["data"] == "quality":
        return "sensor data quality issue"
    if descriptor["extent"] == "none":
        return "no deviation from the reference"
    extent_phrase = {
        "one": "one sensor",
        "several": "several sensors",
        "all": "all sensors",
    }[descriptor["extent"]]
    return f"{extent_phrase} · {descriptor['persistence']} {descriptor['character']} deviation"


# Previous monitor decision, so the model can judge persistence (transient vs
# sustained vs growing) from real temporal context instead of a single window.
# Mutated only under _AGENT_MONITOR_LOCK.
_MONITOR_PREV_DECISION: dict[str, Any] = {"extent": None, "character": None, "polls": 0, "hx": None}


def _update_monitor_prev_decision(descriptor: dict[str, str], history_summary: dict[str, Any] | None) -> None:
    same = (
        descriptor["extent"] == _MONITOR_PREV_DECISION.get("extent")
        and descriptor["character"] == _MONITOR_PREV_DECISION.get("character")
    )
    _MONITOR_PREV_DECISION.update(
        {
            "extent": descriptor["extent"],
            "character": descriptor["character"],
            "polls": (_MONITOR_PREV_DECISION.get("polls") or 0) + 1 if same else 1,
            "hx": (history_summary or {}).get("x_median"),
        }
    )


# Per-episode vote tally: extent of the last monitor decision and every sampled
# re-decision collected so far in the episode. Mutated only under
# _AGENT_MONITOR_LOCK (the use_llm monitor path is serialized on it).
_MONITOR_VOTE_STATE: dict[str, Any] = {"label": None, "votes": []}
_MONITOR_VOTE_BUDGET_S = 24.0
# Stop growing the tally once it is this long: windows drift over a long episode, so
# very old votes should not keep diluting the estimate forever.
_MONITOR_VOTE_MAX_TALLY = 40


def _perturb_monitor_payload(payload: dict[str, Any], *, seed: int, relative_sigma: float) -> dict[str, Any]:
    """A copy of the monitor payload with the target measurements jittered by typical
    window-to-window noise (multiplicative Gaussian for the non-negative magnitudes,
    additive for the frequency shift). Seeded, so each vote index reproducibly sees the
    same perturbation for a given window."""
    rng = random.Random(seed)
    perturbed = copy.deepcopy(payload)
    for row in perturbed.get("targets") or []:
        before = {key: _safe_float(row.get(key)) for key in ("rms", "peak", "ppv", "spec")}
        for key in ("rms", "peak", "ppv", "spec"):
            value = before.get(key)
            if value is not None:
                row[key] = _round_monitor_value(max(0.0, value * (1.0 + rng.gauss(0.0, relative_sigma))))
        df = _safe_float(row.get("df"))
        if df is not None:
            row["df"] = _round_monitor_value(df + rng.gauss(0.0, max(0.5, abs(df) * relative_sigma)))
        # hx is derived from one of the jittered metrics (hm), so it must move by
        # exactly that metric's jitter factor to stay consistent inside the probe.
        hx = _safe_float(row.get("hx"))
        source = str(row.get("hm") or "")
        if hx is not None and before.get(source) and _safe_float(row.get(source)) is not None:
            row["hx"] = _round_monitor_value(max(0.0, hx * (_safe_float(row.get(source)) / before[source])))
    return perturbed


def _post_monitor_vote(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    sampler: dict[str, Any],
    sensor_ids: list[str],
    timeout_s: float,
) -> str | None:
    """One re-decision: the probe names the affected sensors (grammar-locked to the
    real ids) and the vote is the EXTENT COUNTED from that list — the same derivation
    as the main decision, so probes and decision cannot diverge in definition. The
    sampler dict chooses the mode: greedy (perturbation votes — the only variation is
    the data jitter) or temperature 1 with an explicit open sampler and seed (sampling
    votes — the C plugin's default is greedy top-k 1, which would otherwise make
    temperature and seed no-ops)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Do not include markdown."},
            {"role": "user", "content": prompt},
        ],
        **sampler,
        "max_tokens": 48,
        "timeout_s": round(max(1.0, timeout_s), 3),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "monitor_vote",
                "schema": {
                    "type": "object",
                    "properties": {
                        "affected": {
                            "type": "array",
                            "items": {"enum": sorted(sensor_ids)} if sensor_ids else {"type": "string"},
                        }
                    },
                    "required": ["affected"],
                },
            },
        },
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'EMPTY'}"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s + 5.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = str(((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    raw_affected = json.loads(text).get("affected")
    if not isinstance(raw_affected, list):
        return None
    known = set(sensor_ids)
    affected = [item for item in dict.fromkeys(str(value) for value in raw_affected) if item in known]
    return _extent_from_affected(len(affected), len(sensor_ids))


def _monitor_vote_confidence(
    *,
    result: dict[str, Any],
    descriptor: dict[str, str],
    model_base_url: str,
    model_api_key: str,
    model: str,
    vote_k: int | None = None,
) -> dict[str, Any] | None:
    """Decision-stability agreement for the monitor decision (the uncertainty gate).

    Default mode (VIBRO_MONITOR_VOTE_MODE=sample, user-chosen 2026-07-06): each
    vote re-decides the SAME unmodified window at temperature 1 with a distinct
    seed — a Monte-Carlo estimate of the model's own decision stability on the
    evidence at hand. Alternative mode (=perturb): re-decides GREEDILY on
    measurements jittered by typical window-to-window noise
    (VIBRO_MONITOR_PERTURB_PCT, default 10%), measuring distance to a decision
    boundary instead; more sensitive for a decisive model, which answers
    unanimously under sampling even on borderline windows.
    A new episode (extent change) starts with a burst of k votes (vote_k request
    override, else VIBRO_MONITOR_VOTE_K, default 5; k=1 means a single probe);
    while the situation persists, each poll adds one more. The share is NOT the
    displayed confidence: the popup shows the model's generated evidence-strength
    number, and this share only gates it — below the uncertainty threshold the
    label is flagged uncertain and the share takes over the display (see
    _apply_monitor_llm_decision). Returns None when voting is unavailable (chat
    holding the NPU, endpoint down, too few valid votes). Disable entirely with
    VIBRO_MONITOR_CONFIDENCE=generated.
    """
    if (os.environ.get("VIBRO_MONITOR_CONFIDENCE") or "votes").strip().lower() != "votes":
        return None
    # The gate probes the primary decision axis: would the EXTENT survive
    # re-measurement? (Character/persistence flips are less operationally
    # critical and would make the gate fire on descriptive nuance.)
    expected_code = descriptor["extent"]
    episode_key = f"{descriptor['extent']}|{descriptor['data']}"
    if episode_key != _MONITOR_VOTE_STATE["label"]:
        _MONITOR_VOTE_STATE.update({"label": episode_key, "votes": []})
    votes: list[str] = _MONITOR_VOTE_STATE["votes"]
    k = vote_k if vote_k is not None else _env_int_value("VIBRO_MONITOR_VOTE_K", 5, minimum=1)
    k = max(1, min(10, int(k)))
    if len(votes) < k:
        needed = k - len(votes)  # burst at episode start (or refill after failures)
    elif len(votes) < _MONITOR_VOTE_MAX_TALLY:
        needed = 1  # steady state: one refinement vote per poll
    else:
        needed = 0
    mode = (os.environ.get("VIBRO_MONITOR_VOTE_MODE") or "sample").strip().lower()
    perturb_sigma = _env_float_value("VIBRO_MONITOR_PERTURB_PCT", 10.0, minimum=1.0) / 100.0
    payload = _monitor_llm_payload(result)
    sensor_ids = sorted(
        {str(report.get("sensor_id") or "") for report in result.get("sensor_agent_reports") or []} - {""}
    )
    deadline = time.monotonic() + _MONITOR_VOTE_BUDGET_S
    for _ in range(needed):
        remaining = deadline - time.monotonic()
        if remaining <= 1.0 or _chat_is_inflight():
            break
        vote_index = len(votes) + 1
        if mode == "perturb":
            vote_prompt = _render_monitor_prompt(
                _perturb_monitor_payload(payload, seed=vote_index, relative_sigma=perturb_sigma)
            )
            sampler: dict[str, Any] = {"temperature": 0.0}
        else:
            vote_prompt = _render_monitor_prompt(payload)
            sampler = {"temperature": 1.0, "top_k": 100, "top_p": 1.0, "seed": vote_index}
        try:
            code = _post_monitor_vote(
                base_url=model_base_url,
                api_key=model_api_key,
                model=model,
                prompt=vote_prompt,
                sampler=sampler,
                sensor_ids=sensor_ids,
                timeout_s=min(8.0, remaining),
            )
        except (HTTPError, URLError, TimeoutError, OSError):
            break  # endpoint unavailable: further attempts would fail the same way
        except Exception:
            continue  # one malformed vote does not invalidate the others
        if code:
            votes.append(code)
    if len(votes) < min(2, k):
        return None
    agree = votes.count(expected_code)
    agreement = round(agree / len(votes), 3)
    metadata = {
        "monitor_vote_mode": mode,
        "monitor_vote_n": len(votes),
        "monitor_vote_agree": agree,
        "monitor_vote_agreement": agreement,
        "monitor_vote_codes": votes[-10:],
    }
    return {"agreement": agreement, "metadata": metadata}


_CODES_MONITOR_MODE_ENV = "VIBRO_MONITOR_DECISION_MODE"
_CODES_WORKER_WINDOW_S = 10.5  # read margin so the worker's exact 10 s fits


def _codes_monitor_mode_enabled() -> bool:
    value = (os.environ.get(_CODES_MONITOR_MODE_ENV) or "").strip().lower()
    return value in {"codes", "codes_v3"}


def _apply_codes_monitor_decision(
    result: dict[str, Any],
    *,
    model_base_url: str,
    model_api_key: str,
    model: str,
    model_timeout_s: float,
    config_path: str | None = None,
    process_reader: bool = False,
) -> dict[str, Any]:
    """codes_v3 decision: raw windows -> codec-v1 codes -> finetuned model.

    Replaces the descriptor LLM call when VIBRO_MONITOR_DECISION_MODE=codes:
    the six live windows are encoded by the frozen codec-v1 checkpoint in
    the codec-cpu-venv worker and the codes_v3 SFT model judges the five
    targets against the baseline. Its labels ARE the legacy vocabulary, so
    they merge into the panel fields directly. Every failure is reported as
    a model failure (LLM-only policy) — the deterministic prepass label is
    never promoted to a model decision.
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np

    from . import live_codes

    result_metadata = dict(result.get("model_metadata") or {})

    def _failed(reason: str) -> dict[str, Any]:
        fallbacks = list(result_metadata.get("fallbacks_used") or [])
        fallback = f"codes_v3_monitor:{reason}"
        if fallback not in fallbacks:
            fallbacks.append(fallback)
        result_metadata.update(
            {
                "agent_mode": "codes_v3_monitor",
                "monitor_llm_model_used": False,
                "monitor_llm_fallback_reason": reason,
                "fallbacks_used": fallbacks,
            }
        )
        result["model_metadata"] = result_metadata
        return result

    try:
        registry = SensorRegistry(_resolve_live_sensor_config_path(config_path))
    except Exception as exc:
        return _failed(f"sensor_registry_failed:{format_exception_for_response(exc)}")
    sensors = {sensor.sensor_id: sensor for sensor in registry.sensors.values()}
    slots = ("baseline", *live_codes.LIVE_SLOTS)
    if registry.baseline_sensor_id != "baseline" or \
            not set(slots).issubset(sensors):
        return _failed(
            "sensor_set_mismatch: codes_v3 needs baseline + "
            f"{list(live_codes.LIVE_SLOTS)}, config has {sorted(sensors)}")

    reader_fn = (
        read_sdk_vibrometer_window_via_board_process
        if board_reader_processes_enabled(default=process_reader)
        else read_sdk_vibrometer_window
    )

    # Axis sweep: VIBRO_CODES_AXES="x,y,z" runs the whole read->encode->
    # judge pass once per axis and combines by max severity. Each axis is
    # a legitimate single-axis signal (the training contract); "norm" is
    # deliberately not accepted — a rectified magnitude is out of the
    # codec's training distribution.
    axes_env = (os.environ.get("VIBRO_CODES_AXES") or "").strip().lower()
    axes = [axis.strip() for axis in axes_env.split(",") if axis.strip()]
    for axis in axes:
        if axis not in {"x", "y", "z", "0", "1", "2"}:
            return _failed(f"bad_axis:{axis!r} (single axes only, no norm)")
    axis_passes: list[str | None] = list(axes) or [None]

    worker_python = os.environ.get(
        "VIBRO_CODES_WORKER_PYTHON", "/home/ubuntu/codec-cpu-venv/bin/python")
    worker_script = Path(__file__).resolve().parents[2] / "scripts" / \
        "live_codes_worker.py"
    worker_timeout_s = _env_float_value("VIBRO_CODES_WORKER_TIMEOUT_S", 90.0)
    client = ModelClient(
        base_url=model_base_url,
        api_key=model_api_key,
        model=model,
        base_url_env="MAIN_AGENT_BASE_URL",
        api_key_env="MAIN_AGENT_API_KEY",
        model_env="MAIN_AGENT_MODEL",
        role="qwen_codes_monitor",
        timeout_s=model_timeout_s,
        max_tokens=_env_int_value("AGENT_MONITOR_MAX_TOKENS", 192, minimum=64),
    )

    def _axis_pass(axis_override: str | None):
        """One read->encode->judge pass; returns (outcome, None) or
        (None, failure_reason)."""
        arrays: dict[str, Any] = {}
        rates: dict[str, float] = {}
        for slot in slots:
            sensor = sensors[slot]
            try:
                window = reader_fn(
                    machine_id=slot,
                    acquisition_folder=sensor.acquisition_folder,
                    sensor_name=sensor.hsd_sensor_name,
                    axis=axis_override or sensor.axis,
                    duration_s=_CODES_WORKER_WINDOW_S,
                    require_current=True,
                )
            except Exception as exc:
                return None, (f"live_read_failed:{slot}:"
                              f"{format_exception_for_response(exc)}")
            signal = np.asarray(window.signal, dtype=np.float64)
            fs_hz = float(window.sampling_rate_hz)
            if signal.size < int(10.0 * fs_hz):
                return None, (f"window_too_short:{slot}:{signal.size} "
                              f"samples at {fs_hz:g} Hz")
            arrays[slot] = signal
            rates[slot] = fs_hz

        worker_started_s = time.perf_counter()
        workdir = tempfile.mkdtemp(prefix="codes_monitor_")
        try:
            windows_path = Path(workdir) / "windows.npz"
            codes_path = Path(workdir) / "codes.json"
            rate_scalars = {
                f"fs_hz__{slot}": np.float64(rates[slot]) for slot in slots
            }
            np.savez(windows_path, **arrays, **rate_scalars)
            proc = subprocess.run(
                [worker_python, str(worker_script),
                 "--windows", str(windows_path), "--out", str(codes_path)],
                capture_output=True, text=True, timeout=worker_timeout_s,
            )
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-300:]
                return None, f"codes_worker_failed:{tail}"
            codes_payload = json.loads(codes_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            return None, f"codes_worker_timeout:{worker_timeout_s:g}s"
        except Exception as exc:
            return None, (f"codes_worker_error:"
                          f"{format_exception_for_response(exc)}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        code_slots = codes_payload.get("slots") or {}
        try:
            targets = [
                (slot, float(code_slots[slot]["level_rel_db"]),
                 str(code_slots[slot]["codes"]))
                for slot in live_codes.LIVE_SLOTS
            ]
            prompt = live_codes.build_codes_user_prompt(
                str(code_slots["baseline"]["codes"]), targets, aligned=True)
        except (KeyError, live_codes.CodesPromptError) as exc:
            return None, f"codes_prompt_failed:{exc}"

        model_result = client.call_json_model(
            prompt,
            allowed_labels=None,
            extra_body={"response_format": live_codes.codes_response_format(
                list(live_codes.LIVE_SLOTS))},
            system=live_codes.system_message(),
        )
        if not model_result.ok or not model_result.text:
            return None, model_result.error or "model_call_failed"
        try:
            parsed = live_codes.parse_codes_reply(
                model_result.text, list(live_codes.LIVE_SLOTS))
        except ValueError as exc:
            return None, f"model_output_invalid:{exc}"
        return {
            "verdict": parsed,
            "code_slots": code_slots,
            "rates": rates,
            "provenance": codes_payload.get("provenance") or {},
            "worker_s": time.perf_counter() - worker_started_s,
            "model_metadata": model_result.metadata,
        }, None

    passes: dict[str, dict] = {}
    for axis_override in axis_passes:
        pass_label = axis_override or "config"
        outcome, reason = _axis_pass(axis_override)
        if outcome is None:
            tagged = reason if len(axis_passes) == 1 else \
                f"axis_{pass_label}:{reason}"
            return _failed(tagged)
        passes[pass_label] = outcome

    _SENSOR_RANK = {"normal_relative_to_baseline": 0,
                    "sensor_data_quality_issue": 1,
                    "mild_deviation": 2,
                    "significant_local_deviation": 3}
    _NETWORK_RANK = {"normal_relative_to_reference_sensor": 0,
                     "mild_local_deviation": 1,
                     "mild_multi_sensor_deviation": 2,
                     "localized_vibration_deviation": 3,
                     "multi_sensor_building_wide_vibration_event": 4}
    verdict = {
        "sensor_reports": {
            slot: max((p["verdict"]["sensor_reports"][slot]
                       for p in passes.values()),
                      key=lambda label: _SENSOR_RANK[label])
            for slot in live_codes.LIVE_SLOTS
        },
        "network_status": max(
            (p["verdict"]["network_status"] for p in passes.values()),
            key=lambda label: _NETWORK_RANK[label]),
        "affected_sensor_ids": sorted(
            {slot for p in passes.values()
             for slot in p["verdict"]["affected_sensor_ids"]},
            key=list(live_codes.LIVE_SLOTS).index),
    }
    first_pass = next(iter(passes.values()))
    code_slots = first_pass["code_slots"]
    rates = first_pass["rates"]
    codes_payload = {"provenance": first_pass["provenance"]}
    worker_duration_s = sum(p["worker_s"] for p in passes.values())
    model_result_metadata = next(reversed(passes.values()))["model_metadata"]

    for report in result.get("sensor_agent_reports") or []:
        slot = str(report.get("sensor_id") or "")
        label = verdict["sensor_reports"].get(slot)
        if label:
            report["small_agent_label"] = label
            report_metadata = dict(report.get("model_metadata") or {})
            report_metadata.update(
                {"decision_source": "codes_v3_llm", "model_used": True})
            report["model_metadata"] = report_metadata

    network_label = verdict["network_status"]
    affected = verdict["affected_sensor_ids"]
    deviations = [
        f"{slot}: {label.replace('_', ' ')}"
        for slot, label in sorted(verdict["sensor_reports"].items())
        if label != "normal_relative_to_baseline"
    ]
    axes_note = (f" (axes {', '.join(passes)}; worst per sensor)"
                 if len(passes) > 1 else "")
    summary = (
        "Codes model verdict" + axes_note + ": "
        + network_label.replace("_", " ")
        + (" — " + "; ".join(deviations) if deviations
           else " — all targets normal relative to baseline")
        + ". This is monitoring/triage information, not a certified "
          "structural safety assessment."
    )
    assessment = dict(result.get("network_assessment") or {})
    assessment.update(
        {
            "main_agent_label": network_label,
            "network_label": network_label,
            "affected_sensor_ids": affected,
            "descriptor_text": "codes_v3: " + network_label.replace("_", " "),
            "confidence_source": "codes_v3_schema",
            "label_uncertain": False,
            "explanation": summary,
            "model_metadata": model_result_metadata,
        }
    )
    result["network_assessment"] = assessment
    result["main_agent_explanation"] = summary
    result_metadata.update(
        {
            "agent_mode": "codes_v3_monitor",
            "monitor_llm_model_used": True,
            "monitor_llm_model": model,
            "fallbacks_used": [],
            "codes_monitor": {
                "axes": list(passes),
                "combine_rule": ("max_severity_per_sensor_union_affected"
                                 if len(passes) > 1 else "single_axis"),
                "per_axis": {
                    pass_label: {
                        "network_label": p["verdict"]["network_status"],
                        "affected": p["verdict"]["affected_sensor_ids"],
                        "level_rel_db": {
                            slot: round(float(
                                p["code_slots"][slot]["level_rel_db"]), 2)
                            for slot in live_codes.LIVE_SLOTS
                        },
                    }
                    for pass_label, p in passes.items()
                },
                "codec_checkpoint_step":
                    (codes_payload.get("provenance") or {}).get("checkpoint_step"),
                "codec_checkpoint_sha256":
                    ((codes_payload.get("provenance") or {})
                     .get("checkpoint_sha256") or "")[:16],
                "level_rel_db": {
                    slot: round(float(code_slots[slot]["level_rel_db"]), 2)
                    for slot in live_codes.LIVE_SLOTS
                },
                "worker_duration_s": round(worker_duration_s, 2),
                "fs_native_hz": (
                    next(iter(rates.values()))
                    if len(set(rates.values())) == 1 else None
                ),
                "fs_native_hz_by_slot": rates,
            },
        }
    )
    result["model_metadata"] = result_metadata
    return result


def _apply_monitor_llm_decision(
    result: dict[str, Any],
    *,
    model_base_url: str,
    model_api_key: str,
    model: str,
    model_timeout_s: float = 8.0,
    vote_k: int | None = None,
    config_path: str | None = None,
    process_reader: bool = False,
    start_time_s: float | None = None,
    require_current: bool = True,
    replay_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    if _existing_pipeline_llm_active(result):
        return result
    if _codes_monitor_mode_enabled():
        return _apply_codes_monitor_decision(
            result,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model=model,
            model_timeout_s=model_timeout_s,
            config_path=config_path,
            process_reader=process_reader,
        )

    prompt = _build_monitor_llm_prompt(result)
    prompt_chars = len(prompt)
    prompt_estimated_tokens = _estimate_prompt_tokens(prompt)
    max_prompt_chars, max_prompt_tokens = _monitor_prompt_budget()
    server_timeout_override = _local_genie_timeout_override(model_base_url, model_timeout_s)
    result_metadata = dict(result.get("model_metadata") or {})
    result_metadata.update(
        {
            "monitor_llm_prompt_chars": prompt_chars,
            "monitor_llm_prompt_estimated_tokens": prompt_estimated_tokens,
            "monitor_llm_prompt_budget_chars": max_prompt_chars,
            "monitor_llm_prompt_budget_tokens": max_prompt_tokens,
            "monitor_llm_timeout_s": round(float(model_timeout_s), 3),
            "monitor_llm_server_timeout_override": bool(server_timeout_override),
        }
    )

    if prompt_chars > max_prompt_chars or prompt_estimated_tokens > max_prompt_tokens:
        reason = (
            "prompt_budget_exceeded("
            f"chars={prompt_chars}/{max_prompt_chars},"
            f"est_tokens={prompt_estimated_tokens}/{max_prompt_tokens})"
        )
        result_metadata.update(
            {
                "agent_mode": "single_llm_monitor",
                "monitor_llm_model_used": False,
                "monitor_llm_fallback_reason": reason,
                "fallbacks_used": [f"qwen_autonomous_monitor:{reason}"],
            }
        )
        result["model_metadata"] = result_metadata
        return result

    model_client = ModelClient(
        base_url=model_base_url,
        api_key=model_api_key,
        model=model,
        base_url_env="MAIN_AGENT_BASE_URL",
        api_key_env="MAIN_AGENT_API_KEY",
        model_env="MAIN_AGENT_MODEL",
        role="qwen_autonomous_monitor",
        timeout_s=model_timeout_s,
        max_tokens=_env_int_value("AGENT_MONITOR_MAX_TOKENS", DEFAULT_MONITOR_MAX_TOKENS, minimum=32),
    )
    # Grammar-enforce the response shape on the GenieX shim (schema→GBNF): the decision
    # fields first (affected sensors + character × persistence × data — extent is
    # counted from the affected list, never generated), then the model's own confidence
    # number, then the summary. Adapters without json_schema support ignore
    # response_format and fall back to prompt-guided JSON.
    monitor_sensor_ids = sorted(
        {str(report.get("sensor_id") or "") for report in result.get("sensor_agent_reports") or []} - {""}
    )
    monitor_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "monitor_decision",
            "schema": {
                "type": "object",
                "properties": {
                    "affected": {
                        "type": "array",
                        "items": {"enum": monitor_sensor_ids} if monitor_sensor_ids else {"type": "string"},
                    },
                    "character": {"enum": list(DESCRIPTOR_CHARACTERS)},
                    "persistence": {"enum": list(DESCRIPTOR_PERSISTENCE)},
                    "data": {"enum": list(DESCRIPTOR_DATA)},
                    "confidence": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "required": ["affected", "character", "persistence", "data", "confidence", "summary"],
            },
        },
    }
    monitor_extra_body = {"response_format": monitor_response_format}
    monitor_extra_body.update(server_timeout_override or {})
    model_result = model_client.call_json_model(
        prompt,
        allowed_labels=None,
        extra_body=monitor_extra_body,
    )

    if not model_result.ok or not model_result.data:
        reason = model_result.error or (model_result.metadata or {}).get("fallback_reason") or "model_call_failed"
        result_metadata.update(
            {
                "agent_mode": "single_llm_monitor",
                "monitor_llm_model_used": False,
                "monitor_llm_fallback_reason": reason,
                "fallbacks_used": [f"qwen_autonomous_monitor:{reason}"],
            }
        )
        result["model_metadata"] = result_metadata
        return result

    data = model_result.data
    rejected_terms = find_out_of_domain_component_terms({"data": data, "text": model_result.text})
    if rejected_terms:
        reason = "model_output_rejected_for_out_of_domain_language"
        fallbacks = list(result_metadata.get("fallbacks_used") or [])
        fallback = f"qwen_autonomous_monitor:{reason}"
        if fallback not in fallbacks:
            fallbacks.append(fallback)
        result_metadata.update(
            {
                "agent_mode": "single_llm_monitor",
                "monitor_llm_model_used": False,
                "monitor_llm_fallback_reason": reason,
                "monitor_llm_domain_guard_triggered": True,
                "monitor_llm_rejected_term_count": len(rejected_terms),
                "fallbacks_used": fallbacks,
            }
        )
        result["model_metadata"] = result_metadata
        return result

    def _monitor_llm_failed(reason: str) -> dict[str, Any]:
        # LLM-only policy: an unusable model decision is reported as a model
        # failure. The deterministic prepass label is never silently promoted
        # to "the LLM's decision" (require_llm then surfaces the error).
        fallbacks = list(result_metadata.get("fallbacks_used") or [])
        fallback = f"qwen_autonomous_monitor:{reason}"
        if fallback not in fallbacks:
            fallbacks.append(fallback)
        result_metadata.update(
            {
                "agent_mode": "single_llm_monitor",
                "monitor_llm_model_used": False,
                "monitor_llm_fallback_reason": reason,
                "fallbacks_used": fallbacks,
            }
        )
        result["model_metadata"] = result_metadata
        return result

    descriptor, model_affected, descriptor_error = _parse_monitor_descriptor(
        data, known_sensor_ids=monitor_sensor_ids
    )
    if descriptor is None:
        return _monitor_llm_failed(descriptor_error or "model_output_invalid_descriptor")
    # The descriptor IS the model's decision; the legacy label is a mechanical
    # projection kept for severity mapping, dedupe keys, registry, and panel.
    label = _legacy_label_from_descriptor(descriptor, result.get("network_assessment") or {})
    confidence = _safe_float(data.get("confidence"))
    vote = _monitor_vote_confidence(
        result=result,
        descriptor=descriptor,
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model=model,
        vote_k=vote_k,
    )
    assessment = dict(result.get("network_assessment") or {})
    assessment["descriptor"] = descriptor
    assessment["descriptor_text"] = _descriptor_text(descriptor)
    # The model's own affected list replaces the rule's: the popup extent is
    # counted from this exact list, so the two can never contradict.
    assessment["affected_sensor_ids"] = model_affected
    assessment["main_agent_label"] = label
    assessment["network_label"] = label
    generated = max(0.0, min(1.0, confidence)) if confidence is not None else None
    uncertain_below = _env_float_value("VIBRO_MONITOR_VOTE_UNCERTAIN_BELOW", 0.6)
    if vote is not None:
        assessment["vote_agreement"] = vote["agreement"]
        result_metadata.update(vote["metadata"])
    if vote is not None and vote["agreement"] < uncertain_below:
        # The model is genuinely torn on this label: the sampled agreement share
        # replaces the evidence-strength number and the label is flagged.
        assessment["main_agent_confidence"] = vote["agreement"]
        assessment["confidence_source"] = "model_votes_low_agreement"
        assessment["label_uncertain"] = True
        if generated is not None:
            assessment["main_agent_generated_confidence"] = generated
    elif generated is not None:
        # Displayed confidence = the model's generated evidence-strength number;
        # the vote share above is its uncertainty gate, not the display value.
        assessment["main_agent_confidence"] = generated
        assessment["confidence_source"] = "model"
        assessment["label_uncertain"] = False
    summary = str(data.get("summary") or "").strip()
    if _is_placeholder_monitor_summary(summary):
        return _monitor_llm_failed("model_summary_missing_or_placeholder")
    if "certified structural safety assessment" not in summary.lower():
        summary = (
            f"{summary} This is monitoring/triage information, not a certified structural safety assessment."
        )
    assessment["explanation"] = summary
    result["main_agent_explanation"] = summary
    for key in ("affected_sensor_ids", "significant_sensor_ids", "mild_sensor_ids", "quality_flags", "evidence"):
        values = _string_list(data.get(key))
        if values:
            assessment[key] = values
    assessment["model_metadata"] = model_result.metadata
    result["network_assessment"] = assessment
    _update_monitor_prev_decision(descriptor, result.get("history_summary"))
    result_metadata.update(
        {
            "agent_mode": "single_llm_monitor",
            "monitor_llm_model_used": True,
            "monitor_llm_model": model,
            "fallbacks_used": [],
        }
    )
    result["model_metadata"] = result_metadata
    return result


def _existing_pipeline_llm_active(result: dict[str, Any]) -> bool:
    metadata = result.get("model_metadata") or {}
    if metadata.get("monitor_llm_model_used") is True:
        return True
    reports = result.get("sensor_agent_reports") or []
    if not reports:
        return False
    all_sensor_models_used = all(
        bool((report.get("model_metadata") or {}).get("model_used"))
        for report in reports
    )
    network_metadata = (result.get("network_assessment") or {}).get("model_metadata") or {}
    main_model_used = bool(
        network_metadata.get("model_used")
        or metadata.get("main_model_used")
        or (metadata.get("model_used") and not metadata.get("fallbacks_used"))
    )
    return bool(main_model_used and all_sensor_models_used)


def _build_monitor_llm_prompt(result: dict[str, Any]) -> str:
    return _render_monitor_prompt(_monitor_llm_payload(result))


def _monitor_llm_payload(result: dict[str, Any]) -> dict[str, Any]:
    assessment = result.get("network_assessment") or {}
    sensor_reports = result.get("sensor_agent_reports") or []
    focus_reports, targets_truncated = _monitor_focus_sensor_reports(result)
    payload = {
        "net": _compact_network_assessment(assessment),
        "target_count": len(sensor_reports),
        "targets_truncated": targets_truncated,
        "targets": [_compact_monitor_sensor(report) for report in focus_reports],
    }
    history_n = ((result.get("history_summary") or {}).get("n"))
    if history_n is not None:
        payload["hist_n"] = int(history_n)
    reference_x = (result.get("reference_history") or {}).get("x_median")
    if reference_x is not None:
        payload["ref_hx"] = reference_x
    if _MONITOR_PREV_DECISION.get("extent"):
        payload["prev"] = {
            "extent": _MONITOR_PREV_DECISION["extent"],
            "character": _MONITOR_PREV_DECISION["character"],
            "polls": _MONITOR_PREV_DECISION["polls"],
            "hx": _MONITOR_PREV_DECISION.get("hx"),
        }
    return payload


def _render_monitor_prompt(payload: dict[str, Any]) -> str:
    return (
        "Return JSON only with keys affected, character, persistence, data, confidence and summary. "
        "Interpret only relative building response, reference comparisons, transient events, and sensor-data quality. "
        "affected: the exact target ids that truly deviate from the reference this window "
        "(empty list if none do); list a sensor only when its own numbers justify it. "
        "character: amplitude (level differs), spectral (frequency content shifted: high spec or df), "
        "impulsive (short shock), mixed. "
        "persistence: transient (this window only), sustained (across recent windows, see prev), "
        "growing (rising versus prev). "
        "data: ok, quality (sensor data untrustworthy) or insufficient (too little to judge). "
        "targets: rms/peak/ppv are target-to-reference ratios (near 1 = agreement), spec is spectral distance, "
        "df is the dominant-frequency shift in Hz. "
        "hx, when present, is this window as a multiple of that sensor's own normal median for metric hm: "
        "near 1 typical, large a real departure; absent while the hist_n history is small. "
        "ref_hx well above 1 means the reference itself is not calm and ratios understate deviations. "
        "prev is the previous decision and how many polls it persisted. "
        "confidence: your own 0-to-1 estimate of how clearly these measurements support your descriptor; "
        "scale it with the margin of the evidence and judge it fresh each window. "
        "Use only supplied measurements and make no unmeasured causal claims. "
        "summary is exactly one sentence of at most 18 words. "
        f"Input:{json.dumps(payload, separators=(',', ':'), sort_keys=True)}"
    )


def _compact_network_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    # Quality flags only. The rule's affected/significant/mild lists and label
    # code are deliberately NOT shown to the model: with the affected-array
    # output they are copy bait — the model echoed the rule's full list instead
    # of judging per-sensor numbers ("always all sensors", observed live).
    return {
        "q": assessment.get("quality_flags") or [],
    }


def _monitor_focus_sensor_reports(result: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    reports = list(result.get("sensor_agent_reports") or [])
    if not reports:
        return [], False

    assessment = result.get("network_assessment") or {}
    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    for key in ("affected_sensor_ids", "significant_sensor_ids", "mild_sensor_ids"):
        for sensor_id in _string_list(assessment.get(key)):
            if sensor_id not in seen_ids:
                selected_ids.append(sensor_id)
                seen_ids.add(sensor_id)

    for report in reports:
        sensor_id = str(report.get("sensor_id") or "").strip()
        if not sensor_id or sensor_id in seen_ids:
            continue
        label = str(report.get("small_agent_label") or "").strip()
        if report.get("quality_flags") or (label and label != "normal_relative_to_baseline"):
            selected_ids.append(sensor_id)
            seen_ids.add(sensor_id)

    if selected_ids:
        focus_reports = [
            report
            for report in reports
            if str(report.get("sensor_id") or "").strip() in seen_ids
        ]
    else:
        focus_reports = reports[: min(2, len(reports))]

    limit = _env_int_value("AGENT_MONITOR_MAX_TARGETS", DEFAULT_MONITOR_FOCUS_SENSOR_LIMIT, minimum=1)
    visible_reports = focus_reports[:limit]
    return visible_reports, len(reports) > len(visible_reports)


def _compact_monitor_sensor(report: dict[str, Any]) -> dict[str, Any]:
    # The model can only vary its confidence with the evidence it can see: give it the
    # target-to-reference comparison, not just the sensor id (ids alone made every
    # same-label window an identical prompt, so greedy decoding froze the confidence).
    comparison = report.get("baseline_comparison") or {}
    row: dict[str, Any] = {
        "id": report.get("sensor_id"),
        "rms": _round_monitor_value(comparison.get("rms_ratio")),
        "peak": _round_monitor_value(comparison.get("peak_ratio")),
        "ppv": _round_monitor_value(comparison.get("ppv_ratio")),
        "spec": _round_monitor_value(comparison.get("spectral_distance")),
        "df": _round_monitor_value(comparison.get("dominant_frequency_delta_hz")),
    }
    flags = _string_list(report.get("quality_flags"))
    if flags:
        row["q"] = flags[:3]
    history = report.get("history_context") or {}
    if history.get("x_median") is not None:
        # This window versus the sensor's own learned normal: hx = multiple of the
        # normal median for metric hm. The percentile stays out of the prompt (it
        # saturates and cannot be re-derived under perturbation; hx can).
        row["hx"] = history["x_median"]
        row["hm"] = history.get("metric")
    return row


def _round_monitor_value(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return float(f"{number:.3g}")


def _monitor_prompt_budget() -> tuple[int, int]:
    return (
        _env_int_value("AGENT_MONITOR_MAX_PROMPT_CHARS", DEFAULT_MONITOR_MAX_PROMPT_CHARS, minimum=128),
        _env_int_value(
            "AGENT_MONITOR_MAX_PROMPT_TOKENS",
            DEFAULT_MONITOR_MAX_PROMPT_ESTIMATED_TOKENS,
            minimum=32,
        ),
    )


def _estimate_prompt_tokens(text: str) -> int:
    compact = text.strip()
    if not compact:
        return 0
    return max(1, math.ceil(len(compact) / 3.0))


def _local_genie_timeout_override(model_base_url: str, timeout_s: float) -> dict[str, Any] | None:
    parsed = urlparse(model_base_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return None
    if parsed.port != DEFAULT_GENIE_PORT:
        return None
    return {"timeout_s": round(max(1.0, float(timeout_s)), 3)}


def _compact_text(value: Any, *, max_chars: int) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _is_placeholder_monitor_summary(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return normalized in {"", "no details available", "n/a", "none"}


def _round_float(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, 5)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _agent_llm_status(
    result: dict[str, Any],
    *,
    model_base_url: str,
    model: str,
    require_llm: bool,
) -> dict[str, Any]:
    sensor_reports = result.get("sensor_agent_reports") or []
    result_metadata = result.get("model_metadata") or {}
    if "monitor_llm_model_used" in result_metadata:
        monitor_used = bool(result_metadata.get("monitor_llm_model_used"))
        return {
            "role": "qwen_building_vibration_monitor",
            "mode": "single_llm_monitor_required" if require_llm else "single_llm_monitor_with_fallback",
            "llm_agent_active": monitor_used,
            "model": model,
            "base_url": model_base_url,
            "main_model_used": monitor_used,
            "small_model_used_count": 0,
            "small_model_expected_count": 0,
            "fallbacks_used": list(result_metadata.get("fallbacks_used") or []),
        }
    small_used_count = sum(
        1
        for report in sensor_reports
        if bool((report.get("model_metadata") or {}).get("model_used"))
    )
    expected_small_count = len(sensor_reports)
    network_metadata = (result.get("network_assessment") or {}).get("model_metadata") or {}
    main_model_used = bool(network_metadata.get("model_used") or result_metadata.get("main_model_used"))
    fallbacks_used = list(result_metadata.get("fallbacks_used") or [])
    all_small_used = expected_small_count > 0 and small_used_count == expected_small_count
    llm_agent_active = bool(main_model_used and all_small_used)
    return {
        "role": "qwen_building_vibration_monitor",
        "mode": "llm_required" if require_llm else "llm_with_deterministic_fallback",
        "llm_agent_active": llm_agent_active,
        "model": model,
        "base_url": model_base_url,
        "main_model_used": main_model_used,
        "small_model_used_count": small_used_count,
        "small_model_expected_count": expected_small_count,
        "fallbacks_used": fallbacks_used,
    }


def _build_agent_popup(result: dict[str, Any]) -> dict[str, Any]:
    assessment = result.get("network_assessment") or {}
    label = str(
        assessment.get("network_label")
        or assessment.get("main_agent_label")
        or assessment.get("python_rule_label")
        or "insufficient_data"
    )
    affected = [str(item) for item in assessment.get("affected_sensor_ids") or []]
    significant = [str(item) for item in assessment.get("significant_sensor_ids") or []]
    mild = [str(item) for item in assessment.get("mild_sensor_ids") or []]
    sensor_reports = result.get("sensor_agent_reports") or []
    anomalous_reports = [
        report
        for report in sensor_reports
        if str(report.get("small_agent_label") or "") in ANOMALOUS_SENSOR_LABELS
    ]
    anomaly = label in ANOMALOUS_NETWORK_LABELS or bool(anomalous_reports)
    severity = _agent_severity(label, significant, mild, anomalous_reports)
    confidence = _safe_float(assessment.get("main_agent_confidence"))
    label_uncertain = bool(assessment.get("label_uncertain"))
    vote_agreement = _safe_float(assessment.get("vote_agreement"))
    history = result.get("history_summary") or {}
    history_learning = bool(history.get("learning"))
    reference_history = result.get("reference_history") or {}
    descriptor = assessment.get("descriptor")
    descriptor_text = str(assessment.get("descriptor_text") or "").strip()
    explanation = str(assessment.get("explanation") or result.get("main_agent_explanation") or "").strip()
    sensor_text = ", ".join(affected[:6]) if affected else "none"
    confidence_text = f" Confidence {confidence * 100:.1f}%." if confidence is not None else ""
    if history.get("x_median") is not None:
        confidence_text += (
            f" Versus its own normal history: {history.get('sensor')} {history.get('metric')} at "
            f"{history['x_median']:.1f}x its normal median (p{history.get('percentile', 0):.1f} of "
            f"{history.get('n', 0)} learned windows)."
        )
    reference_x = _safe_float(reference_history.get("x_median"))
    if reference_x is not None and reference_x >= 2.0:
        confidence_text += (
            f" Caution: the reference sensor itself is at {reference_x:.1f}x its normal level, "
            "so target-to-reference ratios understate deviations."
        )
    if label_uncertain:
        agreement_text = (
            f" only {vote_agreement * 100:.0f}% of re-checks kept this label"
            if vote_agreement is not None
            else " re-checks disagreed on this label"
        )
        confidence_text += f" Label uncertain:{agreement_text}."
    headline = descriptor_text.capitalize() if descriptor_text else _labelize(label)
    message = (
        f"{headline}. Affected sensors: {sensor_text}.{confidence_text} {explanation}".strip()
        if anomaly
        else f"{headline}. No popup condition detected."
    )
    return {
        "show_popup": bool(anomaly),
        "severity": severity,
        "title": "Building Vibration Anomaly" if anomaly else "Building Vibration Normal",
        "message": message,
        "network_label": label,
        "confidence": confidence,
        "label_uncertain": label_uncertain,
        "vote_agreement": vote_agreement,
        "descriptor": descriptor,
        "descriptor_text": descriptor_text or None,
        "reference_x_median": reference_history.get("x_median"),
        "history_learning": history_learning,
        "history_n": history.get("n"),
        "history_percentile": history.get("percentile"),
        "history_x_median": history.get("x_median"),
        "history_sensor": history.get("sensor"),
        "history_metric": history.get("metric"),
        "affected_sensor_ids": affected,
        "significant_sensor_ids": significant,
        "mild_sensor_ids": mild,
        "quality_flags": list(assessment.get("quality_flags") or []),
        "dedupe_key": "|".join([label, ",".join(affected), ",".join(significant), ",".join(mild)]),
    }


def _agent_severity(
    label: str,
    significant: list[str],
    mild: list[str],
    anomalous_reports: list[dict[str, Any]],
) -> str:
    sensor_labels = {str(report.get("small_agent_label") or "") for report in anomalous_reports}
    if label == "multi_sensor_structure_wide_vibration_event" or len(significant) >= 2:
        return "critical"
    if label == "localized_vibration_deviation" or significant or sensor_labels & {
        "significant_local_deviation",
        "impulsive_or_shock_event",
    }:
        return "warning"
    if label == "sensor_network_quality_issue" or "possible_sensor_issue" in sensor_labels:
        return "quality"
    if label in {"mild_local_deviation", "mild_multi_sensor_deviation"} or mild:
        return "notice"
    return "normal"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _labelize(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("_", " ") if text else "--"


def _resolve_live_sensor_config_path(value: str | None) -> Path:
    raw = _clean(value) or os.environ.get("VIBRO_BUILDING_SENSOR_CONFIG") or DEFAULT_LIVE_SENSOR_CONFIG
    path = Path(raw).expanduser()
    if path.is_absolute():
        return _validate_live_sensor_config_path(path)

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return _validate_live_sensor_config_path(cwd_path)

    project_path = (Path(__file__).resolve().parents[2] / path).resolve()
    return _validate_live_sensor_config_path(project_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VibroAgent loaded-model webchat")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument(
        "--npu",
        action="store_true",
        default=_env_bool("WEBCHAT_START_GENIE", False),
        help="Start the Qualcomm Genie OpenAI-compatible adapter inside the webchat process.",
    )
    parser.add_argument("--genie-host", default=os.environ.get("GENIE_ADAPTER_HOST", DEFAULT_GENIE_HOST))
    parser.add_argument(
        "--genie-port",
        type=int,
        default=_env_int_value("GENIE_ADAPTER_PORT", DEFAULT_GENIE_PORT, minimum=1),
    )
    parser.add_argument("--qairt-sdk-root", default=os.environ.get("QAIRT_SDK_ROOT"))
    parser.add_argument("--genie-config", default=os.environ.get("GENIE_CONFIG") or str(DEFAULT_GENIE_CONFIG))
    parser.add_argument("--genie-runner", default=os.environ.get("GENIE_RUNNER"))
    parser.add_argument(
        "--genie-persistent",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("GENIE_PERSISTENT", True),
        help="Keep one loaded Genie dialog process alive between NPU chat requests.",
    )
    parser.add_argument(
        "--genie-persistent-runner",
        default=os.environ.get("GENIE_PERSISTENT_RUNNER"),
        help="Path to the persistent Genie helper binary. Built automatically when missing.",
    )
    parser.add_argument(
        "--allow-genie-non-htp",
        action="store_true",
        help="Allow --npu to serve a non-QnnHtp Genie config for adapter smoke tests.",
    )
    parser.add_argument("--genie-model", default=os.environ.get("GENIE_MODEL", DEFAULT_GENIE_MODEL))
    parser.add_argument(
        "--genie-prompt-format",
        choices=["plain", "qwen3"],
        default=os.environ.get("GENIE_PROMPT_FORMAT", DEFAULT_GENIE_PROMPT_FORMAT),
    )
    parser.add_argument(
        "--genie-timeout-s",
        type=float,
        default=_env_float_value("GENIE_TIMEOUT_S", 300.0, minimum=1.0),
    )
    parser.add_argument(
        "--genie-max-prompt-chars",
        type=int,
        default=_env_int_value("GENIE_ADAPTER_MAX_PROMPT_CHARS", 12000, minimum=1000),
    )
    parser.add_argument(
        "--genie-max-output-tokens",
        type=int,
        default=_env_int_value(
            "GENIE_MAX_OUTPUT_TOKENS",
            DEFAULT_GENIE_MAX_OUTPUT_TOKENS,
            minimum=16,
        ),
    )
    args = parser.parse_args()

    runtime = None
    if args.npu:
        runtime = _start_embedded_genie_runtime(
            host=args.genie_host,
            port=args.genie_port,
            sdk_root=args.qairt_sdk_root,
            config_path=args.genie_config,
            runner_path=args.genie_runner,
            persistent=args.genie_persistent,
            persistent_runner_path=args.genie_persistent_runner,
            model=args.genie_model,
            prompt_format=args.genie_prompt_format,
            timeout_s=args.genie_timeout_s,
            max_prompt_chars=args.genie_max_prompt_chars,
            max_output_tokens=args.genie_max_output_tokens,
            require_qnn_htp=not args.allow_genie_non_htp,
        )
        if runtime.serving:
            os.environ["QWEN_BASE_URL"] = runtime.base_url
            os.environ["QWEN_API_KEY"] = os.environ.get("QWEN_API_KEY", "EMPTY")
            os.environ["QWEN_MODEL"] = runtime.model

    httpd = ThreadingHTTPServer((args.host, args.port), WebchatHandler)
    httpd.embedded_genie_runtime = runtime  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"VibroAgent webchat running at {url}")
    if runtime is not None:
        if runtime.serving:
            print(f"Embedded Genie adapter running at {runtime.base_url}")
            print(f"Genie model: {runtime.model}")
            print(f"Genie config: {runtime.config_path}")
            print(f"Genie persistent mode: {runtime.persistent}")
            if runtime.persistent_runner_path:
                print(f"Genie persistent runner: {runtime.persistent_runner_path}")
            print(f"NPU offload detected: {runtime.npu_offload}")
            if runtime.warning:
                print(f"NPU warning: {runtime.warning}")
        else:
            print(f"Embedded Genie adapter failed: {runtime.startup_error}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if runtime is not None:
            runtime.shutdown()


_NPU_CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NPU Chat</title>
  <style>
    .assistant-md .md-body p { margin: 0 0 8px; }
    .assistant-md .md-body p:last-child { margin-bottom: 0; }
    .assistant-md .md-body h3 { font-size: 14px; margin: 8px 0 4px; }
    .assistant-md .md-body h4 { font-size: 13px; margin: 8px 0 4px; }
    .assistant-md .md-body ul.md-ul { margin: 4px 0 8px; padding-left: 18px; }
    .assistant-md .md-body li { margin: 2px 0; }
    .md-code { background: rgba(4,105,177,0.12); padding: 1px 5px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
    .md-pre { margin: 8px 0; padding: 10px 12px; background: #eef3f9; border: 1px solid #d6dee7; border-radius: 3px; overflow-x: auto; }
    .md-pre code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre; color: #26312d; }

    :root {
      color-scheme: light;
      --bg: #eef1f5;
      --panel: #ffffff;
      --line: #d6dee7;
      --ink: #1a2433;
      --muted: #5d6b7c;
      --accent: #0469b1;
      --accent-soft: #e1eef9;
      --user: #03234e;
      --error: #d0021b;
      --shadow: 0 14px 36px rgba(3, 35, 78, 0.10);
      --st-blue: #0469b1;
      --st-navy: #03234e;
      --st-blue-line: #0469b1;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 "Helvetica Neue", Helvetica, Arial, "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
    }
    body::before { display: none; content: none; }
    .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
    .mark { width: 58px; height: 44px; display: grid; place-items: center; flex: 0 0 auto; background: transparent; }
    .mark img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .subtitle { margin-top: 2px; color: var(--muted); font-size: 12px; }
    .stfoot { background: var(--st-navy); color: #cdd7e4; font-size: 12px; display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 0; border-image: none; margin: 0 -18px -18px; padding: 9px 18px; }
    .stfoot strong { color: #fff; font-weight: 700; }
    .stfoot .tag { color: #8aa0bd; letter-spacing: 0.3px; }
    button, textarea, input { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto auto;
      max-width: 920px;
      margin: 0 auto;
      padding: 18px;
      gap: 12px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 44px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 720;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      min-width: 0;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #9aa5a1;
      flex: 0 0 auto;
    }
    .dot.ok { background: var(--accent); }
    .dot.bad { background: var(--error); }
    .status span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: min(52vw, 520px);
    }
    main {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      min-height: 0;
      display: grid;
      grid-template-rows: 1fr;
    }
    .messages {
      overflow: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .message {
      width: fit-content;
      max-width: min(76ch, 86%);
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #fbfcfb;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      align-self: flex-end;
      background: #eef3ff;
      border-color: #cbd8f5;
      color: #172d5c;
    }
    .message.assistant {
      align-self: flex-start;
    }
    .message.error {
      align-self: center;
      background: #fff0f0;
      border-color: #edc4c4;
      color: var(--error);
    }
    .speed {
      margin-top: 8px;
      padding-top: 7px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      white-space: normal;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: end;
      padding: 12px 0 0;
    }
    textarea {
      min-height: 46px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 11px 12px;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    button {
      min-height: 46px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 0 16px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      transition: background .15s ease;
    }
    button.primary:hover { background: #035a99; border-color: #035a99; }
    button:disabled { cursor: not-allowed; opacity: .62; }
    footer {
      display: grid;
      gap: 8px;
    }
    .meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }
    .meta span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    @media (max-width: 640px) {
      .app { padding: 12px; }
      header { align-items: flex-start; flex-direction: column; }
      .status span:last-child { max-width: 100%; }
      main { min-height: 55vh; }
      form { grid-template-columns: 1fr; }
      button { width: 100%; }
      .message { max-width: 100%; }
      .meta { flex-direction: column; gap: 2px; }
    }


    /* ST-inspired direct NPU console. */
    :root {
      --bg: #f3f5f7;
      --panel: #ffffff;
      --line: #d8e0e8;
      --ink: #17243a;
      --muted: #617084;
      --accent: #0469b1;
      --accent-soft: #eaf4fb;
      --shadow: 0 3px 12px rgba(3, 35, 78, 0.08);
    }
    body::before { display: none; content: none; }
    .app {
      max-width: none;
      min-height: 100vh;
      padding: 0;
      gap: 0;
      grid-template-rows: 68px minmax(0, 1fr) auto 34px;
    }
    header {
      width: 100%;
      min-height: 68px;
      padding: 0 24px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(3,35,78,.04);
    }
    .brand { gap: 14px; }
    .mark {
      width: 58px;
      height: 44px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      background: transparent;
    }
    .mark img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    h1 { font-size: 20px; font-weight: 750; letter-spacing: -.2px; }
    .subtitle { margin-top: 3px; }
    .top-actions { display: flex; align-items: center; gap: 18px; }
    .app-nav { display: flex; align-items: center; gap: 18px; }
    .nav-link { display: inline-flex; align-items: center; min-height: 65px; padding: 0 2px; border-bottom: 3px solid transparent; color: var(--st-navy); text-decoration: none; font-size: 13px; font-weight: 700; }
    .nav-link:hover { color: var(--st-blue); border-bottom-color: var(--st-blue); }
    .status { min-height: 32px; padding: 5px 10px; border: 1px solid var(--line); border-radius: 999px; background: #f8fafc; font-size: 12px; }

    main {
      width: calc(100% - 48px);
      max-width: 1180px;
      min-height: 0;
      margin: 16px auto 0;
      border-radius: 7px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .messages { padding: 24px; gap: 14px; background: #fff; }
    .message { max-width: min(78ch, 84%); padding: 12px 14px; border-radius: 4px; background: #fff; }
    .message.assistant { border-left: 3px solid var(--st-cyan, #00a3e0); }
    .message.user { border-right: 3px solid var(--st-blue); background: #eaf4fb; color: var(--st-navy); }
    .message.error { border-left: 3px solid var(--error); }

    footer {
      width: calc(100% - 48px);
      max-width: 1180px;
      margin: 12px auto 16px;
      padding: 12px;
      gap: 9px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      box-shadow: var(--shadow);
    }
    form { padding: 0; }
    textarea { min-height: 52px; border-radius: 4px; resize: none; }
    textarea:focus { border-color: var(--st-blue); box-shadow: 0 0 0 3px rgba(4,105,177,.12); }
    button { min-height: 42px; border-radius: 3px; }
    button.primary { background: var(--st-blue); border-color: var(--st-blue); }
    .meta { padding: 0 2px; }
    .stfoot { width: 100%; margin: 0; padding: 8px 24px; }

    @media (max-width: 720px) {
      .app { grid-template-rows: auto minmax(0, 1fr) auto auto; }
      header { padding: 12px 16px; align-items: flex-start; }
      .top-actions { width: 100%; flex-wrap: wrap; justify-content: space-between; gap: 8px; }
      .app-nav { gap: 12px; }
      .nav-link { min-height: 34px; }
      main, footer { width: calc(100% - 24px); }
      main { margin-top: 12px; }
      footer { margin-bottom: 12px; }
      .messages { padding: 16px; }
      .message { max-width: 96%; }
    }

    /* ST logo. */
    body::before { display: none !important; content: none !important; }
    .mark {
      width: 112px !important;
      height: 48px !important;
      min-width: 112px !important;
      padding: 0 !important;
      border: 0 !important;
      border-radius: 0 !important;
      background: transparent !important;
      color: transparent !important;
      font-size: 0 !important;
      letter-spacing: 0 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
    }
    .mark::after { display: none !important; content: none !important; }
    .st-logo {
      display: block;
      width: 112px;
      max-width: 112px;
      max-height: 48px;
      height: auto;
      object-fit: contain;
    }
    .stfoot { border-top: 0 !important; border-image: none !important; }

  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark"><img class="st-logo" src="/assets/st-logo.png" alt="STMicroelectronics logo"></div>
        <div>
          <h1>VibroAgent</h1>
          <div class="subtitle">Direct NPU console</div>
        </div>
      </div>
      <div class="top-actions">
        <nav class="app-nav" aria-label="Application navigation">
          <a class="nav-link" href="/">Analysis chat</a>
          <a class="nav-link" href="/graph">Live data</a>
        </nav>
        <div class="status"><span id="dot" class="dot"></span><span id="status">Connecting</span></div>
      </div>
    </header>
    <main>
      <div id="messages" class="messages" aria-live="polite"></div>
    </main>
    <footer>
      <form id="chatForm">
        <textarea id="input" required placeholder="Ask the local model about registered building-vibration sensors"></textarea>
        <button id="clear" type="button">Clear</button>
        <button id="send" class="primary" type="submit">Send</button>
      </form>
      <div class="meta"><span id="model">Model</span><span id="endpoint">Endpoint</span></div>
    </footer>
    <div class="stfoot"><span><strong>STMicroelectronics</strong> &middot; VibroAgent edge LLM</span><span class="tag">life.augmented</span></div>
  </div>
  <script>
    const els = {
      dot: document.getElementById("dot"),
      status: document.getElementById("status"),
      messages: document.getElementById("messages"),
      form: document.getElementById("chatForm"),
      input: document.getElementById("input"),
      send: document.getElementById("send"),
      clear: document.getElementById("clear"),
      model: document.getElementById("model"),
      endpoint: document.getElementById("endpoint"),
    };

    let messages = [
      { role: "system", content: "You are a building-vibration monitoring assistant. Discuss only registered sensors, reference comparisons, vibration metrics, frequency content, transient events, data quality, and monitoring actions." }
    ];

    function setStatus(kind, text) {
      els.dot.className = "dot" + (kind ? " " + kind : "");
      els.status.textContent = text;
    }

    function addMessage(role, text) {
      const node = document.createElement("div");
      node.className = "message " + role;
      node.textContent = text;
      els.messages.appendChild(node);
      els.messages.scrollTop = els.messages.scrollHeight;
      return node;
    }

    function renderMarkdownInline(text) {
      text = text.replace(/`([^`\n]+)`/g, (match, code) => `<code class="md-code">${code}</code>`);
      text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
      text = text
        .replace(/^\s*#{4,6}\s+(.*)$/gm, "<h4>$1</h4>")
        .replace(/^\s*###\s+(.*)$/gm, "<h4>$1</h4>")
        .replace(/^\s*##\s+(.*)$/gm, "<h3>$1</h3>")
        .replace(/^\s*#\s+(.*)$/gm, "<h3>$1</h3>");
      text = text.replace(/(?:^|\n)((?:[ \t]*[-*][ \t]+.*(?:\n|$))+)/g, (match, items) => {
        const lis = items.replace(/\n+$/, "").split(/\n/).map((line) => `<li>${line.replace(/^[ \t]*[-*][ \t]+/, "")}</li>`).join("");
        return `\n<ul class="md-ul">${lis}</ul>`;
      });
      text = text.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
      text = text
        .replace(/<br>\s*(<(?:ul|h3|h4)[\s>])/g, "$1")
        .replace(/(<\/(?:ul|h3|h4)>)\s*<br>/g, "$1");
      return `<p>${text}</p>`
        .replace(/<p>\s*(<(?:ul|h3|h4)[\s>])/g, "$1")
        .replace(/(<\/(?:ul|h3|h4)>)\s*<\/p>/g, "$1")
        .replace(/<p>(?:\s|<br>)*<\/p>/g, "");
    }

    function renderMarkdown(src) {
      const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const parts = String(src == null ? "" : src).split(/```/);
      let out = "";
      for (let i = 0; i < parts.length; i += 1) {
        if (i % 2 === 1) {
          out += `<pre class="md-pre"><code>${esc(parts[i].replace(/^\n/, "").replace(/\s+$/, ""))}</code></pre>`;
        } else {
          out += renderMarkdownInline(esc(parts[i]));
        }
      }
      return out;
    }

    function revealMarkdown(node, text) {
      const full = String(text == null ? "" : text);
      node.classList.add("assistant-md");
      let body = node.querySelector(".md-body");
      if (!body) { node.textContent = ""; body = document.createElement("div"); body.className = "md-body"; node.appendChild(body); }
      if (node._reveal) { clearInterval(node._reveal); node._reveal = null; }
      const scroll = () => { els.messages.scrollTop = els.messages.scrollHeight; };
      const tokens = full.split(/(\s+)/);
      if (tokens.length <= 3) { body.innerHTML = renderMarkdown(full); scroll(); return; }
      let i = 0;
      const step = Math.max(1, Math.ceil(tokens.length / 24));
      node._reveal = setInterval(() => {
        i += step;
        body.innerHTML = renderMarkdown(tokens.slice(0, Math.min(i, tokens.length)).join(""));
        scroll();
        if (i >= tokens.length) { clearInterval(node._reveal); node._reveal = null; body.innerHTML = renderMarkdown(full); }
      }, 18);
    }

    function appendSpeed(node, metrics) {
      const tokens = metrics && Number.isFinite(Number(metrics.completion_tokens)) ? Number(metrics.completion_tokens) : null;
      const totalSeconds = metrics && Number.isFinite(Number(metrics.duration_s)) ? Number(metrics.duration_s) : null;
      const genSeconds = metrics && Number.isFinite(Number(metrics.generation_s)) ? Number(metrics.generation_s) : null;
      const rate = metrics && Number.isFinite(Number(metrics.tokens_per_second)) ? Number(metrics.tokens_per_second) : null;
      if (tokens === null || rate === null) return;
      const parts = [`${rate.toFixed(2)} tok/s`, `${tokens} tokens`];
      if (genSeconds !== null) parts.push(`${genSeconds.toFixed(2)} s gen`);
      if (totalSeconds !== null && (genSeconds === null || Math.abs(totalSeconds - genSeconds) > 0.05)) {
        parts.push(`${totalSeconds.toFixed(2)} s total`);
      }
      const meta = document.createElement("div");
      meta.className = "speed";
      meta.textContent = parts.join(" · ");
      node.appendChild(meta);
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    async function loadConfig() {
      try {
        const res = await fetch("/api/config");
        const data = await res.json();
        els.model.textContent = data.model || "Model";
        els.endpoint.textContent = data.base_url || "Endpoint";
        if (data.npu && data.npu.serving && data.npu.npu_offload) {
          setStatus("ok", "QnnHtp ready");
        } else if (data.npu && data.npu.enabled) {
          setStatus("bad", data.npu.startup_error || "NPU not ready");
        } else {
          setStatus("", "Using configured endpoint");
        }
      } catch (err) {
        setStatus("bad", "Config unavailable");
      }
    }

    async function sendMessage(text) {
      messages.push({ role: "user", content: text });
      addMessage("user", text);
      const pending = addMessage("assistant", "Thinking...");
      els.send.disabled = true;
      els.clear.disabled = true;
      setStatus("", "Running");
      try {
        const res = await fetch("/api/npu-chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages,
            timeout_s: 300,
            max_tokens: 256,
            max_prompt_chars: 9000,
          })
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.message || data.error || "Request failed");
        revealMarkdown(pending, data.answer || "");
        appendSpeed(pending, data.metrics);
        messages = data.messages || messages;
        messages.push({ role: "assistant", content: data.answer || "" });
        els.model.textContent = data.model || els.model.textContent;
        els.endpoint.textContent = data.base_url || els.endpoint.textContent;
        const rate = data.metrics && Number.isFinite(Number(data.metrics.tokens_per_second))
          ? ` · ${Number(data.metrics.tokens_per_second).toFixed(2)} tok/s`
          : "";
        setStatus(data.npu && data.npu.npu_offload ? "ok" : "", data.npu && data.npu.npu_offload ? `QnnHtp ready${rate}` : `Ready${rate}`);
      } catch (err) {
        pending.remove();
        messages.pop();
        addMessage("error", String(err.message || err));
        setStatus("bad", "Request failed");
      } finally {
        els.send.disabled = false;
        els.clear.disabled = false;
        els.input.focus();
      }
    }

    els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = els.input.value.trim();
      if (!text) return;
      els.input.value = "";
      sendMessage(text);
    });

    els.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        els.form.requestSubmit();
      }
    });

    els.clear.addEventListener("click", () => {
      messages = [messages[0]];
      els.messages.innerHTML = "";
      els.input.focus();
    });

    loadConfig();
  </script>
</body>
</html>
"""

_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VibroAgent Webchat</title>
  <style>
    .assistant-md .md-body p { margin: 0 0 8px; }
    .assistant-md .md-body p:last-child { margin-bottom: 0; }
    .assistant-md .md-body h3 { font-size: 14px; margin: 8px 0 4px; }
    .assistant-md .md-body h4 { font-size: 13px; margin: 8px 0 4px; }
    .assistant-md .md-body ul.md-ul { margin: 4px 0 8px; padding-left: 18px; }
    .assistant-md .md-body li { margin: 2px 0; }
    .md-code { background: rgba(4,105,177,0.12); padding: 1px 5px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
    .md-pre { margin: 8px 0; padding: 10px 12px; background: #eef3f9; border: 1px solid #d6dee7; border-radius: 3px; overflow-x: auto; }
    .md-pre code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre; color: #26312d; }

    :root {
      color-scheme: light;
      --paper: #eef1f5;
      --surface: #ffffff;
      --line: #d6dee7;
      --ink: #1a2433;
      --muted: #5d6b7c;
      --teal: #0469b1;
      --teal-2: #e1eef9;
      --rust: #03234e;
      --amber: #f6a800;
      --bad: #d0021b;
      --shadow: 0 10px 28px rgba(3, 35, 78, 0.10);
      --st-blue: #0469b1;
      --st-cyan: #00a3e0;
      --st-navy: #03234e;
      --st-blue-line: #0469b1;
    }

    * { box-sizing: border-box; }

    html,
    body {
      height: 100%;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
      font: 14px/1.5 "Helvetica Neue", Helvetica, Arial, "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }

    body::before { display: none; content: none; }

    button, input, textarea {
      font: inherit;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 28px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }

    .mark {
      width: 58px;
      height: 44px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      background: transparent;
    }
    .mark img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 720;
    }

    .subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .nav-link {
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      padding: 0 4px;
      border: 0;
      border-bottom: 3px solid transparent;
      border-radius: 0;
      color: var(--st-navy);
      text-decoration: none;
      font-weight: 700;
      background: transparent;
      white-space: nowrap;
      letter-spacing: 0.2px;
    }
    .nav-link:hover { color: var(--st-blue); border-bottom-color: var(--st-blue); }

    .stfoot {
      grid-column: 1 / -1;
      background: var(--st-navy);
      color: #cdd7e4;
      font-size: 12px;
      padding: 9px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 0;
      border-image: none;
    }
    .stfoot a { color: #fff; text-decoration: none; }
    .stfoot strong { color: #fff; font-weight: 700; }
    .stfoot .tag { color: #8aa0bd; letter-spacing: 0.3px; }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--amber);
      box-shadow: 0 0 0 3px rgba(242, 195, 93, 0.25);
    }

    .dot.ok {
      background: var(--teal);
      box-shadow: 0 0 0 3px rgba(40, 102, 92, 0.2);
    }

    .dot.bad {
      background: var(--bad);
      box-shadow: 0 0 0 3px rgba(181, 61, 61, 0.18);
    }

    main {
      display: grid;
      grid-template-columns: minmax(440px, 1fr) minmax(300px, 336px);
      gap: 12px;
      padding: 12px;
      min-height: 0;
      overflow: hidden;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 6px 20px rgba(3, 35, 78, 0.08);
      min-height: 0;
    }

    .chat {
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }

    .connection-details {
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }

    .connection-details > summary {
      min-height: 42px;
      padding: 0 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--st-navy);
      cursor: pointer;
      font-weight: 700;
      list-style: none;
      user-select: none;
    }

    .connection-details > summary::-webkit-details-marker { display: none; }

    .connection-details > summary::after {
      content: "▾";
      color: var(--muted);
      font-size: 12px;
      transition: transform .15s ease;
    }

    .connection-details[open] > summary::after { transform: rotate(180deg); }

    .connection-summary {
      margin-left: auto;
      max-width: min(42vw, 520px);
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .settings {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(170px, 0.9fr) minmax(170px, 0.9fr) auto;
      gap: 10px;
      padding: 4px 14px 14px;
      background: #f8fafc;
    }

    .settings-links {
      grid-column: 1 / -1;
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }

    .settings-links a { color: var(--st-blue); text-decoration: none; font-weight: 650; }

    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }

    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 2px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      outline: none;
    }

    input:focus, textarea:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(40, 102, 92, 0.13);
    }

    .check {
      align-self: end;
      height: 38px;
      padding: 0 12px;
      border: 1px solid var(--teal);
      border-radius: 2px;
      background: var(--teal-2);
      color: var(--teal);
      cursor: pointer;
      font-weight: 650;
    }

    .messages {
      overflow: auto;
      padding: 18px 18px 10px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 320px;
      background: #ffffff;
    }

    .message {
      max-width: min(760px, 88%);
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      white-space: pre-wrap;
      box-shadow: 0 2px 8px rgba(3, 35, 78, 0.04);
    }

    .message.user {
      align-self: flex-end;
      background: var(--teal);
      color: #fff;
      border-color: var(--teal);
    }

    .message.assistant {
      align-self: flex-start;
    }

    .message.error {
      border-color: rgba(181, 61, 61, 0.35);
      background: #fff3f0;
      color: #7e2b2b;
    }

    .composer {
      padding: 10px 12px 12px;
      border-top: 1px solid var(--line);
      background: #fff;
      display: grid;
      gap: 8px;
    }

    .quick {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .quick button {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
      color: var(--ink);
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 650;
    }

    .quick button:hover { border-color: #9bbbd4; background: var(--teal-2); color: var(--st-blue); }

    .send-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    textarea {
      min-height: 52px;
      max-height: 168px;
      resize: none;
      border-radius: 6px;
    }

    .send {
      height: 42px;
      min-width: 96px;
      border: 0;
      border-radius: 6px;
      background: var(--st-blue);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      letter-spacing: 0.2px;
      transition: background .15s ease;
    }
    .send:hover { background: #035a99; }

    .send:disabled,
    .check:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    .diagnostics {
      display: grid;
      grid-template-rows: auto auto 1fr;
      overflow: hidden;
    }

    .side-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .side-head h2 {
      margin: 0;
      font-size: 15px;
    }

    .side-title {
      min-width: 0;
    }

    .side-title span {
      display: block;
      margin-top: 1px;
      color: var(--muted);
      font-size: 11px;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      font-size: 12px;
      background: #fff;
    }

    .chip.ok { color: #17643f; border-color: #a9d7bd; background: #f1fbf5; }
    .chip.warn { color: #8a5a00; border-color: #ead19c; background: #fff9e9; }
    .chip.bad { color: #9b1c2f; border-color: #e1aeb7; background: #fff3f5; }

    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }

    .metric {
      border-left: 3px solid var(--line);
      padding: 2px 0 2px 9px;
      min-width: 0;
    }

    .metric strong {
      display: block;
      font-size: 17px;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
    }

    .metric.primary {
      border-color: var(--teal);
      grid-column: 1 / -1;
    }

    .visual {
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }

    canvas {
      width: 100%;
      display: block;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }

    .feature-canvas { height: 112px; flex: 0 0 112px; }

    .spectrum-launch {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 58px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .spectrum-launch-copy {
      display: grid;
      gap: 2px;
      min-width: 0;
    }

    .spectrum-launch-copy strong { color: var(--ink); }

    .spectrum-launch-copy span {
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .psd-open-action {
      flex: 0 0 auto;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid var(--st-blue);
      border-radius: 2px;
      background: var(--teal-2);
      color: var(--st-blue);
      cursor: pointer;
      font-weight: 700;
      white-space: nowrap;
    }

    .psd-floating {
      position: fixed;
      z-index: 1000;
      top: 70px;
      left: 50%;
      width: min(1040px, calc(100vw - 36px));
      height: min(790px, calc(100vh - 92px));
      min-width: 560px;
      min-height: 460px;
      transform: translateX(-50%);
      resize: both;
      overflow: hidden;
      border: 1px solid #aebdcd;
      border-radius: 5px;
      background: var(--surface);
      box-shadow: 0 24px 70px rgba(3, 35, 78, 0.28);
    }

    .psd-floating[hidden] { display: none !important; }

    .psd-floating.maximized {
      top: 10px !important;
      left: 10px !important;
      width: calc(100vw - 20px) !important;
      height: calc(100vh - 20px) !important;
      transform: none !important;
      resize: none;
    }

    .psd-floating-header {
      height: 46px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 10px 0 14px;
      border-bottom: 1px solid var(--line);
      background: var(--st-navy);
      color: #fff;
      cursor: move;
      user-select: none;
      touch-action: none;
    }

    .psd-floating-heading {
      min-width: 0;
      display: flex;
      align-items: baseline;
      gap: 10px;
    }

    .psd-floating-heading strong { font-size: 14px; }

    .psd-floating-heading span {
      color: #c9d8e9;
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .psd-window-actions {
      display: flex;
      gap: 6px;
      flex: 0 0 auto;
    }

    .psd-window-action {
      height: 28px;
      padding: 0 9px;
      border: 1px solid rgba(255,255,255,0.42);
      border-radius: 2px;
      background: rgba(255,255,255,0.08);
      color: #fff;
      cursor: pointer;
      font-size: 11px;
      font-weight: 700;
    }

    .psd-floating-body {
      height: calc(100% - 46px);
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(130px, 0.42fr) minmax(260px, 1fr) auto;
      overflow: hidden;
    }

    .psd-chart-block {
      min-height: 0;
      display: grid;
      grid-template-rows: 28px 1fr;
      overflow: hidden;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }

    .psd-chart-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 0 10px;
      color: var(--muted);
      background: #f5f8fb;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      font-weight: 650;
    }

    .waveform-canvas,
    .spectrum-canvas {
      width: 100%;
      height: 100%;
      min-height: 0;
      border-bottom: 0;
    }

    .spectrum-head {
      min-height: 38px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 7px 10px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .spectrum-title {
      display: grid;
      min-width: 0;
      color: var(--ink);
      font-weight: 700;
      line-height: 1.2;
    }

    .psd-summary {
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .psd-controls {
      display: flex;
      gap: 6px;
      align-items: end;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    .psd-control {
      display: grid;
      gap: 2px;
      color: var(--muted);
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .psd-control select {
      height: 28px;
      min-width: 72px;
      border: 1px solid var(--line);
      border-radius: 2px;
      background: #fff;
      color: var(--ink);
      font: 600 11px system-ui, sans-serif;
      padding: 0 5px;
    }

    .psd-control input {
      width: 82px;
      height: 28px;
      padding: 0 6px;
      border: 1px solid var(--line);
      border-radius: 2px;
      background: #fff;
      color: var(--ink);
      font: 600 11px system-ui, sans-serif;
    }

    .psd-filter-note {
      flex-basis: 100%;
      color: var(--muted);
      font-size: 10px;
      font-weight: 500;
      text-transform: none;
      letter-spacing: 0;
    }

    .psd-action {
      height: 28px;
      border: 1px solid var(--st-blue);
      border-radius: 2px;
      background: var(--teal-2);
      color: var(--st-blue);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
      padding: 0 12px;
    }

    .psd-action:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    .psd-insights {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
      min-height: 34px;
      padding: 6px 10px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      font-size: 11px;
    }

    .psd-pill {
      display: inline-flex;
      gap: 4px;
      align-items: center;
      border: 1px solid #cad6e2;
      border-radius: 999px;
      background: #f7fafc;
      color: #29435f;
      padding: 3px 7px;
      white-space: nowrap;
    }

    .psd-pill strong { color: var(--st-blue); }

    .technical-details {
      border-top: 1px solid var(--line);
      background: #fbfcfe;
    }

    .technical-details > summary {
      min-height: 42px;
      padding: 0 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      cursor: pointer;
      color: var(--st-navy);
      font-size: 12px;
      font-weight: 700;
      list-style: none;
    }

    .technical-details > summary::-webkit-details-marker { display: none; }

    .technical-details > summary::before {
      content: "+";
      width: 18px;
      height: 18px;
      display: inline-grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }

    .technical-details[open] > summary::before { content: "−"; }

    .technical-details-body {
      border-top: 1px solid var(--line);
    }

    .call-box {
      max-height: 150px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }

    #payload { max-height: 260px; }

    pre {
      margin: 0;
      padding: 12px 14px;
      overflow: auto;
      color: #34413c;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }

    @media (max-width: 1040px) {
      body {
        overflow: auto;
      }

      .app {
        height: auto;
        min-height: 100vh;
      }

      main {
        grid-template-columns: 1fr;
        overflow: visible;
      }

      .diagnostics { min-height: 440px; }
    }

    @media (max-width: 760px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      main {
        padding: 10px;
      }

      .settings {
        grid-template-columns: 1fr;
      }

      .message {
        max-width: 96%;
      }

      .send-row {
        grid-template-columns: 1fr;
      }

      .send {
        width: 100%;
      }

      .psd-floating,
      .psd-floating.maximized {
        top: 8px !important;
        left: 8px !important;
        width: calc(100vw - 16px) !important;
        height: calc(100vh - 16px) !important;
        min-width: 0;
        min-height: 0;
        transform: none !important;
        resize: none;
      }

      .psd-floating-body {
        grid-template-rows: auto minmax(110px, 0.35fr) minmax(230px, 1fr) auto;
      }

      .spectrum-head {
        grid-template-columns: 1fr;
        max-height: 210px;
        overflow-y: auto;
      }

      .psd-controls { justify-content: flex-start; }

      .psd-window-action:first-child { display: none; }
    }


    /* ST-inspired application UI: restrained spacing, white surfaces and blue navigation. */
    :root {
      --paper: #f3f5f7;
      --surface: #ffffff;
      --line: #d8e0e8;
      --ink: #17243a;
      --muted: #617084;
      --teal: #0469b1;
      --teal-2: #eaf4fb;
      --shadow: 0 3px 12px rgba(3, 35, 78, 0.08);
      --st-blue: #0469b1;
      --st-cyan: #00a3e0;
      --st-navy: #03234e;
    }

    body { background: var(--paper); }
    body::before { display: none; content: none; }
    .app { grid-template-rows: 68px minmax(0, 1fr) 34px; }

    header {
      min-height: 68px;
      padding: 0 24px;
      box-shadow: 0 1px 0 rgba(3, 35, 78, 0.04);
    }

    .brand { gap: 14px; }
    .mark {
      width: 58px;
      height: 44px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      background: transparent;
    }
    .mark img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    h1 { font-size: 20px; font-weight: 750; letter-spacing: -0.2px; }
    .subtitle { margin-top: 3px; font-size: 12px; }
    .top-actions { gap: 18px; }
    .app-nav { display: flex; align-items: center; gap: 18px; }
    .nav-link { min-height: 65px; padding: 0 2px; font-size: 13px; }
    .status {
      min-height: 32px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
      font-size: 12px;
    }

    main {
      width: 100%;
      max-width: 1600px;
      margin: 0 auto;
      grid-template-columns: minmax(0, 1fr) minmax(292px, 320px);
      gap: 16px;
      padding: 16px 24px;
    }

    .panel {
      border-radius: 7px;
      box-shadow: var(--shadow);
    }

    .connection-details { background: #fff; }
    .connection-details > summary {
      min-height: 44px;
      padding: 0 16px;
      border-left: 3px solid var(--st-blue);
      font-size: 13px;
    }
    .settings {
      grid-template-columns: minmax(220px, 1.4fr) minmax(180px, 1fr) minmax(160px, .8fr) auto;
      padding: 14px 16px 16px;
      background: #f7f9fb;
    }
    .settings-links { display: none; }

    input, textarea, select { border-radius: 3px; }
    input:focus, textarea:focus, select:focus {
      border-color: var(--st-blue);
      box-shadow: 0 0 0 3px rgba(4, 105, 177, 0.12);
    }
    .check {
      min-width: 126px;
      border-radius: 3px;
      background: #fff;
    }

    .messages {
      padding: 22px 22px 14px;
      gap: 14px;
      background: #fff;
    }
    .message {
      max-width: min(780px, 84%);
      padding: 12px 14px;
      border-radius: 4px;
      box-shadow: none;
    }
    .message.assistant {
      border-left: 3px solid var(--st-cyan);
      background: #fff;
    }
    .message.user {
      border-color: #bed8ec;
      border-right: 3px solid var(--st-blue);
      background: #eaf4fb;
      color: var(--st-navy);
    }
    .message.error { border-left: 3px solid var(--bad); }

    .composer { padding: 11px 14px 14px; gap: 10px; }
    .quick { gap: 7px; }
    .quick button {
      border-radius: 3px;
      background: #fff;
      padding: 6px 10px;
      font-weight: 650;
    }
    .quick button:hover { border-color: var(--st-blue); }
    textarea { border-radius: 4px; }
    .send {
      min-width: 104px;
      border-radius: 3px;
      box-shadow: 0 2px 4px rgba(3, 35, 78, 0.12);
    }

    .diagnostics { grid-template-rows: auto auto minmax(0, 1fr); }
    .side-head {
      min-height: 62px;
      padding: 13px 15px;
      border-top: 0;
      background: #fff;
    }
    .side-head h2 { font-size: 16px; }
    .side-title span { margin-top: 3px; }
    .metrics { gap: 0; padding: 0; }
    .metric {
      min-height: 69px;
      padding: 13px 14px;
      border-left: 0;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .metric:nth-child(odd):not(.primary) { border-right: 0; }
    .metric strong { margin-top: 4px; font-size: 16px; }
    .metric.primary {
      min-height: 78px;
      padding-left: 15px;
      border-right: 0;
      border-left: 3px solid var(--st-blue);
      background: #f6fbff;
    }

    canvas { background: #fff; }
    .feature-canvas { height: 104px; flex-basis: 104px; }
    .feature-canvas[hidden] { display: none !important; }
    .spectrum-launch {
      min-height: 70px;
      margin: 12px;
      padding: 11px 12px;
      border: 1px solid #c9d9e7;
      border-radius: 5px;
      background: #f7fbfe;
    }
    .spectrum-launch-copy strong { color: var(--st-navy); }
    .psd-open-action {
      min-height: 36px;
      border-radius: 3px;
      background: var(--st-blue);
      color: #fff;
    }
    .psd-open-action:hover { background: #035a99; }

    .technical-details { margin-top: auto; background: #fff; }
    .technical-details > summary { min-height: 45px; padding: 0 14px; }
    .technical-details-body { background: #f8fafc; }

    .stfoot { padding: 8px 24px; }

    .psd-floating {
      top: 28px;
      width: min(1220px, calc(100vw - 36px));
      height: min(860px, calc(100vh - 56px));
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(3, 35, 78, 0.24);
    }
    .psd-floating-header {
      height: 54px;
      padding: 0 12px 0 18px;
      border-top: 0;
      border-image: none;
      background: #fff;
      color: var(--st-navy);
      box-shadow: 0 1px 0 rgba(3,35,78,.04);
    }
    .psd-floating-heading { align-items: center; }
    .psd-floating-heading strong { font-size: 16px; }
    .psd-floating-heading span { color: var(--muted); }
    .psd-window-action {
      height: 31px;
      border-color: #c9d5e1;
      border-radius: 3px;
      background: #fff;
      color: var(--st-navy);
    }
    .psd-window-action:hover { border-color: var(--st-blue); color: var(--st-blue); }
    .psd-floating-body {
      height: calc(100% - 54px);
      grid-template-rows: auto minmax(135px, .36fr) minmax(275px, 1fr) auto;
    }

    .spectrum-head {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto;
      grid-template-areas: "spectrum-title spectrum-primary" "spectrum-advanced spectrum-advanced";
      gap: 10px 18px;
      align-items: end;
      max-height: none;
      padding: 12px 14px 0;
      background: #fff;
    }
    .spectrum-title-group { grid-area: spectrum-title; min-width: 0; }
    .spectrum-title { font-size: 15px; }
    .spectrum-description {
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .psd-primary-controls {
      grid-area: spectrum-primary;
      display: flex;
      align-items: end;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .psd-primary-controls .psd-control select { min-width: 110px; }
    .psd-primary-controls .psd-control:first-child select { min-width: 145px; }
    .psd-action { height: 31px; border-radius: 3px; background: var(--st-blue); color: #fff; }
    .psd-action:hover { background: #035a99; }

    .psd-advanced {
      grid-area: spectrum-advanced;
      margin: 0 -14px;
      border-top: 1px solid var(--line);
      background: #f7f9fb;
    }
    .psd-advanced > summary {
      min-height: 38px;
      padding: 0 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      list-style: none;
      color: var(--st-navy);
      font-size: 11px;
      font-weight: 700;
    }
    .psd-advanced > summary::-webkit-details-marker { display: none; }
    .psd-advanced > summary::after {
      content: "+";
      margin-left: auto;
      color: var(--st-blue);
      font-size: 16px;
      font-weight: 500;
    }
    .psd-advanced[open] > summary::after { content: "−"; }
    .psd-advanced > summary span:last-child {
      color: var(--muted);
      font-weight: 500;
    }
    .psd-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(122px, 1fr));
      gap: 9px;
      padding: 0 14px 13px;
      align-items: end;
      justify-content: stretch;
    }
    .psd-control select,
    .psd-control input { width: 100%; min-width: 0; height: 31px; border-radius: 3px; }
    .psd-filter-note { grid-column: 1 / -1; padding-top: 2px; }

    .psd-chart-block { background: #fff; }
    .psd-chart-label { min-height: 31px; padding: 0 14px; background: #f7f9fb; }
    .psd-insights {
      min-height: 40px;
      flex-wrap: nowrap;
      overflow-x: auto;
      padding: 7px 12px;
      scrollbar-width: thin;
    }
    .psd-pill { flex: 0 0 auto; border-radius: 3px; background: #f8fafc; }

    @media (max-width: 1040px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; grid-template-rows: auto 1fr auto; }
      header { min-height: 68px; }
      main { grid-template-columns: 1fr; padding: 14px; overflow: visible; }
      .diagnostics { min-height: 390px; }
    }

    @media (max-width: 760px) {
      header { padding: 12px 16px; align-items: flex-start; flex-direction: column; }
      .top-actions { width: 100%; justify-content: flex-start; flex-wrap: wrap; gap: 8px 12px; }
      .app-nav { width: 100%; gap: 14px; }
      .nav-link { min-height: 34px; }
      .status { flex: 1 0 100%; width: 100%; max-width: 100%; min-width: 0; }
      .status span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      main { grid-template-columns: 1fr; padding: 10px; }
      .settings { grid-template-columns: 1fr; }
      .message { max-width: 96%; }
      .quick button { flex: 1 1 calc(50% - 7px); }
      .spectrum-launch { align-items: stretch; flex-direction: column; }
      .psd-open-action { width: 100%; }
      .spectrum-head {
        grid-template-columns: 1fr;
        grid-template-areas: "spectrum-title" "spectrum-primary" "spectrum-advanced";
        padding-top: 10px;
      }
      .psd-primary-controls { justify-content: flex-start; }
      .psd-primary-controls .psd-control { flex: 1 1 120px; }
      .psd-action { flex: 1 1 100%; }
      .psd-advanced { margin: 0 -14px; }
      .psd-floating-body { grid-template-rows: auto minmax(105px, .3fr) minmax(220px, 1fr) auto; }
      .psd-floating-heading span { display: none; }
    }

    /* ST logo. */
    body::before { display: none !important; content: none !important; }
    .mark {
      width: 112px !important;
      height: 48px !important;
      min-width: 112px !important;
      padding: 0 !important;
      border: 0 !important;
      border-radius: 0 !important;
      background: transparent !important;
      color: transparent !important;
      font-size: 0 !important;
      letter-spacing: 0 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
    }
    .mark::after { display: none !important; content: none !important; }
    .st-logo {
      display: block;
      width: 112px;
      max-width: 112px;
      max-height: 48px;
      height: auto;
      object-fit: contain;
    }
    .stfoot { border-top: 0 !important; border-image: none !important; }

  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark"><img class="st-logo" src="/assets/st-logo.png" alt="STMicroelectronics logo"></div>
        <div>
          <h1>VibroAgent</h1>
          <div class="subtitle">Building vibration monitoring</div>
        </div>
      </div>
      <div class="top-actions">
        <nav class="app-nav" aria-label="Application navigation">
          <a class="nav-link" href="/graph">Live data</a>
          <a class="nav-link" href="/npu-chat">NPU console</a>
        </nav>
        <div class="status" aria-live="polite"><span id="statusDot" class="dot"></span><span id="statusText">Model not checked</span></div>
      </div>
    </header>
    <main>
      <section class="panel chat" aria-label="Chat">
        <details id="connectionDetails" class="connection-details">
          <summary><span>Model connection</span><span id="connectionSummary" class="connection-summary">Using server defaults</span></summary>
          <form id="settingsForm" class="settings" autocomplete="off">
            <label>Base URL
              <input id="baseUrl" autocomplete="off" spellcheck="false">
            </label>
            <label>Model
              <input id="model" autocomplete="off" spellcheck="false">
            </label>
            <label>API key
              <input id="apiKey" type="password" autocomplete="off" placeholder="EMPTY">
            </label>
            <button id="checkModel" class="check" type="button">Check connection</button>
          </form>
        </details>
        <div id="messages" class="messages" aria-live="polite"></div>
        <form id="chatForm" class="composer">
          <div class="quick" aria-label="Building-vibration prompts">
            <button type="button" data-prompt="Analyze the latest building-vibration sensor network relative to the reference sensor.">Analyze network</button>
            <button type="button" data-prompt="Read the current 5-second baseline building-sensor window and summarize its signal statistics.">Read baseline</button>
            <button type="button" data-prompt="Read the saved 5-second baseline building-sensor window and summarize its signal statistics.">Replay saved</button>
            <button type="button" id="psdFftButton">Open spectrum</button>
          </div>
          <div class="send-row">
            <textarea id="message" required placeholder="Ask about sensors, floor comparisons, RMS, variance, or frequency content"></textarea>
            <button id="send" class="send" type="submit">Send</button>
          </div>
        </form>
      </section>
      <aside class="panel diagnostics" aria-label="Diagnostics">
        <div class="side-head">
          <div class="side-title"><h2>Building analysis</h2><span>Reference-relative sensor summary</span></div>
          <span id="modelActivity" class="chip">Waiting</span>
        </div>
        <div class="metrics">
          <div class="metric primary"><span>Assessment</span><strong id="prediction">--</strong></div>
          <div class="metric"><span>Confidence</span><strong id="confidence">--</strong></div>
          <div class="metric"><span id="rmsMetricLabel">RMS / Ratio</span><strong id="rmsRatio">--</strong></div>
          <div class="metric"><span>Dominant / Rate</span><strong id="dominantHz">--</strong></div>
          <div class="metric"><span>Status</span><strong id="vibrationStatus">--</strong></div>
        </div>
        <div class="visual">
          <canvas id="featureCanvas" class="feature-canvas" width="720" height="280" hidden></canvas>
          <div class="spectrum-launch">
            <div class="spectrum-launch-copy">
              <strong>Building vibration spectrum</strong>
              <span id="psdCompactSummary">Welch PSD · periodic Hann · filter off</span>
            </div>
            <button id="psdOpenButton" class="psd-open-action" type="button">Open spectrum</button>
          </div>
          <details class="technical-details">
            <summary><span>Technical details</span><span id="toolChip" class="chip">No tool call</span></summary>
            <div class="technical-details-body">
              <pre id="mcpCall" class="call-box">MCP call will appear here.</pre>
              <pre id="payload">No building-analysis payload yet.</pre>
            </div>
          </details>
        </div>
      </aside>
    </main>
      <footer class="stfoot"><span><strong>STMicroelectronics</strong> &middot; VibroAgent &mdash; building vibration monitoring</span><span class="tag">life.augmented</span></footer>
  </div>
  <section id="psdFloatingPanel" class="psd-floating" role="dialog" aria-modal="false" aria-labelledby="psdDialogTitle" hidden>
    <div id="psdFloatingHeader" class="psd-floating-header">
      <div class="psd-floating-heading">
        <strong id="psdDialogTitle">Building vibration spectrum</strong>
        <span id="psdFloatingStatus">Drag to move · resize from the lower-right corner</span>
      </div>
      <div class="psd-window-actions">
        <button id="psdMaximize" class="psd-window-action" type="button" aria-label="Maximize spectrum window">Maximize</button>
        <button id="psdClose" class="psd-window-action" type="button" aria-label="Close spectrum window">Close</button>
      </div>
    </div>
    <div class="psd-floating-body">
      <div class="spectrum-head">
        <div class="spectrum-title-group">
          <div class="spectrum-title"><span id="psdTitle">Welch PSD</span><span id="psdSummary" class="psd-summary">100 s · averaged spectrum</span></div>
          <div class="spectrum-description">Frequency identifies where building vibration is concentrated; RMS and variance describe its overall strength.</div>
        </div>
        <div class="psd-primary-controls">
          <label class="psd-control">Board
            <select id="psdSensor">
              <option value="">Loading boards...</option>
            </select>
          </label>
          <label class="psd-control">Axis
            <select id="psdAxis">
              <option value="norm">Magnitude (overview)</option>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z" selected>Z (board default)</option>
            </select>
          </label>
          <label class="psd-control">Frequency view
            <select id="psdRange">
              <option value="50">0–50 Hz</option>
              <option value="200" selected>0–200 Hz</option>
              <option value="500">0–500 Hz</option>
              <option value="1000">0–1 kHz</option>
              <option value="5000">0–5 kHz</option>
              <option value="full">Full spectrum</option>
            </select>
          </label>
          <button id="psdButton" class="psd-action" type="button">Run analysis</button>
        </div>
        <details class="psd-advanced">
          <summary><span>Advanced spectrum settings</span><span>estimator, window, scale and filters</span></summary>
          <div class="psd-controls">
            <label class="psd-control">Mode
              <select id="psdEstimator">
                <option value="welch" selected>Welch</option>
                <option value="periodogram">Raw FFT</option>
              </select>
            </label>
            <label class="psd-control">Window
              <select id="psdWindow">
                <option value="hann" selected>Periodic Hann</option>
                <option value="hamming">Periodic Hamming</option>
                <option value="boxcar">Rectangular</option>
              </select>
            </label>
            <label class="psd-control">X scale
              <select id="psdScale">
                <option value="linear" selected>Linear</option>
                <option value="log">Log</option>
              </select>
            </label>
            <label class="psd-control">Filter
              <select id="psdFilter">
                <option value="none" selected>None</option>
                <option value="highpass">High-pass</option>
                <option value="lowpass">Low-pass</option>
                <option value="bandpass">Band-pass</option>
                <option value="notch_50">50 Hz notch</option>
                <option value="notch_60">60 Hz notch</option>
                <option value="notch">Custom notch</option>
              </select>
            </label>
            <label id="psdLowCutoffField" class="psd-control" hidden>Low / cutoff Hz
              <input id="psdLowCutoff" type="number" min="0.001" step="0.1" value="1">
            </label>
            <label id="psdHighCutoffField" class="psd-control" hidden>High / cutoff Hz
              <input id="psdHighCutoff" type="number" min="0.001" step="1" value="6000">
            </label>
            <label id="psdNotchFrequencyField" class="psd-control" hidden>Notch Hz
              <input id="psdNotchFrequency" type="number" min="0.001" step="0.1" value="50">
            </label>
            <label id="psdNotchWidthField" class="psd-control" hidden>-3 dB width Hz
              <input id="psdNotchWidth" type="number" min="0.001" step="0.1" value="2">
            </label>
            <label id="psdFilterOrderField" class="psd-control" hidden>Order
              <select id="psdFilterOrder">
                <option value="2">2</option>
                <option value="4" selected>4</option>
                <option value="6">6</option>
                <option value="8">8</option>
              </select>
            </label>
            <span class="psd-filter-note">Filters are zero-phase analysis/display filters. They do not alter saved acquisition data or replace anti-alias filtering.</span>
          </div>
        </details>
      </div>
      <div class="psd-chart-block">
        <div class="psd-chart-label"><span>Time-domain overview · min/max envelope</span><span id="psdWaveformLabel">Unfiltered</span></div>
        <canvas id="waveformCanvas" class="waveform-canvas" width="1000" height="300" aria-label="Centered time-domain acceleration min/max overview"></canvas>
      </div>
      <div class="psd-chart-block">
        <div class="psd-chart-label"><span>Power spectral density</span><span>dB re 1 g²/Hz</span></div>
        <canvas id="psdCanvas" class="spectrum-canvas" width="1000" height="560" aria-label="Power spectral density"></canvas>
      </div>
      <div id="psdInsights" class="psd-insights">PSD variance is the area under the mean-removed PSD curve in g²; PSD RMS is its square root in g.</div>
    </div>
  </section>
  <script>
    const els = {
      baseUrl: document.getElementById("baseUrl"),
      model: document.getElementById("model"),
      apiKey: document.getElementById("apiKey"),
      checkModel: document.getElementById("checkModel"),
      connectionSummary: document.getElementById("connectionSummary"),
      messages: document.getElementById("messages"),
      form: document.getElementById("chatForm"),
      message: document.getElementById("message"),
      send: document.getElementById("send"),
      statusDot: document.getElementById("statusDot"),
      statusText: document.getElementById("statusText"),
      modelActivity: document.getElementById("modelActivity"),
      toolChip: document.getElementById("toolChip"),
      prediction: document.getElementById("prediction"),
      confidence: document.getElementById("confidence"),
      rmsMetricLabel: document.getElementById("rmsMetricLabel"),
      rmsRatio: document.getElementById("rmsRatio"),
      dominantHz: document.getElementById("dominantHz"),
      vibrationStatus: document.getElementById("vibrationStatus"),
      mcpCall: document.getElementById("mcpCall"),
      payload: document.getElementById("payload"),
      canvas: document.getElementById("featureCanvas"),
      psdButton: document.getElementById("psdButton"),
      psdQuickButton: document.getElementById("psdFftButton"),
      psdOpenButton: document.getElementById("psdOpenButton"),
      psdCompactSummary: document.getElementById("psdCompactSummary"),
      psdFloatingPanel: document.getElementById("psdFloatingPanel"),
      psdFloatingHeader: document.getElementById("psdFloatingHeader"),
      psdFloatingStatus: document.getElementById("psdFloatingStatus"),
      psdClose: document.getElementById("psdClose"),
      psdMaximize: document.getElementById("psdMaximize"),
      waveformCanvas: document.getElementById("waveformCanvas"),
      psdCanvas: document.getElementById("psdCanvas"),
      psdWaveformLabel: document.getElementById("psdWaveformLabel"),
      psdTitle: document.getElementById("psdTitle"),
      psdSummary: document.getElementById("psdSummary"),
      psdInsights: document.getElementById("psdInsights"),
      psdSensor: document.getElementById("psdSensor"),
      psdEstimator: document.getElementById("psdEstimator"),
      psdWindow: document.getElementById("psdWindow"),
      psdAxis: document.getElementById("psdAxis"),
      psdScale: document.getElementById("psdScale"),
      psdRange: document.getElementById("psdRange"),
      psdFilter: document.getElementById("psdFilter"),
      psdLowCutoffField: document.getElementById("psdLowCutoffField"),
      psdLowCutoff: document.getElementById("psdLowCutoff"),
      psdHighCutoffField: document.getElementById("psdHighCutoffField"),
      psdHighCutoff: document.getElementById("psdHighCutoff"),
      psdNotchFrequencyField: document.getElementById("psdNotchFrequencyField"),
      psdNotchFrequency: document.getElementById("psdNotchFrequency"),
      psdNotchWidthField: document.getElementById("psdNotchWidthField"),
      psdNotchWidth: document.getElementById("psdNotchWidth"),
      psdFilterOrderField: document.getElementById("psdFilterOrderField"),
      psdFilterOrder: document.getElementById("psdFilterOrder"),
    };

    const fmt = (value, digits = 2) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits);
    };

    const finiteMetricNumber = (value) => {
      if (value === null || value === undefined) return null;
      if (typeof value === "string" && value.trim() === "") return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    };

    const fmtScientific = (value, digits = 3) => {
      const parsed = finiteMetricNumber(value);
      return parsed === null ? "--" : parsed.toExponential(digits).replace("e+", "e");
    };

    const labelize = (value) => value ? String(value).replace(/_/g, " ") : "--";

    const toolSourceLabel = (source) => {
      const key = String(source || "");
      const labels = {
        deterministic_router: "automatic building route",
        model_tool_call: "model-selected tool",
        text_tool_call: "model-selected tool",
        unsupported_model_tool_fallback: "automatic supported building route",
        unsupported_tool_fallback: "automatic supported building route",
        webchat: "webchat action",
      };
      return labels[key] || labelize(key);
    };

    const toolNameLabel = (name) => {
      const key = String(name || "");
      const labels = {
        run_autonomous_sensor_check: "Building sensor check",
        run_building_vibration_agent_pipeline: "Building vibration pipeline",
        read_building_sensor_window: "Sensor window read",
        psd_spectrum: "Building vibration spectrum",
      };
      return labels[key] || labelize(key);
    };

    const answerSourceLabel = (data) => {
      if (data && data.model_response_used && data.model_explanation_only) return "Model-generated explanation";
      return "No model-generated explanation";
    };

    function setStatus(kind, text) {
      els.statusDot.className = "dot" + (kind ? " " + kind : "");
      els.statusText.textContent = text;
    }

    function setModelActivity(kind, text) {
      els.modelActivity.className = "chip" + (kind ? " " + kind : "");
      els.modelActivity.textContent = text;
    }

    function updateConnectionSummary() {
      const endpoint = String(els.baseUrl.value || "").replace(/^https?:\/\//, "").replace(/\/$/, "");
      const model = String(els.model.value || "model");
      els.connectionSummary.textContent = endpoint ? `${model} · ${endpoint}` : model;
    }

    function resizeComposer() {
      els.message.style.height = "auto";
      els.message.style.height = `${Math.min(168, Math.max(52, els.message.scrollHeight))}px`;
    }

    function addMessage(role, text) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.textContent = text;
      els.messages.appendChild(node);
      els.messages.scrollTop = els.messages.scrollHeight;
      return node;
    }

    function renderMarkdownInline(text) {
      text = text.replace(/`([^`\n]+)`/g, (match, code) => `<code class="md-code">${code}</code>`);
      text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
      text = text
        .replace(/^\s*#{4,6}\s+(.*)$/gm, "<h4>$1</h4>")
        .replace(/^\s*###\s+(.*)$/gm, "<h4>$1</h4>")
        .replace(/^\s*##\s+(.*)$/gm, "<h3>$1</h3>")
        .replace(/^\s*#\s+(.*)$/gm, "<h3>$1</h3>");
      text = text.replace(/(?:^|\n)((?:[ \t]*[-*][ \t]+.*(?:\n|$))+)/g, (match, items) => {
        const lis = items.replace(/\n+$/, "").split(/\n/).map((line) => `<li>${line.replace(/^[ \t]*[-*][ \t]+/, "")}</li>`).join("");
        return `\n<ul class="md-ul">${lis}</ul>`;
      });
      text = text.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
      text = text
        .replace(/<br>\s*(<(?:ul|h3|h4)[\s>])/g, "$1")
        .replace(/(<\/(?:ul|h3|h4)>)\s*<br>/g, "$1");
      return `<p>${text}</p>`
        .replace(/<p>\s*(<(?:ul|h3|h4)[\s>])/g, "$1")
        .replace(/(<\/(?:ul|h3|h4)>)\s*<\/p>/g, "$1")
        .replace(/<p>(?:\s|<br>)*<\/p>/g, "");
    }

    function renderMarkdown(src) {
      const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const parts = String(src == null ? "" : src).split(/```/);
      let out = "";
      for (let i = 0; i < parts.length; i += 1) {
        if (i % 2 === 1) {
          out += `<pre class="md-pre"><code>${esc(parts[i].replace(/^\n/, "").replace(/\s+$/, ""))}</code></pre>`;
        } else {
          out += renderMarkdownInline(esc(parts[i]));
        }
      }
      return out;
    }

    function revealMarkdown(node, text) {
      const full = String(text == null ? "" : text);
      node.classList.add("assistant-md");
      let body = node.querySelector(".md-body");
      if (!body) { node.textContent = ""; body = document.createElement("div"); body.className = "md-body"; node.appendChild(body); }
      if (node._reveal) { clearInterval(node._reveal); node._reveal = null; }
      const scroll = () => { els.messages.scrollTop = els.messages.scrollHeight; };
      const tokens = full.split(/(\s+)/);
      if (tokens.length <= 3) { body.innerHTML = renderMarkdown(full); scroll(); return; }
      let i = 0;
      const step = Math.max(1, Math.ceil(tokens.length / 24));
      node._reveal = setInterval(() => {
        i += step;
        body.innerHTML = renderMarkdown(tokens.slice(0, Math.min(i, tokens.length)).join(""));
        scroll();
        if (i >= tokens.length) { clearInterval(node._reveal); node._reveal = null; body.innerHTML = renderMarkdown(full); }
      }, 18);
    }

    async function loadConfig() {
      const res = await fetch("/api/config");
      const data = await res.json();
      els.baseUrl.value = data.base_url || "http://127.0.0.1:1234/v1";
      els.model.value = data.model || "qwen3.5-4b-instruct-revised";
      updateConnectionSummary();
      if (data.npu && data.npu.enabled) {
        if (data.npu.serving) {
          const mode = data.npu.npu_offload ? "QNN HTP/NPU" : "Genie adapter";
          setStatus(data.npu.npu_offload ? "ok" : "", `${mode}: ${els.model.value}`);
          if (data.npu.warning) addMessage("assistant", data.npu.warning);
        } else {
          setStatus("bad", "NPU adapter failed");
          addMessage("error", data.npu.startup_error || "Embedded Genie adapter failed to start.");
        }
      } else {
        setStatus("", `Configured: ${els.model.value}`);
      }
    }

    async function checkModel() {
      els.checkModel.disabled = true;
      setStatus("", "Checking model");
      const params = new URLSearchParams({
        base_url: els.baseUrl.value,
        api_key: els.apiKey.value || "EMPTY",
      });
      try {
        const res = await fetch(`/api/models?${params}`);
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || "model check failed");
        if (data.models.length && !data.models.includes(els.model.value)) {
          els.model.value = data.models[0];
        }
        updateConnectionSummary();
        setStatus("ok", data.models.length ? `Loaded: ${els.model.value}` : "Endpoint reachable");
      } catch (err) {
        setStatus("bad", "Model unreachable");
        addMessage("error", String(err.message || err));
      } finally {
        els.checkModel.disabled = false;
      }
    }

    function updateDiagnostics(data) {
      if (data.model_response_used && data.model_explanation_only) {
        setModelActivity("ok", "Model-generated explanation");
      } else {
        setModelActivity("bad", "No model answer · no fallback");
      }
      const tool = (data.tool_results || [])[0];
      const call = (data.mcp_calls || [])[0] || (tool ? {
        source: tool.source,
        name: tool.name,
        arguments: tool.arguments,
      } : null);
      const result = tool && tool.result && typeof tool.result === "object" ? tool.result : null;
      els.toolChip.textContent = call ? `${toolNameLabel(call.name)} · ${toolSourceLabel(call.source)}` : "No tool call";
      els.mcpCall.textContent = call
        ? JSON.stringify({
            answer_source: answerSourceLabel(data),
            tool_route: toolSourceLabel(call.source),
            tool: toolNameLabel(call.name),
            arguments: call.arguments,
            note: call.source === "deterministic_router"
              ? "The building-vibration host selected the monitoring tool directly; the final answer can still come from the model."
              : call.note,
            parse_error: call.parse_error,
          }, null, 2)
        : "MCP call will appear here.";
      if (!result) return;

      const network = result.network_assessment || {};
      const reports = Array.isArray(result.sensor_agent_reports) ? result.sensor_agent_reports : [];
      if (network.network_label || network.main_agent_label) {
        const affected = Array.isArray(network.affected_sensor_ids) ? network.affected_sensor_ids : [];
        const focusId = affected[0] || (reports[0] && reports[0].sensor_id) || "";
        const focusReport = reports.find((item) => item && item.sensor_id === focusId) || reports[0] || {};
        const metrics = focusReport.metrics || result.baseline_metrics || {};
        const comparison = focusReport.baseline_comparison || {};
        const label = network.network_label || network.main_agent_label;
        const confidence = Number(network.main_agent_confidence);
        const qualityFlags = Array.isArray(network.quality_flags) ? network.quality_flags : [];

        els.prediction.textContent = labelize(label);
        els.confidence.textContent = Number.isFinite(confidence) ? `${fmt(confidence * 100, 1)}%` : "--";
        els.rmsMetricLabel.textContent = focusReport.sensor_id ? "RMS ratio" : "Reference RMS";
        els.rmsRatio.textContent = Number.isFinite(Number(comparison.rms_ratio))
          ? fmt(comparison.rms_ratio, 2)
          : (Number.isFinite(Number(metrics.acceleration_rms_g)) ? `${fmt(metrics.acceleration_rms_g, 5)} g` : "--");
        els.dominantHz.textContent = Number.isFinite(Number(metrics.dominant_frequency_hz))
          ? `${fmt(metrics.dominant_frequency_hz, 2)} Hz`
          : "--";
        els.vibrationStatus.textContent = qualityFlags.length
          ? `${qualityFlags.length} quality flag${qualityFlags.length === 1 ? "" : "s"}`
          : (affected.length ? `${affected.length} affected sensor${affected.length === 1 ? "" : "s"}` : "reference comparison");
      } else {
        const metadata = result.metadata || result.source_metadata || {};
        const sourceMode = metadata.data_is_current === false
          ? "replay/stale"
          : metadata.data_is_current === true
            ? "current"
            : labelize(metadata.data_source_mode);
        els.prediction.textContent = "building sensor window";
        els.confidence.textContent = result.sample_count !== undefined ? `${result.sample_count} samples` : "--";
        els.rmsMetricLabel.textContent = "RMS";
        const rmsValue = result.rms ?? result.acceleration_rms_g;
        els.rmsRatio.textContent = Number.isFinite(Number(rmsValue)) ? `${fmt(rmsValue, 5)} g` : "--";
        const dominantValue = result.dominant_frequency_hz;
        els.dominantHz.textContent = Number.isFinite(Number(dominantValue))
          ? `${fmt(dominantValue, 2)} Hz`
          : (result.sampling_rate_hz ? `${fmt(result.sampling_rate_hz, 0)} Hz rate` : "--");
        els.vibrationStatus.textContent = sourceMode || labelize(result.status) || "--";
      }
      els.payload.textContent = JSON.stringify(result, null, 2);
      els.canvas.hidden = false;
      drawFeatureBars(result);
    }

    let lastFeatureResult = null;
    let lastPsdResult = null;
    let psdRequestInFlight = false;
    let psdSensorOptions = [];
    let psdBaselineSensorId = "";

    function psdSensorOptionLabel(sensor) {
      const role = sensor && sensor.role ? ` · ${sensor.role}` : "";
      const location = sensor && sensor.location ? ` · ${sensor.location}` : "";
      return `${sensor.sensor_id}${role}${location}`;
    }

    function applySelectedPsdBoardAxis() {
      const selected = selectedPsdSensorConfig();
      const configuredAxis = selected && selected.axis ? String(selected.axis).toLowerCase() : "";
      if (["norm", "x", "y", "z"].includes(configuredAxis)) {
        els.psdAxis.value = configuredAxis;
      }
    }

    function selectedPsdSensorConfig() {
      const selectedId = els.psdSensor ? els.psdSensor.value : "";
      return psdSensorOptions.find((sensor) => sensor.sensor_id === selectedId) || null;
    }

    function selectedPsdSensorLabel() {
      const selected = selectedPsdSensorConfig();
      return selected ? selected.sensor_id : (els.psdSensor.value || psdBaselineSensorId || "baseline");
    }

    async function loadPsdSensorOptions() {
      const previous = els.psdSensor.value;
      try {
        const response = await fetch("/api/live-sensors?prewarm=0", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok || !payload.ok) throw new Error(payload.message || payload.error || "board list failed");
        psdSensorOptions = Array.isArray(payload.sensors) ? payload.sensors : [];
        psdBaselineSensorId = payload.baseline_sensor_id || "";
        if (!psdSensorOptions.length) throw new Error("the live sensor registry contains no boards");

        els.psdSensor.replaceChildren();
        psdSensorOptions.forEach((sensor) => {
          const option = document.createElement("option");
          option.value = sensor.sensor_id;
          option.textContent = psdSensorOptionLabel(sensor);
          els.psdSensor.appendChild(option);
        });
        const preferred = psdSensorOptions.some((sensor) => sensor.sensor_id === previous)
          ? previous
          : (psdSensorOptions.some((sensor) => sensor.sensor_id === psdBaselineSensorId)
            ? psdBaselineSensorId
            : psdSensorOptions[0].sensor_id);
        els.psdSensor.value = preferred;
        applySelectedPsdBoardAxis();
      } catch (error) {
        psdSensorOptions = [];
        psdBaselineSensorId = "baseline";
        els.psdSensor.replaceChildren();
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Baseline (server default)";
        els.psdSensor.appendChild(option);
        els.psdFloatingStatus.textContent = `Board list unavailable · server baseline will be used`;
      }
    }

    function updatePsdFilterControls() {
      const selected = els.psdFilter.value;
      const highPass = selected === "highpass" || selected === "bandpass";
      const lowPass = selected === "lowpass" || selected === "bandpass";
      const notch = selected === "notch" || selected === "notch_50" || selected === "notch_60";
      els.psdLowCutoffField.hidden = !highPass;
      els.psdHighCutoffField.hidden = !lowPass;
      els.psdNotchFrequencyField.hidden = selected !== "notch";
      els.psdNotchWidthField.hidden = !notch;
      els.psdFilterOrderField.hidden = selected === "none";
    }

    function selectedFilterLabel() {
      const selected = els.psdFilter.value;
      if (selected === "highpass") return `High-pass ${els.psdLowCutoff.value || "auto"} Hz`;
      if (selected === "lowpass") return `Low-pass ${els.psdHighCutoff.value || "auto"} Hz`;
      if (selected === "bandpass") return `Band-pass ${els.psdLowCutoff.value || "auto"}–${els.psdHighCutoff.value || "auto"} Hz`;
      if (selected === "notch_50") return `50 Hz notch · ${els.psdNotchWidth.value || "2"} Hz width`;
      if (selected === "notch_60") return `60 Hz notch · ${els.psdNotchWidth.value || "2"} Hz width`;
      if (selected === "notch") return `${els.psdNotchFrequency.value || "50"} Hz notch · ${els.psdNotchWidth.value || "2"} Hz width`;
      return "Unfiltered";
    }

    function appendFilterParameters(params) {
      const selected = els.psdFilter.value;
      params.set("filter_type", selected);
      if (selected !== "none") params.set("filter_order", els.psdFilterOrder.value || "4");
      if (selected === "highpass" || selected === "bandpass") {
        params.set("low_cutoff_hz", els.psdLowCutoff.value);
      }
      if (selected === "lowpass" || selected === "bandpass") {
        params.set("high_cutoff_hz", els.psdHighCutoff.value);
      }
      if (selected === "notch") {
        params.set("notch_frequency_hz", els.psdNotchFrequency.value);
      }
      if (selected === "notch" || selected === "notch_50" || selected === "notch_60") {
        params.set("notch_width_hz", els.psdNotchWidth.value);
      }
    }

    function redrawPsdWindow() {
      if (els.psdFloatingPanel.hidden) return;
      drawWaveformChart(lastPsdResult);
      drawPsdChart(lastPsdResult);
    }

    function openPsdFloatingWindow({ run = false } = {}) {
      els.psdFloatingPanel.hidden = false;
      els.psdFloatingPanel.setAttribute("aria-hidden", "false");
      requestAnimationFrame(redrawPsdWindow);
      if (run && !psdRequestInFlight) loadPsdFft();
    }

    function closePsdFloatingWindow() {
      els.psdFloatingPanel.hidden = true;
      els.psdFloatingPanel.setAttribute("aria-hidden", "true");
      els.psdOpenButton.focus();
    }

    function togglePsdMaximize() {
      const maximized = els.psdFloatingPanel.classList.toggle("maximized");
      els.psdMaximize.textContent = maximized ? "Restore" : "Maximize";
      requestAnimationFrame(redrawPsdWindow);
    }

    function initializePsdWindowDrag() {
      let drag = null;
      const finish = (event) => {
        if (!drag) return;
        try { els.psdFloatingHeader.releasePointerCapture(event.pointerId); } catch (_) {}
        drag = null;
      };
      els.psdFloatingHeader.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || event.target.closest("button") || els.psdFloatingPanel.classList.contains("maximized")) return;
        const rect = els.psdFloatingPanel.getBoundingClientRect();
        els.psdFloatingPanel.style.transform = "none";
        els.psdFloatingPanel.style.left = `${rect.left}px`;
        els.psdFloatingPanel.style.top = `${rect.top}px`;
        drag = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
        els.psdFloatingHeader.setPointerCapture(event.pointerId);
        event.preventDefault();
      });
      els.psdFloatingHeader.addEventListener("pointermove", (event) => {
        if (!drag) return;
        const rect = els.psdFloatingPanel.getBoundingClientRect();
        const maxLeft = Math.max(0, window.innerWidth - Math.min(rect.width, window.innerWidth));
        const maxTop = Math.max(0, window.innerHeight - 46);
        const left = Math.max(0, Math.min(maxLeft, event.clientX - drag.dx));
        const top = Math.max(0, Math.min(maxTop, event.clientY - drag.dy));
        els.psdFloatingPanel.style.left = `${left}px`;
        els.psdFloatingPanel.style.top = `${top}px`;
      });
      els.psdFloatingHeader.addEventListener("pointerup", finish);
      els.psdFloatingHeader.addEventListener("pointercancel", finish);
      els.psdFloatingHeader.addEventListener("dblclick", (event) => {
        if (!event.target.closest("button")) togglePsdMaximize();
      });
    }

    function drawFeatureBars(result) {
      lastFeatureResult = result;
      const canvas = els.canvas;
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(280, Math.round(rect.width || 320));
      const height = Math.max(120, Math.round(rect.height || 150));
      const bitmapW = Math.round(width * ratio);
      const bitmapH = Math.round(height * ratio);
      if (canvas.width !== bitmapW || canvas.height !== bitmapH) {
        canvas.width = bitmapW;
        canvas.height = bitmapH;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfa";
      ctx.fillRect(0, 0, width, height);

      const network = result.network_assessment || {};
      const reports = Array.isArray(result.sensor_agent_reports) ? result.sensor_agent_reports : [];
      const affected = Array.isArray(network.affected_sensor_ids) ? network.affected_sensor_ids : [];
      const focusId = affected[0] || (reports[0] && reports[0].sensor_id) || "";
      const focusReport = reports.find((item) => item && item.sensor_id === focusId) || reports[0] || {};
      const buildingMetrics = focusReport.metrics || (network.network_label ? result.baseline_metrics : null);
      let values;
      if (buildingMetrics) {
        const comparison = focusReport.baseline_comparison || {};
        const allMetrics = [result.baseline_metrics, ...(result.sensor_metrics || [])].filter(Boolean);
        const maxRms = Math.max(1e-9, ...allMetrics.map((item) => Math.abs(Number(item.acceleration_rms_g) || 0)));
        const maxPeak = Math.max(1e-9, ...allMetrics.map((item) => Math.abs(Number(item.acceleration_peak_g) || 0)));
        values = [
          ["RMS", buildingMetrics.acceleration_rms_g, maxRms],
          ["Peak", buildingMetrics.acceleration_peak_g, maxPeak],
          ["Crest", buildingMetrics.crest_factor, 8],
          ["RMS ratio", comparison.rms_ratio, 5],
          ["Dominant", buildingMetrics.dominant_frequency_hz, 500],
        ];
      } else {
        const readRms = result.rms ?? result.acceleration_rms_g;
        const readValues = [result.min, result.max, result.mean, result.std, readRms]
          .map((value) => Number(value))
          .filter((value) => Number.isFinite(value));
        const maxAbs = Math.max(1e-9, ...readValues.map((value) => Math.abs(value)));
        values = [
          ["Min", result.min, maxAbs],
          ["Max", result.max, maxAbs],
          ["Mean", result.mean, maxAbs],
          ["Std", result.std, maxAbs],
          ["RMS", readRms, maxAbs],
        ];
      }
      const margin = 22;
      const barGap = 12;
      const axisY = height - 22;
      const topPad = 20;
      const barWidth = (width - margin * 2 - barGap * (values.length - 1)) / values.length;

      ctx.strokeStyle = "#d7ddd7";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin, axisY);
      ctx.lineTo(width - margin, axisY);
      ctx.stroke();

      ctx.textAlign = "center";
      values.forEach(([label, raw, max], index) => {
        const value = Number(raw) || 0;
        const normalized = Math.max(0.04, Math.min(1, Math.abs(value) / max));
        const x = margin + index * (barWidth + barGap);
        const barHeight = normalized * (axisY - topPad);
        const y = axisY - barHeight;
        const cx = x + barWidth / 2;
        ctx.fillStyle = index === 0 || index === 3 ? "#0469b1" : "#03234e";
        ctx.fillRect(x, y, barWidth, barHeight);
        ctx.fillStyle = "#1f2a26";
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.fillText(fmt(value, index === 4 ? 1 : 2), cx, Math.max(11, y - 4));
        ctx.fillStyle = "#66736e";
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(label, cx, height - 7);
      });
      ctx.textAlign = "left";
    }

    function compactPsdPayload(data) {
      const compact = Object.assign({}, data || {});
      [
        "frequency_hz",
        "psd_g2_per_hz",
        "psd_db_re_1_g2_per_hz",
        "asd_g_per_sqrt_hz",
        "peak_envelope_frequency_hz",
        "peak_envelope_psd_g2_per_hz",
        "peak_envelope_db_re_1_g2_per_hz",
        "waveform_time_s",
        "waveform_min_g",
        "waveform_max_g",
        "waveform_mean_g",
        "impact_spectrum_frequency_hz",
        "impact_spectrum_psd_g2_per_hz",
        "impact_spectrum_db_re_1_g2_per_hz",
      ].forEach((key) => delete compact[key]);
      return compact;
    }

    function renderPsdInsights(data) {
      const container = els.psdInsights;
      container.replaceChildren();
      const addPill = (label, value) => {
        const pill = document.createElement("span");
        pill.className = "psd-pill";
        const strong = document.createElement("strong");
        strong.textContent = label;
        const text = document.createElement("span");
        text.textContent = value;
        pill.append(strong, text);
        container.appendChild(pill);
      };
      if (!data) {
        container.textContent = "PSD variance is the area under the mean-removed PSD curve in g²; PSD RMS is its square root in g.";
        return;
      }
      const source = data.source || {};
      const sourceId = source.sensor_id || data.machine_id || "unknown board";
      const sourceDetail = source.location ? `${sourceId} · ${source.location}` : sourceId;
      addPill("Source", sourceDetail);
      const sourceFolder = source.acquisition_folder || data.acquisition_folder;
      if (sourceFolder) {
        const normalizedFolder = String(sourceFolder).replace(/\\/g, "/").replace(/\/$/, "");
        addPill("Data folder", normalizedFolder.split("/").pop() || normalizedFolder);
      }
      addPill("PSD RMS", `${fmt(data.psd_rms_g, 5)} g`);
      const psdVariance = finiteMetricNumber(data.psd_variance_g2 ?? data.integrated_psd_power_g2);
      if (psdVariance !== null) addPill("PSD variance", `${fmtScientific(psdVariance)} g²`);
      const analysisStats = data.analysis_stats || {};
      const timeVariance = finiteMetricNumber(data.centered_variance_g2 ?? analysisStats.variance_g2);
      if (timeVariance !== null) addPill("Time variance", `${fmtScientific(timeVariance)} g²`);
      addPill("Window", data.window_name === "hann" ? "Periodic Hann" : `Periodic ${labelize(data.window_name)}`);
      const filter = data.filter || {};
      addPill("Filter", filter.label || "None · mean removed");
      if (filter.applied) addPill("Phase", filter.phase_response || "zero phase");
      if (data.waveform_downsampled) addPill("Waveform", `${data.waveform_points} min/max buckets`);
      if (data.axis_warning) addPill("Axis note", "Use X/Y/Z for diagnosis");
      if (Number.isFinite(Number(data.power_consistency_ratio))) {
        addPill("Power check", `${fmt(Number(data.power_consistency_ratio) * 100, 1)}%`);
      }
      const varianceConsistency = finiteMetricNumber(data.variance_consistency_ratio);
      if (varianceConsistency !== null) {
        addPill("Variance check", `${fmt(varianceConsistency * 100, 1)}%`);
      }
      if (data.estimator === "welch") {
        addPill("Average", `${data.segment_count || 1} segments · ${fmt(data.segment_duration_s, 2)} s`);
      } else {
        addPill("Resolution", `${fmt(data.fft_resolution_hz, 5)} Hz`);
      }
      const bands = Array.isArray(data.frequency_bands) ? data.frequency_bands : [];
      const strongestBand = bands
        .filter((band) => Number.isFinite(Number(band.fraction)))
        .sort((a, b) => Number(b.fraction) - Number(a.fraction))[0];
      if (strongestBand) addPill("Main band", `${strongestBand.label} · ${fmt(Number(strongestBand.fraction) * 100, 1)}%`);
      const impact = data.impact_analysis || {};
      const impactPeak = finiteMetricNumber(data.impact_peak_frequency_hz ?? impact.peak_frequency_hz);
      if (impact.detected) {
        if (impactPeak !== null) {
          const prominence = finiteMetricNumber(impact.peak_prominence_db);
          const duration = finiteMetricNumber(impact.analysis_duration_s);
          const confidence = impact.peak_reliable ? "repeatable" : "low confidence";
          const details = [
            `${fmt(impactPeak, 2)} Hz`,
            prominence !== null ? `${fmt(prominence, 1)} dB prom.` : null,
            duration !== null ? `${fmt(duration, 2)} s response` : null,
            confidence,
          ].filter(Boolean).join(" · ");
          addPill("Impact ring-down", details);
        } else if (impact.status === "need_more_post_event_data") {
          addPill("Impact", "detected near window end · wait briefly and run again");
        } else {
          addPill("Impact", "broadband event · no repeatable ring-down resonance");
        }
        if (impact.clipping_suspected) {
          addPill("Impact quality", "possible sensor clipping · repeat with a gentler tap");
        }
      }
      const peaks = (Array.isArray(data.top_peaks) ? data.top_peaks : []).slice(0, 4);
      if (data.peak_is_competing && Number.isFinite(Number(data.competing_peak_frequency_hz))) {
        addPill(
          "Competing peaks",
          `${fmt(data.peak_frequency_hz, 2)} / ${fmt(data.competing_peak_frequency_hz, 2)} Hz · ${fmt(data.peak_lead_db, 2)} dB apart`,
        );
      } else if (Number.isFinite(Number(data.peak_lead_db))) {
        addPill("Peak separation", `${fmt(data.peak_lead_db, 2)} dB`);
      }
      peaks.forEach((peak, index) => {
        const prominence = Number.isFinite(Number(peak.prominence_db)) ? ` · ${fmt(peak.prominence_db, 1)} dB prom.` : "";
        addPill(`#${index + 1}`, `${fmt(peak.frequency_hz, 2)} Hz${prominence}`);
      });
    }

    function updatePsdDiagnostics(data) {
      const metadata = data.metadata || {};
      const source = data.source || {};
      const sourceId = source.sensor_id || data.machine_id || "unknown board";
      if (source.sensor_id && Array.from(els.psdSensor.options).some((option) => option.value === source.sensor_id)) {
        els.psdSensor.value = source.sensor_id;
      }
      const sourceMode = metadata.data_is_current === true ? "current" : "replay/stale";
      const estimatorName = data.estimator === "welch" ? "Welch PSD" : "Raw FFT PSD";
      const filter = data.filter || {};
      const filterLabel = filter.label || "None · mean removed";
      els.toolChip.textContent = "psd_spectrum (webchat)";
      els.prediction.textContent = estimatorName;
      els.confidence.textContent = data.sample_count !== undefined ? `${data.sample_count} samples` : "--";
      const psdVariance = finiteMetricNumber(data.psd_variance_g2 ?? data.integrated_psd_power_g2);
      els.rmsMetricLabel.textContent = "PSD RMS / Variance";
      els.rmsRatio.textContent = data.psd_rms_g !== undefined
        ? `${fmt(data.psd_rms_g, 5)} g / ${fmtScientific(psdVariance)} g²`
        : "--";
      const steadyPeak = finiteMetricNumber(data.peak_frequency_hz);
      const impactPeak = finiteMetricNumber(data.impact_peak_frequency_hz ?? (data.impact_analysis || {}).peak_frequency_hz);
      els.dominantHz.textContent = steadyPeak !== null
        ? `${fmt(steadyPeak, 2)} Hz`
        : (impactPeak !== null ? `${fmt(impactPeak, 2)} Hz impact` : "--");
      els.vibrationStatus.textContent = sourceMode;
      els.psdTitle.textContent = estimatorName;
      els.mcpCall.textContent = JSON.stringify({
        source: "webchat",
        name: "psd_spectrum",
        arguments: {
          sensor_id: source.sensor_id || data.machine_id,
          estimator: data.estimator,
          axis: data.axis,
          duration_s: data.requested_duration_s,
          window: data.window_name,
          plot_scale: data.plot_scale,
          filter_type: data.filter_requested || filter.type || "none",
          low_cutoff_hz: filter.low_cutoff_hz,
          high_cutoff_hz: filter.high_cutoff_hz,
          notch_frequency_hz: filter.notch_frequency_hz,
          notch_width_hz: filter.notch_width_hz,
          filter_order: filter.order,
          require_current: false,
          max_bins: data.points_returned,
        },
      }, null, 2);
      els.payload.textContent = JSON.stringify(compactPsdPayload(data), null, 2);
      const impact = data.impact_analysis || {};
      const peak = steadyPeak !== null
        ? `${fmt(steadyPeak, 2)} Hz averaged peak`
        : (impactPeak !== null
          ? `${fmt(impactPeak, 2)} Hz impact ring-down`
          : (impact.status === "need_more_post_event_data"
            ? "impact needs more post-event data"
            : (impact.detected ? "broadband impact · no ring-down peak" : "no distinct peak")));
      const peakDetail = data.peak_is_competing && Number.isFinite(Number(data.competing_peak_frequency_hz))
        ? `${peak} · ${fmt(data.competing_peak_frequency_hz, 2)} Hz · levels are ${fmt(data.peak_lead_db, 2)} dB apart`
        : peak;
      const averaging = data.estimator === "welch" ? `${data.segment_count} averages` : "single record";
      const windowLabel = data.window_name === "hann" ? "periodic Hann" : `periodic ${labelize(data.window_name)}`;
      const summary = `${sourceId} · ${fmt(data.window_duration_s, 1)} s · ${fmt(data.fft_resolution_hz, 4)} Hz · ${windowLabel} · ${averaging} · ${peakDetail}`;
      els.psdSummary.textContent = summary;
      els.psdCompactSummary.textContent = `${sourceId} · ${estimatorName} · ${filterLabel} · ${peak}`;
      els.psdFloatingStatus.textContent = `${sourceId} · ${data.axis.toUpperCase()} · ${sourceMode} · ${filterLabel}`;
      els.psdWaveformLabel.textContent = filter.applied ? filterLabel : "Unfiltered · mean removed";
      renderPsdInsights(data);
    }

    function drawWaveformChart(data) {
      const canvas = els.waveformCanvas;
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(280, Math.round(rect.width || 320));
      const height = Math.max(110, Math.round(rect.height || 128));
      const bitmapW = Math.round(width * ratio);
      const bitmapH = Math.round(height * ratio);
      if (canvas.width !== bitmapW || canvas.height !== bitmapH) {
        canvas.width = bitmapW;
        canvas.height = bitmapH;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfa";
      ctx.fillRect(0, 0, width, height);

      const times = data && Array.isArray(data.waveform_time_s) ? data.waveform_time_s : [];
      const mins = data && Array.isArray(data.waveform_min_g) ? data.waveform_min_g : [];
      const maxs = data && Array.isArray(data.waveform_max_g) ? data.waveform_max_g : [];
      const means = data && Array.isArray(data.waveform_mean_g) ? data.waveform_mean_g : [];
      const points = times.map((timeValue, index) => ({
        t: Number(timeValue),
        min: Number(mins[index]),
        max: Number(maxs[index]),
        mean: Number(means[index]),
      })).filter((point) => (
        Number.isFinite(point.t)
        && Number.isFinite(point.min)
        && Number.isFinite(point.max)
        && Number.isFinite(point.mean)
      ));

      const margin = { left: 58, right: 16, top: 18, bottom: 30 };
      const plotW = Math.max(1, width - margin.left - margin.right);
      const plotH = Math.max(1, height - margin.top - margin.bottom);
      ctx.strokeStyle = "#d6dee7";
      ctx.lineWidth = 1;
      ctx.strokeRect(margin.left, margin.top, plotW, plotH);
      ctx.fillStyle = "#5d6b7c";
      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Time (s)", margin.left + plotW / 2, height - 6);
      ctx.save();
      ctx.translate(12, margin.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Centered acceleration (g)", 0, 0);
      ctx.restore();

      if (!points.length) {
        ctx.textAlign = "center";
        ctx.fillStyle = "#5d6b7c";
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("Run the vibration spectrum", width / 2, height / 2);
        return;
      }

      const xMin = Math.min(...points.map((point) => point.t));
      const rawXMax = Math.max(...points.map((point) => point.t));
      const xMax = rawXMax > xMin ? rawXMax : xMin + 1;
      const maxAbs = Math.max(
        ...points.map((point) => Math.max(Math.abs(point.min), Math.abs(point.max))),
      );
      const amplitude = Number.isFinite(maxAbs) && maxAbs > 0 ? maxAbs * 1.08 : 1e-6;
      const yMin = -amplitude;
      const yMax = amplitude;
      const xScale = (value) => margin.left + ((value - xMin) / Math.max(1e-12, xMax - xMin)) * plotW;
      const yScale = (value) => margin.top + ((yMax - value) / Math.max(1e-12, yMax - yMin)) * plotH;
      const formatAcceleration = (value) => Math.abs(value) > 0 && Math.abs(value) < 0.001
        ? value.toExponential(1)
        : fmt(value, 3);

      ctx.strokeStyle = "rgba(214, 222, 231, 0.9)";
      ctx.fillStyle = "#5d6b7c";
      for (let index = 0; index <= 4; index += 1) {
        const fraction = index / 4;
        const timeValue = xMin + (xMax - xMin) * fraction;
        const x = xScale(timeValue);
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotH);
        ctx.stroke();
        ctx.textAlign = index === 0 ? "left" : (index === 4 ? "right" : "center");
        ctx.fillText(fmt(timeValue, xMax - xMin < 10 ? 2 : 1), x, height - 17);
      }
      for (let index = 0; index <= 4; index += 1) {
        const value = yMax - (yMax - yMin) * index / 4;
        const y = yScale(value);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(formatAcceleration(value), margin.left - 5, y + 3);
      }

      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xScale(point.t);
        const y = yScale(point.max);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      for (let index = points.length - 1; index >= 0; index -= 1) {
        const point = points[index];
        ctx.lineTo(xScale(point.t), yScale(point.min));
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(4, 105, 177, 0.16)";
      ctx.fill();

      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xScale(point.t);
        const y = yScale(point.mean);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "#0469b1";
      ctx.lineWidth = 1.3;
      ctx.stroke();

      const impact = data.impact_analysis || {};
      const eventTime = finiteMetricNumber(impact.event_time_s);
      if (impact.detected && eventTime !== null && eventTime >= xMin && eventTime <= xMax) {
        const eventX = xScale(eventTime);
        ctx.save();
        ctx.strokeStyle = "#7b2cbf";
        ctx.lineWidth = 1.3;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(eventX, margin.top);
        ctx.lineTo(eventX, margin.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#7b2cbf";
        ctx.font = "700 10px system-ui, sans-serif";
        ctx.textAlign = eventX > margin.left + plotW * 0.75 ? "right" : "left";
        ctx.fillText("impact", eventX + (eventX > margin.left + plotW * 0.75 ? -4 : 4), margin.top + 12);
        ctx.restore();
      }

      ctx.fillStyle = "#03234e";
      ctx.font = "600 10px system-ui, sans-serif";
      ctx.textAlign = "left";
      const displayMode = data.waveform_downsampled ? "min/max envelope + bucket mean" : "full-rate waveform";
      ctx.fillText(displayMode, margin.left + 6, margin.top + 12);
    }

    function drawPsdChart(data) {
      lastPsdResult = data || null;
      const canvas = els.psdCanvas;
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(280, Math.round(rect.width || 320));
      const height = Math.max(180, Math.round(rect.height || 220));
      const bitmapW = Math.round(width * ratio);
      const bitmapH = Math.round(height * ratio);
      if (canvas.width !== bitmapW || canvas.height !== bitmapH) {
        canvas.width = bitmapW;
        canvas.height = bitmapH;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfa";
      ctx.fillRect(0, 0, width, height);

      const floorFreqs = data && Array.isArray(data.frequency_hz) ? data.frequency_hz : [];
      const floorDb = data && Array.isArray(data.psd_db_re_1_g2_per_hz) ? data.psd_db_re_1_g2_per_hz : [];
      const envelopeFreqs = data && Array.isArray(data.peak_envelope_frequency_hz) ? data.peak_envelope_frequency_hz : [];
      const envelopeDb = data && Array.isArray(data.peak_envelope_db_re_1_g2_per_hz) ? data.peak_envelope_db_re_1_g2_per_hz : [];
      const impactFreqs = data && Array.isArray(data.impact_spectrum_frequency_hz) ? data.impact_spectrum_frequency_hz : [];
      const impactDb = data && Array.isArray(data.impact_spectrum_db_re_1_g2_per_hz) ? data.impact_spectrum_db_re_1_g2_per_hz : [];
      const xMode = els.psdScale ? els.psdScale.value : "linear";
      const rangeMode = els.psdRange ? els.psdRange.value : "200";
      const validFrequency = (frequency) => Number.isFinite(frequency) && (xMode === "log" ? frequency > 0 : frequency >= 0);
      const allFloorPoints = floorFreqs.map((frequency, index) => ({
        f: Number(frequency),
        db: Number(floorDb[index]),
      })).filter((point) => validFrequency(point.f) && Number.isFinite(point.db));
      const allEnvelopePoints = envelopeFreqs.map((frequency, index) => ({
        f: Number(frequency),
        db: Number(envelopeDb[index]),
      })).filter((point) => validFrequency(point.f) && Number.isFinite(point.db));
      const allImpactPoints = impactFreqs.map((frequency, index) => ({
        f: Number(frequency),
        db: Number(impactDb[index]),
      })).filter((point) => validFrequency(point.f) && Number.isFinite(point.db));

      const margin = { left: 58, right: 16, top: 24, bottom: 34 };
      const plotW = Math.max(1, width - margin.left - margin.right);
      const plotH = Math.max(1, height - margin.top - margin.bottom);
      const x0 = margin.left;
      const y0 = margin.top + plotH;

      ctx.strokeStyle = "#d6dee7";
      ctx.lineWidth = 1;
      ctx.strokeRect(margin.left, margin.top, plotW, plotH);
      ctx.fillStyle = "#5d6b7c";
      ctx.font = "11px system-ui, sans-serif";
      ctx.save();
      ctx.translate(13, margin.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("PSD level (dB re 1 g²/Hz)", 0, 0);
      ctx.restore();

      if (!allFloorPoints.length && !allEnvelopePoints.length) {
        ctx.textAlign = "center";
        ctx.fillStyle = "#5d6b7c";
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("Run the vibration spectrum", width / 2, height / 2);
        return;
      }

      const allPoints = allFloorPoints.concat(allEnvelopePoints);
      const availableMax = Math.max(...allPoints.map((point) => point.f));
      const requestedMax = rangeMode === "full" ? availableMax : Number(rangeMode);
      let xMax = Number.isFinite(requestedMax) && requestedMax > 0
        ? Math.min(availableMax, requestedMax)
        : availableMax;
      let floorPoints = allFloorPoints.filter((point) => point.f <= xMax);
      let envelopePoints = allEnvelopePoints.filter((point) => point.f <= xMax);
      let impactPoints = allImpactPoints.filter((point) => point.f <= xMax);
      if (!floorPoints.length && !envelopePoints.length && !impactPoints.length) {
        floorPoints = allFloorPoints;
        envelopePoints = allEnvelopePoints;
        impactPoints = allImpactPoints;
        xMax = availableMax;
      }

      const visiblePoints = floorPoints.concat(envelopePoints, impactPoints);
      const xMin = xMode === "log"
        ? Math.min(...visiblePoints.map((point) => point.f).filter((frequency) => frequency > 0))
        : 0;
      xMax = Math.max(xMin * (xMode === "log" ? 1.01 : 1), xMax);
      const rangeLabel = xMax >= availableMax * 0.999
        ? "full"
        : `0–${xMax >= 1000 ? `${fmt(xMax / 1000, 1)}k` : fmt(xMax, 0)} Hz`;
      ctx.textAlign = "center";
      ctx.fillText(`Frequency (Hz, ${xMode} · ${rangeLabel})`, margin.left + plotW / 2, height - 7);

      let yMax = Math.max(...visiblePoints.map((point) => point.db));
      const rawYMin = Math.min(...visiblePoints.map((point) => point.db));
      if (!Number.isFinite(yMax) || !Number.isFinite(rawYMin)) return;
      yMax = Math.ceil((yMax + 3) / 5) * 5;
      let yMin = Math.max(rawYMin, yMax - 100);
      yMin = Math.floor((yMin - 3) / 5) * 5;
      if (Math.abs(yMax - yMin) < 5) yMin = yMax - 20;

      const logMin = Math.log10(Math.max(Number.MIN_VALUE, xMin));
      const logMax = Math.log10(Math.max(Number.MIN_VALUE, xMax));
      const xScale = (frequency) => {
        if (xMode === "log") {
          const value = Math.log10(Math.max(xMin, frequency));
          return x0 + ((value - logMin) / Math.max(1e-12, logMax - logMin)) * plotW;
        }
        return x0 + (Math.max(0, Math.min(xMax, frequency)) / Math.max(1e-12, xMax)) * plotW;
      };
      const yScale = (db) => y0 - ((Math.max(yMin, Math.min(yMax, db)) - yMin) / (yMax - yMin)) * plotH;

      ctx.strokeStyle = "rgba(214, 222, 231, 0.9)";
      ctx.fillStyle = "#5d6b7c";
      ctx.font = "10px system-ui, sans-serif";
      for (let i = 0; i <= 5; i += 1) {
        const db = yMax - (yMax - yMin) * i / 5;
        const y = yScale(db);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(`${Math.round(db)} dB`, margin.left - 5, y + 3);
      }

      const tickValues = [];
      if (xMode === "log") {
        for (let i = 0; i <= 4; i += 1) tickValues.push(10 ** (logMin + (logMax - logMin) * i / 4));
      } else {
        for (let i = 0; i <= 4; i += 1) tickValues.push(xMax * i / 4);
      }
      const formatFrequency = (tick) => tick >= 1000
        ? `${fmt(tick / 1000, tick >= 10000 ? 0 : 1)}k`
        : (tick < 1 ? fmt(tick, 2) : fmt(tick, tick < 10 ? 1 : 0));
      tickValues.forEach((tick, index) => {
        const x = xScale(tick);
        ctx.textAlign = index === 0 ? "left" : (index === tickValues.length - 1 ? "right" : "center");
        ctx.fillText(formatFrequency(tick), x, height - 18);
      });

      const strokeSeries = (points, strokeStyle, lineWidth, dash = []) => {
        if (!points.length) return;
        ctx.save();
        ctx.setLineDash(dash);
        ctx.beginPath();
        points.forEach((point, index) => {
          const x = xScale(point.f);
          const y = yScale(point.db);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        ctx.restore();
      };
      strokeSeries(floorPoints, "#0469b1", 1.8);
      strokeSeries(envelopePoints, "rgba(246, 168, 0, 0.72)", 1.1);
      strokeSeries(impactPoints, "#7b2cbf", 1.6, [6, 4]);

      const markerSeries = envelopePoints.length ? envelopePoints : floorPoints;
      const nearestPlottedPoint = (targetFrequency) => {
        if (!Number.isFinite(targetFrequency) || !markerSeries.length) return null;
        let nearest = markerSeries[0];
        let nearestDistance = Math.abs(nearest.f - targetFrequency);
        for (let index = 1; index < markerSeries.length; index += 1) {
          const candidate = markerSeries[index];
          const distance = Math.abs(candidate.f - targetFrequency);
          if (distance < nearestDistance) {
            nearest = candidate;
            nearestDistance = distance;
          }
        }
        return nearest;
      };

      const peaks = (data && Array.isArray(data.top_peaks) ? data.top_peaks : []).slice(0, 5);
      let primaryOutsideRange = false;
      peaks.forEach((peak, index) => {
        const labelFrequency = Number(peak.frequency_hz);
        const targetFrequency = Number.isFinite(Number(peak.bin_frequency_hz))
          ? Number(peak.bin_frequency_hz)
          : labelFrequency;
        if (!validFrequency(targetFrequency)) return;
        if (targetFrequency > xMax) {
          if (index === 0) primaryOutsideRange = true;
          return;
        }
        const plottedPoint = nearestPlottedPoint(targetFrequency);
        if (!plottedPoint) return;
        const x = xScale(plottedPoint.f);
        const y = yScale(plottedPoint.db);
        if (index === 0) {
          ctx.save();
          ctx.strokeStyle = "#d0021b";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(x, y0);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
          ctx.strokeStyle = "#d0021b";
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.restore();

          const label = `${fmt(labelFrequency, 2)} Hz`;
          ctx.font = "700 11px system-ui, sans-serif";
          const labelWidth = ctx.measureText(label).width + 10;
          const preferredX = x + 7;
          const labelX = Math.max(margin.left + 2, Math.min(width - margin.right - labelWidth - 2, preferredX));
          const labelY = Math.max(margin.top + 3, y - 20);
          ctx.fillStyle = "rgba(255,255,255,0.94)";
          ctx.fillRect(labelX, labelY, labelWidth, 16);
          ctx.fillStyle = "#03234e";
          ctx.textAlign = "left";
          ctx.fillText(label, labelX + 5, labelY + 12);
        } else {
          ctx.strokeStyle = "rgba(3, 35, 78, 0.38)";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(x, y0);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
      });

      const impact = data && data.impact_analysis ? data.impact_analysis : {};
      const impactPeakFrequency = finiteMetricNumber(data && (data.impact_peak_frequency_hz ?? impact.peak_frequency_hz));
      if (impactPeakFrequency !== null && impactPoints.length && impactPeakFrequency <= xMax) {
        let nearestImpact = impactPoints[0];
        let nearestDistance = Math.abs(nearestImpact.f - impactPeakFrequency);
        for (let index = 1; index < impactPoints.length; index += 1) {
          const candidate = impactPoints[index];
          const distance = Math.abs(candidate.f - impactPeakFrequency);
          if (distance < nearestDistance) {
            nearestImpact = candidate;
            nearestDistance = distance;
          }
        }
        const impactX = xScale(nearestImpact.f);
        const impactY = yScale(nearestImpact.db);
        ctx.save();
        ctx.strokeStyle = "#7b2cbf";
        ctx.fillStyle = "#ffffff";
        ctx.lineWidth = 1.8;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(impactX, y0);
        ctx.lineTo(impactX, impactY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(impactX, impactY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        const label = `Impact ${fmt(impactPeakFrequency, 2)} Hz`;
        ctx.font = "700 11px system-ui, sans-serif";
        const labelWidth = ctx.measureText(label).width + 10;
        const labelX = Math.max(margin.left + 2, Math.min(width - margin.right - labelWidth - 2, impactX + 7));
        const labelY = Math.min(y0 - 18, Math.max(margin.top + 19, impactY + 6));
        ctx.fillStyle = "rgba(255,255,255,0.94)";
        ctx.fillRect(labelX, labelY, labelWidth, 16);
        ctx.fillStyle = "#7b2cbf";
        ctx.textAlign = "left";
        ctx.fillText(label, labelX + 5, labelY + 12);
        ctx.restore();
      }

      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#0469b1";
      ctx.fillText("— median floor", margin.left + 6, margin.top + 12);
      ctx.fillStyle = "#b16f00";
      ctx.fillText("— peak envelope", margin.left + 100, margin.top + 12);
      if (impactPoints.length) {
        ctx.fillStyle = "#7b2cbf";
        ctx.fillText("- - impact ring-down", margin.left + 205, margin.top + 12);
      }
      if (primaryOutsideRange) {
        ctx.fillStyle = "#d0021b";
        ctx.textAlign = "right";
        ctx.fillText("Primary peak is outside this frequency view", width - margin.right - 4, margin.top + 12);
      }
    }

    async function loadPsdFft() {
      if (psdRequestInFlight) return;
      psdRequestInFlight = true;
      if (els.psdFloatingPanel.hidden) openPsdFloatingWindow();
      els.psdButton.disabled = true;
      els.psdQuickButton.disabled = true;
      els.psdOpenButton.disabled = true;
      els.psdSensor.disabled = true;
      const estimator = els.psdEstimator.value;
      const estimatorLabel = estimator === "welch" ? "Welch PSD" : "raw FFT periodogram";
      const sensorId = selectedPsdSensorLabel();
      setStatus("", `Computing ${estimatorLabel} for ${sensorId}`);
      els.psdSummary.textContent = "reading and computing...";
      els.psdFloatingStatus.textContent = `${sensorId} · Computing · ${selectedFilterLabel()}`;
      els.psdWaveformLabel.textContent = selectedFilterLabel();
      els.psdInsights.textContent = "Reading the vibration window and computing the filtered waveform and spectrum...";
      const params = new URLSearchParams({
        estimator,
        window: els.psdWindow.value,
        axis: els.psdAxis.value,
        plot_scale: els.psdScale.value,
        duration_s: "100",
        max_bins: "1800",
        waveform_max_points: "900",
        welch_resolution_hz: "0.25",
        overlap_fraction: "0.5",
        require_current: "0",
      });
      const selectedSensor = selectedPsdSensorConfig();
      if (selectedSensor && selectedSensor.sensor_id) {
        params.set("sensor_id", selectedSensor.sensor_id);
      } else if (els.psdSensor.value) {
        params.set("sensor_id", els.psdSensor.value);
      }
      appendFilterParameters(params);
      try {
        const res = await fetch(`/api/psd-fft?${params}`, { cache: "no-store" });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.message || data.error || "PSD computation failed");
        updatePsdDiagnostics(data);
        drawWaveformChart(data);
        drawPsdChart(data);
        const steadyPeak = finiteMetricNumber(data.peak_frequency_hz);
        const impact = data.impact_analysis || {};
        const impactPeak = finiteMetricNumber(data.impact_peak_frequency_hz ?? impact.peak_frequency_hz);
        const peakText = steadyPeak !== null
          ? (data.peak_is_competing && Number.isFinite(Number(data.competing_peak_frequency_hz))
            ? `peaks ${fmt(steadyPeak, 2)} / ${fmt(data.competing_peak_frequency_hz, 2)} Hz are ${fmt(data.peak_lead_db, 2)} dB apart`
            : `dominant peak ${fmt(steadyPeak, 2)} Hz`)
          : (impactPeak !== null
            ? `impact ring-down peak ${fmt(impactPeak, 2)} Hz${impact.peak_reliable ? "" : " (low confidence)"}`
            : (impact.status === "need_more_post_event_data"
              ? "impact detected · wait briefly and run again"
              : (impact.detected ? "broadband impact · no resonant ring-down peak" : "no distinct peak")));
        const actualSource = (data.source || {}).sensor_id || data.machine_id || sensorId;
        setStatus("ok", `${actualSource} · ${estimatorLabel} ready · ${peakText}`);
      } catch (err) {
        const message = String(err.message || err);
        els.psdSummary.textContent = "Spectrum failed";
        els.psdFloatingStatus.textContent = message;
        els.psdInsights.textContent = message;
        if (!lastPsdResult) {
          drawWaveformChart(null);
          drawPsdChart(null);
        }
        setStatus("bad", "Spectrum failed");
      } finally {
        psdRequestInFlight = false;
        els.psdButton.disabled = false;
        els.psdQuickButton.disabled = false;
        els.psdOpenButton.disabled = false;
        els.psdSensor.disabled = false;
      }
    }

    async function sendMessage(text) {
      const message = text.trim();
      if (!message) return;
      addMessage("user", message);
      els.message.value = "";
      resizeComposer();
      els.send.disabled = true;
      // Pause autonomous agent checks for the duration of this chat: the single NPU serializes
      // every request, so an in-flight/queued check would starve this chat and it would never
      // reach the model. fetchAgentMonitor() early-returns while chatBusy is set.
      chatBusy = true;
      setStatus("", "Reading sensors and waiting for model explanation");
      setModelActivity("", "Model working");
      const pending = addMessage("assistant", "Reading the requested building data and generating the model explanation…");
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            base_url: els.baseUrl.value,
            model: els.model.value,
            api_key: els.apiKey.value || "EMPTY",
            timeout_s: 300,
            max_tokens: 256,
            max_prompt_chars: 9000,
            model_tool_routing: false,
          }),
        });
        let data = {};
        try {
          data = await res.json();
        } catch (_jsonError) {
          data = {};
        }
        if (!res.ok || !data.ok) {
          const error = new Error(data.message || data.error || `chat failed (${res.status})`);
          error.code = data.error || "chat_failed";
          throw error;
        }
        if (!data.model_request_sent || !data.model_response_used || !data.model_explanation_only || !String(data.answer || "").trim()) {
          const error = new Error("The server returned no verified model-generated explanation. No fallback was shown.");
          error.code = "model_explanation_failed";
          throw error;
        }
        revealMarkdown(pending, data.answer);
        updateDiagnostics(data);
        setStatus("ok", `Model answered: ${data.model}`);
      } catch (err) {
        pending.remove();
        const messageText = String(err.message || err);
        addMessage("error", messageText);
        const timedOut = err.code === "model_timeout" || /timeout|timed out/i.test(messageText);
        setModelActivity("bad", timedOut ? "Model timeout · no fallback" : "No model answer · no fallback");
        setStatus("bad", timedOut ? "Model timed out" : "Model explanation failed");
      } finally {
        els.send.disabled = false;
        els.message.focus();
        // Resume agent checks; the periodic timer will run the next check on its own.
        chatBusy = false;
      }
    }

    els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(els.message.value);
    });

    els.checkModel.addEventListener("click", checkModel);
    els.baseUrl.addEventListener("input", updateConnectionSummary);
    els.model.addEventListener("input", updateConnectionSummary);
    els.message.addEventListener("input", resizeComposer);
    els.message.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        els.form.requestSubmit();
      }
    });
    els.psdButton.addEventListener("click", loadPsdFft);
    els.psdQuickButton.addEventListener("click", () => openPsdFloatingWindow({ run: true }));
    els.psdOpenButton.addEventListener("click", () => openPsdFloatingWindow({ run: !lastPsdResult }));
    els.psdClose.addEventListener("click", closePsdFloatingWindow);
    els.psdMaximize.addEventListener("click", togglePsdMaximize);
    els.psdSensor.addEventListener("change", () => {
      applySelectedPsdBoardAxis();
      const selectedId = selectedPsdSensorLabel();
      const displayedId = lastPsdResult && lastPsdResult.source
        ? lastPsdResult.source.sensor_id
        : (lastPsdResult ? lastPsdResult.machine_id : null);
      if (lastPsdResult && displayedId && displayedId !== selectedId) {
        lastPsdResult = null;
        drawWaveformChart(null);
        drawPsdChart(null);
        renderPsdInsights(null);
      }
      els.psdSummary.textContent = `${selectedId} selected · press Run`;
      els.psdFloatingStatus.textContent = `${selectedId} · source locked for the next run`;
      els.psdCompactSummary.textContent = `${selectedId} · spectrum not run`;
    });
    els.psdScale.addEventListener("change", () => { if (lastPsdResult) loadPsdFft(); });
    els.psdRange.addEventListener("change", () => { if (lastPsdResult) drawPsdChart(lastPsdResult); });
    els.psdEstimator.addEventListener("change", () => {
      els.psdTitle.textContent = els.psdEstimator.value === "welch" ? "Welch PSD" : "Raw FFT PSD";
    });
    els.psdFilter.addEventListener("change", () => {
      updatePsdFilterControls();
      els.psdWaveformLabel.textContent = selectedFilterLabel();
    });
    [els.psdLowCutoff, els.psdHighCutoff, els.psdNotchFrequency, els.psdNotchWidth].forEach((input) => {
      input.addEventListener("input", () => { els.psdWaveformLabel.textContent = selectedFilterLabel(); });
    });
    initializePsdWindowDrag();
    updatePsdFilterControls();
    loadPsdSensorOptions();
    if (window.ResizeObserver) {
      const psdResizeObserver = new ResizeObserver(() => requestAnimationFrame(redrawPsdWindow));
      psdResizeObserver.observe(els.psdFloatingPanel);
    }
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !els.psdFloatingPanel.hidden) closePsdFloatingWindow();
    });

    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => sendMessage(button.dataset.prompt));
    });

    let featureResizeRaf = 0;
    window.addEventListener("resize", () => {
      if (!lastFeatureResult && !lastPsdResult) return;
      cancelAnimationFrame(featureResizeRaf);
      featureResizeRaf = requestAnimationFrame(() => {
        if (lastFeatureResult) drawFeatureBars(lastFeatureResult);
        redrawPsdWindow();
      });
    });

    loadConfig().then(() => {
      addMessage("assistant", "Ready. Ask about the registered building sensors, compare them with the reference, or open the spectrum viewer.");
      resizeComposer();
    }).catch((err) => addMessage("error", String(err.message || err)));
  </script>
</body>
</html>
"""


_GRAPH_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VibroAgent Live Graph</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #eef1f5;
      --surface: #ffffff;
      --line: #d6dee7;
      --ink: #1a2433;
      --muted: #5d6b7c;
      --teal: #0469b1;
      --teal-2: #e1eef9;
      --rust: #03234e;
      --amber: #f6a800;
      --bad: #d0021b;
      --shadow: 0 10px 28px rgba(3, 35, 78, 0.10);
      --st-blue: #0469b1;
      --st-cyan: #00a3e0;
      --st-navy: #03234e;
      --st-blue-line: #0469b1;
    }

    * { box-sizing: border-box; }

    html,
    body {
      height: 100%;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
      font: 14px/1.5 "Helvetica Neue", Helvetica, Arial, "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }

    body::before { display: none; content: none; }

    button, input, select {
      font: inherit;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 0;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 28px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }

    .mark {
      width: 58px;
      height: 44px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      background: transparent;
    }
    .mark img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 720;
    }

    .subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .nav-link {
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      padding: 0 4px;
      border: 0;
      border-bottom: 3px solid transparent;
      border-radius: 0;
      color: var(--st-navy);
      text-decoration: none;
      font-weight: 700;
      background: transparent;
      white-space: nowrap;
      letter-spacing: 0.2px;
    }
    .nav-link:hover { color: var(--st-blue); border-bottom-color: var(--st-blue); }

    .stfoot {
      grid-column: 1 / -1;
      background: var(--st-navy);
      color: #cdd7e4;
      font-size: 12px;
      padding: 9px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 0;
      border-image: none;
    }
    .stfoot a { color: #fff; text-decoration: none; }
    .stfoot strong { color: #fff; font-weight: 700; }
    .stfoot .tag { color: #8aa0bd; letter-spacing: 0.3px; }

    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--amber);
      box-shadow: 0 0 0 3px rgba(242, 195, 93, 0.25);
    }

    .dot.ok {
      background: var(--teal);
      box-shadow: 0 0 0 3px rgba(40, 102, 92, 0.2);
    }

    .dot.bad {
      background: var(--bad);
      box-shadow: 0 0 0 3px rgba(181, 61, 61, 0.18);
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(190px, 1.4fr) minmax(96px, 0.6fr) minmax(96px, 0.6fr) minmax(96px, 0.6fr) minmax(120px, 0.8fr) minmax(116px, 0.7fr) auto auto auto auto auto auto;
      gap: 10px;
      padding: 12px 16px;
      background: #fbfcfa;
      border-bottom: 1px solid var(--line);
    }

    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }

    input,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 2px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      outline: none;
    }

    input:focus,
    select:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(40, 102, 92, 0.13);
    }

    .checkline {
      display: flex;
      align-items: end;
      gap: 8px;
      padding-bottom: 8px;
      color: var(--ink);
      font-size: 13px;
      white-space: nowrap;
    }

    .checkline input {
      width: auto;
      transform: translateY(-1px);
    }

    .button {
      align-self: end;
      height: 38px;
      padding: 0 12px;
      border: 1px solid var(--teal);
      border-radius: 2px;
      background: var(--teal-2);
      color: var(--teal);
      cursor: pointer;
      font-weight: 650;
      white-space: nowrap;
    }

    .button.secondary {
      border-color: var(--line);
      background: #fff;
      color: var(--ink);
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      padding: 16px;
      min-height: 0;
      overflow: hidden;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 3px;
      box-shadow: var(--shadow);
      min-height: 0;
    }

    .graph-panel {
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
      white-space: nowrap;
    }

    .canvas-wrap {
      min-height: 0;
      padding: 14px;
    }

    canvas {
      width: 100%;
      height: 100%;
      min-height: 420px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: #fbfcfa;
    }

    .side {
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }

    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 10px;
      background: #fbfcfa;
      min-width: 0;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }

    .metric strong {
      display: block;
      font-size: 17px;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }

    .timing {
      padding: 0 12px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .timing table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      line-height: 1.25;
    }

    .timing th,
    .timing td {
      padding: 6px 4px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }

    .timing th:first-child,
    .timing td:first-child {
      text-align: left;
      max-width: 128px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .timing tr:last-child td {
      border-bottom: 0;
    }

    .timing .warn {
      color: var(--bad);
      font-weight: 700;
    }

    .timing td.good { color: var(--teal); }
    .timing td.amber { color: #9a7400; }
    .timing td.bad { color: var(--bad); font-weight: 700; }
    .timing td.spark { padding: 2px 4px; width: 56px; }
    .timing svg.spark-svg { display: block; width: 52px; height: 16px; }
    .timing .state {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
    }
    .timing .state.live { color: var(--teal); background: var(--teal-2); }
    .timing .state.stale { color: var(--bad); background: rgba(181, 61, 61, 0.12); }
    .timing .state.err { color: #fff; background: var(--bad); }

    .legend {
      display: flex;
      flex: 1 1 auto;
      flex-wrap: wrap;
      gap: 4px 10px;
      align-items: center;
      justify-content: center;
      margin: 0 10px;
      max-height: 34px;
      overflow: hidden;
    }

    .legend:empty { display: none; }

    .legend .lg {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
    }

    .legend .lg.muted { opacity: 0.45; }

    .legend .sw {
      width: 10px;
      height: 10px;
      border-radius: 2px;
      flex: none;
    }

    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      font-size: 11px;
      line-height: 1.45;
      background: #fbfcfa;
      color: #26312d;
    }

    .agent-alert {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 20;
      width: min(420px, calc(100vw - 36px));
      border: 1px solid rgba(181, 61, 61, 0.35);
      border-left: 5px solid var(--bad);
      border-radius: 3px;
      background: #fff;
      box-shadow: 0 18px 46px rgba(31, 42, 38, 0.2);
      padding: 14px;
      display: grid;
      gap: 10px;
    }

    .agent-alert.hidden {
      display: none;
    }

    .agent-alert.notice {
      border-color: rgba(169, 86, 53, 0.35);
      border-left-color: var(--rust);
    }

    .agent-alert.quality {
      border-color: rgba(242, 195, 93, 0.45);
      border-left-color: var(--amber);
    }

    .agent-alert-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 10px;
    }

    .agent-alert h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }

    .agent-alert p {
      margin: 0;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.4;
    }

    .agent-alert .meta {
      color: var(--muted);
      font-size: 12px;
    }

    .anomaly-log {
      position: fixed;
      right: 18px;
      top: 84px;
      bottom: 52px;
      z-index: 19;
      width: min(460px, calc(100vw - 36px));
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface, #fff);
      box-shadow: 0 18px 46px rgba(3, 35, 78, 0.18);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .anomaly-log.hidden {
      display: none;
    }

    .anomaly-log-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }

    .anomaly-log-head h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
    }

    .anomaly-log-head .panel-subtitle {
      display: block;
      word-break: break-all;
    }

    .anomaly-log-actions {
      display: flex;
      gap: 8px;
      flex: 0 0 auto;
    }

    .anomaly-log-list {
      overflow-y: auto;
      padding: 10px 14px;
      display: grid;
      gap: 10px;
      align-content: start;
    }

    .anomaly-item {
      border: 1px solid var(--line);
      border-left: 4px solid var(--bad);
      border-radius: 4px;
      background: #fff;
      padding: 10px 12px;
      display: grid;
      gap: 6px;
      cursor: pointer;
    }

    .anomaly-item.notice {
      border-left-color: var(--rust);
    }

    .anomaly-item.quality {
      border-left-color: var(--amber);
    }

    .anomaly-item-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
    }

    .anomaly-item-head strong {
      font-size: 13px;
    }

    .anomaly-item .when {
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }

    .anomaly-item .meta {
      color: var(--muted);
      font-size: 12px;
    }

    .anomaly-item .summary {
      display: none;
      margin: 0;
      font-size: 12.5px;
      line-height: 1.45;
      color: var(--ink);
    }

    .anomaly-item.open .summary {
      display: block;
    }

    .anomaly-log-empty {
      color: var(--muted);
      font-size: 12.5px;
      padding: 8px 2px;
    }

    @media (max-width: 900px) {
      body {
        overflow: auto;
      }

      .app {
        min-height: 100vh;
        height: auto;
      }

      header,
      .top-actions {
        align-items: flex-start;
      }

      header,
      .controls,
      main {
        grid-template-columns: 1fr;
      }

      main {
        overflow: visible;
      }

      canvas {
        height: 420px;
      }
    }


    /* ST-inspired live-data UI. */
    :root {
      --paper: #f3f5f7;
      --surface: #ffffff;
      --line: #d8e0e8;
      --ink: #17243a;
      --muted: #617084;
      --teal: #0469b1;
      --teal-2: #eaf4fb;
      --shadow: 0 3px 12px rgba(3, 35, 78, 0.08);
    }
    body { background: var(--paper); }
    body::before { display: none; content: none; }
    .app { grid-template-rows: 68px auto minmax(0, 1fr) 34px; }
    header { min-height: 68px; padding: 0 24px; box-shadow: 0 1px 0 rgba(3,35,78,.04); }
    .brand { gap: 14px; }
    .mark {
      width: 58px;
      height: 44px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      background: transparent;
    }
    .mark img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    h1 { font-size: 20px; font-weight: 750; letter-spacing: -.2px; }
    .subtitle { margin-top: 3px; }
    .top-actions { gap: 18px; }
    .app-nav { display: flex; align-items: center; gap: 18px; }
    .nav-link { min-height: 65px; padding: 0 2px; font-size: 13px; }
    .status { min-height: 32px; padding: 5px 10px; border: 1px solid var(--line); border-radius: 999px; background: #f8fafc; font-size: 12px; }

    .graph-toolbar { border-bottom: 1px solid var(--line); background: #fff; }
    .graph-toolbar-inner { width: 100%; max-width: 1600px; margin: 0 auto; padding: 11px 24px; }
    .toolbar-primary { display: flex; align-items: end; gap: 10px; flex-wrap: wrap; }
    .toolbar-primary label { min-width: 116px; }
    .toolbar-primary label:first-child { min-width: 220px; flex: 1 1 260px; max-width: 430px; }
    label { font-size: 11px; font-weight: 650; }
    input, select { height: 36px; padding: 7px 9px; border-radius: 3px; }
    input:focus, select:focus { border-color: var(--st-blue); box-shadow: 0 0 0 3px rgba(4,105,177,.12); }
    .button { height: 36px; border-radius: 3px; background: var(--st-blue); color: #fff; }
    .button:hover { background: #035a99; }
    .button.secondary { border-color: #c9d5e1; background: #fff; color: var(--st-navy); }

    .toolbar-more { margin-top: 10px; border-top: 1px solid var(--line); }
    .toolbar-more > summary {
      min-height: 36px;
      display: flex;
      align-items: center;
      gap: 10px;
      list-style: none;
      cursor: pointer;
      color: var(--st-navy);
      font-size: 11px;
      font-weight: 700;
    }
    .toolbar-more > summary::-webkit-details-marker { display: none; }
    .toolbar-more > summary::after { content: "+"; margin-left: auto; color: var(--st-blue); font-size: 16px; font-weight: 500; }
    .toolbar-more[open] > summary::after { content: "−"; }
    .toolbar-more > summary span:last-child { color: var(--muted); font-weight: 500; }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 10px;
      padding: 2px 0 12px;
      border: 0;
      background: transparent;
    }
    .checkline { align-self: stretch; min-height: 36px; align-items: center; padding: 0 8px; border: 1px solid var(--line); border-radius: 3px; background: #f8fafc; }
    .controls .button { align-self: end; }

    main { width: 100%; max-width: 1600px; margin: 0 auto; grid-template-columns: minmax(0,1fr) 320px; gap: 16px; padding: 16px 24px; }
    .panel { border-radius: 7px; box-shadow: var(--shadow); }
    .panel-head { min-height: 54px; padding: 12px 15px; background: #fff; }
    .graph-panel .panel-head { border-top: 3px solid var(--st-blue); }
    .panel-head h2 { font-size: 16px; }
    .panel-subtitle { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }
    .canvas-wrap { padding: 12px; background: #fff; }
    canvas { border-radius: 4px; background: #fff; }

    .side { display: flex; flex-direction: column; overflow-y: auto; }
    .side > .panel-head { flex: 0 0 auto; border-top: 3px solid var(--st-cyan); }
    .metrics { flex: 0 0 auto; gap: 0; padding: 0; }
    .metric { min-height: 66px; padding: 12px 13px; border: 0; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); border-radius: 0; background: #fff; }
    .metric:nth-child(even) { border-right: 0; }
    .metric strong { font-size: 16px; }

    .side-details { flex: 0 0 auto; border-bottom: 1px solid var(--line); background: #fff; }
    .side-details > summary {
      min-height: 42px;
      padding: 0 13px;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      list-style: none;
      color: var(--st-navy);
      font-size: 11px;
      font-weight: 700;
    }
    .side-details > summary::-webkit-details-marker { display: none; }
    .side-details > summary::after { content: "+"; margin-left: auto; color: var(--st-blue); font-size: 16px; font-weight: 500; }
    .side-details[open] > summary::after { content: "−"; }
    .side-details > summary span:nth-child(2) { color: var(--muted); font-weight: 500; }
    .timing { padding: 0 12px 12px; border: 0; }
    .raw-details { margin-top: auto; }
    pre { max-height: 280px; padding: 12px 13px; border-top: 1px solid var(--line); background: #f8fafc; }

    .agent-alert { right: 24px; bottom: 50px; border-radius: 6px; box-shadow: 0 18px 46px rgba(3,35,78,.18); }
    .stfoot { padding: 8px 24px; }

    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; grid-template-rows: auto auto 1fr auto; }
      header { padding: 12px 16px; align-items: flex-start; flex-direction: column; }
      .top-actions { width: 100%; justify-content: flex-start; flex-wrap: wrap; gap: 8px 12px; }
      .app-nav { width: 100%; gap: 14px; }
      .nav-link { min-height: 34px; }
      .status { flex: 1 0 100%; width: 100%; max-width: 100%; min-width: 0; }
      .status span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .graph-toolbar-inner { padding: 10px 14px; }
      .toolbar-primary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar-primary label { min-width: 0; }
      .toolbar-primary label:first-child { grid-column: 1 / -1; max-width: none; }
      .toolbar-primary .button { width: 100%; }
      main { grid-template-columns: 1fr; padding: 12px 14px; overflow: visible; }
      .graph-panel { min-height: 470px; }
      .side { min-height: 360px; }
    }

    @media (max-width: 520px) {
      .toolbar-primary { grid-template-columns: 1fr 1fr; }
      .toolbar-primary label:nth-child(2),
      .toolbar-primary label:nth-child(3) { min-width: 0; }
      .controls { grid-template-columns: 1fr; }
      main { padding: 10px; }
      .canvas-wrap { padding: 8px; }
      canvas { min-height: 340px; }
    }

    /* ST logo. */
    body::before { display: none !important; content: none !important; }
    .mark {
      width: 112px !important;
      height: 48px !important;
      min-width: 112px !important;
      padding: 0 !important;
      border: 0 !important;
      border-radius: 0 !important;
      background: transparent !important;
      color: transparent !important;
      font-size: 0 !important;
      letter-spacing: 0 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
    }
    .mark::after { display: none !important; content: none !important; }
    .st-logo {
      display: block;
      width: 112px;
      max-width: 112px;
      max-height: 48px;
      height: auto;
      object-fit: contain;
    }
    .stfoot { border-top: 0 !important; border-image: none !important; }

  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark"><img class="st-logo" src="/assets/st-logo.png" alt="STMicroelectronics logo"></div>
        <div>
          <h1>VibroAgent</h1>
          <div class="subtitle">Live building vibration data</div>
        </div>
      </div>
      <div class="top-actions">
        <nav class="app-nav" aria-label="Application navigation">
          <a class="nav-link" href="/">Analysis chat</a>
          <a class="nav-link" href="/npu-chat">NPU console</a>
        </nav>
        <div class="status" aria-live="polite"><span id="statusDot" class="dot"></span><span id="statusText">Waiting</span></div>
      </div>
    </header>

    <section class="graph-toolbar" aria-label="Graph controls">
      <div class="graph-toolbar-inner">
        <div class="toolbar-primary">
          <label>Board
            <select id="sensorSelect">
              <option value="">Loading boards</option>
            </select>
          </label>
          <label>Axis
            <select id="axis">
              <option value="norm">Norm</option>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </label>
          <label>Window
            <select id="duration">
              <option value="0.5">0.5 s</option>
              <option value="1" selected>1 s</option>
              <option value="2">2 s</option>
              <option value="5">5 s</option>
            </select>
          </label>
          <button id="toggle" class="button" type="button">Pause</button>
          <button id="refreshNow" class="button secondary" type="button">Refresh now</button>
          <button id="anomalyLogBtn" class="button secondary" type="button" title="Saved anomaly popups from the agent monitor">Anomalies</button>
        </div>
        <details class="toolbar-more">
          <summary><span>View and agent settings</span><span>refresh, point count and comparison overlays</span></summary>
          <div class="controls">
            <label>Refresh interval
              <select id="refresh">
                <option value="500">0.5 s</option>
                <option value="1000" selected>1 s</option>
                <option value="2000">2 s</option>
              </select>
            </label>
            <label>Agent interval
              <select id="agentInterval">
                <option value="5000">5 s</option>
                <option value="10000">10 s</option>
                <option value="30000" selected>30 s</option>
                <option value="60000">60 s</option>
              </select>
            </label>
            <label title="Re-decision probes behind the confidence gate: more probes = finer uncertainty detection, fewer = faster checks.">Confidence probes
              <select id="voteK">
                <option value="5" selected>5</option>
                <option value="3">3</option>
                <option value="1">1</option>
              </select>
            </label>
            <label>Plot points
              <input id="points" type="number" min="200" max="5000" step="100" value="1800">
            </label>
            <label class="checkline"><input id="center" type="checkbox" checked> Center signal</label>
            <label class="checkline" title="Scale every board to one shared amplitude axis so target boards are directly comparable to the baseline."><input id="sharedScale" type="checkbox"> Shared amplitude scale</label>
            <label class="checkline" title="Overlay the baseline board's waveform (faint) on every target panel."><input id="overlayBaseline" type="checkbox"> Baseline overlay</label>
            <button id="agentToggle" class="button" type="button" aria-pressed="true" title="Live agent is enabled; click to turn it off.">Agent: On</button>
          </div>
        </details>
      </div>
    </section>

    <main>
      <section class="panel graph-panel" aria-label="Live waveform graphs">
        <div class="panel-head">
          <h2 id="chartTitle">Acceleration</h2>
          <div id="legend" class="legend" aria-hidden="true"></div>
          <span id="sourceChip" class="chip">No data</span>
        </div>
        <div class="canvas-wrap">
          <canvas id="waveform"></canvas>
        </div>
      </section>

      <aside class="panel side" aria-label="Live graph metrics">
        <div class="panel-head">
          <div><h2>Sensor summary</h2><span class="panel-subtitle">Latest decoded window</span></div>
          <span id="liveChip" class="chip">--</span>
        </div>
        <div class="metrics">
          <div class="metric"><span>Samples</span><strong id="samples">--</strong></div>
          <div class="metric"><span>Rate</span><strong id="rate">--</strong></div>
          <div class="metric"><span>Mean</span><strong id="mean">--</strong></div>
          <div class="metric"><span>RMS</span><strong id="rms">--</strong></div>
          <div class="metric"><span>Peak-to-peak</span><strong id="peak">--</strong></div>
          <div class="metric"><span>Live age</span><strong id="age">--</strong></div>
          <div class="metric"><span>SDK decode</span><strong id="decode">--</strong></div>
          <div class="metric"><span>SDK gap</span><strong id="behind">--</strong></div>
        </div>
        <details class="side-details">
          <summary><span>Board timing</span><span>age, SDK gap and trend</span></summary>
          <div class="timing" aria-label="Decode timing by board">
            <table>
              <thead>
                <tr><th>Board</th><th>Age</th><th>Gap</th><th>Trend</th></tr>
              </thead>
              <tbody id="timingBody">
                <tr><td colspan="4">No timing yet.</td></tr>
              </tbody>
            </table>
          </div>
        </details>
        <details class="side-details raw-details">
          <summary><span>Raw response</span><span>developer payload</span></summary>
          <pre id="payload">No payload yet.</pre>
        </details>
      </aside>
    </main>

    <section id="agentAlert" class="agent-alert hidden" role="alert" aria-live="assertive">
      <div class="agent-alert-head">
        <h2 id="agentAlertTitle">Building Vibration Anomaly</h2>
        <button id="agentDismiss" class="button secondary" type="button">Dismiss</button>
      </div>
      <p id="agentAlertBody"></p>
      <div id="agentAlertMeta" class="meta"></div>
    </section>

    <section id="anomalyLog" class="anomaly-log hidden" role="dialog" aria-label="Saved anomaly popups">
      <div class="anomaly-log-head">
        <div>
          <h2>Saved anomalies</h2>
          <span id="anomalyLogInfo" class="panel-subtitle">Loading saved anomaly windows</span>
        </div>
        <div class="anomaly-log-actions">
          <button id="anomalyLogRefresh" class="button secondary" type="button">Refresh</button>
          <button id="anomalyLogClose" class="button secondary" type="button">Close</button>
        </div>
      </div>
      <div id="anomalyLogList" class="anomaly-log-list"><div class="anomaly-log-empty">Loading saved anomalies.</div></div>
    </section>
      <footer class="stfoot"><span><strong>STMicroelectronics</strong> &middot; VibroAgent &mdash; live acceleration</span><span class="tag">life.augmented</span></footer>
  </div>

  <script>
    const els = {
      sensorSelect: document.getElementById("sensorSelect"),
      axis: document.getElementById("axis"),
      duration: document.getElementById("duration"),
      refresh: document.getElementById("refresh"),
      agentInterval: document.getElementById("agentInterval"),
      voteK: document.getElementById("voteK"),
      points: document.getElementById("points"),
      center: document.getElementById("center"),
      sharedScale: document.getElementById("sharedScale"),
      overlayBaseline: document.getElementById("overlayBaseline"),
      legend: document.getElementById("legend"),
      agentToggle: document.getElementById("agentToggle"),
      toggle: document.getElementById("toggle"),
      refreshNow: document.getElementById("refreshNow"),
      statusDot: document.getElementById("statusDot"),
      statusText: document.getElementById("statusText"),
      chartTitle: document.getElementById("chartTitle"),
      sourceChip: document.getElementById("sourceChip"),
      liveChip: document.getElementById("liveChip"),
      samples: document.getElementById("samples"),
      rate: document.getElementById("rate"),
      mean: document.getElementById("mean"),
      rms: document.getElementById("rms"),
      peak: document.getElementById("peak"),
      age: document.getElementById("age"),
      decode: document.getElementById("decode"),
      behind: document.getElementById("behind"),
      timingBody: document.getElementById("timingBody"),
      payload: document.getElementById("payload"),
      canvas: document.getElementById("waveform"),
      agentAlert: document.getElementById("agentAlert"),
      agentAlertTitle: document.getElementById("agentAlertTitle"),
      agentAlertBody: document.getElementById("agentAlertBody"),
      agentAlertMeta: document.getElementById("agentAlertMeta"),
      agentDismiss: document.getElementById("agentDismiss"),
      anomalyLogBtn: document.getElementById("anomalyLogBtn"),
      anomalyLog: document.getElementById("anomalyLog"),
      anomalyLogInfo: document.getElementById("anomalyLogInfo"),
      anomalyLogList: document.getElementById("anomalyLogList"),
      anomalyLogRefresh: document.getElementById("anomalyLogRefresh"),
      anomalyLogClose: document.getElementById("anomalyLogClose"),
    };

    let running = true;
    let agentRunning = true;
    let timer = null;
    let agentTimer = null;
    let inFlight = false;
    let agentInFlight = false;
    let chatBusy = false;
    let latestPayload = null;
    let sensorOptions = [];
    let baselineSensorId = "";
    const BOARD_COLORS = ["#0469b1", "#00a3e0", "#03234e", "#00857a", "#f6a800", "#5b6b7c", "#6aa5d8", "#2f80c0"];
    const gapHistory = new Map();
    let lastAgentAlertKey = "";
    let lastAgentAlertAt = 0;
    const ALL_SENSORS = "__all__";
    const AGENT_ALERT_COOLDOWN_MS = 30000;

    const fmt = (value, digits = 3) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits);
    };

    const numeric = (value) => {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };

    const labelize = (value) => value ? String(value).replace(/_/g, " ") : "--";

    const fmtMs = (seconds) => {
      const value = numeric(seconds);
      if (value === null) return "--";
      if (value < 1) return `${Math.round(value * 1000)} ms`;
      return `${fmt(value, 2)} s`;
    };

    const fmtLag = (seconds) => {
      const value = numeric(seconds);
      if (value === null) return "--";
      if (value < 0.01) return `${Math.round(value * 1000)} ms`;
      return `${fmt(value, 2)} s`;
    };

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

    function diagnosticsFor(reading) {
      const metadata = reading && reading.metadata ? reading.metadata : {};
      const diagnostics = reading && reading.diagnostics ? reading.diagnostics : {};
      return {
        decode_s: numeric(diagnostics.decode_duration_s),
        server_s: numeric(diagnostics.server_duration_s),
        live_s: numeric(diagnostics.live_latency_s ?? diagnostics.data_file_age_s ?? metadata.data_file_age_s),
        sdk_gap_s: numeric(
          diagnostics.sdk_timestamp_gap_s
          ?? diagnostics.decoded_latest_gap_s
          ?? metadata.sdk_latest_timestamp_gap_s
        ),
        file_age_s: numeric(diagnostics.data_file_age_s ?? metadata.data_file_age_s),
      };
    }

    function pushGapHistory(board, gap, age) {
      const key = String(board || "sensor");
      const arr = gapHistory.get(key) || [];
      arr.push({ gap: Number.isFinite(gap) ? gap : null, age: Number.isFinite(age) ? age : null });
      while (arr.length > 40) arr.shift();
      gapHistory.set(key, arr);
      return arr;
    }

    function thresholdClass(value, goodMax, amberMax) {
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      if (value <= goodMax) return "good";
      if (value <= amberMax) return "amber";
      return "bad";
    }

    function statePill(kind) {
      if (kind === "live") return '<span class="state live">live</span>';
      if (kind === "stale") return '<span class="state stale">stale</span>';
      return '<span class="state err">err</span>';
    }

    function sparklineSvg(history) {
      const points = (history || []).map((item) => (item && Number.isFinite(item.gap)) ? item.gap : null);
      const valid = points.filter((value) => value !== null);
      if (valid.length < 2) return '<svg class="spark-svg" viewBox="0 0 52 16" preserveAspectRatio="none"></svg>';
      const max = Math.max(...valid, 0.05);
      const min = Math.min(...valid, 0);
      const span = Math.max(1e-6, max - min);
      const n = points.length;
      const coords = points.map((value, index) => {
        if (value === null) return null;
        const px = (index / (n - 1)) * 52;
        const py = 15 - ((value - min) / span) * 14;
        return `${px.toFixed(1)},${py.toFixed(1)}`;
      }).filter(Boolean).join(" ");
      const last = valid[valid.length - 1];
      const color = last <= 0.25 ? "#0469b1" : (last <= 0.75 ? "#9a7400" : "#d0021b");
      return `<svg class="spark-svg" viewBox="0 0 52 16" preserveAspectRatio="none"><polyline points="${coords}" fill="none" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/></svg>`;
    }

    function renderTimingTable(readings, failures = []) {
      const rows = (readings || []).map((reading) => {
        const sensor = reading.sensor_config || {};
        const board = reading.machine_id || sensor.sensor_id || "sensor";
        const safeBoard = escapeHtml(board);
        const timing = diagnosticsFor(reading);
        const current = !!(reading.metadata || {}).data_is_current;
        const history = pushGapHistory(board, timing.sdk_gap_s, timing.live_s);
        const ageClass = thresholdClass(timing.live_s, 1, 5);
        const gapClass = thresholdClass(timing.sdk_gap_s, 0.25, 0.75);
        return `<tr>`
          + `<td title="${safeBoard}">${statePill(current ? "live" : "stale")} ${safeBoard}</td>`
          + `<td class="${ageClass}">${fmtLag(timing.live_s)}</td>`
          + `<td class="${gapClass}">${fmtLag(timing.sdk_gap_s)}</td>`
          + `<td class="spark">${sparklineSvg(history)}</td>`
          + `</tr>`;
      });
      (failures || []).forEach((failure) => {
        const board = escapeHtml(failure.sensor_id || "sensor");
        const error = escapeHtml(failure.error || "read failed");
        rows.push(`<tr><td title="${error}">${statePill("err")} ${board}</td><td class="bad">--</td><td class="bad">--</td><td class="spark"></td></tr>`);
      });
      els.timingBody.innerHTML = rows.length ? rows.join("") : `<tr><td colspan="4">No timing yet.</td></tr>`;
    }

    function setStatus(kind, text) {
      els.statusDot.className = "dot" + (kind ? " " + kind : "");
      els.statusText.textContent = text;
    }

    function fallbackSensorOptions() {
      return [{
        sensor_id: "stwinbox_iis3dwb",
        location: "Latest acquisition",
        role: "sensor",
        acquisition_folder: "",
        hsd_sensor_name: "iis3dwb_acc",
        axis: "norm",
      }];
    }

    function selectedSensorConfig() {
      if (els.sensorSelect.value === ALL_SENSORS) return null;
      return sensorOptions.find((sensor) => sensor.sensor_id === els.sensorSelect.value) || sensorOptions[0] || null;
    }

    function allSensorsSelected() {
      return els.sensorSelect.value === ALL_SENSORS;
    }

    function sensorOptionLabel(sensor) {
      const location = sensor.location ? ` · ${sensor.location}` : "";
      return `${sensor.sensor_id}${location} · ${sensor.role || "sensor"}`;
    }

    function renderSensorOptions(payload) {
      sensorOptions = Array.isArray(payload.sensors) && payload.sensors.length ? payload.sensors : fallbackSensorOptions();
      baselineSensorId = payload.baseline_sensor_id || "";
      const preferred = sensorOptions.length > 1 ? ALL_SENSORS : (payload.baseline_sensor_id || sensorOptions[0].sensor_id);
      els.sensorSelect.replaceChildren();
      if (sensorOptions.length > 1) {
        const allOption = document.createElement("option");
        allOption.value = ALL_SENSORS;
        allOption.textContent = `All boards (${sensorOptions.length})`;
        els.sensorSelect.appendChild(allOption);
      }
      sensorOptions.forEach((sensor) => {
        const option = document.createElement("option");
        option.value = sensor.sensor_id;
        option.textContent = sensorOptionLabel(sensor);
        els.sensorSelect.appendChild(option);
      });
      els.sensorSelect.value = preferred === ALL_SENSORS || sensorOptions.some((sensor) => sensor.sensor_id === preferred)
        ? preferred
        : sensorOptions[0].sensor_id;
    }

    async function loadSensors() {
      try {
        const res = await fetch("/api/live-sensors?process_reader=1&prewarm=1", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.message || data.error || "sensor list failed");
        renderSensorOptions(data);
      } catch (err) {
        renderSensorOptions({ sensors: fallbackSensorOptions() });
        setStatus("bad", "Sensor list failed");
      }
    }

    function graphUrl(sensor) {
      const selected = sensor === undefined ? selectedSensorConfig() : sensor;
      const params = new URLSearchParams({
        machine_id: selected ? selected.sensor_id : "stwinbox_iis3dwb",
        sensor_name: selected && selected.hsd_sensor_name ? selected.hsd_sensor_name : "iis3dwb_acc",
        axis: els.axis.value,
        duration_s: els.duration.value,
        max_points: els.points.value,
        require_current: "1",
        process_reader: "1",
      });
      if (selected && selected.acquisition_folder) {
        params.set("acquisition_folder", selected.acquisition_folder);
      }
      return `/api/live-vibrometer?${params}`;
    }

    async function readAllSensorResults() {
      setStatus("", `Reading ${sensorOptions.length} boards in parallel`);
      return Promise.all(sensorOptions.map((sensor) => readSensorResult(sensor)));
    }

    function clearReadout(message, failures = []) {
      latestPayload = null;
      els.chartTitle.textContent = "Live IIS3DWB acceleration";
      els.sourceChip.textContent = "No current live data";
      els.liveChip.textContent = "Error";
      els.samples.textContent = "--";
      els.rate.textContent = "--";
      els.mean.textContent = "--";
      els.rms.textContent = "--";
      els.peak.textContent = "--";
      els.age.textContent = "--";
      els.decode.textContent = "--";
      els.behind.textContent = "--";
      renderTimingTable([], failures);
      els.payload.textContent = JSON.stringify({
        ok: false,
        error: "read_failed",
        message,
        failures,
      }, null, 2);
      drawWaveform([], []);
    }

    async function fetchReading() {
      if (inFlight) return;
      inFlight = true;
      setStatus("", "Reading");
      try {
        if (allSensorsSelected()) {
          const results = await readAllSensorResults();
          const readings = results.filter((item) => item.ok).map((item) => item.data);
          const failures = results.filter((item) => !item.ok);
          latestPayload = {
            ok: readings.length > 0,
            status: !readings.length ? "error" : (failures.length ? "partial" : "ok"),
            mode: "all",
            readings,
            failures,
          };
          updateMultiReadout(readings, failures);
          setStatus(readings.length && !failures.length ? "ok" : "bad", `${readings.length}/${sensorOptions.length} boards`);
        } else {
          const data = await readSensor(selectedSensorConfig());
          latestPayload = data;
          updateReadout(data);
          setStatus("ok", "Live");
        }
      } catch (err) {
        const message = String(err.message || err);
        setStatus("bad", "Read failed");
        clearReadout(message, [{ sensor_id: allSensorsSelected() ? "all_boards" : ((selectedSensorConfig() || {}).sensor_id || "sensor"), error: message }]);
      } finally {
        inFlight = false;
      }
    }

    async function fetchAgentMonitor(force = false) {
      // Yield the NPU to an in-flight MCP chat: skip the check (even a forced one) while the
      // user's chat request is running so it isn't starved on the single serialized NPU.
      if (chatBusy) return;
      if ((!agentRunning && !force) || agentInFlight) return;
      agentInFlight = true;
      els.agentToggle.classList.remove("secondary");
      els.agentToggle.title = "";
      els.agentToggle.textContent = "Agent: Checking…";
      try {
        const params = new URLSearchParams({
          axis: els.axis.value,
          duration_s: els.duration.value,
          require_current: "1",
          require_llm: "0",
          use_llm: "1",
          model_timeout_s: "15",
          vote_k: els.voteK.value,
          skip_cache: "1",
          process_reader: "1",
        });
        const res = await fetch(`/api/agent-monitor?${params}`, { cache: "no-store" });
        const data = await res.json();
        if (data && data.paused_for_chat) {
          // A chat is using the NPU; skip this check without touching the panel or alert.
          els.agentToggle.textContent = agentRunning ? "Agent: On · Paused" : "Agent: Off";
          return;
        }
        if (!res.ok || !data.ok) throw new Error(data.message || data.error || "agent monitor failed");
        updateAgentMonitor(data);
      } catch (err) {
        els.agentToggle.textContent = "Agent: Retry";
        els.agentToggle.title = String(err.message || err);
        els.agentToggle.classList.add("secondary");
        if (agentRunning) window.setTimeout(() => fetchAgentMonitor(true), 5000);
      } finally {
        agentInFlight = false;
      }
    }

    function updateAgentMonitor(data) {
      const popup = data.popup || {};
      els.agentToggle.classList.remove("secondary");
      const status = labelize(popup.network_label || "checked");
      const cachedAge = Number((data.diagnostics || {}).cached_age_s);
      const checked = Number.isFinite(cachedAge)
        ? (cachedAge < 1 ? "now" : `${Math.round(cachedAge)}s ago`)
        : "now";
      const alert = data.anomaly && popup.show_popup;
      els.agentToggle.title = `Live agent enabled · ${status} · checked ${checked}. Click to turn it off.`;
      els.agentToggle.textContent = `Agent: On · ${alert ? "Alert" : status} · ${checked}`;
      els.agentToggle.setAttribute("aria-pressed", "true");
      if (data.anomaly && popup.show_popup) {
        showAgentAlert(popup);
      } else if (!data.anomaly) {
        hideAgentAlert();
      }
      if ((data.anomaly_registration || {}).registered) {
        refreshAnomalyLog();
      }
    }

    function showAgentAlert(popup) {
      const now = Date.now();
      const key = popup.dedupe_key || popup.network_label || popup.message || "agent-alert";
      const shouldNotify = !(key === lastAgentAlertKey && now - lastAgentAlertAt < AGENT_ALERT_COOLDOWN_MS);

      els.agentAlert.className = `agent-alert ${popup.severity || "warning"}`;
      els.agentAlertTitle.textContent = popup.title || "Building Vibration Anomaly";
      els.agentAlertBody.textContent = popup.message || "The building vibration agent detected an anomaly.";
      const affected = Array.isArray(popup.affected_sensor_ids) && popup.affected_sensor_ids.length
        ? popup.affected_sensor_ids.join(", ")
        : "none";
      const confidence = popup.confidence != null && Number.isFinite(Number(popup.confidence))
        ? `${fmt(Number(popup.confidence) * 100, 1)}%`
        : "--";
      const isNum = (value) => value != null && Number.isFinite(Number(value));
      const uncertain = popup.label_uncertain
        ? ` · Label uncertain (${isNum(popup.vote_agreement) ? fmt(Number(popup.vote_agreement) * 100, 0) + "% re-check agreement" : "re-checks disagree"})`
        : "";
      const history = isNum(popup.history_x_median)
        ? ` · vs normal: ${fmt(Number(popup.history_x_median), 1)}× median (p${fmt(Number(popup.history_percentile), 1)})`
        : (popup.history_learning ? ` · history: learning (${popup.history_n || 0} windows)` : "");
      els.agentAlertMeta.textContent = `Status: ${popup.descriptor_text || labelize(popup.network_label)} · Affected: ${affected}${history} · Confidence: ${confidence}${uncertain}`;

      if (shouldNotify) {
        lastAgentAlertKey = key;
        lastAgentAlertAt = now;
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification(els.agentAlertTitle.textContent, { body: els.agentAlertBody.textContent });
        }
      }
    }

    function hideAgentAlert() {
      els.agentAlert.classList.add("hidden");
    }

    // The registry appends one record per agent poll, so a persisting anomaly repeats every
    // few seconds — and its affected-sensor list jitters between polls. Collapse consecutive
    // records with the same label and severity into one episode; the affected set is the
    // union across the run.
    function collapseAnomalyWindows(windows) {
      const groups = [];
      for (const record of windows) { // newest first from the API
        const key = `${record.network_label || ""}|${record.severity || ""}`;
        const last = groups[groups.length - 1];
        if (last && last.key === key) {
          last.count += 1;
          last.oldest = record;
          for (const id of record.affected_sensor_ids || []) last.affected.add(id);
        } else {
          groups.push({ key, count: 1, latest: record, oldest: record, affected: new Set(record.affected_sensor_ids || []) });
        }
      }
      return groups;
    }

    async function refreshAnomalyLog() {
      try {
        const res = await fetch("/api/anomaly-windows?limit=1000", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.message || data.error || "anomaly list failed");
        renderAnomalyLog(data);
      } catch (err) {
        els.anomalyLogInfo.textContent = "Load failed";
        els.anomalyLogList.replaceChildren(anomalyLogNote(`Could not load saved anomalies: ${String(err.message || err)}`));
      }
    }

    function anomalyLogNote(text) {
      const note = document.createElement("div");
      note.className = "anomaly-log-empty";
      note.textContent = text;
      return note;
    }

    function renderAnomalyLog(data) {
      const windows = Array.isArray(data.windows) ? data.windows : [];
      const groups = collapseAnomalyWindows(windows);
      els.anomalyLogBtn.textContent = groups.length ? `Anomalies (${groups.length})` : "Anomalies";
      els.anomalyLogInfo.textContent = windows.length
        ? `${groups.length} episodes from ${windows.length} saved windows · ${data.jsonl_path || ""}`
        : `No saved anomaly windows yet · ${data.jsonl_path || ""}`;
      if (!windows.length) {
        els.anomalyLogList.replaceChildren(anomalyLogNote("No anomaly popups have been saved yet. Each popup the agent raises is appended here automatically."));
        return;
      }
      els.anomalyLogList.replaceChildren(...groups.map(renderAnomalyItem));
    }

    function renderAnomalyItem(group) {
      const record = group.latest;
      const item = document.createElement("article");
      item.className = `anomaly-item ${record.severity || "warning"}`;
      const head = document.createElement("div");
      head.className = "anomaly-item-head";
      const title = document.createElement("strong");
      title.textContent = (record.descriptor_text || labelize(record.network_label)) + (group.count > 1 ? ` (×${group.count})` : "");
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = fmtAnomalyTime(record.created_at_utc);
      head.append(title, when);
      const meta = document.createElement("div");
      meta.className = "meta";
      const affected = group.affected.size ? [...group.affected].sort().join(", ") : "none";
      const confidence = record.confidence != null && Number.isFinite(Number(record.confidence)) ? `${fmt(Number(record.confidence) * 100, 1)}%` : "--";
      const history = record.history_x_median != null && Number.isFinite(Number(record.history_x_median))
        ? ` · vs normal: ${fmt(Number(record.history_x_median), 1)}× (p${fmt(Number(record.history_percentile), 1)})`
        : "";
      const span = group.count > 1 ? ` · first ${fmtAnomalyTime(group.oldest.created_at_utc)}` : "";
      meta.textContent = `Severity: ${record.severity || "--"} · Affected: ${affected}${history} · Confidence: ${confidence}${span}`;
      const summary = document.createElement("p");
      summary.className = "summary";
      summary.textContent = record.summary || "No summary recorded.";
      item.append(head, meta, summary);
      item.addEventListener("click", () => item.classList.toggle("open"));
      return item;
    }

    function fmtAnomalyTime(value) {
      const date = new Date(value || "");
      return Number.isNaN(date.getTime()) ? String(value || "--") : date.toLocaleString();
    }

    function toggleAnomalyLog(force) {
      const show = typeof force === "boolean" ? force : els.anomalyLog.classList.contains("hidden");
      els.anomalyLog.classList.toggle("hidden", !show);
      if (show) refreshAnomalyLog();
    }

    async function readSensor(sensor) {
      const result = await readSensorResult(sensor);
      if (!result.ok) {
        const sensorId = sensor && sensor.sensor_id ? `${sensor.sensor_id}: ` : "";
        throw new Error(sensorId + result.error);
      }
      return result.data;
    }

    async function readSensorResult(sensor) {
      try {
        const res = await fetch(graphUrl(sensor), { cache: "no-store" });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          throw new Error(data.message || data.error || "read failed");
        }
        data.sensor_config = sensor || null;
        return { ok: true, sensor_id: sensor && sensor.sensor_id, data };
      } catch (err) {
        return {
          ok: false,
          sensor_id: sensor && sensor.sensor_id ? sensor.sensor_id : "sensor",
          location: sensor && sensor.location,
          role: sensor && sensor.role,
          error: String(err.message || err),
        };
      }
    }

    function updateReadout(data) {
      const stats = data.stats || {};
      const metadata = data.metadata || {};
      const selected = selectedSensorConfig();
      const sensorId = data.machine_id || metadata.machine_id || (selected && selected.sensor_id) || "sensor";
      const samples = data.samples_g || [];
      const mean = Number(stats.mean_g) || 0;
      const plotted = els.center.checked ? samples.map((value) => value - mean) : samples.slice();
      const peakToPeak = Number(stats.max_g) - Number(stats.min_g);
      const age = metadata.data_file_age_s;
      const timing = diagnosticsFor(data);

      els.chartTitle.textContent = `${sensorId} ${String(data.axis || "norm").toUpperCase()} acceleration ${els.center.checked ? "(centered)" : "(raw)"}`;
      els.sourceChip.textContent = `${selected && selected.role ? selected.role : "sensor"} · ${metadata.data_is_current ? "Current file" : "Stale or missing"}`;
      els.liveChip.textContent = `${fmt(data.window_duration_s, 2)} s`;
      els.samples.textContent = String(data.sample_count || "--");
      els.rate.textContent = data.sampling_rate_hz ? `${data.sampling_rate_hz} Hz` : "--";
      els.mean.textContent = `${fmt(stats.mean_g, 4)} g`;
      els.rms.textContent = `${fmt(stats.rms_g, 4)} g`;
      els.peak.textContent = `${fmt(peakToPeak, 4)} g`;
      els.age.textContent = age === null || age === undefined ? "--" : `${fmt(age, 2)} s`;
      els.decode.textContent = fmtMs(timing.decode_s);
      els.behind.textContent = fmtLag(timing.sdk_gap_s);
      renderTimingTable([data]);
      els.payload.textContent = JSON.stringify({
        sensor_id: sensorId,
        location: selected && selected.location,
        role: selected && selected.role,
        acquisition_folder: data.acquisition_folder,
        sensor_name: data.sensor_name,
        axis: data.axis,
        sample_count: data.sample_count,
        points_returned: data.points_returned,
        downsample_stride: data.downsample_stride,
        sampling_rate_hz: data.sampling_rate_hz,
        window_duration_s: data.window_duration_s,
        stats: data.stats,
        metadata: data.metadata,
        diagnostics: data.diagnostics,
      }, null, 2);
      drawWaveform(plotted, data.time_s || []);
    }

    function updateMultiReadout(readings, failures = []) {
      const count = readings.length;
      const sampleCounts = readings.map((item) => Number(item.sample_count) || 0);
      const rates = readings.map((item) => Number(item.sampling_rate_hz) || 0).filter(Boolean);
      const ages = readings
        .map((item) => item.metadata ? Number(item.metadata.data_file_age_s) : NaN)
        .filter((value) => Number.isFinite(value));
      const rmsValues = readings.map((item) => Number((item.stats || {}).rms_g) || 0);
      const maxPeakToPeak = Math.max(...readings.map((item) => {
        const stats = item.stats || {};
        return (Number(stats.max_g) || 0) - (Number(stats.min_g) || 0);
      }));
      const latestAge = ages.length ? Math.max(...ages) : null;
      const timings = readings.map(diagnosticsFor);
      const decodeValues = timings.map((item) => item.decode_s).filter((value) => value !== null);
      const sdkGapValues = timings.map((item) => item.sdk_gap_s).filter((value) => value !== null);
      const maxDecode = decodeValues.length ? Math.max(...decodeValues) : null;
      const maxSdkGap = sdkGapValues.length ? Math.max(...sdkGapValues) : null;

      els.chartTitle.textContent = `All ${count} IIS3DWB waveforms ${els.center.checked ? "(centered)" : "(raw)"}`;
      els.sourceChip.textContent = failures.length
        ? `${failures.length} read failed`
        : readings.every((item) => (item.metadata || {}).data_is_current)
          ? "All current"
          : "Some stale or missing";
      els.liveChip.textContent = `${count}/${count + failures.length} live`;
      els.samples.textContent = sampleCounts.length ? String(Math.min(...sampleCounts)) : "--";
      els.rate.textContent = rates.length ? `${Math.round(rates.reduce((sum, value) => sum + value, 0) / rates.length)} Hz avg` : "--";
      els.mean.textContent = `${count} sensors`;
      els.rms.textContent = rmsValues.length ? `${fmt(Math.max(...rmsValues), 4)} g max` : "--";
      els.peak.textContent = Number.isFinite(maxPeakToPeak) ? `${fmt(maxPeakToPeak, 4)} g max` : "--";
      els.age.textContent = latestAge === null ? "--" : `${fmt(latestAge, 2)} s max`;
      els.decode.textContent = maxDecode === null ? "--" : `${fmtMs(maxDecode)} max`;
      els.behind.textContent = maxSdkGap === null ? "--" : `${fmtLag(maxSdkGap)} max`;
      renderTimingTable(readings, failures);
      els.payload.textContent = JSON.stringify({
        mode: "all_sensors",
        axis: els.axis.value,
        read_count: count,
        failed_count: failures.length,
        failures,
        sensors: readings.map((item) => {
          const sensor = item.sensor_config || {};
          return {
            sensor_id: item.machine_id || sensor.sensor_id,
            location: sensor.location,
            role: sensor.role,
            acquisition_folder: item.acquisition_folder || sensor.acquisition_folder,
            sensor_name: item.sensor_name,
            sample_count: item.sample_count,
            points_returned: item.points_returned,
            sampling_rate_hz: item.sampling_rate_hz,
            window_duration_s: item.window_duration_s,
            stats: item.stats,
            metadata: item.metadata,
            diagnostics: item.diagnostics,
          };
        }),
      }, null, 2);
      drawMultiWaveforms(readings, failures);
    }

    function resizeCanvas() {
      const canvas = els.canvas;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width * ratio));
      const height = Math.max(300, Math.floor(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return { width, height, ratio };
    }

    function drawWaveform(samples, times) {
      const ctx = els.canvas.getContext("2d");
      const { width, height, ratio } = resizeCanvas();
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfa";
      ctx.fillRect(0, 0, width, height);

      const padLeft = 54 * ratio;
      const padRight = 20 * ratio;
      const padTop = 24 * ratio;
      const padBottom = 42 * ratio;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;

      ctx.strokeStyle = "#d7ddd7";
      ctx.lineWidth = 1 * ratio;
      for (let i = 0; i <= 5; i += 1) {
        const y = padTop + (plotHeight * i / 5);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(width - padRight, y);
        ctx.stroke();
      }
      for (let i = 0; i <= 6; i += 1) {
        const x = padLeft + (plotWidth * i / 6);
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, height - padBottom);
        ctx.stroke();
      }

      if (!samples.length) {
        ctx.fillStyle = "#66736e";
        ctx.font = `${14 * ratio}px system-ui, sans-serif`;
        ctx.fillText("Waiting for live samples", padLeft, padTop + 32 * ratio);
        return;
      }

      let min = Math.min(...samples);
      let max = Math.max(...samples);
      if (Math.abs(max - min) < 1e-9) {
        min -= 0.001;
        max += 0.001;
      }
      const padding = (max - min) * 0.12;
      min -= padding;
      max += padding;

      const firstTime = Number(times[0]) || 0;
      const lastTime = Number(times[times.length - 1]) || samples.length - 1;
      const spanTime = Math.max(1e-9, lastTime - firstTime);
      const xAt = (index) => {
        const t = Number(times[index]);
        const rel = Number.isFinite(t) ? (t - firstTime) / spanTime : index / Math.max(1, samples.length - 1);
        return padLeft + rel * plotWidth;
      };
      const yAt = (value) => padTop + (max - value) / (max - min) * plotHeight;

      if (min < 0 && max > 0) {
        const zeroY = yAt(0);
        ctx.strokeStyle = "#03234e";
        ctx.lineWidth = 1 * ratio;
        ctx.beginPath();
        ctx.moveTo(padLeft, zeroY);
        ctx.lineTo(width - padRight, zeroY);
        ctx.stroke();
      }

      ctx.strokeStyle = "#0469b1";
      ctx.lineWidth = 2 * ratio;
      ctx.beginPath();
      samples.forEach((value, index) => {
        const x = xAt(index);
        const y = yAt(Number(value) || 0);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#9aa39e";
      ctx.font = `${10 * ratio}px system-ui, sans-serif`;
      ctx.textAlign = "right";
      for (let i = 0; i <= 5; i += 1) {
        const gridY = padTop + (plotHeight * i / 5);
        ctx.fillText(`${fmt(max - (max - min) * i / 5, 4)}`, padLeft - 6 * ratio, gridY + 3 * ratio);
      }
      ctx.textAlign = "left";
      ctx.fillText("g", 10 * ratio, padTop - 8 * ratio);
      for (let i = 0; i <= 6; i += 1) {
        const tickX = padLeft + (plotWidth * i / 6);
        ctx.textAlign = i === 0 ? "left" : (i === 6 ? "right" : "center");
        ctx.fillText(fmtSpanMs(spanTime * i / 6), tickX, height - padBottom + 16 * ratio);
      }
      ctx.textAlign = "left";
      ctx.fillStyle = "#66736e";
      ctx.font = `${11 * ratio}px system-ui, sans-serif`;
      ctx.fillText(`${fmt(spanTime, 2)} s window`, padLeft, height - 12 * ratio);
    }

    const fmtSpanMs = (s) => {
      const v = Number(s);
      if (!Number.isFinite(v)) return "";
      return v < 1 ? `${Math.round(v * 1000)}ms` : `${fmt(v, 2)}s`;
    };

    function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
      const words = String(text).split(/\s+/);
      let line = "";
      let yy = y;
      for (const word of words) {
        const test = line ? line + " " + word : word;
        if (ctx.measureText(test).width > maxWidth && line) {
          ctx.fillText(line, x, yy);
          line = word;
          yy += lineHeight;
          if (yy > y + lineHeight * 4) { ctx.fillText("…", x, yy); return; }
        } else {
          line = test;
        }
      }
      if (line) ctx.fillText(line, x, yy);
    }

    function boardColorFor(id, index) {
      if (id) {
        const pos = sensorOptions.findIndex((sensor) => sensor.sensor_id === id);
        if (pos >= 0) return BOARD_COLORS[pos % BOARD_COLORS.length];
      }
      return BOARD_COLORS[(index || 0) % BOARD_COLORS.length];
    }

    function isBaselineReading(reading) {
      const cfg = (reading && reading.sensor_config) || {};
      const id = reading && (reading.machine_id || cfg.sensor_id);
      return cfg.role === "baseline" || (baselineSensorId && id === baselineSensorId);
    }

    function renderLegend(panels) {
      if (!els.legend) return;
      if (!panels || panels.length <= 1) { els.legend.replaceChildren(); return; }
      const frag = document.createDocumentFragment();
      panels.forEach((panel) => {
        const lg = document.createElement("span");
        lg.className = "lg" + (panel.kind === "failure" ? " muted" : "");
        const sw = document.createElement("span");
        sw.className = "sw";
        sw.style.background = panel.color;
        const text = document.createElement("span");
        text.textContent = panel.label + (panel.kind === "failure" ? " ⚠" : "");
        lg.append(sw, text);
        frag.appendChild(lg);
      });
      els.legend.replaceChildren(frag);
    }

    function drawMultiWaveforms(readings, failures = []) {
      const ctx = els.canvas.getContext("2d");
      const { width, height, ratio } = resizeCanvas();
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfa";
      ctx.fillRect(0, 0, width, height);

      const centered = els.center.checked;
      const readingList = readings || [];
      const baseline = readingList.find((reading) => isBaselineReading(reading));
      const baselineId = baseline ? (baseline.machine_id || (baseline.sensor_config || {}).sensor_id) : null;
      const baselineSamples = baseline
        ? (baseline.samples_g || []).map((value) => centered ? Number(value) - (Number((baseline.stats || {}).mean_g) || 0) : Number(value))
        : null;

      const panels = [
        ...readingList.map((reading) => {
          const mean = Number((reading.stats || {}).mean_g) || 0;
          const samples = (reading.samples_g || []).map((value) => centered ? Number(value) - mean : Number(value));
          const id = reading.machine_id || (reading.sensor_config || {}).sensor_id || "sensor";
          return {
            kind: "reading",
            id,
            label: id,
            color: boardColorFor(id),
            samples,
            times: reading.time_s || [],
            current: !!(reading.metadata || {}).data_is_current,
          };
        }),
        ...(failures || []).map((failure) => ({
          kind: "failure",
          id: failure.sensor_id || "sensor",
          label: failure.sensor_id || "sensor",
          color: "#d0021b",
          message: failure.error || "read failed",
        })),
      ];

      renderLegend(panels);

      let sharedRange = null;
      if (els.sharedScale.checked) {
        let lo = Infinity;
        let hi = -Infinity;
        panels.forEach((panel) => {
          if (panel.kind !== "reading") return;
          panel.samples.forEach((value) => { if (value < lo) lo = value; if (value > hi) hi = value; });
        });
        if (Number.isFinite(lo) && Number.isFinite(hi)) {
          if (Math.abs(hi - lo) < 1e-9) { lo -= 0.001; hi += 0.001; }
          const pad = (hi - lo) * 0.12;
          sharedRange = { min: lo - pad, max: hi + pad };
        }
      }

      const overlay = (els.overlayBaseline.checked && baselineSamples && baselineSamples.length) ? baselineSamples : null;
      const columns = width >= 900 * ratio ? 2 : 1;
      const rows = Math.max(1, Math.ceil(panels.length / columns));
      const gap = 14 * ratio;
      const outerPad = 16 * ratio;
      const cellWidth = (width - outerPad * 2 - gap * (columns - 1)) / columns;
      const cellHeight = (height - outerPad * 2 - gap * (rows - 1)) / rows;
      const axisU = (els.axis.value || "norm").toUpperCase();

      panels.forEach((panel, index) => {
        const col = index % columns;
        const row = Math.floor(index / columns);
        const x = outerPad + col * (cellWidth + gap);
        const y = outerPad + row * (cellHeight + gap);
        drawWaveformPanel(ctx, {
          x,
          y,
          width: cellWidth,
          height: cellHeight,
          samples: panel.samples || [],
          times: panel.times || [],
          title: `${panel.label} · ${axisU}`,
          chip: panel.kind === "failure" ? "failed" : (panel.current ? "current" : "stale"),
          message: panel.message,
          color: panel.color,
          range: sharedRange,
          overlay: (panel.kind === "reading" && panel.id !== baselineId) ? overlay : null,
          ratio,
        });
      });
    }

    function drawWaveformPanel(ctx, panel) {
      const { x, y, width, height, samples, times, title, chip, message, ratio } = panel;
      const color = panel.color || "#0469b1";
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x, y, width, height);
      ctx.strokeStyle = "#d7ddd7";
      ctx.lineWidth = 1 * ratio;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = "#1f2a26";
      ctx.font = `${12 * ratio}px system-ui, sans-serif`;
      ctx.textAlign = "left";
      ctx.fillText(title, x + 12 * ratio, y + 20 * ratio);
      ctx.fillStyle = chip === "current" ? "#0469b1" : (chip === "stale" ? "#9a7400" : "#d0021b");
      ctx.textAlign = "right";
      ctx.fillText(chip, x + width - 12 * ratio, y + 20 * ratio);
      ctx.textAlign = "left";

      const padLeft = 50 * ratio;
      const padRight = 14 * ratio;
      const padTop = 34 * ratio;
      const padBottom = 30 * ratio;
      const plotX = x + padLeft;
      const plotY = y + padTop;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;

      if (!samples.length) {
        ctx.strokeStyle = "#ecefec";
        ctx.strokeRect(plotX, plotY, plotWidth, plotHeight);
        ctx.fillStyle = "#66736e";
        ctx.font = `${11 * ratio}px system-ui, sans-serif`;
        wrapText(ctx, message || "waiting", plotX + 8 * ratio, plotY + 18 * ratio, plotWidth - 16 * ratio, 13 * ratio);
        return;
      }

      let min;
      let max;
      if (panel.range) {
        min = panel.range.min;
        max = panel.range.max;
      } else {
        min = Math.min(...samples);
        max = Math.max(...samples);
        if (Math.abs(max - min) < 1e-9) { min -= 0.001; max += 0.001; }
        const padding = (max - min) * 0.12;
        min -= padding;
        max += padding;
      }

      const firstTime = Number(times[0]) || 0;
      const lastTime = Number(times[times.length - 1]) || samples.length - 1;
      const spanTime = Math.max(1e-9, lastTime - firstTime);
      const xAt = (index) => {
        const t = Number(times[index]);
        const rel = Number.isFinite(t) ? (t - firstTime) / spanTime : index / Math.max(1, samples.length - 1);
        return plotX + rel * plotWidth;
      };
      const yAt = (value) => plotY + (max - value) / (max - min) * plotHeight;

      ctx.strokeStyle = "#e7ebe7";
      ctx.fillStyle = "#9aa39e";
      ctx.font = `${9 * ratio}px system-ui, sans-serif`;
      ctx.lineWidth = 1 * ratio;
      ctx.textAlign = "right";
      for (let i = 0; i <= 2; i += 1) {
        const gridY = plotY + (plotHeight * i / 2);
        ctx.beginPath();
        ctx.moveTo(plotX, gridY);
        ctx.lineTo(plotX + plotWidth, gridY);
        ctx.stroke();
        ctx.fillText(fmt(max - (max - min) * i / 2, 3), plotX - 4 * ratio, gridY + 3 * ratio);
      }
      ctx.textAlign = "left";

      if (min < 0 && max > 0) {
        const zeroY = yAt(0);
        ctx.strokeStyle = "#cdb6a6";
        ctx.beginPath();
        ctx.moveTo(plotX, zeroY);
        ctx.lineTo(plotX + plotWidth, zeroY);
        ctx.stroke();
      }

      if (panel.overlay && panel.overlay.length) {
        const overlay = panel.overlay;
        ctx.strokeStyle = "rgba(102, 115, 110, 0.4)";
        ctx.lineWidth = 1 * ratio;
        ctx.beginPath();
        overlay.forEach((value, index) => {
          const lineX = plotX + (index / Math.max(1, overlay.length - 1)) * plotWidth;
          const lineY = yAt(Number(value) || 0);
          if (index === 0) ctx.moveTo(lineX, lineY);
          else ctx.lineTo(lineX, lineY);
        });
        ctx.stroke();
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6 * ratio;
      ctx.beginPath();
      samples.forEach((value, index) => {
        const lineX = xAt(index);
        const lineY = yAt(Number(value) || 0);
        if (index === 0) ctx.moveTo(lineX, lineY);
        else ctx.lineTo(lineX, lineY);
      });
      ctx.stroke();

      ctx.fillStyle = "#9aa39e";
      ctx.font = `${9 * ratio}px system-ui, sans-serif`;
      ctx.fillText("g", x + 8 * ratio, plotY + plotHeight / 2);
      ctx.textAlign = "left";
      ctx.fillText("0", plotX, y + height - 8 * ratio);
      ctx.textAlign = "center";
      ctx.fillText(fmtSpanMs(spanTime / 2), plotX + plotWidth / 2, y + height - 8 * ratio);
      ctx.textAlign = "right";
      ctx.fillText(fmtSpanMs(spanTime), plotX + plotWidth, y + height - 8 * ratio);
      ctx.textAlign = "left";
    }

    function restartLoop() {
      if (timer) clearInterval(timer);
      if (running) {
        timer = setInterval(fetchReading, Math.max(250, Number(els.refresh.value) || 1000));
      }
    }

    function restartAgentLoop() {
      if (agentTimer) clearInterval(agentTimer);
      if (agentRunning) {
        agentTimer = setInterval(fetchAgentMonitor, Math.max(5000, Number(els.agentInterval.value) || 30000));
      }
    }

    els.agentToggle.addEventListener("click", () => {
      agentRunning = !agentRunning;
      els.agentToggle.textContent = agentRunning ? "Agent: On" : "Agent: Off";
      els.agentToggle.title = agentRunning
        ? "Live agent is enabled; click to turn it off."
        : "Live agent is disabled; click to turn it on.";
      els.agentToggle.setAttribute("aria-pressed", String(agentRunning));
      els.agentToggle.classList.toggle("secondary", !agentRunning);
      if (agentRunning) fetchAgentMonitor(true);
      else hideAgentAlert();
      restartAgentLoop();
    });
    els.toggle.addEventListener("click", () => {
      running = !running;
      els.toggle.textContent = running ? "Pause" : "Start";
      if (running) fetchReading();
      restartLoop();
    });
    els.refreshNow.addEventListener("click", fetchReading);
    els.sensorSelect.addEventListener("change", () => {
      fetchReading();
      restartLoop();
    });
    [els.axis, els.duration, els.refresh, els.points].forEach((control) => {
      control.addEventListener("change", () => {
        fetchReading();
        restartLoop();
      });
    });
    [els.axis, els.duration, els.agentInterval, els.voteK].forEach((control) => {
      control.addEventListener("change", () => {
        fetchAgentMonitor(true);
        restartAgentLoop();
      });
    });
    els.agentDismiss.addEventListener("click", hideAgentAlert);
    els.anomalyLogBtn.addEventListener("click", () => toggleAnomalyLog());
    els.anomalyLogRefresh.addEventListener("click", refreshAnomalyLog);
    els.anomalyLogClose.addEventListener("click", () => toggleAnomalyLog(false));
    function redrawLatest() {
      if (latestPayload && latestPayload.mode === "all") updateMultiReadout(latestPayload.readings || [], latestPayload.failures || []);
      else if (latestPayload) updateReadout(latestPayload);
    }
    [els.center, els.sharedScale, els.overlayBaseline].forEach((control) => {
      control.addEventListener("change", redrawLatest);
    });
    window.addEventListener("resize", () => {
      if (latestPayload && latestPayload.mode === "all") updateMultiReadout(latestPayload.readings || [], latestPayload.failures || []);
      else if (latestPayload) updateReadout(latestPayload);
      else drawWaveform([], []);
    });

    drawWaveform([], []);
    refreshAnomalyLog();
    loadSensors().finally(() => {
      fetchReading();
      fetchAgentMonitor(true);
      restartLoop();
      restartAgentLoop();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
