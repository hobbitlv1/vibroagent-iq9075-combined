#!/usr/bin/env bash
# Direct VibroAgent launcher.
#
# This script captures the manual launch method used from this workspace:
#   1) Qualcomm Genie/OpenAI-compatible NPU model server on 127.0.0.1:8910
#   2) VibroAgent webchat on 127.0.0.1:7860, pointed at that model server
#   3) Native-callback ST HSDatalog logger writing stdatalog_examples/live_*
#
# Usage:
#   ./vibroagent_direct.sh [start|stop|restart|status|logs|start-webchat|stop-webchat|restart-webchat|start-logger|stop-logger|restart-logger]
#
# Logs and pid files are stored in:
#   vibrodiag_mcp_prototype/.run/
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$ROOT/vibrodiag_mcp_prototype"
EXAMPLES="$ROOT/stdatalog_examples"
PYTHON="$ROOT/vibroagent-venv/bin/python"
RUN_DIR="$PROTO/.run"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
NPU_HOST="${NPU_HOST:-127.0.0.1}"
NPU_PORT="${NPU_PORT:-8910}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-7860}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-$ROOT/v2.46.0.260424/qairt/2.46.0.260424}"
GENIE_CONFIG="${GENIE_CONFIG:-$ROOT/models/merged_4k_6000s/merged_6000_ctx4096_calib1-genie-w4a16-qualcomm_qcs9075/genie_config.json}"
# Server-side per-generation timeout for the NPU model server. Keep it aligned with the
# agent client budget (AGENT_MODEL_TIMEOUT_S) so a stuck/runaway generation is reclaimed
# (the server restarts the runner) instead of being orphaned on the single serial NPU,
# where it would block every following check. With greedy decoding a valid check finishes
# in seconds; this is only the ceiling for a pathological generation.
GENIE_TIMEOUT_S="${GENIE_TIMEOUT_S:-300}"
GENIE_READY_WAIT_S="${GENIE_READY_WAIT_S:-360}"
GENIE_ADAPTER_MAX_PROMPT_CHARS="${GENIE_ADAPTER_MAX_PROMPT_CHARS:-12000}"
GENIE_MAX_OUTPUT_TOKENS="${GENIE_MAX_OUTPUT_TOKENS:-1024}"
GENIE_PERSISTENT_RUNNER="${GENIE_PERSISTENT_RUNNER:-$PROTO/.build/genie_persistent_runner}"
# --- Model backend selection ---
# geniex = Qualcomm GenieX OpenAI server (llama.cpp + GGML Hexagon backend,
#          GGUF models, runtime-configurable context, grammar-constrained
#          json_schema). Default since 2026-07-02 (validated end-to-end on the
#          NPU). Install per https://geniex.aihub.qualcomm.com, then
#          `geniex-py pull "$GENIEX_MODEL"` once before first start.
# genie  = legacy QAIRT/Genie adapter on :8910 (qairt bundle: finetuned model,
#          NPU, ctx/kv baked at 4096, ignores per-request sampling params).
MODEL_BACKEND="${MODEL_BACKEND:-geniex}"
# The pip distribution ships the Python SDK (geniex-py) without `geniex serve`;
# the stack serves it through vibroagent_mcp.geniex_openai_server using the
# GenieX virtualenv's Python.
GENIEX_PYTHON="${GENIEX_PYTHON:-$HOME/geniex-venv/bin/python}"
GENIEX_HOST="${GENIEX_HOST:-127.0.0.1}"
GENIEX_PORT="${GENIEX_PORT:-18181}"
# Q4_0 is the one GGUF precision that lands on the Hexagon NPU; other quants
# fall back to GPU/CPU (per GenieX model docs).
GENIEX_MODEL="${GENIEX_MODEL:-unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0}"
GENIEX_DEVICE_MAP="${GENIEX_DEVICE_MAP:-npu}"
# 6144 is the measured HTP0 ceiling: 8192 fails KV-cache fastrpc_mmap (~1 GiB).
GENIEX_N_CTX="${GENIEX_N_CTX:-6144}"
GENIEX_MAX_OUTPUT_TOKENS="${GENIEX_MAX_OUTPUT_TOKENS:-1024}"
GENIEX_TIMEOUT_S="${GENIEX_TIMEOUT_S:-$GENIE_TIMEOUT_S}"
GENIEX_READY_WAIT_S="${GENIEX_READY_WAIT_S:-240}"

QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-$GENIE_TIMEOUT_S}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
QWEN_MAX_PROMPT_CHARS="${QWEN_MAX_PROMPT_CHARS:-9000}"
QWEN_MODEL_TOOL_ROUTING="${QWEN_MODEL_TOOL_ROUTING:-0}"
LOGGER_SENSOR="${LOGGER_SENSOR:-iis3dwb_acc}"
LOGGER_STATS_S="${LOGGER_STATS_S:-5}"
LOGGER_STALE_TIMEOUT_S="${LOGGER_STALE_TIMEOUT_S:-2}"
# Rollover rewrites active .dat files and can break readers that tail the stream.
# Keep it opt-in. Set LOGGER_MAX_DAT_MB to a positive value to enable it.
LOGGER_MAX_DAT_MB="${LOGGER_MAX_DAT_MB:-0}"
LOGGER_ROLLOVER_KEEP_MB="${LOGGER_ROLLOVER_KEEP_MB:-16}"
LOGGER_ROLLOVER_ALIGN_BYTES="${LOGGER_ROLLOVER_ALIGN_BYTES:-0}"
VIBRO_BOARD_READER_PROCESSES="${VIBRO_BOARD_READER_PROCESSES:-1}"
VIBRO_BOARD_READER_TIMEOUT_S="${VIBRO_BOARD_READER_TIMEOUT_S:-12}"
# Building-agent model limits. The local Qwen/Genie adapter emits text-style tool
# calls and the main fusion response can exceed the old 384-token Genie cap
# before its finalization block, so keep the cap high enough for reliable agent engagement.
AGENT_MODEL_MAX_TOKENS="${AGENT_MODEL_MAX_TOKENS:-1024}"
AGENT_MODEL_TIMEOUT_S="${AGENT_MODEL_TIMEOUT_S:-300}"
# Agent-check monitor limits. These apply only to the popup triage path, which must
# fail fast instead of wedging the single serial NPU behind a long generation.
AGENT_MONITOR_MODEL_TIMEOUT_S="${AGENT_MONITOR_MODEL_TIMEOUT_S:-5}"
AGENT_MONITOR_MAX_TOKENS="${AGENT_MONITOR_MAX_TOKENS:-48}"
AGENT_MONITOR_MAX_PROMPT_CHARS="${AGENT_MONITOR_MAX_PROMPT_CHARS:-1100}"
AGENT_MONITOR_MAX_PROMPT_TOKENS="${AGENT_MONITOR_MAX_PROMPT_TOKENS:-320}"
AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-2}"
AGENT_MAX_AXIS_CHECKS="${AGENT_MAX_AXIS_CHECKS:-0}"
AGENT_MAX_RECHECKS="${AGENT_MAX_RECHECKS:-0}"
# The single NPU serializes every request, so run autonomous per-sensor agents one
# at a time. Flooding it concurrently only makes queued agents time out and fall
# back to the deterministic rule. Raise this only for a batching remote endpoint.
AGENT_MAX_CONCURRENCY="${AGENT_MAX_CONCURRENCY:-1}"

SDK_PYTHONPATH="$PROTO/src:$ROOT/stdatalog_core:$ROOT/stdatalog_pnpl:$ROOT/stdatalog_dtk:$ROOT/stdatalog_gui"

# The webchat talks to whichever backend is active.
if [ "$MODEL_BACKEND" = "geniex" ]; then
  MODEL_API_HOST="$GENIEX_HOST"
  MODEL_API_PORT="$GENIEX_PORT"
  MODEL_API_MODEL="$GENIEX_MODEL"
else
  MODEL_API_HOST="$NPU_HOST"
  MODEL_API_PORT="$NPU_PORT"
  MODEL_API_MODEL="$MODEL_ID"
fi

MODEL_PID_FILE="$RUN_DIR/model_server_npu.pid"
WEBCHAT_PID_FILE="$RUN_DIR/webchat.pid"
LOGGER_PID_FILE="$RUN_DIR/logger.pid"
MODEL_LOG="$RUN_DIR/model_server_npu.log"
WEBCHAT_LOG="$RUN_DIR/webchat.log"
LOGGER_LOG="$RUN_DIR/logger.log"
PERSISTENT_RUNNER_LOG="${GENIE_PERSISTENT_RUNNER}.log"

mkdir -p "$RUN_DIR"

green=$'\e[32m'
yellow=$'\e[33m'
red=$'\e[31m'
off=$'\e[0m'

say() { printf '%s\n' "$*"; }
ok() { printf '%s%s%s\n' "$green" "$*" "$off"; }
warn() { printf '%s%s%s\n' "$yellow" "$*" "$off"; }
err() { printf '%s%s%s\n' "$red" "$*" "$off"; }

pid_alive() {
  local pid="${1:-}"
  [ -n "$pid" ] && ps -o pid= -p "$pid" >/dev/null 2>&1
}

pid_from_file() {
  local file="$1"
  [ -f "$file" ] && tr -d '[:space:]' < "$file" || true
}

logger_rollover_enabled() {
  awk -v value="${LOGGER_MAX_DAT_MB:-0}" 'BEGIN {exit (value + 0 > 0) ? 0 : 1}'
}

logger_rollover_args() {
  logger_rollover_enabled || return 0
  printf " --max-dat-file-mb '%s' --rollover-keep-mb '%s' --rollover-align-bytes '%s'"     "$LOGGER_MAX_DAT_MB" "$LOGGER_ROLLOVER_KEEP_MB" "$LOGGER_ROLLOVER_ALIGN_BYTES"
}

port_pid() {
  ss -ltnp 2>/dev/null | awk -v port=":$1" '$4 ~ port"$" {print $0}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1
}

port_listening() {
  ss -ltn 2>/dev/null | awk -v port=":$1" '$4 ~ port"$" {found=1} END {exit found ? 0 : 1}'
}

wait_for_port() {
  local port="$1" timeout_s="${2:-30}" i=0
  while [ "$i" -lt "$timeout_s" ]; do
    port_listening "$port" && return 0
    sleep 1
    i=$((i + 1))
  done
  return 1
}

model_start_failed() {
  grep -qiE "Traceback|RuntimeError|SystemExit|GenieDialog_create failed|GenieDialogConfig_createFromJson|persistent Genie runner startup|Failed to open input file|Can't access model file|NSPModel" "$MODEL_LOG" 2>/dev/null
}

print_model_start_failure() {
  err "NPU model server exited before binding port $NPU_PORT. See $MODEL_LOG"
  tail -80 "$MODEL_LOG" 2>/dev/null | sed 's/^/  /'
  if [ -f "$PERSISTENT_RUNNER_LOG" ]; then
    warn "Persistent Genie runner log tail: $PERSISTENT_RUNNER_LOG"
    tail -60 "$PERSISTENT_RUNNER_LOG" 2>/dev/null | sed 's/^/  /'
  fi
}

wait_for_model_server() {
  local timeout_s="${1:-30}" i=0 pid
  while [ "$i" -lt "$timeout_s" ]; do
    port_listening "$NPU_PORT" && return 0
    pid="$(pid_from_file "$MODEL_PID_FILE")"
    if [ -n "$pid" ] && ! pid_alive "$pid"; then
      return 2
    fi
    model_start_failed && return 2
    sleep 1
    i=$((i + 1))
  done
  return 1
}

print_genie_bundle_recovery() {
  err "Configured Genie model bundle is incomplete: $GENIE_CONFIG"
  say "Restore the missing artifacts under: $(dirname "$GENIE_CONFIG")"
  say "or start with a complete bundle, for example:"
  say "  sudo env GENIE_CONFIG=/path/to/complete/genie_config.json bash $0 start"
  say "When using sudo, put GENIE_CONFIG after 'sudo env' so root receives the override."
}

validate_genie_bundle() {
  "$PYTHON" - "$GENIE_CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path

config = Path(sys.argv[1]).expanduser().resolve()
try:
    payload = json.loads(config.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"Invalid Genie config {config}: {exc}")
    raise SystemExit(1)

dialog = payload.get("dialog") if isinstance(payload, dict) else None
if not isinstance(dialog, dict):
    print(f"Invalid Genie config {config}: missing dialog object")
    raise SystemExit(1)

base = config.parent
required: list[tuple[str, str]] = []

def add(label: str, value: object) -> None:
    if isinstance(value, str) and value.strip():
        required.append((label, value.strip()))

tokenizer = dialog.get("tokenizer")
if isinstance(tokenizer, dict):
    add("dialog.tokenizer.path", tokenizer.get("path"))

engine = dialog.get("engine")
if isinstance(engine, dict):
    backend = engine.get("backend")
    if isinstance(backend, dict):
        add("dialog.engine.backend.extensions", backend.get("extensions"))
    model = engine.get("model")
    if isinstance(model, dict):
        binary = model.get("binary")
        if isinstance(binary, dict):
            ctx_bins = binary.get("ctx-bins")
            if isinstance(ctx_bins, list):
                for index, item in enumerate(ctx_bins):
                    add(f"dialog.engine.model.binary.ctx-bins[{index}]", item)
            else:
                add("dialog.engine.model.binary.ctx-bins", ctx_bins)

missing = 0
for label, reference in required:
    candidate = Path(reference)
    resolved = candidate if candidate.is_absolute() else base / candidate
    if not resolved.exists():
        detail = ""
        if resolved.is_symlink():
            detail = f" (broken symlink -> {os.readlink(resolved)})"
        print(f"Missing required Genie model artifact: {resolved}{detail} [{label} in {config}]")
        missing += 1

raise SystemExit(1 if missing else 0)
PY
}

ensure_paths() {
  local missing=0
  for path in "$PYTHON" "$PROTO/src" "$ROOT/stdatalog_core" "$ROOT/stdatalog_pnpl" "$ROOT/stdatalog_dtk" "$ROOT/stdatalog_gui" "$EXAMPLES/vibroagent_two_vibrometer_logger.py"; do
    if [ ! -e "$path" ]; then
      err "Missing required path: $path"
      missing=1
    fi
  done
  if [ "$MODEL_BACKEND" = "geniex" ]; then
    # GenieX needs only its venv; the QAIRT bundle checks below are genie-only.
    if [ ! -x "$GENIEX_PYTHON" ]; then
      err "MODEL_BACKEND=geniex but $GENIEX_PYTHON is missing. Install GenieX first (see https://geniex.aihub.qualcomm.com) and pull $GENIEX_MODEL."
      missing=1
    fi
    [ "$missing" -eq 0 ]
    return
  fi
  for path in "$QAIRT_SDK_ROOT" "$GENIE_CONFIG"; do
    if [ ! -e "$path" ]; then
      err "Missing required NPU path: $path"
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    if ! validate_genie_bundle; then
      print_genie_bundle_recovery
      missing=1
    fi
  fi
  [ "$missing" -eq 0 ]
}

start_model_server() {
  local pid
  pid="$(pid_from_file "$MODEL_PID_FILE")"
  if pid_alive "$pid"; then
    warn "NPU model server already running: pid $pid"
    return 0
  fi
  if port_listening "$NPU_PORT"; then
    warn "Port $NPU_PORT is already in use by pid $(port_pid "$NPU_PORT")"
    return 0
  fi
  rm -f "$MODEL_PID_FILE"

  if [ "$MODEL_BACKEND" = "geniex" ]; then
    if [ ! -x "$GENIEX_PYTHON" ]; then
      err "MODEL_BACKEND=geniex but $GENIEX_PYTHON is missing. Install GenieX first (see https://geniex.aihub.qualcomm.com) and pull $GENIEX_MODEL."
      return 1
    fi
    say "Starting GenieX model server (llama.cpp/Hexagon, n_ctx=$GENIEX_N_CTX) on http://$GENIEX_HOST:$GENIEX_PORT/v1 ..."
    : > "$MODEL_LOG"
    setsid -f bash -c "
      echo \$\$ > '$MODEL_PID_FILE'
      exec env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH='$PROTO/src' \
        GENIEX_MODEL='$GENIEX_MODEL' \
        GENIEX_DEVICE_MAP='$GENIEX_DEVICE_MAP' \
        GENIEX_N_CTX='$GENIEX_N_CTX' \
        GENIEX_MAX_OUTPUT_TOKENS='$GENIEX_MAX_OUTPUT_TOKENS' \
        GENIEX_TIMEOUT_S='$GENIEX_TIMEOUT_S' \
        '$GENIEX_PYTHON' -u -m vibroagent_mcp.geniex_openai_server \
          --host '$GENIEX_HOST' \
          --port '$GENIEX_PORT' \
        >> '$MODEL_LOG' 2>&1
    "
    local waited=0
    while [ "$waited" -lt "$GENIEX_READY_WAIT_S" ]; do
      if curl -sf -m 3 "http://$GENIEX_HOST:$GENIEX_PORT/v1/models" > /dev/null 2>&1; then
        ok "GenieX model server is up: pid $(pid_from_file "$MODEL_PID_FILE")"
        return 0
      fi
      sleep 2
      waited=$((waited + 2))
    done
    err "GenieX server did not become ready on port $GENIEX_PORT within ${GENIEX_READY_WAIT_S}s. See $MODEL_LOG"
    return 1
  fi

  say "Starting NPU model server on http://$NPU_HOST:$NPU_PORT/v1 ..."
  : > "$MODEL_LOG"
  setsid -f bash -c "
    echo \$\$ > '$MODEL_PID_FILE'
    exec env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH='$SDK_PYTHONPATH' \
      QAIRT_SDK_ROOT='$QAIRT_SDK_ROOT' \
      GENIE_CONFIG='$GENIE_CONFIG' \
      GENIE_MODEL='$MODEL_ID' \
      GENIE_PROMPT_FORMAT='qwen3' \
      GENIE_PERSISTENT='1' \
      GENIE_TIMEOUT_S='$GENIE_TIMEOUT_S' \
      GENIE_ADAPTER_MAX_PROMPT_CHARS='$GENIE_ADAPTER_MAX_PROMPT_CHARS' \
      GENIE_MAX_OUTPUT_TOKENS='$GENIE_MAX_OUTPUT_TOKENS' \
      GENIE_PERSISTENT_RUNNER='$GENIE_PERSISTENT_RUNNER' \
      '$PYTHON' -u -m vibroagent_mcp.genie_openai_server \
        --host '$NPU_HOST' \
        --port '$NPU_PORT' \
      >> '$MODEL_LOG' 2>&1
  "

  wait_for_model_server "$GENIE_READY_WAIT_S"
  local wait_rc=$?
  if [ "$wait_rc" -eq 0 ]; then
    ok "NPU model server is up: pid $(pid_from_file "$MODEL_PID_FILE")"
  elif [ "$wait_rc" -eq 2 ]; then
    print_model_start_failure
    return 1
  else
    err "NPU model server did not become ready on port $NPU_PORT within ${GENIE_READY_WAIT_S}s. See $MODEL_LOG"
    return 1
  fi
}

start_webchat() {
  local pid
  pid="$(pid_from_file "$WEBCHAT_PID_FILE")"
  if pid_alive "$pid"; then
    warn "Webchat already running: pid $pid"
    return 0
  fi
  if port_listening "$WEB_PORT"; then
    warn "Port $WEB_PORT is already in use by pid $(port_pid "$WEB_PORT")"
    return 0
  fi

  say "Starting webchat on http://$WEB_HOST:$WEB_PORT ..."
  : > "$WEBCHAT_LOG"
  setsid -f bash -c "
    echo \$\$ > '$WEBCHAT_PID_FILE'
    exec env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH='$SDK_PYTHONPATH' \
      QWEN_BASE_URL='http://$MODEL_API_HOST:$MODEL_API_PORT/v1' \
      QWEN_API_KEY='EMPTY' \
      QWEN_MODEL='$MODEL_API_MODEL' \
      QWEN_TIMEOUT_S='$QWEN_TIMEOUT_S' \
      QWEN_MAX_TOKENS='$QWEN_MAX_TOKENS' \
      QWEN_MAX_PROMPT_CHARS='$QWEN_MAX_PROMPT_CHARS' \
      QWEN_MODEL_TOOL_ROUTING='$QWEN_MODEL_TOOL_ROUTING' \
      VIBRO_BOARD_READER_PROCESSES='$VIBRO_BOARD_READER_PROCESSES' \
      VIBRO_BOARD_READER_TIMEOUT_S='$VIBRO_BOARD_READER_TIMEOUT_S' \
      AGENT_MODEL_MAX_TOKENS='$AGENT_MODEL_MAX_TOKENS' \
      AGENT_MODEL_TIMEOUT_S='$AGENT_MODEL_TIMEOUT_S' \
      AGENT_MONITOR_MODEL_TIMEOUT_S='$AGENT_MONITOR_MODEL_TIMEOUT_S' \
      AGENT_MONITOR_MAX_TOKENS='$AGENT_MONITOR_MAX_TOKENS' \
      AGENT_MONITOR_MAX_PROMPT_CHARS='$AGENT_MONITOR_MAX_PROMPT_CHARS' \
      AGENT_MONITOR_MAX_PROMPT_TOKENS='$AGENT_MONITOR_MAX_PROMPT_TOKENS' \
      AGENT_MAX_STEPS='$AGENT_MAX_STEPS' \
      AGENT_MAX_AXIS_CHECKS='$AGENT_MAX_AXIS_CHECKS' \
      AGENT_MAX_RECHECKS='$AGENT_MAX_RECHECKS' \
      AGENT_MAX_CONCURRENCY='$AGENT_MAX_CONCURRENCY' \
      '$PYTHON' -u -m vibroagent_mcp.webchat_server \
        --host '$WEB_HOST' \
        --port '$WEB_PORT' \
      >> '$WEBCHAT_LOG' 2>&1
  "

  if wait_for_port "$WEB_PORT" 30; then
    ok "Webchat is up: http://$WEB_HOST:$WEB_PORT pid $(pid_from_file "$WEBCHAT_PID_FILE")"
  else
    err "Webchat did not bind port $WEB_PORT. See $WEBCHAT_LOG"
    return 1
  fi
}

start_logger() {
  local pid
  pid="$(pid_from_file "$LOGGER_PID_FILE")"
  if pid_alive "$pid"; then
    warn "Logger already running: pid $pid"
    return 0
  fi

  say "Starting native callback logger ..."
  : > "$LOGGER_LOG"
  setsid -f bash -c "
    echo \$\$ > '$LOGGER_PID_FILE'
    cd '$EXAMPLES'
    exec env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH='$ROOT/stdatalog_core:$ROOT/stdatalog_pnpl:$ROOT/stdatalog_dtk:$ROOT/stdatalog_gui' \
      '$PYTHON' -u '$EXAMPLES/vibroagent_two_vibrometer_logger.py' \
        --output-root '$EXAMPLES' \
        --sensor '$LOGGER_SENSOR' \
        --stats-s '$LOGGER_STATS_S' \
        --stale-timeout-s '$LOGGER_STALE_TIMEOUT_S'$(logger_rollover_args) \
      >> '$LOGGER_LOG' 2>&1
  "

  local i=0
  while [ "$i" -lt 30 ]; do
    grep -q "Logging via native callbacks" "$LOGGER_LOG" 2>/dev/null && {
      ok "Logger is up: pid $(pid_from_file "$LOGGER_PID_FILE")"
      return 0
    }
    if grep -qiE "Traceback|SystemExit|Need at least|Native libhs_datalog_v2 open failed|zero HSDatalog|Unexpected communication backend" "$LOGGER_LOG" 2>/dev/null; then
      err "Logger failed to start. See $LOGGER_LOG"
      tail -80 "$LOGGER_LOG" 2>/dev/null | sed 's/^/  /'
      return 1
    fi
    sleep 1
    i=$((i + 1))
  done

  warn "Logger process started, but ready line was not seen within 30s. Check $LOGGER_LOG"
}

start_all() {
  ensure_paths || return 1
  say "== VibroAgent direct launcher: start =="
  local rc=0
  # Independent services still start even if one fails (the logger does not need
  # the model server), but any failure is surfaced via a nonzero exit so callers
  # and automation never assume the full stack is up when it is not.
  start_model_server || rc=1
  start_webchat || rc=1
  start_logger || rc=1
  say "----------------------------------------"
  status_all || rc=1
  return "$rc"
}

stop_pid_file() {
  local name="$1" file="$2" signal="${3:-INT}" timeout_s="${4:-15}"
  local pid i=0
  pid="$(pid_from_file "$file")"
  if ! pid_alive "$pid"; then
    say "$name not running."
    rm -f "$file"
    return 0
  fi

  say "Stopping $name with SIG$signal: pid $pid ..."
  kill "-$signal" "$pid" 2>/dev/null || true
  while [ "$i" -lt "$timeout_s" ]; do
    pid_alive "$pid" || {
      ok "$name stopped."
      rm -f "$file"
      return 0
    }
    sleep 1
    i=$((i + 1))
  done

  warn "$name did not stop after ${timeout_s}s; sending SIGTERM."
  kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  if pid_alive "$pid"; then
    warn "$name still alive; sending SIGKILL."
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

stop_logger() {
  local pid i=0 stopped=0
  pid="$(pid_from_file "$LOGGER_PID_FILE")"
  if ! pid_alive "$pid"; then
    say "Logger not running."
    rm -f "$LOGGER_PID_FILE"
    return 0
  fi

  say "Stopping logger with SIGINT so boards receive stop_log: pid $pid ..."
  kill -INT "$pid" 2>/dev/null || true
  while [ "$i" -lt 90 ]; do
    stopped="$(grep -cE 'Stopped (baseline|target_)' "$LOGGER_LOG" 2>/dev/null || true)"
    pid_alive "$pid" || {
      ok "Logger stopped cleanly. stop_log confirmations: $stopped"
      rm -f "$LOGGER_PID_FILE"
      return 0
    }
    sleep 1
    i=$((i + 1))
  done

  warn "Logger did not exit after 90s. Sending SIGTERM as last resort."
  kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  if pid_alive "$pid"; then
    warn "Logger still alive; sending SIGKILL."
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$LOGGER_PID_FILE"
}

# Persistent Genie runner pids that belong to THIS instance only. A runner's argv
# carries our specific GENIE_CONFIG, so we match on that and never signal another
# VibroAgent instance's (or another user's) live model server on a shared host.
our_genie_runner_pids() {
  local pid cmd
  for pid in $(pgrep -f '[g]enie_persistent_runner' 2>/dev/null || true); do
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    case "$cmd" in
      *"$GENIE_CONFIG"*) printf '%s\n' "$pid" ;;
    esac
  done
}

stop_orphaned_genie_runners() {
  local pids pid
  pids="$(our_genie_runner_pids)"
  [ -z "$pids" ] && return 0
  warn "Stopping this instance's Genie persistent runner(s): $(printf '%s ' $pids)"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  sleep 2
  for pid in $(our_genie_runner_pids); do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

stop_all() {
  say "== VibroAgent direct launcher: stop =="
  stop_logger
  stop_pid_file "webchat" "$WEBCHAT_PID_FILE" INT 15
  stop_pid_file "NPU model server" "$MODEL_PID_FILE" INT 15
  stop_orphaned_genie_runners
  ok "All direct-launch services stopped."
}

status_one() {
  local name="$1" file="$2" port="${3:-}"
  local pid
  pid="$(pid_from_file "$file")"
  if pid_alive "$pid"; then
    ok "$name: UP pid $pid"
    return 0
  elif [ -n "$port" ] && port_listening "$port"; then
    warn "$name: UP on port $port pid $(port_pid "$port") (pid file stale)"
    return 0
  fi
  err "$name: DOWN"
  return 1
}

logger_rollover_status() {
  local pid
  pid="$(pid_from_file "$LOGGER_PID_FILE")"
  if ! logger_rollover_enabled; then
    if pid_alive "$pid" && grep -azq -- '--max-dat-file-mb' "/proc/$pid/cmdline" 2>/dev/null; then
      warn "DAT rollover    : ACTIVE in current logger; restart logger to disable"
    else
      say "DAT rollover    : DISABLED (set LOGGER_MAX_DAT_MB>0 to enable)"
    fi
    return 0
  fi
  if pid_alive "$pid"; then
    if grep -azq -- '--max-dat-file-mb' "/proc/$pid/cmdline" 2>/dev/null; then
      ok "DAT rollover    : ACTIVE (${LOGGER_MAX_DAT_MB} MB cap, ${LOGGER_ROLLOVER_KEEP_MB} MB tail)"
    else
      warn "DAT rollover    : configured for next start only; current logger lacks rollover flags"
    fi
  else
    say "DAT rollover    : configured for next start (${LOGGER_MAX_DAT_MB} MB cap, ${LOGGER_ROLLOVER_KEEP_MB} MB tail)"
  fi
}

status_all() {
  local rc=0
  status_one "model server ($MODEL_BACKEND)" "$MODEL_PID_FILE" "$MODEL_API_PORT" || rc=1
  status_one "Webchat" "$WEBCHAT_PID_FILE" "$WEB_PORT" || rc=1
  status_one "Logger" "$LOGGER_PID_FILE" || rc=1
  say "Webchat URL: http://$WEB_HOST:$WEB_PORT"
  say "Model URL  : http://$MODEL_API_HOST:$MODEL_API_PORT/v1 ($MODEL_BACKEND)"
  say "Run dir    : $RUN_DIR"
  logger_rollover_status
  return "$rc"
}

tail_logs() {
  say "Tailing logs. Press Ctrl+C to stop."
  tail -f "$MODEL_LOG" "$WEBCHAT_LOG" "$LOGGER_LOG"
}

case "${1:-start}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    sleep 3
    start_all
    ;;
  status)
    status_all
    ;;
  logs)
    tail_logs
    ;;
  start-webchat)
    ensure_paths || exit 1
    start_webchat
    ;;
  stop-webchat)
    stop_pid_file "webchat" "$WEBCHAT_PID_FILE" INT 15
    ;;
  restart-webchat)
    ensure_paths || exit 1
    stop_pid_file "webchat" "$WEBCHAT_PID_FILE" INT 15
    sleep 3
    start_webchat
    ;;
  start-logger)
    start_logger
    ;;
  stop-logger)
    stop_logger
    ;;
  restart-logger)
    stop_logger
    sleep 3
    start_logger
    ;;
  start-model)
    ensure_paths || exit 1
    start_model_server
    ;;
  stop-model)
    stop_pid_file "NPU model server" "$MODEL_PID_FILE" INT 15
    stop_orphaned_genie_runners
    ;;
  restart-model)
    # Restart only the NPU model server (e.g. after a code/runner change) without
    # bouncing the logger, which could leave the STWIN.box boards wedged.
    stop_pid_file "NPU model server" "$MODEL_PID_FILE" INT 15
    stop_orphaned_genie_runners
    sleep 2
    ensure_paths || exit 1
    start_model_server
    ;;
  *)
    err "Usage: $0 [start|stop|restart|status|logs|start-webchat|stop-webchat|restart-webchat|start-logger|stop-logger|restart-logger|start-model|stop-model|restart-model]"
    exit 2
    ;;
esac
