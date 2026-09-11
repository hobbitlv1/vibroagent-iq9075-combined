#!/usr/bin/env bash
# vibroagent.sh — manage the current VibroAgent production stack:
#   1) fine-tuned codes_v3 model on Hexagon through GenieX (:18181)
#   2) webchat, live graphs, and monitoring UI (:7860)
#   3) live STWIN.box acquisition or immutable recorded-data replay
#
# Usage:
#   ./vibroagent.sh [start|stop|restart|status|stop-model|restart-webchat]
#
# VIBRO_CODES_AXES=x,y,z optionally evaluates each axis and combines the
# strict schema verdicts by maximum severity. The configured z axis is the
# default. Logs and pidfiles are stored under .run/.
set -u

# ---- resolve paths (script lives at repo root) -----------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$REPO/vibrodiag_mcp_prototype"
EX="$REPO/stdatalog_examples"
VENV="$REPO/vibroagent-venv/bin/python"
RUN_DIR="$REPO/.run"
mkdir -p "$RUN_DIR"

# ---- config (override via environment if needed) ---------------------------
GENIEX_PORT="${GENIEX_PORT:-18181}"
CODES_GGUF="${CODES_GGUF:-$REPO/models/qwen3_4b_codes_v3_Q4_0_embq8.gguf}"
CODES_GGUF_SHA256="${CODES_GGUF_SHA256:-3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2}"
# Split the pure-attention Qwen3-4B across two Hexagon sessions so its 6144-token
# KV cache is not constrained by HTP0 alone.
CODES_DEVICE_MAP="${CODES_DEVICE_MAP:-llama_cpp:HTP0,HTP1}"
CODES_HEXAGON_NDEV="${CODES_HEXAGON_NDEV:-2}"
CODES_N_CTX="${CODES_N_CTX:-6144}"
WEB_PORT="${WEB_PORT:-7860}"
# Bind the webchat to the LAN interface only (default-route source IP): reachable from the
# LAN/router but NOT on tailscale0 or loopback. Set WEB_HOST=127.0.0.1 for local-only, or
# WEB_HOST=0.0.0.0 for every interface including Tailscale. The webchat has no auth.
LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)"
WEB_HOST="${WEB_HOST:-${LAN_IP:-127.0.0.1}}"
if [ "$WEB_HOST" = "0.0.0.0" ]; then WEB_LOCAL_HOST=127.0.0.1; else WEB_LOCAL_HOST="$WEB_HOST"; fi
BASELINE_SERIAL="003F003D3530500820323641"
LOGGER_PY="$EX/vibroagent_two_vibrometer_logger.py"
NATIVE_PROBE_PY="$EX/vibroagent_hsd_native_probe.py"
# Data-source mode is fixed at SETUP time (./setup.sh --offline writes it):
# "live"    = USB logger drives real STWIN.box boards (default)
# "offline" = the web service reads immutable recording windows directly;
#             no logger starts, so the supplied .dat files are never modified.
VIBRO_MODE="$(cat "$REPO/.vibro_mode" 2>/dev/null || echo live)"
REPLAY_SOURCE_ROOT="${VIBRO_REPLAY_SOURCE_ROOT:-$REPO/recordings}"
REPLAY_MANIFEST="${VIBRO_REPLAY_MANIFEST:-$REPLAY_SOURCE_ROOT/recordings_manifest.json}"
EXPECTED_BOARDS="${EXPECTED_BOARDS:-6}"
GENIEX_READY_WAIT_S="${GENIEX_READY_WAIT_S:-360}"
# All services must import THIS tree's stdatalog packages, not whichever tree the
# venv's editable installs happen to point at.
SDK_PYTHONPATH="$REPO/stdatalog_core:$REPO/stdatalog_pnpl"

c_grn=$'\e[32m'; c_red=$'\e[31m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
say() { printf '%s\n' "$*"; }
ok()  { printf '%s%s%s\n' "$c_grn" "$*" "$c_off"; }
warn(){ printf '%s%s%s\n' "$c_yel" "$*" "$c_off"; }
err() { printf '%s%s%s\n' "$c_red" "$*" "$c_off"; }

port_listening() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
port_pid()       { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1; }
logger_pid()     { pgrep -f "vibroagent_two_vibrometer_logger\.py" 2>/dev/null | head -1; }

wait_for_port() { # port, timeout_s
  local p="$1" t="${2:-60}" i=0
  while [ "$i" -lt "$t" ]; do port_listening "$p" && return 0; sleep 1; i=$((i+1)); done
  return 1
}

webchat_probe_url() {
  local host="$WEB_HOST"
  case "$host" in
    0.0.0.0|::) host="127.0.0.1" ;;
    \[*\]) ;;  # Already bracketed IPv6 literal.
    *:*) host="[$host]" ;;
  esac
  printf 'http://%s:%s' "$host" "$WEB_PORT"
}

logger_is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 1 ] || return 1
  "$VENV" - "$pid" "$REPO" "$PROTO/scripts" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from safe_pipeline_stop import process_role, read_process
proc = read_process(int(sys.argv[1]))
sys.exit(0 if proc is not None and process_role(proc, Path(sys.argv[2]).resolve()) == "logger" else 1)
PY
}

board_count() {
  local n=0 pth
  for pth in /sys/bus/usb/devices/*; do
    [ "$(cat "$pth/idVendor" 2>/dev/null)" = "0483" ] && \
    [ "$(cat "$pth/idProduct" 2>/dev/null)" = "5744" ] && n=$((n+1))
  done
  echo "$n"
}
baseline_present() {
  local pth
  for pth in /sys/bus/usb/devices/*; do
    [ "$(cat "$pth/serial" 2>/dev/null)" = "$BASELINE_SERIAL" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------- start
start_geniex() {
  if port_listening "$GENIEX_PORT"; then warn "geniex codes server already on :$GENIEX_PORT (pid $(port_pid "$GENIEX_PORT")) — skipping"; return 0; fi
  local device_map="${GENIEX_DEVICE_MAP:-$CODES_DEVICE_MAP}"
  local hexagon_ndev="${GGML_HEXAGON_NDEV:-$CODES_HEXAGON_NDEV}"
  local n_ctx="${GENIEX_N_CTX:-$CODES_N_CTX}"
  say "Starting geniex codes server on :$GENIEX_PORT ($(basename "$CODES_GGUF"), $device_map, n_ctx=$n_ctx) ..."
  ( cd "$REPO" || exit 1
    export GENIEX_PORT
    export GENIEX_DEVICE_MAP="$device_map"
    export GGML_HEXAGON_NDEV="$hexagon_ndev"
    export GENIEX_N_CTX="$n_ctx"
    export GENIEX_EXPECT_MODEL_SHA256="${GENIEX_EXPECT_MODEL_SHA256:-$CODES_GGUF_SHA256}"
    nohup bash "$PROTO/scripts/serve_codes_v3_geniex.sh" "$CODES_GGUF" \
      > "$RUN_DIR/geniex.log" 2>&1 &
    echo $! > "$RUN_DIR/geniex.pid" )
  # Wait for the port, but fail FAST if the daemon dies during startup —
  # a blind port wait turns a load error into a silent 6-minute hang.
  local i=0 pid; pid="$(cat "$RUN_DIR/geniex.pid")"
  while [ "$i" -lt "$GENIEX_READY_WAIT_S" ]; do
    if port_listening "$GENIEX_PORT"; then
      ok "  geniex ready (codes model on $device_map, n_ctx=$n_ctx, hash-verified) — pid $pid"
      return 0
    fi
    if ! ps -o pid= -p "$pid" >/dev/null 2>&1; then
      err "  geniex daemon exited during startup — last log lines:"
      tail -12 "$RUN_DIR/geniex.log" 2>/dev/null | sed 's/^/  /'
      return 1
    fi
    sleep 1; i=$((i+1))
  done
  err "  geniex did not open :$GENIEX_PORT in ${GENIEX_READY_WAIT_S}s — see $RUN_DIR/geniex.log"
  return 1
}

start_webchat() {
  if port_listening "$WEB_PORT"; then warn "Webchat already on :$WEB_PORT (pid $(port_pid "$WEB_PORT")) — skipping"; return 0; fi
  say "Starting webchat on :$WEB_PORT ..."
  ( cd "$PROTO" || exit 1
    export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:$GENIEX_PORT/v1}"
    export VIBRO_CODES_WORKER_PYTHON="${VIBRO_CODES_WORKER_PYTHON:-$HOME/codec-cpu-venv/bin/python}"
    export QWEN_API_KEY="EMPTY" QWEN_MODEL="${QWEN_MODEL:-qwen3_4b_codes_v3}"
    export QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"
    export QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
    export QWEN_MAX_PROMPT_CHARS="${QWEN_MAX_PROMPT_CHARS:-9000}"
    export QWEN_MODEL_TOOL_ROUTING="${QWEN_MODEL_TOOL_ROUTING:-0}"
    export AGENT_MODEL_MAX_TOKENS="${AGENT_MODEL_MAX_TOKENS:-1024}"
    export AGENT_MODEL_TIMEOUT_S="${AGENT_MODEL_TIMEOUT_S:-300}"
    # $PROTO/src first so THIS tree's vibroagent_mcp wins over the venv's editable
    # install (which can point at a sibling checkout); VIBRO_ALLOWED_HSD_DIR pins the
    # data sandbox to the selected acquisition root for the same reason.
    # pyarrow shim: the venv's pyarrow is renamed pyarrow.disabled_sigbus (SIGBUS
    # via dask.dataframe, Jun 15); the webchat gets the probed-safe 22.0.0 copy
    # from .run/pyarrow_shim instead. Webchat only.
    export PYTHONPATH="$PROTO/src:$SDK_PYTHONPATH:$RUN_DIR/pyarrow_shim${PYTHONPATH:+:$PYTHONPATH}"
    export VIBRO_MODE
    if [ "$VIBRO_MODE" = "offline" ]; then
      export VIBRO_OFFLINE_REPLAY=1
      export VIBRO_ACQUISITION_ROOT="$REPLAY_SOURCE_ROOT"
      export VIBRO_REPLAY_MANIFEST="$REPLAY_MANIFEST"
      export VIBRO_ALLOWED_HSD_DIR="$REPLAY_SOURCE_ROOT"
    else
      export VIBRO_OFFLINE_REPLAY=0
      unset VIBRO_ACQUISITION_ROOT VIBRO_REPLAY_MANIFEST
      export VIBRO_ALLOWED_HSD_DIR="$EX"
    fi
    nohup "$VENV" -m vibroagent_mcp.webchat_server \
      --host "$WEB_HOST" --port "$WEB_PORT" \
      > "$RUN_DIR/webchat.log" 2>&1 &
    echo $! > "$RUN_DIR/webchat.pid" )
  if wait_for_port "$WEB_PORT" 30; then ok "  Webchat up — http://$WEB_HOST:$WEB_PORT  (graph: /graph)"
  else err "  Webchat did not open :$WEB_PORT — see $RUN_DIR/webchat.log"; fi
}

start_logger() {
  safe_stop_services logger --check-start || return $?
  if [ -n "$(logger_pid)" ]; then warn "Logger already running (pid $(logger_pid)) — skipping"; return 0; fi
  if [ "$VIBRO_MODE" = "offline" ]; then
    say "OFFLINE mode (set at setup): immutable acquisition replay — no logger starts."
    if [ ! -f "$REPLAY_MANIFEST" ]; then
      err "  Missing replay manifest: $REPLAY_MANIFEST"
      return 1
    fi
    local folder
    for folder in live_baseline live_target_1 live_target_2 live_target_3 live_target_4 live_target_5; do
      if [ ! -s "$REPLAY_SOURCE_ROOT/$folder/iis3dwb_acc.dat" ]; then
        err "  Missing or empty immutable recording: $REPLAY_SOURCE_ROOT/$folder/iis3dwb_acc.dat"
        return 1
      fi
    done
    ok "  Replay source ready: $REPLAY_SOURCE_ROOT"
    say "  Graphs read fixed windows directly; codec-v1 + codes_v3 run only at manifest timestamps."
    return 0
  fi
  local n; n="$(board_count)"
  if ! baseline_present || [ "$n" -lt 2 ]; then
    err "  Skipping logger: need baseline + >=1 target, but found $n board(s) and baseline_present=$(baseline_present && echo yes || echo no)."
    err "  Reconnect boards, then: ./vibroagent.sh start   (NPU+webchat already up will be skipped)"
    return 1
  fi
  say "Checking native libhs_datalog_v2 before logger start ..."
  ( cd "$REPO" || exit 1
    PYTHONPATH="$SDK_PYTHONPATH" PYTHONUNBUFFERED=1 "$VENV" -u \
      "$NATIVE_PROBE_PY" --output-root "$EX" --expected-count "$EXPECTED_BOARDS" \
      > "$RUN_DIR/native_probe.log" 2>&1 )
  local probe_rc=$?
  if [ "$probe_rc" -ne 0 ]; then
    err "  Skipping logger: native HSDatalog probe failed with code $probe_rc — see $RUN_DIR/native_probe.log"
    tail -80 "$RUN_DIR/native_probe.log" 2>/dev/null | sed 's/^/  /'
    return 1
  fi

  say "Starting logger ($n boards detected, native HSD probe OK) ..."
  ( cd "$REPO" || exit 1
    PYTHONPATH="$SDK_PYTHONPATH" PYTHONUNBUFFERED=1 nohup "$VENV" -u \
      "$LOGGER_PY" --output-root "$EX" --sensor iis3dwb_acc --stats-s 5 --stale-timeout-s 2 \
      > "$RUN_DIR/logger.log" 2>&1 &
    echo $! > "$RUN_DIR/logger.pid" )
  # wait for acquisition threads (or a hard error); boards start one by one,
  # so keep waiting while the Started count is still rising instead of
  # reporting the first board's count as the total
  local i=0 started=0 prev=0 stall=0
  while [ "$i" -lt 45 ]; do
    started="$(grep -cE 'Started .* on device' "$RUN_DIR/logger.log" 2>/dev/null)"
    [ "${started:-0}" -ge "$n" ] && break
    if [ "${started:-0}" -gt "$prev" ]; then prev="$started"; stall=0; else stall=$((stall+1)); fi
    [ "${started:-0}" -gt 0 ] && [ "$stall" -ge 8 ] && break
    # NB: pattern must not match the logger's success line
    # "Native libhs_datalog_v2 opened correctly with N device(s)."
    grep -qiE 'Traceback|SystemExit|Need at least|Native libhs_datalog_v2 open failed|serial/v1 fallback|zero HSDatalog|Unexpected communication backend' "$RUN_DIR/logger.log" 2>/dev/null && break
    sleep 1; i=$((i+1))
  done
  started="$(grep -cE 'Started .* on device' "$RUN_DIR/logger.log" 2>/dev/null)"
  local pid; pid="$(cat "$RUN_DIR/logger.pid" 2>/dev/null)"
  if [ "${started:-0}" -ge 1 ] && logger_is_alive "$pid"; then ok "  Logger up — $started board(s) acquiring (pid $pid)"
  else err "  Logger failed to start — see $RUN_DIR/logger.log"; return 1; fi
  # prewarm per-board reader workers so the first 'all boards' poll is fast
  local sensors_path="/api/live-sensors"
  [ "${STACK:-codes}" = "gemma" ] && sensors_path="/api/vibro/sensors"
  if curl --noproxy '*' --fail --silent --show-error --max-time 90 \
      "$(webchat_probe_url)$sensors_path?process_reader=1&prewarm=1" -o /dev/null; then
    say "  Reader workers prewarmed."
  else
    warn "  Reader prewarm unavailable; acquisition remains running and readers initialize on demand."
  fi
  if ! logger_is_alive "$pid"; then
    err "  Logger exited during reader prewarm — see $RUN_DIR/logger.log"
    return 1
  fi
  return 0
}

do_start() {
  say "== VibroAgent: launching the codes_v3 production stack =="
  start_geniex || return $?
  start_webchat || return $?
  start_logger || return $?
  say "----------------------------------------"
  do_status
}

# ---------------------------------------------------------------------------- stop
safe_stop_services() {
  local service="$1"; shift
  # Shutdown uses only stdlib/Linux pidfds, never the optional application venv.
  /usr/bin/python3 -I "$PROTO/scripts/safe_pipeline_stop.py" \
    --repo "$REPO" --service "$service" \
    --timeout "${VIBRO_SAFE_STOP_TIMEOUT_S:-120}" "$@"
}

stop_logger() { safe_stop_services logger; }

stop_by_port() { safe_stop_services "$1"; }


do_stop() {
  say "== VibroAgent: safe shutdown (no forced board termination) =="
  safe_stop_services all "$@"
}

# ---------------------------------------------------------------------------- status
do_restart() {
  do_stop || return $?
  if [ "${MODE:-${VIBRO_MODE:-live}}" != offline ]; then
    say "Board shutdown confirmed; allowing USB to settle ..."
    sleep 6
  fi
  do_start
}

do_status() {
  if port_listening "$GENIEX_PORT"; then ok "geniex codes LLM : UP   :$GENIEX_PORT (pid $(port_pid "$GENIEX_PORT"))"
  else err "geniex codes LLM : DOWN :$GENIEX_PORT"; fi
  port_listening "$WEB_PORT"  && ok  "Webchat          : UP   :$WEB_PORT (pid $(port_pid "$WEB_PORT"))" || err "Webchat          : DOWN :$WEB_PORT"
  if [ "$VIBRO_MODE" = "offline" ]; then
    if [ -f "$REPLAY_MANIFEST" ]; then ok "Replay source     : READY (immutable)"; else err "Replay source     : MISSING"; fi
    say "Replay root       : $REPLAY_SOURCE_ROOT"
    say "Data writer       : NONE"
  else
    if [ -n "$(logger_pid)" ]; then ok "Logger           : UP   (pid $(logger_pid))"; else err "Logger           : DOWN"; fi
    local n; n="$(board_count)"
    say "Boards detected  : $n / 6   (baseline $(baseline_present && echo present || echo MISSING))"
  fi
}

# ---------------------------------------------------------------------------- main
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${1:-}" != "--control-locked" ]]; then
    exec flock --nonblock --conflict-exit-code 75 --close "$RUN_DIR/control.lock" \
      "$REPO/vibroagent.sh" --control-locked "$@"
  fi
  shift
case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop "${@:2}" ;;
  restart) do_restart ;;
  stop-model) stop_by_port geniex "$GENIEX_PORT" ;;
  restart-webchat) stop_by_port webchat "$WEB_PORT" && start_webchat ;;
  status)  do_status ;;
  *) err "Usage: $0 [start|stop|restart|status|stop-model|restart-webchat]"; exit 2 ;;
esac
fi
