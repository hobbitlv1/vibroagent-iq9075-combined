#!/usr/bin/env bash
# vibroagent.sh — launch/stop the 3 live services for the VibroAgent prototype:
#   1) model server on the NPU (Qualcomm Genie / Hexagon DSP)   :8910
#   2) webchat (graph + chat + live readings panel)             :7860
#   3) logger (drives the 6 STWIN.box boards over USB)
#
# Usage:
#   ./vibroagent.sh [start|stop|restart|status|stop-npu|restart-webchat]   (default: start)
#   stop-npu / restart-webchat leave the logger (and its USB boards) untouched.
#
# Default stack (VIBRO_STACK=codes): geniex serves the finetuned codes_v3
# GGUF across HTP0+HTP1 (:18181, sha-pinned, 6K context) and the webchat
# monitor decides from codec-v1 codes. One command does everything:
# ./vibroagent.sh start
# Legacy comparison stack: VIBRO_STACK=genie ./vibroagent.sh start  (QAIRT
# genie on :8910, merged LoRA, descriptor monitor — schema-stale).
# Axis sweep: VIBRO_CODES_AXES=x,y,z judges each axis separately and combines
# by max severity (~3x NPU time per poll). Default: the per-sensor config axis.
#
# Notes:
#   * The workspace folder is still named vibrodiag_mcp_prototype/ (the venv lives
#     inside it); the Python package is vibroagent_mcp.
#   * Logs + pidfiles go to .run/ ; tail them with:  tail -f .run/*.log
set -u

# ---- resolve paths (script lives at repo root) -----------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$REPO/vibrodiag_mcp_prototype"
EX="$REPO/stdatalog_examples"
VENV="$PROTO/.venv/bin/python"
RUN_DIR="$REPO/.run"
mkdir -p "$RUN_DIR"

# ---- config (override via environment if needed) ---------------------------
# VIBRO_STACK=codes (default): finetuned codes_v3 GGUF on geniex
#   llama_cpp:HTP0,HTP1 with a 6K context (:18181) + codec-v1 codes monitor
#   (VIBRO_MONITOR_DECISION_MODE=codes).
# VIBRO_STACK=genie: the legacy QAIRT genie server on :8910 (merged LoRA,
#   descriptor monitor) — schema-stale, kept for comparison only.
STACK="${VIBRO_STACK:-codes}"
GENIEX_PORT="${GENIEX_PORT:-18181}"
# Default = the hash-verified download from setup_models.sh; the legacy board
# path is the fallback for deployments predating the in-repo models/ layout.
if [ -f "$REPO/models/qwen3_4b_codes_v3_Q4_0_embq8.gguf" ]; then
  CODES_GGUF="${CODES_GGUF:-$REPO/models/qwen3_4b_codes_v3_Q4_0_embq8.gguf}"
else
  CODES_GGUF="${CODES_GGUF:-/media/ubuntu/Drive/downloads_heavy/qwen3_4b_codes_v3_Q4_0_embq8.gguf}"
fi
CODES_GGUF_SHA256="${CODES_GGUF_SHA256:-3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2}"
# Split the pure-attention Qwen3-4B across two Hexagon sessions so its 6144-token
# KV cache is not constrained by HTP0 alone.
CODES_DEVICE_MAP="${CODES_DEVICE_MAP:-llama_cpp:HTP0,HTP1}"
CODES_HEXAGON_NDEV="${CODES_HEXAGON_NDEV:-2}"
CODES_N_CTX="${CODES_N_CTX:-6144}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
NPU_PORT="${NPU_PORT:-8910}"
WEB_PORT="${WEB_PORT:-7860}"
# Bind the webchat to the LAN interface only (default-route source IP): reachable from the
# LAN/router but NOT on tailscale0 or loopback. Set WEB_HOST=127.0.0.1 for local-only, or
# WEB_HOST=0.0.0.0 for every interface including Tailscale. The webchat has no auth.
LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)"
WEB_HOST="${WEB_HOST:-${LAN_IP:-127.0.0.1}}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-$REPO/v2.46.0.260424/qairt/2.46.0.260424}"
GENIE_CONFIG="${GENIE_CONFIG:-$REPO/models/merged_4k_6000s/merged_6000_ctx4096_calib1-genie-w4a16-qualcomm_qcs9075/genie_config.json}"
BASELINE_SERIAL="003F003D3530500820323641"
LOGGER_PY="$EX/vibroagent_two_vibrometer_logger.py"
REPLAY_PY="$EX/vibroagent_replay_logger.py"
NATIVE_PROBE_PY="$EX/vibroagent_hsd_native_probe.py"
# Data-source mode is fixed at SETUP time (./setup.sh --offline writes it):
# "live"    = USB logger drives real STWIN.box boards (default)
# "offline" = the replay logger streams the recorded examples/ acquisitions;
#             the USB logger NEVER starts (it would overwrite the replayed
#             .dat files), so no boards and no USB permissions are needed.
VIBRO_MODE="$(cat "$REPO/.vibro_mode" 2>/dev/null || echo live)"
EXPECTED_BOARDS="${EXPECTED_BOARDS:-6}"
GENIE_TIMEOUT_S="${GENIE_TIMEOUT_S:-300}"
GENIE_READY_WAIT_S="${GENIE_READY_WAIT_S:-360}"
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
logger_pid()     { pgrep -f "vibroagent_(two_vibrometer|replay)_logger\.py" 2>/dev/null | head -1; }

wait_for_port() { # port, timeout_s
  local p="$1" t="${2:-60}" i=0
  while [ "$i" -lt "$t" ]; do port_listening "$p" && return 0; sleep 1; i=$((i+1)); done
  return 1
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
start_npu() {
  if port_listening "$NPU_PORT"; then warn "NPU model server already on :$NPU_PORT (pid $(port_pid "$NPU_PORT")) — skipping"; return 0; fi
  say "Starting NPU model server on :$NPU_PORT ..."
  ( cd "$PROTO" || exit 1
    export QAIRT_SDK_ROOT GENIE_CONFIG
    export GENIE_MODEL="$MODEL_ID" GENIE_PROMPT_FORMAT="qwen3"
    export GENIE_TIMEOUT_S
    export GENIE_ADAPTER_MAX_PROMPT_CHARS="${GENIE_ADAPTER_MAX_PROMPT_CHARS:-12000}"
    # 1024, not 384: the all-sensor fusion response can exceed 384 tokens before
    # its finalization block; a truncated JSON answer makes the agent fall back
    # and the no-fallback chat path then reports a model-explanation error.
    # Keep in sync with vibroagent_direct.sh and the adapter default.
    export GENIE_MAX_OUTPUT_TOKENS="${GENIE_MAX_OUTPUT_TOKENS:-1024}"
    export PYTHONUNBUFFERED=1
    # $PROTO/src first so this tree's NPU adapter wins over any editable install
    # that may point at a sibling checkout.
    export PYTHONPATH="$PROTO/src:$SDK_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$QAIRT_SDK_ROOT/lib/hexagon-v73/unsigned:${LD_LIBRARY_PATH:-}"
    nohup "$VENV" -u -m vibroagent_mcp.genie_openai_server \
      --host 127.0.0.1 --port "$NPU_PORT" --model-id "$MODEL_ID" \
      > "$RUN_DIR/npu.log" 2>&1 &
    echo $! > "$RUN_DIR/npu.pid" )
  if wait_for_port "$NPU_PORT" "$GENIE_READY_WAIT_S"; then ok "  NPU ready (model preloaded on Hexagon DSP) — pid $(cat "$RUN_DIR/npu.pid")"
  else err "  NPU did not become ready on :$NPU_PORT in ${GENIE_READY_WAIT_S}s — see $RUN_DIR/npu.log"; fi
}

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
    nohup bash "$PROTO/scripts/serve_qwen35_npu.sh" "$CODES_GGUF" \
      > "$RUN_DIR/geniex.log" 2>&1 &
    echo $! > "$RUN_DIR/geniex.pid" )
  # Wait for the port, but fail FAST if the daemon dies during startup —
  # a blind port wait turns a load error into a silent 6-minute hang.
  local i=0 pid; pid="$(cat "$RUN_DIR/geniex.pid")"
  while [ "$i" -lt "$GENIE_READY_WAIT_S" ]; do
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
  err "  geniex did not open :$GENIEX_PORT in ${GENIE_READY_WAIT_S}s — see $RUN_DIR/geniex.log"
  return 1
}

start_webchat() {
  if port_listening "$WEB_PORT"; then warn "Webchat already on :$WEB_PORT (pid $(port_pid "$WEB_PORT")) — skipping"; return 0; fi
  say "Starting webchat on :$WEB_PORT ..."
  ( cd "$PROTO" || exit 1
    if [ "$STACK" = "codes" ]; then
      # codes stack: consume the geniex GGUF-on-HTP endpoint and decide via
      # codec-v1 codes (webchat VIBRO_MONITOR_DECISION_MODE=codes path).
      export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:$GENIEX_PORT/v1}"
      export VIBRO_MONITOR_DECISION_MODE="${VIBRO_MONITOR_DECISION_MODE:-codes}"
    fi
    export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:$NPU_PORT/v1}" QWEN_API_KEY="EMPTY" QWEN_MODEL="$MODEL_ID"
    export QWEN_TIMEOUT_S="${QWEN_TIMEOUT_S:-300}"
    export QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
    export QWEN_MAX_PROMPT_CHARS="${QWEN_MAX_PROMPT_CHARS:-9000}"
    export QWEN_MODEL_TOOL_ROUTING="${QWEN_MODEL_TOOL_ROUTING:-0}"
    export AGENT_MODEL_MAX_TOKENS="${AGENT_MODEL_MAX_TOKENS:-1024}"
    export AGENT_MODEL_TIMEOUT_S="${AGENT_MODEL_TIMEOUT_S:-300}"
    # $PROTO/src first so THIS tree's vibroagent_mcp wins over the venv's editable
    # install (which can point at a sibling checkout); VIBRO_ALLOWED_HSD_DIR pins the
    # live-data sandbox to THIS tree's stdatalog_examples for the same reason.
    # pyarrow shim: the venv's pyarrow is renamed pyarrow.disabled_sigbus (SIGBUS
    # via dask.dataframe, Jun 15); the webchat gets the probed-safe 22.0.0 copy
    # from .run/pyarrow_shim instead. Webchat only — logger/NPU never import it.
    export PYTHONPATH="$PROTO/src:$SDK_PYTHONPATH:$RUN_DIR/pyarrow_shim${PYTHONPATH:+:$PYTHONPATH}"
    export VIBRO_ALLOWED_HSD_DIR="$EX"
    nohup "$VENV" -m vibroagent_mcp.webchat_server \
      --host "$WEB_HOST" --port "$WEB_PORT" \
      > "$RUN_DIR/webchat.log" 2>&1 &
    echo $! > "$RUN_DIR/webchat.pid" )
  if wait_for_port "$WEB_PORT" 30; then ok "  Webchat up — http://$WEB_HOST:$WEB_PORT  (graph: /graph)"
  else err "  Webchat did not open :$WEB_PORT — see $RUN_DIR/webchat.log"; fi
}

start_logger() {
  if [ -n "$(logger_pid)" ]; then warn "Logger already running (pid $(logger_pid)) — skipping"; return 0; fi
  if [ "$VIBRO_MODE" = "offline" ]; then
    say "OFFLINE mode (set at setup): replaying recorded acquisitions — USB logger stays down."
    local replay_src="$REPO/examples"
    [ -f "$REPO/recordings/recordings_manifest.json" ] && replay_src="$REPO/recordings"
    say "  replay source: $replay_src"
    ( cd "$REPO" || exit 1
      PYTHONUNBUFFERED=1 nohup python3 -u \
        "$REPLAY_PY" --output-root "$EX" --source-root "$replay_src" --stats-s 5 \
        > "$RUN_DIR/logger.log" 2>&1 &
      echo $! > "$RUN_DIR/logger.pid" )
    local i=0
    while [ "$i" -lt 15 ]; do
      [ "$(grep -c 'Replay started' "$RUN_DIR/logger.log" 2>/dev/null)" -ge 1 ] && break
      grep -qiE 'Traceback|SystemExit' "$RUN_DIR/logger.log" 2>/dev/null && break
      sleep 1; i=$((i+1))
    done
    if [ "$(grep -c 'Replay started' "$RUN_DIR/logger.log" 2>/dev/null)" -ge 1 ]; then
      ok "  Replay logger up: $(grep -c 'Replay started' "$RUN_DIR/logger.log") recorded acquisitions streaming in real time (looping)."
    else
      err "  Replay logger failed — see $RUN_DIR/logger.log"
      tail -20 "$RUN_DIR/logger.log" 2>/dev/null | sed 's/^/  /'
      return 1
    fi
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
  if [ "${started:-0}" -ge 1 ]; then ok "  Logger up — $started board(s) acquiring (pid $(cat "$RUN_DIR/logger.pid"))"
  else err "  Logger failed to start — see $RUN_DIR/logger.log"; return 1; fi
  # prewarm per-board reader workers so the first 'all boards' poll is fast
  curl -s -m 90 "http://127.0.0.1:$WEB_PORT/api/live-sensors?process_reader=1&prewarm=1" -o /dev/null \
    && say "  Reader workers prewarmed."
}

do_start() {
  say "== VibroAgent: launching all services (stack: $STACK) =="
  if [ "$STACK" = "codes" ]; then start_geniex; else start_npu; fi
  start_webchat
  start_logger
  say "----------------------------------------"
  do_status
}

# ---------------------------------------------------------------------------- stop
stop_logger() {
  # Graceful only. The logger MUST run stop_log on every board over USB before it
  # exits; killing it mid-stream (SIGTERM/SIGKILL) leaves boards in a bad state and
  # drops them off the bus (needs a physical replug). So: SIGINT, wait generously,
  # nudge once more, and treat SIGTERM/SIGKILL strictly as a last resort.
  local p; p="$(logger_pid)"
  [ -z "$p" ] && [ -f "$RUN_DIR/logger.pid" ] && p="$(cat "$RUN_DIR/logger.pid")"
  if [ -z "${p:-}" ] || ! ps -o pid= -p "$p" >/dev/null 2>&1; then
    say "Logger not running."; rm -f "$RUN_DIR/logger.pid"; return 0
  fi
  local nstarted; nstarted="$(grep -cE 'Started .* on device' "$RUN_DIR/logger.log" 2>/dev/null)"
  say "Stopping logger (SIGINT -> clean stop_log on ${nstarted:-?} board(s)) pid $p ..."
  kill -INT "$p" 2>/dev/null
  # Wait by PROGRESS, not a blind timer. With the logger started unbuffered, each
  # board's "Stopped <role>" line appears as it is released, so keep waiting while the
  # count is still rising and only conclude the logger is wedged in a native call
  # after a real stall. A normally-progressing stop is always allowed to finish;
  # only a true hang is force-killed.
  local i=0 nudged=0 prev=-1 stall=0 nstopped=0 forced=0
  while [ "$i" -lt 120 ]; do
    ps -o pid= -p "$p" >/dev/null 2>&1 || { ok "  Logger exited cleanly after ${i}s."; break; }
    nstopped="$(grep -cE 'Stopped (baseline|target_)' "$RUN_DIR/logger.log" 2>/dev/null)"
    if [ "$nstopped" -gt "$prev" ]; then prev="$nstopped"; stall=0; else stall=$((stall+1)); fi
    if [ "$stall" -eq 12 ] && [ "$nudged" -eq 0 ]; then warn "  no progress for 12s; resending SIGINT"; kill -INT "$p" 2>/dev/null; nudged=1; fi
    if [ "$stall" -ge 25 ]; then warn "  no stop_log progress for 25s — logger appears wedged in a native HSD call."; break; fi
    sleep 1; i=$((i+1))
  done
  if ps -o pid= -p "$p" >/dev/null 2>&1; then
    err "  Logger not exiting (stopped ${nstopped}/${nstarted:-?} boards). LAST resort SIGTERM."
    err "  A board mid-stream may need a replug; this wedge is almost always a marginal USB link."
    forced=1
    kill "$p" 2>/dev/null; sleep 3
    ps -o pid= -p "$p" >/dev/null 2>&1 && kill -9 "$p" 2>/dev/null
  fi
  say "  stop_log confirmations: ${nstopped:-0}/${nstarted:-?} board(s)."
  sleep 2  # let USB settle before any re-acquire
  local n; n="$(board_count)"
  if [ "$n" -lt 6 ]; then
    warn "  Only $n/6 boards enumerated after stop — one may have dropped; replug if a board is missing on next start."
  elif [ "$forced" -eq 1 ] || { [ "${nstarted:-0}" -gt 0 ] && [ "${nstopped:-0}" -lt "${nstarted:-0}" ]; }; then
    warn "  All $n boards still enumerated, but logger shutdown was not clean (${nstopped:-0}/${nstarted:-?} stop_log confirmations)."
    warn "  If the next start cannot acquire a board, replug the affected board(s)."
  else
    ok "  All $n boards still enumerated — clean."
  fi
  rm -f "$RUN_DIR/logger.pid"
}

stop_by_port() { # name, port  (webchat/NPU only — these do NOT hold the USB boards)
  local name="$1" port="$2" p; p="$(port_pid "$port")"
  if [ -n "$p" ]; then
    say "Stopping $name (pid $p, :$port) ..."; kill "$p" 2>/dev/null
    local i=0; while [ "$i" -lt 6 ]; do ps -o pid= -p "$p" >/dev/null 2>&1 || break; sleep 1; i=$((i+1)); done
    ps -o pid= -p "$p" >/dev/null 2>&1 && { warn "  still alive after ${i}s, SIGKILL"; kill -9 "$p" 2>/dev/null; }
    ok "  $name stopped."
  else say "$name not running."; fi
  rm -f "$RUN_DIR/${name}.pid"
}

release_dsp() {
  # kill any orphaned Genie runner still holding /dev/fastrpc-cdsp
  local pid
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    if ls -l "/proc/$pid/fd" 2>/dev/null | grep -q fastrpc-cdsp; then
      warn "  Genie runner $pid still holds the DSP — terminating"
      kill "$pid" 2>/dev/null; sleep 2
      ps -o pid= -p "$pid" >/dev/null 2>&1 && kill -9 "$pid" 2>/dev/null
    fi
  done
}

do_stop() {
  say "== VibroAgent: stopping all services =="
  stop_logger
  stop_by_port webchat "$WEB_PORT"
  stop_by_port npu "$NPU_PORT"
  stop_by_port geniex "$GENIEX_PORT"
  release_dsp
  ok "All stopped."
}

# ---------------------------------------------------------------------------- status
do_status() {
  local n; n="$(board_count)"
  if port_listening "$GENIEX_PORT"; then ok "geniex codes LLM : UP   :$GENIEX_PORT (pid $(port_pid "$GENIEX_PORT"))"
  elif [ "$STACK" = "codes" ]; then err "geniex codes LLM : DOWN :$GENIEX_PORT"; fi
  port_listening "$NPU_PORT"  && ok  "NPU model server : UP   :$NPU_PORT (pid $(port_pid "$NPU_PORT"))" || { [ "$STACK" = "genie" ] && err "NPU model server : DOWN :$NPU_PORT"; }
  port_listening "$WEB_PORT"  && ok  "Webchat          : UP   :$WEB_PORT (pid $(port_pid "$WEB_PORT"))" || err "Webchat          : DOWN :$WEB_PORT"
  if [ -n "$(logger_pid)" ]; then ok "Logger           : UP   (pid $(logger_pid))"; else err "Logger           : DOWN"; fi
  say "Boards detected  : $n / 6   (baseline $(baseline_present && echo present || echo MISSING))"
}

# ---------------------------------------------------------------------------- main
case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; say "Letting USB settle before restart ..."; sleep 6; do_start ;;
  stop-npu)        stop_by_port npu "$NPU_PORT"; release_dsp; ok "NPU stopped, DSP released." ;;
  restart-webchat) stop_by_port webchat "$WEB_PORT"; start_webchat ;;
  status)  do_status ;;
  *) err "Usage: $0 [start|stop|restart|status]"; exit 2 ;;
esac
