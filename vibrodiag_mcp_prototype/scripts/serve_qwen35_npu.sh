#!/usr/bin/env bash
# Serve a Qwen3.5 requant GGUF on the Hexagon NPU as an OpenAI-compatible endpoint,
# backed by geniex (llama.cpp/HTP) via the repo's geniex_openai_server bridge.
#
# RUN THIS IN YOUR OWN TERMINAL (e.g. the `! ...` prompt), not from an automated
# tool call: a backgrounded DSP-holding daemon gets killed at the tool boundary,
# but a synchronous foreground server in your shell stays up.
#
# Usage:
#   serve_qwen35_npu.sh [variant|/abs/path.gguf]
#   GENIEX_ENABLE_VISION=1 serve_qwen35_npu.sh    # load mmproj-BF16 -> VLM endpoint
#   GENIEX_MMPROJ=/path/mmproj.gguf serve_qwen35_npu.sh   # explicit projector
#
# Vision is OPT-IN only: presence of the mmproj file on disk does not switch the
# production text endpoint into an (unmeasured) VLM configuration by itself.
#
# VISION is EXPERIMENTAL and CPU-only on this geniex build (board-measured
# 2026-07-23). Three constraints: (1) an image on an HTP device_map hard-
# SEGFAULTS the daemon (CLIP tower ops unsupported on HTP), so vision forces
# GENIEX_DEVICE_MAP=llama_cpp:cpu; (2) the CPU VLM handle is single-shot, so the
# server reloads it (~3.3s) before each request; (3) each reload leaks ~4.9 GB
# that isn't reclaimed, so vision is hard-capped (GENIEX_RELOAD_BUDGET, default 2
# construction attempts) and the server must be RESTARTED after the initial handle
# plus ~2 replacement generations. Text-only serving is unaffected and stays on
# HTP. Vision is off by default; production text uses no mmproj. For a dedicated
# vision worker, set GENIEX_VISION_REQUIRE_IMAGE=1 so stray text traffic can't
# spend the budget.
#
# Once it prints "Serving GenieX OpenAI-compatible API", point the apps at it:
#   webchat:  QWEN_BASE_URL=http://127.0.0.1:18181/v1  python -m vibroagent_mcp.webchat_server ...
#   agent:    MAIN_AGENT_BASE_URL=http://127.0.0.1:18181/v1 \
#             SMALL_AGENT_BASE_URL=http://127.0.0.1:18181/v1  (+ your logger pipeline)
#
# Health check from another shell:
#   curl -s http://127.0.0.1:18181/health
#   curl -s http://127.0.0.1:18181/v1/chat/completions -H 'content-type: application/json' \
#     -d '{"messages":[{"role":"user","content":"Reply OK."}],"max_tokens":8,"temperature":0}'
set -euo pipefail

REPO=/media/ubuntu/Drive/zip1/vibrodiag_mcp_prototype
REPO_SRC="$REPO/src"
MODELS=/media/ubuntu/Drive/llm_models/qwen35-4b-htp-requant

# Full-file sha256 pins from the requant receipts (hobbitlv/qwen35-4b-htp-requant
# receipts.json): the hash, not the filename, is what separates a board-proven
# requant from the shipped artifact. The server hashes the files once at startup
# and fails closed on mismatch.
declare -A GGUF_SHA256=(
  [ssmout-q8_0-embd-q4_0]=a609cfb0bbe29f07a828413f807556171748ddf16c5f27f01c3a576e92093ce2
  [ssmout-q8_0-embd-q6_k]=909804cc14654b137824d4ed9e3f95e4e1eabd34762ddeea6689cfda29dfee8b
  [ssmout-q4_0-embd-q4_0]=22c69371d02b1d791f13b130a98db16af036a3e8c1db201daee09e65d7e372e0
  [ssmout-q4_0-embd-q6_k]=f7dd84c7c3910dffd319c251b810cdb4871a098ca36a7cdbdb351f26ce08ce50
)
MMPROJ_BF16_SHA256=302b92d565080b9cc0281186979ae75a7429ec23d14f6f7607a035539b21f3a6

# Default: the recommended requant (ssm_out q8_0, token_embd q4_0 — +23% vs shipped,
# preserves vendor ssm_out precision). Override with $1 (a variant name) or an
# absolute path (which skips the hash pin — no receipt to pin against).
VARIANT="${1:-ssmout-q8_0-embd-q4_0}"
EXPECT_MODEL_SHA256=""
MODEL_LABEL=""
if [ -f "$VARIANT" ]; then
  GGUF="$VARIANT"
  MODEL_LABEL="$(basename "$VARIANT")"
  # Explicit paths have no requant receipt; honor a caller-supplied pin so
  # e.g. the codes_v3 monitor GGUF still fails closed on byte drift.
  EXPECT_MODEL_SHA256="${GENIEX_EXPECT_MODEL_SHA256:-}"
  if [ -z "$EXPECT_MODEL_SHA256" ]; then
    echo "note: explicit path given; no hash pin for $GGUF (set GENIEX_EXPECT_MODEL_SHA256 to pin)" >&2
  fi
else
  GGUF="$MODELS/Qwen3.5-4B-${VARIANT}.gguf"
  EXPECT_MODEL_SHA256="${GGUF_SHA256[$VARIANT]:-}"
  MODEL_LABEL="$VARIANT"
  [ -n "$EXPECT_MODEL_SHA256" ] || { echo "unknown variant: $VARIANT (known: ${!GGUF_SHA256[*]})" >&2; exit 1; }
fi
[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF" >&2; exit 1; }

# Vision (multimodal) is explicit opt-in; see header. mmproj hash is pinned only
# for the known BF16 projector.
MMPROJ="${GENIEX_MMPROJ:-}"
EXPECT_MMPROJ_SHA256=""
if [ -z "$MMPROJ" ] && [ "${GENIEX_ENABLE_VISION:-0}" = "1" ]; then
  MMPROJ="$MODELS/mmproj-BF16.gguf"
fi
if [ -n "$MMPROJ" ]; then
  [ -f "$MMPROJ" ] || { echo "mmproj not found: $MMPROJ (hf download hobbitlv/qwen35-4b-htp-requant mmproj-BF16.gguf --local-dir $MODELS, with HF_HUB_DISABLE_XET=1)" >&2; exit 1; }
  [ "$(basename "$MMPROJ")" = "mmproj-BF16.gguf" ] && EXPECT_MMPROJ_SHA256="$MMPROJ_BF16_SHA256"
fi

HOST="${GENIEX_HOST:-127.0.0.1}"
PORT="${GENIEX_PORT:-18181}"
# device_map MUST resolve to the llama_cpp plugin on HTP. 'npu' routes GGUF to QAIRT
# (which rejects .gguf); 'llama_cpp:HTP0' is the proven selector (== smoke's --device).
# Vision forces CPU (see header): HTP + image = segfault on this build.
if [ -n "$MMPROJ" ]; then
  DEVICE_MAP="${GENIEX_DEVICE_MAP:-llama_cpp:cpu}"
else
  DEVICE_MAP="${GENIEX_DEVICE_MAP:-llama_cpp:HTP0}"
fi
# 6144 is the measured n_ctx ceiling on HTP0 (8192's KV cache exceeds the fastrpc mmap limit).
N_CTX="${GENIEX_N_CTX:-6144}"
# A comma-separated HTP device map needs one registered Hexagon session per
# selected device. The VibroAgent launcher sets this to 2 for HTP0,HTP1.
HEXAGON_NDEV="${GGML_HEXAGON_NDEV:-1}"
GIT_COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
# Assert the model resolves to the device we asked for (catches npu->QAIRT
# misroute). Track the device map's compute unit so the vision (CPU) path does
# not trip an HTP0 assertion.
EXPECT_DEVICE="${GENIEX_EXPECT_DEVICE:-${DEVICE_MAP##*:}}"

echo "Serving $GGUF"
echo "  device_map=$DEVICE_MAP  n_ctx=$N_CTX  vision=${MMPROJ:-off}  ->  http://$HOST:$PORT/v1"
[ -n "$EXPECT_MODEL_SHA256" ] && echo "  model hash pin: $EXPECT_MODEL_SHA256 (verified by the server at startup)"
exec env GGML_HEXAGON_NDEV="$HEXAGON_NDEV" PYTHONPATH="$REPO_SRC" \
  GENIEX_EXPECT_DEVICE="$EXPECT_DEVICE" \
  GENIEX_EXPECT_MODEL_SHA256="$EXPECT_MODEL_SHA256" \
  GENIEX_EXPECT_MMPROJ_SHA256="$EXPECT_MMPROJ_SHA256" \
  GENIEX_MODEL_LABEL="$MODEL_LABEL" \
  VIBROAGENT_GIT_COMMIT="$GIT_COMMIT" \
  /home/ubuntu/geniex-venv/bin/python -u -m vibroagent_mcp.geniex_openai_server \
  --host "$HOST" --port "$PORT" --model "$GGUF" --device-map "$DEVICE_MAP" --n-ctx "$N_CTX" \
  ${MMPROJ:+--mmproj-path "$MMPROJ"}
