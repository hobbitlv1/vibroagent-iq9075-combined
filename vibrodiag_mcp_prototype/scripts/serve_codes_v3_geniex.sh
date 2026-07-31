#!/usr/bin/env bash
# Serve the fine-tuned codes_v3 GGUF on the Qualcomm Hexagon NPU through
# GenieX and expose it as an OpenAI-compatible HTTP endpoint.
#
# The primary launcher invokes this script automatically. To run it directly:
#
#   ./vibrodiag_mcp_prototype/scripts/serve_codes_v3_geniex.sh \
#       ./models/qwen3_4b_codes_v3_Q4_0_embq8.gguf
#
# Environment overrides:
#   GENIEX_PYTHON, GENIEX_HOST, GENIEX_PORT, GENIEX_DEVICE_MAP,
#   GGML_HEXAGON_NDEV, GENIEX_N_CTX, GENIEX_EXPECT_MODEL_SHA256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$PROTO/.." && pwd)"

GENIEX_PYTHON="${GENIEX_PYTHON:-$HOME/geniex-venv/bin/python}"
GGUF="${1:-$REPO/models/qwen3_4b_codes_v3_Q4_0_embq8.gguf}"
MODEL_SHA256="3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2"
EXPECT_MODEL_SHA256="${GENIEX_EXPECT_MODEL_SHA256:-$MODEL_SHA256}"

HOST="${GENIEX_HOST:-127.0.0.1}"
PORT="${GENIEX_PORT:-18181}"
DEVICE_MAP="${GENIEX_DEVICE_MAP:-llama_cpp:HTP0,HTP1}"
HEXAGON_NDEV="${GGML_HEXAGON_NDEV:-2}"
N_CTX="${GENIEX_N_CTX:-6144}"
EXPECT_DEVICE="${GENIEX_EXPECT_DEVICE:-${DEVICE_MAP#llama_cpp:}}"
MODEL_LABEL="${GENIEX_MODEL_LABEL:-codes_v3}"
GIT_COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

[ -x "$GENIEX_PYTHON" ] || {
    echo "GenieX Python not found: $GENIEX_PYTHON (run ./setup_geniex.sh)" >&2
    exit 1
}
[ -f "$GGUF" ] || {
    echo "codes_v3 GGUF not found: $GGUF (run ./setup_models.sh)" >&2
    exit 1
}

echo "Serving codes_v3 model: $GGUF"
echo "  device_map=$DEVICE_MAP  n_ctx=$N_CTX"
echo "  endpoint=http://$HOST:$PORT/v1"
echo "  sha256 pin=$EXPECT_MODEL_SHA256"

exec env \
    GGML_HEXAGON_NDEV="$HEXAGON_NDEV" \
    PYTHONPATH="$PROTO/src" \
    GENIEX_EXPECT_DEVICE="$EXPECT_DEVICE" \
    GENIEX_EXPECT_MODEL_SHA256="$EXPECT_MODEL_SHA256" \
    GENIEX_MODEL_LABEL="$MODEL_LABEL" \
    VIBROAGENT_GIT_COMMIT="$GIT_COMMIT" \
    "$GENIEX_PYTHON" -u -m vibroagent_mcp.geniex_openai_server \
        --host "$HOST" \
        --port "$PORT" \
        --model "$GGUF" \
        --model-label "$MODEL_LABEL" \
        --device-map "$DEVICE_MAP" \
        --n-ctx "$N_CTX"
