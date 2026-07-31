#!/usr/bin/env bash
# setup_geniex.sh — install the Qualcomm GenieX runtime used by VibroAgent.
#
# GenieX (https://github.com/qualcomm/GenieX, docs: https://geniex.aihub.qualcomm.com)
# is llama.cpp with the GGML Hexagon backend and runs the fine-tuned codes_v3
# GGUF directly on the NPU. The model itself is installed separately by
# setup_models.sh so the base Qwen model is not downloaded unnecessarily.
#
# The GenieX venv is intentionally separate from the app venv:
# vibroagent_mcp.geniex_openai_server runs with the GenieX venv's Python.
set -euo pipefail

GENIEX_VENV="${GENIEX_VENV:-$HOME/geniex-venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x "$GENIEX_VENV/bin/python" ]; then
    echo "== creating GenieX virtualenv at $GENIEX_VENV (via uv)"
    uv venv "$GENIEX_VENV"
fi

echo "== installing geniex from PyPI (via uv)"
uv pip install --quiet --python "$GENIEX_VENV/bin/python" geniex
"$GENIEX_VENV/bin/geniex-py" version

echo "== GenieX runtime ready: $GENIEX_VENV"
echo "   setup_models.sh installs the hash-pinned codes_v3 GGUF."
