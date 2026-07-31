#!/usr/bin/env bash
# setup_geniex.sh — install the Qualcomm GenieX runtime and auto-download the model.
#
# GenieX (https://github.com/qualcomm/GenieX, docs: https://geniex.aihub.qualcomm.com)
# is the DEFAULT model backend: llama.cpp with the GGML Hexagon backend, running
# GGUF models directly on the NPU. It needs neither the QAIRT SDK bundle nor a
# Genie-compiled model:
#   - the `geniex` package installs from public PyPI,
#   - the model is pulled from Hugging Face into ~/.cache/geniex (~2.3 GB).
#
# The GenieX venv is intentionally separate from the app venv:
# vibroagent_mcp.geniex_openai_server runs with the GenieX venv's Python and
# imports only stdlib + geniex (vibroagent_direct.sh points GENIEX_PYTHON here).
set -euo pipefail

GENIEX_VENV="${GENIEX_VENV:-$HOME/geniex-venv}"
# Q4_0 is the one GGUF precision that lands on the Hexagon NPU; other quants
# fall back to GPU/CPU (per GenieX model docs).
GENIEX_MODEL="${GENIEX_MODEL:-unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0}"

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

echo "== pulling model $GENIEX_MODEL from Hugging Face (cache: ~/.cache/geniex)"
"$GENIEX_VENV/bin/geniex-py" pull "$GENIEX_MODEL"

echo "== cached models:"
"$GENIEX_VENV/bin/geniex-py" ls

echo "== done. Start the stack with:  ./vibroagent_direct.sh start"
echo "   (MODEL_BACKEND=geniex is the default; GENIEX_PYTHON=$GENIEX_VENV/bin/python)"
