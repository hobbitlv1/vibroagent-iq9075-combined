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
#
# The runtime version is pinned. An unpinned install would take whatever PyPI
# serves that day, which is many releases past the one this stack was built on.
#   GENIEX_VERSION   install exactly this release (overrides everything below)
# Without it, an already-installed release is kept as is, and a fresh
# environment gets GENIEX_DEFAULT_VERSION.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENIEX_VENV="${GENIEX_VENV:-$HOME/geniex-venv}"
# The release geniex_openai_server.py was written against.
GENIEX_DEFAULT_VERSION="0.3.16"
# Pinned versions of everything geniex pulls in (compiled from constraints/codec.in).
CONSTRAINTS="${VIBRO_CONSTRAINTS:-$REPO/constraints/codec.txt}"

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (pinned release, checksum-verified)"
    "$REPO/setup_uv.sh"
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x "$GENIEX_VENV/bin/python" ]; then
    echo "== creating GenieX virtualenv at $GENIEX_VENV (via uv)"
    uv venv "$GENIEX_VENV"
fi

installed_version="$("$GENIEX_VENV/bin/python" -c \
    'from importlib.metadata import version; print(version("geniex"))' 2>/dev/null || true)"
case "$installed_version" in
    ''|*[!0-9.]*) installed_version="" ;;
esac
if [ -n "${GENIEX_VERSION:-}" ]; then
    geniex_version="$GENIEX_VERSION"
elif [ -n "$installed_version" ]; then
    geniex_version="$installed_version"
    echo "== keeping installed geniex $installed_version (set GENIEX_VERSION to change it)"
else
    geniex_version="$GENIEX_DEFAULT_VERSION"
fi

echo "== installing geniex==$geniex_version from PyPI (via uv)"
uv pip install --quiet --python "$GENIEX_VENV/bin/python" --constraint "$CONSTRAINTS" \
    "geniex==$geniex_version"
"$GENIEX_VENV/bin/geniex-py" version

echo "== GenieX runtime ready: $GENIEX_VENV"
echo "   setup_models.sh installs the hash-pinned codes_v3 GGUF."
