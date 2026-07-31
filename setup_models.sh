#!/usr/bin/env bash
# setup_models.sh — download the fine-tuned codes_v3 model automatically.
#
# The frozen codec checkpoint (models/codec_v1/best.pt, 15 MB) is versioned in
# this repository; the fine-tuned LLM is not (2.4 GB). This script pulls it
# from the private Hugging Face weights repo and verifies its pinned sha256
# before anything may serve it.
#
#   HF_TOKEN            read token for the private weights repo; falls back to
#                       ~/.cache/huggingface/token (hf auth login)
#   VIBRO_MODELS_REPO   default hobbitlv/vibroagent-models
#
# Idempotent — a present, hash-verified GGUF is left untouched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_REPO="${VIBRO_MODELS_REPO:-hobbitlv/vibroagent-models}"
GGUF_NAME="qwen3_4b_codes_v3_Q4_0_embq8.gguf"
GGUF_SHA256="3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2"
TARGET="$REPO/models/$GGUF_NAME"

if [ -f "$TARGET" ]; then
    echo "== verifying existing $TARGET"
    if echo "$GGUF_SHA256  $TARGET" | sha256sum -c --quiet -; then
        echo "== codes_v3 GGUF already in place and hash-verified"
        exit 0
    fi
    echo "== existing file FAILS verification — re-downloading"
    rm -f "$TARGET"
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: the weights repo is private — set HF_TOKEN or run 'hf auth login'." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== downloading $GGUF_NAME from $MODELS_REPO (2.4 GB)"
mkdir -p "$REPO/models"
HF_TOKEN="$HF_TOKEN" uv tool run --from 'huggingface_hub[cli]' \
    hf download "$MODELS_REPO" "$GGUF_NAME" --local-dir "$REPO/models"

echo "== verifying sha256"
echo "$GGUF_SHA256  $TARGET" | sha256sum -c -
echo "== codes_v3 GGUF ready: $TARGET"
