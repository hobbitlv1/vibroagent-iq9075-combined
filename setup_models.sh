#!/usr/bin/env bash
# setup_models.sh — download the fine-tuned codes_v3 model automatically.
#
# The frozen codec checkpoint (models/codec_v1/best.pt, 15 MB) is versioned in
# this repository; the fine-tuned LLM is not (2.4 GB). This script first tries
# the application repository's private GitHub release, then falls back to the
# Hugging Face weights repository. The final GGUF is always verified against
# its pinned sha256 before anything may serve it.
#
#   GitHub access       `gh auth login`, or GITHUB_TOKEN when gh is unavailable
#   HF_TOKEN            fallback read token; also reads the cached hf login
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
# A public weights repo downloads anonymously; a private one needs HF_TOKEN
# (or an 'hf auth login' token on disk) from an account with read access.

# ---- source 1: GitHub release assets on the application repo -----------------
# The GGUF ships as two byte-split parts (GitHub caps release assets at 2 GB).
# Reassembly is a pure byte concatenation; each part is sha256-verified against
# release_assets_manifest.json first, and the final file must STILL match the
# hard pin below — corruption anywhere fails closed.
GH_REPO="${VIBRO_GH_REPO:-$(git -C "$REPO" remote get-url origin 2>/dev/null \
    | sed -E 's#(git@github\.com:|https://github\.com/)##; s#\.git$##')}"
RELEASE_TAG="${VIBRO_RELEASE_TAG:-weights-v1}"
github_release_fetch() {
    local asset="$1" dest="$2"
    if command -v gh >/dev/null 2>&1; then
        gh release download "$RELEASE_TAG" -R "$GH_REPO" -p "$asset" \
            -O "$dest" --clobber
    else
        [ -n "${GITHUB_TOKEN:-}" ] || return 1
        local asset_id
        asset_id="$(curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
            "https://api.github.com/repos/$GH_REPO/releases/tags/$RELEASE_TAG" \
            | python3 -c "import json,sys;print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='$asset'))")" || return 1
        curl -sfL -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/octet-stream" \
            -o "$dest" "https://api.github.com/repos/$GH_REPO/releases/assets/$asset_id"
    fi
}
if [ -n "$GH_REPO" ]; then
    echo "== trying GitHub release $GH_REPO@$RELEASE_TAG"
    WORK="$REPO/models/.parts"; mkdir -p "$WORK"
    if github_release_fetch "release_assets_manifest.json" "$WORK/release_assets_manifest.json"; then
        ok=1
        while read -r part sha; do
            echo "==   part $part"
            github_release_fetch "$part" "$WORK/$part" || { ok=0; break; }
            echo "$sha  $WORK/$part" | sha256sum -c --quiet - || { ok=0; break; }
        done < <(python3 -c "
import json
m = json.load(open('$WORK/release_assets_manifest.json'))
for p in m['gguf']['parts']:
    print(p['name'], p['sha256'])")
        if [ "$ok" = "1" ]; then
            cat "$WORK"/*.part > "$TARGET"
            rm -rf "$WORK"
            echo "== verifying reassembled sha256"
            echo "$GGUF_SHA256  $TARGET" | sha256sum -c -
            echo "== codes_v3 GGUF ready (GitHub release): $TARGET"
            exit 0
        fi
        echo "== GitHub release path incomplete — falling back to Hugging Face"
        rm -rf "$WORK"
    else
        echo "== no release assets reachable — falling back to Hugging Face"
    fi
fi

# ---- source 2: Hugging Face weights repo ------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== downloading $GGUF_NAME from $MODELS_REPO (2.4 GB)"
mkdir -p "$REPO/models"
HF_TOKEN="${HF_TOKEN:-}" uv tool run --from 'huggingface_hub[cli]' \
    hf download "$MODELS_REPO" "$GGUF_NAME" --local-dir "$REPO/models"

echo "== verifying sha256"
echo "$GGUF_SHA256  $TARGET" | sha256sum -c -
echo "== codes_v3 GGUF ready: $TARGET"
