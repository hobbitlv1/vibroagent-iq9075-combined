#!/usr/bin/env bash
# Download and verify the VibroAgent-Gemma Q8 GGUF from this repository's release assets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${VIBROGEMMA_BUNDLE:-$ROOT/models/vibroagent-gemma-g1}"
GGUF_NAME="gemma-4-e2b-g1-Q8_0.gguf"
GGUF_SHA256="11ceefee8d62080072fe2b65f68beab5e0f31ad716c640178ed93b5ac0eb31d6"
TARGET="$BUNDLE/$GGUF_NAME"

verify() { echo "$GGUF_SHA256  $1" | sha256sum -c --quiet -; }
mkdir -p "$BUNDLE"
if [ -f "$TARGET" ] && verify "$TARGET"; then
    echo "== VibroAgent-Gemma Q8 model already present and verified"
    exit 0
fi

WORK="$(mktemp -d "$BUNDLE/.download.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
if [ -n "${VIBROGEMMA_GGUF_SOURCE:-}" ]; then
    echo "== copying local model source"
    cp "$VIBROGEMMA_GGUF_SOURCE" "$WORK/$GGUF_NAME"
else
    GH_REPO="${VIBRO_GH_REPO:-$(git -C "$ROOT" remote get-url origin 2>/dev/null \
        | sed -E 's#(git@github\.com:|https://github\.com/)##; s#\.git$##')}"
    RELEASE_TAG="${VIBRO_GEMMA_RELEASE_TAG:-weights-v1}"
    [ -n "$GH_REPO" ] || { echo "cannot resolve GitHub repository" >&2; exit 1; }
    echo "== downloading $GGUF_NAME parts from $GH_REPO@$RELEASE_TAG"
    if command -v gh >/dev/null 2>&1; then
        gh release download "$RELEASE_TAG" -R "$GH_REPO" \
            -p "$GGUF_NAME*.part" -D "$WORK" --clobber
    else
        auth=()
        [ -z "${GITHUB_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
        curl -fsSL "${auth[@]}" \
            "https://api.github.com/repos/$GH_REPO/releases/tags/$RELEASE_TAG" \
            -o "$WORK/release.json" || {
                echo "release metadata unavailable; install gh and run 'gh auth login', or set GITHUB_TOKEN" >&2
                exit 1
            }
        python3 - "$WORK/release.json" "$GGUF_NAME" > "$WORK/assets.tsv" <<'PY'
import json, sys
release, prefix = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2]
assets = sorted(
    (asset for asset in release.get("assets", [])
     if asset.get("name", "").startswith(prefix) and asset.get("name", "").endswith(".part")),
    key=lambda asset: asset["name"],
)
for asset in assets:
    print(asset["name"], asset["id"], asset["browser_download_url"], sep="\t")
PY
        while IFS=$'\t' read -r name asset_id public_url; do
            [ -n "$name" ] || continue
            if [ -n "${GITHUB_TOKEN:-}" ]; then
                curl -fsSL "${auth[@]}" -H "Accept: application/octet-stream" \
                    "https://api.github.com/repos/$GH_REPO/releases/assets/$asset_id" -o "$WORK/$name"
            else
                curl -fsSL "$public_url" -o "$WORK/$name"
            fi
        done < "$WORK/assets.tsv"
    fi
    mapfile -t parts < <(find "$WORK" -maxdepth 1 -type f -name "$GGUF_NAME*.part" -print | sort)
    [ "${#parts[@]}" -ge 2 ] || { echo "no complete Q8 split-part set found in the release" >&2; exit 1; }
    cat "${parts[@]}" > "$WORK/$GGUF_NAME"
fi

verify "$WORK/$GGUF_NAME" || { echo "VibroAgent-Gemma Q8 SHA-256 mismatch" >&2; exit 1; }
mv "$WORK/$GGUF_NAME" "$TARGET"
echo "== verified model ready: $TARGET"
