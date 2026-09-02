#!/usr/bin/env bash
# Download and verify the VibroAgent-Gemma Q8 GGUF from the combined repository release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${VIBROGEMMA_BUNDLE:-$ROOT/models/vibroagent-gemma-g1}"
ASSET_MANIFEST="$ROOT/models/gemma_release_assets_manifest.json"
ASSET_MANIFEST_NAME="$(basename "$ASSET_MANIFEST")"
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
    GH_REPO="${VIBRO_GH_REPO:-hobbitlv1/vibroagent-iq9075-combined}"
    RELEASE_TAG="${VIBRO_GEMMA_RELEASE_TAG:-weights-v1}"
    [ -n "$GH_REPO" ] || { echo "cannot resolve GitHub repository" >&2; exit 1; }
    echo "== downloading $GGUF_NAME parts from $GH_REPO@$RELEASE_TAG"
    if command -v gh >/dev/null 2>&1; then
        gh release download "$RELEASE_TAG" -R "$GH_REPO" \
            -p "$ASSET_MANIFEST_NAME" -p "$GGUF_NAME*.part" -D "$WORK" --clobber
    else
        auth=()
        [ -z "${GITHUB_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
        curl -fsSL "${auth[@]}" \
            "https://api.github.com/repos/$GH_REPO/releases/tags/$RELEASE_TAG" \
            -o "$WORK/release.json" || {
                echo "release metadata unavailable; install gh and run 'gh auth login', or set GITHUB_TOKEN" >&2
                exit 1
            }
        python3 - "$WORK/release.json" "$GGUF_NAME" "$ASSET_MANIFEST_NAME" > "$WORK/assets.tsv" <<'PY'
import json, sys
release, prefix, manifest = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3]
assets = sorted(
    (asset for asset in release.get("assets", [])
     if asset.get("name") == manifest
     or (asset.get("name", "").startswith(prefix) and asset.get("name", "").endswith(".part"))),
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

    [ -f "$WORK/$ASSET_MANIFEST_NAME" ] || { echo "release asset manifest is missing" >&2; exit 1; }
    cmp -s "$ASSET_MANIFEST" "$WORK/$ASSET_MANIFEST_NAME" || {
        echo "release asset manifest does not match the reviewed repository copy" >&2
        exit 1
    }

    parts=()
    while IFS=$'\t' read -r name size sha; do
        part="$WORK/$name"
        [ -f "$part" ] || { echo "missing release part: $name" >&2; exit 1; }
        [ "$(stat -c %s "$part")" = "$size" ] || { echo "size mismatch: $name" >&2; exit 1; }
        echo "$sha  $part" | sha256sum -c --quiet - || { echo "SHA-256 mismatch: $name" >&2; exit 1; }
        parts+=("$part")
    done < <(python3 - "$ASSET_MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
for part in manifest["gguf"]["parts"]:
    print(part["name"], part["size_bytes"], part["sha256"], sep="\t")
PY
)
    [ "${#parts[@]}" -ge 2 ] || { echo "release manifest contains no complete split-part set" >&2; exit 1; }
    cat "${parts[@]}" > "$WORK/$GGUF_NAME"
fi

verify "$WORK/$GGUF_NAME" || { echo "VibroAgent-Gemma Q8 SHA-256 mismatch" >&2; exit 1; }
mv "$WORK/$GGUF_NAME" "$TARGET"
echo "== verified model ready: $TARGET"
