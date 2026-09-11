#!/usr/bin/env bash
# setup_recordings.sh — place the offline-mode recordings (5-minute .dat sets).
#
# Offline mode replays full-length recorded acquisitions; those are too large
# for git (~280 MB for six 5-minute IIS3DWB streams), so this script first
# tries the combined repository's GitHub release and then falls
# back to the Hugging Face weights repository. Every downloaded file is
# verified against the recording manifest's sha256 pins. The web service reads
# fixed timestamp windows directly from ./recordings/live_*; no replay process
# opens those source files for writing.
#
#   GitHub access       `gh auth login`, or GITHUB_TOKEN when gh is unavailable
#   HF_TOKEN            fallback read token; also reads the cached hf login
#   VIBRO_MODELS_REPO   default hobbitlv/vibroagent-models
#
# Idempotent — a present, fully hash-verified recording set is left untouched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_REPO="${VIBRO_MODELS_REPO:-hobbitlv/vibroagent-models}"
TARGET="$REPO/recordings"

verify_recordings() {
    python3 - "${1:-$TARGET}" "$REPO/models/release_assets_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "recordings_manifest.json"
if not manifest_path.is_file():
    sys.exit(1)
release = json.loads(Path(sys.argv[2]).read_text())
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != release["recordings"]["manifest_sha256"]:
    sys.exit(1)
manifest = json.loads(manifest_path.read_text())
if not manifest.get("slots") or any(not entry.get("files") for entry in manifest["slots"].values()):
    sys.exit(1)
for slot, entry in manifest["slots"].items():
    for name, sha in entry["files"].items():
        path = root / slot / name
        if not path.is_file():
            sys.exit(1)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            sys.exit(1)
print("recordings verified:", ", ".join(
    f"{slot} {entry['seconds']:.0f}s" for slot, entry in sorted(manifest["slots"].items())))
PY
}

if verify_recordings 2>/dev/null; then
    echo "== offline recordings already in place and hash-verified"
    exit 0
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
# A public weights repo downloads anonymously; a private one needs HF_TOKEN
# (or an 'hf auth login' token on disk) from an account with read access.

# ---- source 1: GitHub release asset on the established weights repo ----------
GH_REPO="${VIBRO_GH_REPO:-hobbitlv1/vibroagent-iq9075-combined}"
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
mkdir -p "$TARGET"
WORK="$(mktemp -d "$TARGET/.download.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
if [ -n "$GH_REPO" ]; then
    echo "== trying GitHub release $GH_REPO@$RELEASE_TAG"
    ARCHIVE_SHA="$(python3 - "$REPO/models/release_assets_manifest.json" <<'HASH'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["recordings"]["sha256"])
HASH
)"
    mkdir -p "$WORK/extracted"
    if github_release_fetch "offline_recordings.tar.gz" "$WORK/recordings.tar.gz" &&
            echo "$ARCHIVE_SHA  $WORK/recordings.tar.gz" | sha256sum -c --quiet - &&
            tar xzf "$WORK/recordings.tar.gz" -C "$WORK/extracted" &&
            verify_recordings "$WORK/extracted"; then
        cp -r "$WORK/extracted/." "$TARGET/"
        echo "== offline recordings ready (GitHub release): $TARGET"
        exit 0
    fi
    echo "== GitHub release recordings unavailable or invalid — falling back to Hugging Face"
fi

# ---- source 2: Hugging Face weights repo ------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== downloading 5-minute recordings from $MODELS_REPO (~280 MB)"
mkdir -p "$TARGET"
HF_TOKEN="${HF_TOKEN:-}" uv tool run --from 'huggingface_hub[cli]' \
    hf download "$MODELS_REPO" --include 'offline_recordings/*' \
    --local-dir "$WORK/hf"
echo "== verifying manifest sha256 pins"
verify_recordings "$WORK/hf/offline_recordings"
# Publish only after every recording matches the reviewed manifest.
cp -r "$WORK/hf/offline_recordings/." "$TARGET/"
echo "== offline recordings ready: $TARGET"
