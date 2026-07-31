#!/usr/bin/env bash
# setup_recordings.sh — place the offline-mode recordings (5-minute .dat sets).
#
# Offline mode replays full-length recorded acquisitions; those are too large
# for git (~280 MB for six 5-minute IIS3DWB streams), so this script downloads
# them from the private Hugging Face weights repo into ./recordings/live_* and
# verifies every file against the recording manifest's sha256 pins. The
# replay logger streams from these folders; they are placed ONCE here and
# never modified afterwards.
#
#   HF_TOKEN            read token for the private weights repo; falls back to
#                       ~/.cache/huggingface/token (hf auth login)
#   VIBRO_MODELS_REPO   default hobbitlv/vibroagent-models
#
# Idempotent — a present, fully hash-verified recording set is left untouched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_REPO="${VIBRO_MODELS_REPO:-hobbitlv/vibroagent-models}"
TARGET="$REPO/recordings"

verify_recordings() {
    python3 - "$TARGET" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "recordings_manifest.json"
if not manifest_path.is_file():
    sys.exit(1)
manifest = json.loads(manifest_path.read_text())
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

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== downloading 5-minute recordings from $MODELS_REPO (~280 MB)"
mkdir -p "$TARGET"
HF_TOKEN="${HF_TOKEN:-}" uv tool run --from 'huggingface_hub[cli]' \
    hf download "$MODELS_REPO" --include 'offline_recordings/*' \
    --local-dir "$TARGET/.download"
# flatten offline_recordings/ into ./recordings/
cp -r "$TARGET/.download/offline_recordings/." "$TARGET/"
rm -rf "$TARGET/.download"

echo "== verifying manifest sha256 pins"
verify_recordings
echo "== offline recordings ready: $TARGET"
