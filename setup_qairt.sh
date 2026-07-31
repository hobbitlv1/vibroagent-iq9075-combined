#!/usr/bin/env bash
# setup_qairt.sh — download the Qualcomm AI Runtime (QAIRT) Community SDK.
#
# ONLY needed for the legacy `genie` model backend (MODEL_BACKEND=genie and the
# vibroagent.sh launcher). The default GenieX backend does not use it.
#
# Downloads the public QAIRT Community SDK zip (~1.8 GB) from Qualcomm Software
# Center and extracts it to $REPO/v<version>/qairt/<version> — exactly the
# default QAIRT_SDK_ROOT that vibroagent.sh expects. By downloading you accept
# Qualcomm's license terms (https://www.qualcomm.com/site/terms-of-use).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAIRT_VERSION="${QAIRT_VERSION:-2.46.0.260424}"
URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/v${QAIRT_VERSION}.zip"
DEST="$REPO/v${QAIRT_VERSION}"
SDK_ROOT="$DEST/qairt/$QAIRT_VERSION"

if [ -d "$SDK_ROOT" ]; then
    echo "== QAIRT $QAIRT_VERSION already present at $SDK_ROOT — nothing to do"
    exit 0
fi

ZIP="${TMPDIR:-/tmp}/qairt-v${QAIRT_VERSION}.zip"
echo "== downloading QAIRT Community SDK $QAIRT_VERSION (~1.8 GB, resumable)"
curl -fL -A "Mozilla/5.0" -C - -o "$ZIP" "$URL"

echo "== extracting to $DEST"
mkdir -p "$DEST"
unzip -q "$ZIP" -d "$DEST"

if [ ! -d "$SDK_ROOT" ]; then
    echo "ERROR: expected layout qairt/$QAIRT_VERSION not found in the zip" >&2
    exit 1
fi
rm -f "$ZIP"

echo "== done: QAIRT_SDK_ROOT=$SDK_ROOT (vibroagent.sh default)"
echo "   The genie backend additionally needs a Genie-compiled model bundle"
echo "   (GENIE_CONFIG) — see README.md, it is not downloadable."
