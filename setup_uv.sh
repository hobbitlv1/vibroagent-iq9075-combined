#!/usr/bin/env bash
# setup_uv.sh — install uv when it is missing, from a pinned and checksum-verified installer.
#
# The other setup scripts call this instead of piping whatever installer the network serves
# that day into a shell. The installer of one fixed uv release is downloaded to a private
# directory and its SHA-256 is checked before it runs; that installer in turn verifies the
# checksum of the uv binary it downloads.
#
# Overrides (set both together when moving to another release):
#   VIBRO_UV_VERSION            uv release to install
#   VIBRO_UV_INSTALLER_SHA256   SHA-256 of https://astral.sh/uv/<version>/install.sh
set -euo pipefail

UV_VERSION="${VIBRO_UV_VERSION:-0.12.3}"
UV_INSTALLER_SHA256="${VIBRO_UV_INSTALLER_SHA256:-a7e3924ea1cd06bf1518c577d635c624ae2e2db030e0fc8ff8cf426224384e17}"

if command -v uv >/dev/null 2>&1; then
    echo "== uv already installed: $(uv --version)"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
INSTALLER="$WORK/uv-install.sh"

echo "== installing uv $UV_VERSION (checksum-verified installer from https://astral.sh/uv)"
curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$INSTALLER"
if ! echo "$UV_INSTALLER_SHA256  $INSTALLER" | sha256sum -c --quiet - >/dev/null 2>&1; then
    echo "uv installer checksum mismatch for version $UV_VERSION; refusing to run it." >&2
    echo "  expected $UV_INSTALLER_SHA256" >&2
    echo "  got      $(sha256sum "$INSTALLER" 2>/dev/null | cut -d' ' -f1)" >&2
    exit 1
fi
sh "$INSTALLER"
