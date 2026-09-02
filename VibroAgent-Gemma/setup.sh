#!/usr/bin/env bash
# Install VibroAgent-Gemma for live six-board use or the board-free LUMO demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINED_ROOT="$(cd "$ROOT/.." && pwd)"
MODE=live
for arg in "$@"; do
    case "$arg" in
        --live) MODE=live ;;
        --demo) MODE=demo ;;
        *) echo "usage: $0 [--live|--demo]" >&2; exit 2 ;;
    esac
done
echo "$MODE" > "$ROOT/.vibro_mode"

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$MODE" = live ]; then
    echo "== preparing USB and pinned STDATALOG-PYSDK v1.3.0"
    "$COMBINED_ROOT/setup_usb.sh"
    "$COMBINED_ROOT/setup_sdk.sh"
else
    echo "== demo mode: USB and STDATALOG setup skipped"
fi

"$ROOT/setup_models.sh"
"$ROOT/setup_geniex.sh"

if [ "$MODE" = live ]; then
    VENV="$ROOT/vibrodiag_mcp_prototype/.run/vibrogemma-venv"
    SDK_DEPS="$(sed -n '/install_requires=\[/,/\]/p' "$COMBINED_ROOT/stdatalog_core/setup.py" \
        | grep -oE '"[^"]+"' | tr -d '"' | grep -v '^stdatalog_')"
    # shellcheck disable=SC2086
    uv pip install --quiet --python "$VENV/bin/python" $SDK_DEPS
    SITE_PKGS="$($VENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' \
        "$COMBINED_ROOT/stdatalog_core" \
        "$COMBINED_ROOT/stdatalog_pnpl" \
        "$COMBINED_ROOT/stdatalog_dtk" \
        "$COMBINED_ROOT/stdatalog_gui" > "$SITE_PKGS/vibroagent_sdk.pth"
fi

echo "== VibroAgent-Gemma setup complete ($MODE)"
if [ "$MODE" = live ]; then
    echo "Next: ./vibroagent.sh start"
else
    echo "Next: ./vibroagent.sh demo target_3   # or target_5"
fi
