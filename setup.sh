#!/usr/bin/env bash
# setup.sh — one-command bootstrap for a fresh clone.
#
#   ./setup.sh            everything needed for the default (codes / GenieX) stack:
#                           1) Linux USB prerequisites: libusb, hsdatalog udev
#                              rules + group                  -> setup_usb.sh
#                           2) stdatalog-pysdk v1.3.0 (pinned commits) + VibroAgent
#                              patches                        -> setup_sdk.sh
#                           3) app virtualenv ./vibroagent-venv + editable install
#                              of vibrodiag_mcp_prototype     (created with uv)
#                           4) codec worker virtualenv ~/codec-cpu-venv
#                              (CPU torch — runs the frozen codec-v1)
#                           5) GenieX runtime + base-model download from
#                              Hugging Face                   -> setup_geniex.sh
#                           6) fine-tuned codes_v3 GGUF, hash-verified
#                                                             -> setup_models.sh
#   ./setup.sh --qairt    all of the above PLUS the 1.8 GB QAIRT Community SDK
#                         (only needed for the legacy MODEL_BACKEND=genie path)
#
# Every step is idempotent — re-running skips what is already in place.
# All virtualenvs and installs use uv (installed automatically if missing).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=live
WANT_QAIRT=0
for arg in "$@"; do
    case "$arg" in
        --offline) MODE=offline ;;
        --qairt)   WANT_QAIRT=1 ;;
        *) echo "unknown option: $arg (supported: --offline --qairt)" >&2; exit 2 ;;
    esac
done
echo "$MODE" > "$REPO/.vibro_mode"
echo "==== data-source mode: $MODE (persisted in .vibro_mode) ===="

if ! command -v uv >/dev/null 2>&1; then
    echo "==== installing uv (https://astral.sh/uv) ===="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$MODE" = "offline" ]; then
    echo "==== [1/6] Linux USB prerequisites — SKIPPED (offline mode, no boards) ===="
else
    echo "==== [1/6] Linux USB prerequisites (libusb, udev rules, hsdatalog group) ===="
    "$REPO/setup_usb.sh"
fi

echo "==== [2/6] STDATALOG-PYSDK (stock v1.3.0 + patches) ===="
"$REPO/setup_sdk.sh"

echo "==== [3/6] app virtualenv (./vibroagent-venv, via uv) ===="
APP_VENV="$REPO/vibroagent-venv"
if [ ! -x "$APP_VENV/bin/python" ]; then
    uv venv "$APP_VENV"
fi
# qwen extra = the OpenAI-compatible client the webchat/agent use to talk to
# the model server — required by the default stack, not optional in practice.
uv pip install --quiet --python "$APP_VENV/bin/python" \
    -e "$REPO/vibrodiag_mcp_prototype[dev,qwen]"
# The ST SDK is imported from the source trees via PYTHONPATH (the launchers
# insist on THIS tree's packages), so it is never pip-installed — install its
# declared dependencies explicitly, except the stdatalog_* packages themselves.
SDK_DEPS="$(sed -n '/install_requires=\[/,/\]/p' "$REPO/stdatalog_core/setup.py" \
    | grep -oE '"[^"]+"' | tr -d '"' | grep -v '^stdatalog_')"
# shellcheck disable=SC2086
uv pip install --quiet --python "$APP_VENV/bin/python" $SDK_DEPS
# Make the patched SDK importable from the venv WITHOUT the launchers'
# PYTHONPATH: a .pth file appends the four source trees to sys.path.
# PYTHONPATH entries still precede site-packages, so the launchers behave
# exactly as before.
SITE_PKGS="$("$APP_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' \
    "$REPO/stdatalog_core" \
    "$REPO/stdatalog_pnpl" \
    "$REPO/stdatalog_dtk" \
    "$REPO/stdatalog_gui" > "$SITE_PKGS/vibroagent_sdk.pth"
echo "app venv ready: $("$APP_VENV/bin/python" --version) at $APP_VENV"
echo "  (SDK importable via $SITE_PKGS/vibroagent_sdk.pth)"

echo "==== [4/6] codec worker virtualenv (~/codec-cpu-venv, CPU torch, via uv) ===="
CODEC_VENV="${VIBRO_CODEC_VENV:-$HOME/codec-cpu-venv}"
if [ ! -x "$CODEC_VENV/bin/python" ]; then
    uv venv "$CODEC_VENV"
fi
uv pip install --quiet --python "$CODEC_VENV/bin/python" numpy scipy
uv pip install --quiet --python "$CODEC_VENV/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cpu
echo "codec venv ready: $("$CODEC_VENV/bin/python" --version) at $CODEC_VENV"

echo "==== [5/6] GenieX runtime + base-model autodownload ===="
"$REPO/setup_geniex.sh"

echo "==== [6/6] fine-tuned codes_v3 GGUF (hash-verified) ===="
"$REPO/setup_models.sh"

if [ "$MODE" = "offline" ]; then
    echo "==== [offline] 5-minute recorded acquisitions (hash-verified) ===="
    "$REPO/setup_recordings.sh"
fi

if [ "$WANT_QAIRT" = "1" ]; then
    echo "==== [optional] QAIRT Community SDK (legacy genie backend) ===="
    "$REPO/setup_qairt.sh"
fi

echo
echo "==== setup complete ===="
echo "Next steps (mode: $MODE):"
if [ "$MODE" = "offline" ]; then
    echo "  1. Start the stack:   ./vibroagent.sh start"
    echo "     (the replay logger streams the recorded examples/ acquisitions in"
    echo "      real time; the USB logger stays down — no boards needed)"
    echo "  2. Open the webchat:  http://<board-lan-ip>:7860 — graphs, chat and"
    echo "     monitor popups reproduce the recording at its original moments."
    echo "  To switch to live boards later:  ./setup.sh   (reruns USB setup)"
else
    echo "  1. If setup_usb.sh just added you to the hsdatalog group, log out and"
    echo "     back in (or reboot) so the membership applies."
    echo "  2. Start the stack:   ./vibroagent.sh start"
    echo "  3. Open the webchat:  http://<board-lan-ip>:7860"
    echo "  For the board-free demo instead:  ./setup.sh --offline"
fi
echo
echo "No board? Run the recorded-data demo instead:"
echo "  $CODEC_VENV/bin/python examples/selftest.py"
echo "  $CODEC_VENV/bin/python examples/run_example.py --dry-run"
