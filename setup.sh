#!/usr/bin/env bash
# setup.sh — one-command bootstrap for a fresh clone.
#
#   ./setup.sh            everything needed for the default (codes / GenieX) stack:
#                           1) stdatalog-pysdk v1.3.0 (pinned commits) + VibroAgent
#                              patches                        -> setup_sdk.sh
#                           2) app virtualenv ./vibroagent-venv + editable install
#                              of vibrodiag_mcp_prototype     (created with uv)
#                           3) codec worker virtualenv ~/codec-cpu-venv
#                              (CPU torch — runs the frozen codec-v1)
#                           4) GenieX runtime + base-model download from
#                              Hugging Face                   -> setup_geniex.sh
#                           5) fine-tuned codes_v3 GGUF, hash-verified
#                                                             -> setup_models.sh
#   ./setup.sh --qairt    all of the above PLUS the 1.8 GB QAIRT Community SDK
#                         (only needed for the legacy MODEL_BACKEND=genie path)
#
# Every step is idempotent — re-running skips what is already in place.
# All virtualenvs and installs use uv (installed automatically if missing).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "==== installing uv (https://astral.sh/uv) ===="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==== [1/5] STDATALOG-PYSDK (stock v1.3.0 + patches) ===="
"$REPO/setup_sdk.sh"

echo "==== [2/5] app virtualenv (./vibroagent-venv, via uv) ===="
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

echo "==== [3/5] codec worker virtualenv (~/codec-cpu-venv, CPU torch, via uv) ===="
CODEC_VENV="${VIBRO_CODEC_VENV:-$HOME/codec-cpu-venv}"
if [ ! -x "$CODEC_VENV/bin/python" ]; then
    uv venv "$CODEC_VENV"
fi
uv pip install --quiet --python "$CODEC_VENV/bin/python" numpy scipy
uv pip install --quiet --python "$CODEC_VENV/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cpu
echo "codec venv ready: $("$CODEC_VENV/bin/python" --version) at $CODEC_VENV"

echo "==== [4/5] GenieX runtime + base-model autodownload ===="
"$REPO/setup_geniex.sh"

echo "==== [5/5] fine-tuned codes_v3 GGUF (hash-verified) ===="
"$REPO/setup_models.sh"

if [ "${1:-}" = "--qairt" ]; then
    echo "==== [optional] QAIRT Community SDK (legacy genie backend) ===="
    "$REPO/setup_qairt.sh"
fi

echo
echo "==== setup complete ===="
echo "Next steps:"
echo "  1. USB permissions for the STWIN.box boards (ST udev rules / hsdatalog"
echo "     group) — see the linux_setup instructions in ST's stdatalog-pysdk repo."
echo "  2. Start the stack:   ./vibroagent.sh start"
echo "  3. Open the webchat:  http://<board-lan-ip>:7860"
echo
echo "No board? Run the recorded-data demo instead:"
echo "  $CODEC_VENV/bin/python examples/selftest.py"
echo "  $CODEC_VENV/bin/python examples/run_example.py --dry-run"
