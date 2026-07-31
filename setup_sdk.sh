#!/usr/bin/env bash
# setup_sdk.sh — materialize the STDATALOG-PYSDK packages this project runs on.
#
# The stock ST SDK is NOT versioned in this repository (it is ST's code, available
# from ST's GitHub). This script:
#   1. fetches stdatalog_core and stdatalog_pnpl at the exact v1.3.0 release
#      commits the project was built against,
#   2. copies the VibroAgent patch overlay (sdk_patches/overlay/) on top.
#
# After it finishes, vibroagent.sh finds both packages at the repo root, exactly
# where its SDK_PYTHONPATH expects them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Commits pinned by the stdatalog-pysdk v1.3.0 umbrella release
# (https://github.com/STMicroelectronics/stdatalog-pysdk, tag v1.3.0).
CORE_SHA=a4824fcdf8c670e457df8f87ec1598d2e73476f8
PNPL_SHA=a6e36d29d9e8b463ef6c4b91f66a2561612bfb46
# dtk and gui remain at the versions pinned by the umbrella SDK release.
DTK_SHA=46ceea78c149f6c2c8d20ae887cd8bf57106ee22
GUI_SHA=708799237b504326ddad8f52227d2ae145028eee

fetch_pinned() {
    local name="$1" sha="$2" dest="$REPO/$1"
    if [ -f "$dest/.stock_sha" ] && [ "$(cat "$dest/.stock_sha")" = "$sha" ]; then
        echo "== $name already present at $sha — skipping fetch"
        return
    fi
    echo "== fetching $name @ $sha"
    rm -rf "$dest"
    mkdir -p "$dest"
    git -C "$dest" init -q
    git -C "$dest" remote add origin "https://github.com/STMicroelectronics/$name.git"
    git -C "$dest" fetch -q --depth 1 origin "$sha"
    git -C "$dest" checkout -q FETCH_HEAD
    rm -rf "$dest/.git"
    echo "$sha" > "$dest/.stock_sha"
}

fetch_pinned stdatalog_core "$CORE_SHA"
fetch_pinned stdatalog_pnpl "$PNPL_SHA"
fetch_pinned stdatalog_dtk  "$DTK_SHA"
fetch_pinned stdatalog_gui  "$GUI_SHA"

echo "== applying VibroAgent SDK patches (sdk_patches/overlay/)"
cp -r "$REPO/sdk_patches/overlay/." "$REPO/"

# ST ships an EMPTY __init__.py at each package repo's top level (a packaging
# stub). With the repo root as the working directory — which is where the
# launchers run — that stub makes the OUTER folder shadow the real inner
# package on sys.path, so `import stdatalog_core.HSD...` fails in every
# service and board-reader worker. Removing the stubs demotes the outer
# folders to namespace portions, which Python skips in favor of the real
# packages found via PYTHONPATH / the venv's vibroagent_sdk.pth.
rm -f "$REPO/stdatalog_core/__init__.py" "$REPO/stdatalog_pnpl/__init__.py" \
      "$REPO/stdatalog_dtk/__init__.py"  "$REPO/stdatalog_gui/__init__.py"

echo "== done: stdatalog_core + stdatalog_pnpl = stock v1.3.0 + VibroAgent patches;"
echo "   stdatalog_dtk + stdatalog_gui = stock v1.3.0 (unpatched, needed on PYTHONPATH)."
echo "   See sdk_patches/README.md for what the patches change and why."
