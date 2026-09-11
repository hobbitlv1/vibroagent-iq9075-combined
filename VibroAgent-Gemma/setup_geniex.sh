#!/usr/bin/env bash
# Build GenieX 0.4.0 with VibroAgent-Gemma external-embedding support.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$ROOT/vibrodiag_mcp_prototype"
RUN="$PROTO/.run"
GENIEX="$RUN/GenieX-v0.4.0"
VENV="$RUN/vibrogemma-venv"
OVERLAY="$RUN/geniex-v0.4.0-vibrogemma"
PATCHES=(
  "$PROTO/patches/0001-feat-add-stateful-external-embedding-prefill.patch"
  "$PROTO/patches/0002-feat-read-state-logits.patch"
  "$PROTO/patches/0003-fix-logit-validation-and-stop-sequences.patch"
)
export UV_CACHE_DIR="${VIBROGEMMA_UV_CACHE:-/tmp/vibroagent-gemma-uv-cache}"

for tool_name in git uv g++; do
    command -v "$tool_name" >/dev/null || { echo "missing tool: $tool_name" >&2; exit 1; }
done
mkdir -p "$RUN"
if [ ! -d "$GENIEX/.git" ]; then
    git clone --branch v0.4.0 --depth 1 https://github.com/qualcomm/GenieX.git "$GENIEX"
fi
git -C "$GENIEX" submodule update --init --recursive third-party/llama.cpp
# Later patches can modify the context added by earlier ones, so checking
# each patch in reverse independently cannot recognize a fully patched tree.
# Compare every valid patch prefix in a disposable index, so an existing
# two-patch installation can receive the third without undoing earlier patches.
applied_patch_count() (
    check_dir="$(mktemp -d "$RUN/.patch-check.XXXXXX")" || return 1
    trap 'rm -rf "$check_dir"' EXIT
    export GIT_INDEX_FILE="$check_dir/index"
    git -C "$GENIEX" read-tree HEAD || return 1
    count=0
    matched_count=-1
    if git -C "$GENIEX" diff --quiet; then matched_count=0; fi
    for patch in "${PATCHES[@]}"; do
        git -C "$GENIEX" apply --cached "$patch" || return 1
        count=$((count + 1))
        if git -C "$GENIEX" diff --quiet; then matched_count="$count"; fi
    done
    [ "$matched_count" -ge 0 ] || return 1
    echo "$matched_count"
)
if ! applied_count="$(applied_patch_count)"; then
    echo "GenieX differs from clean v0.4.0 and every supported patch prefix; preserving local edits." >&2
    exit 1
fi
if [ "$applied_count" -eq "${#PATCHES[@]}" ]; then
    echo "== GenieX patch series already applied"
else
    for patch in "${PATCHES[@]:applied_count}"; do
        git -C "$GENIEX" apply "$patch"
    done
fi

test -x "$VENV/bin/python" || uv venv --python /usr/bin/python3.12 "$VENV"
uv pip install --quiet --python "$VENV/bin/python" \
    'numpy==2.2.6' 'scipy>=1.12' 'onnxruntime>=1.17' 'tokenizers>=0.22' \
    'pyyaml>=6' 'jsonschema>=4' cmake ninja 'pytest>=8' \
    'geniex==0.4.0' 'geniex-llama-cpp==0.4.0'
uv pip install --quiet --python "$VENV/bin/python" \
    --index https://download.pytorch.org/whl/cpu 'torch==2.11.0'
uv pip install --quiet --python "$VENV/bin/python" -e "${PROTO}[qwen]"

SITE="$("$VENV/bin/python" -c 'from pathlib import Path; import geniex; print(Path(geniex.__file__).parent / "lib")')"
OBJECTS="$RUN/geniex-plugin-objects"
mkdir -p "$OVERLAY" "$OBJECTS"
cp -a "$SITE/." "$OVERLAY/"
for source_name in htp_session llm params plugin profiler threadpool vlm; do
    g++ -DGENIEX_SHARED -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_SUBPROCESS \
      -DPROJECT_SOURCE_DIR=\"$GENIEX/sdk\" -Dgeniex_llama_cpp_EXPORTS \
      -I"$GENIEX/sdk/include" -I"$GENIEX/sdk/plugins/llama_cpp/include" \
      -I"$GENIEX/third-party/llama.cpp/include" -I"$GENIEX/third-party/llama.cpp/ggml/include" \
      -I"$GENIEX/third-party/llama.cpp/common" -I"$GENIEX/third-party/llama.cpp/vendor" \
      -I"$GENIEX/sdk/model-manager/include" -I"$GENIEX/third-party/llama.cpp/tools/mtmd" \
      -O3 -DNDEBUG -std=gnu++17 -fPIC -o "$OBJECTS/$source_name.o" \
      -c "$GENIEX/sdk/plugins/llama_cpp/src/$source_name.cpp"
done
g++ -fPIC -O3 -shared -Wl,-soname,libgeniex_plugin.so -Wl,-rpath,'$ORIGIN:$ORIGIN/..' \
  -Wl,--no-as-needed -o "$OVERLAY/llama_cpp/libgeniex_plugin.so" "$OBJECTS"/*.o \
  -L"$OVERLAY/llama_cpp" -L"$OVERLAY" -lllama-common -lgeniex -lmtmd -lllama -lggml -lggml-base
g++ -DGENIEX_SHARED -std=gnu++17 -fPIC -shared -I"$GENIEX/sdk/include" \
  -Wl,-soname,libgeniex_stateful.so -Wl,-rpath,'$ORIGIN' \
  -o "$OVERLAY/libgeniex_stateful.so" "$GENIEX/sdk/compat/llm_stateful_bridge.cpp" \
  -L"$OVERLAY" -lgeniex

echo "== patched GenieX 0.4.0 ready: $OVERLAY"
sha256sum "$OVERLAY/llama_cpp/libgeniex_plugin.so" "$OVERLAY/libgeniex_stateful.so"
