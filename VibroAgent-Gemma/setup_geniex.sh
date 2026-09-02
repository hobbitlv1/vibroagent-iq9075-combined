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
for patch in "${PATCHES[@]}"; do
    if git -C "$GENIEX" apply --check "$patch" 2>/dev/null; then
        git -C "$GENIEX" apply "$patch"
    elif ! git -C "$GENIEX" apply --reverse --check "$patch" 2>/dev/null; then
        echo "GenieX is neither clean v0.4.0 nor already patched for $(basename "$patch")" >&2
        exit 1
    fi
done

test -x "$VENV/bin/python" || uv venv --python /usr/bin/python3.12 --system-site-packages "$VENV"
uv pip install --quiet --python "$VENV/bin/python" \
    'numpy==2.2.6' 'scipy>=1.12' 'onnxruntime>=1.17' 'tokenizers>=0.22' \
    'pyyaml>=6' 'jsonschema>=4' cmake ninja 'pytest>=8' \
    'geniex==0.4.0' 'geniex-llama-cpp==0.4.0'
uv pip install --quiet --python "$VENV/bin/python" \
    --index https://download.pytorch.org/whl/cpu 'torch==2.11.0'
uv pip install --quiet --python "$VENV/bin/python" -e "$PROTO[qwen]"

SITE="$($VENV/bin/python -c 'from pathlib import Path; import geniex; print(Path(geniex.__file__).parent / "lib")')"
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
