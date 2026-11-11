#!/usr/bin/env bash
# Build GenieX with VibroAgent-Gemma external-embedding support.
#
# The stock runtime wheels provide the core and the llama.cpp libraries. This script rebuilds
# only the llama.cpp plugin from the patched source and adds libgeniex_stateful.so, which
# exports the stateful decode, generate-from-state, and state-logits calls the stock core lacks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$ROOT/vibrodiag_mcp_prototype"
RUN="$PROTO/.run"
# One release everywhere: source tag, runtime wheels, and the server's required version.
GENIEX_VERSION="0.10.0"
GENIEX_TAG="v$GENIEX_VERSION"
GENIEX="$RUN/GenieX-$GENIEX_TAG"
VENV="$RUN/vibrogemma-venv"
OVERLAY="$RUN/geniex-$GENIEX_TAG-vibrogemma"
PATCHES=(
  "$PROTO/patches/0001-feat-add-stateful-external-embedding-prefill.patch"
  "$PROTO/patches/0002-feat-read-state-logits.patch"
  "$PROTO/patches/0003-fix-logit-validation-and-stop-sequences.patch"
)
# The release is a tag, and a tag can be moved. This source is compiled into the library the
# model server loads, so the checkout must be the exact commit the patch series was written for.
GENIEX_COMMIT="${VIBROGEMMA_GENIEX_COMMIT:-071f609995470bdb0e0769ab964754adfaab7e47}"
# Pinned versions of the full Python dependency closure (compiled from constraints/gemma.in).
CONSTRAINTS="${VIBROGEMMA_CONSTRAINTS:-$ROOT/constraints/gemma.txt}"
# uv installs whatever its cache hands it, so the cache has to belong to this user alone. A
# fixed name under /tmp could be created and seeded by any other account first; the default
# therefore lives in the project's private run directory.
export UV_CACHE_DIR="${VIBROGEMMA_UV_CACHE:-$RUN/uv-cache}"
if [ -e "$UV_CACHE_DIR" ] && { [ ! -O "$UV_CACHE_DIR" ] || [ -n "$(find "$UV_CACHE_DIR" -maxdepth 0 -perm -002 2>/dev/null)" ]; }; then
    echo "refusing uv cache $UV_CACHE_DIR: it must be owned by $(id -un) and not world-writable" >&2
    exit 1
fi

for tool_name in git uv g++; do
    command -v "$tool_name" >/dev/null || { echo "missing tool: $tool_name" >&2; exit 1; }
done
mkdir -p "$RUN"
if [ ! -d "$GENIEX/.git" ]; then
    git clone --branch "$GENIEX_TAG" --depth 1 https://github.com/qualcomm/GenieX.git "$GENIEX"
fi
geniex_head="$(git -C "$GENIEX" rev-parse HEAD)"
if [ "$geniex_head" != "$GENIEX_COMMIT" ]; then
    echo "GenieX checkout is at ${geniex_head:-an unknown commit}, expected $GENIEX_COMMIT (tag $GENIEX_TAG); refusing to build it." >&2
    echo "Remove $GENIEX to fetch it again, or set VIBROGEMMA_GENIEX_COMMIT after reviewing the upstream change." >&2
    exit 1
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
    echo "GenieX differs from clean $GENIEX_TAG and every supported patch prefix; preserving local edits." >&2
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
uv pip install --quiet --python "$VENV/bin/python" --constraint "$CONSTRAINTS" \
    'numpy==2.2.6' 'scipy>=1.12' 'onnxruntime>=1.17' 'tokenizers>=0.22' \
    'pyyaml>=6' 'jsonschema>=4' cmake ninja 'pytest>=8' \
    "geniex==$GENIEX_VERSION" "geniex-llama-cpp==$GENIEX_VERSION"
# Only the torch wheel itself comes from the PyTorch CPU index (--no-deps). The second
# command keeps that wheel and installs its dependencies from PyPI at the pinned versions.
uv pip install --quiet --python "$VENV/bin/python" --constraint "$CONSTRAINTS" --no-deps \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
uv pip install --quiet --python "$VENV/bin/python" --constraint "$CONSTRAINTS" 'torch==2.11.0'
uv pip install --quiet --python "$VENV/bin/python" --constraint "$CONSTRAINTS" -e "${PROTO}[qwen]"

SITE="$("$VENV/bin/python" -c 'from pathlib import Path; import geniex; print(Path(geniex.__file__).parent / "lib")')"
OBJECTS="$RUN/geniex-$GENIEX_TAG-plugin-objects"
mkdir -p "$OVERLAY" "$OBJECTS"
cp -a "$SITE/." "$OVERLAY/"
# Every source of the upstream plugin target (sdk/plugins/llama_cpp/src/*.cpp).
for source_name in decision htp_session llm params plugin profiler threadpool vlm; do
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

echo "== patched GenieX $GENIEX_VERSION ready: $OVERLAY"
sha256sum "$OVERLAY/llama_cpp/libgeniex_plugin.so" "$OVERLAY/libgeniex_stateful.so"
