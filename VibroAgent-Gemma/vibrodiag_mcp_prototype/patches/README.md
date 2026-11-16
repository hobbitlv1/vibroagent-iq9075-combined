# GenieX 0.11.0 patches used by VibroAgent-Gemma

`setup_geniex.sh` clones Qualcomm GenieX at tag `v0.11.0`, checks that the checkout is commit `ae375185109277a2a019b37c037f65bd6e47578c`, initializes its pinned `llama.cpp` submodule, and applies these patches in order:

1. `0001-feat-add-stateful-external-embedding-prefill.patch` adds a bounded stateful decode API to the C ABI, llama.cpp plugin, and Python binding. It accepts either token IDs or caller-supplied float embedding rows, retains the resulting KV state, and generates from that state. VibroAgent-Gemma uses this to place six groups of 14 encoder rows directly into Gemma's input sequence.
2. `0002-feat-read-state-logits.patch` exposes selected logits from the latest stateful decode without sampling or changing the KV state. The runtime reads the two binary-choice logits for the global decision and each of the five target slots.
3. `0003-fix-logit-validation-and-stop-sequences.patch` validates token/sequence IDs before passing them to the native API and handles stop sequences across generated token boundaries, in both the one-shot and the stateful generation paths.

Run `setup_geniex.sh` to build these changes into the installed runtime; copying the patches alone does not rebuild existing libraries.

## Why the patches are still needed on 0.11.0

GenieX 0.9.0 added caller-supplied embeddings to `generate`, and 0.3.18 added `forward_logits`. Neither covers this deployment:

- `generate(input_embd=...)` takes embeddings as the whole prompt. Text tokens and embedding rows cannot be interleaved in one sequence, and the KV cache is dropped after every embedding call.
- `forward_logits` accepts token IDs only and always starts from an empty cache, so it cannot read logits after embedding rows.

The decision path decodes text and six embedding groups into one retained sequence and reads two choice logits at six positions, which is what patches 1 and 2 provide. If upstream later accepts embedding input in `forward_logits` with a way to select positions and token IDs, the series can be dropped.

## How the build uses them

The build keeps the stock GenieX 0.11.0 shared libraries from the `geniex` and `geniex-llama-cpp` packages, rebuilds only the llama.cpp plugin from the patched source, and adds `libgeniex_stateful.so` as a compatibility bridge that exports the three new calls. The new plugin methods are declared after every upstream virtual method, so the stock core, which only knows the earlier ones, keeps working. Runtime health checks require external-embedding decode and state-logit probes to pass before the web application starts.

## NPU host buffers

GenieX 0.11.0 turns on the Hexagon backend's host buffers, which hand the NPU the model as one buffer of several gigabytes (2.3 GiB for this model). The IQ-9075 cannot map a buffer that large: the load fails with `fastrpc_mmap failed`, the runtime log says "possibly out of IOVA space", and the kernel logs "failed to map buffer". It fails the same way with the multi-session board layout that 0.11.0 picks by itself when `GGML_HEXAGON_NDEV` is unset. The model server therefore sets `GGML_HEXAGON_HOSTBUF=0` before it loads GenieX, unless the variable is already set, and the model is mapped in 64 MiB pieces as it was in 0.10.0. Set `GENIEX_LOG=info` to see the runtime's own log when a load fails.

## Recorded reference logits

`parity_reference.json` holds the decision logits of the two bundled LUMO windows for the `improve-binary-s20261002` model, recorded on the IQ-9075 with the patched 0.4.0, 0.10.0 and 0.11.0 runtimes. With a model server running, compare it against one entry:

```bash
vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python vibrodiag_mcp_prototype/scripts/check_geniex_parity.py --list
vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python vibrodiag_mcp_prototype/scripts/check_geniex_parity.py \
  --reference geniex-0.11.0-cpu-4-threads --bundle models/vibroagent-gemma-improve-binary-s20261002
```

Add `--record <name>` to store the observed logits as a new entry; the GenieX version, device and thread count are taken from the server's `/health`.

What the recorded runs show:

| Runtime | Device, threads | Against 0.4.0 CPU at 8 threads | Demo time per window |
|---|---|---|---:|
| 0.4.0 | CPU, 8 threads on cores 4-7 | reference | 33 s |
| 0.4.0 | CPU, 4 threads on cores 4-7 | same answers, logits differ by up to 1.3 | 31 s |
| 0.4.0 | NPU, 6 threads | same answers, logits differ by up to 5.4 | 14 s |
| 0.10.0 | CPU, 8 threads on cores 0-7 | identical logits | 25 s |
| 0.10.0 | CPU, 8 threads on cores 4-7 | identical logits | 113 s |
| 0.10.0 | CPU, 4 threads on cores 4-7 | identical to 0.4.0 at 4 threads | 26 s |
| 0.10.0 | NPU, 2 or 6 threads | same answers, logits differ by up to 3.0 | 14 s |
| 0.11.0 | CPU, 8 threads on cores 0-7 | same answers, logits differ by up to 2.2 | 16 s |
| 0.11.0 | CPU, 8 threads on cores 4-7 | identical to 0.11.0 at 8 threads on cores 0-7 | 106 s |
| 0.11.0 | CPU, 4 threads on cores 4-7 | same answers, logits differ by up to 1.9 | 17 s |
| 0.11.0 | NPU, 2 threads on cores 6-7 | same answers, logits differ by up to 3.1 | 12 s |

Releases 0.4.0 and 0.10.0 give the same CPU bits at the same thread count. 0.11.0 does not: its llama.cpp evaluates Gemma 4 differently and more than twice as fast, so its CPU logits differ from the earlier releases by up to 2.2, with every answer on the bundled windows unchanged. Within 0.11.0, 4 and 8 threads differ by at most 0.4, and each setting repeats its own logits exactly. On the NPU, 0.11.0 gives the same answers as 0.10.0 with logits up to 0.7 apart, and repeats its own logits exactly, also across a restart.

The demo time is the wall time of `demo/run_lumo.py` for one window. It includes that client's start-up and encoding, about 10 s on two cores. The model server's own latency per window:

| Runtime | NPU | CPU, 4 threads on cores 4-7 | CPU, 8 threads on cores 0-7 |
|---|---:|---:|---:|
| 0.10.0 | 3.3 s | 16 s | not measured |
| 0.11.0 | 2.2 s | 7.3 s | 6 s |

GenieX starts one thread per machine core unless told otherwise, and its CPU backend waits by spinning, which is why 8 threads on 4 pinned cores take over 100 s. The model server therefore passes an explicit count: `VIBROGEMMA_N_THREADS`, or by default one thread per core it is pinned to (`VIBROGEMMA_INFERENCE_CPUS`). The 0.4.0 and 0.10.0 CPU logits cannot be reproduced exactly on 0.11.0 at any thread count.

## Moving to another GenieX release

The series is written against one exact release. To move it, port the three patches to the new tag, then change together: `GENIEX_VERSION` and the pinned commit in `setup_geniex.sh`, the default paths in `vibroagent.sh`, `REQUIRED_GENIEX_VERSION` in `vibrogemma_geniex_server.py`, the expected version in `scripts/check_vibrogemma_health.py`, and the `geniex` pins in `constraints/gemma.in` (then recompile `gemma.txt`). Check `sizeof` of the new structs against the Python definitions, because they embed `geniex_ProfileData`, whose size changes between releases. Read the release's changes to the Hexagon backend defaults as well: 0.11.0 turned on host buffers, which this board cannot map. Before deploying, run `scripts/check_geniex_parity.py` against the previous runtime's entry at the same thread count on the CPU and on the NPU, and add entries for the new runtime with `--record`.

The patched GenieX-derived files remain under Qualcomm's BSD 3-Clause licence; see `GENIEX_LICENSE`.
