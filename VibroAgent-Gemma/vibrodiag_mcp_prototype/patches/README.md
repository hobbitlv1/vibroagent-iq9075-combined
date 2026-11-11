# GenieX 0.10.0 patches used by VibroAgent-Gemma

`setup_geniex.sh` clones Qualcomm GenieX at tag `v0.10.0`, checks that the checkout is commit `071f609995470bdb0e0769ab964754adfaab7e47`, initializes its pinned `llama.cpp` submodule, and applies these patches in order:

1. `0001-feat-add-stateful-external-embedding-prefill.patch` adds a bounded stateful decode API to the C ABI, llama.cpp plugin, and Python binding. It accepts either token IDs or caller-supplied float embedding rows, retains the resulting KV state, and generates from that state. VibroAgent-Gemma uses this to place six groups of 14 encoder rows directly into Gemma's input sequence.
2. `0002-feat-read-state-logits.patch` exposes selected logits from the latest stateful decode without sampling or changing the KV state. The runtime reads the two binary-choice logits for the global decision and each of the five target slots.
3. `0003-fix-logit-validation-and-stop-sequences.patch` validates token/sequence IDs before passing them to the native API and handles stop sequences across generated token boundaries, in both the one-shot and the stateful generation paths.

Run `setup_geniex.sh` to build these changes into the installed runtime; copying the patches alone does not rebuild existing libraries.

## Why the patches are still needed on 0.10.0

GenieX 0.9.0 added caller-supplied embeddings to `generate`, and 0.3.18 added `forward_logits`. Neither covers this deployment:

- `generate(input_embd=...)` takes embeddings as the whole prompt. Text tokens and embedding rows cannot be interleaved in one sequence, and the KV cache is dropped after every embedding call.
- `forward_logits` accepts token IDs only and always starts from an empty cache, so it cannot read logits after embedding rows.

The decision path decodes text and six embedding groups into one retained sequence and reads two choice logits at six positions, which is what patches 1 and 2 provide. If upstream later accepts embedding input in `forward_logits` with a way to select positions and token IDs, the series can be dropped.

## How the build uses them

The build keeps the stock GenieX 0.10.0 shared libraries from the `geniex` and `geniex-llama-cpp` packages, rebuilds only the llama.cpp plugin from the patched source, and adds `libgeniex_stateful.so` as a compatibility bridge that exports the three new calls. The new plugin methods are declared after every upstream virtual method, so the stock core, which only knows the earlier ones, keeps working. Runtime health checks require external-embedding decode and state-logit probes to pass before the web application starts.

## Recorded reference logits

`parity_reference.json` holds the decision logits of the two bundled LUMO windows for the `improve-binary-s20261002` model, recorded on the IQ-9075 with the patched 0.4.0 and 0.10.0 runtimes. With a model server running, compare it against one entry:

```bash
vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python vibrodiag_mcp_prototype/scripts/check_geniex_parity.py --list
vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python vibrodiag_mcp_prototype/scripts/check_geniex_parity.py \
  --reference geniex-0.4.0-cpu-8-threads --bundle models/vibroagent-gemma-improve-binary-s20261002
```

What the recorded runs show:

| Runtime | Device, threads | Against 0.4.0 CPU at 8 threads | Time per window |
|---|---|---|---:|
| 0.4.0 | CPU, 8 threads on cores 4-7 | reference | 33 s |
| 0.10.0 | CPU, 8 threads on cores 0-7 | identical logits | 25 s |
| 0.10.0 | CPU, 8 threads on cores 4-7 | identical logits | 113 s |
| 0.10.0 | CPU, 4 threads on cores 4-7 | same answers, target logits differ by up to 1.3 | 26 s |
| 0.4.0 | CPU, 4 threads on cores 4-7 | identical to 0.10.0 at 4 threads | 31 s |
| 0.4.0 | NPU | same answers, logits differ by up to 5.4 | 14 s |

The logits depend on the thread count, not on the GenieX release: both releases give the same bits at 4 threads and at 8 threads. Between 4 and 8 threads the global answer's logits are unchanged and the five target answers move. The NPU has not been measured on 0.10.0 yet; run the check against the `geniex-0.4.0-npu` entry before relying on it.

GenieX 0.10.0 starts one thread per machine core unless told otherwise, and its CPU backend waits by spinning, which is why 8 threads on 4 pinned cores take 113 s. The model server therefore passes an explicit count: `VIBROGEMMA_N_THREADS`, or by default one thread per core it is pinned to (`VIBROGEMMA_INFERENCE_CPUS`). To reproduce the 0.4.0 CPU numbers exactly, start with `VIBROGEMMA_N_THREADS=8 VIBROGEMMA_INFERENCE_CPUS=0-7`.

## Moving to another GenieX release

The series is written against one exact release. To move it, port the three patches to the new tag, then change together: `GENIEX_VERSION` and the pinned commit in `setup_geniex.sh`, the default paths in `vibroagent.sh`, `REQUIRED_GENIEX_VERSION` in `vibrogemma_geniex_server.py`, the expected version in `scripts/check_vibrogemma_health.py`, and the `geniex` pins in `constraints/gemma.in` (then recompile `gemma.txt`). Check `sizeof` of the new structs against the Python definitions, because they embed `geniex_ProfileData`, whose size changes between releases. Before deploying, run `scripts/check_geniex_parity.py` against the previous runtime's entry at the same thread count, and add an entry for the new runtime.

The patched GenieX-derived files remain under Qualcomm's BSD 3-Clause licence; see `GENIEX_LICENSE`.
