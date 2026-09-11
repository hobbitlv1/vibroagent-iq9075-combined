# GenieX 0.4.0 patches used by VibroAgent-Gemma

`setup_geniex.sh` clones Qualcomm GenieX at tag `v0.4.0`, initializes its pinned `llama.cpp` submodule, and applies these patches in order:

1. `0001-feat-add-stateful-external-embedding-prefill.patch` adds a bounded stateful decode API to the C ABI, llama.cpp plugin, and Python binding. It accepts either token IDs or caller-supplied float embedding rows, retains the resulting KV state, and generates from that state. VibroAgent-Gemma uses this to place six groups of 14 encoder rows directly into Gemma's input sequence.
2. `0002-feat-read-state-logits.patch` exposes selected logits from the latest stateful decode without sampling or changing the KV state. The runtime reads the two binary-choice logits for the global decision and each of the five target slots.
3. `0003-fix-logit-validation-and-stop-sequences.patch` validates token/sequence IDs before passing them to the native API and handles stop sequences across generated token boundaries. Its applicability was checked against the combined project's patched GenieX 0.4.0 source. Run `setup_geniex.sh` to build these changes into the installed runtime; copying the patch alone does not rebuild existing libraries.

The build keeps the stock GenieX 0.4.0 shared libraries, rebuilds only the patched llama.cpp plugin, and adds `libgeniex_stateful.so` as a compatibility bridge. Runtime health checks require external-embedding decode and state-logit probes to pass before the web application starts.

The patched GenieX-derived files remain under Qualcomm's BSD 3-Clause licence; see `GENIEX_LICENSE`.
