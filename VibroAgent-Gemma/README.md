# VibroAgent-Gemma

VibroAgent-Gemma is the continuous-encoder deployment of VibroAgent for the Qualcomm IQ-9075. It preserves the six-board STWIN.box acquisition, live or recorded monitor, spectrum, replay, and chat interface, but replaces the older discrete codec/Qwen decision path with a vibration encoder and Gemma.

> **Research status:** the included G1 checkpoint is explicitly **unpromoted** because its validation gate failed. Its model card reports affected-target F1 `0.4701` and single-target exact-set match `0.3393`. This package makes that checkpoint reproducibly deployable for evaluation and demonstrations; it does not promote it to a production or safety model.

See [../DEPLOYMENT_PATHS.md](../DEPLOYMENT_PATHS.md) for the side-by-side comparison with the older VibroAgent-Codec deployment.

## Requirements

- Qualcomm IQ-9075 Linux system with working Hexagon HTP access
- Python 3.12, Git, `g++`, `curl`, CMake/Ninja (installed into the environment where possible), and at least 12 GB free storage during split-part reconstruction
- For live mode: six STEVAL-STWINBX1 boards running FP-SNS-DATALOG2 v3.2.0 and a powered USB hub
- For offline mode: no boards or USB setup; the six recorded `.dat` inputs are already included
- Access to the combined repository's [VibroAgent-Gemma weight release](https://github.com/hobbitlv1/vibroagent-iq9075-combined/releases): authenticate with `gh auth login` or set `GITHUB_TOKEN` when the assets are private

## Path 1: live six-board deployment

From the combined repository root:

```bash
cd VibroAgent-Gemma
./setup.sh --live
```

If USB setup added your account to `hsdatalog`, log out and back in or reboot. Connect and power-cycle all six boards, then run:

```bash
./vibroagent.sh start
./vibroagent.sh status
```

The launcher prints the LAN URL for the web interface. The model endpoint remains loopback-only at `http://127.0.0.1:18181/v1`.

Stop cleanly so every board receives `stop_log`:

```bash
./vibroagent.sh stop
```

## Path 2: offline web pipeline

This mode starts the full web application and the same encoder + model path without any boards or USB logger:

```bash
cd VibroAgent-Gemma
./setup.sh --offline
./vibroagent.sh start
```

The monitor replays the six included `.dat` acquisitions under the combined repository's `examples/` directory. At replay time `15 s` it overlays the bundled LUMO `DAM4_010` window for target 3; at `45 s` it overlays `DAM6_010` for target 5. The inference claim, six-board read, overlay, encoder input, and model verdict all use that manifest's exact 10-second start time. The scheduler sleeps to the next event instead of continuously running the model.

The `.dat` files are never edited; the LUMO signal is a runtime-only overlay. Both events repeat when the 60-second replay loops. Change speed or looping without changing the recordings:

```bash
VIBRO_REPLAY_SPEED=2 VIBRO_REPLAY_LOOP=0 ./vibroagent.sh start
```

Stop with `./vibroagent.sh stop`. Run `./setup.sh --live` to return to physical boards.

## Path 3: direct board-free LUMO demo

The demo uses the same encoder, continuous soft-token path, Q8 GGUF, structured decoder, and validation code as live inference; only the six-board input is replaced by included 10-second LUMO windows.

```bash
cd VibroAgent-Gemma
./setup.sh --demo
./vibroagent.sh demo target_3
./vibroagent.sh demo target_5
```

`target_3` uses the `DAM4_010` window and `target_5` uses the `DAM6_010` window. To verify the encoder without loading the model:

```bash
vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python \
  demo/run_lumo.py --target target_3 --dry-run
```

The compact source windows and their checksums are versioned under `vibrodiag_mcp_prototype/data/external/lumo/`. Their source, licence, channel mapping, and transformations are documented in [demo/LUMO_ATTRIBUTION.md](demo/LUMO_ATTRIBUTION.md).

In live mode the two UI injection buttons use these same windows as moving, amplitude-matched test overlays for target 3 or target 5. Manual live tests are marked synthetic and are not registered as Replay alerts. Offline scheduled events are identified separately and may be saved in Replay because their timestamp and source are manifest-bound.

## Model assets

The repository includes the pinned encoder, tokenizer, pipeline source, healthy profile, schemas, and manifests under `models/vibroagent-gemma-g1/`. The large Q8 GGUF is downloaded from the GitHub release as lexically ordered split parts and reconstructed byte-for-byte:

```text
gemma-4-e2b-g1-Q8_0.gguf
SHA-256: 11ceefee8d62080072fe2b65f68beab5e0f31ad716c640178ed93b5ac0eb31d6
```

The default release tag is `weights-v1` in `hobbitlv1/vibroagent-iq9075-combined`. The Q8 file is published as three sub-2 GB parts plus `models/gemma_release_assets_manifest.json`. Setup requires the downloaded manifest to match the checked-in copy, verifies every part's size and SHA-256, concatenates in manifest order, then verifies the reconstructed file. Override the tag with `VIBRO_GEMMA_RELEASE_TAG`. To install an already downloaded copy:

```bash
VIBROGEMMA_GGUF_SOURCE=/path/to/gemma-4-e2b-g1-Q8_0.gguf ./setup_models.sh
```

No model or encoder artifact is served until its manifest-bound SHA-256 checks pass.

## Runtime path

1. Live mode writes one IIS3DWB stream per physical board through the patched STDATALOG-PYSDK native callback path. Offline mode starts no logger and maps the same six logical slots to immutable recordings.
2. The monitor reads a synchronized 10-second XYZ episode: the latest complete live episode, or the exact scheduled offline window. Stale, missing, flat, clipped, or collapsed routing remains a deterministic data-quality failure.
3. The frozen ONNX encoder projects the six boards into `84 × 1536` continuous embeddings: 14 rows for the reference and 14 for each of five targets.
4. The GenieX 0.4.0 patch inserts those rows at the model's reserved vibration positions. They are attended as native continuous context; acceleration arrays are not converted into prompt text.
5. The Q8 Gemma checkpoint runs on `llama_cpp:HTP0` with a 4,096-token context. KV state is rebuilt for each classification episode.
6. The structured decoder reads binary logits for one global state and one joint five-target set. Its supported decision labels are `normal`, `unknown_anomaly`, and the deterministic quality class `data_invalid`; operational severities are `none` and `advisory`.
7. The web application validates the complete response before showing live board states, popups, spectrum, replay, or chat context.

The five target records are independent: a global anomaly does not imply that all targets are affected. Localization remains experimental at the validation results above. This is monitoring/triage output, not a certified structural-safety assessment.

## GenieX patch

The complete, reviewable patch series is under `vibrodiag_mcp_prototype/patches/`; [its README](vibrodiag_mcp_prototype/patches/README.md) explains every changed API. `setup_geniex.sh` applies the patches to an exact GenieX `v0.4.0` checkout and builds the patched plugin plus compatibility bridge.

## Verification

Static and focused runtime checks:

```bash
bash -n setup.sh setup_models.sh setup_geniex.sh vibroagent.sh
PYTHONPATH=vibrodiag_mcp_prototype/src:vibrodiag_mcp_prototype \
  vibrodiag_mcp_prototype/.run/vibrogemma-venv/bin/python -m pytest -q \
  vibrodiag_mcp_prototype/tests/test_vibrogemma_live.py \
  vibrodiag_mcp_prototype/tests/test_vibrogemma_offline.py \
  vibrodiag_mcp_prototype/tests/test_vibrogemma_geniex_server.py \
  vibrodiag_mcp_prototype/tests/test_vibrogemma_deployment_health.py
```

Logs and PID files are kept under `vibrodiag_mcp_prototype/.run/` and `.run/`:

```bash
tail -f .run/geniex.log .run/webchat.log .run/logger.log
```
