# G1 verification and possible deployment

The validation gate failed: stop at offline research/replay; production deployment is prohibited.

## 1. Verify the delivery

Verify the ZIP checksum before extraction, then verify every extracted payload against
`PACKAGE_MANIFEST.json`. The manifest contains the selected-checkpoint identities, exact B0/config/
shared-input bindings, frozen-encoder tensor proof, and SHA-256 for every payload member. Reject the
delivery if any member is missing, extra, or changed.

## 2. Prepare the software and base model

From the extracted package root:

```bash
python -m venv .venv-g1
source .venv-g1/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './code[live]'
```

The package intentionally excludes raw datasets, caches, credentials, and the Gemma base weights.
Obtain authorized access to `google/gemma-4-E2B-it` at the exact revision
recorded in `configs/g1-portable-config.yaml`. Do not copy tokens into configuration files.

## 3. Use the selected checkpoint; export is optional

The portable configuration preserves the exact model/loss/output contract, points to the packaged
TRAIN-only healthy profile, and documents that training/evaluation are unauthorized from this
delivery. For the native runtime, set `model.checkpoint` directly to
`checkpoints/lora/epoch-002` so the loader sees the checkpoint-root `adapter/` directory.

An optional offline/export copy can be created with:

```bash
source .venv-g1/bin/activate
python -m vibrogemma.cli export   --config configs/g1-portable-config.yaml   --checkpoint checkpoints/lora/epoch-002   --output-dir models/g1   --keep-adapter   --no-onnx
```

Run the optional ONNX export only after the native export succeeds and validate its generated ONNX
receipt. Do not point the current native runtime at `models/g1`: the exporter stores PEFT under a
different subdirectory than that runtime loader expects. Keep board 0 as the fixed reference and
boards 1-5 in the trained physical zone order.

## 4. Site integration and acceptance

The packaged `code/src/vibrogemma/live/coordinator.py` includes the runtime-parity fix that forwards
G1 `training.language_loss`, `training.structured_set_training`, full `generation`, and
`final_protocol` controls into model construction. Its focused regression test is packaged at
`code/tests/test_live_runtime_config_parity.py`. Treat either file changing without a matching
manifest update as a failed delivery.

Create a site runtime configuration from the repository runtime schema, set `model.checkpoint` to the
packaged selected LoRA checkpoint, and preserve the architecture, marker text, labels, loss/decoding
controls, and system prompt from the portable G1 configuration. Supply the real sensor geometry,
units, sample rates, orientation, and
synchronization evidence. Missing axes must remain masked. Cross-board phase evidence must remain
disabled until synchronization is measured and documented.

First run an offline six-board replay:

```bash
python -m vibrogemma.cli replay --config configs/runtime-site.yaml --input SITE_REPLAY.npz --fast
```

Parse the replay JSON and require this exact assertion before interpreting any result:

```text
result.model.decision_mode == parallel_binary_global_plus_joint_target_set_v1
```

Abort if it differs. This proves the runtime retained G1 joint target-set decoding rather than
silently falling back to independent or legacy decoding.

## 5. Current runtime limitations

- The current alert policy votes only from target outputs. A global `unknown_anomaly` combined with
  an all-zero/unaffected target set does **not** trigger an alert. Monitor the stored global decision
  separately or fix and validate the policy before operational use.
- The packaged healthy profile was fitted from training-corpus provenance. A live/site episode whose
  dataset/acquisition provenance does not match those keys may receive no healthy-profile residual;
  global profile fallback is intentionally disabled. Fit and validate a TRAIN/site-baseline profile
  with matching provenance rather than enabling fallback silently.
- The bundled FastAPI service provides no authentication or TLS. Keep it on localhost, or place it
  behind an authenticated TLS reverse proxy with network access controls.
- G1 is a research artifact. When its validation gate fails it is explicitly unpromoted; even a gate
  pass would still require sealed independent testing and site acceptance before deployment.

Before any alerting, require an untouched, building-independent evaluation; healthy site calibration;
normal/anomaly and every target-zone support; false-alert, persistence, clock-skew, packet-loss, and
sensor-failure tests; and human engineering review. Only after those gates pass should `live` and
`serve` be considered. Preserve `normal`, `unknown_anomaly`, and deterministic `data_invalid`
semantics; do not add a side classifier.
