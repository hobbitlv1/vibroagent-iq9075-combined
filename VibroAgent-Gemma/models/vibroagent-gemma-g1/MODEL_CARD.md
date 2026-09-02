# VibroGemma controlled G1 model card

Status: **unpromoted (validation gate failed)**. This is Path 1 only. In this experiment, E1 means encoder-only
pretraining/selection: E1 has no fresh projector or classifier. Path 2 is explicitly
`E1_ENCODER -> EG1 fresh projector/alignment -> EG1 LoRA`, and is not part of this G1 package.

## Model

G1 keeps the original B0 signal architecture: one reference board plus five fixed target boards,
14 soft tokens per board (6
modal, 6 wideband, and
2 physics), encoder width
192, followed by a vibration-to-Gemma projector and
`google/gemma-4-E2B-it`. Gemma choice logits are the only learned classifier. No production
classifier, source, or damage head was added.

The B0 encoder was frozen throughout G1. Exact tensor comparison proved every
`vibration_tokenizer.*` tensor in selected alignment and selected LoRA equals B0
(`203` tensors; digest
`f655f4a438513f4a0a207b5cbcea52c54fb50dd5d8f0838049d8ca1438fe3ee8`). Alignment started with a fresh projector;
LoRA continued that aligned projector and trained the configured Gemma adapter.

## Training and data boundary

- Alignment: 8 complete epochs; LoRA: 4 complete epochs; 7,080 sampled episodes per epoch per stage.
- Checkpoint selection used validation only. Final evaluation used validation only.
- Downstream datasets: lumo, qugs, upc_jacket_data1011, luh_reversible_beam, geisel_library_xk2021, tallwood_prj4359_v2, bncs_nees_2009_0722, sera_aims_prj2992, four_storey_zenodo_11578_v1.
- Unknown-condition/representation-only rows were not assigned fabricated binary or location labels.
- The healthy profile was fitted on TRAIN only and is included as a deployment dependency.
- Localization labels mean physical damage-source zone, not the sensor with the largest response.
- TEST remained sealed and unmaterialized; this package contains no TEST outputs or claims.

## Validation result

| Gate metric | Observed |
| --- | ---: |
| Overall macro-F1 | 0.9265 |
| Normal recall | 0.9416 |
| Anomaly recall | 0.9101 |
| Affected-target F1 | 0.4701 |
| Single-target exact set match | 0.3393 |
| Single-target all-zero rate | 0.2202 |
| Supported target count | fail |
| Minimum per-target F1 | 0.2927 |

The authoritative complete result is `evidence/evaluation-validation/validation_gate.json`.
Selection metrics are development evidence, not an independent estimate of deployment performance.

## Limitations and intended use

This experimental model is intended for structural-vibration research with a synchronized six-board
layout matching the configured reference/target geometry. Sampling rates and genuine measured axes
vary by source; bandwidth and missing-axis masks preserve those limitations rather than inventing
signals. Some added corpora improve representation or global normal/anomaly coverage without adding
genuine five-zone localization positives. Therefore stronger global discrimination does not by itself
prove reliable localization.

Do not use this output as a life-safety system or as the sole basis for structural decisions. It has
not been evaluated on the sealed external TEST collections, calibrated on the target building, or
validated for alert latency/false-alert rates on deployed six-board hardware. If the validation gate
failed, the model is explicitly unpromoted and must not be put into production.
