# VibroAgent deployment paths

This repository intentionally keeps two independently deployable implementations. They share the same six-board physical installation, STDATALOG-PYSDK patch set, sensor identity contract, spectrum tooling, and safety boundaries. They differ at the signal representation and model-decision layers.

| Area | VibroAgent-Codec (older, repository root) | VibroAgent-Gemma (`VibroAgent-Gemma/`) |
|---|---|---|
| Input window | One configured axis, 10 s per board | Synchronized XYZ, 10 s per board |
| Signal representation | Residual-VQ codec at 400 Hz; 125 frames × 2 codebooks = 250 discrete symbols per sensor | Frozen ONNX vibration encoder; 14 continuous rows per board = 84 rows of width 1,536 |
| What enters the language model | Text prompt containing discrete code tokens and one relative-level value per target | Continuous encoder embeddings inserted directly at six reserved vibration regions inside Gemma's sequence |
| Model | Fine-tuned Qwen3-4B-Instruct-2507, Q4_0 with q8_0 embeddings | Fine-tuned Gemma 4 E2B, Q8_0 |
| HTP placement | `llama_cpp:HTP0,HTP1`, two sessions | `llama_cpp:HTP0`, one session |
| Context | 6,144 tokens | 4,096 tokens |
| GenieX | Stock packaged runtime | GenieX 0.4.0 plus the included external-embedding and state-logit patches |
| Decision method | One grammar-constrained JSON generation using the trained codes prompt | Binary global decision plus a joint five-target set read from selected model logits |
| Model labels | Four relative sensor labels and five network labels | `normal` or `unknown_anomaly`; deterministic quality label `data_invalid`; severities `none` or `advisory` |
| Localization | `affected_sensor_ids` generated in the model response | Exactly five target records; each target is independently affected or normal, so global and local states are not conflated |
| Healthy reference | Recorded/reference window plus relative level | Reference board inside the synchronized episode plus the bundle's frozen healthy feature profile |
| Live UI | Waveforms, PSD, chat, history, replay, and model-authored popup | Desktop live monitor, spectrum, replay, chat, validated target states, and test-only LUMO overlay controls |
| Board-free path | Five-minute immutable STDATALOG replay plus a smaller codec self-test | Included 60-second immutable `.dat` web replay with scheduled LUMO overlays, plus two direct 10-second LUMO checks |
| Large weight | 2.46 GB Q4 GGUF | 4.95 GB Q8 GGUF |

## When to use each

Use **VibroAgent-Codec** when you need the already published, most compact deployment; its immutable five-minute replay; compatibility with the established codes_v3 label vocabulary; or a codec-only demonstration on a non-Qualcomm Linux computer.

Use **VibroAgent-Gemma** to evaluate native continuous vibration ingestion, XYZ preprocessing, the newer binary anomaly contract, or explicit five-target localization. The included checkpoint is research-only and unpromoted: its validation gate failed, with affected-target F1 `0.4701` and single-target exact-set match `0.3393`. Its full verdict demo requires the patched GenieX path and therefore targets the IQ-9075 deployment environment.

Neither path converts raw acceleration arrays to language-model text, exposes raw arrays outside the device, identifies a physical damage mechanism, or provides a certified structural-safety conclusion.

## Deploy VibroAgent-Codec

Live boards:

```bash
./setup.sh
./vibroagent.sh start
```

Immutable recorded demo:

```bash
./setup.sh --offline
./vibroagent.sh start
```

The complete guide remains in [README.md](README.md).

## Deploy VibroAgent-Gemma

Live boards:

```bash
cd VibroAgent-Gemma
./setup.sh --live
./vibroagent.sh start
```

Offline web pipeline (LUMO target 3 at 15 s; target 5 at 45 s):

```bash
cd VibroAgent-Gemma
./setup.sh --offline
./vibroagent.sh start
```

Direct LUMO demo:

```bash
cd VibroAgent-Gemma
./setup.sh --demo
./vibroagent.sh demo target_3
./vibroagent.sh demo target_5
```

The complete guide is [VibroAgent-Gemma/README.md](VibroAgent-Gemma/README.md).

The two live stacks use the same USB fleet, acquisition folders, web port, and model port. Stop one stack cleanly before starting the other.

## Shared acquisition foundation

Both paths call the repository-root `setup_sdk.sh`. It fetches STDATALOG-PYSDK v1.3.0 at pinned upstream commits and applies `sdk_patches/overlay/`. The patch exposes native data-ready callbacks, prevents silent serial fallback, returns acquisition threads for clean shutdown, bounds live decoder allocation, and includes the STWIN.box device-template cache. The current callback implementation also retains the native component-name buffer for the full callback lifetime and routes data by the verified registered device identity.

VibroAgent-Gemma adds no second SDK fork: its `stdatalog_*` and `stdatalog_examples` entries are links to this same reviewed root implementation.
