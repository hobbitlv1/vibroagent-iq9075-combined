# Fine-tuning Qwen3 4B Instruct for the VibroAgent all-sensor agent

The runtime behavior is now a single autonomous LLM pass over all sensors. The model receives one compact prompt containing the reference sensor, every target sensor, relative metrics, and the deterministic precheck labels. It must return one strict JSON object with per-sensor decisions and the overall network decision.

The fine-tuning scripts train that behavior directly. They do not train the removed separate `small_sensor_agent` and `main_building_agent` prompt flow, and they do not train on raw waveforms.

## Runtime output schema

The assistant response must match this shape:

```json
{
  "sensor_reports": [
    {
      "sensor_id": "target_1",
      "local_status": "normal_relative_to_baseline",
      "confidence": 0.82,
      "evidence": ["short numeric reason"]
    }
  ],
  "network_status": "normal_relative_to_reference_sensor",
  "confidence": 0.84,
  "affected_sensor_ids": [],
  "evidence": ["short network reason"],
  "explanation": "short monitoring triage sentence"
}
```

Allowed sensor labels:

```text
normal_relative_to_baseline
mild_deviation
significant_local_deviation
impulsive_or_shock_event
possible_sensor_issue
insufficient_data
```

Allowed network labels:

```text
normal_relative_to_reference_sensor
mild_local_deviation
mild_multi_sensor_deviation
localized_vibration_deviation
multi_sensor_building_wide_vibration_event
sensor_network_quality_issue
insufficient_data
```

## Compact features used

The SFT builder uses the same compact metrics as `AutonomousLlmSensorAgent`:

```text
acceleration_rms_g
acceleration_peak_g
acceleration_peak_to_peak_g
crest_factor
kurtosis
estimated_ppv_mm_s
dominant_frequency_hz
band_powers
quality_flags
```

Baseline comparison fields:

```text
rms_ratio
peak_ratio
peak_to_peak_ratio
ppv_ratio
dominant_frequency_delta_hz
spectral_distance
sensor_quality_flags
baseline_quality_flags
```

## Install dependencies

From `vibrodiag_mcp_prototype/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-finetune.txt
```

## Build new-behavior SFT JSONL

```bash
python scripts/prepare_building_vibration_sft.py \
  --config config/vibration_sft_datasets.example.yaml \
  --output-dir data/vibration_sft \
  --window-duration-s 10 \
  --stride-s 10 \
  --all-sensor-examples 200
```

Outputs:

```text
data/vibration_sft/train.jsonl
data/vibration_sft/validation.jsonl
data/vibration_sft/test.jsonl
data/vibration_sft/all.jsonl
data/vibration_sft/metrics_records.jsonl
data/vibration_sft/dataset_summary.json
```

Each JSONL row now has `task: "all_sensor_agent"`. The user message is the production-style all-sensor prompt. The assistant message contains `sensor_reports`, `network_status`, `affected_sensor_ids`, `evidence`, and `explanation`.

The old `--network-examples` flag is kept only as a deprecated alias for `--all-sensor-examples`. Do not reuse previously generated JSONL from the old behavior; regenerate it with the updated builder.

## Fine-tune Qwen3 4B Instruct with LoRA/QLoRA

For an NVIDIA GPU:

```bash
python scripts/finetune_qwen3_building_vibration_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-jsonl data/vibration_sft/train.jsonl \
  --eval-jsonl data/vibration_sft/validation.jsonl \
  --output-dir outputs/qwen3_4b_vibration_lora \
  --max-length 4096 \
  --epochs 3 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --load-in-4bit
```

For CPU or Mac smoke tests, disable 4-bit loading and use fewer examples/epochs:

```bash
python scripts/finetune_qwen3_building_vibration_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --train-jsonl data/vibration_sft/train.jsonl \
  --eval-jsonl data/vibration_sft/validation.jsonl \
  --output-dir outputs/qwen3_4b_vibration_lora \
  --no-load-in-4bit \
  --epochs 1
```

The trainer still consumes generic TRL chat JSONL rows, so the trainer script itself does not need behavior-specific changes.

## Evaluate the adapter

```bash
python scripts/evaluate_vibration_sft_jsonl.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/qwen3_4b_vibration_lora \
  --test-jsonl data/vibration_sft/test.jsonl \
  --output-jsonl outputs/vibration_eval_predictions.jsonl \
  --limit 200 \
  --load-in-4bit
```

The evaluator now reports exact-example accuracy plus separate per-sensor label accuracy and network-label accuracy.

## Connecting the adapter back to runtime

The runtime calls an OpenAI-compatible endpoint through `ModelClient`. Serve the base model plus adapter with your preferred server that supports PEFT adapters, or merge the adapter into the model and serve the merged model.

The production prompt is generated in `src/vibroagent_mcp/llm_sensor_agent.py` by `_build_all_sensor_prompt()`. The SFT builder imports that helper so prompt shape stays aligned.

## Important notes

- Use the non-thinking Qwen3 model for this classification task: `Qwen/Qwen3-4B-Instruct-2507`.
- Train only on the fixed building sensor and network labels defined by the application schema.
- The model output is monitoring triage, not a certified structural safety assessment.
- FRF data are augmentation only and are marked with `quality_flags: ["frf_derived_features"]`.
- Keep an external dataset for test-only validation, for example LANL, IASC-ASCE, or your own held-out VibroAgent exports.
