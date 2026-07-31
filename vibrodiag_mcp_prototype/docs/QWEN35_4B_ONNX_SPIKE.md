# Qwen3.5-4B ONNX Spike

This is an experimental text-only ONNX Runtime GenAI path for `Qwen/Qwen3.5-4B`.
It does not replace the current Qwen3-4B Genie/QNN NPU bundle.

## Current workspace status

The current machine is not ready for a full export:

- only a few GB of disk are free;
- `torch`, `transformers`, `huggingface_hub`, `onnx`, `onnxscript`, and
  `onnxruntime_genai` are not installed;
- the Hugging Face model weights are about 9.3 GB before export cache and ONNX
  external-data files.

Run the readiness check before downloading anything:

```bash
cd vibrodiag_mcp_prototype
python scripts/qwen35_4b_onnx_readiness.py \
  --output-dir ../models/qwen3_5_4b_onnx_int4_cpu
```

## Export Attempt

Use a machine or mounted volume with at least 35 GB free.

```bash
cd vibrodiag_mcp_prototype
pip install -e '.[qwen,onnx,onnx-export]'

git clone https://github.com/microsoft/onnxruntime-genai.git ../onnxruntime-genai

python ../onnxruntime-genai/src/python/py/models/builder.py \
  --model_name Qwen/Qwen3.5-4B \
  --output ../models/qwen3_5_4b_onnx_int4_cpu \
  --precision int4 \
  --execution_provider cpu \
  --extra_options int4_algo_config=k_quant_linear
```

Notes:

- This is a CPU ONNX Runtime GenAI export first, not a Qualcomm NPU/QNN context
  export.
- The current ONNX GenAI builder recognizes `Qwen3_5ForConditionalGeneration`,
  but the export may still fail on model-specific operators or packaging.
- `Qwen/Qwen3.5-4B` is multimodal; this spike treats it as a text-generation
  model for the VibroAgent chat and JSON agent flows.

## Serve Through the Existing App Boundary

Once the ONNX folder exists and contains `genai_config.json`, start the local
OpenAI-compatible adapter:

```bash
cd vibrodiag_mcp_prototype
python -m vibroagent_mcp.onnx_openai_server \
  --model-path ../models/qwen3_5_4b_onnx_int4_cpu \
  --model-id Qwen/Qwen3.5-4B \
  --execution-provider follow_config \
  --host 127.0.0.1 \
  --port 8920
```

Point VibroAgent at the adapter:

```bash
QWEN_BASE_URL=http://127.0.0.1:8920/v1 \
QWEN_MODEL=Qwen/Qwen3.5-4B \
python -m vibroagent_mcp.webchat_server --host 127.0.0.1 --port 7860
```

The app already talks to OpenAI-compatible endpoints, so no MCP changes are
needed for this experiment.
