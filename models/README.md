# Model weights

## codec_v1/ (included in this repository)

| file | sha256 | role |
|------|--------|------|
| `best.pt` (14.8 MiB) | `a80216b639327f9b4e59447138abc025fddac2d6d5eca68448c4ac14c653a350` | Frozen codec-v1 checkpoint, training step 29000. Residual-VQ conv autoencoder: 2 codebooks x 1024 entries, 125 frames per 10 s / 400 Hz window -> 250 discrete codes. |
| `split_manifest.json` | embedded copy must match the checkpoint | Training split provenance; the loader fail-closes on any mismatch. |

The checkpoint is loaded CPU-only, in eval mode, with deterministic
algorithms enforced. `scripts/selftest.py` proves your environment
reproduces the proven live encoder byte-for-byte before you trust any
output.

## codes_v3 LLM (NOT in this repository — 2.4 GB)

| property | value |
|----------|-------|
| file | `qwen3_4b_codes_v3_Q4_0_embq8.gguf` |
| size | 2 463 745 376 bytes |
| sha256 | `3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2` |
| base model | Qwen/Qwen3-4B-Instruct-2507 |
| finetune | LoRA on 11 000 codes_v3 examples (merged), then Q4_0 quantization with q8_0 embeddings |
| training data | public vibration corpora only (fdsn / lumo / rt345 pools); the building's own sensors were never in training |

Distribution: automatic — `./setup_models.sh` downloads it from the private
Hugging Face weights repo (`hobbitlv/vibroagent-models`, `HF_TOKEN` required)
into `models/` and **verifies the sha256 before anything may serve it**:

```bash
HF_TOKEN=hf_... ./setup_models.sh
# == verifying sha256
# models/qwen3_4b_codes_v3_Q4_0_embq8.gguf: OK
```

The GGUF runs anywhere llama.cpp runs (CPU/GPU) and on the board's Hexagon
NPU through the geniex bridge (`vibrodiag_mcp_prototype/scripts/serve_qwen35_npu.sh`).
