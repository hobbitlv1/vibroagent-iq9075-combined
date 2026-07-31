# Current model artifacts

This directory describes the two artifacts used by the supported codes_v3 production pipeline. Run setup from the repository root before starting VibroAgent.

## Install the downloadable model

~~~bash
./setup_models.sh
~~~

The installer resolves the GGUF from the private GitHub release weights-v1 first and uses the configured Hugging Face repository only as a fallback. It verifies every downloaded part and the reconstructed model before returning successfully.

`release_assets_manifest.json` records the exact release filenames, sizes, and SHA-256 values for both GGUF parts, codec-v1, and the immutable offline-recording archive. The same file is uploaded as a release asset and drives GGUF part verification during installation.

| Property | Production value |
|---|---|
| File | qwen3_4b_codes_v3_Q4_0_embq8.gguf |
| Size | 2,463,745,376 bytes |
| SHA-256 | 3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2 |
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| Fine-tuning | LoRA on 11,000 codes_v3 examples, merged before quantization |
| Quantization | Q4_0 with q8_0 embeddings |
| Training data | Public vibration corpora; data from the monitored building was not used |

By default, the fallback repository is hobbitlv/vibroagent-models. Set VIBRO_MODELS_REPO to use another source. For a private fallback, set HF_TOKEN or authenticate with the Hugging Face CLI.

## Versioned codec checkpoint

The smaller codec-v1 artifacts are versioned with the source:

| File | SHA-256 | Role |
|---|---|---|
| codec_v1/best.pt | a80216b639327f9b4e59447138abc025fddac2d6d5eca68448c4ac14c653a350 | Frozen residual-VQ codec checkpoint from training step 29,000 |
| codec_v1/split_manifest.json | Bound to the checkpoint | Training-split provenance checked by the loader |

For each 10-second, 400 Hz input window, codec-v1 produces 125 latent frames. Two 1,024-entry codebooks produce 250 discrete symbols. The runtime loads the checkpoint on the CPU, selects evaluation mode, enables deterministic algorithms, and fails closed when checkpoint provenance does not match.

Verify the complete codec chain with:

~~~bash
~/codec-cpu-venv/bin/python examples/selftest.py
~~~

The successful result is byte-identical to examples/golden_codes.json.

## Serving

On the IQ-9075, vibroagent.sh starts the GGUF through the GenieX Hexagon backend with two sessions and a 6,144-token context. On another Linux computer, the same GGUF can be served by an OpenAI-compatible llama.cpp server for the recorded-data example. The root README contains the complete setup, architecture, and verification instructions.
