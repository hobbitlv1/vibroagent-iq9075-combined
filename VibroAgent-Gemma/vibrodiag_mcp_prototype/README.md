# VibroAgent-Gemma application package

This package contains the six-board live web application and model bridge used by the parent [VibroAgent-Gemma deployment](../README.md). It is a research monitoring/triage application, not a certified structural-safety system.

The runtime-specific modules are:

- `vibrogemma_live.py`: synchronized six-board capture, deterministic quality checks, encoder handoff, response validation, and optional test-only LUMO overlay;
- `vibrogemma_geniex_server.py`: loopback OpenAI-compatible endpoint for the patched GenieX 0.4.0 external-embedding path;
- `vibrogemma_webchat.py`: desktop live monitor, spectrum, replay, chat, target-state display, and popup coordination;
- `scripts/live_vibrogemma_worker.py`: manifest-verified ONNX encoder worker producing exactly `84 × 1536` continuous embeddings;
- `board_reader_process.py` and `sdk_vibrometer.py`: isolated live STDATALOG readers;
- `spectral.py`: time-domain and spectrum tooling.

Install and run through the parent scripts rather than constructing environments by hand:

```bash
cd ..
./setup.sh --live       # or --demo
./vibroagent.sh start   # live boards
./vibroagent.sh demo target_3
```

Focused tests:

```bash
PYTHONPATH=src:. .run/vibrogemma-venv/bin/python -m pytest -q \
  tests/test_vibrogemma_live.py \
  tests/test_vibrogemma_geniex_server.py \
  tests/test_vibrogemma_deployment_health.py \
  tests/test_sdk_vibrometer.py \
  tests/test_webchat_graph.py
```

The accepted model decision labels are `normal` and `unknown_anomaly`; deterministic acquisition/quality failures use `data_invalid`. Exactly five target records are required and validated for every model verdict.
