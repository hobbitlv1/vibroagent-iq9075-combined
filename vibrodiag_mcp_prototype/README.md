# VibroAgent runtime package

This directory contains the Python application used by the current codes_v3 production stack. Start with the [repository README](../README.md): it is the authoritative setup, operation, architecture, and hardware guide.

## Setup and run

From the repository root:

~~~bash
./setup.sh
./vibroagent.sh start
~~~

For recorded input without STWIN.box hardware:

~~~bash
./setup.sh --offline
./vibroagent.sh start
~~~

The top-level setup creates the application environment, installs the pinned STDATALOG SDK overlay, installs the CPU codec and GenieX environments, and obtains the hash-verified model. The launcher starts the model service and web application in both modes. Live mode also starts the USB callback logger; offline mode starts no data writer and reads the verified recordings in place.

## Current architecture

The supported monitoring path is:

1. SensorRegistry resolves the baseline and five target folders, remapping them to the immutable recording root in offline mode.
2. One persistent board_reader_process child owns the mutable STDATALOG decoder state for each sensor without modifying the underlying acquisition.
3. sdk_vibrometer returns calibrated, measured-rate windows: latest windows for live data or explicit fixed timestamps for replay.
4. The web server calls scripts/live_codes_worker.py for the latest live 10 seconds or once at each scheduled offline timestamp.
5. The worker resamples each selected axis to 400 Hz, loads the frozen architecture from scripts/codec_runtime.py, and encodes it with the pinned codec-v1 checkpoint.
6. live_codes.py builds the byte-exact codes_v3 prompt from 250 symbols per sensor plus relative level values.
7. geniex_openai_server.py serves the fine-tuned Qwen3-4B GGUF on Hexagon through an OpenAI-compatible loopback endpoint.
8. Strict schema and label validation accepts the verdict or exposes the failure to the operator.
9. webchat_server.py renders live waveforms, PSD views, chat, monitor history, and alerts.

The waveform and PSD display path branches before codec inference. Display filters do not alter the acquisition stream or the model input. During offline replay, graphs follow the monotonic virtual position continuously while codec-v1 and codes_v3 remain idle between inference timestamps.

Offline anomalies are already encoded in the supplied target `.dat` files. `offline_replay.py` only schedules reads; it never copies, truncates, appends to, or rewrites the recordings.

## Important paths

| Path | Responsibility |
|---|---|
| config/sensors.live.yaml | Logical IDs, roles, locations, folders, components, and axes |
| src/vibroagent_mcp/sensor_registry.py | Registered-folder resolution and path boundary |
| src/vibroagent_mcp/offline_replay.py | Immutable replay clock and timestamp claim controller |
| src/vibroagent_mcp/board_reader_process.py | Isolated persistent SDK readers |
| src/vibroagent_mcp/sdk_vibrometer.py | Calibrated live-window extraction |
| scripts/codec_runtime.py | Frozen codec-v1 architecture and normalization |
| scripts/live_codes_worker.py | Deterministic codec subprocess |
| scripts/serve_codes_v3_geniex.sh | Hash-pinned GenieX launcher for codes_v3 |
| src/vibroagent_mcp/live_codes.py | Prompt construction, labels, and response validation |
| src/vibroagent_mcp/geniex_openai_server.py | GenieX model adapter |
| src/vibroagent_mcp/webchat_server.py | HTTP API, monitoring coordinator, and UI |
| tests/ | Hardware-independent regression suite |

Raw acceleration stays local. Only codec symbols, relative level values, fixed instructions, and the response schema are sent to the loopback model service. Deterministic data-quality checks remain authoritative.

## Tests

From this directory, after top-level setup:

~~~bash
../vibroagent-venv/bin/python -m pytest tests/
~~~

For the frozen codec reproducibility gate, run from the repository root:

~~~bash
~/codec-cpu-venv/bin/python examples/selftest.py
~~~
