# VibroAgent MCP — Building Vibration Monitoring

This application monitors a building with one reference IIS3DWB sensor and one or more target sensors. It reads synchronized acquisition windows, computes compact vibration metrics, compares every target with the reference, and returns fixed building-monitoring labels through an MCP-enabled webchat.

## Scope

The supported analysis domain is:

```text
building_vibration_relative_monitoring
```

The system reports relative vibration level, dominant frequency, spectral distribution, transient or shock-like events, sensor-data quality, and whether a deviation is local or shared by several sensors. It is a monitoring and triage tool, not a certified structural-safety assessment.

## Main components

- `webchat_server.py`: webchat, live-readings page, and floating PSD window.
- `spectral.py`: periodic-Hann Welch PSD, raw periodogram mode, filtering, peak extraction, and plot aggregation.
- `sensor_registry.py`: maps each sensor ID to one configured acquisition folder and location.
- `sdk_vibrometer.py`: decodes IIS3DWB acquisition windows.
- `llm_sensor_agent.py`: one bounded model call for all selected sensors, with deterministic checks and fixed labels.
- `agent_pipeline.py`: deterministic building metrics and reference comparisons.
- `mcp_server.py`: exposes only building-monitoring tools.
- `qwen_mcp_host.py`: deterministic tool routing and a bounded host-generated operator summary.

## Sensor configuration

Edit `config/sensors.live.yaml`. Every board must have a unique `sensor_id` and acquisition folder:

```yaml
baseline_sensor_id: baseline

sensors:
  - sensor_id: baseline
    location: first_floor_reference
    role: baseline
    acquisition_folder: ../../stdatalog_examples/live_baseline
    hsd_sensor_name: iis3dwb_acc
    axis: z

  - sensor_id: target_1
    location: upper_floor_position
    role: target
    acquisition_folder: ../../stdatalog_examples/live_target_1
    hsd_sensor_name: iis3dwb_acc
    axis: z
```

Use the same axis, time interval, sampling conditions, and mounting approach when comparing floors.

## Installation

From the repository root (or run the top-level `./setup.sh`, which does all of
this plus the SDK and model runtime):

```bash
python -m venv vibroagent-venv
source vibroagent-venv/bin/activate
pip install -e vibrodiag_mcp_prototype
```

For an OpenAI-compatible local model client:

```bash
pip install -e '.[qwen]'
```

## Run the complete live stack

From the repository root:

```bash
./vibroagent.sh start
./vibroagent.sh status
./vibroagent.sh stop
```

The default webchat address is `http://127.0.0.1:7860` and the live graph page is `/graph`.

To run only the webchat:

```bash
cd vibrodiag_mcp_prototype
python -m vibroagent_mcp.webchat_server --host 127.0.0.1 --port 7860
```

## MCP tools

The server exposes only these application tools:

- `list_building_sensors`
- `read_building_sensor_window`
- `run_autonomous_sensor_check`
- `run_building_sensor_agent`
- `run_building_vibration_agent_pipeline`
- `export_building_validation_dataset`

Run the server over stdio:

```bash
python -m vibroagent_mcp.mcp_server --transport stdio
```

Example client call:

```bash
python -m vibroagent_mcp.mcp_client_demo \
  --tool run_autonomous_sensor_check \
  --config-path config/sensors.live.yaml \
  --duration-s 10
```

Read one registered board without running the network assessment:

```bash
python -m vibroagent_mcp.mcp_client_demo \
  --tool read_building_sensor_window \
  --sensor-id target_2 \
  --duration-s 5
```

Add `--allow-recorded-data` only when intentionally reading a saved acquisition.

## PSD display

The floating spectrum window provides:

- periodic-Hann Welch PSD by default;
- optional raw periodogram mode;
- X, Y, Z, or magnitude selection;
- high-pass, low-pass, band-pass, and notch filters;
- a full time-domain overview plus an inspectable spectrum;
- a persistent primary-peak marker;
- frequency and PSD readout while moving the pointer along the curve.

Filters are analysis-only and do not modify stored acquisition files.

## Building labels

Sensor-level labels:

```text
normal_relative_to_baseline
mild_deviation
significant_local_deviation
impulsive_or_shock_event
possible_sensor_issue
insufficient_data
```

Network-level labels:

```text
normal_relative_to_reference_sensor
mild_local_deviation
mild_multi_sensor_deviation
localized_vibration_deviation
multi_sensor_building_wide_vibration_event
sensor_network_quality_issue
insufficient_data
```

Model output is validated against these fixed labels. Out-of-domain component diagnoses are rejected before they can reach the webchat.

## Tests

```bash
pytest -q
```

Focused building and webchat tests:

```bash
pytest -q \
  tests/test_service.py \
  tests/test_agent_pipeline.py \
  tests/test_llm_sensor_agent.py \
  tests/test_mcp_building_pipeline.py \
  tests/test_qwen_host_stability.py \
  tests/test_webchat_graph.py \
  tests/test_spectral.py
```
