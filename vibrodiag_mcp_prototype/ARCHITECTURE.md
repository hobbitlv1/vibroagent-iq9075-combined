# Architecture

## Data flow

```text
Six IIS3DWB boards
        │
        ▼
STDATALOG acquisition logger
        │ one folder per board
        ▼
SensorRegistry
        │ resolves sensor ID → configured folder/component/axis
        ▼
SDK window reader
        │ calibrated acceleration window
        ├──────────────► floating waveform/PSD interface
        │
        ▼
Building metric extraction
        │ RMS, peak, crest factor, kurtosis, PPV estimate,
        │ dominant frequency, band powers, quality flags
        ▼
Reference comparison
        │ target/reference ratios and spectral distance
        ▼
Deterministic precheck + bounded model classification
        │ fixed sensor and network labels
        ▼
MCP result + webchat operator summary
```

## Runtime boundaries

### Acquisition

The logger owns the connected boards and writes each stream to its dedicated folder. The webchat does not reassign board identities. Board selection is resolved through `config/sensors.live.yaml`.

### Signal processing

`agent_pipeline.py` and `spectral.py` perform deterministic numerical processing. Raw sample arrays remain local. Compact metrics, labels, and evidence are the only information passed to the language model.

### Model layer

`llm_sensor_agent.py` uses one bounded request to classify all selected target sensors and the overall network. Deterministic labels remain available as fallback. Returned JSON is checked for schema compliance, valid labels, quality-gate consistency, and building-domain vocabulary.

### MCP and webchat

`mcp_server.py` exposes an allowlisted set of building tools. `qwen_mcp_host.py` routes operator requests to those tools and renders a compact summary without an additional free-form rewrite after a tool result is available.

## Safety and reliability controls

- fixed application-domain labels;
- registered sensor IDs only;
- current-data gate for live requests;
- synchronized reference comparisons;
- deterministic fallback when the model is unavailable;
- output-domain guard before webchat display;
- bounded prompts, output tokens, and request timeouts;
- no certified structural-safety claim.
