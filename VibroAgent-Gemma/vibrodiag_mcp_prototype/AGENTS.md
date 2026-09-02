# Agent Development Rules

- Keep every model prompt and tool description focused on building response, reference comparisons, transient events, sensor quality, and monitoring actions.
- Use only the label sets declared in `schemas.py`.
- Raw acceleration arrays must remain local; send compact metrics to the model.
- Treat deterministic checks as authoritative for stale, missing, flat, or otherwise invalid data.
- Resolve acquisition sources through `SensorRegistry`; never let a model choose an arbitrary folder.
- Do not add certified structural-safety conclusions.
- Keep model requests bounded and return deterministic results when the endpoint fails or times out.
- Validate every model-produced payload before it is exposed through MCP or webchat.
- Changes to acquisition or graph code require focused regression tests and checksum review of the logger, webchat, and spectral modules.
