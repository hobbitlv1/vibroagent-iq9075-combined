# Readings and popup reliability fixes

Applied fixes for live multi-board vibration monitoring:

- Serialized STDATALOG-PYSDK/HSDatalog reads globally with a re-entrant lock, including persistent live decoders, to avoid parallel decoder contention when several boards are displayed or the agent polls at the same time.
- Changed the graph UI to read boards sequentially instead of with `Promise.all`, so one board is decoded at a time.
- Changed live graph requests to `require_current=1`; stale decoded streams or old `.dat` files now show an error instead of leaving an old waveform visible.
- Cleared the waveform and timing panel on read failure so stale graphs are not mistaken for live data.
- Added per-board failure rows/panels when “All boards” is selected, so failed boards are visible instead of silently disappearing.
- Fixed live sensor paths: `config/sensors.live.yaml` now uses relative `../../stdatalog_examples/live_*` folders, and the sensor registry remaps legacy absolute `/home/ubuntu/.../stdatalog_examples/...` paths into the current checkout.
- Fixed agent popup behavior: the monitor endpoint can run the deterministic feature prepass without requiring Qwen/LLM availability, and the UI always shows the in-page anomaly popup even when browser notifications are deduplicated.

Validation run:

```bash
cd vibrodiag_mcp_prototype
PYTHONPATH=src pytest -q
# 90 passed, 3 skipped
```
