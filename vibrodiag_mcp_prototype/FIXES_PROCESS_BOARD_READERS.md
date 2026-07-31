# Process-per-board live readers

This patch adds a long-lived reader process per board for live STDATALOG-PYSDK reads.

## Why

The SDK decoder keeps process-local mutable state. When several boards are decoded from one Python process, reads can become serialized and a board that is launched later can lag behind the others. Running one worker process per board gives every board its own decoder state and lets all boards read in parallel.

## What changed

- Added `src/vibroagent_mcp/board_reader_process.py`.
- The live graph now requests `process_reader=1` and prewarms reader processes from `/api/live-sensors?process_reader=1&prewarm=1`.
- “All boards” graph reads now use parallel browser requests instead of reading boards one after another.
- The agent monitor passes `process_reader=1`, and the building pipeline reads the baseline and target windows concurrently when process readers are enabled.
- Added `/api/board-readers` to inspect worker process status.

## Useful environment variables

- `VIBRO_BOARD_READER_PROCESSES=0` disables process readers.
- `VIBRO_BOARD_READER_TIMEOUT_S=30` changes per-read timeout.
- `VIBRO_BOARD_READER_START_METHOD=spawn` changes multiprocessing start method. `spawn` is the default.
- `VIBRO_BOARD_READER_PARALLELISM=16` caps concurrent board reads inside the agent pipeline.

## Validation

```bash
PYTHONPATH=src pytest -q
# 93 passed, 3 skipped
```
