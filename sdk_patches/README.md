# VibroAgent patches to STDATALOG-PYSDK v1.3.0

This directory contains **everything that differs from the stock ST SDK** — verified
by a byte-level comparison of the whole working tree against the official
`stdatalog-pysdk` v1.3.0 release (submodules `stdatalog_core @ a4824fc` and
`stdatalog_pnpl @ a6e36d2`). Every other SDK file in the original workspace was
byte-identical to stock and is therefore not versioned here; `setup_sdk.sh`
re-fetches it from ST's GitHub.

Contents:

- `overlay/` — the 7 changed files, at their original paths. `setup_sdk.sh`
  copies this directory over the freshly fetched stock tree.
- `stdatalog-pysdk-v1.3.0-to-vibroagent.diff` — the same changes as one unified
  diff (`a/` = stock v1.3.0, `b/` = this project), for review or `patch -p1`.
- `LICENSE.md` — ST's BSD-3-Clause license, which covers all files in `overlay/`
  that derive from ST sources.

## Changed files and why

### 1. `stdatalog_core/stdatalog_core/HSD_link/HSDLink.py`

- **Lower-latency polling** — `SensorAcquisitionThread.run_print()` and
  `run_no_print()` wait `self.stopped.wait(0.001)` between read attempts instead
  of the stock `0.02`. Live acquisition threads check for incoming USB data every
  1 ms instead of every 20 ms, at the cost of more aggressive polling.
- **`fallback_to_serial` flag** — `create_hsd_link(..., fallback_to_serial=True)`
  gained a keyword. When the caller passes `False`, the SDK refuses to silently
  fall back from native HSDv2 USB to the serial/v1 path, so the logger/probe fail
  fast instead of appearing "connected" through a slower, incompatible transport.
- **Acquisition threads returned** — `start_sensor_acquisition_thread()` returns
  the started thread objects so callers can track/join them for clean shutdown in
  multi-board acquisition code.
- **`set_data_ready_callback()` helper** — public static helper that forwards to
  the HSDv2 link's native data-ready callback registration (below).

### 2. `stdatalog_core/stdatalog_core/HSD_link/HSDLink_v2.py`

- Adds `set_data_ready_callback(d_id, comp_name, callback)`, forwarding to the
  communication manager. (2 added lines.)

### 3. `stdatalog_core/stdatalog_core/HSD_link/communication/PnPL_HSD/hsd_dll.py`

- Adds `HSD_DATA_READY_CALLBACK`, a ctypes wrapper for the native HSDv2
  `hs_datalog_set_data_ready_callback` symbol.
- Registered callback objects are kept in `_data_ready_callbacks` so Python does
  not garbage-collect them while native USB endpoint threads may still call them.
- Exposes `hs_datalog_set_data_ready_callback()` as a Python method.

### 4. `stdatalog_core/stdatalog_core/HSD_link/communication/PnPL_HSD/PnPLHSD_com_manager.py`

- Exposes `set_data_ready_callback()` at the communication-manager layer,
  completing the plumbing: `HSDLink` → `HSDLink_v2` → com manager → native DLL.

Together, 2–4 expose the native HSDv2 data-ready callback API so acquisition can
be event-driven (native code pushes data availability) rather than purely
poll-driven. Note: the practical multi-board parallelism in VibroAgent comes from
the process-per-board architecture in
`vibrodiag_mcp_prototype/src/vibroagent_mcp/board_reader_process.py`; these SDK
hooks are the foundation for event-driven acquisition.

### 5. `stdatalog_core/stdatalog_core/HSD/HSDatalog_v2.py`

- **Bounded buffer preallocation for live `.dat` reads.** Stock code preallocates
  `byte_chest` / `raw_data_array` from an estimate proportional to the whole
  file. A live `.dat` kept small on disk by head hole-punching keeps growing its
  *logical* size, so whole-file preallocation grows without bound. The patch
  moves the allocation after `last_index` is known and sizes the buffers from the
  read start to EOF (`file_size - last_index + missing_bytes + 2 * cmplt_pkt_size`).

### 6. `stdatalog_pnpl/stdatalog_pnpl/DTDL/usb_device_catalog.json`
### 7. `stdatalog_pnpl/stdatalog_pnpl/DTDL/dtmi/appconfig/steval_stwinkt1b/FP_SNS_DATALOG2_Datalog2-4.json`

- Device-catalog state as synced on the deployment machine (the SDK updates its
  local DTDL catalog when it meets board firmware). Included so a fresh checkout
  reproduces the exact runtime device-template state without network access.

## Applying

`../setup_sdk.sh` does both steps (fetch stock at the pinned commits, copy
`overlay/` on top). To apply the changes by hand after fetching stock — run
from the repository root:

```bash
patch -p1 < sdk_patches/stdatalog-pysdk-v1.3.0-to-vibroagent.diff
# or simply:
cp -r sdk_patches/overlay/. .
```


## Runtime-cached ST device templates (added 2026-07-31)

`overlay/stdatalog_pnpl/.../appconfig/steval_stwinbx1/` additionally carries
six **unmodified** ST device-template JSONs (notably
`FP_SNS_DATALOG2_Datalog2-10.json`, the template the STWIN.box fleet resolves
to under FP-SNS-DATALOG2 v3.2.0). Stock v1.3.0 does not ship them;
`DeviceCatalogManager` normally downloads them from ST's catalog on first
board contact and caches them into the DTDL tree. Shipping the cache makes a
fresh clone work without that first online round-trip. They are ST content
under the same BSD-3-Clause license as the rest of the overlay.
