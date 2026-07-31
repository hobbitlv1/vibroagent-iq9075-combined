# VibroAgent — on-device LLM building-vibration diagnostics (Qualcomm IQ-9075)

VibroAgent is an edge application that monitors building vibration with six
STMicroelectronics **STWIN.box** (STEVAL-STWINBX1) sensor nodes over USB,
computes spectral features (PSD/FFT) from their **IIS3DWB** wideband
accelerometers, and asks a **locally hosted Qwen3-4B-Instruct** model — running
on the **Hexagon NPU** of a **Qualcomm IQ-9075 EVK (QCS9075)** — to produce
strict-JSON diagnostic verdicts. A web UI serves live graphs, chat, and agent
monitoring.

**Target platform: Linux on Qualcomm IQ-9075 only.** Nothing else is supported
for the live system — but the recorded example acquisitions in `examples/`
make the codes pipeline below runnable on any PC without hardware (see
**Try it without any hardware**).

## The codes_v3 decision path (current production stack)

Since 2026-07-28 the production monitor decides through **discrete codec
codes** instead of textual spectral descriptors: a frozen neural codec
("codec-v1", residual-VQ conv autoencoder) turns each sensor's 10 s window
into **250 single-token code symbols**, and a Qwen3-4B model **fine-tuned on
those codes** (`qwen3_4b_codes_v3_Q4_0_embq8.gguf`) judges every target
sensor strictly relative to the reference sensor:

```
 6x STWIN.box (IIS3DWB, 26.7 kHz, USB)
        │  .dat streams (STDatalog v2)
        ▼
 vibroagent_two_vibrometer_logger.py               [board]
        │  10 s windows, z axis, per-unit sample rate
        ▼
 codec-v1 (frozen RVQ autoencoder, CPU worker)     [board or any PC]
        │  250 code symbols / window + broadband level delta (dB)
        ▼
 byte-exact codes_v3 prompt
        ▼
 Qwen3-4B codes_v3 GGUF                            [Hexagon NPU via GenieX,
        │                                           or llama.cpp on any PC]
        ▼
 {"sensor_reports":[...], "network_status":..., "affected_sensor_ids":[...]}
```

Chain details: most recent 10 s per sensor, z axis, DC removal, exact
rational resample to 400 Hz, `normalize_window` (detrend + unit AC rms — the
codec sees only the *shape*; the absolute *level* travels as one
`level_rel_db` scalar in the prompt), frozen checkpoint
`models/codec_v1/best.pt` (step 29000, 2 codebooks x 1024 entries, 125
frames), symbols from `data/codec_pretrain/r1_token_map.json` (2048 symbols
screened so each is exactly one Qwen3 token), prompt rendered byte-exactly
from `assets/codes_v3_prompt_template.json` (extracted from the immutable
11 000-example training set, golden-tested), and a strictly validated JSON
verdict over fixed vocabularies (per-sensor `normal_relative_to_baseline` /
`mild_deviation` / `significant_local_deviation` /
`sensor_data_quality_issue`; network `normal_relative_to_reference_sensor` …
`multi_sensor_building_wide_vibration_event`). `VIBRO_CODES_AXES=x,y,z`
optionally runs the pass per axis and combines by max severity.

The earlier descriptor path (PSD/FFT metrics → LLM) remains available as the
legacy comparison mode (`VIBRO_STACK=genie`).

## Offline mode (full stack on recorded data — no boards)

The data-source mode is fixed **at setup time** and persisted in
`.vibro_mode`, because the two producers are mutually exclusive: every USB
logger start truncates the acquisition `.dat` files, so in offline mode the
USB logger **never starts** — `stdatalog_examples/vibroagent_replay_logger.py`
runs in its place:

```bash
./setup.sh --offline        # persists mode; skips USB/udev setup entirely
./vibroagent.sh start       # NPU model server + webchat + REPLAY logger
```

`setup.sh --offline` additionally downloads the **5-minute recording set**
(`setup_recordings.sh`, ~280 MB, sha-verified per file into `./recordings/`,
placed once and never modified): real recorded acquisitions from the
six-sensor installation, with **five demonstration events staged into the
target channels** (baseline untouched) so the monitor has something to flag
during replay. The schedule ships in `recordings_manifest.json`:

| time in cycle | sensor | event | measured level vs baseline |
|---|---|---|---|
| 45–55 s | target_2 | repeated impulses (significant) | +13.4 dB |
| 90–100 s | target_4 | tone + harmonics (significant) | +13.1 dB |
| 150–160 s | target_1 | broadband rise (significant) | +13.4 dB |
| 210–220 s | target_3 | repeated impulses (significant) | +9.5 dB |
| 210–220 s | target_5 | mild tone | +5.3 dB |

The replay logger streams the placed recordings into the live acquisition
folders at the ORIGINAL real-time cadence (~37.5 ms per frame), looping over
the 5-minute cycle — **the monitor's verdicts and popups fire at these
moments**, every cycle, as the anomaly enters its 10 s window. Everything downstream is byte- and time-faithful and completely
unaware of the replay — the webchat, monitor, and codec worker read the same
folders through the same code paths as with live boards:

- the **live-readings graphs** redraw the recorded waveforms as they
  "arrive";
- the **chat/agent** analyses the recorded data instead of live streams;
- the **codes monitor** cuts its 10 s windows at the same relative moments
  as during the recording, and — greedy decoding being deterministic —
  verdicts and **popups reproduce at the same registered moments** with the
  same content as they had live.

Switch back to live boards with a plain `./setup.sh` (reruns USB setup and
persists `live`).

## Try it without any hardware

`examples/` contains real recorded acquisitions from all six sensors
(STDatalog `.dat`, 60 s each) plus a pure-numpy reader — no ST SDK, no USB.
The demo drives the **production implementation** (the same
`live_codes_worker.py` and `vibroagent_mcp.live_codes` the board runs);
nothing is re-implemented.

```bash
# environment via uv (https://astral.sh/uv):
uv venv .venv && source .venv/bin/activate
uv pip install numpy scipy
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) Prove your environment reproduces the live encoder BYTE-FOR-BYTE:
python examples/selftest.py
#    ... SELFTEST PASS: byte-identical to the proven live encoder

# 2) Inspect the pipeline without an LLM (codes + the exact prompt):
python examples/run_example.py --dry-run

# 3) Full verdict: fetch the codes_v3 GGUF (hash-verified; needs HF_TOKEN
#    for the private weights repo), serve it with llama.cpp (any PC) ...
./setup_models.sh
llama-server -m models/qwen3_4b_codes_v3_Q4_0_embq8.gguf -c 6144 --port 8080
python examples/run_example.py --endpoint http://localhost:8080/v1
# ... or on the board against GenieX:  --endpoint http://localhost:18181/v1
```

`--offset-s` picks a different 10 s window inside the 60 s recordings;
`--axis x|y|z` selects the accelerometer axis (the live monitor feeds **z**;
the training contract is single-axis, never the vector norm).
`examples/golden_codes.json` freezes the expected codec output for the first
10 s of every example; `selftest.py` fail-closes on any drift in STDatalog
decoding, resampling, normalization, the checkpoint, or symbol
serialization. The GGUF is 2.4 GB and distributed separately — pinned hashes
in `models/README.md`.

## Hardware requisites (full live system)

| item | value |
|------|-------|
| Board | **Qualcomm Addons IQ-9075 EVK** (Dragonwing, QCS9075 SoC), Linux (kernel 6.8 qcom), aarch64 |
| NPU | Hexagon HTP; Q4_0 GGUF served on the NPU by GenieX (`GENIEX_DEVICE_MAP`, context 6144 = measured HTP ceiling) |
| Sensor nodes | **STEVAL-STWINBX1 (STWIN.box)** × 6 — one reference ("baseline") + five targets, mapped by serial in `stdatalog_examples/vibroagent_live_devices.json` |
| Sensor firmware | **FP-SNS-DATALOG2 v3.2.0** (<https://github.com/STMicroelectronics/fp-sns-datalog2>) |
| Accelerometer | **IIS3DWB** ultra-wide-bandwidth 3-axis MEMS: 26.667 kHz ODR (per-unit measured ~26.6–26.8 kHz; the measured rate travels per slot through the whole chain), ±16 g, 0.488 mg/LSB, `int16`, `samples_per_ts=1000` |
| USB | powered hub with ≥ 6 ports; ST udev permissions (`hsdatalog` group) |
| Storage | ≥ 8 GB free for models (GGUF 2.4 GB + caches) |
| Host PC (offline demo only) | any x86-64/ARM machine, Python ≥ 3.10, CPU torch; the 4B Q4_0 GGUF runs in ~3 GB RAM under llama.cpp |

## System architecture

Three long-lived services:

| # | Service | Port | What it does |
|---|---------|------|--------------|
| 1 | NPU model server | `:18181` (GenieX) or `:8910` (Genie) | Serves Qwen3-4B on the Hexagon NPU behind an OpenAI-compatible HTTP API — in the codes stack this is the **sha-pinned codes_v3 GGUF** via `scripts/serve_qwen35_npu.sh`; see **Model backends** below |
| 2 | Webchat server | `:7860` | Live PSD graph, board list, chat, agent monitor endpoints, timing diagnostics (`vibroagent_mcp.webchat_server`); in codes mode it cuts synchronized 10 s windows, runs the codec worker, builds the codes_v3 prompt, queries :18181 and renders the verdict — every failure surfaces as a model failure |
| 3 | Logger | — | Drives the 6 STWIN.box boards over native HSDv2 USB, writing continuous acquisitions into `stdatalog_examples/live_*` (`vibroagent_two_vibrometer_logger.py`) |

## Model backends — you choose

The stack speaks one OpenAI-compatible surface; which runtime sits behind it is
selectable via `VIBRO_STACK` in `vibroagent.sh` (`codes` default | `genie`)
and `MODEL_BACKEND` in `vibroagent_direct.sh`:

### 1. `geniex` — Qualcomm GenieX runtime (**default**)

[GenieX](https://github.com/qualcomm/GenieX) (docs:
<https://geniex.aihub.qualcomm.com>) is llama.cpp with the GGML **Hexagon
backend**: it runs standard **GGUF models directly on the NPU**, no model
compilation and no QAIRT bundle required.

- **Install:** `./setup_geniex.sh` — creates `~/geniex-venv` and installs
  `geniex` from public PyPI.
- **Model:** in the codes stack, the **fine-tuned**
  `qwen3_4b_codes_v3_Q4_0_embq8.gguf` (path `CODES_GGUF`, hash-verified via
  `GENIEX_EXPECT_MODEL_SHA256` before load). **Q4_0 is the one GGUF precision
  that lands on the Hexagon NPU** — other quants fall back to GPU/CPU.
- **Served by** `vibroagent_mcp.geniex_openai_server` on `127.0.0.1:18181`,
  using the GenieX venv's Python (the module needs only stdlib + `geniex`).
- **Capabilities** the legacy backend lacks: per-request `max_tokens` /
  `temperature` / `stop`; `response_format` with `json_object` and
  `json_schema` (compiled here to a GBNF grammar); a raw-GBNF `grammar`
  extra-body field; runtime-configurable context (`GENIEX_N_CTX`, default
  6144 — the measured HTP0 ceiling; 8192 fails because the KV cache needs a
  ~1 GiB fastrpc_mmap, above the Hexagon buffer-mapping limit); explicit
  rejection of prompts the runtime would silently truncate at the context
  limit.

### 2. `genie` — legacy QAIRT/Genie adapter

The original path: `vibroagent_mcp.genie_openai_server` on `:8910`, driving a
**Genie-compiled model bundle** through the QAIRT SDK (native helper:
`genie_persistent_runner.cpp`).

- **Requires** the QAIRT SDK — `./setup_qairt.sh` downloads the public
  Qualcomm AI Runtime Community SDK 2.46.0.260424 (~1.8 GB) and extracts it to
  the exact `QAIRT_SDK_ROOT` the launchers expect.
- **Also requires** a Genie-compiled model (`GENIE_CONFIG`) with context/KV
  **baked at 4096**. This is a local build artifact — it is *not*
  auto-downloadable.
- Ignores per-request sampling parameters; context length is a compile-time
  constant.

### 3. `onnx` — experimental spike

`vibroagent_mcp.onnx_openai_server` (console script `vibroagent-onnx-openai`)
over onnxruntime-genai; status and findings in
`vibrodiag_mcp_prototype/docs/QWEN35_4B_ONNX_SPIKE.md`. Not wired into the
launchers.

Backend selection:

```bash
./vibroagent.sh start                             # codes stack (GenieX + codes_v3 GGUF)
VIBRO_STACK=genie ./vibroagent.sh start           # legacy QAIRT/Genie stack
./vibroagent_direct.sh start                      # per-service launcher (GenieX default)
MODEL_BACKEND=genie ./vibroagent_direct.sh start  # legacy backend
```

Data flow: the logger writes `.dat` streams into per-board acquisition folders →
`sdk_vibrometer.py` reads IIS3DWB windows out of those folders (one dedicated
reader **process per board**, see below) → in codes mode the webchat monitor
sends the windows through `scripts/live_codes_worker.py` (codec-v1 codes, in a
separate CPU-torch venv — `VIBRO_CODES_WORKER_PYTHON`) and asks the codes_v3
model for the verdict; in legacy mode `llm_sensor_agent.py` computes compact
vibration metrics and asks the model for JSON-labeled decisions → webchat
displays readings, PSD comparisons, and agent verdicts.

### Why one process per board

The stock ST SDK keeps process-local mutable decoder/native state; reading
several boards from one Python process serializes and corrupts live reads.
`board_reader_process.py` starts one long-lived worker process per board, each
owning its own SDK reader, persistent live decoder, interpreter state, and
queues. The web API pre-warms them (`/api/live-sensors?process_reader=1&prewarm=1`)
and the browser reads all boards concurrently with `Promise.all`. The agent
pipeline mirrors this with a `ThreadPoolExecutor` whose per-thread reads each
land in a separate worker process.

## Repository layout

```
vibroagent.sh                 Primary launcher: start|stop|restart|status for all 3 services
                              (codes stack by default; VIBRO_STACK=genie for legacy). Binds
                              webchat to the LAN interface (no auth — see Security).
vibroagent_direct.sh          Alternative launcher: per-service subcommands, 127.0.0.1 only.
setup.sh                      One-command bootstrap: runs setup_sdk.sh, creates the app venv
                              (./vibroagent-venv), runs setup_geniex.sh; --qairt adds setup_qairt.sh.
setup_usb.sh                  Linux USB prerequisites: libusb, hsdatalog udev rules + group
                              (verbatim from the production board).
setup_sdk.sh                  Fetches stock STDATALOG-PYSDK v1.3.0 (pinned commits) and applies
                              the sdk_patches/ overlay (incl. cached STWIN.box device templates).
setup_geniex.sh               Installs the GenieX runtime (PyPI) into ~/geniex-venv (via uv).
setup_models.sh               Downloads the fine-tuned codes_v3 GGUF from the private HF
                              weights repo (HF_TOKEN) and verifies its pinned sha256.
setup_qairt.sh                Downloads the QAIRT Community SDK (~1.8 GB, public URL) —
                              only needed for the legacy genie backend.
sdk_patches/                  ONLY the files that differ from stock ST SDK v1.3.0, as a
                              path-preserving overlay + unified diff + docs. See sdk_patches/README.md.
examples/                     BOARD-FREE demo on real recorded data:
├── live_baseline/, live_target_1..5/   60 s recorded acquisitions (iis3dwb_acc.dat +
│                                        device_config.json + acquisition metadata)
├── golden_codes.json         Frozen expected codec output (selftest gate)
├── stdatalog_reader.py       Pure-numpy STDatalog v2 .dat reader (no ST SDK)
├── selftest.py               Byte-exact reproduction gate — run this first
└── run_example.py            .dat -> codes -> prompt -> GGUF -> verdict, end to end
models/
├── codec_v1/best.pt          Frozen codec-v1 checkpoint (15 MB, versioned) + split manifest
└── README.md                 codes_v3 GGUF distribution + pinned hashes (2.4 GB, external)
vibrodiag_mcp_prototype/      The application package (Python package name: vibroagent_mcp).
├── src/vibroagent_mcp/       Runtime modules. Key ones:
│   ├── webchat_server.py         Web UI + REST endpoints + the codes-mode monitor
│   ├── live_codes.py             Byte-exact codes_v3 prompt builder + strict verdict parser
│   ├── llm_sensor_agent.py       Legacy descriptor-mode monitoring loop; LLM-only decisions,
│   │                             deterministic code only measures
│   ├── board_reader_process.py   One long-lived SDK reader process per board
│   ├── sdk_vibrometer.py         Reads IIS3DWB windows from STDATALOG acquisition folders
│   ├── sensor_registry.py        Maps logical sensor IDs → acquisition folders / HSD names
│   ├── geniex_openai_server.py   OpenAI-compatible HTTP server over the GenieX SDK
│   │                             (llama.cpp/Hexagon); grammar-compiled json_schema
│   ├── genie_openai_server.py    Same idea over QAIRT Genie; genie_persistent_runner.cpp
│   │                             is its native helper
│   ├── onnx_openai_server.py     ONNX Runtime backend variant
│   ├── mcp_server.py             MCP tool surface; qwen_mcp_host.py drives it
│   ├── spectral.py / features.py PSD/FFT estimators and vibration metrics
│   ├── baseline.py / classifier.py  Baseline comparison + heuristic labels
│   ├── assets/codes_v3_prompt_template.json   Byte-constant codes_v3 prompt template
│   └── mock_* / *demo*.py        Synthetic-signal demos, no hardware required
├── scripts/
│   ├── live_codes_worker.py      Codec worker the webchat invokes (windows.npz -> codes.json)
│   ├── repr_extractor.py         Frozen-checkpoint loader/extractor + code-symbol map
│   ├── codec_dataset.py          normalize_window and codec data plumbing
│   ├── train_vibration_codec.py  Codec ARCHITECTURE classes (the checkpoint loads through
│   │                             these; training data is not distributed, so the training
│   │                             entry point is inert)
│   ├── vibration_injection.py / block_role_guard.py   Import dependencies of the above
│   ├── serve_qwen35_npu.sh       codes_v3 GGUF -> Hexagon NPU -> OpenAI endpoint (:18181)
│   └── model_backend_probe.py, qwen35_4b_onnx_readiness.py, phase-05 benchmark tooling
├── data/codec_pretrain/r1_token_map.json   2048-symbol code-token map
├── tests/                    Live-stack pytest suite (runs without hardware)
├── config/                   sensors.example.yaml, sensors.live.yaml (live 6-board mapping),
│                             qwen3-4b-instruct-2507-genie.json
├── docs/                     Fine-tune and ONNX-spike notes; figures/gen_*.py regenerate
│                             the architecture figures
├── pyproject.toml            Package metadata; extras: qwen, onnx, onnx-export, dev
└── requirements.txt          Runtime deps: numpy, pydantic, mcp[cli]
stdatalog_examples/           Only the project-owned live-acquisition artifacts:
├── live_baseline/            Baseline board acquisition folder (acquisition_info.json,
├── live_target_1..5/         device_config.json, source_identity.json each; .dat data
│                             is written here at runtime and is git-ignored)
├── vibroagent_live_devices.json / vibrodiag_live_devices.json   Board serial → role maps
├── vibroagent_two_vibrometer_logger.py   The production logger (native callback mode)
├── vibroagent_replay_logger.py           OFFLINE-mode producer: replays examples/ into the
│                                         live folders at original cadence (looping)
├── vibroagent_hsd_native_probe.py        Native HSDv2 connectivity probe
└── vibroagent_polling_logger_legacy.py   Superseded polling-based logger, kept for reference
PSD/                          Standalone Jupyter notebook comparing PSD estimators (Welch,
                              periodogram, direct one-sided FFT) on building-vibration data
```

## What is deliberately NOT in this repository

| Missing piece | Why | How to get it |
|---|---|---|
| **codes_v3 GGUF** (`qwen3_4b_codes_v3_Q4_0_embq8.gguf`, 2.4 GB) | Model weights, too large for git | `./setup_models.sh` — automatic download from the private HF weights repo (`HF_TOKEN`), **sha256-verified** `3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2`; see `models/README.md` |
| Stock STDATALOG-PYSDK (`stdatalog_core/`, `stdatalog_pnpl/`, `stdatalog_dtk/`, `stdatalog_gui/`) | It is ST's unmodified code | `./setup_sdk.sh` fetches the pinned v1.3.0 commits and applies the patches |
| GenieX runtime | Installable on demand | `./setup_geniex.sh` (PyPI) |
| QAIRT SDK (`v2.46.0.260424/`) | Qualcomm-licensed, 1.8 GB | `./setup_qairt.sh` (public Qualcomm Software Center URL); legacy genie backend only |
| Genie-compiled model (`models/merged_4k_6000s/`) | 4-bit compiled artifact; local build output, nowhere to download | Qualcomm Genie compile flow; legacy genie backend only |
| Codec + LLM **training data and pipelines** | Deliberately excluded (public-corpus codec pretraining and the 11k-example codes_v3 SFT set live in the research tree) | Not needed to run anything here |
| `vibrodiag_mcp_prototype/data/` beyond the token map, `validation_exports/` | Captured signals + external datasets, 300+ MB | — |
| ST examples/installers, GUI apps | Stock ST content unused by VibroAgent | ST GitHub, if ever needed |
| `vibroagent-venv/`, `.run/`, logs, backups | Machine state, not source | Recreated by `./setup.sh` / the launchers |

## Setup (fresh IQ-9075, Linux)

One command, run from the repository root, bootstraps everything:

```bash
./setup.sh
```

`setup.sh` downloads and installs **everything** automatically (all
environments are created with **uv**, which it also installs if missing):

0. **Linux USB prerequisites** — libusb-1.0, the `30-hsdatalog.rules` udev
   rules (ST vendor `0483`, DATALOG2 products `5743`/`5744` → group
   `hsdatalog`) and the `hsdatalog` group membership, exactly as on the
   production board (`setup_usb.sh`; needs sudo; log out/in once afterwards);
1. **stdatalog-pysdk** — stock v1.3.0 at the pinned commits + the
   `sdk_patches/` overlay (`setup_sdk.sh`);
2. the **app virtualenv** at `./vibroagent-venv` with an editable install of
   `vibrodiag_mcp_prototype`;
3. the **codec worker virtualenv** at `~/codec-cpu-venv` (CPU torch — runs
   the frozen codec-v1);
4. the **GenieX runtime** (`setup_geniex.sh`);
5. the **fine-tuned codes_v3 GGUF** (2.4 GB), pulled from the private
   Hugging Face weights repo and **sha256-verified** into
   `models/qwen3_4b_codes_v3_Q4_0_embq8.gguf` (`setup_models.sh` — needs
   `HF_TOKEN` in the environment or an `hf auth login` token on disk; the
   launcher picks the downloaded file up automatically).

Idempotent — re-running skips whatever is already in place.

For the legacy genie backend, add the QAIRT SDK download (1.8 GB):

```bash
./setup.sh --qairt
```

The genie backend also needs its compiled model bundle (see "Model backends").

The individual steps remain available as `setup_sdk.sh`, `setup_geniex.sh`,
`setup_qairt.sh`, `setup_models.sh`, `setup_usb.sh`.

USB permissions are handled by `setup_usb.sh` (step 0 above). After the
first run, log out and back in so the new `hsdatalog` group membership
applies to your session.

After setup, the app venv imports the patched SDK directly (a
`vibroagent_sdk.pth` file adds the four source trees to its `sys.path`):

```bash
./vibroagent-venv/bin/python -c "from stdatalog_core.HSD_link.HSDLink import HSDLink"
```

This works from any directory, including the repository root: ST ships an
empty `__init__.py` at each SDK package's top level which would otherwise
shadow the real inner package whenever Python starts with the repo root as
its working directory — `setup_sdk.sh` deletes those stubs after fetching, so
the services, the board-reader worker processes, and interactive Python all
resolve the patched SDK correctly.

## Running

```bash
# All-in-one launcher — codes stack by default (GenieX :18181 + logger + webchat):
./vibroagent.sh              # = start
./vibroagent.sh status
./vibroagent.sh stop
./vibroagent.sh restart      # stops, lets USB settle 6 s, starts
tail -f .run/*.log           # this launcher's logs + pidfiles land in ./.run/

# Backend-selectable per-service launcher, everything on 127.0.0.1:
./vibroagent_direct.sh start
tail -f vibrodiag_mcp_prototype/.run/*.log
```

Environment knobs (all overridable; defaults in the launchers):

- Stack: `VIBRO_STACK` (`codes` | `genie`), `VIBRO_MONITOR_DECISION_MODE`
  (`codes`), `VIBRO_CODES_AXES` (unset = z; e.g. `x,y,z`),
  `VIBRO_CODES_WORKER_PYTHON` (`~/codec-cpu-venv/bin/python`),
  `VIBRO_CODES_WORKER_TIMEOUT_S` (90)
- Codes model: `CODES_GGUF`, `CODES_GGUF_SHA256` (load refuses on mismatch),
  `GENIEX_PORT` (18181), `GENIEX_DEVICE_MAP`, `GENIEX_N_CTX` (6144),
  `GENIEX_PYTHON` (`~/geniex-venv/bin/python`), `GENIEX_MAX_OUTPUT_TOKENS`
- Legacy genie: `MODEL_ID` (`Qwen/Qwen3-4B-Instruct-2507`), `NPU_PORT` (8910),
  `QAIRT_SDK_ROOT`, `GENIE_CONFIG`, `GENIE_TIMEOUT_S` (300), `GENIE_READY_WAIT_S` (360)
- Webchat: `WEB_PORT` (7860); `WEB_HOST` defaults to the LAN-facing IP —
  `127.0.0.1` = local only, `0.0.0.0` = all interfaces
- Boards: `EXPECTED_BOARDS` (6), `VIBRO_BOARD_READER_PROCESSES` (0 disables
  process readers), `VIBRO_BOARD_READER_TIMEOUT_S` (30),
  `VIBRO_BOARD_READER_START_METHOD` (spawn), `VIBRO_BOARD_READER_PARALLELISM` (16)

### Security

The webchat has **no authentication**. `vibroagent.sh` binds it to the
LAN-interface IP by default (reachable on the LAN, not on loopback/Tailscale).
Do not expose it beyond a trusted network.

## Performance (board-measured)

| stage | figure |
|-------|--------|
| codec encode, 6 windows (CPU worker) | a few seconds |
| GenieX GGUF-on-HTP prefill | 342–422 tok/s |
| GenieX decode | ~6–11 tok/s |
| full 6-sensor codes verdict (≈1.9 K-token prompt) | well under 30 s end-to-end |
| logger fleet | 6 units x 26.7 kHz x 3 axes sustained over USB |

## Models

The frozen codec checkpoint is versioned in `models/codec_v1/`; the
fine-tuned codes_v3 GGUF is external with pinned hashes — both documented in
`models/README.md`. Training data and training pipelines are deliberately not
part of this repository.

## Tests

```bash
cd vibrodiag_mcp_prototype && PYTHONPATH=src python -m pytest tests/
```

The suite (283 tests) runs without hardware (mock vibrometers and demo
signals). Four template-provenance tests skip automatically because they
re-derive the prompt asset from the training set, which is not distributed.

## Scope and honest limitations

- **Relative judgments only.** The system reports deviations of targets
  against a reference window. It does not name damage types, does not
  localize faults structurally, and is a monitoring and triage tool, not a
  certified structural-safety assessment.
- **Single-axis contract.** codes_v3 was trained on single-axis windows; the
  live monitor feeds the z axis (never the vector norm).
- **The codec never saw this building.** Training data is public corpora
  only; the recorded examples here demonstrate the pipeline, they were never
  training material.
- **Examples are trimmed.** Each recorded acquisition is the first 60 s of a
  longer session, cut at exact STDatalog frame boundaries; the golden codes
  bind the first 10 s window.

## Ownership & License

This project is licensed under **CC BY-NC-SA 4.0** (Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International) — full terms and the
ownership statement in [`LICENSE.md`](LICENSE.md).

Owners: **Niks KORDJUKOVS** (System Research and Applications),
**Danilo Pau** (System Research and Applications, danilo.pau@st.com),
**STMicroelectronics SRL**.

Third-party components keep their own licenses: `sdk_patches/overlay/**` is
BSD-3-Clause © STMicroelectronics (`sdk_patches/LICENSE.md`), the SDK trees
fetched by `setup_sdk.sh` carry ST's licenses, and the base LLM
Qwen3-4B-Instruct-2507 is Apache-2.0; the fine-tuned codes_v3 GGUF and the
recorded datasets are covered by this project's license.
