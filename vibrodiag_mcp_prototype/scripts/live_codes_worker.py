#!/usr/bin/env python3
"""Encode live sensor windows into codes_v3 model inputs (single shot).

Runs ONLY under the codec-cpu-venv (torch 2.13.0+cpu binding):

  /home/ubuntu/codec-cpu-venv/bin/python scripts/live_codes_worker.py \
      --windows /path/windows.npz --out /path/codes.json

Input npz: one float array per slot (``baseline`` required, then any of
``target_1``..``target_5``) holding raw acceleration in g, plus either a
shared scalar ``fs_hz`` or one ``fs_hz__<slot>`` scalar per window. Each
record must cover at least 10 s; the worker uses the most recent 10 s.

Chain per slot (mirrors the rt345/codes_v3 build recipe):
  last 10 s -> DC removal over the record -> resample_poly to 400 Hz
  (exact rational, reduced) -> trim to 4000 samples -> codec_dataset
  normalize_window (detrend + unit AC rms; level_db = 20*log10(ac_rms))
  -> frozen codec-v1 checkpoint (step 29000) -> 125x2 codes ->
  CodeSymbolMap serialization (250 single-codepoint symbols).

level_rel_db per target = level_db(target) - level_db(baseline).
Output JSON carries codes + levels + full provenance. The prompt itself
is built by vibroagent_mcp.live_codes on the webchat side.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repr_extractor import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_TOKEN_MAP,
    CodecV1Extractor,
    CodeSymbolMap,
    sha256_file,
)

FS_CODEC = 400
WINDOW_S = 10.0
WINDOW_SAMPLES = 4000
SLOTS = ("baseline", "target_1", "target_2", "target_3", "target_4",
         "target_5")
RATE_PREFIX = "fs_hz__"


def resample_to_codec_rate(record: np.ndarray, fs_hz: float) -> np.ndarray:
    """Most-recent 10 s -> DC-removed -> exactly 4000 samples at 400 Hz."""
    from scipy.signal import resample_poly

    x = np.asarray(record, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"record must be 1-D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("record contains nonfinite samples")
    fs = float(fs_hz)
    if not fs > 0:
        raise ValueError(f"fs_hz must be positive, got {fs}")
    native_window = int(round(WINDOW_S * fs))
    if x.size < native_window:
        raise ValueError(
            f"record too short: {x.size} samples < {native_window} "
            f"({WINDOW_S:.0f} s at {fs:g} Hz)")
    x = x[-native_window:]
    x = x - x.mean()
    if abs(fs - FS_CODEC) < 1e-9:
        y = x
    else:
        ratio = Fraction(FS_CODEC, int(round(fs))).limit_denominator(10**9)
        if abs(float(ratio) * fs - FS_CODEC) > 1e-6:
            raise ValueError(
                f"fs_hz {fs:g} has no exact rational path to {FS_CODEC} Hz")
        y = resample_poly(x, ratio.numerator, ratio.denominator)
    if y.size < WINDOW_SAMPLES:
        raise ValueError(
            f"resampled record has {y.size} < {WINDOW_SAMPLES} samples")
    return y[:WINDOW_SAMPLES]


def encode_windows(windows: dict[str, np.ndarray],
                   fs_hz: float | dict[str, float], *,
                   checkpoint: Path, split_manifest: Path,
                   token_map: Path, torch_threads: int) -> dict:
    if "baseline" not in windows:
        raise ValueError("baseline window is required")
    unknown = sorted(set(windows) - set(SLOTS))
    if unknown:
        raise ValueError(f"unknown slots: {unknown}")
    slots = [slot for slot in SLOTS if slot in windows]

    if isinstance(fs_hz, dict):
        missing_rates = sorted(set(slots) - set(fs_hz))
        unknown_rates = sorted(set(fs_hz) - set(slots))
        if missing_rates:
            raise ValueError(f"missing sampling rates for slots: {missing_rates}")
        if unknown_rates:
            raise ValueError(f"sampling rates supplied for unknown slots: {unknown_rates}")
        rates = {slot: float(fs_hz[slot]) for slot in slots}
    else:
        rates = {slot: float(fs_hz) for slot in slots}

    prepared = {slot: resample_to_codec_rate(windows[slot], rates[slot])
                for slot in slots}
    extractor = CodecV1Extractor(
        checkpoint, split_manifest=split_manifest,
        torch_threads=torch_threads)
    symbol_map = CodeSymbolMap(
        token_map,
        codebooks=int(extractor.config.codebooks),
        codebook_size=int(extractor.config.codebook_size),
    )
    bundles = extractor.extract_batch([prepared[slot] for slot in slots])

    unique_rates = set(rates.values())
    result: dict = {
        "schema": "live-codes-worker-v1",
        "fs_native_hz": next(iter(unique_rates)) if len(unique_rates) == 1 else None,
        "fs_native_hz_by_slot": rates,
        "fs_codec_hz": FS_CODEC,
        "window_s": WINDOW_S,
        "slots": {},
        "provenance": {
            **extractor.provenance,
            "token_map": str(token_map),
            "token_map_sha256": sha256_file(token_map),
        },
    }
    baseline_level = None
    for slot, bundle in zip(slots, bundles):
        codes_text = symbol_map.encode(bundle.codes)
        entry = {
            "codes": codes_text,
            "n_codes": len(codes_text),
            "level_db": float(bundle.level_db),
        }
        if slot == "baseline":
            baseline_level = float(bundle.level_db)
        result["slots"][slot] = entry
    for slot, entry in result["slots"].items():
        if slot != "baseline":
            entry["level_rel_db"] = entry["level_db"] - baseline_level
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True,
                        help="npz with per-slot raw g arrays + fs_hz scalar")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-manifest", type=Path,
                        default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--token-map", type=Path, default=DEFAULT_TOKEN_MAP)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    payload = np.load(args.windows)
    rate_names = [name for name in payload.files if name.startswith(RATE_PREFIX)]
    if rate_names and "fs_hz" in payload.files:
        raise SystemExit(
            "windows npz must carry either shared fs_hz or per-slot "
            "fs_hz__<slot> scalars, not both")
    if rate_names:
        fs_hz: float | dict[str, float] = {
            name[len(RATE_PREFIX):]: float(np.asarray(payload[name]).reshape(()))
            for name in rate_names
        }
    elif "fs_hz" in payload.files:
        fs_hz = float(np.asarray(payload["fs_hz"]).reshape(()))
    else:
        raise SystemExit(
            "windows npz must carry shared fs_hz or per-slot fs_hz__<slot> scalars")
    windows = {
        name: payload[name] for name in payload.files
        if name != "fs_hz" and not name.startswith(RATE_PREFIX)
    }
    result = encode_windows(
        windows, fs_hz, checkpoint=args.checkpoint,
        split_manifest=args.split_manifest, token_map=args.token_map,
        torch_threads=args.torch_threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"encoded {sorted(result['slots'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
