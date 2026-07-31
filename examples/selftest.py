#!/usr/bin/env python3
"""Verify the frozen chain: decode examples -> codec-v1 codes == golden.

Runs entirely offline (no LLM) through the PRODUCTION worker implementation
(``live_codes_worker.encode_windows``).  PASS means your environment
reproduces the proven live encoder byte-for-byte: STDatalog decode,
exact-rational resampling, window normalization, the frozen codec-v1
checkpoint and the code-symbol serialization all agree with the recorded
golden output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vibrodiag_mcp_prototype" / "scripts"))
sys.path.insert(0, str(REPO / "examples"))

from live_codes_worker import encode_windows  # noqa: E402
from stdatalog_reader import read_acquisition  # noqa: E402

SLOT_FOLDERS = {
    "baseline": "live_baseline", "target_1": "live_target_1",
    "target_2": "live_target_2", "target_3": "live_target_3",
    "target_4": "live_target_4", "target_5": "live_target_5",
}


def main() -> int:
    golden = json.loads(
        (REPO / "examples/golden_codes.json").read_text(encoding="utf-8"))
    axis = golden["window"]["axis"]
    offset_s = float(golden["window"]["offset_s"])

    windows, rates = {}, {}
    for slot, folder in SLOT_FOLDERS.items():
        stream = read_acquisition(REPO / "examples" / folder)
        fs = float(round(stream.fs_hz))
        start = int(round(offset_s * fs))
        n = int(round(10.0 * fs))
        windows[slot] = stream.axis(axis)[start:start + n]
        rates[slot] = fs

    encoded = encode_windows(
        windows, rates,
        checkpoint=REPO / "models/codec_v1/best.pt",
        split_manifest=REPO / "models/codec_v1/split_manifest.json",
        token_map=(REPO / "vibrodiag_mcp_prototype/data/codec_pretrain/"
                   "r1_token_map.json"),
        torch_threads=4)
    failures = 0
    for slot, want in golden["slots"].items():
        got = encoded["slots"][slot]
        codes_ok = got["codes"] == want["codes"]
        level_ok = float(got["level_db"]) == float(want["level_db"])
        status = "ok" if codes_ok and level_ok else "MISMATCH"
        print(f"  {slot:9s} codes={'ok' if codes_ok else 'MISMATCH'} "
              f"level={'ok' if level_ok else 'MISMATCH'}  [{status}]")
        failures += 0 if codes_ok and level_ok else 1
    checkpoint_sha = encoded["provenance"]["checkpoint_sha256"]
    print(f"  checkpoint sha256 {checkpoint_sha[:16]}... step "
          f"{encoded['provenance']['checkpoint_step']}")
    if failures:
        print(f"SELFTEST FAIL: {failures} slot(s) drifted")
        return 1
    print("SELFTEST PASS: byte-identical to the proven live encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
