#!/usr/bin/env python3
"""Run one bundled LUMO window through the VibroAgent-Gemma encoder and model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "vibrodiag_mcp_prototype"
for path in (PROTO / "src", PROTO):
    sys.path.insert(0, str(path))

from scripts.live_vibrogemma_worker import encode_live_window  # noqa: E402
from vibroagent_mcp.vibrogemma_live import SLOTS, _lumo_training_replay  # noqa: E402


def _vibration_payload(target: str, bundle: Path) -> dict:
    arrays, rates, capture, timing, positions, masks = _lumo_training_replay(target)
    episode_id = f"lumo-demo-{target}"
    encoder_input = {
        **arrays,
        **{f"fs_hz__{slot}": rates[slot] for slot in SLOTS},
        **{f"metadata__{slot}": capture[slot] for slot in SLOTS},
        **{f"sensor_position__{slot}": positions[slot] for slot in SLOTS},
        **{f"axis_mask__{slot}": masks[slot] for slot in SLOTS},
        "episode_id": episode_id,
        "window_start_utc": timing["window_start_utc"],
        "window_end_utc": timing["window_end_utc"],
        "window_time_source": timing["window_time_source"],
        "capture_metadata": timing,
    }
    return encode_live_window(encoder_input, bundle, amplitude_calibrated=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("target_3", "target_5"), default="target_3")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models" / "vibroagent-gemma-g1")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18181/v1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="encode and verify without calling GenieX")
    args = parser.parse_args()

    vibration = _vibration_payload(args.target, args.bundle.resolve())
    embedding_sha256 = hashlib.sha256(base64.b64decode(vibration["embeddings_b64"])).hexdigest()
    summary = {
        "target": args.target,
        "shape": vibration["shape"],
        "embedding_sha256": embedding_sha256,
        "source": vibration["metadata"].get("pipeline_source"),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    request_body = json.dumps(
        {
            "model": "vibrogemma",
            "messages": [{"role": "user", "content": "Classify this LUMO vibration episode."}],
            "temperature": 0,
            "max_tokens": 320,
            "vibration": vibration,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=request_body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = json.loads(payload["choices"][0]["message"]["content"])
    print(json.dumps({**summary, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
