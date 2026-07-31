#!/usr/bin/env python3
"""End-to-end offline demo: recorded .dat acquisitions -> codes -> LLM verdict.

Runs the EXACT production implementation — ``live_codes_worker.encode_windows``
(the codec worker the webchat invokes on the board) and
``vibroagent_mcp.live_codes`` (the byte-exact prompt builder / strict parser) —
on the recorded example acquisitions, so the full system can be exercised
without any board:

  iis3dwb_acc.dat (26.7 kHz, 3-axis, g)
    -> z axis, one 10 s window per sensor slot
    -> frozen codec-v1 -> 250 code symbols + broadband level delta
    -> byte-exact codes_v3 prompt
    -> finetuned Qwen3-4B GGUF via any OpenAI-compatible endpoint
    -> strictly parsed relative-to-baseline verdict.

Examples:

  # 1) Inspect the pipeline without any LLM (codes + prompt only):
  python examples/run_example.py --dry-run

  # 2) Full verdict against a local llama.cpp server:
  #    llama-server -m qwen3_4b_codes_v3_Q4_0_embq8.gguf -c 6144 --port 8080
  python examples/run_example.py --endpoint http://localhost:8080/v1

  # 3) On the board, against the geniex HTP service:
  python examples/run_example.py --endpoint http://localhost:18181/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vibrodiag_mcp_prototype" / "scripts"))
sys.path.insert(0, str(REPO / "vibrodiag_mcp_prototype" / "src"))
sys.path.insert(0, str(REPO / "examples"))

from live_codes_worker import WINDOW_S, encode_windows  # noqa: E402
from stdatalog_reader import read_acquisition  # noqa: E402
from vibroagent_mcp.live_codes import (  # noqa: E402
    LIVE_SLOTS,
    build_codes_user_prompt,
    codes_response_format,
    parse_codes_reply,
    system_message,
)

DEFAULT_CHECKPOINT = REPO / "models" / "codec_v1" / "best.pt"
DEFAULT_SPLIT_MANIFEST = REPO / "models" / "codec_v1" / "split_manifest.json"
DEFAULT_TOKEN_MAP = (REPO / "vibrodiag_mcp_prototype" / "data" /
                     "codec_pretrain" / "r1_token_map.json")

SLOT_FOLDERS = {
    "baseline": "live_baseline",
    "target_1": "live_target_1",
    "target_2": "live_target_2",
    "target_3": "live_target_3",
    "target_4": "live_target_4",
    "target_5": "live_target_5",
}


def load_windows(examples_root: Path, axis: str, offset_s: float,
                 ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """One window per available slot, all cut at the same record offset.

    Each STWIN.box unit runs on its own clock, so the measured output data
    rate differs slightly per sensor; rates travel per slot, rounded to an
    integer exactly like the live monitor feeds the worker."""
    windows: dict[str, np.ndarray] = {}
    rates: dict[str, float] = {}
    for slot, folder in SLOT_FOLDERS.items():
        root = examples_root / folder
        if not (root / "iis3dwb_acc.dat").is_file():
            continue
        stream = read_acquisition(root)
        fs_hz = float(round(stream.fs_hz))
        signal = stream.axis(axis)
        n_window = int(round(WINDOW_S * fs_hz))
        start = int(round(offset_s * fs_hz))
        if start + n_window > signal.size:
            raise SystemExit(
                f"{slot}: window at offset {offset_s:.1f}s exceeds the "
                f"{signal.size / fs_hz:.1f}s recording")
        windows[slot] = signal[start:start + n_window]
        rates[slot] = fs_hz
        print(f"  {slot:9s} <- {folder}/iis3dwb_acc.dat "
              f"({signal.size / fs_hz:.1f}s at {fs_hz:.0f} Hz, window at "
              f"{offset_s:.1f}-{offset_s + WINDOW_S:.1f}s)")
    if "baseline" not in windows:
        raise SystemExit(f"no baseline acquisition under {examples_root}")
    return windows, rates


def call_llm(endpoint: str, model: str, messages: list[dict], schema: dict | None,
             timeout_s: float) -> str:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if schema is not None:
        body["response_format"] = schema
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-root", type=Path,
                        default=REPO / "examples")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z",
                        help="single axis fed to the codec (the codes_v3 "
                             "training contract is single-axis; z default)")
    parser.add_argument("--offset-s", type=float, default=0.0,
                        help="window start inside each recording (seconds)")
    parser.add_argument("--endpoint", default="http://localhost:8080/v1",
                        help="OpenAI-compatible base URL serving the codes_v3 GGUF")
    parser.add_argument("--model", default="qwen3_4b_codes_v3",
                        help="model name field passed to the endpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-manifest", type=Path,
                        default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--token-map", type=Path, default=DEFAULT_TOKEN_MAP)
    parser.add_argument("--no-schema", action="store_true",
                        help="do not send response_format json_schema "
                             "(for servers without grammar support)")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="stop after printing codes and the exact prompt")
    parser.add_argument("--save-prompt", type=Path, default=None,
                        help="write the assembled messages JSON here")
    args = parser.parse_args()

    print(f"[1/4] decoding acquisitions under {args.examples_root} "
          f"(axis={args.axis}) ...")
    windows, rates = load_windows(args.examples_root, args.axis, args.offset_s)

    print(f"[2/4] encoding {len(windows)} windows through frozen codec-v1 ...")
    encoded = encode_windows(
        windows, rates, checkpoint=args.checkpoint,
        split_manifest=args.split_manifest, token_map=args.token_map,
        torch_threads=4)
    slots = encoded["slots"]
    for slot in sorted(slots):
        entry = slots[slot]
        rel = (f"  level_rel_db={entry['level_rel_db']:+7.2f}"
               if "level_rel_db" in entry else "  (reference)")
        print(f"  {slot:9s} {entry['n_codes']} codes{rel}")

    targets = [(slot, slots[slot]["level_rel_db"], slots[slot]["codes"])
               for slot in LIVE_SLOTS if slot in slots]
    target_names = [slot for slot, _, _ in targets]
    print("[3/4] building the byte-exact codes_v3 prompt ...")
    user_prompt = build_codes_user_prompt(
        slots["baseline"]["codes"], targets, aligned=True)
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "user", "content": user_prompt},
    ]
    if args.save_prompt:
        args.save_prompt.write_text(
            json.dumps(messages, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"  messages -> {args.save_prompt}")

    if args.dry_run:
        print("--- system ---")
        print(messages[0]["content"])
        print("--- user (long code lines elided) ---")
        for line in user_prompt.splitlines():
            print(line if len(line) < 200 else line[:160] + " ...")
        print("[dry-run] stopping before the LLM call")
        return 0

    print(f"[4/4] querying {args.endpoint} (model={args.model}) ...")
    schema = None if args.no_schema else codes_response_format(target_names)
    reply = call_llm(args.endpoint, args.model, messages, schema,
                     args.timeout_s)
    verdict = parse_codes_reply(reply, target_names)

    print("\n=== verdict (relative to baseline sensor) ===")
    for slot in target_names:
        print(f"  {slot:9s} {verdict['sensor_reports'][slot]}")
    print(f"  network   {verdict['network_status']}")
    print(f"  affected  {verdict['affected_sensor_ids'] or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
