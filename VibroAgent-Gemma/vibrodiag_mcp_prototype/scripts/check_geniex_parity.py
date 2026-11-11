#!/usr/bin/env python3
"""Compare a running model server's decision logits with numbers recorded on an earlier runtime.

Runs the two bundled LUMO windows through the server and reports, per window, whether every binary
answer matches the reference and how far the choice logits moved. Use it after changing the GenieX
release, the device (CPU or NPU), or the thread count:

    check_geniex_parity.py --list
    check_geniex_parity.py --reference geniex-0.4.0-cpu-8-threads --bundle <bundle dir>

The exit status is 0 when every answer matches the reference, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROTO = Path(__file__).resolve().parents[1]
ROOT = PROTO.parent
DEFAULT_REFERENCE_FILE = PROTO / "patches" / "parity_reference.json"
WINDOWS = ("target_3", "target_5")


def compare_window(reference: dict, observed: dict) -> dict:
    """Compare one window's choice logits; each row is the pair of logits for one binary answer."""
    ref_logits = reference["choice_logits"]
    got_logits = observed["choice_logits"]
    if len(ref_logits) != len(got_logits) or any(len(a) != 2 or len(b) != 2 for a, b in zip(ref_logits, got_logits)):
        raise ValueError("reference and observed logits do not have the same answer positions")
    ref_answers = [row[1] > row[0] for row in ref_logits]
    got_answers = [row[1] > row[0] for row in got_logits]
    differing = [index for index, (a, b) in enumerate(zip(ref_answers, got_answers)) if a != b]
    return {
        "same_input": reference["embedding_sha256"] == observed["embedding_sha256"],
        "answers_match": not differing,
        "differing_positions": differing,
        "identical": ref_logits == got_logits,
        "max_logit_diff": max(abs(a - b) for ref, got in zip(ref_logits, got_logits) for a, b in zip(ref, got)),
        "reference_margins": [row[1] - row[0] for row in ref_logits],
        "observed_margins": [row[1] - row[0] for row in got_logits],
    }


def run_window(target: str, bundle: Path, endpoint: str, timeout: float) -> dict:
    command = [
        sys.executable, str(ROOT / "demo" / "run_lumo.py"),
        "--target", target, "--bundle", str(bundle), "--endpoint", endpoint, "--timeout", str(timeout),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{target}: demo run failed: {completed.stderr.strip()[-400:]}")
    document = json.loads(completed.stdout)
    model = document["result"]["model"]
    return {
        "embedding_sha256": document["embedding_sha256"],
        "choice_logits": model["gemma_binary_choice_logits"],
        "model_sha256": model.get("checkpoint_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", help="name of a recorded reference (see --list)")
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--bundle", type=Path, help="model bundle the running server was started with")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18181/v1")
    parser.add_argument("--timeout", type=float, default=400.0)
    parser.add_argument("--list", action="store_true", help="list the recorded references and exit")
    args = parser.parse_args()

    recorded = json.loads(args.reference_file.read_text())
    references = recorded["references"]
    if args.list:
        for name, entry in references.items():
            print(f"{name}: GenieX {entry['geniex_version']}, {entry['device_map']}, "
                  f"{entry['n_threads']} threads. {entry.get('note', '')}".rstrip())
        return 0
    if not args.reference or not args.bundle:
        parser.error("--reference and --bundle are required unless --list is given")
    if args.reference not in references:
        parser.error(f"unknown reference {args.reference!r}; choose from: {', '.join(references)}")
    reference = references[args.reference]

    all_match = True
    for target in WINDOWS:
        observed = run_window(target, args.bundle.resolve(), args.endpoint, args.timeout)
        if observed["model_sha256"] != recorded["model_sha256"]:
            print(f"{target}: the server runs a different model ({observed['model_sha256']}) than the reference "
                  f"({recorded['model_sha256']}); these numbers are not comparable")
            return 1
        outcome = compare_window(reference["windows"][target], observed)
        all_match = all_match and outcome["same_input"] and outcome["answers_match"]
        print(f"{target}:")
        print(f"  reference margins  {[round(value, 3) for value in outcome['reference_margins']]}")
        print(f"  observed margins   {[round(value, 3) for value in outcome['observed_margins']]}")
        print(f"  same encoder input: {outcome['same_input']}; all answers match: {outcome['answers_match']}"
              + (f" (differ at positions {outcome['differing_positions']})" if outcome["differing_positions"] else "")
              + f"; logits identical: {outcome['identical']}; largest logit difference: {outcome['max_logit_diff']:.6f}")
    print("PASS: every answer matches the reference" if all_match else "FAIL: answers or inputs differ from the reference")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
