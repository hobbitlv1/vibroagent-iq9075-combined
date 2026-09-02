#!/usr/bin/env python3
"""Validate that a listening VibroAgent-Gemma server is the exact deployable runtime."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

EXPECTED_CONTRACT = "vibrogemma-geniex-live-v2"
EXPECTED_DECISION_MODE = "parallel_binary_global_plus_joint_target_set_v1"
EXPECTED_LABEL_CONTRACT = {
    "classes": ["normal", "unknown_anomaly"],
    "quality_class": "data_invalid",
    "severities": ["none", "advisory"],
}


def validate_health(
    payload: Any,
    *,
    model_name: str,
    model_sha256: str,
    manifest_sha256: str,
    device_map: str,
    n_ctx: int,
    decision_mode: str = EXPECTED_DECISION_MODE,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["health response is not a JSON object"]

    expected = {
        "status": "ok",
        "deployment_contract_version": EXPECTED_CONTRACT,
        "model": model_name,
        "model_sha256": model_sha256.lower(),
        "bundle_manifest_sha256": manifest_sha256.lower(),
        "geniex_version": "0.4.0",
        "device_map": device_map,
        "n_ctx": n_ctx,
        "external_embeddings": True,
        "decision_mode": decision_mode,
        "label_contract": EXPECTED_LABEL_CONTRACT,
    }
    for key, expected_value in expected.items():
        actual = payload.get(key)
        if actual != expected_value:
            errors.append(f"{key} mismatch: {actual!r} != {expected_value!r}")

    probe = payload.get("runtime_probe")
    if not isinstance(probe, dict) or probe.get("passed") is not True:
        errors.append(f"runtime_probe did not pass: {probe!r}")
    elif probe.get("external_embedding_decode") is not True:
        errors.append("runtime_probe did not verify external-embedding decode")
    elif decision_mode == EXPECTED_DECISION_MODE and probe.get("state_logits") is not True:
        errors.append("runtime_probe did not verify state_logits")
    elif (
        decision_mode == "factorized_global_plus_independent_targets_v1"
        and probe.get("generate_from_state") is not True
    ):
        errors.append("runtime_probe did not verify generate_from_state")

    return errors


def fetch_health(url: str, timeout_s: float) -> Any:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout_s) as response:  # noqa: S310 - fixed local deployment URL.
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--device-map", required=True)
    parser.add_argument("--n-ctx", required=True, type=int)
    parser.add_argument("--decision-mode", default=EXPECTED_DECISION_MODE)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    try:
        payload = fetch_health(args.url, args.timeout)
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"VibroAgent-Gemma health check failed: {exc}", file=sys.stderr)
        return 1

    errors = validate_health(
        payload,
        model_name=args.model_name,
        model_sha256=args.model_sha256,
        manifest_sha256=args.manifest_sha256,
        device_map=args.device_map,
        n_ctx=args.n_ctx,
        decision_mode=args.decision_mode,
    )
    if errors:
        print("VibroAgent-Gemma deployment contract mismatch:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "VibroAgent-Gemma deployment verified: "
        f"{args.model_name}, {EXPECTED_DECISION_MODE}, {EXPECTED_CONTRACT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
