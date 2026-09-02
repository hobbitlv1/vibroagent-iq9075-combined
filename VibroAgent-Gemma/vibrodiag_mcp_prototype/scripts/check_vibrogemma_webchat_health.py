#!/usr/bin/env python3
"""Validate that the live UI process includes the fail-closed VibroGemma wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

EXPECTED = {
    "ok": True,
    "app": "vibroagent",
    "inference": "gemma",
    "deployment_contract_version": "vibroagent-gemma-web-live-v2",
    "failure_policy": "data_invalid_never_inherited_normal",
    "monitor_decision_mode": "vibrogemma",
}


def validate_health(
    payload: object,
    *,
    model_base_url: str,
    model: str,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["health response is not a JSON object"]
    expected = {
        **EXPECTED,
        "model_base_url": model_base_url.rstrip("/"),
        "model": model,
    }
    return [
        f"{key} mismatch: {payload.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(args.url, timeout=args.timeout) as response:  # noqa: S310 - local service URL.
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"VibroAgent webchat health check failed: {exc}", file=sys.stderr)
        return 1
    errors = validate_health(
        payload,
        model_base_url=args.model_base_url,
        model=args.model,
    )
    if errors:
        print("VibroAgent webchat deployment contract mismatch:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("VibroAgent webchat deployment verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
