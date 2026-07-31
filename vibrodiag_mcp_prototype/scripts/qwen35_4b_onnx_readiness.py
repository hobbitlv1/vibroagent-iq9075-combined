#!/usr/bin/env python3
"""Check whether the current machine is ready for a Qwen3.5-4B ONNX export."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_MIN_FREE_GB = 35.0
HF_API_ROOT = "https://huggingface.co/api/models"
HF_RAW_ROOT = "https://huggingface.co"
REQUIRED_EXPORT_PACKAGES = (
    "torch",
    "transformers",
    "huggingface_hub",
    "onnx",
    "onnxscript",
    "onnxruntime_genai",
)
REQUIRED_SERVE_PACKAGES = ("onnxruntime_genai",)


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "vibroagent-qwen35-onnx-readiness"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object response from {url}")
    return payload


def _model_api_url(model_id: str) -> str:
    return f"{HF_API_ROOT}/{model_id}?blobs=true"


def _model_config_url(model_id: str) -> str:
    return f"{HF_RAW_ROOT}/{model_id}/raw/main/config.json"


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024**3)


def _weight_bytes(model_info: dict[str, Any]) -> int:
    total = 0
    siblings = model_info.get("siblings")
    if not isinstance(siblings, list):
        return total
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        name = str(sibling.get("rfilename") or "")
        if name.endswith((".safetensors", ".bin", ".pt")):
            try:
                total += int(sibling.get("size") or 0)
            except (TypeError, ValueError):
                pass
    return total


def _missing_packages(packages: tuple[str, ...]) -> list[str]:
    return [package for package in packages if importlib.util.find_spec(package) is None]


def _disk_free_gb(path: Path) -> float:
    path = path.expanduser()
    existing = path if path.exists() else path.parent
    usage = shutil.disk_usage(existing)
    return _bytes_to_gb(usage.free)


def _print_status(label: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "BLOCKED"
    print(f"[{marker}] {label}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="models/qwen3_5_4b_onnx_int4_cpu")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    blocked = False

    print(f"Model: {args.model_id}")
    print(f"Output directory: {output_dir}")

    try:
        model_info = _fetch_json(_model_api_url(args.model_id))
        config = _fetch_json(_model_config_url(args.model_id))
    except urllib.error.HTTPError as exc:
        print(f"[BLOCKED] Hugging Face metadata: HTTP {exc.code} for {exc.url}")
        return 2
    except Exception as exc:
        print(f"[BLOCKED] Hugging Face metadata: {exc.__class__.__name__}: {exc}")
        return 2

    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    expected_architecture = "Qwen3_5ForConditionalGeneration"
    arch_ok = architecture == expected_architecture
    _print_status(
        "architecture",
        arch_ok,
        f"{architecture!r}; expected {expected_architecture!r} for the current ONNX GenAI builder path",
    )
    blocked = blocked or not arch_ok

    gated = bool(model_info.get("gated"))
    _print_status("model access", not gated, "public model" if not gated else "gated model requires HF login/token")
    blocked = blocked or gated

    weights_gb = _bytes_to_gb(_weight_bytes(model_info))
    used_storage = model_info.get("usedStorage")
    storage_gb = _bytes_to_gb(int(used_storage)) if isinstance(used_storage, int) else weights_gb
    print(f"Approximate HF storage: {storage_gb:.1f} GB; weight files: {weights_gb:.1f} GB")

    free_gb = _disk_free_gb(output_dir)
    required_gb = max(args.min_free_gb, weights_gb * 3.0)
    disk_ok = free_gb >= required_gb
    _print_status(
        "disk",
        disk_ok,
        f"{free_gb:.1f} GB free; recommend at least {required_gb:.1f} GB for weights, export cache, and ONNX external data",
    )
    blocked = blocked or not disk_ok

    missing_export = _missing_packages(REQUIRED_EXPORT_PACKAGES)
    _print_status(
        "export packages",
        not missing_export,
        "installed" if not missing_export else "missing " + ", ".join(missing_export),
    )
    blocked = blocked or bool(missing_export)

    missing_serve = _missing_packages(REQUIRED_SERVE_PACKAGES)
    _print_status(
        "serve packages",
        not missing_serve,
        "installed" if not missing_serve else "missing " + ", ".join(missing_serve),
    )

    print()
    if blocked:
        print("Readiness result: blocked. Do not start the full Qwen3.5-4B export on this machine yet.")
        print("Next steps:")
        print("  1. Free or mount enough storage for the model/cache/export output.")
        print("  2. Install export dependencies with: pip install -e '.[qwen,onnx,onnx-export]'")
        print("  3. Re-run this readiness check before downloading weights.")
        return 2

    print("Readiness result: OK to attempt the Qwen3.5-4B text-only ONNX export.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
