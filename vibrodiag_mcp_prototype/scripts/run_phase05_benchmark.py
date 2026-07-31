#!/usr/bin/env python3
"""Run one arm of the Phase 0.5 benchmark against an OpenAI-compatible endpoint.

Replays the frozen eval prompts (built by build_phase05_eval_set.py) verbatim:
same messages, same temperature (0.0), same max_tokens, same json_schema
response_format. Works against the GenieX shim (:18181), vLLM, llama-server,
or any OpenAI-compatible server — stdlib only, no packages needed, so it can be
copied to any machine together with eval.jsonl.

Predictions are appended one JSON line at a time (flushed immediately); rerun
with the same --out to resume after an interruption. Latency and token usage
are recorded per example.

Examples:
  # Arm A: Qwen3-4B through the GenieX shim (deployment condition)
  python3 scripts/run_phase05_benchmark.py \
      --eval-jsonl data/phase05_benchmark/eval.jsonl \
      --base-url http://127.0.0.1:18181/v1 \
      --model 'unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0' \
      --arm qwen3_4b_geniex_npu \
      --out data/phase05_benchmark/predictions_qwen3_4b_geniex_npu.jsonl

  # Arm B: big model on another machine (vLLM)
  python3 scripts/run_phase05_benchmark.py \
      --eval-jsonl eval.jsonl --base-url http://HOST:8000/v1 \
      --model Qwen/Qwen3-235B-A22B-Instruct-2507 --arm qwen3_235b \
      --concurrency 8 --out predictions_qwen3_235b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def post_chat(
    base_url: str,
    api_key: str | None,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def call_example(
    example: dict[str, Any],
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    use_schema: bool,
    retries: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": example["messages"],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_schema and example.get("response_format"):
        payload["response_format"] = example["response_format"]

    last_error: str | None = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            response = post_chat(base_url, api_key, payload, timeout_s)
            latency_s = time.monotonic() - started
            text = str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            return {
                "example_id": example["example_id"],
                "ok": True,
                "text": text,
                "latency_s": round(latency_s, 3),
                "usage": response.get("usage"),
                "attempt": attempt,
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            last_error = f"http_{exc.code}: {detail}"
            # 4xx: the request itself is rejected (context overflow, bad schema);
            # retrying the same bytes cannot succeed.
            if 400 <= exc.code < 500:
                break
        except Exception as exc:  # URLError, timeout, connection reset, ...
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30.0, 2.0 ** (attempt + 1)))
    return {
        "example_id": example["example_id"],
        "ok": False,
        "text": None,
        "latency_s": None,
        "usage": None,
        "attempt": retries,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:18181/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--arm", required=True, help="arm name recorded in every prediction line")
    parser.add_argument("--out", required=True, help="predictions JSONL (append; reruns resume)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N pending examples")
    parser.add_argument("--concurrency", type=int, default=1, help="keep 1 for the single-flight NPU shim")
    parser.add_argument("--no-schema", action="store_true", help="omit response_format (prompt-guided JSON only)")
    args = parser.parse_args()

    eval_path = Path(args.eval_jsonl).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(eval_path)
    done_ids: set[str] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            if row.get("ok"):
                done_ids.add(str(row.get("example_id")))
    pending = [e for e in examples if e["example_id"] not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"arm={args.arm} examples={len(examples)} done={len(done_ids)} pending={len(pending)} "
        f"schema={not args.no_schema} concurrency={args.concurrency}",
        flush=True,
    )
    if not pending:
        print("nothing to do")
        return

    write_lock = threading.Lock()
    progress = {"done": 0, "failed": 0}
    out_file = out_path.open("a", encoding="utf-8")

    def run_one(example: dict[str, Any]) -> None:
        result = call_example(
            example,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            use_schema=not args.no_schema,
            retries=args.retries,
        )
        result["arm"] = args.arm
        result["model"] = args.model
        result["schema_enforced"] = not args.no_schema
        with write_lock:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            progress["done"] += 1
            if not result["ok"]:
                progress["failed"] += 1
            n = progress["done"]
            if result["ok"]:
                lat = result.get("latency_s")
                usage = result.get("usage") or {}
                out_tok = usage.get("completion_tokens")
                print(
                    f"[{n}/{len(pending)}] {example['example_id']}: ok latency={lat}s out_tokens={out_tok}",
                    flush=True,
                )
            else:
                print(f"[{n}/{len(pending)}] {example['example_id']}: FAILED {result['error']}", flush=True)

    started = time.monotonic()
    if args.concurrency <= 1:
        for example in pending:
            run_one(example)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(run_one, pending))
    elapsed = time.monotonic() - started
    print(
        f"finished: {progress['done'] - progress['failed']} ok, {progress['failed']} failed, "
        f"{elapsed / 60:.1f} min total, predictions -> {out_path}",
        flush=True,
    )
    out_file.close()


if __name__ == "__main__":
    main()
