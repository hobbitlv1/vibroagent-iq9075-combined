#!/usr/bin/env python3
"""Evaluate a Qwen vibration all-sensor LoRA adapter on SFT JSONL examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all-sensor VibroAgent SFT JSON accuracy")
    parser.add_argument("--test-jsonl", default="data/vibration_sft/test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter directory")
    parser.add_argument("--output-jsonl", default="outputs/vibration_eval_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    test_path = Path(args.test_jsonl).expanduser().resolve()
    if not test_path.exists():
        raise SystemExit(f"Test JSONL does not exist: {test_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    examples = list(read_jsonl(test_path))
    if args.limit is not None:
        examples = examples[: args.limit]

    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = new_counts()
    by_task: dict[str, dict[str, int]] = {}
    with output_path.open("w", encoding="utf-8") as out:
        for example in examples:
            gold_payload = parse_json_payload(example["messages"][-1]["content"])
            task = infer_task(example, gold_payload)
            prompt_messages = [m for m in example["messages"] if m.get("role") != "assistant"]
            prediction_text = generate(model, tokenizer, prompt_messages, args.max_new_tokens, args.temperature)
            pred_payload = parse_json_payload(prediction_text)
            score = score_prediction(gold_payload, pred_payload, example)
            add_counts(totals, score)
            add_counts(by_task.setdefault(task, new_counts()), score)
            out.write(json.dumps({
                "task": task,
                **score,
                "prediction_text": prediction_text,
                "prediction_json": pred_payload,
                "gold_json": gold_payload,
                "source": example.get("source"),
                "metadata": example.get("metadata"),
            }, ensure_ascii=False) + "\n")

    print_counts("Overall", totals)
    for task, counts in sorted(by_task.items()):
        print_counts(task, counts)
    print(f"Predictions written to {output_path}")


def new_counts() -> dict[str, int]:
    return {
        "examples_total": 0,
        "examples_exact": 0,
        "sensor_total": 0,
        "sensor_correct": 0,
        "network_total": 0,
        "network_correct": 0,
        "invalid_json": 0,
    }


def add_counts(total: dict[str, int], score: dict[str, Any]) -> None:
    total["examples_total"] += int(score.get("examples_total", 1))
    total["examples_exact"] += int(bool(score.get("example_exact")))
    total["sensor_total"] += int(score.get("sensor_total") or 0)
    total["sensor_correct"] += int(score.get("sensor_correct") or 0)
    total["network_total"] += int(score.get("network_total") or 0)
    total["network_correct"] += int(score.get("network_correct") or 0)
    total["invalid_json"] += int(not bool(score.get("prediction_json_valid")))


def print_counts(name: str, counts: dict[str, int]) -> None:
    exact = ratio(counts["examples_exact"], counts["examples_total"])
    sensor = ratio(counts["sensor_correct"], counts["sensor_total"])
    network = ratio(counts["network_correct"], counts["network_total"])
    print(f"{name}: exact={counts['examples_exact']}/{counts['examples_total']} ({exact:.3f}), "
          f"sensor={counts['sensor_correct']}/{counts['sensor_total']} ({sensor:.3f}), "
          f"network={counts['network_correct']}/{counts['network_total']} ({network:.3f}), "
          f"invalid_json={counts['invalid_json']}")


def ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def score_prediction(gold: dict[str, Any], pred: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    gold_sensors = sensor_label_map(gold)
    pred_sensors = sensor_label_map(pred)
    gold_network = str(gold.get("network_status") or gold.get("network_label") or gold.get("main_agent_label") or "")
    pred_network = str(pred.get("network_status") or pred.get("network_label") or pred.get("main_agent_label") or "")

    if not gold_sensors and gold.get("local_status"):
        sensor_id = str(gold.get("sensor_id") or example.get("metadata", {}).get("sensor_id") or "sensor")
        gold_sensors = {sensor_id: str(gold.get("local_status"))}
        if pred.get("local_status"):
            pred_sensors = {sensor_id: str(pred.get("local_status"))}

    sensor_total = len(gold_sensors)
    sensor_correct = sum(1 for sensor_id, label in gold_sensors.items() if pred_sensors.get(sensor_id) == label)
    network_total = 1 if gold_network else 0
    network_correct = 1 if network_total and pred_network == gold_network else 0
    if sensor_total or network_total:
        example_exact = sensor_correct == sensor_total and network_correct == network_total
    else:
        gold_label = str(example.get("label") or "")
        pred_label = pred_network or next(iter(pred_sensors.values()), "")
        example_exact = bool(gold_label and pred_label == gold_label)
    return {
        "examples_total": 1,
        "example_exact": example_exact,
        "prediction_json_valid": bool(pred),
        "sensor_total": sensor_total,
        "sensor_correct": sensor_correct,
        "network_total": network_total,
        "network_correct": network_correct,
        "gold_sensor_labels": gold_sensors,
        "pred_sensor_labels": pred_sensors,
        "gold_network_label": gold_network or None,
        "pred_network_label": pred_network or None,
    }


def sensor_label_map(payload: dict[str, Any]) -> dict[str, str]:
    items = payload.get("sensor_reports") or payload.get("sensor_decisions") or payload.get("sensors")
    reports: list[dict[str, Any]] = []
    if isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, dict):
                reports.append({"sensor_id": str(value.get("sensor_id") or key), **value})
    elif isinstance(items, (list, tuple)):
        reports = [item for item in items if isinstance(item, dict)]
    out: dict[str, str] = {}
    for report in reports:
        sensor_id = str(report.get("sensor_id") or "")
        label = str(report.get("local_status") or report.get("small_agent_label") or report.get("label") or "")
        if sensor_id and label:
            out[sensor_id] = label
    return out


def infer_task(example: dict[str, Any], gold_payload: dict[str, Any]) -> str:
    task = example.get("task")
    if task:
        return str(task)
    if gold_payload.get("sensor_reports"):
        return "all_sensor_agent"
    if gold_payload.get("network_status"):
        return "main_building_agent"
    return "small_sensor_agent"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json_payload(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    main()
