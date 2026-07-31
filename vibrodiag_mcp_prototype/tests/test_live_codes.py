"""Golden tests for the codes_v3 live prompt/parse layer.

The authority for the prompt is the immutable training set in
data/codes_sft_v3 — every test here compares against those bytes, never
against a builder script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vibroagent_mcp import live_codes

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "codes_sft_v3" / "train.jsonl"
VAL = REPO / "data" / "codes_sft_v3" / "val.jsonl"

_TGT_RE = re.compile(
    r'^Target sensor "(target_\d+)" level_rel_db=([+-]?\d+\.\d) '
    r'codes: (\S+)$', re.M)
_REF_RE = re.compile(r'^Reference sensor "baseline" codes: (\S+)$', re.M)
_ALIGN_RE = re.compile(r"^Reference alignment: (.+)$", re.M)


def _examples(limit_per_shape: int = 40):
    """Yield (example, n_targets) covering both shapes and alignments."""
    buckets: dict[tuple[int, bool], int] = {}
    for path in (TRAIN, VAL):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                example = json.loads(line)
                user = example["messages"][1]["content"]
                n_targets = len(_TGT_RE.findall(user))
                aligned = "time-aligned" in _ALIGN_RE.search(user).group(1)
                key = (n_targets, aligned)
                if buckets.get(key, 0) >= limit_per_shape:
                    continue
                buckets[key] = buckets.get(key, 0) + 1
                yield example, n_targets


def test_template_asset_matches_dataset():
    """The committed asset must re-derive byte-identically from the data."""
    asset = json.loads(
        (REPO / "src" / "vibroagent_mcp" / "assets" /
         "codes_v3_prompt_template.json").read_text(encoding="utf-8"))
    systems, headers, tails, aligns = set(), set(), {}, {}
    for example, n_targets in _examples(limit_per_shape=200):
        systems.add(example["messages"][0]["content"])
        user = example["messages"][1]["content"]
        match_align = _ALIGN_RE.search(user)
        key = "aligned" if "time-aligned" in match_align.group(1) \
            else "fallback"
        aligns[key] = match_align.group(1)
        headers.add(user[:match_align.start()])
        last = None
        for match in _TGT_RE.finditer(user):
            last = match
        tails["multi" if n_targets == 5 else "single"] = user[last.end():]
    assert systems == {asset["system"]}
    assert headers == {asset["header"]}
    assert tails == asset["tails"]
    assert aligns == asset["alignment_variants"]


def test_prompt_rebuild_is_byte_exact():
    """Rebuilding stored prompts from their parsed pieces must round-trip."""
    checked = 0
    for example, _n in _examples():
        user = example["messages"][1]["content"]
        reference = _REF_RE.search(user).group(1)
        targets = [(slot, float(level), codes)
                   for slot, level, codes in _TGT_RE.findall(user)]
        aligned = "time-aligned" in _ALIGN_RE.search(user).group(1)
        rebuilt = live_codes.build_codes_user_prompt(
            reference, targets, aligned=aligned)
        assert rebuilt == user
        assert example["messages"][0]["content"] == \
            live_codes.system_message()
        checked += 1
    assert checked >= 100


def test_assistant_labels_within_vocabulary():
    for example, _n in _examples(limit_per_shape=200):
        verdict = json.loads(example["messages"][2]["content"])
        for report in verdict["sensor_reports"]:
            assert report["local_status"] in live_codes.SENSOR_LABELS
        assert verdict["network_status"] in live_codes.NETWORK_LABELS


def test_build_rejects_wrong_shapes():
    codes = "x" * live_codes.CODES_PER_WINDOW
    with pytest.raises(live_codes.CodesPromptError):
        live_codes.build_codes_user_prompt(
            codes, [("target_2", 0.0, codes)], aligned=True)
    with pytest.raises(live_codes.CodesPromptError):
        live_codes.build_codes_user_prompt(
            codes, [("target_1", 0.0, codes[:-1])], aligned=True)
    with pytest.raises(live_codes.CodesPromptError):
        live_codes.build_codes_user_prompt(
            codes[:-1], [("target_1", 0.0, codes)], aligned=True)
    slots = [(slot, 0.0, codes) for slot in live_codes.LIVE_SLOTS[:3]]
    with pytest.raises(live_codes.CodesPromptError):
        live_codes.build_codes_user_prompt(codes, slots, aligned=True)


def test_parse_accepts_training_answers():
    for example, _n in _examples(limit_per_shape=50):
        user = example["messages"][1]["content"]
        targets = [slot for slot, _, _ in _TGT_RE.findall(user)]
        parsed = live_codes.parse_codes_reply(
            example["messages"][2]["content"], targets)
        assert sorted(parsed["sensor_reports"]) == sorted(targets)
        assert parsed["network_status"] in live_codes.NETWORK_LABELS


def test_parse_rejects_drift():
    targets = list(live_codes.LIVE_SLOTS)
    good = {
        "sensor_reports": [
            {"sensor_id": slot,
             "local_status": "normal_relative_to_baseline"}
            for slot in targets],
        "network_status": "normal_relative_to_reference_sensor",
        "affected_sensor_ids": [],
    }
    assert live_codes.parse_codes_reply(json.dumps(good), targets)

    for mutate in (
        lambda d: d.update(network_status="calm"),
        lambda d: d["sensor_reports"].pop(),
        lambda d: d["sensor_reports"].append(
            dict(d["sensor_reports"][0])),
        lambda d: d["sensor_reports"][0].update(local_status="broken"),
        lambda d: d["sensor_reports"][0].update(sensor_id="target_9"),
        lambda d: d.update(affected_sensor_ids=["target_9"]),
        lambda d: d.update(extra_key=1),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ValueError):
            live_codes.parse_codes_reply(json.dumps(bad), targets)
    with pytest.raises(ValueError):
        live_codes.parse_codes_reply("no json here", targets)


def test_response_format_schema_shape():
    fmt = live_codes.codes_response_format(live_codes.LIVE_SLOTS)
    schema = fmt["json_schema"]["schema"]
    assert schema["properties"]["sensor_reports"]["minItems"] == 5
    assert set(schema["required"]) == {
        "sensor_reports", "network_status", "affected_sensor_ids"}
