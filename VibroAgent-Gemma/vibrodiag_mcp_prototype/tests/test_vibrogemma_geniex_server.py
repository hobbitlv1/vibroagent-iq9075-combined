import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

import vibroagent_mcp.vibrogemma_geniex_server as geniex_server

from vibroagent_mcp.vibrogemma_geniex_server import (
    _choice_strings,
    _apply_deterministic_quality_gate,
    _canonical_structured_rows,
    _classification_explanation,
    _factorized_user_content,
    _factorized_kv_cache_enabled,
    _factorized_summary,
    _embedding_audit,
    _marker,
    _shared_factorized_segments,
    _structured_choices,
    _structured_decision_block,
    _structured_segments,
    _structured_user_content,
    _user_content,
    classification_grammar,
    global_classification_grammar,
    mixed_segments,
    parse_global_record,
    parse_record,
    parse_target_record,
    target_classification_grammar,
    VibroGemmaRuntime,
)


RECORD = """CLASSIFICATION
T1|0|normal|none
T2|1|modal_frequency_shift|warning
T3|0|data_invalid|advisory
T4|1|unknown_anomaly|advisory
T5|0|normal|none
END"""


def _g1_config(target_query_order="deterministic_per_episode"):
    return {
        "model": {
            "marker_text": {
                "baseline_start": "",
                "baseline_end": "",
                "target_start_template": "",
                "target_end": "",
            }
        },
        "training": {
            "language_loss": {
                "prompt_prior_guard": True,
                "target_decision_semantics": "damage_source_zone",
            },
            "structured_set_training": {
                "enabled": True,
                "decision_separator": " ",
                "target_query_order": target_query_order,
                "choice_tokens": ["0", "1"],
            },
        },
    }


def _integrity_bundle(tmp_path: Path) -> tuple[Path, Path, str, dict[str, str]]:
    bundle = tmp_path / "bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    artifacts = {
        "vibrogemma_config.json": json.dumps(
            {
                "labels": {
                    "classes": ["normal", "unknown_anomaly"],
                    "severities": ["none", "advisory"],
                    "quality_class": "data_invalid",
                },
                "generation": {
                    "structured_set_decoding": False,
                    "factorized_decoding": False,
                },
                "model": {"system_prompt": "test system prompt"},
            }
        ),
        "tokenizer/tokenizer.json": "{}",
        "tokenizer/tokenizer_config.json": "{}",
        "token_layout.json": "{}",
        "alert.schema.json": "{}",
        "model.gguf": "tiny test model",
    }
    hashes = {}
    files = []
    for relative, content in artifacts.items():
        path = bundle / relative
        path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest})
    manifest_path = bundle / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps({"bundle_version": "test-bundle-v1", "files": files}), encoding="utf-8"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return bundle, bundle / "model.gguf", manifest_sha, hashes


def test_classification_contract_and_grammar():
    raw, targets = parse_record(RECORD)
    assert raw == RECORD
    assert [target["sensor_id"] for target in targets] == [f"target_{i}" for i in range(1, 6)]
    assert len(_choice_strings()) == 23
    assert "1|modal_frequency_shift|warning" in _choice_strings()
    assert "1|mounting_or_sensor_fault|critical" in _choice_strings()
    grammar = classification_grammar()
    assert '"CLASSIFICATION"' in grammar
    assert '"1|unknown_anomaly|advisory"' in grammar
    assert '"1|damping_change|critical"' in grammar


def test_live_prompt_matches_exported_bundle_training_context():
    prompt = _user_content(
        {"model": {"marker_text": {"baseline_start": "", "baseline_end": "", "target_start_template": "", "target_end": ""}}},
        "quality sentinel",
        "evidence sentinel",
    )
    assert "quality sentinel" in prompt
    assert "evidence sentinel" in prompt
    assert "After END, emit EXPLANATION" in prompt
    assert "equally plausible" not in prompt


def test_runtime_uses_exported_bundle_vocabulary_and_prompt(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    bundle = root / "models" / "vibroagent-gemma-g1"
    monkeypatch.delenv("VIBROGEMMA_GGUF_SHA256", raising=False)
    monkeypatch.setattr(
        geniex_server,
        "_verify_bundle_identity",
        lambda *_args: {
            "bundle_version": None,
            "manifest_sha256": None,
            "artifact_sha256": {},
            "model_sha256": None,
        },
    )
    runtime = VibroGemmaRuntime(
        model_path=bundle / "gemma-4-e2b-g1-Q8_0.gguf",
        bundle=bundle,
        device_map="llama_cpp:HTP0",
        n_ctx=4096,
        n_batch=512,
    )
    assert runtime.classes == tuple(runtime.config["labels"]["classes"])
    assert runtime.severities == tuple(runtime.config["labels"]["severities"])
    assert runtime.system_prompt == runtime.config["model"]["system_prompt"]


def test_runtime_verifies_bundle_manifest_and_reports_identity(tmp_path, monkeypatch):
    bundle, model_path, manifest_sha, hashes = _integrity_bundle(tmp_path)
    monkeypatch.setenv("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", manifest_sha)
    monkeypatch.delenv("VIBROGEMMA_GGUF_SHA256", raising=False)
    runtime = VibroGemmaRuntime(
        model_path=model_path, bundle=bundle, device_map="test", n_ctx=128, n_batch=16
    )
    health = runtime.health()
    assert health["bundle_version"] == "test-bundle-v1"
    assert health["manifest_sha256"] == manifest_sha
    assert health["bundle_manifest_sha256"] == manifest_sha
    assert health["checkpoint_sha256"] == hashes["model.gguf"]
    assert health["verified_artifact_sha256"] == hashes


def test_runtime_rejects_manifest_or_artifact_mismatch(tmp_path, monkeypatch):
    bundle, model_path, manifest_sha, _hashes = _integrity_bundle(tmp_path)
    monkeypatch.setenv("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="manifest SHA-256 mismatch"):
        VibroGemmaRuntime(
            model_path=model_path, bundle=bundle, device_map="test", n_ctx=128, n_batch=16
        )
    monkeypatch.setenv("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", manifest_sha)
    (bundle / "token_layout.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact (size|SHA-256) mismatch"):
        VibroGemmaRuntime(
            model_path=model_path, bundle=bundle, device_map="test", n_ctx=128, n_batch=16
        )


def test_runtime_keeps_manifestless_legacy_bundle_compatible(tmp_path, monkeypatch):
    bundle, model_path, _manifest_sha, _hashes = _integrity_bundle(tmp_path)
    (bundle / "bundle_manifest.json").unlink()
    monkeypatch.delenv("VIBROGEMMA_BUNDLE_MANIFEST_SHA256", raising=False)
    monkeypatch.delenv("VIBROGEMMA_GGUF_SHA256", raising=False)
    runtime = VibroGemmaRuntime(
        model_path=model_path, bundle=bundle, device_map="test", n_ctx=128, n_batch=16
    )
    health = runtime.health()
    assert health["bundle_version"] is None
    assert health["bundle_manifest_sha256"] is None
    assert health["verified_artifact_sha256"] == {}
    assert health["checkpoint_sha256"] is None


def test_record_rejects_semantically_invalid_tuple():
    with pytest.raises(ValueError, match="invalid target tuple"):
        parse_record(RECORD.replace("T1|0|normal|none", "T1|1|normal|advisory"))


def test_factorized_binary_contract_and_localization():
    classes = ("normal", "unknown_anomaly")
    severities = ("none", "advisory")
    raw_global, global_class, global_severity = parse_global_record(
        "GLOBAL_CLASSIFICATION\nG|unknown_anomaly|advisory\nEND",
        classes,
        severities,
    )
    assert raw_global.startswith("GLOBAL_CLASSIFICATION")
    targets = []
    for index in range(1, 6):
        affected = index == 3
        _, target = parse_target_record(
            f"TARGET_CLASSIFICATION\nT{index}|{int(affected)}|"
            f"{'unknown_anomaly|advisory' if affected else 'normal|none'}\nEND",
            index,
            classes,
            severities,
        )
        targets.append(target)
    state, localization, consistent, explanation = _factorized_summary(
        global_class, global_severity, targets
    )
    assert (state, localization, consistent) == ("advisory", "partial_target_set", True)
    assert "target 3" in explanation
    normal_targets = [
        {"sensor_id": f"target_{index}", "affected": False, "class": "normal", "severity": "none"}
        for index in range(1, 6)
    ]
    assert _factorized_summary("normal", "none", normal_targets)[0] == "normal"
    assert "GLOBAL_CLASSIFICATION" in global_classification_grammar(classes, severities)
    assert "TARGET_CLASSIFICATION" in target_classification_grammar(3, classes, severities)


def test_factorized_prompt_preserves_training_prior_and_localization_semantics():
    config = {
        "model": {
            "marker_text": {
                "baseline_start": "",
                "baseline_end": "",
                "target_start_template": "",
                "target_end": "",
            }
        }
    }
    prompt = _factorized_user_content(config, "quality", "evidence", target_index=3)
    assert "normal and unknown_anomaly as equally plausible" in prompt
    assert "excitation magnitude" in prompt
    assert "physical location belongs to fixed monitored zone T3" in prompt
    assert "global anomaly into an all-target anomaly" in prompt
    assert "global anomaly into a target anomaly" not in prompt


def test_data_invalid_is_applied_only_by_quality_gate():
    targets = [
        {"sensor_id": f"target_{index}", "affected": False, "class": "normal", "severity": "none"}
        for index in range(1, 6)
    ]
    quality = {"reference": {"valid": True}, **{f"target_{index}": {"valid": True} for index in range(1, 6)}}
    quality["target_3"]["valid"] = False
    _apply_deterministic_quality_gate(targets, quality, "data_invalid")
    assert [target["class"] for target in targets] == [
        "normal", "normal", "data_invalid", "normal", "normal"
    ]
    assert "data_invalid" not in target_classification_grammar(3, ("normal", "unknown_anomaly"), ("none", "advisory"))


def test_factorized_prompts_share_one_prefill_prefix():
    groups = [
        [("tokens", [1]), ("embeddings", [[2.0]]), ("tokens", [3, 4, 5])],
        [("tokens", [1]), ("embeddings", [[2.0]]), ("tokens", [3, 4, 6])],
    ]
    shared, suffixes = _shared_factorized_segments(groups)
    assert shared[-1] == ("tokens", [3, 4])
    assert suffixes == [[("tokens", [5])], [("tokens", [6])]]


def test_factorized_kv_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VIBROGEMMA_DISABLE_KV_CACHE", "1")
    assert _factorized_kv_cache_enabled() is False


def test_structured_decoder_scores_one_global_choice_and_all_target_masks():
    rows = [
        [0.0, 3.0],  # global anomaly
        [4.0, 0.0],  # target 1 normal
        [0.0, 4.0],  # target 2 affected
        [4.0, 0.0],  # target 3 normal
        [0.0, 4.0],  # target 4 affected
        [4.0, 0.0],  # target 5 normal
    ]
    global_choice, target_mask, scores, global_probability, target_probabilities = (
        _structured_choices(rows, [0.0] * 6)
    )
    assert global_choice == 1
    assert target_mask == 0b01010
    assert len(scores) == 32
    assert global_probability > 0.9
    assert all(probability > 0.9 for probability in target_probabilities)


def test_structured_decision_block_matches_checkpoint_token_layout():
    lines = _structured_decision_block().splitlines()
    assert lines == [
        "DIRECT_DECISIONS",
        "global: ",
        "target_1: ",
        "target_2: ",
        "target_3: ",
        "target_4: ",
        "target_5: ",
        "END DIRECT_DECISIONS",
    ]


def test_g1_structured_prompt_uses_packaged_instruction_not_legacy_block():
    prompt = _structured_user_content(_g1_config(), "quality", "evidence")
    assert "six binary decisions below are scored in parallel" in prompt
    assert "localized source zone belongs" in prompt
    assert "Treat 0 and 1 as equally plausible" in prompt
    assert "DIRECT_DECISIONS" not in prompt


def test_g1_structured_queries_use_placeholders_and_episode_order():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(ids=[ord(char) for char in text])

    formatted = "prefix" + "".join(_marker(index) for index in range(6)) + "assistant"
    segments, positions, choice_ids, slot_order = _structured_segments(
        CharacterTokenizer(),
        formatted,
        [[float(index)] for index in range(84)],
        _g1_config(),
        "episode-123",
    )
    token_tail = "".join(
        "".join(chr(value) for value in values)
        for kind, values in segments
        if kind == "tokens"
    )
    assert token_tail.endswith(
        "assistant\nSTRUCTURED_BINARY_DECISIONS"
        "\nGLOBAL= ?\nT4= ?\nT3= ?\nT1= ?\nT5= ?\nT2= ?"
        "\nEND_STRUCTURED_BINARY_DECISIONS"
    )
    assert positions == sorted(positions)
    assert choice_ids == [ord("0"), ord("1")]
    assert slot_order == [0, 4, 3, 1, 5, 2]
    rows = [[float(slot), -float(slot)] for slot in slot_order]
    assert _canonical_structured_rows(rows, slot_order) == [
        [float(slot), -float(slot)] for slot in range(6)
    ]


def test_g1_fixed_query_order_remains_supported():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(ids=[ord(char) for char in text])

    formatted = "".join(_marker(index) for index in range(6)) + "assistant"
    _segments, _positions, _choice_ids, slot_order = _structured_segments(
        CharacterTokenizer(),
        formatted,
        [[float(index)] for index in range(84)],
        _g1_config("fixed"),
        "",
    )
    assert slot_order == list(range(6))


def test_g1_response_matches_packaged_alert_schema_and_exposes_logit_audit(monkeypatch):
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = object()
    runtime.tokenizer = object()
    runtime.config = {
        "generation": {"confidence_calibration_path": None},
        "model": {"gemma_model_id": "google/gemma-4-E2B-it"},
    }
    runtime.classes = ("normal", "unknown_anomaly")
    runtime.severities = ("none", "advisory")
    runtime.quality_class = "data_invalid"
    runtime.structured_set_decoding = True
    runtime.factorized_decoding = False
    runtime.model_path = Path("gemma-4-e2b-g1-Q4_0_embq8.gguf")
    runtime.model_sha256 = "a" * 64
    runtime.bundle_version = "test-bundle-v1"
    runtime.bundle_manifest_sha256 = "b" * 64
    runtime.lock = threading.Lock()
    runtime.inference_started_mono = None
    runtime.inference_label = "idle"

    logits = [[2.0, -1.0], *([[-1.0, 2.0]] * 5)]
    audit = {
        "gemma_binary_choice_slot_order": ["GLOBAL", "T1", "T2", "T3", "T4", "T5"],
        "gemma_binary_choice_token_ids": [236771, 236770],
        "gemma_binary_choice_logits": logits,
        "gemma_binary_query_slot_ids": [0, 4, 3, 1, 5, 2],
        "gemma_binary_query_slot_order": ["GLOBAL", "T4", "T3", "T1", "T5", "T2"],
        "gemma_binary_query_positions": [369, 375, 381, 387, 393, 399],
    }

    def structured_record(_embeddings, _quality_text, _evidence_text, _episode_id):
        targets = [
            {
                "sensor_id": f"target_{index}",
                "affected": index == 5,
                "class": "unknown_anomaly" if index == 5 else "normal",
                "severity": "advisory" if index == 5 else "none",
            }
            for index in range(1, 6)
        ]
        return (
            "GLOBAL_CLASSIFICATION\nG|normal|none\nEND",
            "normal",
            "none",
            [
                f"TARGET_CLASSIFICATION\nT{index}|{int(index == 5)}|"
                f"{'unknown_anomaly|advisory' if index == 5 else 'normal|none'}\nEND"
                for index in range(1, 6)
            ],
            targets,
            [float(value) for value in range(32)],
            0.8,
            [0.91, 0.82, 0.73, 0.64, 0.55],
            audit,
        )

    runtime._structured_record = structured_record
    monkeypatch.setattr(geniex_server, "_decode_embeddings", lambda _payload: object())
    monkeypatch.setattr(
        geniex_server,
        "_embedding_audit",
        lambda _embeddings: {
            "embedding_sha256_by_slot": {},
            "embedding_rms_by_slot": {},
            "distinct_board_embeddings": True,
        },
    )
    quality = {
        "reference": {"valid": True},
        **{f"target_{index}": {"valid": True} for index in range(1, 6)},
    }
    response = runtime.classify(
        {
            "model": "vibrogemma",
            "vibration": {
                "quality_text": "all six boards valid",
                "evidence_text": "auditable evidence",
                "quality": quality,
                "window": {
                    "start_utc": "2026-08-31T00:00:00+00:00",
                    "duration_seconds": 10.0,
                    "effective_duration_seconds": 9.75,
                },
                "metadata": {"episode_id": "episode-123"},
                "evidence": {"boards": {}},
            },
        }
    )
    result = json.loads(response["choices"][0]["message"]["content"])

    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "models" / "vibroagent-gemma-g1" / "alert.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda error: list(error.path))
    assert not errors, "\n".join(error.message for error in errors)

    assert result["window"]["episode_id"] == "episode-123"
    assert result["window"]["duration_seconds"] == 10.0
    assert result["window"]["effective_duration_seconds"] == 9.75
    assert [target["class"] for target in result["targets"]] == [
        "normal",
        "normal",
        "normal",
        "normal",
        "unknown_anomaly",
    ]
    assert [target["choice_probability"] for target in result["targets"]] == [None] * 5
    assert [target["raw_choice_probability"] for target in result["targets"]] == [
        0.91,
        0.82,
        0.73,
        0.64,
        0.55,
    ]
    assert all(target["confidence_operational"] is False for target in result["targets"])
    assert all(
        set(target) == {
            "sensor_id",
            "affected",
            "class",
            "severity",
            "choice_probability",
            "raw_choice_probability",
            "confidence_operational",
        }
        for target in result["targets"]
    )
    model = result["model"]
    assert model["label_contract"] == {
        "classes": ["normal", "unknown_anomaly"],
        "quality_class": "data_invalid",
        "severities": ["none", "advisory"],
    }
    assert model["checkpoint_sha256"] == "a" * 64
    assert model["bundle_version"] == "test-bundle-v1"
    assert model["bundle_manifest_sha256"] == "b" * 64
    assert model["raw_global_choice_probability"] == 0.8
    assert model["global_choice_probability"] is None
    assert model["global_confidence_operational"] is False
    assert model["explanation_source"] == "deterministic_physics_evidence"
    assert model["explanation_grounded"] is True
    assert model["confidence_calibration"] == {
        "path": None,
        "target_operational": False,
        "target_validation_auroc": None,
        "global_operational": False,
        "global_validation_auroc": None,
    }
    assert model["gemma_binary_choice_logits"] == logits
    assert model["gemma_binary_choice_token_ids"] == [236771, 236770]
    assert model["gemma_binary_choice_slot_order"] == ["GLOBAL", "T1", "T2", "T3", "T4", "T5"]
    assert model["gemma_binary_query_slot_ids"] == [0, 4, 3, 1, 5, 2]
    assert model["gemma_binary_query_slot_order"] == ["GLOBAL", "T4", "T3", "T1", "T5", "T2"]


def test_embedding_audit_rejects_collapsed_board_tokens():
    import numpy as np

    embeddings = np.ones((84, 1536), dtype=np.float32)
    with pytest.raises(ValueError, match="byte-identical"):
        _embedding_audit(embeddings)

    for board in range(6):
        embeddings[board * 14 : (board + 1) * 14] *= board + 1
    audit = _embedding_audit(embeddings)
    assert audit["distinct_board_embeddings"] is True
    assert len(audit["embedding_sha256_by_slot"]) == 6
    assert set(audit["embedding_rms_by_slot"]) == {
        "baseline",
        "target_1",
        "target_2",
        "target_3",
        "target_4",
        "target_5",
    }


def test_loaded_runtime_probe_exercises_external_embeddings_and_state_logits():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=True):
            del add_special_tokens
            return SimpleNamespace(ids=[ord(char) for char in text])

    class FakeModel:
        def __init__(self):
            self.decoded_tokens = 0
            self.embedding_rows = 0
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def decode(self, *, input_ids=None, input_embd=None):
            if input_ids is not None:
                self.decoded_tokens += len(input_ids)
            if input_embd is not None:
                assert len(input_embd) == 1
                assert len(input_embd[0]) == 1536
                self.embedding_rows += 1

        @staticmethod
        def state_logits(token_ids):
            assert len(token_ids) == 2
            return [-0.25, 0.25]

    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = FakeModel()
    runtime.tokenizer = CharacterTokenizer()
    runtime.config = _g1_config("fixed")
    runtime.structured_set_decoding = True
    runtime.n_batch = 8

    probe = runtime._probe_loaded_runtime()
    assert probe["passed"] is True
    assert probe["external_embedding_decode"] is True
    assert probe["state_logits"] is True
    assert probe["generate_from_state"] is False
    assert probe["choice_token_ids"] == [ord("0"), ord("1")]
    assert runtime.model.decoded_tokens > 0
    assert runtime.model.embedding_rows == 1
    assert runtime.model.reset_count == 2


def test_loaded_runtime_probe_accepts_factorized_generation_api():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=True):
            del add_special_tokens
            return SimpleNamespace(ids=[ord(char) for char in text])

    class FakeModel:
        def reset(self):
            pass

        def decode(self, **_kwargs):
            pass

        def generate_from_state(self, **_kwargs):
            pass

    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = FakeModel()
    runtime.tokenizer = CharacterTokenizer()
    runtime.config = _g1_config("fixed")
    runtime.structured_set_decoding = False
    runtime.n_batch = 8

    probe = runtime._probe_loaded_runtime()
    assert probe["passed"] is True
    assert probe["state_logits"] is False
    assert probe["generate_from_state"] is True


def test_record_explanation_and_watchdog_are_bounded():
    _, targets = parse_record(RECORD)
    assert _classification_explanation(targets) == (
        "Signal quality was insufficient for target 3."
    )
    all_affected = [
        {"sensor_id": f"target_{index}", "affected": True, "class": "unknown_anomaly"}
        for index in range(1, 6)
    ]
    assert "does not isolate a single target" in _classification_explanation(all_affected)
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.inference_started_mono = time.monotonic() - 5
    runtime.inference_label = "vibration classification"
    age_s, label = runtime.stuck_inference(1)
    assert age_s >= 5
    assert label == "vibration classification"


def test_mixed_segments_replaces_each_marker_with_fourteen_rows():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(ids=[ord(char) for char in text])

    prompt = "prefix" + "".join(f" before {_marker(i)} after" for i in range(6)) + " suffix"
    embeddings = [[float(i)] for i in range(84)]
    segments = mixed_segments(CharacterTokenizer(), prompt, embeddings)
    embedding_segments = [values for kind, values in segments if kind == "embeddings"]
    assert [len(values) for values in embedding_segments] == [14] * 6
    assert embedding_segments[0][0] == [0.0]
    assert embedding_segments[-1][-1] == [83.0]


def test_mixed_segments_prepends_bos_exactly_once():
    class CharacterTokenizer:
        @staticmethod
        def token_to_id(token):
            return 2 if token == "<bos>" else None

        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            ids = []
            if text.startswith("<bos>"):
                ids.append(2)
                text = text[len("<bos>") :]
            ids.extend(ord(char) for char in text)
            return SimpleNamespace(ids=ids)

    embeddings = [[float(index)] for index in range(84)]
    prompt = "prefix" + "".join(_marker(index) for index in range(6)) + "suffix"
    for formatted in (prompt, "<bos>" + prompt):
        segments = mixed_segments(CharacterTokenizer(), formatted, embeddings)
        token_ids = [
            token_id
            for kind, values in segments
            if kind == "tokens"
            for token_id in values
        ]
        assert token_ids[0] == 2
        assert token_ids.count(2) == 1


def test_text_chat_does_not_require_vibration_payload():
    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert messages[-1] == {"role": "user", "content": "Status?"}
            assert kwargs["add_generation_prompt"] is True
            return "formatted prompt"

    class FakeModel:
        tokenizer = FakeTokenizer()

        @staticmethod
        def generate(prompt, **kwargs):
            assert prompt == "formatted prompt"
            assert kwargs["max_new_tokens"] == 48
            profile = SimpleNamespace(
                prompt_tokens=12,
                generated_tokens=3,
                decode_time=100_000,
                ttft=50_000,
                decode_speed=30.0,
            )
            return SimpleNamespace(text="All systems nominal.", profile=profile)

    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = FakeModel()
    runtime.tokenizer = object()
    runtime.lock = threading.Lock()
    response = runtime.classify(
        {"model": "vibrogemma", "messages": [{"role": "user", "content": "Status?"}], "max_tokens": 48}
    )
    assert response["choices"][0]["message"]["content"] == "All systems nominal."
    assert response["usage"]["total_tokens"] == 15


def test_structured_record_rejects_identical_slot_margins(monkeypatch):
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda *_args, **_kwargs: "formatted-prompt"
        )
    )
    runtime.tokenizer = object()
    runtime.config = {
        **_g1_config("fixed"),
        "generation": {"target_set_cardinality_bias": [0.0] * 6},
    }
    runtime.system_prompt = "system"
    runtime._state_logits_at_positions = lambda *_args: [[2.0, 1.0]] * 6
    monkeypatch.setattr(
        geniex_server,
        "_structured_segments",
        lambda *_args: ([('tokens', [1])], [0, 1, 2, 3, 4, 5], [10, 11], list(range(6))),
    )

    with pytest.raises(RuntimeError, match="identical across all six slots"):
        runtime._structured_record(
            [[0.1] * 1536 for _ in range(84)],
            "quality",
            "evidence",
            "episode-1",
        )


def test_structured_record_allows_distinct_all_normal_margins(monkeypatch):
    runtime = VibroGemmaRuntime.__new__(VibroGemmaRuntime)
    runtime.model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda *_args, **_kwargs: "formatted-prompt"
        )
    )
    runtime.tokenizer = object()
    runtime.config = {
        **_g1_config("fixed"),
        "generation": {"target_set_cardinality_bias": [0.0] * 6},
    }
    runtime.system_prompt = "system"
    runtime._state_logits_at_positions = lambda *_args: [
        [3.0 + index * 0.1, 0.0] for index in range(6)
    ]
    monkeypatch.setattr(
        geniex_server,
        "_structured_segments",
        lambda *_args: ([('tokens', [1])], [0, 1, 2, 3, 4, 5], [10, 11], list(range(6))),
    )

    record = runtime._structured_record(
        [[0.1] * 1536 for _ in range(84)],
        "quality",
        "evidence",
        "episode-1",
    )
    assert record[1] == "normal"
    assert all(target["class"] == "normal" for target in record[4])
    assert len(set(record[-1]["gemma_binary_choice_logit_margins"])) == 6
