"""The parity check compares answers and logits; the recorded reference must stay well formed."""

import json

import pytest

from scripts.check_geniex_parity import DEFAULT_REFERENCE_FILE, WINDOWS, compare_window, reference_entry


def _window(logits, sha="a" * 64):
    return {"embedding_sha256": sha, "choice_logits": logits}


def test_identical_logits_match_exactly():
    logits = [[27.4, 26.9], [16.0, 8.8]]

    outcome = compare_window(_window(logits), _window([row[:] for row in logits]))

    assert outcome["identical"] and outcome["answers_match"] and outcome["same_input"]
    assert outcome["max_logit_diff"] == 0


def test_shifted_logits_with_the_same_answers_still_match():
    outcome = compare_window(_window([[2.0, 1.0], [0.0, 3.0]]), _window([[2.5, 1.25], [0.5, 2.0]]))

    assert outcome["answers_match"] and not outcome["identical"]
    assert outcome["max_logit_diff"] == pytest.approx(1.0)
    assert outcome["observed_margins"] == pytest.approx([-1.25, 1.5])


def test_a_flipped_answer_is_reported_with_its_position():
    outcome = compare_window(_window([[2.0, 1.0], [0.0, 0.2]]), _window([[2.0, 1.0], [0.3, 0.2]]))

    assert not outcome["answers_match"]
    assert outcome["differing_positions"] == [1]


def test_a_different_encoder_input_is_flagged():
    outcome = compare_window(_window([[2.0, 1.0]], sha="a" * 64), _window([[2.0, 1.0]], sha="b" * 64))

    assert not outcome["same_input"]


def test_mismatched_answer_positions_are_rejected():
    with pytest.raises(ValueError):
        compare_window(_window([[2.0, 1.0], [0.0, 1.0]]), _window([[2.0, 1.0]]))


def test_recorded_reference_is_complete():
    recorded = json.loads(DEFAULT_REFERENCE_FILE.read_text())

    assert len(recorded["model_sha256"]) == 64
    assert recorded["references"]
    for name, entry in recorded["references"].items():
        assert {"geniex_version", "device_map", "n_threads", "windows"} <= set(entry), name
        assert set(entry["windows"]) == set(WINDOWS), name
        for window in entry["windows"].values():
            assert len(window["embedding_sha256"]) == 64
            assert len(window["choice_logits"]) == 6
            assert all(len(row) == 2 for row in window["choice_logits"])


def test_same_thread_count_gives_the_same_logits_on_both_geniex_releases():
    references = json.loads(DEFAULT_REFERENCE_FILE.read_text())["references"]

    for window in WINDOWS:
        old = references["geniex-0.4.0-cpu-8-threads"]["windows"][window]
        new = references["geniex-0.10.0-cpu-8-threads"]["windows"][window]
        assert compare_window(old, new)["identical"], window


def test_recording_takes_the_runtime_description_from_the_server():
    health = {"geniex_version": "0.10.0", "device_map": "llama_cpp:HTP0", "n_threads": 6, "status": "ok"}
    windows = {target: {**_window([[1.0, 2.0]] * 6), "seconds": 14.2, "model_sha256": "c" * 64} for target in WINDOWS}

    entry = reference_entry(health, windows, "first NPU run")

    assert (entry["geniex_version"], entry["device_map"], entry["n_threads"]) == ("0.10.0", "llama_cpp:HTP0", 6)
    assert entry["note"] == "first NPU run"
    assert set(entry["windows"]) == set(WINDOWS)
    assert set(entry["windows"]["target_3"]) == {"embedding_sha256", "choice_logits"}


def test_recording_refuses_a_server_that_does_not_report_its_thread_count():
    health = {"geniex_version": "0.10.0", "device_map": "llama_cpp:cpu"}
    windows = {target: _window([[1.0, 2.0]] * 6) for target in WINDOWS}

    with pytest.raises(ValueError, match="n_threads"):
        reference_entry(health, windows, "")


def test_recording_needs_both_windows():
    health = {"geniex_version": "0.10.0", "device_map": "llama_cpp:cpu", "n_threads": 4}

    with pytest.raises(ValueError, match="both bundled windows"):
        reference_entry(health, {"target_3": _window([[1.0, 2.0]] * 6)}, "")
