"""Regression checks for the September pipeline audit; no USB or model server."""
import importlib
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vibroagent_mcp import vibrogemma_live as live, vibrogemma_webchat as web
from vibroagent_mcp.vibration_quality import raw_window_quality

PROTO = Path(__file__).resolve().parents[1]
BUNDLE = PROTO.parent / "models/vibroagent-gemma-g1"


def test_raw_quality_distinguishes_quiet_noise_flatline_and_clipping(monkeypatch):
    quiet = np.random.default_rng(4).normal(0, 1e-5, (3, 1000))
    assert not raw_window_quality(quiet)["flatline"]
    assert raw_window_quality(np.ones((3, 1000)))["flatline"]
    quiet[0] = 8
    assert raw_window_quality(quiet, clipping_by_axis=dict.fromkeys("xyz", 8))["clipped_fraction"] == pytest.approx(1/3)
    monkeypatch.setenv("VIBROGEMMA_FLATLINE_MAX_PTP_G", "nan")
    with pytest.raises(ValueError):
        raw_window_quality(quiet)


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="packaged preprocessor unavailable")
def test_authoritative_quality_survives_real_preprocessor(monkeypatch):
    from scripts import live_vibrogemma_worker as worker
    t = np.arange(10000) / 1000
    payload = {slot: np.stack([0.01*np.sin((10+i)*t), 0.01*np.cos((12+i)*t), 1+0.01*np.sin((14+i)*t)]).astype(np.float32)
               for i, slot in enumerate(live.SLOTS)}
    payload.update({f"fs_hz__{slot}": 1000.0 for slot in live.SLOTS})
    payload["target_1"][0] = 8.0
    payload["target_2"][:] = 1.0
    payload["metadata__target_1"] = {"clipped_fraction": 1/3, "clipping_abs_g_by_axis": dict.fromkeys("xyz", 8)}
    episode = worker._episode(payload, BUNDLE, amplitude_calibrated=True)
    *_, cls = worker._load_contracts(BUNDLE)
    prepared = cls.from_config(worker._runtime_config(BUNDLE)).prepare(episode)
    worker._preserve_acquisition_quality(episode, prepared)
    assert prepared.episode.boards[1].quality.clipped_fraction >= 1/3
    assert not prepared.episode.boards[1].quality.is_valid
    assert not prepared.episode.boards[2].quality.is_valid
    assert prepared.quality_features[1, 2] == pytest.approx(1/3)
    assert prepared.quality_features[2, 0] == 0
    assert "acquisition_flatline" in prepared.episode.boards[2].quality.notes


def test_downsampling_rejects_alias_and_preserves_in_band_signal():
    t = np.arange(10000)/1000
    out = live._resample_board_window(np.tile(np.sin(2*np.pi*70*t), (3, 1)), source_rate_hz=1000, target_rate_hz=100)
    assert np.sqrt(np.mean(out[:, 100:-100]**2)) < 0.01
    retained = live._resample_board_window(np.tile(np.sin(2*np.pi*10*t), (3, 1)), source_rate_hz=1000, target_rate_hz=100)
    assert np.sqrt(np.mean(retained[:, 100:-100]**2)) == pytest.approx(2**-0.5, abs=0.01)


def snapshot(window_id="NEW", rms=0.1):
    targets = [{"sensor_id": f"target_{i}", "affected": False, "class": "normal", "severity": "none"} for i in range(1, 6)]
    return {"result": {"window_id": window_id, "baseline_sensor_id": "baseline",
                       "all_sensor_metrics": [{"sensor_id": "target_1", "acceleration_rms_g": rms}],
                       "model_metadata": {"monitor_llm_model_used": True,
                                          "vibrogemma_monitor": {"targets": targets, "global_class": "normal"}}}}


def test_chat_rejects_superseded_and_stale_browser_context(monkeypatch):
    monkeypatch.setattr(web, "_LATEST_MONITOR", snapshot())
    monkeypatch.setattr(web, "_LATEST_MONITOR_AT", time.monotonic())
    monkeypatch.setattr(web.webchat_server, "urlopen", lambda *_a, **_k: pytest.fail("stale context sent to model"))
    out = web._gemma_chat_payload({"message": "Explain", "context": {"available": True, "window_id": "OLD"}})
    assert out["context"]["available"] is False
    monkeypatch.setattr(web, "_LATEST_MONITOR_AT", 0)
    out = web._gemma_chat_payload({"message": "Explain", "context": {"available": True}})
    assert out["context"]["stale"]


def test_chat_snapshot_is_atomic_and_client_cannot_replace_labels(monkeypatch):
    monkeypatch.setattr(web, "_LATEST_MONITOR", snapshot())
    monkeypatch.setattr(web, "_LATEST_MONITOR_AT", time.monotonic())
    context, result, _ = web._resolve_chat_context({"available": True, "network_state": "critical", "targets": []})
    web._LATEST_MONITOR["result"]["window_id"] = "NEXT"
    assert context["window_id"] == result["window_id"] == "NEW"
    assert context["network_state"] == "normal" and len(context["targets"]) == 5


def test_saved_chat_uses_server_records_not_client_claims(monkeypatch):
    monkeypatch.setattr(web, "_load_replay_detail", lambda _: {"window_id": "SAVED", "pipeline_result": snapshot("SAVED")["result"]})
    monkeypatch.setattr(web, "_saved_evidence_text", lambda _: "saved evidence")
    context, result, evidence = web._resolve_chat_context({"source": "replay", "window_id": "SAVED", "targets": [{"affected": True}]})
    assert context["network_state"] == "normal"
    assert result["window_id"] == context["window_id"] == "SAVED"
    assert evidence == "saved evidence"


def test_spectrum_uses_overlay_reader_and_server_owned_chat(monkeypatch):
    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "target_3")
    sentinel = object()
    monkeypatch.setattr(web, "_lumo_display_reader", lambda _: sentinel)
    def spectrum(query, *, reader):
        assert reader is sentinel
        return {"ok": True, "machine_id": "target_3", "axis": "x", "psd_rms_g": 0.5}
    monkeypatch.setattr(web.webchat_server, "_psd_fft_payload", spectrum)
    result = web._spectrum_payload({})
    context, _, _ = web._resolve_chat_context({"source": "spectrum", "context_id": result["context_id"], "rms_g": 99})
    assert context["rms_g"] == 0.5


def test_monitor_publishes_current_failure_rejects_old_generation(monkeypatch):
    monkeypatch.setenv("VIBRO_LUMO_TRAINING_INJECTION", "target_3")
    monkeypatch.setattr(web, "_INJECTION_GENERATION", 7)
    failure = {"ok": False, "error": "live_data_required"}
    assert web._monitor_matches_current_injection(failure, 7)
    assert not web._monitor_matches_current_injection(failure, 6)
    monkeypatch.setattr(web, "_pin_current_thread", lambda *_: None)
    monkeypatch.setattr(web.time, "sleep", lambda _: (_ for _ in ()).throw(StopIteration()))
    monkeypatch.setattr(web, "_LATEST_MONITOR", None)
    monkeypatch.setattr(web, "_LATEST_MONITOR_AT", 0)
    with pytest.raises(StopIteration):
        web._monitor_loop(lambda _: failure)
    assert web._LATEST_MONITOR["error"] == "live_data_required"


def test_hybrid_does_not_claim_fixture_phase_synchronization(monkeypatch):
    values = {slot: np.ones((3, 100), dtype=np.float32) for slot in live.SLOTS}
    positions = {slot: np.zeros(3) for slot in live.SLOTS}
    masks = {slot: np.array([True, True, False]) for slot in live.SLOTS}
    monkeypatch.setattr(live, "_lumo_training_replay", lambda _: (values, dict.fromkeys(live.SLOTS, 10), {}, {"source_fixture": "test", "episode_provenance": {}}, positions, masks))
    live._lumo_overlay_templates.cache_clear()
    try:
        _, metadata = live._lumo_overlay_templates("target_3", 10, 100, 1)
        assert metadata["overlay_phase_synchronized"]
        assert not metadata["phase_synchronized"] and metadata["synchronization_verification_id"] is None
    finally:
        live._lumo_overlay_templates.cache_clear()


def test_profile_capture_passes_selected_bundle_to_all_calls(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROTO / "scripts"))
    capture = importlib.import_module("scripts.capture_live_healthy_profile")
    selected = tmp_path / "bundle"
    calls = []
    class Preprocessor:
        physics_extractor = SimpleNamespace(profile_feature_names=["f"])
        @classmethod
        def from_config(cls, _):
            return cls()
        def prepare(self, episode):
            return SimpleNamespace(episode=episode, quality_features=np.zeros((6,13)), profile_feature_values=np.ones((6,1)), profile_feature_mask=np.ones((6,1)))
    def contracts(bundle):
        calls.append(bundle)
        return SimpleNamespace(__module__="vibrogemma.contracts"), None, None, None, Preprocessor
    monkeypatch.setattr(capture, "_load_contracts", contracts)
    monkeypatch.setattr(capture, "_runtime_config", lambda _: {"model": {"multiresolution": {"physics": {}}}})
    monkeypatch.setattr(capture, "_devices", lambda _: dict.fromkeys(live.SLOTS, {}))
    monkeypatch.setattr(capture, "_latest_starts", lambda _: dict.fromkeys(live.SLOTS, 1000))
    monkeypatch.setattr(capture, "_read_block", lambda *_: (dict.fromkeys(live.SLOTS, np.ones((3, 550))), dict.fromkeys(live.SLOTS, 10)))
    def episode(payload, bundle, **kwargs):
        calls.append(bundle)
        return SimpleNamespace(boards=[SimpleNamespace(quality=SimpleNamespace(is_valid=True)) for _ in live.SLOTS])
    monkeypatch.setattr(capture, "_episode", episode)
    monkeypatch.setattr(capture, "_preserve_acquisition_quality", lambda *_: None)
    profile = SimpleNamespace(save=lambda _: None, lookup=lambda **_: True)
    class Builder:
        def __init__(self, _): pass
        def add(self, *_a, **_k): pass
        def finalize(self, **_): return profile
    monkeypatch.setitem(sys.modules, "vibrogemma.signal.profile", SimpleNamespace(HealthyProfile=SimpleNamespace(load=lambda _: profile), HealthyProfileBuilder=Builder))
    monkeypatch.setattr(sys, "argv", ["capture", "--bundle", str(selected), "--output", str(tmp_path / "healthy.json")])
    assert capture.main() == 0
    assert len(calls) == 9 and all(bundle == selected for bundle in calls)


@pytest.mark.skipif(not (PROTO / "scripts/prepare_vibrogemma.sh").is_file(), reason="Combined deployment uses setup_models.sh; preparer is workspace-only")
def test_setup_and_launch_share_pinned_q8_defaults(tmp_path):
    defaults = PROTO / "config/vibrogemma_deployment.sh"
    command = f'PROTO="{tmp_path}/vibrodiag_mcp_prototype"; source "{defaults}"; printf "%s\\n" "$VIBROGEMMA_BUNDLE" "$VIBROGEMMA_GGUF" "$VIBROGEMMA_GGUF_SHA256"'
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin"}).stdout.splitlines()
    assert result[0] == str(tmp_path / "models/vibrogemma-g1-multidomain-20260830")
    assert result[1].endswith("gemma-4-e2b-g1-Q8_0.gguf") and len(result[2]) == 64
    for script in [PROTO / "scripts/prepare_vibrogemma.sh", PROTO.parent / "vibroagent.sh"]:
        assert 'source "$PROTO/config/vibrogemma_deployment.sh"' in script.read_text()
        subprocess.run(["bash", "-n", str(script)], check=True)
    setup = (PROTO / "scripts/prepare_vibrogemma.sh").read_text()
    assert '"$F16" "$GGUF" Q8_0' in setup and "vibrogemma-final" not in setup


@pytest.mark.parametrize("member", ["bundle_manifest.json", "../escape.json"])
@pytest.mark.skipif(not (PROTO / "scripts/prepare_vibrogemma.sh").is_file(), reason="Combined deployment uses setup_models.sh; preparer is workspace-only")
def test_setup_bundle_extraction_is_confined_to_staging(tmp_path, member):
    import zipfile
    archive = tmp_path / "bundle.zip"
    output = tmp_path / "staging"
    output.mkdir()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member, "{}")
    setup = (PROTO / "scripts/prepare_vibrogemma.sh").read_text()
    code = setup.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    result = subprocess.run([sys.executable, "-c", code, str(archive), str(output)], capture_output=True, text=True)
    if member.startswith("../"):
        assert result.returncode != 0 and "Unsafe bundle" in result.stderr
        assert not (tmp_path / "escape.json").exists()
    else:
        assert result.returncode == 0 and (output / member).read_text() == "{}"
