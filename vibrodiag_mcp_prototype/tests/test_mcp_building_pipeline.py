import json

import numpy as np
import pytest

from vibroagent_mcp.schemas import NETWORK_LABELS, SENSOR_LABELS, ensure_building_domain_only
from vibroagent_mcp.service import (
    list_building_sensors_service,
    run_building_sensor_agent_service,
    run_building_vibration_agent_pipeline_service,
)


def _write_config(tmp_path):
    t = np.arange(64, dtype=float) / 8.0
    baseline_signal = 0.001 * np.sin(2 * np.pi * 1.0 * t)
    target_signal = 0.002 * np.sin(2 * np.pi * 1.0 * t)
    baseline = tmp_path / "baseline.csv"
    target = tmp_path / "target.csv"
    baseline.write_text("time_s,acceleration_g\n" + "\n".join(f"{a},{b}" for a, b in zip(t, baseline_signal)), encoding="utf-8")
    target.write_text("time_s,acceleration_g\n" + "\n".join(f"{a},{b}" for a, b in zip(t, target_signal)), encoding="utf-8")
    config = tmp_path / "sensors.json"
    config.write_text(
        json.dumps(
            {
                "baseline_sensor_id": "baseline",
                "sensors": [
                    {"sensor_id": "baseline", "location": "calm", "role": "baseline", "replay_signal_path": str(baseline), "axis": "z", "sampling_rate_hz": 8},
                    {"sensor_id": "target", "location": "target_room", "role": "target", "replay_signal_path": str(target), "axis": "z", "sampling_rate_hz": 8},
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_building_service_wrappers_return_required_shape(tmp_path, monkeypatch):
    monkeypatch.delenv("SMALL_AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("MAIN_AGENT_BASE_URL", raising=False)
    config = _write_config(tmp_path)

    listing = list_building_sensors_service(config_path=str(config))
    assert listing["schema_version"] == "0.5.0"
    assert listing["baseline_sensor_id"] == "baseline"

    result = run_building_vibration_agent_pipeline_service(
        config_path=str(config),
        baseline_sensor_id="baseline",
        sensor_ids=["target"],
        start_time_s=0,
        duration_s=8,
        axis="z",
        require_current=False,
    )

    assert result["schema_version"] == "0.5.0"
    assert result["window_id"]
    assert result["sensor_agent_reports"]
    assert result["sensor_agent_reports"][0]["small_agent_label"] in SENSOR_LABELS
    assert result["network_assessment"]["network_label"] in NETWORK_LABELS
    assert "model_metadata" in result
    ensure_building_domain_only(result)


def test_single_sensor_agent_wrapper_returns_report(tmp_path):
    config = _write_config(tmp_path)

    result = run_building_sensor_agent_service(
        config_path=str(config),
        sensor_id="target",
        baseline_sensor_id="baseline",
        start_time_s=0,
        duration_s=8,
        axis="z",
        require_current=False,
    )

    assert result["sensor_agent_report"]["sensor_id"] == "target"
    assert result["sensor_agent_report"]["small_agent_label"] in SENSOR_LABELS


def test_single_sensor_agent_requires_sensor_id(tmp_path):
    config = _write_config(tmp_path)

    # An empty sensor_id used to silently assess every registered sensor and
    # present the first report as "the" requested sensor.
    with pytest.raises(ValueError, match="requires sensor_id"):
        run_building_sensor_agent_service(
            config_path=str(config),
            sensor_id="",
            baseline_sensor_id="baseline",
            require_current=False,
        )


def test_domain_guard_catches_legacy_snake_case_labels():
    # Models trained on legacy machine-readable labels emit single tokens such
    # as "bearing_diagnosis"; the guard must treat separators as spaces.
    with pytest.raises(ValueError):
        ensure_building_domain_only({"prediction": {"label": "bearing_diagnosis"}})
    with pytest.raises(ValueError):
        ensure_building_domain_only(["possible inner-race signature"])
