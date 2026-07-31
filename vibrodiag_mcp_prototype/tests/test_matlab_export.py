import csv
import json
from pathlib import Path

import numpy as np

from vibroagent_mcp.llm_sensor_agent import AutonomousLlmSensorAgent
from vibroagent_mcp.matlab_export import VALIDATION_COLUMNS, export_validation_dataset
from vibroagent_mcp.service import export_building_validation_dataset_service


def _clear_model_env(monkeypatch):
    for name in (
        "SMALL_AGENT_BASE_URL",
        "SMALL_AGENT_MODEL",
        "SMALL_AGENT_API_KEY",
        "MAIN_AGENT_BASE_URL",
        "MAIN_AGENT_MODEL",
        "MAIN_AGENT_API_KEY",
        "QWEN_BASE_URL",
        "QWEN_MODEL",
        "QWEN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_config(tmp_path: Path) -> Path:
    t = np.arange(40, dtype=float) / 4.0
    baseline_signal = 0.0005 * np.sin(2 * np.pi * 0.5 * t)
    target_signal = 0.0018 * np.sin(2 * np.pi * 0.5 * t)
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
                    {"sensor_id": "baseline", "location": "calm", "role": "baseline", "replay_signal_path": str(baseline), "axis": "z", "sampling_rate_hz": 4},
                    {"sensor_id": "target", "location": "target", "role": "target", "replay_signal_path": str(target), "axis": "z", "sampling_rate_hz": 4},
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_matlab_export_writes_required_files_and_columns(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    config = _write_config(tmp_path)
    agent = AutonomousLlmSensorAgent(
        config_path=config,
        start_time_s=0,
        duration_s=10,
        require_current=False,
        mode="replay",
    )
    result = agent.run_window()

    export = export_validation_dataset(
        pipeline_result=result,
        windows=agent.last_windows,
        output_dir=tmp_path / "exports",
    )

    assert Path(export["validation_windows_csv"]).exists()
    assert Path(export["agent_reports_jsonl"]).exists()
    assert Path(export["network_assessments_jsonl"]).exists()
    with Path(export["validation_windows_csv"]).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == VALIDATION_COLUMNS
        rows = list(reader)
    assert rows
    assert {"baseline", "target"} <= {row["sensor_id"] for row in rows}


def test_export_service_returns_matlab_paths(tmp_path):
    config = _write_config(tmp_path)
    result = export_building_validation_dataset_service(
        config_path=str(config),
        baseline_sensor_id="baseline",
        sensor_ids=["target"],
        start_time_s=0,
        duration_s=10,
        axis="z",
        output_dir=str(tmp_path / "service_exports"),
        require_current=False,
    )

    assert result["schema_version"] == "0.5.0"
    assert result["status"] == "ok"
    assert Path(result["matlab_export"]["validation_windows_csv"]).exists()


def test_export_service_returns_structured_error_for_unwritable_output(tmp_path):
    config = _write_config(tmp_path)
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("already a file", encoding="utf-8")

    result = export_building_validation_dataset_service(
        config_path=str(config),
        baseline_sensor_id="baseline",
        sensor_ids=["target"],
        start_time_s=0,
        duration_s=10,
        axis="z",
        output_dir=str(output_file),
        require_current=False,
    )

    assert result["schema_version"] == "0.5.0"
    assert result["status"] == "error"
    assert result["error"] == "matlab_export_failed"
    assert result["window_id"]
    assert "network_assessment" in result
