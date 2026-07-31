from pathlib import Path

import pytest

from vibroagent_mcp.sensor_registry import SensorRegistry


def test_sensor_registry_loads_yaml_and_resolves_selected(tmp_path):
    baseline = tmp_path / "baseline.csv"
    target = tmp_path / "target.csv"
    baseline.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    target.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.yaml"
    config.write_text(
        f"""
baseline_sensor_id: baseline
sensors:
  - sensor_id: baseline
    location: calm_room
    role: baseline
    replay_signal_path: {baseline}
    axis: z
    sampling_rate_hz: 4
  - sensor_id: floor_2
    location: second_floor
    role: target
    replay_signal_path: {target}
    axis: z
    sampling_rate_hz: 4
""",
        encoding="utf-8",
    )

    registry = SensorRegistry(config)

    assert registry.baseline().sensor_id == "baseline"
    assert registry.targets()[0].sensor_id == "floor_2"
    assert registry.resolve_selected(["floor_2"])[0].location == "second_floor"


def test_sensor_registry_rejects_duplicate_sensor_ids(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.yaml"
    config.write_text(
        f"""
baseline_sensor_id: baseline
sensors:
  - sensor_id: baseline
    location: a
    role: baseline
    replay_signal_path: {data}
  - sensor_id: baseline
    location: b
    role: target
    replay_signal_path: {data}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sensor_id"):
        SensorRegistry(config)


def test_sensor_registry_rejects_missing_baseline(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.json"
    config.write_text(
        f"""{{
  "baseline_sensor_id": "missing",
  "sensors": [
    {{"sensor_id": "floor_2", "location": "second", "role": "target", "replay_signal_path": "{data}"}}
  ]
}}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Baseline sensor"):
        SensorRegistry(config)


def test_sensor_registry_rejects_invalid_axis(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.yaml"
    config.write_text(
        f"""
baseline_sensor_id: baseline
sensors:
  - sensor_id: baseline
    location: a
    role: baseline
    replay_signal_path: {data}
    axis: vertical
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="axis"):
        SensorRegistry(config)


def test_sensor_registry_accepts_numeric_string_sampling_rate(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.json"
    config.write_text(
        f"""{{
  "baseline_sensor_id": "baseline",
  "sensors": [
    {{"sensor_id": "baseline", "location": "a", "role": "baseline", "replay_signal_path": "{data}", "sampling_rate_hz": "4.0"}}
  ]
}}""",
        encoding="utf-8",
    )

    registry = SensorRegistry(config)

    assert registry.baseline().sampling_rate_hz == 4


def test_sensor_registry_rejects_invalid_sampling_rate(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n0\n0\n0\n" * 10, encoding="utf-8")
    config = tmp_path / "sensors.json"
    config.write_text(
        f"""{{
  "baseline_sensor_id": "baseline",
  "sensors": [
    {{"sensor_id": "baseline", "location": "a", "role": "baseline", "replay_signal_path": "{data}", "sampling_rate_hz": 0}}
  ]
}}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        SensorRegistry(config)


def test_sensor_registry_remaps_legacy_stdatalog_examples_absolute_path(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    config_dir = repo / "vibrodiag_mcp_prototype" / "config"
    live_dir = repo / "stdatalog_examples" / "live_baseline"
    config_dir.mkdir(parents=True)
    live_dir.mkdir(parents=True)
    config = config_dir / "sensors.yaml"
    config.write_text(
        """
baseline_sensor_id: baseline
sensors:
  - sensor_id: baseline
    location: reference
    role: baseline
    acquisition_folder: /home/ubuntu/Downloads/stdatalog-pysdk/stdatalog_examples/live_baseline
    axis: z
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    registry = SensorRegistry(config)

    assert registry.baseline().acquisition_folder == str(live_dir.resolve())
