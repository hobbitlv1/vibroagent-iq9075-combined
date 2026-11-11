"""The chat model chooses ``config_path`` for every MCP tool; it must stay inside allowed roots."""

from pathlib import Path

import pytest

from vibroagent_mcp import service
from vibroagent_mcp.service import _resolve_building_sensor_config, allowed_sensor_config_roots

PROJECT_ROOT = Path(service.__file__).resolve().parents[2]


def test_default_project_registry_resolves(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("VIBRO_BUILDING_SENSOR_CONFIG", raising=False)
    resolved = _resolve_building_sensor_config("config/sensors.live.yaml")
    assert resolved == (PROJECT_ROOT / "config" / "sensors.live.yaml").resolve()
    assert allowed_sensor_config_roots() == [PROJECT_ROOT]


@pytest.mark.parametrize(
    "escape",
    [
        "/etc/passwd",
        "/etc/definitely-not-a-real-file.yaml",
        "~/.ssh/config",
        "../../../../../../etc/hostname",
        "config/../../../../../etc/hostname",
    ],
)
def test_paths_outside_the_roots_are_refused_without_revealing_existence(monkeypatch, escape):
    monkeypatch.delenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("VIBRO_BUILDING_SENSOR_CONFIG", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _resolve_building_sensor_config(escape)
    message = str(excinfo.value)
    assert "not inside an allowed sensor config directory" in message
    # Only the caller's own string is echoed: no resolved path, no "exists"/"missing" wording.
    assert message.count("/etc") == escape.count("/etc")
    assert "does not exist" not in message and "exist" not in message.replace("existing", "")


def test_tmp_registry_needs_an_explicit_allowed_directory(tmp_path, monkeypatch):
    config = tmp_path / "sensors.yaml"
    config.write_text("sensors: []\n", encoding="utf-8")

    monkeypatch.delenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("VIBRO_BUILDING_SENSOR_CONFIG", raising=False)
    with pytest.raises(ValueError, match="not inside an allowed sensor config directory"):
        _resolve_building_sensor_config(str(config))

    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", str(tmp_path))
    assert _resolve_building_sensor_config(str(config)) == config.resolve()

    # The launcher-configured registry's own directory is trusted without the extra variable.
    monkeypatch.delenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", raising=False)
    monkeypatch.setenv("VIBRO_BUILDING_SENSOR_CONFIG", str(config))
    assert _resolve_building_sensor_config("config/sensors.live.yaml") == config.resolve()


def test_symlink_inside_the_project_cannot_point_outside(tmp_path, monkeypatch):
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: value\n", encoding="utf-8")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    link = allowed / "sensors.yaml"
    link.symlink_to(outside)
    monkeypatch.delenv("VIBRO_BUILDING_SENSOR_CONFIG", raising=False)
    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", str(allowed))
    with pytest.raises(ValueError, match="not inside an allowed sensor config directory"):
        _resolve_building_sensor_config(str(link))


def test_tool_services_refuse_an_outside_config_before_reading_it(monkeypatch):
    monkeypatch.delenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("VIBRO_BUILDING_SENSOR_CONFIG", raising=False)
    with pytest.raises(ValueError, match="not inside an allowed sensor config directory"):
        service.list_building_sensors_service(config_path="/etc/passwd")
    with pytest.raises(ValueError, match="not inside an allowed sensor config directory"):
        service.read_building_sensor_window_service(config_path="/etc/passwd", sensor_id="baseline")
