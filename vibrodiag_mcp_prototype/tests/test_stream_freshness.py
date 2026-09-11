import json
import os

import pytest

from vibroagent_mcp import webchat_server

@pytest.mark.parametrize("kind,current", [
    ("fresh", True), ("stale", False), ("empty", False),
    ("missing", False), ("directory", False), ("future", False),
    ("invalid_name", False),
])
def test_live_sensor_stream_status_uses_files_without_decoding(tmp_path, monkeypatch, kind, current):
    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VIBRO_CURRENT_FILE_MAX_AGE_S", "3")
    monkeypatch.setattr(webchat_server.time, "time", lambda: 1000.0)
    monkeypatch.setattr(webchat_server, "board_reader_process_statuses", lambda: {})
    monkeypatch.setattr(webchat_server, "prewarm_board_reader_processes",
                        lambda *_: pytest.fail("status polling must not prewarm or decode"))
    folder = tmp_path / "live_baseline"
    folder.mkdir()
    data = folder / "iis3dwb_acc.dat"
    if kind == "directory":
        data.mkdir()
    elif kind not in {"missing", "invalid_name"}:
        data.write_bytes(b"" if kind == "empty" else b"data")
    if data.exists():
        modified = 900.0 if kind == "stale" else 1001.0 if kind == "future" else 999.0
        os.utime(data, (modified, modified))
    config = tmp_path / "sensors.json"
    config.write_text(json.dumps({"baseline_sensor_id": "baseline", "sensors": [{
        "sensor_id": "baseline", "role": "baseline", "acquisition_folder": str(folder),
        "hsd_sensor_name": "../outside" if kind == "invalid_name" else "iis3dwb_acc",
    }]}))
    payload = webchat_server._live_sensors_payload({
        "config_path": [str(config)], "process_reader": ["0"], "prewarm": ["0"],
    })
    assert payload["ok"] is True, payload
    stream = payload["sensors"][0]["stream"]
    assert stream["data_is_current"] is current
    assert stream["data_currentness_max_age_s"] == 3.0
    if kind in {"missing", "invalid_name"}:
        assert stream["data_file_age_s"] is None
