import ctypes
import json
import os
import signal
import sys
from types import SimpleNamespace

import pytest

stdatalog = pytest.importorskip(
    "stdatalog_examples.vibroagent_two_vibrometer_logger",
    reason="stdatalog_examples is only present on the logger board")
NativeCallbackStream = stdatalog.NativeCallbackStream


def _payload(values):
    return (ctypes.c_uint8 * len(values))(*values)


def test_native_callback_writes_by_registered_stream_when_source_metadata_mismatches(tmp_path, capsys):
    stream = NativeCallbackStream(
        device_id=3,
        sensor_id="target_1",
        component_name="iis3dwb_acc",
        folder=tmp_path,
        hsd_link=object(),
    )
    stream.open()
    try:
        data = _payload([1, 2, 3, 4])

        result = stream._on_data_ready(99, b"unexpected_component", data, len(data))

        assert result == 0
        assert stream.bytes_written == 4
        assert stream.chunks_written == 1
        assert stream.source_mismatches == 1
        assert stream.last_callback_device_id == 99
        assert stream.last_callback_component_name == "unexpected_component"
        assert stream.dat_path.read_bytes() == b"\x01\x02\x03\x04"
        assert "Writing by registered callback identity" in capsys.readouterr().out
    finally:
        stream.close()


def test_native_callback_strict_source_identity_rejects_mismatch(tmp_path):
    stream = NativeCallbackStream(
        device_id=3,
        sensor_id="target_1",
        component_name="iis3dwb_acc",
        folder=tmp_path,
        hsd_link=object(),
        strict_source_identity=True,
    )
    stream.open()
    try:
        data = _payload([1, 2, 3, 4])

        result = stream._on_data_ready(99, b"unexpected_component", data, len(data))

        assert result == 0
        assert stream.bytes_written == 0
        assert stream.chunks_written == 0
        assert stream.source_mismatches == 1
        assert stream.dat_path.read_bytes() == b""
    finally:
        stream.close()


@pytest.mark.parametrize("stop_result", [
    True, (True, "stopped"), False, (False, "failed"), None,
    RuntimeError("USB error"), SystemExit("native response free failed"),
])
@pytest.mark.parametrize("signal_during_start", [False, True])
def test_shutdown_handles_repeated_signals_and_reports_native_failures(
    tmp_path, monkeypatch, capsys, stop_result, signal_during_start
):
    handlers = {}
    monkeypatch.setattr(stdatalog.signal, "signal", lambda number, handler: handlers.__setitem__(number, handler))
    events = []
    sessions = [
        stdatalog.BoardSession(
            device_id=device_id,
            sensor_id=f"target_{device_id}",
            serial=f"serial-{device_id}",
            folder=tmp_path / f"target_{device_id}",
            firmware={},
            active_sensors=["iis3dwb_acc"],
            logged_sensors=["iis3dwb_acc"],
            disabled_sensors=["other"],
            streams=[NativeCallbackStream(
                device_id=device_id,
                sensor_id=f"target_{device_id}",
                component_name="iis3dwb_acc",
                folder=tmp_path / f"target_{device_id}",
                hsd_link=object(),
            )],
            noop_components=["other"],
        )
        for device_id in range(2)
    ]

    def repeated_signals(*_args):
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGINT):
            handlers[number](number, None)

    def stop_log(_link, device_id):
        assert all(stream.stopping for session in sessions for stream in session.streams)
        events.append(("stop", device_id))
        repeated_signals()
        if device_id == 1:
            if isinstance(stop_result, BaseException):
                raise stop_result
            return stop_result
        return (True, "stopped")

    def set_callback(_link, device_id, component, callback):
        if callback is None:
            events.append(("unregister", device_id))
        return True

    def start_log(_link, device_id, **_kwargs):
        if signal_during_start and device_id == 1:
            repeated_signals()
        return (True, "started")

    link = SimpleNamespace(close=lambda: events.append(("close", None)) or True)
    monkeypatch.setattr(stdatalog, "HSDLink", SimpleNamespace(
        get_devices=lambda _link: [0, 1],
        set_data_ready_callback=set_callback,
        start_log=start_log,
        stop_log=stop_log,
        save_json_device_file=lambda *_args: True,
        save_json_acq_info_file=lambda *_args: True,
    ))
    monkeypatch.setattr(stdatalog, "_open_native_hsd_link", lambda _root: link)
    monkeypatch.setattr(stdatalog, "_visible_devices", lambda *_args: [])
    monkeypatch.setattr(stdatalog, "_resolve_roles", lambda *_args, **_kwargs: [0, 1])
    monkeypatch.setattr(stdatalog, "_prepare_sessions", lambda *_args: (sessions, [
        {"device_id": session.device_id, "sensor_id": session.sensor_id, "disabled_sensors": ["other"]}
        for session in sessions
    ]))
    monkeypatch.setattr(stdatalog, "_write_manifest", lambda *_args: None)
    monkeypatch.setattr(stdatalog, "_write_live_acquisition_info", lambda *_args: None)
    monkeypatch.setattr(stdatalog, "_restore_disabled_sensors", lambda _link, device_id, _sensors: events.append(("restore", device_id)))
    monkeypatch.setattr(stdatalog.time, "sleep", repeated_signals)
    monkeypatch.setattr(sys, "argv", ["logger", "--output-root", str(tmp_path)])

    failed = not (stop_result is True or stop_result == (True, "stopped"))
    if failed:
        with pytest.raises(SystemExit) as exc:
            stdatalog.main()
        assert exc.value.code == 1
    else:
        stdatalog.main()

    assert [event for event in events if event[0] == "stop"] == [("stop", 1), ("stop", 0)]
    assert ("close", None) in events
    assert all(stream.fd is None for session in sessions for stream in session.streams)
    assert (("unregister", 1) not in events) == failed
    assert (("restore", 1) not in events) == failed
    output = capsys.readouterr().out
    assert "Stopped target_0 ->" in output
    assert ("Stopped target_1 ->" not in output) == failed
    report = json.loads(next(line.removeprefix("Shutdown result: ") for line in output.splitlines()
                             if line.startswith("Shutdown result: ")))
    assert report["pid"] == os.getpid()
    assert report["started"] == ["target_0", "target_1"]
    assert report["stopped"] == (["target_0"] if failed else ["target_0", "target_1"])
    assert bool(report["errors"]) == failed
