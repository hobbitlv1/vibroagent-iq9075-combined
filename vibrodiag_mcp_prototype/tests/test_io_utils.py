import numpy as np
import pytest

from vibroagent_mcp.io_utils import load_signal_file, save_signal_file


def test_load_signal_file_skips_time_column_for_multichannel_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_DATA_DIR", str(tmp_path))
    path = tmp_path / "window.csv"
    path.write_text(
        "time_s,x,y,z\n"
        + "\n".join(f"{idx / 10:.1f},{1 + idx},{2 + idx},{3 + idx}" for idx in range(40)),
        encoding="utf-8",
    )

    signal = load_signal_file(path)

    assert signal.shape == (40,)
    assert signal[0] == 1.0
    assert signal[-1] == 40.0


def test_load_signal_file_rejects_non_integer_channel(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_DATA_DIR", str(tmp_path))
    path = tmp_path / "window.csv"
    path.write_text("0,1\n1,2\n" * 20, encoding="utf-8")

    with pytest.raises(ValueError, match="channel"):
        load_signal_file(path, channel="x")  # type: ignore[arg-type]


def test_save_signal_file_validates_sampling_rate_and_finite_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBRO_ALLOWED_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        save_signal_file(tmp_path / "bad_rate.csv", np.ones(32), sampling_rate_hz=0)
    with pytest.raises(ValueError, match="finite"):
        save_signal_file(tmp_path / "bad_signal.csv", np.asarray([1.0, np.nan]), sampling_rate_hz=10)
