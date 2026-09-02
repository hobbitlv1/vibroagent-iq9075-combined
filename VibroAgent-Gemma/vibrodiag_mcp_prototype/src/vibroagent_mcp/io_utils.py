"""Safe vibration-signal input utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


def allowed_data_dir() -> Path:
    """Return the directory from which the MCP tool is allowed to read files."""
    return Path(os.environ.get("VIBRO_ALLOWED_DATA_DIR", os.getcwd())).resolve()


def validate_signal_path(signal_path: str | os.PathLike[str]) -> Path:
    """Validate a file path before loading it from an LLM-callable MCP tool.

    This prevents the model/tool caller from reading arbitrary files.
    """
    path = Path(signal_path).expanduser().resolve()
    root = allowed_data_dir()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"signal_path must be inside VIBRO_ALLOWED_DATA_DIR={root}. Got: {path}"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Signal file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Signal path is not a file: {path}")
    if path.suffix.lower() not in {".csv", ".txt", ".npy"}:
        raise ValueError("Supported signal file formats are .csv, .txt, and .npy")
    return path


def load_signal_file(
    signal_path: str | os.PathLike[str],
    channel: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Load a 1D vibration signal from CSV/TXT/NPY."""
    path = validate_signal_path(signal_path)

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            # Accept simple one-line CSV/TXT headers such as "time_s,amplitude".
            arr = np.loadtxt(path, delimiter=delimiter, skiprows=1)

    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 0:
        raise ValueError("Loaded signal has no samples")
    if arr.ndim == 1:
        signal = arr
    elif arr.ndim == 2:
        if channel is None:
            # Common case: first column may be time, followed by one or more
            # acceleration channels.
            if arr.shape[1] >= 2 and np.all(np.diff(arr[:, 0]) >= 0):
                channel = 1
            else:
                channel = 0
        else:
            try:
                channel = int(channel)
            except (TypeError, ValueError):
                raise ValueError("channel must be an integer column index") from None
        if channel < 0 or channel >= arr.shape[1]:
            raise ValueError(f"channel must be in [0, {arr.shape[1] - 1}]")
        signal = arr[:, channel]
    else:
        raise ValueError("Loaded signal must be a 1D or 2D array")

    signal = signal[np.isfinite(signal)]
    if max_samples is not None and max_samples > 0:
        signal = signal[:max_samples]
    if signal.size < 32:
        raise ValueError("Signal must contain at least 32 finite samples")
    return signal.astype(np.float64)


def validate_output_signal_path(output_path: str | os.PathLike[str], overwrite: bool = False) -> Path:
    """Validate a path before writing a synthetic signal from an MCP-callable tool.

    The file must stay inside VIBRO_ALLOWED_DATA_DIR and must use .csv, .txt, or .npy.
    """
    path = Path(output_path).expanduser().resolve()
    root = allowed_data_dir()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output_path must be inside VIBRO_ALLOWED_DATA_DIR={root}. Got: {path}"
        ) from exc

    if path.suffix.lower() not in {".csv", ".txt", ".npy"}:
        raise ValueError("Supported output file formats are .csv, .txt, and .npy")
    if not path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")
    return path


def save_signal_file(
    output_path: str | os.PathLike[str],
    signal: np.ndarray,
    sampling_rate_hz: int,
    include_time_column: bool = True,
    overwrite: bool = False,
) -> Path:
    """Save a synthetic signal to CSV/TXT/NPY inside the allowed data directory."""
    path = validate_output_signal_path(output_path, overwrite=overwrite)
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if x.size == 0:
        raise ValueError("signal must contain at least one sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain only finite samples")

    if path.suffix.lower() == ".npy":
        np.save(path, x)
    else:
        if include_time_column:
            t = np.arange(x.size, dtype=np.float64) / float(sampling_rate_hz)
            data = np.column_stack([t, x])
            header = "time_s,amplitude"
        else:
            data = x
            header = "amplitude"
        delimiter = "," if path.suffix.lower() == ".csv" else " "
        np.savetxt(path, data, delimiter=delimiter, header=header, comments="# ")
    return path
