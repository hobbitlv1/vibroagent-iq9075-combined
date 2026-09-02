from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from ..exceptions import DataContractError


@dataclass(slots=True)
class BufferSlice:
    values: np.ndarray
    timestamps_s: np.ndarray
    sample_mask: np.ndarray
    packet_loss_count: int


class TimestampedRingBuffer:
    def __init__(self, *, channels: int, sample_rate_hz: float, capacity_seconds: float) -> None:
        if channels <= 0 or sample_rate_hz <= 0 or capacity_seconds <= 0:
            raise ValueError("channels, sample_rate_hz and capacity_seconds must be positive")
        self.channels = channels
        self.sample_rate_hz = float(sample_rate_hz)
        self.capacity = int(np.ceil(sample_rate_hz * capacity_seconds))
        self.values = np.zeros((channels, self.capacity), dtype=np.float32)
        self.timestamps = np.full(self.capacity, np.nan, dtype=np.float64)
        self.valid = np.zeros(self.capacity, dtype=bool)
        self.write_index = 0
        self.size = 0
        self.packet_loss_count = 0
        self.lock = RLock()

    def append(
        self,
        values: np.ndarray,
        *,
        start_time_s: float,
        timestamps_s: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
        packet_loss_count: int = 0,
    ) -> None:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != self.channels:
            raise DataContractError(f"Expected [{self.channels},samples], got {matrix.shape}")
        count = matrix.shape[1]
        if count == 0:
            return
        if timestamps_s is None:
            timestamps = start_time_s + np.arange(count, dtype=np.float64) / self.sample_rate_hz
        else:
            timestamps = np.asarray(timestamps_s, dtype=np.float64)
            if timestamps.shape != (count,):
                raise DataContractError("timestamps_s shape mismatch")
        valid = np.ones(count, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.shape != (count,):
            raise DataContractError("valid_mask shape mismatch")
        if count > self.capacity:
            matrix = matrix[:, -self.capacity :]
            timestamps = timestamps[-self.capacity :]
            valid = valid[-self.capacity :]
            count = self.capacity
        with self.lock:
            first = min(count, self.capacity - self.write_index)
            second = count - first
            self.values[:, self.write_index : self.write_index + first] = matrix[:, :first]
            self.timestamps[self.write_index : self.write_index + first] = timestamps[:first]
            self.valid[self.write_index : self.write_index + first] = valid[:first]
            if second:
                self.values[:, :second] = matrix[:, first:]
                self.timestamps[:second] = timestamps[first:]
                self.valid[:second] = valid[first:]
            self.write_index = (self.write_index + count) % self.capacity
            self.size = min(self.capacity, self.size + count)
            self.packet_loss_count += int(packet_loss_count)

    def _ordered(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            return (
                np.empty((self.channels, 0), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=bool),
            )
        start = (self.write_index - self.size) % self.capacity
        if start + self.size <= self.capacity:
            return (
                self.values[:, start : start + self.size].copy(),
                self.timestamps[start : start + self.size].copy(),
                self.valid[start : start + self.size].copy(),
            )
        first = self.capacity - start
        return (
            np.concatenate((self.values[:, start:], self.values[:, : self.size - first]), axis=1),
            np.concatenate((self.timestamps[start:], self.timestamps[: self.size - first])),
            np.concatenate((self.valid[start:], self.valid[: self.size - first])),
        )

    def latest_timestamp(self) -> float | None:
        with self.lock:
            if self.size == 0:
                return None
            index = (self.write_index - 1) % self.capacity
            value = self.timestamps[index]
            return float(value) if np.isfinite(value) else None

    def earliest_timestamp(self) -> float | None:
        with self.lock:
            if self.size == 0:
                return None
            index = (self.write_index - self.size) % self.capacity
            value = self.timestamps[index]
            return float(value) if np.isfinite(value) else None

    def extract(self, start_time_s: float, end_time_s: float, *, expected_samples: int) -> BufferSlice:
        if end_time_s <= start_time_s:
            raise ValueError("end_time_s must be after start_time_s")
        with self.lock:
            values, timestamps, valid = self._ordered()
            packet_loss_count = self.packet_loss_count
        selected = (timestamps >= start_time_s) & (timestamps < end_time_s)
        selected_values = values[:, selected]
        selected_timestamps = timestamps[selected]
        selected_valid = valid[selected]
        output = np.zeros((self.channels, expected_samples), dtype=np.float32)
        output_timestamps = start_time_s + np.arange(expected_samples, dtype=np.float64) / self.sample_rate_hz
        output_valid = np.zeros(expected_samples, dtype=bool)
        if selected_timestamps.size:
            indices = np.rint((selected_timestamps - start_time_s) * self.sample_rate_hz).astype(np.int64)
            in_range = (indices >= 0) & (indices < expected_samples)
            indices = indices[in_range]
            output[:, indices] = selected_values[:, in_range]
            output_valid[indices] = selected_valid[in_range]
        return BufferSlice(output, output_timestamps, output_valid, packet_loss_count)
