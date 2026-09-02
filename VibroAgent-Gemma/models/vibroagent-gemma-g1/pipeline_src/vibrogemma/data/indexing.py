from __future__ import annotations

from pathlib import Path

from ..utils import atomic_write_text
from .manifest import DatasetManifest


def write_index_template(manifest: DatasetManifest, config: dict, destination: str | Path) -> None:
    """Write a strict JSONL template. Every channel path must be supplied explicitly."""

    example = {
        "dataset_id": manifest.id,
        "source_file": "relative/path/to/source.h5",
        "structure_id": "publisher_structure_identifier",
        "session_id": "complete_acquisition_session_identifier",
        "experiment_id": "publisher_experiment_identifier",
        "sample_rate_hz": manifest.reader.get("sample_rate_hz"),
        "unit": "g|mg|m/s^2",
        "start_time_s": 0.0,
        "condition": "publisher_state_label",
        "healthy": True,
        "severity": "none|advisory|warning|critical",
        "class_name": "normal|modal_frequency_shift|damping_change|impact_transient|periodic_excitation|broadband_increase|mounting_or_sensor_fault|unknown_anomaly|data_invalid",
        "affected_targets": [],
        "phase_synchronized": False,
        "synchronization_verification_id": None,
        "boards": [
            {
                "sensor_id": "reference" if index == 0 else f"target_{index}",
                "signal_path": "/exact/numeric/node",
                "channel_indices": {"x": index},
                "axes_available": ["x"],
                "sensor_position": None,
                "orientation_matrix": None,
            }
            for index in range(6)
        ],
    }
    import json

    destination_path = Path(destination)
    atomic_write_text(destination_path, json.dumps(example, sort_keys=True) + "\n")
