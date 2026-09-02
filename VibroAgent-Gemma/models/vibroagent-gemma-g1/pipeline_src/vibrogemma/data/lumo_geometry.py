"""Canonical LUMO measurement and damage-zone geometry.

The publisher's measurement levels and reversible damage levels are ordered from
the top of the mast to its base.  VibroGemma has five target slots for six damage
levels, so DAM1 and DAM2 share the upper target zone.  Every other target uses the
nearest available acceleration measurement level.  ML4 is retained as the context
reference because it is the unused accelerometer furthest from a damage level.
"""

from __future__ import annotations

from typing import Any, Final

LUMO_SENSOR_ELEVATIONS_M: Final[dict[str, float]] = {
    "ML1": 8.95,
    "ML2": 8.00,
    "ML3": 7.00,
    "ML4": 5.95,
    "ML5": 5.00,
    "ML6": 4.00,
    "ML7": 2.95,
    "ML8": 2.00,
    "ML9": 1.00,
}

LUMO_DAMAGE_ELEVATIONS_M: Final[dict[str, float]] = {
    "DAM1": 8.30,
    "DAM2": 7.10,
    "DAM3": 4.90,
    "DAM4": 3.70,
    "DAM5": 1.90,
    "DAM6": 0.70,
}

LUMO_REFERENCE_MEASUREMENT_LEVEL: Final = "ML4"
LUMO_TARGET_MEASUREMENT_LEVELS: Final[dict[str, str]] = {
    "target_1": "ML2",
    "target_2": "ML5",
    "target_3": "ML6",
    "target_4": "ML8",
    "target_5": "ML9",
}
LUMO_TARGET_DAMAGE_LEVELS: Final[dict[str, list[str]]] = {
    "target_1": ["DAM1", "DAM2"],
    "target_2": ["DAM3"],
    "target_3": ["DAM4"],
    "target_4": ["DAM5"],
    "target_5": ["DAM6"],
}
LUMO_RELEASED_DAMAGE_LEVELS: Final[tuple[str, ...]] = ("DAM3", "DAM4", "DAM6")
LUMO_PANEL_LEVELS: Final[tuple[str, ...]] = (
    LUMO_REFERENCE_MEASUREMENT_LEVEL,
    *(LUMO_TARGET_MEASUREMENT_LEVELS[f"target_{index}"] for index in range(1, 6)),
)
LUMO_DAMAGE_TO_TARGET: Final[dict[str, int]] = {
    damage: int(target.rsplit("_", 1)[-1])
    for target, damage_levels in LUMO_TARGET_DAMAGE_LEVELS.items()
    for damage in damage_levels
}

# Geometry inputs are dimensionless inside the encoder.  Preserve the official
# vertical ordering while making the normalization explicit and reproducible.
LUMO_POSITION_NORMALIZATION_M: Final = max(LUMO_SENSOR_ELEVATIONS_M.values())
LUMO_POSITION_BASIS: Final = (
    "publisher_vertical_elevation_normalized_by_highest_accelerometer;"
    "horizontal_coordinates_unavailable"
)


def normalized_lumo_sensor_position(measurement_level: str) -> tuple[float, float, float]:
    """Return an XYZ-shaped position containing the sourced normalized elevation."""

    level = str(measurement_level).strip().upper()
    try:
        elevation = LUMO_SENSOR_ELEVATIONS_M[level]
    except KeyError as exc:
        raise ValueError(f"unknown LUMO measurement level: {measurement_level!r}") from exc
    return (0.0, 0.0, elevation / LUMO_POSITION_NORMALIZATION_M)


def lumo_zone_vertical_distance_m(target: str, damage_level: str) -> float:
    """Return the physical vertical distance between one target sensor and damage."""

    target_name = str(target).strip().lower()
    damage_name = str(damage_level).strip().upper()
    try:
        sensor_level = LUMO_TARGET_MEASUREMENT_LEVELS[target_name]
        sensor_elevation = LUMO_SENSOR_ELEVATIONS_M[sensor_level]
        damage_elevation = LUMO_DAMAGE_ELEVATIONS_M[damage_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown LUMO target/damage pair: {target!r}, {damage_level!r}"
        ) from exc
    return round(abs(sensor_elevation - damage_elevation), 9)


LUMO_ZONE_CONTRACT: Final[dict[str, Any]] = {
    "schema": "vibrogemma-lumo-zone-contract-v0.19.6.9-geometry-aligned",
    "reference_measurement_level": LUMO_REFERENCE_MEASUREMENT_LEVEL,
    "target_zones": {
        target: {
            "measurement_level": LUMO_TARGET_MEASUREMENT_LEVELS[target],
            "damage_levels": list(damage_levels),
            "maximum_vertical_distance_m": max(
                lumo_zone_vertical_distance_m(target, damage)
                for damage in damage_levels
            ),
        }
        for target, damage_levels in LUMO_TARGET_DAMAGE_LEVELS.items()
    },
    "sensor_elevations_m": dict(LUMO_SENSOR_ELEVATIONS_M),
    "damage_elevations_m": dict(LUMO_DAMAGE_ELEVATIONS_M),
    "position_normalization_m": LUMO_POSITION_NORMALIZATION_M,
    "position_basis": LUMO_POSITION_BASIS,
    "released_damage_levels_expected": list(LUMO_RELEASED_DAMAGE_LEVELS),
    "mapping_status": "project_defined_geometry_aligned_zone_representatives",
    "positive_label_semantics": "known structural damage location lies in target zone",
    "exclusive_waveform_effect_claimed": False,
}


__all__ = [
    "LUMO_DAMAGE_ELEVATIONS_M",
    "LUMO_DAMAGE_TO_TARGET",
    "LUMO_PANEL_LEVELS",
    "LUMO_POSITION_BASIS",
    "LUMO_POSITION_NORMALIZATION_M",
    "LUMO_REFERENCE_MEASUREMENT_LEVEL",
    "LUMO_RELEASED_DAMAGE_LEVELS",
    "LUMO_SENSOR_ELEVATIONS_M",
    "LUMO_TARGET_DAMAGE_LEVELS",
    "LUMO_TARGET_MEASUREMENT_LEVELS",
    "LUMO_ZONE_CONTRACT",
    "lumo_zone_vertical_distance_m",
    "normalized_lumo_sensor_position",
]
