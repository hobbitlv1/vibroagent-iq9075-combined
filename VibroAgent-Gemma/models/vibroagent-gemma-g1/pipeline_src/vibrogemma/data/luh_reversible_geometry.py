"""Publisher-grounded geometry for the LUH reversible-damage beam dataset."""

from __future__ import annotations

from typing import Final

LUH_STRUCTURE_ID: Final = "luh_reversible_damage_cantilever"
LUH_BEAM_LENGTH_M: Final = 1.205
LUH_ACCELEROMETER_COUNT: Final = 15
LUH_FISHPLATE_COUNT: Final = 9
LUH_SAMPLE_RATE_HZ: Final = 1200.0
LUH_EXCITATION_UPPER_HZ: Final = 250.0
LUH_MEASURED_AXIS: Final = "z"

# Figure 1 of Wolniak et al. (2023) places Acc1..Acc15 at 75 mm spacing,
# beginning 10 mm from the free tip.  Sensors alternate across the beam width.
LUH_ACCELEROMETER_POSITIONS_M: Final = {
    index: (
        0.010 + 0.075 * (index - 1),
        0.020 if index % 2 == 0 else -0.020,
        0.0,
    )
    for index in range(1, LUH_ACCELEROMETER_COUNT + 1)
}

# Geometry presented to the encoder is dimensionless and support-rooted, like
# the existing LUMO convention.  Acc1 is the farthest instrumented sensor from
# the clamp, so its axial coordinate defines one; target indices then progress
# monotonically from the flexible end toward the support.
LUH_POSITION_NORMALIZATION_M: Final = (
    LUH_BEAM_LENGTH_M - LUH_ACCELEROMETER_POSITIONS_M[1][0]
)
LUH_POSITION_BASIS: Final = (
    "publisher_longitudinal_coordinate_projected_to_canonical_structural_height;"
    "support_is_zero;farthest_instrumented_sensor_is_one;lateral_coordinate_provenance_only"
)

# Acc15 is near the clamp and electromagnetic shaker.  It is context only and
# is not claimed to be unaffected by damage, particularly the F9 condition.
LUH_REFERENCE_ACCELEROMETER: Final = 15
LUH_TARGET_ACCELEROMETERS: Final = {
    "target_1": 2,
    "target_2": 5,
    "target_3": 8,
    "target_4": 11,
    "target_5": 14,
}

# Each fishplate begins every 120 mm and is 130 mm long.  The contiguous
# five-zone projection is symmetric about F5 and is fixed for every record.
LUH_FISHPLATE_CENTERS_M: Final = {
    index: 0.065 + 0.120 * (index - 1)
    for index in range(1, LUH_FISHPLATE_COUNT + 1)
}
LUH_FISHPLATE_TO_TARGET: Final = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 4,
    7: 4,
    8: 5,
    9: 5,
}
LUH_DAMAGE_MECHANISMS: Final = ("discrete", "gaussian", "uniformly")
LUH_GEOMETRY_BASIS: Final = (
    "Wolniak et al. (2023), Fig. 1: 1.205 m beam, Acc1..Acc15 at 75 mm "
    "spacing, and F1..F9 reversible fishplate positions"
)


def luh_target_for_fishplate(fishplate: int) -> int:
    """Return the fixed five-zone target containing one publisher fishplate."""

    try:
        return LUH_FISHPLATE_TO_TARGET[int(fishplate)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"LUH fishplate must be 1..9, got {fishplate!r}") from exc


def luh_panel_accelerometers() -> tuple[int, int, int, int, int, int]:
    """Return the immutable context-plus-five-target accelerometer panel."""

    return (
        LUH_REFERENCE_ACCELEROMETER,
        *(LUH_TARGET_ACCELEROMETERS[f"target_{index}"] for index in range(1, 6)),
    )


def normalized_luh_sensor_position(accelerometer: int) -> tuple[float, float, float]:
    """Return a publisher sensor position in the encoder's dimensionless basis."""

    try:
        x_from_free_tip_m, _, _ = LUH_ACCELEROMETER_POSITIONS_M[
            int(accelerometer)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"LUH accelerometer must be 1..{LUH_ACCELEROMETER_COUNT}, "
            f"got {accelerometer!r}"
        ) from exc
    normalized_height = (
        LUH_BEAM_LENGTH_M - x_from_free_tip_m
    ) / LUH_POSITION_NORMALIZATION_M
    return (0.0, 0.0, normalized_height)


__all__ = [
    "LUH_ACCELEROMETER_COUNT",
    "LUH_ACCELEROMETER_POSITIONS_M",
    "LUH_BEAM_LENGTH_M",
    "LUH_DAMAGE_MECHANISMS",
    "LUH_EXCITATION_UPPER_HZ",
    "LUH_FISHPLATE_CENTERS_M",
    "LUH_FISHPLATE_COUNT",
    "LUH_FISHPLATE_TO_TARGET",
    "LUH_GEOMETRY_BASIS",
    "LUH_MEASURED_AXIS",
    "LUH_POSITION_BASIS",
    "LUH_POSITION_NORMALIZATION_M",
    "LUH_REFERENCE_ACCELEROMETER",
    "LUH_SAMPLE_RATE_HZ",
    "LUH_STRUCTURE_ID",
    "LUH_TARGET_ACCELEROMETERS",
    "luh_panel_accelerometers",
    "luh_target_for_fishplate",
    "normalized_luh_sensor_position",
]
