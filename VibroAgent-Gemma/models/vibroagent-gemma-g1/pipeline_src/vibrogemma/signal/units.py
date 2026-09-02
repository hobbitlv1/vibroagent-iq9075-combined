from __future__ import annotations

import numpy as np

from ..exceptions import DataContractError

STANDARD_GRAVITY = 9.80665
UNCALIBRATED_UNITS = {
    "release_native",
    "release-native",
    "native",
    "uncalibrated",
    "arbitrary",
    "arbitrary_units",
    "a.u.",
    "au",
}


def canonical_acceleration_unit(unit: str) -> tuple[str, bool]:
    """Return a canonical unit name and whether absolute amplitude is calibrated.

    ``release_native`` is intentionally supported for access-controlled public
    datasets whose waveform scale is documented but whose physical acceleration
    unit is not.  It is *not* converted to g.  Downstream code must mask every
    absolute-amplitude physics feature while retaining scale-invariant temporal,
    spectral, modal, transmissibility, and coherence evidence.
    """

    unit_normalized = unit.strip().lower().replace("²", "2")
    if unit_normalized in {"g", "gravity", "gravities"}:
        return "g", True
    if unit_normalized == "mg":
        return "mg", True
    if unit_normalized in {"m/s2", "m/s^2", "mps2"}:
        return "m/s^2", True
    if unit_normalized in {"mm/s2", "mm/s^2"}:
        return "mm/s^2", True
    if unit_normalized in UNCALIBRATED_UNITS:
        return "release_native", False
    raise DataContractError(f"Unsupported acceleration unit: {unit}")


def acceleration_to_analysis_units(values: np.ndarray, unit: str) -> tuple[np.ndarray, bool, str]:
    """Convert calibrated acceleration to g or preserve explicit native scale.

    Returns ``(values, calibrated, canonical_unit)``.  For uncalibrated input,
    the numeric values are preserved exactly (apart from float32 conversion).
    The caller is responsible for robust episode normalisation and for masking
    physics features that imply a physical amplitude unit.
    """

    canonical, calibrated = canonical_acceleration_unit(unit)
    array = np.asarray(values, dtype=np.float32)
    if not calibrated:
        return array, False, canonical
    return acceleration_to_g(array, canonical), True, canonical


def acceleration_to_g(values: np.ndarray, unit: str) -> np.ndarray:
    unit_normalized, calibrated = canonical_acceleration_unit(unit)
    array = np.asarray(values, dtype=np.float32)
    if not calibrated:
        raise DataContractError(
            f"Cannot convert uncalibrated acceleration unit {unit!r} to g; "
            "use acceleration_to_analysis_units() and propagate the calibration mask"
        )
    if unit_normalized == "g":
        return array
    if unit_normalized == "mg":
        return array / 1000.0
    if unit_normalized == "m/s^2":
        return array / STANDARD_GRAVITY
    if unit_normalized == "mm/s^2":
        return array / (1000.0 * STANDARD_GRAVITY)
    raise DataContractError(f"Unsupported acceleration unit: {unit}")


def g_to_acceleration(values: np.ndarray, unit: str) -> np.ndarray:
    unit_normalized, calibrated = canonical_acceleration_unit(unit)
    array = np.asarray(values, dtype=np.float32)
    if not calibrated:
        raise DataContractError(f"Cannot express g in uncalibrated unit {unit!r}")
    if unit_normalized == "g":
        return array
    if unit_normalized == "mg":
        return array * 1000.0
    if unit_normalized == "m/s^2":
        return array * STANDARD_GRAVITY
    if unit_normalized == "mm/s^2":
        return array * 1000.0 * STANDARD_GRAVITY
    raise DataContractError(f"Unsupported acceleration unit: {unit}")
