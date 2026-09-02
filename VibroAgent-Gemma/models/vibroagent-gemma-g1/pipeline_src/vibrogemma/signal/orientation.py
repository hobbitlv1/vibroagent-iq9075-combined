from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import DataContractError


@dataclass(frozen=True, slots=True)
class OrientationResult:
    values: np.ndarray
    axis_mask: np.ndarray
    matrix: np.ndarray
    calibrated: bool
    applied: bool
    reason: str


def validate_orientation_matrix(
    matrix: np.ndarray,
    *,
    orthogonality_tolerance: float = 2e-3,
    determinant_tolerance: float = 2e-3,
) -> np.ndarray:
    """Validate a sensor-to-building rotation matrix.

    The convention used throughout the project is ``a_building = R @ a_sensor``.
    Reflections are rejected because an orientation calibration must be a proper
    right-handed rotation.
    """

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise DataContractError(f"orientation_matrix must have shape [3,3], got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise DataContractError("orientation_matrix must contain finite values")
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3), atol=orthogonality_tolerance, rtol=0.0):
        raise DataContractError("orientation_matrix must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=determinant_tolerance, rtol=0.0):
        raise DataContractError(
            f"orientation_matrix must be a proper rotation with determinant +1, got {determinant:.6f}"
        )
    return rotation.astype(np.float32)


def apply_orientation(
    values: np.ndarray,
    axis_mask: np.ndarray,
    orientation_matrix: np.ndarray | None,
) -> OrientationResult:
    """Rotate genuine tri-axis data into the common building frame when safe.

    A matrix that mixes axes cannot be applied when one or more physical axes are
    absent because doing so would synthesize information. In that case the native
    sensor frame is preserved and the calibrated matrix is still returned to the
    model as metadata.
    """

    matrix = np.asarray(values, dtype=np.float32)
    present = np.asarray(axis_mask, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise DataContractError(f"values must have shape [3,samples], got {matrix.shape}")
    if present.shape != (3,):
        raise DataContractError("axis_mask must have shape [3]")

    if orientation_matrix is None:
        identity = np.eye(3, dtype=np.float32)
        return OrientationResult(matrix.copy(), present.copy(), identity, False, False, "orientation_not_provided")

    rotation = validate_orientation_matrix(orientation_matrix)
    if not present.all():
        return OrientationResult(
            matrix.copy(),
            present.copy(),
            rotation,
            True,
            False,
            "rotation_not_applied_missing_genuine_axis",
        )

    rotated = (rotation @ matrix).astype(np.float32, copy=False)
    return OrientationResult(rotated, np.ones(3, dtype=bool), rotation, True, True, "rotated_to_building_frame")


def orientation_feature_vector(
    matrix: np.ndarray,
    *,
    calibrated: bool,
    applied: bool,
) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise DataContractError("orientation feature matrix must have shape [3,3]")
    return np.concatenate(
        (
            rotation.reshape(-1),
            np.asarray([float(calibrated), float(applied)], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
