"""OpenSees-based, train-only augmentation utilities for VibroGemma."""

from .opensees_v017 import (
    OpenSeesRunError,
    TwinParameters,
    build_tcl_model,
    run_counterfactual_family,
)
from .conditioning_v017 import (
    MEASURED_TRANSFER_CALIBRATION_SCHEMA,
    MeasuredTransferCalibration,
    apply_simulated_transfer,
    estimate_dataset_calibration,
    fit_measured_transfer_calibration,
    measured_transfer_calibration_error,
)
from .lumo_fe_v019 import (
    LUMO_ABAQUS_SOURCE_SHA256,
    LUMO_FE_SCHEMA,
    LumoFEModel,
    LumoFEScales,
    build_lumo_eigen_tcl,
    load_lumo_fe_model,
)
from .lumo_inverse_v019 import (
    LUMO_INVERSE_CALIBRATION_SCHEMA,
    LumoInverseCalibration,
    MeasuredModalExample,
    extract_output_only_modes,
    fit_lumo_inverse_calibration,
)
from .modal_synthesis_v019 import SensorModalState, synthesize_modal_family
from .synthetic_dataset_v017 import (
    CALIBRATED_SYNTHETIC_SOURCE_KIND,
    INVERSE_FE_SYNTHETIC_SOURCE_KIND,
    LEGACY_SYNTHETIC_SOURCE_KIND,
    LUMO_INVERSE_FE_SCHEMA,
    REQUIRE_CALIBRATED_SYNTHETIC_ENV,
    REQUIRE_INVERSE_FE_SYNTHETIC_ENV,
    SYNTHETIC_MANIFEST_ENV,
    audit_synthetic_manifest,
    wrap_episode_dataset_class,
)

__all__ = [
    "OpenSeesRunError",
    "TwinParameters",
    "build_tcl_model",
    "run_counterfactual_family",
    "apply_simulated_transfer",
    "estimate_dataset_calibration",
    "fit_measured_transfer_calibration",
    "measured_transfer_calibration_error",
    "MeasuredTransferCalibration",
    "MEASURED_TRANSFER_CALIBRATION_SCHEMA",
    "LUMO_ABAQUS_SOURCE_SHA256",
    "LUMO_FE_SCHEMA",
    "LumoFEModel",
    "LumoFEScales",
    "build_lumo_eigen_tcl",
    "load_lumo_fe_model",
    "LUMO_INVERSE_CALIBRATION_SCHEMA",
    "LumoInverseCalibration",
    "MeasuredModalExample",
    "extract_output_only_modes",
    "fit_lumo_inverse_calibration",
    "SensorModalState",
    "synthesize_modal_family",
    "CALIBRATED_SYNTHETIC_SOURCE_KIND",
    "INVERSE_FE_SYNTHETIC_SOURCE_KIND",
    "LEGACY_SYNTHETIC_SOURCE_KIND",
    "LUMO_INVERSE_FE_SCHEMA",
    "REQUIRE_CALIBRATED_SYNTHETIC_ENV",
    "REQUIRE_INVERSE_FE_SYNTHETIC_ENV",
    "SYNTHETIC_MANIFEST_ENV",
    "audit_synthetic_manifest",
    "wrap_episode_dataset_class",
]
