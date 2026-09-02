"""Signal preparation for the three-branch VibroGemma tokenizer."""

from .multiresolution import (
    MultiResolutionEpisodePreprocessor,
    MultiResolutionSettings,
    PreparedEpisode,
    PreparedEpisodeBatch,
    PreparedEpisodeContext,
    prepared_episode_batch,
    stack_prepared_episode_batch,
)
from .orientation import apply_orientation, orientation_feature_vector, validate_orientation_matrix
from .physics import PhysicsFeatureExtractor
from .profile import HealthyProfile, HealthyProfileBuilder
from .quality import assess_quality
from .resample import StatefulFIRDecimator, resample_masked_signal
from .spectral import compute_wideband_features

__all__ = [
    "HealthyProfile",
    "HealthyProfileBuilder",
    "MultiResolutionEpisodePreprocessor",
    "MultiResolutionSettings",
    "PhysicsFeatureExtractor",
    "PreparedEpisode",
    "PreparedEpisodeBatch",
    "PreparedEpisodeContext",
    "StatefulFIRDecimator",
    "apply_orientation",
    "assess_quality",
    "compute_wideband_features",
    "orientation_feature_vector",
    "prepared_episode_batch",
    "stack_prepared_episode_batch",
    "resample_masked_signal",
    "validate_orientation_matrix",
]
