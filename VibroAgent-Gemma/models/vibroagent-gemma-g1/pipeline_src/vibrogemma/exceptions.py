class VibroGemmaError(RuntimeError):
    """Base exception for the package."""


class ConfigurationError(VibroGemmaError):
    """Configuration is missing, inconsistent, or unsafe."""


class ProvenanceError(VibroGemmaError):
    """A dataset violates the public-data or licence policy."""


class DataContractError(VibroGemmaError):
    """Input data do not satisfy the six-board signal contract."""


class ModelCompatibilityError(VibroGemmaError):
    """A model/runtime does not expose the required Gemma interfaces."""


class LiveAcquisitionError(VibroGemmaError):
    """Live STWIN.box acquisition failed or became invalid."""
