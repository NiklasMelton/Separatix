"""Custom exceptions and warnings for separatix."""


class SeparatixError(Exception):
    """Base exception for separatix."""


class DensificationError(SeparatixError):
    """Raised when dense conversion is required but disallowed or impossible."""


class DensificationWarning(UserWarning):
    """Warning emitted when sparse data are densified or subsampled."""


class ProbeRecipeError(SeparatixError):
    """Base error for malformed or unsupported serialized probe recipes."""


class UnsupportedProbeRecipeVersion(ProbeRecipeError):
    """Raised when a recipe schema version is not supported by this package."""


class ProbeRecipeCompatibilityError(ProbeRecipeError):
    """Raised when a valid recipe is incompatible with the current runtime."""
