"""Custom exceptions and warnings for separatix."""


class SeparatixError(Exception):
    """Base exception for separatix."""


class DensificationError(SeparatixError):
    """Raised when dense conversion is required but disallowed or impossible."""


class DensificationWarning(UserWarning):
    """Warning emitted when sparse data are densified or subsampled."""
