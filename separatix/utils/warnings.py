"""Helpers for recording warnings in reports."""

from __future__ import annotations

import warnings


def record_warning(message: str, container: list[str], category: type[Warning]) -> None:
    """Emit and record a warning message."""
    container.append(message)
    warnings.warn(message, category, stacklevel=2)
