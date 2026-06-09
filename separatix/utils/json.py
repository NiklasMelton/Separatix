"""JSON compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    """Convert common NumPy values into JSON-friendly builtins."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value
