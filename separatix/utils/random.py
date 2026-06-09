"""Random-state utilities."""

from __future__ import annotations

import numpy as np


def make_rng(random_state: int | None) -> np.random.Generator:
    """Create a deterministic NumPy generator."""
    return np.random.default_rng(random_state)
