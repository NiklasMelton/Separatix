"""Preprocessing helpers."""

from __future__ import annotations

from typing import Any


def build_preprocessing_summary(X: Any, *, is_sparse: bool) -> dict[str, Any]:
    """Summarize preprocessing state for the report."""
    return {
        "input_type": type(X).__name__,
        "is_sparse": is_sparse,
    }
