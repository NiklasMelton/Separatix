"""Preprocessing helpers."""

from __future__ import annotations

from typing import Any


def build_preprocessing_summary(X: Any, *, is_sparse: bool) -> dict[str, Any]:
    """Summarize preprocessing state for the report."""
    return {
        "input_type": type(X).__name__,
        "is_sparse": is_sparse,
        "probe_feature_scaling": "fold_local_standard_scaler",
        "probe_scaler_centering": not is_sparse,
        "geometry_coordinate_space": "input_coordinates_unscaled",
        "random_feature_order": "scale_then_expand",
    }
