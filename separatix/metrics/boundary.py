"""Boundary candidate diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_boundary_candidates(
    y: np.ndarray,
    neighborhood: dict[str, Any],
    probes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract boundary candidate indices from ambiguity and disagreement."""
    local_entropy = np.asarray(neighborhood.get("local_entropy", []), dtype=float)
    local_ambiguity = np.asarray(neighborhood.get("local_ambiguity", []), dtype=float)
    if local_entropy.size == 0 or local_ambiguity.size == 0:
        return {
            "candidate_indices": [],
            "candidate_fraction": 0.0,
            "boundary_sample_size": 0,
            "class_composition": {},
            "warning": "Boundary diagnostics unavailable.",
        }
    entropy_threshold = float(np.quantile(local_entropy, 0.75))
    ambiguity_threshold = max(0.33, float(np.quantile(local_ambiguity, 0.75)))
    candidate_mask = (local_entropy >= entropy_threshold) | (
        local_ambiguity >= ambiguity_threshold
    )
    linear_preds = probes.get("linear", {}).get("predictions")
    knn_preds = probes.get("knn", {}).get("predictions")
    if (
        linear_preds is not None
        and knn_preds is not None
        and len(linear_preds) == candidate_mask.shape[0]
    ):
        disagreement = np.asarray(linear_preds) != np.asarray(knn_preds)
        candidate_mask = candidate_mask | disagreement
    indices = np.flatnonzero(candidate_mask)
    counts = {
        str(cls): int(np.sum(y[indices] == cls))
        for cls in np.unique(y[indices])
        if indices.size
    }
    return {
        "candidate_indices": indices.tolist(),
        "candidate_fraction": float(indices.shape[0] / max(1, y.shape[0])),
        "boundary_sample_size": int(indices.shape[0]),
        "class_composition": counts,
        "warning": "Boundary sample is very small."
        if indices.shape[0] < max(10, len(np.unique(y)))
        else None,
    }
