"""Neighborhood diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from separatix.config import ProfilerConfig
from separatix.densify import ensure_dense_or_sample
from separatix.sampling import cap_samples_for_budget


def compute_neighborhood_diagnostics(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Compute neighborhood overlap and ambiguity diagnostics."""
    X_used, y_used, sample_info = cap_samples_for_budget(
        X, y, config=config, reason="neighbors"
    )
    k = min(len(y_used) - 1, min(15, max(3, int(np.sqrt(len(y_used))))))
    if k < 1:
        return {
            "mean_local_entropy": 0.0,
            "high_entropy_fraction": 0.0,
            "same_class_neighbor_fraction": 1.0,
            "cross_class_neighbor_fraction": 0.0,
            "mean_local_ambiguity": 0.0,
            "sampling": sample_info,
        }

    X_fit = X_used
    if sparse.issparse(X_used):
        try:
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_fit)
        except TypeError:
            dense_info = ensure_dense_or_sample(
                X_used,
                y_used,
                reason="neighborhood_diagnostics",
                config=config,
                report_context=report_context,
            )
            if dense_info["skipped"]:
                return {
                    "sampling": sample_info,
                    "skipped_reason": "dense conversion unavailable",
                }
            X_fit = dense_info["X"]
            y_used = dense_info["y"]
            nn = NearestNeighbors(n_neighbors=min(k, len(y_used) - 1))
            nn.fit(X_fit)
    else:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_fit)

    indices = nn.kneighbors(X_fit, n_neighbors=k + 1, return_distance=False)[:, 1:]
    entropies = []
    ambiguities = []
    same_class = []
    enemy_distances = []
    distances, idxs = nn.kneighbors(X_fit, n_neighbors=k + 1, return_distance=True)
    for row_i, neigh in enumerate(indices):
        neigh_labels = y_used[neigh]
        counts = np.bincount(neigh_labels, minlength=len(np.unique(y_used)))
        probs = counts / max(1, counts.sum())
        positive_probs = probs[probs > 0]
        ent = -np.sum(positive_probs * np.log(positive_probs))
        entropies.append(float(ent / max(np.log(max(2, len(probs))), 1e-9)))
        ambiguities.append(float(1.0 - probs.max()))
        same_class.append(float(np.mean(neigh_labels == y_used[row_i])))
        row_dist = distances[row_i, 1:]
        enemy_mask = neigh_labels != y_used[row_i]
        if np.any(enemy_mask):
            enemy_distances.append(float(np.min(row_dist[enemy_mask])))
    entropies_arr = np.array(entropies)
    return {
        "mean_local_entropy": float(entropies_arr.mean()),
        "high_entropy_fraction": float(np.mean(entropies_arr >= 0.5)),
        "same_class_neighbor_fraction": float(np.mean(same_class)),
        "cross_class_neighbor_fraction": float(1.0 - np.mean(same_class)),
        "nearest_enemy_distance_estimate": float(np.mean(enemy_distances))
        if enemy_distances
        else None,
        "mean_local_ambiguity": float(np.mean(ambiguities)),
        "local_entropy": entropies,
        "local_ambiguity": ambiguities,
        "sampling": sample_info,
    }
