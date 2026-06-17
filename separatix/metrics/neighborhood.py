"""Neighborhood diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from separatix.config import ProfilerConfig
from separatix.densify import ensure_dense_or_sample
from separatix.models.probes import _ensure_dense_X_for_multilabel
from separatix.sampling import cap_multilabel_samples_for_budget, cap_samples_for_budget


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


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _label_entropy(values: np.ndarray) -> float:
    """Return mean binary entropy across label columns."""
    probs = values.mean(axis=0)
    mask = (probs > 0) & (probs < 1)
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(ent / np.log(2.0)))


def compute_multilabel_neighborhood_diagnostics(
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Compute local label-set similarity diagnostics."""
    X_used, Y_used, sample_info = cap_multilabel_samples_for_budget(
        X,
        Y,
        config=config,
        reason="neighbors",
    )
    Y_dense = _dense_multilabel_matrix(Y_used)
    k = min(Y_dense.shape[0] - 1, min(15, max(3, int(np.sqrt(Y_dense.shape[0])))))
    if k < 1:
        return {
            "mean_neighbor_jaccard": 1.0,
            "mean_neighbor_hamming_distance": 0.0,
            "low_jaccard_neighbor_fraction": 0.0,
            "mean_local_label_entropy": 0.0,
            "high_entropy_label_fraction": 0.0,
            "label_cardinality_local_std": 0.0,
            "all_zero_neighbor_pair_fraction": 0.0,
            "empty_union_jaccard_convention": "empty unions are scored as 0.0",
            "sampling": sample_info,
        }

    X_fit = X_used
    if sparse.issparse(X_used):
        try:
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X_fit)
        except TypeError:
            dense_info = _ensure_dense_X_for_multilabel(
                X_used,
                Y_used,
                reason="multilabel_neighborhood_diagnostics",
                config=config,
                report_context=report_context,
            )
            if dense_info["skipped"]:
                return {
                    "sampling": sample_info,
                    "skipped_reason": "dense conversion unavailable",
                }
            X_fit = dense_info["X"]
            Y_dense = _dense_multilabel_matrix(dense_info["Y"])
            nn = NearestNeighbors(n_neighbors=min(k + 1, Y_dense.shape[0]))
            nn.fit(X_fit)
    else:
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_fit)

    indices = nn.kneighbors(X_fit, n_neighbors=k + 1, return_distance=False)[:, 1:]
    jaccards: list[float] = []
    hamming_values: list[float] = []
    local_entropies: list[float] = []
    cardinality_stds: list[float] = []
    local_jaccard_means: list[float] = []
    local_hamming_means: list[float] = []
    empty_union_count = 0
    pair_count = 0
    for row_i, neighbors in enumerate(indices):
        row = Y_dense[row_i].astype(bool)
        neighbor_labels = Y_dense[neighbors].astype(bool)
        local_entropies.append(_label_entropy(neighbor_labels.astype(float)))
        cardinality_stds.append(float(np.std(neighbor_labels.sum(axis=1))))
        for neighbor in neighbor_labels:
            union = np.logical_or(row, neighbor)
            intersection = np.logical_and(row, neighbor)
            pair_count += 1
            if not np.any(union):
                empty_union_count += 1
                jaccards.append(0.0)
            else:
                jaccards.append(float(np.sum(intersection) / np.sum(union)))
            hamming_values.append(float(np.mean(row != neighbor)))
        start = pair_count - len(neighbor_labels)
        local_jaccard_means.append(float(np.mean(jaccards[start:pair_count])))
        local_hamming_means.append(float(np.mean(hamming_values[start:pair_count])))

    local_entropy_array = np.asarray(local_entropies, dtype=float)
    jaccard_array = np.asarray(jaccards, dtype=float)
    return {
        "mean_neighbor_jaccard": float(np.mean(jaccard_array)),
        "mean_neighbor_hamming_distance": float(np.mean(hamming_values)),
        "low_jaccard_neighbor_fraction": float(np.mean(jaccard_array < 0.25)),
        "mean_local_label_entropy": float(np.mean(local_entropy_array)),
        "high_entropy_label_fraction": float(np.mean(local_entropy_array >= 0.5)),
        "label_cardinality_local_std": float(np.mean(cardinality_stds)),
        "all_zero_neighbor_pair_fraction": float(
            empty_union_count / max(1, pair_count)
        ),
        "empty_union_jaccard_convention": "empty unions are scored as 0.0",
        "local_neighbor_jaccard": local_jaccard_means,
        "local_neighbor_hamming_distance": local_hamming_means,
        "local_label_entropy": local_entropies,
        "local_cardinality_std": cardinality_stds,
        "sampling": sample_info,
    }
