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


def _slice_rows(X: Any, indices: np.ndarray) -> Any:
    return X[indices, :] if sparse.issparse(X) else X[indices]


def _fit_neighbors(
    X_used: Any,
    y_like: np.ndarray,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> tuple[Any, np.ndarray | None]:
    """Return a dense or sparse neighbor-fit matrix and aligned groups."""
    if not sparse.issparse(X_used):
        return X_used, groups
    try:
        NearestNeighbors(n_neighbors=2).fit(X_used)
        return X_used, groups
    except TypeError:
        dense_info = ensure_dense_or_sample(
            X_used,
            y_like,
            reason=reason,
            config=config,
            report_context=report_context,
            groups=groups,
        )
        if dense_info["skipped"]:
            return None, None
        return dense_info["X"], dense_info.get("groups")


def _normalize_entropy(counts: np.ndarray) -> float:
    probs = counts / max(1, counts.sum())
    positive_probs = probs[probs > 0]
    if positive_probs.size == 0:
        return 0.0
    return float(
        -np.sum(positive_probs * np.log(positive_probs))
        / max(np.log(max(2, len(probs))), 1e-9)
    )


def _singlelabel_summary(
    X_fit: Any,
    y_used: np.ndarray,
    *,
    groups: np.ndarray | None,
    cross_group: bool,
) -> dict[str, Any]:
    """Compute row-level or cross-group single-label neighborhood metrics."""
    n_samples = len(y_used)
    k = min(n_samples - 1, min(15, max(3, int(np.sqrt(n_samples)))))
    if k < 1:
        return {
            "mean_local_entropy": 0.0,
            "high_entropy_fraction": 0.0,
            "same_class_neighbor_fraction": 1.0,
            "cross_class_neighbor_fraction": 0.0,
            "mean_local_ambiguity": 0.0,
            "local_entropy": [],
            "local_ambiguity": [],
        }

    n_query = n_samples if cross_group and groups is not None else k + 1
    nn = NearestNeighbors(n_neighbors=n_query)
    nn.fit(X_fit)
    distances, indices = nn.kneighbors(X_fit, n_neighbors=n_query, return_distance=True)
    entropies: list[float] = []
    ambiguities: list[float] = []
    same_class: list[float] = []
    enemy_distances: list[float] = []
    for row_i in range(n_samples):
        neigh = indices[row_i]
        row_dist = distances[row_i]
        mask = neigh != row_i
        if cross_group and groups is not None:
            mask &= groups[neigh] != groups[row_i]
        neigh = neigh[mask][:k]
        row_dist = row_dist[mask][:k]
        if neigh.size == 0:
            continue
        neigh_labels = y_used[neigh]
        counts = np.bincount(neigh_labels, minlength=len(np.unique(y_used)))
        entropies.append(_normalize_entropy(counts))
        probs = counts / max(1, counts.sum())
        ambiguities.append(float(1.0 - probs.max()))
        same_class.append(float(np.mean(neigh_labels == y_used[row_i])))
        enemy_mask = neigh_labels != y_used[row_i]
        if np.any(enemy_mask):
            enemy_distances.append(float(np.min(row_dist[enemy_mask])))
    if not entropies:
        return {
            "mean_local_entropy": 0.0,
            "high_entropy_fraction": 0.0,
            "same_class_neighbor_fraction": 1.0,
            "cross_class_neighbor_fraction": 0.0,
            "nearest_enemy_distance_estimate": None,
            "mean_local_ambiguity": 0.0,
            "local_entropy": [],
            "local_ambiguity": [],
            "warning": "No cross-group neighbors were available.",
        }
    entropies_arr = np.asarray(entropies, dtype=float)
    same_class_arr = np.asarray(same_class, dtype=float)
    return {
        "mean_local_entropy": float(np.mean(entropies_arr)),
        "high_entropy_fraction": float(np.mean(entropies_arr >= 0.5)),
        "same_class_neighbor_fraction": float(np.mean(same_class_arr)),
        "cross_class_neighbor_fraction": float(1.0 - np.mean(same_class_arr)),
        "nearest_enemy_distance_estimate": float(np.mean(enemy_distances))
        if enemy_distances
        else None,
        "mean_local_ambiguity": float(np.mean(ambiguities)),
        "local_entropy": entropies,
        "local_ambiguity": ambiguities,
    }


def compute_neighborhood_diagnostics(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute neighborhood overlap and ambiguity diagnostics."""
    X_used, y_used, sample_info = cap_samples_for_budget(
        X,
        y,
        config=config,
        reason="neighbors",
        groups=groups,
    )
    sample_indices = np.asarray(sample_info["indices"], dtype=int)
    groups_used = groups[sample_indices] if groups is not None else None
    X_fit, dense_groups = _fit_neighbors(
        X_used,
        y_used,
        reason="neighborhood_diagnostics",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if X_fit is None:
        return {
            "sampling": sample_info,
            "row_indices": sample_indices.tolist(),
            "skipped_reason": "dense conversion unavailable",
        }
    row_level = _singlelabel_summary(
        X_fit,
        y_used,
        groups=dense_groups,
        cross_group=False,
    )
    if dense_groups is None:
        return {
            **row_level,
            "sampling": sample_info,
            "row_indices": sample_indices.tolist(),
            "primary_scope": "row_level",
        }
    cross_group = _singlelabel_summary(
        X_fit,
        y_used,
        groups=dense_groups,
        cross_group=True,
    )
    return {
        **cross_group,
        "cross_group": cross_group,
        "row_level": row_level,
        "sampling": sample_info,
        "row_indices": sample_indices.tolist(),
        "primary_scope": "cross_group",
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


def _multilabel_summary(
    X_fit: Any,
    Y_dense: np.ndarray,
    *,
    groups: np.ndarray | None,
    cross_group: bool,
) -> dict[str, Any]:
    """Compute row-level or cross-group multilabel neighborhood metrics."""
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
            "local_neighbor_jaccard": [],
            "local_neighbor_hamming_distance": [],
            "local_label_entropy": [],
            "local_cardinality_std": [],
        }
    n_query = Y_dense.shape[0] if cross_group and groups is not None else k + 1
    nn = NearestNeighbors(n_neighbors=n_query)
    nn.fit(X_fit)
    indices = nn.kneighbors(X_fit, n_neighbors=n_query, return_distance=False)
    jaccards: list[float] = []
    hamming_values: list[float] = []
    local_entropies: list[float] = []
    cardinality_stds: list[float] = []
    local_jaccard_means: list[float] = []
    local_hamming_means: list[float] = []
    empty_union_count = 0
    pair_count = 0
    for row_i in range(Y_dense.shape[0]):
        neighbors = indices[row_i]
        mask = neighbors != row_i
        if cross_group and groups is not None:
            mask &= groups[neighbors] != groups[row_i]
        neighbors = neighbors[mask][:k]
        if neighbors.size == 0:
            continue
        row = Y_dense[row_i].astype(bool)
        neighbor_labels = Y_dense[neighbors].astype(bool)
        local_entropies.append(_label_entropy(neighbor_labels.astype(float)))
        cardinality_stds.append(float(np.std(neighbor_labels.sum(axis=1))))
        row_jaccards: list[float] = []
        row_hamming: list[float] = []
        for neighbor in neighbor_labels:
            union = np.logical_or(row, neighbor)
            intersection = np.logical_and(row, neighbor)
            pair_count += 1
            if not np.any(union):
                empty_union_count += 1
                score = 0.0
            else:
                score = float(np.sum(intersection) / np.sum(union))
            row_jaccards.append(score)
            row_hamming.append(float(np.mean(row != neighbor)))
        jaccards.extend(row_jaccards)
        hamming_values.extend(row_hamming)
        local_jaccard_means.append(float(np.mean(row_jaccards)))
        local_hamming_means.append(float(np.mean(row_hamming)))
    if not jaccards:
        return {
            "mean_neighbor_jaccard": 1.0,
            "mean_neighbor_hamming_distance": 0.0,
            "low_jaccard_neighbor_fraction": 0.0,
            "mean_local_label_entropy": 0.0,
            "high_entropy_label_fraction": 0.0,
            "label_cardinality_local_std": 0.0,
            "all_zero_neighbor_pair_fraction": 0.0,
            "empty_union_jaccard_convention": "empty unions are scored as 0.0",
            "local_neighbor_jaccard": [],
            "local_neighbor_hamming_distance": [],
            "local_label_entropy": [],
            "local_cardinality_std": [],
            "warning": "No cross-group neighbors were available.",
        }
    jaccard_array = np.asarray(jaccards, dtype=float)
    local_entropy_array = np.asarray(local_entropies, dtype=float)
    return {
        "mean_neighbor_jaccard": float(np.mean(jaccard_array)),
        "mean_neighbor_hamming_distance": float(np.mean(hamming_values)),
        "low_jaccard_neighbor_fraction": float(np.mean(jaccard_array < 0.25)),
        "mean_local_label_entropy": float(np.mean(local_entropy_array)),
        "high_entropy_label_fraction": float(np.mean(local_entropy_array >= 0.5)),
        "label_cardinality_local_std": float(np.mean(cardinality_stds))
        if cardinality_stds
        else 0.0,
        "all_zero_neighbor_pair_fraction": float(
            empty_union_count / max(1, pair_count)
        ),
        "empty_union_jaccard_convention": "empty unions are scored as 0.0",
        "local_neighbor_jaccard": local_jaccard_means,
        "local_neighbor_hamming_distance": local_hamming_means,
        "local_label_entropy": local_entropies,
        "local_cardinality_std": cardinality_stds,
    }


def compute_multilabel_neighborhood_diagnostics(
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute local label-set similarity diagnostics."""
    X_used, Y_used, sample_info = cap_multilabel_samples_for_budget(
        X,
        Y,
        config=config,
        reason="neighbors",
        groups=groups,
    )
    sample_indices = np.asarray(sample_info["indices"], dtype=int)
    groups_used = groups[sample_indices] if groups is not None else None
    Y_dense = _dense_multilabel_matrix(Y_used)
    if sparse.issparse(X_used):
        dense_info = _ensure_dense_X_for_multilabel(
            X_used,
            Y_used,
            reason="multilabel_neighborhood_diagnostics",
            config=config,
            report_context=report_context,
            groups=groups_used,
        )
        if dense_info["skipped"]:
            return {
                "sampling": sample_info,
                "row_indices": sample_indices.tolist(),
                "skipped_reason": "dense conversion unavailable",
            }
        X_fit = dense_info["X"]
        Y_dense = _dense_multilabel_matrix(dense_info["Y"])
        groups_used = dense_info.get("groups")
    else:
        X_fit = X_used
    row_level = _multilabel_summary(
        X_fit, Y_dense, groups=groups_used, cross_group=False
    )
    if groups_used is None:
        return {
            **row_level,
            "sampling": sample_info,
            "row_indices": sample_indices.tolist(),
            "primary_scope": "row_level",
        }
    cross_group = _multilabel_summary(
        X_fit, Y_dense, groups=groups_used, cross_group=True
    )
    return {
        **cross_group,
        "cross_group": cross_group,
        "row_level": row_level,
        "sampling": sample_info,
        "row_indices": sample_indices.tolist(),
        "primary_scope": "cross_group",
    }
