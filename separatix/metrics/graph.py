"""Graph fragmentation diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.sampling import cap_multilabel_samples_for_budget, cap_samples_for_budget


def _empty_fragmentation(indices_size: int, *, warning: str) -> dict[str, Any]:
    return {
        "component_count": 0,
        "largest_component_fraction": 1.0 if indices_size else 0.0,
        "component_size_entropy": 0.0,
        "small_component_count": 0,
        "cross_class_edge_density": 0.0,
        "graph_fragmentation_score": 0.0,
        "warning": warning,
    }


def _knn_graph(
    X_boundary: Any,
    *,
    groups: np.ndarray | None,
    cross_group: bool,
) -> csr_matrix | None:
    """Build a symmetric neighborhood graph, optionally excluding same-group edges."""
    n_rows = X_boundary.shape[0]
    if n_rows < 3:
        return None
    k = min(n_rows - 1, 10)
    query_neighbors = n_rows if cross_group and groups is not None else k + 1
    nn = NearestNeighbors(n_neighbors=query_neighbors)
    nn.fit(X_boundary)
    indices = nn.kneighbors(
        X_boundary, n_neighbors=query_neighbors, return_distance=False
    )
    rows: list[int] = []
    cols: list[int] = []
    for row_idx in range(n_rows):
        neighbors = indices[row_idx]
        mask = neighbors != row_idx
        if cross_group and groups is not None:
            mask &= groups[neighbors] != groups[row_idx]
        neighbors = neighbors[mask][:k]
        for col_idx in neighbors.tolist():
            rows.append(row_idx)
            cols.append(col_idx)
    if not rows:
        return None
    data = np.ones(len(rows), dtype=int)
    graph = csr_matrix((data, (rows, cols)), shape=(n_rows, n_rows))
    return graph.maximum(graph.T)


def _fragmentation_summary(
    X_boundary: Any,
    y_boundary: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    cross_group: bool = False,
) -> dict[str, Any]:
    graph = _knn_graph(X_boundary, groups=groups, cross_group=cross_group)
    if graph is None:
        return _empty_fragmentation(
            len(y_boundary),
            warning="Not enough boundary candidates for graph diagnostics.",
        )
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    probs = sizes / max(1, sizes.sum())
    entropy = float(
        -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
        / max(np.log(max(2, len(sizes))), 1e-9)
    )
    rows, cols = graph.nonzero()
    valid = rows < cols
    cross_class = (
        np.mean(y_boundary[rows[valid]] != y_boundary[cols[valid]])
        if np.any(valid)
        else 0.0
    )
    largest = float(sizes.max() / max(1, sizes.sum()))
    component_count_scaled = min(1.0, n_components / max(3, len(y_boundary) / 20))
    fragmentation = float(
        np.clip(np.mean([1.0 - largest, component_count_scaled, entropy]), 0.0, 1.0)
    )
    return {
        "component_count": int(n_components),
        "largest_component_fraction": largest,
        "component_size_entropy": entropy,
        "small_component_count": int(np.sum(sizes <= 3)),
        "cross_class_edge_density": float(cross_class),
        "graph_fragmentation_score": fragmentation,
    }


def _bootstrap_fragmentation(
    X_boundary: Any,
    y_boundary: np.ndarray,
    *,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
    cross_group: bool = False,
) -> dict[str, int | float | None]:
    budget = cast(dict[str, object], BUDGETS[config.budget])
    repeats = int(cast(int, budget["bootstrap_repeats"]))
    if repeats <= 0 or len(y_boundary) < 3:
        return {
            "graph_fragmentation_bootstrap_repeats": 0,
            "graph_fragmentation_bootstrap_mean": None,
            "graph_fragmentation_bootstrap_std": None,
        }

    rng = np.random.default_rng(config.random_state)
    scores: list[float] = []
    if groups is None:
        for _ in range(repeats):
            sample_indices = rng.integers(0, len(y_boundary), size=len(y_boundary))
            if np.unique(sample_indices).size < 3:
                continue
            summary = _fragmentation_summary(
                X_boundary[sample_indices],
                y_boundary[sample_indices],
                cross_group=cross_group,
            )
            if "warning" not in summary:
                scores.append(float(summary["graph_fragmentation_score"]))
    else:
        unique_groups = np.unique(groups)
        for _ in range(repeats):
            sampled_groups = rng.choice(
                unique_groups, size=unique_groups.shape[0], replace=True
            )
            row_blocks = [
                np.flatnonzero(groups == group_id) for group_id in sampled_groups
            ]
            sample_indices = np.concatenate(row_blocks).astype(int)
            if np.unique(sample_indices).size < 3:
                continue
            summary = _fragmentation_summary(
                X_boundary[sample_indices],
                y_boundary[sample_indices],
                groups=groups[sample_indices],
                cross_group=cross_group,
            )
            if "warning" not in summary:
                scores.append(float(summary["graph_fragmentation_score"]))
    return {
        "graph_fragmentation_bootstrap_repeats": len(scores),
        "graph_fragmentation_bootstrap_mean": float(np.mean(scores))
        if scores
        else None,
        "graph_fragmentation_bootstrap_std": float(np.std(scores)) if scores else None,
    }


def compute_graph_fragmentation(
    X: Any,
    y: np.ndarray,
    boundary: dict[str, Any],
    *,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute fragmentation diagnostics over boundary candidates."""
    indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    if indices.size < 3:
        return {
            **_empty_fragmentation(
                indices.size,
                warning="Not enough boundary candidates for graph diagnostics.",
            ),
            "graph_fragmentation_bootstrap_repeats": 0,
            "graph_fragmentation_bootstrap_mean": None,
            "graph_fragmentation_bootstrap_std": None,
        }

    X_boundary = X[indices] if hasattr(X, "__getitem__") else X
    y_boundary = y[indices]
    groups_boundary = groups[indices] if groups is not None else None
    X_boundary, y_boundary, sample_info = cap_samples_for_budget(
        X_boundary,
        y_boundary,
        config=config,
        reason="boundary",
        groups=groups_boundary,
    )
    groups_used = (
        groups_boundary[np.asarray(sample_info["indices"], dtype=int)]
        if groups_boundary is not None
        else None
    )
    row_level = _fragmentation_summary(X_boundary, y_boundary)
    if groups_used is None:
        return {
            **row_level,
            **_bootstrap_fragmentation(X_boundary, y_boundary, config=config),
            "sampling": sample_info,
            "primary_scope": "row_level",
        }
    cross_group = _fragmentation_summary(
        X_boundary,
        y_boundary,
        groups=groups_used,
        cross_group=True,
    )
    return {
        **cross_group,
        **_bootstrap_fragmentation(
            X_boundary,
            y_boundary,
            config=config,
            groups=groups_used,
            cross_group=True,
        ),
        "cross_group": cross_group,
        "row_level": row_level,
        "sampling": sample_info,
        "primary_scope": "cross_group",
    }


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _multilabel_edge_metrics(
    Y_boundary: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> dict[str, float]:
    """Return multilabel edge disagreement summaries."""
    valid = rows < cols
    if not np.any(valid):
        return {
            "mean_edge_label_jaccard": 1.0,
            "mean_edge_hamming_distance": 0.0,
            "low_label_overlap_edge_fraction": 0.0,
        }
    rows = rows[valid]
    cols = cols[valid]
    first = Y_boundary[rows].astype(bool)
    second = Y_boundary[cols].astype(bool)
    intersection = np.logical_and(first, second).sum(axis=1).astype(float)
    union = np.logical_or(first, second).sum(axis=1).astype(float)
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )
    hamming = np.mean(first != second, axis=1)
    return {
        "mean_edge_label_jaccard": float(np.mean(jaccard)),
        "mean_edge_hamming_distance": float(np.mean(hamming)),
        "low_label_overlap_edge_fraction": float(np.mean(jaccard < 0.25)),
    }


def _multilabel_component_metrics(
    Y_boundary: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Return component-level multilabel diversity summaries."""
    component_diversities: list[float] = []
    component_cardinality_variances: list[float] = []
    for component in np.unique(labels):
        component_indices = np.flatnonzero(labels == component)
        component_rows = Y_boundary[component_indices]
        if component_rows.shape[0] == 0:
            continue
        row_keys = {tuple(row.tolist()) for row in component_rows}
        component_diversities.append(len(row_keys) / max(1, component_rows.shape[0]))
        component_cardinality_variances.append(
            float(np.var(component_rows.sum(axis=1).astype(float)))
        )
    return {
        "component_label_diversity": float(np.mean(component_diversities))
        if component_diversities
        else 0.0,
        "component_cardinality_variance": float(
            np.mean(component_cardinality_variances)
        )
        if component_cardinality_variances
        else 0.0,
    }


def _multilabel_fragmentation_summary(
    X_boundary: Any,
    Y_boundary: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    cross_group: bool = False,
) -> dict[str, Any]:
    graph = _knn_graph(X_boundary, groups=groups, cross_group=cross_group)
    if graph is None:
        return {
            "component_count": 0,
            "largest_component_fraction": 1.0 if Y_boundary.shape[0] else 0.0,
            "component_size_entropy": 0.0,
            "small_component_count": 0,
            "graph_fragmentation_score": 0.0,
            "mean_edge_label_jaccard": 1.0,
            "mean_edge_hamming_distance": 0.0,
            "low_label_overlap_edge_fraction": 0.0,
            "component_label_diversity": 0.0,
            "component_cardinality_variance": 0.0,
            "warning": (
                "Not enough multilabel boundary candidates for graph diagnostics."
            ),
        }
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    probs = sizes / max(1, sizes.sum())
    entropy = float(
        -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
        / max(np.log(max(2, len(sizes))), 1e-9)
    )
    largest = float(sizes.max() / max(1, sizes.sum()))
    component_count_scaled = min(1.0, n_components / max(3, len(Y_boundary) / 20))
    rows, cols = graph.nonzero()
    return {
        "component_count": int(n_components),
        "largest_component_fraction": largest,
        "component_size_entropy": entropy,
        "small_component_count": int(np.sum(sizes <= 3)),
        "graph_fragmentation_score": float(
            np.clip(np.mean([1.0 - largest, component_count_scaled, entropy]), 0.0, 1.0)
        ),
        **_multilabel_edge_metrics(Y_boundary, rows, cols),
        **_multilabel_component_metrics(Y_boundary, labels),
    }


def compute_multilabel_graph_fragmentation(
    X: Any,
    Y: Any,
    boundary: dict[str, Any],
    *,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute graph fragmentation summaries over multilabel boundary candidates."""
    indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    if indices.size < 3:
        return _multilabel_fragmentation_summary(
            np.zeros((0, 1)), np.zeros((0, 1), dtype=int)
        )
    X_boundary = X[indices] if hasattr(X, "__getitem__") else X
    Y_boundary = Y[indices] if not hasattr(Y, "toarray") else Y[indices, :]
    groups_boundary = groups[indices] if groups is not None else None
    X_boundary, Y_boundary, sample_info = cap_multilabel_samples_for_budget(
        X_boundary,
        Y_boundary,
        config=config,
        reason="boundary",
        groups=groups_boundary,
    )
    Y_dense = _dense_multilabel_matrix(Y_boundary)
    groups_used = (
        groups_boundary[np.asarray(sample_info["indices"], dtype=int)]
        if groups_boundary is not None
        else None
    )
    row_level = _multilabel_fragmentation_summary(X_boundary, Y_dense)
    if groups_used is None:
        return {
            **row_level,
            "sampling": sample_info,
            "primary_scope": "row_level",
        }
    cross_group = _multilabel_fragmentation_summary(
        X_boundary,
        Y_dense,
        groups=groups_used,
        cross_group=True,
    )
    return {
        **cross_group,
        "cross_group": cross_group,
        "row_level": row_level,
        "sampling": sample_info,
        "primary_scope": "cross_group",
    }
