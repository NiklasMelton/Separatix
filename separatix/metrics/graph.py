"""Graph fragmentation diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.sampling import cap_samples_for_budget


def _empty_fragmentation(indices_size: int) -> dict[str, Any]:
    return {
        "component_count": 0,
        "largest_component_fraction": 1.0 if indices_size else 0.0,
        "component_size_entropy": 0.0,
        "small_component_count": 0,
        "cross_class_edge_density": 0.0,
        "graph_fragmentation_score": 0.0,
        "warning": "Not enough boundary candidates for graph diagnostics.",
    }


def _fragmentation_summary(X_boundary: Any, y_boundary: np.ndarray) -> dict[str, Any]:
    if len(y_boundary) < 3:
        return _empty_fragmentation(len(y_boundary))

    k = min(len(y_boundary) - 1, 10)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_boundary)
    graph = nn.kneighbors_graph(X_boundary, mode="connectivity")
    graph = graph.maximum(graph.T)
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
    for _ in range(repeats):
        sample_indices = rng.integers(0, len(y_boundary), size=len(y_boundary))
        if np.unique(sample_indices).size < 3:
            continue
        summary = _fragmentation_summary(
            X_boundary[sample_indices],
            y_boundary[sample_indices],
        )
        if "warning" not in summary:
            scores.append(float(summary["graph_fragmentation_score"]))

    return {
        "graph_fragmentation_bootstrap_repeats": len(scores),
        "graph_fragmentation_bootstrap_mean": float(np.mean(scores))
        if scores
        else None,
        "graph_fragmentation_bootstrap_std": float(np.std(scores))
        if scores
        else None,
    }


def compute_graph_fragmentation(
    X: Any,
    y: np.ndarray,
    boundary: dict[str, Any],
    *,
    config: ProfilerConfig,
) -> dict[str, Any]:
    """Compute fragmentation diagnostics over boundary candidates."""
    indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    if indices.size < 3:
        return {
            **_empty_fragmentation(indices.size),
            "graph_fragmentation_bootstrap_repeats": 0,
            "graph_fragmentation_bootstrap_mean": None,
            "graph_fragmentation_bootstrap_std": None,
        }

    X_boundary = X[indices]
    y_boundary = y[indices]
    X_boundary, y_boundary, sample_info = cap_samples_for_budget(
        X_boundary, y_boundary, config=config, reason="boundary"
    )
    return {
        **_fragmentation_summary(X_boundary, y_boundary),
        **_bootstrap_fragmentation(X_boundary, y_boundary, config=config),
        "sampling": sample_info,
    }
