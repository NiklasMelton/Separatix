"""Graph fragmentation diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from separatix.config import ProfilerConfig
from separatix.sampling import cap_samples_for_budget


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
            "component_count": 0,
            "largest_component_fraction": 1.0 if indices.size else 0.0,
            "component_size_entropy": 0.0,
            "small_component_count": 0,
            "cross_class_edge_density": 0.0,
            "graph_fragmentation_score": 0.0,
            "warning": "Not enough boundary candidates for graph diagnostics.",
        }

    X_boundary = X[indices]
    y_boundary = y[indices]
    X_boundary, y_boundary, sample_info = cap_samples_for_budget(
        X_boundary, y_boundary, config=config, reason="boundary"
    )
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
        "sampling": sample_info,
    }
