"""Geometry reliability diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances

from separatix.config import ProfilerConfig
from separatix.densify import ensure_dense_or_sample
from separatix.sampling import cap_samples_for_budget


def compute_geometry_diagnostics(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute cheap geometry reliability metrics."""
    X_used, y_used, sample_info = cap_samples_for_budget(
        X, y, config=config, reason="neighbors", groups=groups
    )
    groups_used = (
        groups[np.asarray(sample_info["indices"], dtype=int)]
        if groups is not None
        else None
    )
    if sparse.issparse(X_used):
        svd = TruncatedSVD(
            n_components=min(10, max(2, X_used.shape[1] - 1)),
            random_state=config.random_state,
        )
        embedding = svd.fit_transform(X_used)
        explained = svd.explained_variance_ratio_
        feature_scale_range = None
    else:
        centered = np.asarray(X_used)
        pca = PCA(
            n_components=min(10, centered.shape[1], max(1, centered.shape[0] - 1)),
            random_state=config.random_state,
        )
        embedding = pca.fit_transform(centered)
        explained = pca.explained_variance_ratio_
        std = np.std(centered, axis=0)
        feature_scale_range = float(
            std.max() / max(std[std > 0].min() if np.any(std > 0) else 1.0, 1e-9)
        )

    dense_for_dist = ensure_dense_or_sample(
        X_used,
        y_used,
        reason="geometry_distance_concentration",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if dense_for_dist["skipped"]:
        concentration = None
    else:
        sample = dense_for_dist["X"]
        if sample.shape[0] > 250:
            sample = sample[:250]
        dists = pairwise_distances(sample)
        tri = dists[np.triu_indices_from(dists, k=1)]
        concentration = (
            float((tri.max() - tri.min()) / max(tri.mean(), 1e-9)) if tri.size else 0.0
        )

    intrinsic_dim = float(np.sum(explained > (1.0 / max(embedding.shape[1], 1))))
    effective_rank = float(
        np.exp(-np.sum(explained * np.log(np.clip(explained, 1e-12, None))))
    )
    return {
        "feature_scale_range_estimate": feature_scale_range,
        "effective_rank_estimate": effective_rank,
        "intrinsic_dimension_proxy": intrinsic_dim,
        "distance_concentration_proxy": concentration,
        "high_dimensionality_flag": bool(X.shape[1] > max(100, X.shape[0] // 2)),
        "sample_to_feature_ratio": float(X.shape[0] / max(1, X.shape[1])),
        "sampling": sample_info,
    }
