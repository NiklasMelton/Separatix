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
    if sample_info.get("support_preserved") is False or X_used.shape[0] == 0:
        return {
            "feature_scale_range_estimate": None,
            "effective_rank_estimate": None,
            "intrinsic_dimension_proxy": None,
            "distance_concentration_proxy": None,
            "high_dimensionality_flag": bool(X.shape[1] > max(100, X.shape[0] // 2)),
            "sample_to_feature_ratio": float(X.shape[0] / max(1, X.shape[1])),
            "degenerate_geometry": None,
            "skipped_reason": sample_info.get("skip_reason"),
            "sampling": sample_info,
        }
    groups_used = (
        groups[np.asarray(sample_info["indices"], dtype=int)]
        if groups is not None
        else None
    )
    if sparse.issparse(X_used):
        feature_means = np.asarray(X_used.mean(axis=0)).ravel()
        feature_squares = np.asarray(X_used.multiply(X_used).mean(axis=0)).ravel()
        sampled_variance = np.maximum(0.0, feature_squares - feature_means**2)
    else:
        sampled_variance = np.var(np.asarray(X_used), axis=0)
    all_constant = bool(np.all(sampled_variance <= 1e-12))

    if all_constant:
        embedding = np.zeros((X_used.shape[0], 1), dtype=float)
        explained = np.asarray([0.0])
        feature_scale_range = None if sparse.issparse(X_used) else 1.0
    elif X_used.shape[1] == 1:
        column = X_used.toarray() if sparse.issparse(X_used) else np.asarray(X_used)
        embedding = np.asarray(column, dtype=float).reshape(-1, 1)
        explained = np.asarray([1.0])
        feature_scale_range = None if sparse.issparse(X_used) else 1.0
    elif sparse.issparse(X_used):
        svd = TruncatedSVD(
            n_components=min(10, X_used.shape[1] - 1, max(1, X_used.shape[0] - 1)),
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

    explained = np.asarray(explained, dtype=float)
    explained = np.where(np.isfinite(explained) & (explained > 0.0), explained, 0.0)
    explained_total = float(np.sum(explained))
    degenerate_geometry = explained_total <= 1e-12
    if degenerate_geometry:
        explained = np.zeros_like(explained)
        warning = "Feature geometry is degenerate because all sampled variance is zero."
        warnings = report_context.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(warning)
    else:
        explained = explained / explained_total

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
        if not np.isfinite(concentration):
            concentration = None

    intrinsic_dim = float(np.sum(explained > (1.0 / max(embedding.shape[1], 1))))
    effective_rank = (
        0.0
        if degenerate_geometry
        else float(np.exp(-np.sum(explained * np.log(np.clip(explained, 1e-12, None)))))
    )
    return {
        "feature_scale_range_estimate": feature_scale_range,
        "effective_rank_estimate": effective_rank,
        "intrinsic_dimension_proxy": intrinsic_dim,
        "distance_concentration_proxy": concentration,
        "high_dimensionality_flag": bool(X.shape[1] > max(100, X.shape[0] // 2)),
        "sample_to_feature_ratio": float(X.shape[0] / max(1, X.shape[1])),
        "degenerate_geometry": degenerate_geometry,
        "sampling": sample_info,
    }
