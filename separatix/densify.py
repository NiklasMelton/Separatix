"""Dense conversion helpers."""

from __future__ import annotations

from math import floor
from typing import Any

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.exceptions import DensificationError, DensificationWarning
from separatix.sampling import (
    grouped_regression_subsample_indices,
    multilabel_subsample_indices,
    stratified_subsample_indices,
)
from separatix.utils.warnings import record_warning


def ensure_dense_or_sample(
    X: Any,
    y: np.ndarray,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a dense matrix, optionally after stratified subsampling."""
    densification_events = report_context.setdefault("densification_events", [])
    warnings_list = report_context.setdefault("warnings", [])
    skipped = report_context.setdefault("skipped_diagnostics", [])

    if not sparse.issparse(X):
        return {
            "X": np.asarray(X),
            "y": y,
            "groups": groups,
            "indices": np.arange(y.shape[0], dtype=int),
            "performed": False,
            "skipped": False,
        }

    dtype = X.dtype if X.dtype is not None else np.dtype(float)
    estimated_mb = X.shape[0] * X.shape[1] * np.dtype(dtype).itemsize / 1024**2
    event = {
        "operation": "densify",
        "reason": reason,
        "input_shape": [int(X.shape[0]), int(X.shape[1])],
        "estimated_full_dense_mb": float(estimated_mb),
        "max_dense_mb": config.max_dense_mb,
        "policy": config.densify_policy,
        "sampling_used": False,
        "n_original": int(X.shape[0]),
        "n_used": int(X.shape[0]),
        "status": "performed",
    }

    if estimated_mb <= config.max_dense_mb:
        dense = X.toarray()
        densification_events.append(event)
        if config.warn_on_densify:
            record_warning(
                f"Sparse input densified for {reason}.",
                warnings_list,
                DensificationWarning,
            )
        return {
            "X": dense,
            "y": y,
            "groups": groups,
            "indices": np.arange(y.shape[0], dtype=int),
            "performed": True,
            "skipped": False,
        }

    if config.densify_policy == "fail":
        message = (
            f"Dense conversion for {reason} would exceed "
            f"max_dense_mb={config.max_dense_mb}."
        )
        raise DensificationError(message)

    if config.densify_policy == "skip":
        event["status"] = "skipped"
        densification_events.append(event)
        skipped.append(
            {
                "name": reason,
                "reason": "dense conversion exceeds configured memory budget",
            }
        )
        return {
            "X": None,
            "y": y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }

    max_rows = floor(
        (config.max_dense_mb * 1024**2) / (X.shape[1] * np.dtype(dtype).itemsize)
    )
    n_used = min(X.shape[0], max_rows, config.max_samples or X.shape[0])
    if n_used < min(config.min_dense_samples, X.shape[0]):
        skipped.append({"name": reason, "reason": "dense subsample would be too small"})
        event["status"] = "skipped_too_small"
        event["n_used"] = int(max(n_used, 0))
        densification_events.append(event)
        if config.densify_policy == "warn_and_sample":
            return {
                "X": None,
                "y": y,
                "groups": groups,
                "indices": np.array([], dtype=int),
                "performed": False,
                "skipped": True,
            }
        raise DensificationError(f"Unable to densify enough samples for {reason}.")

    if groups is not None:
        from separatix.sampling import grouped_stratified_subsample_indices

        indices = grouped_stratified_subsample_indices(
            y,
            groups,
            n_samples=n_used,
            random_state=config.random_state,
        )
    else:
        indices = stratified_subsample_indices(
            y,
            n_samples=n_used,
            random_state=config.random_state,
        )
    if indices.size == 0 or not np.array_equal(np.unique(y[indices]), np.unique(y)):
        skipped.append(
            {
                "name": reason,
                "reason": (
                    "no support-preserving dense subsample fits the memory budget"
                ),
            }
        )
        event["status"] = "skipped_too_small"
        event["n_used"] = int(indices.size)
        densification_events.append(event)
        return {
            "X": None,
            "y": y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }
    dense = X[indices, :].toarray()
    event["sampling_used"] = True
    event["n_used"] = int(indices.shape[0])
    event["status"] = "performed_on_subsample"
    densification_events.append(event)
    if config.warn_on_densify:
        record_warning(
            (
                "Sparse input was "
                f"{'group-aware ' if groups is not None else ''}"
                f"stratified-subsampled then densified for {reason}."
            ),
            warnings_list,
            DensificationWarning,
        )
    return {
        "X": dense,
        "y": y[indices],
        "groups": groups[indices] if groups is not None else None,
        "indices": indices,
        "performed": True,
        "skipped": False,
    }


def ensure_dense_or_sample_regression(
    X: Any,
    Y: np.ndarray,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a dense matrix for regression, using non-stratified subsampling."""
    densification_events = report_context.setdefault("densification_events", [])
    warnings_list = report_context.setdefault("warnings", [])
    skipped = report_context.setdefault("skipped_diagnostics", [])

    if not sparse.issparse(X):
        return {
            "X": np.asarray(X),
            "y": Y,
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": False,
            "skipped": False,
        }

    dtype = X.dtype if X.dtype is not None else np.dtype(float)
    estimated_mb = X.shape[0] * X.shape[1] * np.dtype(dtype).itemsize / 1024**2
    event = {
        "operation": "densify",
        "reason": reason,
        "input_shape": [int(X.shape[0]), int(X.shape[1])],
        "estimated_full_dense_mb": float(estimated_mb),
        "max_dense_mb": config.max_dense_mb,
        "policy": config.densify_policy,
        "sampling_used": False,
        "n_original": int(X.shape[0]),
        "n_used": int(X.shape[0]),
        "status": "performed",
    }
    if estimated_mb <= config.max_dense_mb:
        dense = X.toarray()
        densification_events.append(event)
        if config.warn_on_densify:
            record_warning(
                f"Sparse input densified for {reason}.",
                warnings_list,
                DensificationWarning,
            )
        return {
            "X": dense,
            "y": Y,
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": True,
            "skipped": False,
        }
    if config.densify_policy == "fail":
        raise DensificationError(
            f"Dense conversion for {reason} would exceed "
            f"max_dense_mb={config.max_dense_mb}."
        )
    if config.densify_policy == "skip":
        event["status"] = "skipped"
        densification_events.append(event)
        skipped.append(
            {
                "name": reason,
                "reason": "dense conversion exceeds configured memory budget",
            }
        )
        return {
            "X": None,
            "y": Y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }
    max_rows = floor(
        (config.max_dense_mb * 1024**2) / (X.shape[1] * np.dtype(dtype).itemsize)
    )
    n_used = min(X.shape[0], max_rows, config.max_samples or X.shape[0])
    if n_used < min(config.min_dense_samples, X.shape[0]):
        skipped.append({"name": reason, "reason": "dense subsample would be too small"})
        event["status"] = "skipped_too_small"
        event["n_used"] = int(max(n_used, 0))
        densification_events.append(event)
        if config.densify_policy == "warn_and_sample":
            return {
                "X": None,
                "y": Y,
                "groups": groups,
                "indices": np.array([], dtype=int),
                "performed": False,
                "skipped": True,
            }
        raise DensificationError(f"Unable to densify enough samples for {reason}.")

    if groups is not None:
        indices = grouped_regression_subsample_indices(
            groups,
            n_samples=n_used,
            random_state=config.random_state,
        )
    else:
        from separatix.sampling import random_subsample_indices

        indices = random_subsample_indices(
            X.shape[0],
            n_samples=n_used,
            random_state=config.random_state,
        )
    if indices.size < 4:
        skipped.append(
            {
                "name": reason,
                "reason": "no evaluable regression subsample fits the memory budget",
            }
        )
        event["status"] = "skipped_too_small"
        event["n_used"] = int(indices.size)
        densification_events.append(event)
        return {
            "X": None,
            "y": Y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }
    dense = X[indices, :].toarray()
    event["sampling_used"] = True
    event["sampling_method"] = "random"
    event["n_used"] = int(indices.shape[0])
    event["status"] = "performed_on_subsample"
    densification_events.append(event)
    if config.warn_on_densify:
        record_warning(
            f"Sparse input was randomly subsampled then densified for {reason}.",
            warnings_list,
            DensificationWarning,
        )
    return {
        "X": dense,
        "y": Y[indices],
        "groups": groups[indices] if groups is not None else None,
        "indices": indices,
        "performed": True,
        "skipped": False,
    }


def ensure_dense_or_sample_multilabel(
    X: Any,
    Y: Any,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Densify sparse multilabel features while preserving all aligned arrays."""
    if not sparse.issparse(X):
        return {
            "X": np.asarray(X),
            "Y": Y,
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": False,
            "skipped": False,
        }

    densification_events = report_context.setdefault("densification_events", [])
    warnings_list = report_context.setdefault("warnings", [])
    skipped = report_context.setdefault("skipped_diagnostics", [])
    dtype = X.dtype if X.dtype is not None else np.dtype(float)
    estimated_mb = X.shape[0] * X.shape[1] * np.dtype(dtype).itemsize / 1024**2
    event = {
        "operation": "densify_features",
        "reason": reason,
        "input_shape": [int(X.shape[0]), int(X.shape[1])],
        "estimated_full_dense_mb": float(estimated_mb),
        "max_dense_mb": config.max_dense_mb,
        "policy": config.densify_policy,
        "sampling_used": False,
        "n_original": int(X.shape[0]),
        "n_used": int(X.shape[0]),
        "status": "performed",
    }
    if estimated_mb <= config.max_dense_mb:
        densification_events.append(event)
        if config.warn_on_densify:
            record_warning(
                f"Sparse input densified for {reason}.",
                warnings_list,
                DensificationWarning,
            )
        return {
            "X": X.toarray(),
            "Y": Y,
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": True,
            "skipped": False,
        }
    if config.densify_policy == "fail":
        raise DensificationError(
            f"Dense conversion for {reason} would exceed "
            f"max_dense_mb={config.max_dense_mb}."
        )
    if config.densify_policy == "skip":
        event["status"] = "skipped"
        densification_events.append(event)
        skipped.append(
            {
                "name": reason,
                "reason": "dense conversion exceeds configured memory budget",
            }
        )
        return {
            "X": None,
            "Y": Y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }

    max_rows = floor(
        (config.max_dense_mb * 1024**2) / (X.shape[1] * np.dtype(dtype).itemsize)
    )
    n_used = min(X.shape[0], max_rows, config.max_samples or X.shape[0])
    indices, method = multilabel_subsample_indices(
        Y,
        n_samples=n_used,
        config=config,
        groups=groups,
    )
    if indices.size < min(config.min_dense_samples, X.shape[0]) or indices.size == 0:
        event["status"] = "skipped_too_small"
        event["n_used"] = int(indices.size)
        densification_events.append(event)
        skipped.append(
            {
                "name": reason,
                "reason": (
                    "no support-preserving dense subsample fits the memory budget"
                ),
            }
        )
        return {
            "X": None,
            "Y": Y,
            "groups": groups,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }
    event.update(
        {
            "sampling_used": True,
            "sampling_method": method,
            "n_used": int(indices.size),
            "status": "performed_on_subsample",
        }
    )
    densification_events.append(event)
    if config.warn_on_densify:
        record_warning(
            f"Sparse input was multilabel-subsampled then densified for {reason}.",
            warnings_list,
            DensificationWarning,
        )
    Y_used = Y[indices, :] if sparse.issparse(Y) else np.asarray(Y)[indices]
    return {
        "X": X[indices, :].toarray(),
        "Y": Y_used,
        "groups": groups[indices] if groups is not None else None,
        "indices": indices,
        "performed": True,
        "skipped": False,
    }


def ensure_dense_multilabel_target(
    X: Any,
    Y: Any,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Densify a sparse indicator target only within the configured budget."""
    if not sparse.issparse(Y):
        return {
            "X": X,
            "Y": np.asarray(Y, dtype=np.int8),
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": False,
            "skipped": False,
        }
    estimated_mb = Y.shape[0] * Y.shape[1] * np.dtype(np.int8).itemsize / 1024**2
    events = report_context.setdefault("densification_events", [])
    event = {
        "operation": "densify_target",
        "reason": reason,
        "input_shape": [int(Y.shape[0]), int(Y.shape[1])],
        "estimated_full_dense_mb": float(estimated_mb),
        "max_dense_mb": config.max_dense_mb,
        "policy": config.densify_policy,
        "sampling_used": False,
        "n_original": int(Y.shape[0]),
        "n_used": int(Y.shape[0]),
        "status": "performed",
    }
    if estimated_mb <= config.max_dense_mb:
        events.append(event)
        return {
            "X": X,
            "Y": Y.toarray().astype(np.int8, copy=False),
            "groups": groups,
            "indices": np.arange(Y.shape[0], dtype=int),
            "performed": True,
            "skipped": False,
        }
    if config.densify_policy == "fail":
        raise DensificationError(
            f"Dense target conversion for {reason} would exceed "
            f"max_dense_mb={config.max_dense_mb}."
        )
    if config.densify_policy == "skip":
        indices = np.array([], dtype=int)
    else:
        max_rows = floor(
            (config.max_dense_mb * 1024**2)
            / max(1, Y.shape[1] * np.dtype(np.int8).itemsize)
        )
        n_used = min(Y.shape[0], max_rows, config.max_samples or Y.shape[0])
        indices, method = multilabel_subsample_indices(
            Y,
            n_samples=n_used,
            config=config,
            groups=groups,
        )
        event["sampling_method"] = method
    if indices.size < min(config.min_dense_samples, Y.shape[0]):
        event.update({"status": "skipped", "n_used": int(indices.size)})
        events.append(event)
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": reason,
                "reason": "dense target conversion exceeds configured memory budget",
            }
        )
        return {
            "X": None,
            "Y": None,
            "groups": None,
            "indices": np.array([], dtype=int),
            "performed": False,
            "skipped": True,
        }
    event.update(
        {
            "sampling_used": True,
            "n_used": int(indices.size),
            "status": "performed_on_subsample",
        }
    )
    events.append(event)
    return {
        "X": X[indices, :] if sparse.issparse(X) else np.asarray(X)[indices],
        "Y": Y[indices, :].toarray().astype(np.int8, copy=False),
        "groups": groups[indices] if groups is not None else None,
        "indices": indices,
        "performed": True,
        "skipped": False,
    }
