"""Dense conversion helpers."""

from __future__ import annotations

from math import floor
from typing import Any

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.exceptions import DensificationError, DensificationWarning
from separatix.sampling import stratified_subsample_indices
from separatix.utils.warnings import record_warning


def ensure_dense_or_sample(
    X: Any,
    y: np.ndarray,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Return a dense matrix, optionally after stratified subsampling."""
    densification_events = report_context.setdefault("densification_events", [])
    warnings_list = report_context.setdefault("warnings", [])
    skipped = report_context.setdefault("skipped_diagnostics", [])

    if not sparse.issparse(X):
        return {"X": np.asarray(X), "y": y, "performed": False, "skipped": False}

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
        return {"X": dense, "y": y, "performed": True, "skipped": False}

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
        return {"X": None, "y": y, "performed": False, "skipped": True}

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
            return {"X": None, "y": y, "performed": False, "skipped": True}
        raise DensificationError(f"Unable to densify enough samples for {reason}.")

    indices = stratified_subsample_indices(
        y,
        n_samples=n_used,
        random_state=config.random_state,
    )
    dense = X[indices, :].toarray()
    event["sampling_used"] = True
    event["n_used"] = int(indices.shape[0])
    event["status"] = "performed_on_subsample"
    densification_events.append(event)
    if config.warn_on_densify:
        record_warning(
            f"Sparse input was stratified-subsampled then densified for {reason}.",
            warnings_list,
            DensificationWarning,
        )
    return {"X": dense, "y": y[indices], "performed": True, "skipped": False}
