"""Dataset audit metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_dataset_audit(
    X: Any,
    y: np.ndarray,
    *,
    classes: np.ndarray,
    is_sparse: bool,
) -> dict[str, Any]:
    """Compute cheap dataset audit statistics."""
    class_ids, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    imbalance_ratio = float(counts.max() / max(1, counts.min()))
    result: dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(classes.shape[0]),
        "class_counts": {
            str(classes[i]): int(counts[idx]) for idx, i in enumerate(class_ids)
        },
        "class_proportions": {
            str(classes[i]): float(proportions[idx]) for idx, i in enumerate(class_ids)
        },
        "imbalance_ratio": imbalance_ratio,
        "is_sparse": is_sparse,
    }
    if is_sparse:
        density = (
            float(X.nnz / (X.shape[0] * X.shape[1]))
            if X.shape[0] and X.shape[1]
            else 0.0
        )
        result.update(
            {
                "nnz": int(X.nnz),
                "density": density,
                "sparsity_fraction": float(1.0 - density),
                "estimated_dense_memory_mb": float(
                    X.shape[0] * X.shape[1] * X.dtype.itemsize / 1024**2
                ),
                "dtype": str(X.dtype),
            }
        )
    else:
        result.update(
            {
                "dtype": str(X.dtype),
                "constant_feature_fraction": float(
                    np.mean(np.nanstd(X, axis=0) == 0.0)
                ),
                "estimated_dense_memory_mb": float(X.nbytes / 1024**2),
            }
        )
    return result
