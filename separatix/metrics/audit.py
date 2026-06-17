"""Dataset audit metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from separatix.validation import ValidatedMultilabelInput


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


def _top_label_counts(
    names: np.ndarray, counts: np.ndarray, *, limit: int = 10
) -> list[dict[str, Any]]:
    """Return the largest multilabel counts in report-friendly form."""
    order = np.argsort(-counts)[:limit]
    return [
        {"label": str(names[idx]), "count": int(counts[idx])}
        for idx in order
        if counts[idx] > 0
    ]


def compute_multilabel_audit(
    validated: ValidatedMultilabelInput,
) -> dict[str, Any]:
    """Compute cheap multilabel dataset audit statistics."""
    Y = validated.Y
    row_counts = np.asarray(Y.sum(axis=1)).ravel().astype(float)
    constant_count = int(np.sum(~validated.usable_label_mask))
    rare_count = int(
        np.sum(
            (validated.label_counts > 0)
            & (validated.label_counts < 2)
            & (validated.n_samples - validated.label_counts >= 2)
        )
    )
    result: dict[str, Any] = {
        "target_type": "multilabel",
        "n_samples": validated.n_samples,
        "n_features": validated.n_features,
        "n_labels": validated.n_labels,
        "usable_label_count": int(np.sum(validated.usable_label_mask)),
        "constant_or_too_rare_label_count": constant_count,
        "rare_label_count": rare_count,
        "label_cardinality_mean": float(np.mean(row_counts)),
        "label_cardinality_std": float(np.std(row_counts)),
        "label_density": float(np.mean(row_counts) / max(1, validated.n_labels)),
        "all_zero_sample_count": validated.all_zero_sample_count,
        "all_zero_sample_fraction": float(
            validated.all_zero_sample_count / max(1, validated.n_samples)
        ),
        "is_sparse_X": validated.is_sparse_X,
        "is_sparse_Y": validated.is_sparse_Y,
        "label_count_summary": {
            "min": int(np.min(validated.label_counts)),
            "median": float(np.median(validated.label_counts)),
            "max": int(np.max(validated.label_counts)),
        },
        "label_prevalence_summary": {
            "min": float(np.min(validated.label_prevalence)),
            "median": float(np.median(validated.label_prevalence)),
            "max": float(np.max(validated.label_prevalence)),
        },
        "top_label_counts": _top_label_counts(
            validated.label_names, validated.label_counts
        ),
    }
    X = validated.X
    if validated.is_sparse_X:
        density = (
            float(X.nnz / (X.shape[0] * X.shape[1]))
            if X.shape[0] and X.shape[1]
            else 0.0
        )
        result.update(
            {
                "X_nnz": int(X.nnz),
                "X_density": density,
                "X_sparsity_fraction": float(1.0 - density),
                "X_estimated_dense_memory_mb": float(
                    X.shape[0] * X.shape[1] * X.dtype.itemsize / 1024**2
                ),
                "X_dtype": str(X.dtype),
            }
        )
    else:
        result.update(
            {
                "X_dtype": str(X.dtype),
                "X_constant_feature_fraction": float(
                    np.mean(np.nanstd(X, axis=0) == 0.0)
                ),
                "X_estimated_dense_memory_mb": float(X.nbytes / 1024**2),
            }
        )
    return result
