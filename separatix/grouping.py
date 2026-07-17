"""Helpers for validating and summarizing group identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GroupInfo:
    """Validated internal representation of sample groups."""

    encoded: np.ndarray
    n_groups: int
    group_sizes: np.ndarray


def _is_missing_group_value(value: Any) -> bool:
    """Return whether one group value should be treated as missing."""
    if value is None:
        return True
    if value is np.ma.masked:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    try:
        unequal = value != value
        if isinstance(unequal, (bool, np.bool_)):
            return bool(unequal)
        # pandas.NA and similar scalar sentinels propagate an indeterminate
        # comparison result instead of a bool.
        if type(unequal).__name__ == "NAType":
            return True
    except (TypeError, ValueError):
        return True
    return False


def validate_groups(groups: Any, *, n_samples: int) -> GroupInfo | None:
    """Validate optional grouping identifiers and encode them to integers."""
    if groups is None:
        return None

    if hasattr(groups, "to_numpy"):
        groups = groups.to_numpy()
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if values.shape[0] != n_samples:
        raise ValueError("groups must have the same number of rows as X and y.")

    encoded = np.empty(n_samples, dtype=int)
    mapping: dict[Any, int] = {}
    next_id = 0
    for idx, raw in enumerate(values.tolist()):
        if _is_missing_group_value(raw):
            raise ValueError("groups contains missing values.")
        if isinstance(raw, (float, np.floating)) and not np.isfinite(raw):
            raise ValueError("groups contains non-finite values.")
        try:
            hash(raw)
        except TypeError as exc:
            raise ValueError("groups values must be hashable.") from exc
        if raw not in mapping:
            mapping[raw] = next_id
            next_id += 1
        encoded[idx] = mapping[raw]

    group_sizes = np.bincount(encoded).astype(int, copy=False)
    return GroupInfo(
        encoded=encoded,
        n_groups=int(group_sizes.shape[0]),
        group_sizes=group_sizes,
    )


def summarize_groups(groups: np.ndarray | None) -> dict[str, Any]:
    """Return report-safe summary metadata for optional grouping."""
    if groups is None:
        return {
            "provided": False,
            "group_count": None,
            "group_size_summary": None,
        }

    sizes = np.bincount(groups).astype(int, copy=False)
    return {
        "provided": True,
        "group_count": int(sizes.shape[0]),
        "group_size_summary": {
            "min": int(np.min(sizes)),
            "median": float(np.median(sizes)),
            "max": int(np.max(sizes)),
        },
    }


def singlelabel_group_support(
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Return class-level support metadata for grouped single-label evaluation."""
    classes = np.unique(y)
    groups_per_class = {
        int(cls): int(np.unique(groups[y == cls]).shape[0]) for cls in classes
    }
    supported_classes = [int(cls) for cls in classes if groups_per_class[int(cls)] >= 2]
    skipped_classes = [int(cls) for cls in classes if groups_per_class[int(cls)] < 2]
    evaluable_mask = np.isin(y, supported_classes)
    return {
        "groups_per_class": groups_per_class,
        "supported_classes": supported_classes,
        "skipped_classes": skipped_classes,
        "evaluable_mask": evaluable_mask,
    }


def multilabel_group_support(
    Y: Any,
    groups: np.ndarray,
) -> np.ndarray:
    """Return a mask of multilabel columns with group support on both sides."""
    n_labels = int(Y.shape[1])
    positive_group_counts = np.zeros(n_labels, dtype=int)
    negative_group_counts = np.zeros(n_labels, dtype=int)
    for label_idx in range(n_labels):
        if sparse.issparse(Y):
            positives = np.asarray(Y[:, label_idx].toarray()).ravel() > 0
        else:
            positives = np.asarray(Y)[:, label_idx] > 0
        positive_group_counts[label_idx] = int(np.unique(groups[positives]).shape[0])
        negative_group_counts[label_idx] = int(np.unique(groups[~positives]).shape[0])
    return (positive_group_counts >= 2) & (negative_group_counts >= 2)
