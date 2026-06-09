"""Sampling utilities for separatix."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.utils.random import make_rng


class BudgetConfig(TypedDict):
    """Per-budget runtime limits."""

    max_probe_samples: int
    max_neighbor_samples: int
    max_boundary_samples: int
    cv_folds: int
    bootstrap_repeats: int
    run_kernel_probe: bool
    run_persistent_topology: bool | str


def stratified_subsample_indices(
    y: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None = None,
    min_per_class: int = 1,
) -> np.ndarray:
    """Return stratified subsample indices preserving class proportions."""
    y = np.asarray(y)
    if n_samples >= y.shape[0]:
        return np.arange(y.shape[0], dtype=int)

    rng = make_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = classes.shape[0]
    chosen: list[int] = []

    if n_samples < n_classes:
        picked_classes = rng.choice(classes, size=n_samples, replace=False)
        for cls in picked_classes:
            cls_idx = np.flatnonzero(y == cls)
            chosen.append(int(rng.choice(cls_idx)))
        return np.array(sorted(chosen), dtype=int)

    base = np.maximum(
        min_per_class,
        np.floor(n_samples * (counts / counts.sum())).astype(int),
    )
    base = np.minimum(base, counts)
    total = int(base.sum())

    while total > n_samples:
        for i in np.argsort(-base):
            if total == n_samples:
                break
            if base[i] > min_per_class:
                base[i] -= 1
                total -= 1
    while total < n_samples:
        for i in np.argsort(-(counts - base)):
            if total == n_samples:
                break
            if base[i] < counts[i]:
                base[i] += 1
                total += 1

    for cls, take in zip(classes, base):  # noqa: B905
        cls_idx = np.flatnonzero(y == cls)
        sampled = rng.choice(cls_idx, size=int(take), replace=False)
        chosen.extend(sampled.tolist())
    return np.array(sorted(chosen), dtype=int)


def cap_samples_for_budget(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    reason: str,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Optionally cap sample count for expensive diagnostics."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    if reason == "neighbors":
        max_allowed = budget["max_neighbor_samples"]
    elif reason == "boundary":
        max_allowed = budget["max_boundary_samples"]
    else:
        max_allowed = budget["max_probe_samples"]
    if config.max_samples is not None:
        max_allowed = min(max_allowed, config.max_samples)
    if y.shape[0] <= max_allowed:
        return (
            X,
            y,
            {
                "reason": reason,
                "sampled": False,
                "n_original": int(y.shape[0]),
                "n_used": int(y.shape[0]),
            },
        )

    indices = stratified_subsample_indices(
        y,
        n_samples=max_allowed,
        random_state=config.random_state,
    )
    X_used = X[indices] if not sparse.issparse(X) else X[indices, :]
    return (
        X_used,
        y[indices],
        {
            "reason": reason,
            "sampled": True,
            "n_original": int(y.shape[0]),
            "n_used": int(indices.shape[0]),
        },
    )
