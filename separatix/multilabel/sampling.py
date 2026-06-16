"""Sampling and validation split helpers for multilabel diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold, ShuffleSplit

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.sampling import BudgetConfig
from separatix.utils.random import make_rng


def _dense_y(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    if sparse.issparse(Y):
        return Y.toarray().astype(np.int8, copy=False)
    return np.asarray(Y)


def _iterative_splitters() -> tuple[Any, Any] | None:
    """Return optional iterative stratification splitters when installed."""
    try:
        from iterstrat.ml_stratifiers import (  # type: ignore[import-not-found]
            MultilabelStratifiedKFold,
            MultilabelStratifiedShuffleSplit,
        )
    except ImportError:
        return None
    return MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit


def _require_iterative(config: ProfilerConfig) -> tuple[Any, Any] | None:
    """Return iterative splitters or raise when the config requires them."""
    splitters = _iterative_splitters()
    if splitters is None and config.multilabel_stratification == "iterative":
        raise ImportError(
            "multilabel_stratification='iterative' requires the optional "
            "'iterative-stratification' dependency. Install separatix with the "
            "'multilabel' extra or choose multilabel_stratification='auto'."
        )
    return splitters


def _heuristic_multilabel_indices(
    Y: Any,
    *,
    n_samples: int,
    random_state: int | None,
    min_per_label: int = 1,
) -> np.ndarray:
    """Return deterministic multilabel subsample indices preserving rare labels."""
    Y_dense = _dense_y(Y)
    n_total = Y_dense.shape[0]
    if n_samples >= n_total:
        return np.arange(n_total, dtype=int)

    rng = make_rng(random_state)
    chosen: set[int] = set()
    label_counts = Y_dense.sum(axis=0)
    for label_idx in np.argsort(label_counts):
        positives = np.flatnonzero(Y_dense[:, label_idx])
        if positives.size == 0:
            continue
        take = min(min_per_label, positives.size, max(0, n_samples - len(chosen)))
        if take == 0:
            break
        available = np.asarray([idx for idx in positives if idx not in chosen])
        if available.size == 0:
            continue
        sampled = rng.choice(available, size=min(take, available.size), replace=False)
        chosen.update(int(idx) for idx in sampled)

    remaining = np.asarray([idx for idx in range(n_total) if idx not in chosen])
    fill = n_samples - len(chosen)
    if fill > 0 and remaining.size > 0:
        sampled = rng.choice(remaining, size=min(fill, remaining.size), replace=False)
        chosen.update(int(idx) for idx in sampled)
    return np.asarray(sorted(chosen), dtype=int)


def multilabel_subsample_indices(
    Y: Any,
    *,
    n_samples: int,
    config: ProfilerConfig,
) -> tuple[np.ndarray, str]:
    """Return multilabel-aware subsample indices and the method used."""
    if n_samples >= Y.shape[0]:
        return np.arange(Y.shape[0], dtype=int), "none"
    splitters = _require_iterative(config)
    if config.multilabel_stratification != "heuristic" and splitters is not None:
        _, splitter_cls = splitters
        try:
            splitter = splitter_cls(
                n_splits=1,
                train_size=n_samples,
                random_state=config.random_state,
            )
            train_idx, _ = next(splitter.split(np.zeros((Y.shape[0], 1)), _dense_y(Y)))
            return np.asarray(sorted(train_idx), dtype=int), "iterative"
        except (TypeError, ValueError):
            if config.multilabel_stratification == "iterative":
                raise

    return (
        _heuristic_multilabel_indices(
            Y,
            n_samples=n_samples,
            random_state=config.random_state,
        ),
        "heuristic",
    )


def cap_multilabel_samples_for_budget(
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    reason: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Optionally cap sample count for expensive multilabel diagnostics."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    if reason == "neighbors":
        max_allowed = budget["max_neighbor_samples"]
    elif reason == "boundary":
        max_allowed = budget["max_boundary_samples"]
    else:
        max_allowed = budget["max_probe_samples"]
    if config.max_samples is not None:
        max_allowed = min(max_allowed, config.max_samples)
    if Y.shape[0] <= max_allowed:
        return (
            X,
            Y,
            {
                "reason": reason,
                "sampled": False,
                "n_original": int(Y.shape[0]),
                "n_used": int(Y.shape[0]),
                "stratification_method": "none",
            },
        )

    indices, method = multilabel_subsample_indices(
        Y,
        n_samples=max_allowed,
        config=config,
    )
    X_used = X[indices] if not sparse.issparse(X) else X[indices, :]
    Y_used = Y[indices] if not sparse.issparse(Y) else Y[indices, :]
    Y_dense = _dense_y(Y)
    used_dense = _dense_y(Y_used)
    original_positive = Y_dense.sum(axis=0)
    used_positive = used_dense.sum(axis=0)
    return (
        X_used,
        Y_used,
        {
            "reason": reason,
            "sampled": True,
            "n_original": int(Y.shape[0]),
            "n_used": int(indices.shape[0]),
            "stratification_method": method,
            "rare_label_preservation_attempted": True,
            "labels_with_no_positive_after_sampling": int(
                np.sum((original_positive > 0) & (used_positive == 0))
            ),
            "labels_with_too_few_positive_after_sampling": int(
                np.sum((used_positive > 0) & (used_positive < 2))
            ),
        },
    )


def choose_multilabel_cv(
    Y: Any,
    *,
    max_folds: int,
    config: ProfilerConfig,
) -> tuple[Any | None, str]:
    """Choose a multilabel validation strategy based on label support."""
    Y_dense = _dense_y(Y)
    positives = Y_dense.sum(axis=0)
    negatives = Y_dense.shape[0] - positives
    supported = np.minimum(positives, negatives)
    supported = supported[supported >= 2]
    if supported.size == 0:
        return None, "resubstitution_low_reliability"
    min_count = int(np.min(supported))
    if min_count >= 5:
        n_splits = min(max_folds, 5)
    elif min_count >= 3:
        n_splits = min(max_folds, 3)
    elif min_count >= 2:
        n_splits = min(max_folds, 2)
    else:
        return None, "resubstitution_low_reliability"

    splitters = _require_iterative(config)
    if config.multilabel_stratification != "heuristic" and splitters is not None:
        splitter_cls, _ = splitters
        return (
            splitter_cls(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.random_state,
            ),
            "iterative",
        )
    return (
        KFold(n_splits=n_splits, shuffle=True, random_state=config.random_state),
        "heuristic",
    )


def choose_multilabel_holdout(
    Y: Any,
    *,
    repeats: int,
    config: ProfilerConfig,
) -> tuple[Any | None, str]:
    """Choose a repeated multilabel holdout splitter."""
    if repeats <= 0:
        return None, "disabled"
    splitters = _require_iterative(config)
    if config.multilabel_stratification != "heuristic" and splitters is not None:
        _, splitter_cls = splitters
        return (
            splitter_cls(
                n_splits=repeats,
                test_size=0.25,
                random_state=config.random_state,
            ),
            "iterative",
        )
    return (
        ShuffleSplit(
            n_splits=repeats,
            test_size=0.25,
            random_state=config.random_state,
        ),
        "heuristic",
    )
