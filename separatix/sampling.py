"""Sampling and split-selection utilities for separatix."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, TypedDict, cast

import numpy as np
from scipy import sparse
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

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


class PrecomputedSplitter:
    """Simple splitter backed by explicit row-index folds."""

    def __init__(self, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._splits = splits

    def split(
        self, X: Any, y: Any | None = None, groups: Any | None = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield precomputed train and test row indices."""
        yield from self._splits


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


def _group_rows(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return unique groups and row indices for each group."""
    unique = np.unique(groups)
    return unique, [np.flatnonzero(groups == group_id) for group_id in unique]


def _greedy_group_indices(
    group_rows: list[np.ndarray],
    group_scores: list[float],
    *,
    n_samples: int,
) -> np.ndarray:
    """Choose whole groups without splitting them across rows."""
    selected: list[int] = []
    used_rows = 0
    remaining = set(range(len(group_rows)))
    while remaining and used_rows < n_samples:
        fitting = [
            idx
            for idx in remaining
            if used_rows == 0 or used_rows + group_rows[idx].shape[0] <= n_samples
        ]
        if not fitting:
            break
        best = max(
            fitting,
            key=lambda idx: (group_scores[idx], -group_rows[idx].shape[0], -idx),
        )
        selected.append(best)
        used_rows += int(group_rows[best].shape[0])
        remaining.remove(best)
    if not selected and group_rows:
        selected = [
            min(
                range(len(group_rows)),
                key=lambda idx: (group_rows[idx].shape[0], -group_scores[idx], idx),
            )
        ]
    if not selected:
        return np.array([], dtype=int)
    return np.sort(np.concatenate([group_rows[idx] for idx in selected]).astype(int))


def grouped_stratified_subsample_indices(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Return group-preserving row indices for single-label diagnostics."""
    if n_samples >= y.shape[0]:
        return np.arange(y.shape[0], dtype=int)
    unique_groups, group_rows = _group_rows(groups)
    if unique_groups.shape[0] == y.shape[0]:
        return stratified_subsample_indices(
            y,
            n_samples=n_samples,
            random_state=random_state,
        )
    cumulative_class_counts = {int(cls): 0 for cls in np.unique(y)}
    group_scores: list[float] = []
    for rows in group_rows:
        group_y = y[rows]
        score = 0.0
        group_classes, group_counts = np.unique(group_y, return_counts=True)
        for cls, count in zip(group_classes, group_counts):  # noqa: B905
            score += float(count) / (1.0 + cumulative_class_counts[int(cls)])
        group_scores.append(score)
        for cls, count in zip(group_classes, group_counts):  # noqa: B905
            cumulative_class_counts[int(cls)] += int(count)
    return _greedy_group_indices(group_rows, group_scores, n_samples=n_samples)


def cap_samples_for_budget(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    reason: str,
    groups: np.ndarray | None = None,
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
                "indices": list(range(int(y.shape[0]))),
                "group_sampling": bool(groups is not None),
            },
        )

    if groups is not None:
        indices = grouped_stratified_subsample_indices(
            y,
            groups,
            n_samples=max_allowed,
            random_state=config.random_state,
        )
    else:
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
            "indices": indices.tolist(),
            "group_sampling": bool(groups is not None),
        },
    )


def random_subsample_indices(
    n_total: int,
    *,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Return deterministic random row indices without label stratification."""
    if n_samples >= n_total:
        return np.arange(n_total, dtype=int)
    rng = make_rng(random_state)
    return np.sort(rng.choice(np.arange(n_total), size=n_samples, replace=False))


def cap_regression_samples_for_budget(
    X: Any,
    Y: np.ndarray,
    *,
    config: ProfilerConfig,
    reason: str,
    groups: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Optionally cap regression sample count for expensive diagnostics."""
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
                "indices": list(range(int(Y.shape[0]))),
                "group_sampling": bool(groups is not None),
            },
        )
    indices = random_subsample_indices(
        Y.shape[0],
        n_samples=max_allowed,
        random_state=config.random_state,
    )
    X_used = X[indices] if not sparse.issparse(X) else X[indices, :]
    return (
        X_used,
        Y[indices],
        {
            "reason": reason,
            "sampled": True,
            "n_original": int(Y.shape[0]),
            "n_used": int(indices.shape[0]),
            "indices": indices.tolist(),
            "group_sampling": bool(groups is not None),
        },
    )


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    if sparse.issparse(Y):
        dense = Y.toarray()
    else:
        dense = np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _iterative_splitters() -> tuple[Any, Any] | None:
    """Return optional iterative multilabel splitters when installed."""
    try:
        from iterstrat.ml_stratifiers import (  # type: ignore[import-not-found]
            MultilabelStratifiedKFold,
            MultilabelStratifiedShuffleSplit,
        )
    except ImportError:
        return None
    return MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit


def _require_iterative_multilabel_splitters(
    config: ProfilerConfig,
) -> tuple[Any, Any] | None:
    """Return iterative multilabel splitters or raise when required."""
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
    Y_dense = _dense_multilabel_matrix(Y)
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


def _is_single_column_multilabel(Y: Any) -> bool:
    """Return whether a multilabel target is the degenerate one-column case."""
    return _dense_multilabel_matrix(Y).shape[1] == 1


def multilabel_subsample_indices(
    Y: Any,
    *,
    n_samples: int,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return multilabel-aware subsample indices and the method used."""
    if n_samples >= Y.shape[0]:
        return np.arange(Y.shape[0], dtype=int), "none"
    Y_dense = _dense_multilabel_matrix(Y)
    if groups is not None:
        _, group_rows = _group_rows(groups)
        running_support = np.zeros(Y_dense.shape[1], dtype=float)
        group_scores: list[float] = []
        for rows in group_rows:
            group_sum = Y_dense[rows].sum(axis=0).astype(float)
            score = float(np.sum(group_sum / np.maximum(1.0, running_support + 1.0)))
            group_scores.append(score)
            running_support += group_sum
        return (
            _greedy_group_indices(group_rows, group_scores, n_samples=n_samples),
            "group_heuristic",
        )
    if _is_single_column_multilabel(Y_dense):
        indices = stratified_subsample_indices(
            Y_dense[:, 0],
            n_samples=n_samples,
            random_state=config.random_state,
        )
        return indices, "binary_stratified"
    splitters = _require_iterative_multilabel_splitters(config)
    if config.multilabel_stratification != "heuristic" and splitters is not None:
        _, splitter_cls = splitters
        try:
            splitter = splitter_cls(
                n_splits=1,
                train_size=n_samples,
                random_state=config.random_state,
            )
            train_idx, _ = next(splitter.split(np.zeros((Y.shape[0], 1)), Y_dense))
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
    groups: np.ndarray | None = None,
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
                "indices": list(range(int(Y.shape[0]))),
                "group_sampling": bool(groups is not None),
            },
        )

    indices, method = multilabel_subsample_indices(
        Y,
        n_samples=max_allowed,
        config=config,
        groups=groups,
    )
    X_used = X[indices] if not sparse.issparse(X) else X[indices, :]
    Y_used = Y[indices] if not sparse.issparse(Y) else Y[indices, :]
    original_positive = _dense_multilabel_matrix(Y).sum(axis=0)
    used_positive = _dense_multilabel_matrix(Y_used).sum(axis=0)
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
            "indices": indices.tolist(),
            "group_sampling": bool(groups is not None),
            "labels_with_no_positive_after_sampling": int(
                np.sum((original_positive > 0) & (used_positive == 0))
            ),
            "labels_with_too_few_positive_after_sampling": int(
                np.sum((used_positive > 0) & (used_positive < 2))
            ),
        },
    )


def grouped_singlelabel_cv(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_folds: int,
    random_state: int | None,
) -> tuple[Any | None, str]:
    """Choose the best available group-safe CV splitter for single-label data."""
    group_support = min(np.unique(groups[y == cls]).shape[0] for cls in np.unique(y))
    all_rows = np.arange(y.shape[0], dtype=int)
    for n_splits in range(min(max_folds, int(group_support)), 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(np.zeros((y.shape[0], 1)), y, groups))
        if all(
            np.intersect1d(groups[train_idx], groups[test_idx]).size == 0
            and np.unique(y[train_idx]).shape[0] == np.unique(y).shape[0]
            for train_idx, test_idx in splits
        ):
            return splitter, "stratified_group"
        fallback = GroupKFold(n_splits=n_splits)
        splits = list(fallback.split(np.zeros((y.shape[0], 1)), y, groups))
        if all(
            np.intersect1d(groups[train_idx], groups[test_idx]).size == 0
            and np.unique(y[train_idx]).shape[0] == np.unique(y).shape[0]
            for train_idx, test_idx in splits
        ):
            return fallback, "group_kfold"
        unique_groups, group_rows = _group_rows(groups)
        group_class_counts = np.asarray(
            [
                np.bincount(y[rows], minlength=np.unique(y).shape[0]).astype(float)
                for rows in group_rows
            ]
        )
        order = np.argsort(-group_class_counts.sum(axis=1))
        fold_group_indices: list[list[int]] = [[] for _ in range(n_splits)]
        fold_sizes = np.zeros(n_splits, dtype=int)
        fold_class_totals = np.zeros(
            (n_splits, group_class_counts.shape[1]),
            dtype=float,
        )
        for idx in order.tolist():
            best_fold = min(
                range(n_splits),
                key=lambda fold_idx: (
                    float(
                        np.var(fold_class_totals[fold_idx] + group_class_counts[idx])
                    ),
                    int(fold_sizes[fold_idx]),
                    fold_idx,
                ),
            )
            fold_group_indices[best_fold].append(idx)
            fold_sizes[best_fold] += int(group_rows[idx].shape[0])
            fold_class_totals[best_fold] += group_class_counts[idx]
        heuristic_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in fold_group_indices:
            if not fold:
                heuristic_splits = []
                break
            test_idx = np.sort(
                np.concatenate([group_rows[idx] for idx in fold]).astype(int)
            )
            train_idx = np.setdiff1d(all_rows, test_idx)
            if np.unique(y[train_idx]).shape[0] != np.unique(y).shape[0]:
                heuristic_splits = []
                break
            heuristic_splits.append((train_idx, test_idx))
        if heuristic_splits:
            return PrecomputedSplitter(heuristic_splits), "group_heuristic"
    return None, "group_split_unavailable"


def _assign_groups_to_multilabel_folds(
    Y_dense: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Greedily assign whole groups to multilabel folds."""
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < n_splits:
        return None
    group_rows = [np.flatnonzero(groups == group_id) for group_id in unique_groups]
    group_label_counts = np.asarray([Y_dense[rows].sum(axis=0) for rows in group_rows])
    order = np.argsort(-group_label_counts.sum(axis=1))
    fold_group_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_label_totals = np.zeros((n_splits, Y_dense.shape[1]), dtype=float)
    for idx in order.tolist():
        best_fold = min(
            range(n_splits),
            key=lambda fold_idx: (
                float(
                    np.sum((fold_label_totals[fold_idx] + group_label_counts[idx]) ** 2)
                ),
                int(fold_sizes[fold_idx]),
                fold_idx,
            ),
        )
        fold_group_indices[best_fold].append(idx)
        fold_sizes[best_fold] += int(group_rows[idx].shape[0])
        fold_label_totals[best_fold] += group_label_counts[idx]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_rows = np.arange(Y_dense.shape[0], dtype=int)
    for fold in fold_group_indices:
        if not fold:
            return None
        test_idx = np.sort(
            np.concatenate([group_rows[idx] for idx in fold]).astype(int)
        )
        train_idx = np.setdiff1d(all_rows, test_idx)
        if test_idx.size == 0 or train_idx.size == 0:
            return None
        splits.append((train_idx, test_idx))
    return splits


def choose_multilabel_cv(
    Y: Any,
    *,
    max_folds: int,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
) -> tuple[Any | None, str]:
    """Choose a multilabel validation strategy based on label support."""
    Y_dense = _dense_multilabel_matrix(Y)
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

    if groups is not None:
        splits = _assign_groups_to_multilabel_folds(
            Y_dense,
            groups,
            n_splits=n_splits,
        )
        if splits is None:
            return None, "group_split_unavailable"
        return PrecomputedSplitter(splits), "group_heuristic"

    if _is_single_column_multilabel(Y_dense):
        return (
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.random_state,
            ),
            "binary_stratified",
        )

    splitters = _require_iterative_multilabel_splitters(config)
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
    groups: np.ndarray | None = None,
) -> tuple[Any | None, str]:
    """Choose a repeated multilabel holdout splitter."""
    if repeats <= 0:
        return None, "disabled"
    if groups is not None:
        return (
            GroupShuffleSplit(
                n_splits=repeats,
                test_size=0.25,
                random_state=config.random_state,
            ),
            "group_shuffle",
        )
    if _is_single_column_multilabel(Y):
        return (
            StratifiedShuffleSplit(
                n_splits=repeats,
                test_size=0.25,
                random_state=config.random_state,
            ),
            "binary_stratified",
        )
    splitters = _require_iterative_multilabel_splitters(config)
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
