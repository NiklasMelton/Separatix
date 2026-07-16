"""Sampling and split-selection utilities for separatix."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypedDict, cast

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.utils.random import make_rng


@contextmanager
def _isolated_legacy_numpy_rng() -> Generator[None, None, None]:
    """Restore NumPy's legacy global RNG after optional third-party splitters."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


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

    def get_n_splits(
        self, X: Any | None = None, y: Any | None = None, groups: Any | None = None
    ) -> int:
        """Return the number of stored splits."""
        return len(self._splits)


def stratified_subsample_indices(
    y: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None = None,
    min_per_class: int = 2,
) -> np.ndarray:
    """Return stratified subsample indices preserving class proportions."""
    y = np.asarray(y)
    if n_samples >= y.shape[0]:
        return np.arange(y.shape[0], dtype=int)

    rng = make_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    chosen: list[int] = []

    required = np.minimum(counts, min_per_class)
    if n_samples < int(required.sum()):
        return np.array([], dtype=int)

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
            if used_rows + group_rows[idx].shape[0] <= n_samples
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
    if not selected:
        return np.array([], dtype=int)
    return np.sort(np.concatenate([group_rows[idx] for idx in selected]).astype(int))


def _constrained_group_indices(
    group_rows: list[np.ndarray],
    group_support: np.ndarray,
    *,
    required_support: int,
    n_samples: int,
    random_state: int | None,
) -> np.ndarray:
    """Select whole groups under a hard cap while preserving support states."""
    if not group_rows or n_samples <= 0:
        return np.array([], dtype=int)
    sizes = np.asarray([rows.shape[0] for rows in group_rows], dtype=int)
    eligible = np.flatnonzero(sizes <= n_samples)
    if eligible.size == 0:
        return np.array([], dtype=int)
    support = np.asarray(group_support, dtype=bool)
    if support.ndim == 1:
        support = support.reshape(-1, 1)
    if np.any(np.sum(support[eligible], axis=0) < required_support):
        return np.array([], dtype=int)

    rng = make_rng(random_state)
    eligible_support = support[eligible]
    eligible_sizes = sizes[eligible]
    state_frequency = np.maximum(1, eligible_support.sum(axis=0))
    rarity = np.sum(eligible_support / state_frequency, axis=1).astype(float)
    rarity_bonus = 0.2 * rarity / max(float(np.sum(rarity)), 1.0)
    tie_order = rng.permutation(eligible.size).astype(float) + 1.0
    tie_bonus = 0.05 * tie_order / float(np.sum(tie_order))
    objective = -(eligible_sizes.astype(float) + rarity_bonus + tie_bonus)
    constraint_matrix = sparse.vstack(
        [
            sparse.csr_matrix(eligible_sizes.reshape(1, -1)),
            sparse.csr_matrix(eligible_support.T.astype(float)),
        ],
        format="csr",
    )
    lower = np.concatenate(
        [np.asarray([-np.inf]), np.full(support.shape[1], required_support)]
    )
    upper = np.concatenate(
        [np.asarray([float(n_samples)]), np.full(support.shape[1], np.inf)]
    )
    try:
        solution = milp(
            c=objective,
            integrality=np.ones(eligible.size, dtype=int),
            bounds=Bounds(np.zeros(eligible.size), np.ones(eligible.size)),
            constraints=LinearConstraint(constraint_matrix, lower, upper),
            options={"presolve": True},
        )
    except (TypeError, ValueError, RuntimeError):
        solution = None
    if solution is not None:
        if solution.x is None:
            return np.array([], dtype=int)
        selected_groups = eligible[np.flatnonzero(solution.x >= 0.5)]
        if selected_groups.size:
            selected_rows = np.sort(
                np.concatenate([group_rows[idx] for idx in selected_groups]).astype(int)
            )
            selected_support = support[selected_groups].sum(axis=0)
            if selected_rows.size <= n_samples and np.all(
                selected_support >= required_support
            ):
                return selected_rows

    # Keep a deterministic fallback for unexpected optimizer/runtime failures.
    best: tuple[int, tuple[int, ...]] | None = None
    attempts = max(16, min(128, len(group_rows) * 4))
    state_frequency = np.maximum(1, support[eligible].sum(axis=0))
    for attempt in range(attempts):
        selected: list[int] = []
        used = 0
        state_counts = np.zeros(support.shape[1], dtype=int)
        remaining = set(int(idx) for idx in eligible.tolist())
        jitter = rng.random(len(group_rows)) if attempt else np.zeros(len(group_rows))
        while np.any(state_counts < required_support):
            fitting = [idx for idx in remaining if used + sizes[idx] <= n_samples]
            if not fitting:
                break
            unmet = state_counts < required_support
            useful = [idx for idx in fitting if np.any(support[idx] & unmet)]
            if not useful:
                break
            choice = max(
                useful,
                key=lambda idx: (
                    float(np.sum((support[idx] & unmet) / state_frequency)),
                    int(np.sum(support[idx] & unmet)),
                    -int(sizes[idx]),
                    float(jitter[idx]),
                    -idx,
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
            used += int(sizes[choice])
            state_counts += support[choice].astype(int)
        if np.any(state_counts < required_support):
            continue
        # Fill toward the cap without ever compromising the selected support.
        for idx in sorted(
            remaining,
            key=lambda item: (-int(np.sum(support[item])), sizes[item], item),
        ):
            if used + sizes[idx] <= n_samples:
                selected.append(idx)
                used += int(sizes[idx])
        signature = tuple(sorted(selected))
        candidate = (used, signature)
        if (
            best is None
            or candidate[0] > best[0]
            or (candidate[0] == best[0] and candidate[1] < best[1])
        ):
            best = candidate
    if best is None:
        return np.array([], dtype=int)
    return np.sort(np.concatenate([group_rows[idx] for idx in best[1]]).astype(int))


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
    classes = np.unique(y)
    support = np.asarray(
        [[np.any(y[rows] == cls) for cls in classes] for rows in group_rows],
        dtype=bool,
    )
    return _constrained_group_indices(
        group_rows,
        support,
        required_support=2,
        n_samples=n_samples,
        random_state=random_state,
    )


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
                "support_preserved": True,
                "skip_reason": None,
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
    support_preserved = bool(
        indices.size > 0
        and np.array_equal(np.unique(y[indices]), np.unique(y))
        and all(
            np.sum(y[indices] == cls) >= min(2, np.sum(y == cls))
            for cls in np.unique(y)
        )
    )
    if support_preserved and groups is not None:
        support_preserved = all(
            np.unique(groups[indices][y[indices] == cls]).shape[0] >= 2
            for cls in np.unique(y)
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
            "support_preserved": support_preserved,
            "skip_reason": None
            if support_preserved
            else "no support-preserving sample fits the configured hard cap",
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


def grouped_regression_subsample_indices(
    groups: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Return whole-group regression sample indices under a hard row cap."""
    _, group_rows = _group_rows(groups)
    return _constrained_group_indices(
        group_rows,
        np.ones((len(group_rows), 1), dtype=bool),
        required_support=2,
        n_samples=n_samples,
        random_state=random_state,
    )


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
                "support_preserved": True,
                "skip_reason": None,
            },
        )
    if groups is not None:
        indices = grouped_regression_subsample_indices(
            groups,
            n_samples=max_allowed,
            random_state=config.random_state,
        )
    else:
        indices = (
            random_subsample_indices(
                Y.shape[0],
                n_samples=max_allowed,
                random_state=config.random_state,
            )
            if max_allowed >= 4
            else np.array([], dtype=int)
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
            "support_preserved": bool(
                indices.size >= 4
                and (groups is None or np.unique(groups[indices]).shape[0] >= 2)
            ),
            "skip_reason": None
            if indices.size >= 4
            else "no evaluable regression sample fits the configured hard cap",
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
    min_per_label: int = 2,
) -> np.ndarray:
    """Return deterministic multilabel subsample indices preserving rare labels."""
    n_total = int(Y.shape[0])
    if n_samples >= n_total:
        return np.arange(n_total, dtype=int)

    rng = make_rng(random_state)
    if sparse.issparse(Y):
        Y_csc = Y.tocsc()
        sparse_chosen: set[int] = set()
        all_rows = np.arange(n_total, dtype=int)
        positive_counts = np.asarray(Y_csc.sum(axis=0)).ravel().astype(int)
        label_order = np.argsort(np.minimum(positive_counts, n_total - positive_counts))
        for label_idx in label_order.tolist():
            positives = Y_csc[:, label_idx].indices.astype(int, copy=False)
            positive_mask = np.zeros(n_total, dtype=bool)
            positive_mask[positives] = True
            for side_mask in (positive_mask, ~positive_mask):
                already = sum(bool(side_mask[idx]) for idx in sparse_chosen)
                need = max(0, min_per_label - already)
                if need == 0:
                    continue
                available = all_rows[side_mask]
                available = np.asarray(
                    [idx for idx in available.tolist() if idx not in sparse_chosen],
                    dtype=int,
                )
                if available.size < need or len(sparse_chosen) + need > n_samples:
                    return np.array([], dtype=int)
                sampled = rng.choice(available, size=need, replace=False)
                sparse_chosen.update(int(idx) for idx in sampled.tolist())
        remaining = np.asarray(
            [idx for idx in range(n_total) if idx not in sparse_chosen], dtype=int
        )
        fill = n_samples - len(sparse_chosen)
        if fill > 0:
            sparse_chosen.update(
                int(idx)
                for idx in rng.choice(remaining, size=fill, replace=False).tolist()
            )
        return np.asarray(sorted(sparse_chosen), dtype=int)

    Y_dense = _dense_multilabel_matrix(Y)
    chosen: set[int] = set()
    states = np.concatenate([Y_dense > 0, Y_dense == 0], axis=1)
    state_counts = states.sum(axis=0)
    required = np.minimum(state_counts, min_per_label)
    if n_samples < int(np.max(required)) or np.any(state_counts < min_per_label):
        return np.array([], dtype=int)
    selected_support = np.zeros(states.shape[1], dtype=int)
    while np.any(selected_support < required):
        remaining = np.asarray([idx for idx in range(n_total) if idx not in chosen])
        if remaining.size == 0 or len(chosen) >= n_samples:
            return np.array([], dtype=int)
        unmet = selected_support < required
        rarity = 1.0 / np.maximum(1, state_counts)
        scores = np.sum(states[remaining][:, unmet] * rarity[unmet], axis=1)
        best_score = float(np.max(scores))
        candidates = remaining[np.flatnonzero(np.isclose(scores, best_score))]
        selected = int(rng.choice(candidates))
        chosen.add(selected)
        selected_support += states[selected].astype(int)

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
    Y_dense = None if sparse.issparse(Y) else _dense_multilabel_matrix(Y)
    if groups is not None:
        unique_groups, group_rows = _group_rows(groups)
        if unique_groups.shape[0] == Y.shape[0]:
            indices, method = multilabel_subsample_indices(
                Y,
                n_samples=n_samples,
                config=config,
                groups=None,
            )
            return indices, f"group_rows_{method}"
        support_mb = (
            len(group_rows) * 2 * Y.shape[1] * np.dtype(float).itemsize / 1024**2
        )
        if support_mb > config.max_dense_mb:
            return np.array([], dtype=int), "group_memory_unavailable"
        if sparse.issparse(Y):
            Y_csc = Y.tocsc()
            support = np.zeros((len(group_rows), 2 * Y.shape[1]), dtype=bool)
            for label_idx in range(Y.shape[1]):
                positives = np.zeros(Y.shape[0], dtype=bool)
                positives[Y_csc[:, label_idx].indices] = True
                for group_idx, rows in enumerate(group_rows):
                    support[group_idx, label_idx] = bool(np.any(positives[rows]))
                    support[group_idx, Y.shape[1] + label_idx] = bool(
                        np.any(~positives[rows])
                    )
        else:
            assert Y_dense is not None
            support = np.asarray(
                [
                    np.concatenate(
                        [
                            np.any(Y_dense[rows] > 0, axis=0),
                            np.any(Y_dense[rows] == 0, axis=0),
                        ]
                    )
                    for rows in group_rows
                ],
                dtype=bool,
            )
        return (
            _constrained_group_indices(
                group_rows,
                support,
                required_support=2,
                n_samples=n_samples,
                random_state=config.random_state,
            ),
            "group_heuristic",
        )
    if Y.shape[1] == 1:
        single_column = (
            np.asarray(Y[:, 0].toarray()).ravel()
            if sparse.issparse(Y)
            else np.asarray(Y_dense)[:, 0]
        )
        indices = stratified_subsample_indices(
            single_column,
            n_samples=n_samples,
            random_state=config.random_state,
        )
        return indices, "binary_stratified"
    if sparse.issparse(Y):
        return (
            _heuristic_multilabel_indices(
                Y,
                n_samples=n_samples,
                random_state=config.random_state,
            ),
            "heuristic",
        )
    assert Y_dense is not None
    splitters = _require_iterative_multilabel_splitters(config)
    if config.multilabel_stratification != "heuristic" and splitters is not None:
        _, splitter_cls = splitters
        try:
            splitter = splitter_cls(
                n_splits=1,
                train_size=n_samples,
                random_state=config.random_state,
            )
            with _isolated_legacy_numpy_rng():
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
                "support_preserved": True,
                "skip_reason": None,
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
    original_positive = np.asarray(Y.sum(axis=0)).ravel()
    used_positive = np.asarray(Y_used.sum(axis=0)).ravel()
    used_negative = indices.shape[0] - used_positive
    support_preserved = bool(
        indices.size > 0
        and np.all(used_positive >= np.minimum(original_positive, 2))
        and np.all(used_negative >= np.minimum(Y.shape[0] - original_positive, 2))
    )
    if support_preserved and groups is not None:
        used_groups = groups[indices]
        Y_used_dense = _dense_multilabel_matrix(Y_used)
        support_preserved = all(
            np.unique(used_groups[Y_used_dense[:, label] > 0]).shape[0] >= 2
            and np.unique(used_groups[Y_used_dense[:, label] == 0]).shape[0] >= 2
            for label in range(Y_used_dense.shape[1])
        )
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
            "support_preserved": support_preserved,
            "skip_reason": None
            if support_preserved
            else "no multilabel support-preserving sample fits the configured hard cap",
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
            and np.unique(y[test_idx]).shape[0] == np.unique(y).shape[0]
            for train_idx, test_idx in splits
        ):
            return splitter, "stratified_group"
        fallback = GroupKFold(n_splits=n_splits)
        splits = list(fallback.split(np.zeros((y.shape[0], 1)), y, groups))
        if all(
            np.intersect1d(groups[train_idx], groups[test_idx]).size == 0
            and np.unique(y[train_idx]).shape[0] == np.unique(y).shape[0]
            and np.unique(y[test_idx]).shape[0] == np.unique(y).shape[0]
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
            if (
                np.unique(y[train_idx]).shape[0] != np.unique(y).shape[0]
                or np.unique(y[test_idx]).shape[0] != np.unique(y).shape[0]
            ):
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
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Assign whole groups rare-state-first across multilabel folds."""
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < n_splits:
        return None
    group_rows = [np.flatnonzero(groups == group_id) for group_id in unique_groups]
    positive_counts = np.asarray([Y_dense[rows].sum(axis=0) for rows in group_rows])
    negative_counts = np.asarray(
        [(Y_dense[rows] == 0).sum(axis=0) for rows in group_rows]
    )
    group_state_counts = np.concatenate([positive_counts, negative_counts], axis=1)
    total_states = np.maximum(1.0, group_state_counts.sum(axis=0).astype(float))
    rng = make_rng(random_state)
    rarity_scores = np.asarray(
        np.sum(group_state_counts / total_states, axis=1), dtype=float
    )
    n_groups = int(unique_groups.shape[0])
    jitter = np.asarray(rng.random(n_groups), dtype=float) * 1e-9
    order = np.asarray(
        sorted(
            range(n_groups),
            key=lambda idx: (-float(rarity_scores[idx]), -float(jitter[idx]), idx),
        ),
        dtype=int,
    )
    fold_group_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_state_totals = np.zeros((n_splits, group_state_counts.shape[1]), dtype=float)
    target_states = total_states / n_splits
    for idx in order.tolist():
        best_fold = min(
            range(n_splits),
            key=lambda fold_idx: (
                float(
                    np.sum(
                        (
                            (
                                fold_state_totals
                                + np.eye(n_splits, dtype=float)[:, [fold_idx]]
                                * group_state_counts[idx]
                            )
                            - target_states[None, :]
                        )
                        ** 2
                        / total_states[None, :]
                    )
                ),
                int(fold_sizes[fold_idx]),
                fold_idx,
            ),
        )
        fold_group_indices[best_fold].append(idx)
        fold_sizes[best_fold] += int(group_rows[idx].shape[0])
        fold_state_totals[best_fold] += group_state_counts[idx]
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


def _valid_multilabel_splits(
    Y_dense: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    groups: np.ndarray | None = None,
) -> bool:
    """Return whether every multilabel fold preserves both sides of each label."""
    for train_idx, test_idx in splits:
        if train_idx.size == 0 or test_idx.size == 0:
            return False
        if (
            groups is not None
            and np.intersect1d(groups[train_idx], groups[test_idx]).size
        ):
            return False
        for indices in (train_idx, test_idx):
            positives = Y_dense[indices].sum(axis=0)
            if np.any(positives == 0) or np.any(positives == indices.size):
                return False
    return True


def _heuristic_multilabel_splits(
    Y_dense: np.ndarray,
    *,
    n_splits: int,
    random_state: int | None,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Build deterministic support-aware multilabel folds."""
    row_groups = groups if groups is not None else np.arange(Y_dense.shape[0])
    for attempt in range(32):
        seed = None if random_state is None else random_state + attempt
        splits = _assign_groups_to_multilabel_folds(
            Y_dense,
            row_groups,
            n_splits=n_splits,
            random_state=seed,
        )
        if splits is not None and _valid_multilabel_splits(
            Y_dense, splits, groups=groups
        ):
            return splits
    return None


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
        return None, "multilabel_split_unavailable"
    min_count = int(np.min(supported))
    if min_count >= 5:
        n_splits = min(max_folds, 5)
    elif min_count >= 3:
        n_splits = min(max_folds, 3)
    elif min_count >= 2:
        n_splits = min(max_folds, 2)
    else:
        return None, "multilabel_split_unavailable"

    splitters = _require_iterative_multilabel_splitters(config)
    for folds in range(n_splits, 1, -1):
        if groups is None and _is_single_column_multilabel(Y_dense):
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=config.random_state,
            )
            splits = list(
                splitter.split(np.zeros((Y_dense.shape[0], 1)), Y_dense[:, 0])
            )
            if _valid_multilabel_splits(Y_dense, splits):
                return PrecomputedSplitter(splits), "binary_stratified"
        if (
            groups is None
            and config.multilabel_stratification != "heuristic"
            and splitters is not None
        ):
            splitter_cls, _ = splitters
            splitter = splitter_cls(
                n_splits=folds,
                shuffle=True,
                random_state=config.random_state,
            )
            try:
                with _isolated_legacy_numpy_rng():
                    iterative_splits = list(
                        splitter.split(np.zeros((Y_dense.shape[0], 1)), Y_dense)
                    )
            except ValueError:
                iterative_splits = []
            if _valid_multilabel_splits(Y_dense, iterative_splits):
                return PrecomputedSplitter(iterative_splits), "iterative"
        heuristic_splits = _heuristic_multilabel_splits(
            Y_dense,
            n_splits=folds,
            random_state=config.random_state,
            groups=groups,
        )
        if heuristic_splits is not None:
            method = "group_heuristic" if groups is not None else "heuristic"
            return PrecomputedSplitter(heuristic_splits), method
    unavailable = (
        "group_split_unavailable"
        if groups is not None
        else "multilabel_split_unavailable"
    )
    return None, unavailable


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
    Y_dense = _dense_multilabel_matrix(Y)
    if groups is None and _is_single_column_multilabel(Y_dense):
        splitter = StratifiedShuffleSplit(
            n_splits=max(repeats * 4, repeats),
            test_size=0.25,
            random_state=config.random_state,
        )
        binary_valid = [
            split
            for split in splitter.split(np.zeros((Y_dense.shape[0], 1)), Y_dense[:, 0])
            if _valid_multilabel_splits(Y_dense, [split])
        ][:repeats]
        if len(binary_valid) == repeats:
            return PrecomputedSplitter(binary_valid), "binary_stratified"
    splitters = _require_iterative_multilabel_splitters(config)
    if (
        groups is None
        and config.multilabel_stratification != "heuristic"
        and splitters is not None
    ):
        _, splitter_cls = splitters
        splitter = splitter_cls(
            n_splits=max(repeats * 4, repeats),
            test_size=0.25,
            random_state=config.random_state,
        )
        try:
            with _isolated_legacy_numpy_rng():
                iterative_valid = [
                    split
                    for split in splitter.split(
                        np.zeros((Y_dense.shape[0], 1)), Y_dense
                    )
                    if _valid_multilabel_splits(Y_dense, [split])
                ][:repeats]
        except ValueError:
            iterative_valid = []
        if len(iterative_valid) == repeats:
            return PrecomputedSplitter(iterative_valid), "iterative"

    heuristic_valid: list[tuple[np.ndarray, np.ndarray]] = []
    for attempt in range(max(32, repeats * 8)):
        seed = None if config.random_state is None else config.random_state + attempt
        for folds in (4, 3, 2):
            splits = _heuristic_multilabel_splits(
                Y_dense,
                n_splits=folds,
                random_state=seed,
                groups=groups,
            )
            if splits:
                heuristic_valid.append(splits[attempt % len(splits)])
                break
        if len(heuristic_valid) >= repeats:
            method = "group_heuristic" if groups is not None else "heuristic"
            return PrecomputedSplitter(heuristic_valid[:repeats]), method
    unavailable = (
        "group_split_unavailable"
        if groups is not None
        else "multilabel_split_unavailable"
    )
    return None, unavailable
