import numpy as np

from separatix.config import ProfilerConfig
from separatix.sampling import (
    _constrained_group_indices,
    cap_regression_samples_for_budget,
    cap_samples_for_budget,
    grouped_stratified_subsample_indices,
    stratified_subsample_indices,
)


def test_stratified_sampling_preserves_classes() -> None:
    y = np.array([0] * 40 + [1] * 30 + [2] * 20)
    idx = stratified_subsample_indices(y, n_samples=30, random_state=0)
    sampled = y[idx]
    assert set(sampled.tolist()) == {0, 1, 2}


def test_stratified_sampling_is_deterministic() -> None:
    y = np.array([0] * 50 + [1] * 50)
    idx_a = stratified_subsample_indices(y, n_samples=20, random_state=3)
    idx_b = stratified_subsample_indices(y, n_samples=20, random_state=3)
    assert np.array_equal(idx_a, idx_b)


def test_sampling_smaller_than_required_support_is_unavailable() -> None:
    y = np.array([0, 1, 2, 3, 4, 5])
    idx = stratified_subsample_indices(y, n_samples=3, random_state=0)
    assert idx.shape[0] == 0


def test_cap_samples_for_budget() -> None:
    X = np.arange(600).reshape(200, 3)
    y = np.array([0] * 100 + [1] * 100)
    config = ProfilerConfig(budget="fast", max_samples=50, random_state=0)
    X_used, y_used, info = cap_samples_for_budget(X, y, config=config, reason="probe")
    assert X_used.shape[0] == 50
    assert y_used.shape[0] == 50
    assert info["sampled"] is True


def test_group_sampling_never_exceeds_hard_cap() -> None:
    y = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 4)
    groups = np.repeat(np.arange(4), 6)
    indices = grouped_stratified_subsample_indices(
        y,
        groups,
        n_samples=18,
        random_state=0,
    )
    assert indices.size <= 18
    assert np.unique(groups[indices]).size == 3
    assert all(np.sum(groups[indices] == group) == 6 for group in groups[indices])
    for cls in (0, 1):
        assert np.unique(groups[indices][y[indices] == cls]).size >= 2


def test_oversized_groups_make_hard_cap_infeasible() -> None:
    y = np.asarray([0] * 20 + [1] * 20)
    groups = np.asarray([0] * 20 + [1] * 20)
    indices = grouped_stratified_subsample_indices(
        y,
        groups,
        n_samples=10,
        random_state=0,
    )
    assert indices.size == 0


def test_grouped_regression_sampling_preserves_groups() -> None:
    X = np.arange(120, dtype=float).reshape(40, 3)
    Y = np.arange(40, dtype=float).reshape(-1, 1)
    groups = np.repeat(np.arange(8), 5)
    config = ProfilerConfig(max_samples=12, random_state=0)
    X_used, Y_used, info = cap_regression_samples_for_budget(
        X,
        Y,
        config=config,
        reason="probe",
        groups=groups,
    )
    assert X_used.shape[0] == Y_used.shape[0] <= 12
    assert X_used.shape[0] % 5 == 0
    assert info["support_preserved"] is True


def test_group_sampler_finds_maximal_feasible_whole_group_subset() -> None:
    group_rows = [np.arange(0, 6), np.arange(6, 11), np.arange(11, 15)]
    indices = _constrained_group_indices(
        group_rows,
        np.ones((3, 1), dtype=bool),
        required_support=2,
        n_samples=10,
        random_state=0,
    )
    assert indices.size == 10
    assert set(indices) == set(group_rows[0]) | set(group_rows[2])
