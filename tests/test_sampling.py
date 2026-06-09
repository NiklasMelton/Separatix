import numpy as np

from separatix.config import ProfilerConfig
from separatix.sampling import cap_samples_for_budget, stratified_subsample_indices


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


def test_sampling_smaller_than_class_count() -> None:
    y = np.array([0, 1, 2, 3, 4, 5])
    idx = stratified_subsample_indices(y, n_samples=3, random_state=0)
    assert idx.shape[0] == 3


def test_cap_samples_for_budget() -> None:
    X = np.arange(600).reshape(200, 3)
    y = np.array([0] * 100 + [1] * 100)
    config = ProfilerConfig(budget="fast", max_samples=50, random_state=0)
    X_used, y_used, info = cap_samples_for_budget(X, y, config=config, reason="probe")
    assert X_used.shape[0] == 50
    assert y_used.shape[0] == 50
    assert info["sampled"] is True
