from importlib.util import find_spec

import numpy as np
import pytest

from separatix.config import ProfilerConfig
from separatix.sampling import (
    cap_multilabel_samples_for_budget,
    choose_multilabel_cv,
    choose_multilabel_holdout,
)


def test_multilabel_heuristic_sampling_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Y = np.zeros((200, 4), dtype=int)
    Y[:100, 0] = 1
    Y[80:160, 1] = 1
    Y[150:180, 2] = 1
    Y[190:, 3] = 1
    config = ProfilerConfig(
        budget="fast",
        max_samples=40,
        random_state=7,
        multilabel_stratification="heuristic",
    )
    _, Y_a, info_a = cap_multilabel_samples_for_budget(
        X, Y, config=config, reason="probe"
    )
    _, Y_b, info_b = cap_multilabel_samples_for_budget(
        X, Y, config=config, reason="probe"
    )
    assert np.array_equal(Y_a, Y_b)
    assert info_a == info_b
    assert info_a["stratification_method"] == "heuristic"


def test_multilabel_cv_gracefully_falls_back_to_heuristic() -> None:
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]])
    config = ProfilerConfig(multilabel_stratification="heuristic", random_state=0)
    cv, method = choose_multilabel_cv(Y, max_folds=5, config=config)
    assert cv is not None
    assert method == "heuristic"


def test_single_column_multilabel_uses_binary_stratified_splitters() -> None:
    Y = np.array([[0], [0], [1], [1], [0], [1]])
    config = ProfilerConfig(multilabel_stratification="auto", random_state=0)
    cv, cv_method = choose_multilabel_cv(Y, max_folds=5, config=config)
    holdout, holdout_method = choose_multilabel_holdout(Y, repeats=3, config=config)

    assert cv is not None
    assert holdout is not None
    assert cv_method == "binary_stratified"
    assert holdout_method == "binary_stratified"


def test_iterative_mode_requires_optional_dependency_when_absent() -> None:
    if find_spec("iterstrat") is not None:
        pytest.skip("iterative-stratification is installed in this environment")
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]])
    config = ProfilerConfig(multilabel_stratification="iterative", random_state=0)
    with pytest.raises(ImportError):
        choose_multilabel_cv(Y, max_folds=5, config=config)
