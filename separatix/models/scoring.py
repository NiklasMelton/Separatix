"""Scoring helpers for model probes."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def choose_cv(y: np.ndarray, max_folds: int) -> object:
    """Choose a stratified validation strategy based on class counts."""
    min_count = min(Counter(y).values())
    if min_count >= 5:
        n_splits = min(max_folds, 5)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    if min_count >= 3:
        n_splits = min(max_folds, 3)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return StratifiedShuffleSplit(n_splits=3, test_size=0.33, random_state=0)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Summarize classification predictions."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": [float(x) for x in recalls],
    }
