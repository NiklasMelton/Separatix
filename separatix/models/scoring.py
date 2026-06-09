"""Scoring helpers for model probes."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _prepared_estimator(estimator: Any, train_size: int) -> Any:
    """Clone an estimator and shrink kNN neighborhoods to fit the training fold."""
    fitted = clone(estimator)
    if hasattr(fitted, "get_params") and "n_neighbors" in fitted.get_params():
        n_neighbors = int(fitted.get_params()["n_neighbors"])
        fitted.set_params(n_neighbors=max(1, min(n_neighbors, train_size)))
    return fitted


def choose_cv(y: np.ndarray, max_folds: int) -> object | None:
    """Choose a stratified validation strategy based on class counts."""
    min_count = min(Counter(y).values())
    if min_count >= 5:
        n_splits = min(max_folds, 5)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    if min_count >= 3:
        n_splits = min(max_folds, 3)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    if min_count >= 2:
        n_splits = min(max_folds, 2)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return None


def evaluate_estimator(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    *,
    cv: Any | None,
) -> tuple[np.ndarray, str]:
    """Return predictions with a partitioned CV path or a low-reliability fallback."""
    if cv is None:
        fitted = _prepared_estimator(estimator, len(y)).fit(X, y)
        return fitted.predict(X), "resubstitution_low_reliability"

    predictions = np.empty_like(y)
    for train_idx, test_idx in cv.split(X, y):
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            X[train_idx], y[train_idx]
        )
        predictions[test_idx] = fitted.predict(X[test_idx])
    return predictions, "cross_validation"


def summarize_stability(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    *,
    repeats: int,
    random_state: int | None,
) -> dict[str, int | float | None]:
    """Estimate score stability using repeated stratified holdout splits."""
    if repeats <= 0 or len(np.unique(y)) < 2 or np.min(np.bincount(y)) < 2:
        return {
            "stability_repeats": 0,
            "stability_balanced_accuracy_mean": None,
            "stability_balanced_accuracy_std": None,
        }

    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=0.25,
        random_state=random_state,
    )
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            X[train_idx], y[train_idx]
        )
        preds = fitted.predict(X[test_idx])
        scores.append(float(balanced_accuracy_score(y[test_idx], preds)))

    return {
        "stability_repeats": len(scores),
        "stability_balanced_accuracy_mean": float(np.mean(scores)) if scores else None,
        "stability_balanced_accuracy_std": float(np.std(scores)) if scores else None,
    }


def most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_labels: np.ndarray | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the most frequently confused class pairs."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, Any]] = []
    for i, class_a in enumerate(labels):
        for j, class_b in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            if class_labels is not None:
                label_a: Any = class_labels[int(class_a)]
                label_b: Any = class_labels[int(class_b)]
            else:
                label_a = int(class_a)
                label_b = int(class_b)
            pairs.append(
                {
                    "class_a": label_a.item() if hasattr(label_a, "item") else label_a,
                    "class_b": label_b.item() if hasattr(label_b, "item") else label_b,
                    "count": int(matrix[i, j]),
                }
            )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_k]


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize classification predictions."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    summary: dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": [float(x) for x in recalls],
    }
    if len(np.unique(y_true)) > 2:
        summary["most_confused_pairs"] = most_confused_pairs(
            y_true, y_pred, class_labels=class_labels
        )
    return summary
