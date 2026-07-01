"""Scoring helpers for model probes."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from separatix.config import ProfilerConfig
from separatix.sampling import choose_multilabel_holdout, grouped_singlelabel_cv


def _prepared_estimator(estimator: Any, train_size: int) -> Any:
    """Clone an estimator and shrink kNN neighborhoods to fit the training fold."""
    fitted = clone(estimator)
    if hasattr(fitted, "get_params") and "n_neighbors" in fitted.get_params():
        n_neighbors = int(fitted.get_params()["n_neighbors"])
        fitted.set_params(n_neighbors=max(1, min(n_neighbors, train_size)))
    return fitted


def choose_cv(
    y: np.ndarray,
    max_folds: int,
    *,
    groups: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[object | None, str]:
    """Choose a stratified validation strategy based on class counts."""
    if groups is not None:
        return grouped_singlelabel_cv(
            y,
            groups,
            max_folds=max_folds,
            random_state=random_state,
        )
    min_count = min(Counter(y).values())
    if min_count >= 5:
        n_splits = min(max_folds, 5)
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=0
        ), "stratified"
    if min_count >= 3:
        n_splits = min(max_folds, 3)
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=0
        ), "stratified"
    if min_count >= 2:
        n_splits = min(max_folds, 2)
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=0
        ), "stratified"
    return None, "resubstitution_low_reliability"


def evaluate_estimator(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    *,
    cv: Any | None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return predictions with a partitioned CV path or a low-reliability fallback."""
    if cv is None:
        fitted = _prepared_estimator(estimator, len(y)).fit(X, y)
        return fitted.predict(X), "resubstitution_low_reliability"

    predictions = np.empty_like(y)
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for train_idx, test_idx in split_iter:
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            X[train_idx], y[train_idx]
        )
        predictions[test_idx] = fitted.predict(X[test_idx])
    return (
        predictions,
        "group_cross_validation" if groups is not None else "cross_validation",
    )


def summarize_stability(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    *,
    repeats: int,
    random_state: int | None,
    groups: np.ndarray | None = None,
) -> dict[str, int | float | None]:
    """Estimate score stability using repeated stratified holdout splits."""
    if repeats <= 0 or len(np.unique(y)) < 2 or np.min(np.bincount(y)) < 2:
        return {
            "stability_repeats": 0,
            "stability_balanced_accuracy_mean": None,
            "stability_balanced_accuracy_std": None,
        }

    splitter = (
        GroupShuffleSplit(
            n_splits=repeats,
            test_size=0.25,
            random_state=random_state,
        )
        if groups is not None
        else StratifiedShuffleSplit(
            n_splits=repeats,
            test_size=0.25,
            random_state=random_state,
        )
    )
    scores: list[float] = []
    split_iter = (
        splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    )
    for train_idx, test_idx in split_iter:
        if np.unique(y[train_idx]).shape[0] < np.unique(y).shape[0]:
            continue
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


class MultilabelPriorDummy(BaseEstimator, ClassifierMixin):
    """Per-label prevalence baseline for multilabel diagnostics."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the dummy baseline."""
        self.threshold = threshold

    def fit(self, X: Any, y: Any) -> MultilabelPriorDummy:
        """Fit per-label prevalence values."""
        Y = _dense_multilabel_matrix(y)
        self.prevalence_ = np.mean(Y, axis=0)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels whose prevalence meets the configured threshold."""
        positives = (self.prevalence_ >= self.threshold).astype(np.int8)
        return np.tile(positives, (X.shape[0], 1))


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _slice_rows(X: Any, indices: np.ndarray) -> Any:
    """Slice dense or sparse rows."""
    return X[indices, :] if sparse.issparse(X) else X[indices]


def evaluate_multilabel_estimator(
    estimator: Any,
    X: Any,
    Y: Any,
    *,
    cv: Any | None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return multilabel predictions with CV or a low-reliability fallback."""
    Y_dense = _dense_multilabel_matrix(Y)
    if cv is None:
        fitted = _prepared_estimator(estimator, Y_dense.shape[0]).fit(X, Y_dense)
        return (
            _dense_multilabel_matrix(fitted.predict(X)),
            "resubstitution_low_reliability",
        )

    predictions = np.zeros_like(Y_dense, dtype=np.int8)
    split_iter = (
        cv.split(X, Y_dense, groups) if groups is not None else cv.split(X, Y_dense)
    )
    for train_idx, test_idx in split_iter:
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            _slice_rows(X, train_idx), Y_dense[train_idx]
        )
        predictions[test_idx] = _dense_multilabel_matrix(
            fitted.predict(_slice_rows(X, test_idx))
        )
    return (
        predictions,
        "group_cross_validation" if groups is not None else "cross_validation",
    )


def _per_label_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    *,
    label_names: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return compact per-label metric details and summary statistics."""
    details: list[dict[str, Any]] = []
    f1_values: list[float] = []
    rare_f1_values: list[float] = []
    for idx in range(Y_true.shape[1]):
        true_col = Y_true[:, idx]
        pred_col = Y_pred[:, idx]
        precision = float(precision_score(true_col, pred_col, zero_division=0))
        recall = float(recall_score(true_col, pred_col, zero_division=0))
        f1 = (
            0.0
            if precision + recall == 0.0
            else (2.0 * precision * recall / (precision + recall))
        )
        if np.unique(true_col).shape[0] < 2:
            balanced = None
        else:
            balanced = float(balanced_accuracy_score(true_col, pred_col))
        positive_count = int(np.sum(true_col))
        f1_values.append(f1)
        if positive_count <= max(5, int(0.05 * Y_true.shape[0])):
            rare_f1_values.append(f1)
        details.append(
            {
                "label": str(label_names[idx]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced,
                "positive_count": positive_count,
                "prevalence": float(positive_count / max(1, Y_true.shape[0])),
            }
        )

    f1_array = np.asarray(f1_values, dtype=float)
    worst = sorted(details, key=lambda item: item["f1"])[:5]
    best = sorted(details, key=lambda item: item["f1"], reverse=True)[:5]
    summary: dict[str, Any] = {
        "per_label_f1_mean": float(np.mean(f1_array)) if f1_array.size else None,
        "per_label_f1_median": float(np.median(f1_array)) if f1_array.size else None,
        "per_label_f1_p10": float(np.percentile(f1_array, 10))
        if f1_array.size
        else None,
        "rare_label_f1_mean": float(np.mean(rare_f1_values))
        if rare_f1_values
        else None,
    }
    return details, {**summary, "worst_labels": worst, "best_labels": best}


def _micro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Return multilabel micro F1 for binary indicator matrices."""
    true_bool = Y_true.astype(bool)
    pred_bool = Y_pred.astype(bool)
    tp = float(np.logical_and(true_bool, pred_bool).sum())
    fp = float(np.logical_and(~true_bool, pred_bool).sum())
    fn = float(np.logical_and(true_bool, ~pred_bool).sum())
    denom = (2.0 * tp) + fp + fn
    return 0.0 if denom == 0.0 else float((2.0 * tp) / denom)


def _sample_f1_and_jaccard(
    Y_true: np.ndarray, Y_pred: np.ndarray
) -> tuple[float, float]:
    """Return sample-averaged F1 and Jaccard with empty unions scored as 0."""
    true_bool = Y_true.astype(bool)
    pred_bool = Y_pred.astype(bool)
    intersections = np.logical_and(true_bool, pred_bool).sum(axis=1).astype(float)
    true_counts = true_bool.sum(axis=1).astype(float)
    pred_counts = pred_bool.sum(axis=1).astype(float)
    unions = np.logical_or(true_bool, pred_bool).sum(axis=1).astype(float)
    f1_denoms = true_counts + pred_counts
    sample_f1 = np.divide(
        2.0 * intersections,
        f1_denoms,
        out=np.zeros_like(intersections, dtype=float),
        where=f1_denoms > 0,
    )
    sample_jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    return float(np.mean(sample_f1)), float(np.mean(sample_jaccard))


def summarize_multilabel_predictions(
    Y_true: Any,
    Y_pred: Any,
    *,
    label_names: np.ndarray,
) -> dict[str, Any]:
    """Summarize multilabel predictions with separate primary metrics."""
    true_dense = _dense_multilabel_matrix(Y_true)
    pred_dense = _dense_multilabel_matrix(Y_pred)
    per_label, per_label_summary = _per_label_metrics(
        true_dense,
        pred_dense,
        label_names=label_names,
    )
    samples_f1, sample_jaccard = _sample_f1_and_jaccard(true_dense, pred_dense)
    balanced_values = [
        item["balanced_accuracy"]
        for item in per_label
        if item["balanced_accuracy"] is not None
    ]
    macro_f1_values = [float(item["f1"]) for item in per_label]
    return {
        "micro_f1": _micro_f1(true_dense, pred_dense),
        "macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
        "samples_f1": samples_f1,
        "sample_jaccard": sample_jaccard,
        "hamming_loss": float(np.mean(true_dense != pred_dense)),
        "subset_accuracy": float(accuracy_score(true_dense, pred_dense)),
        "label_balanced_accuracy_macro": float(np.mean(balanced_values))
        if balanced_values
        else None,
        "primary_metrics": ["micro_f1", "macro_f1", "sample_jaccard"],
        "per_label_metrics": per_label,
        "per_label_summary": per_label_summary,
    }


def summarize_multilabel_stability(
    estimator: Any,
    X: Any,
    Y: Any,
    *,
    repeats: int,
    random_state: int | None,
    config: ProfilerConfig,
    label_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate multilabel metric stability with repeated holdout splits."""
    if repeats <= 0:
        return {
            "stability_repeats": 0,
            "stability_method": "disabled",
            "stability_micro_f1_std": None,
            "stability_macro_f1_std": None,
            "stability_sample_jaccard_std": None,
        }

    holdout, method = choose_multilabel_holdout(
        Y,
        repeats=repeats,
        config=config,
        groups=groups,
    )
    if holdout is None:
        return {
            "stability_repeats": 0,
            "stability_method": method,
            "stability_micro_f1_std": None,
            "stability_macro_f1_std": None,
            "stability_sample_jaccard_std": None,
        }

    Y_dense = _dense_multilabel_matrix(Y)
    scores: dict[str, list[float]] = {
        "micro_f1": [],
        "macro_f1": [],
        "sample_jaccard": [],
    }
    split_iter = (
        holdout.split(X, Y_dense, groups)
        if groups is not None
        else holdout.split(X, Y_dense)
    )
    for train_idx, test_idx in split_iter:
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            _slice_rows(X, train_idx), Y_dense[train_idx]
        )
        preds = fitted.predict(_slice_rows(X, test_idx))
        summary = summarize_multilabel_predictions(
            Y_dense[test_idx],
            preds,
            label_names=label_names,
        )
        for metric in scores:
            scores[metric].append(float(summary[metric]))

    result: dict[str, Any] = {
        "stability_repeats": len(scores["micro_f1"]),
        "stability_method": method,
    }
    for metric, values in scores.items():
        result[f"stability_{metric}_mean"] = float(np.mean(values)) if values else None
        result[f"stability_{metric}_std"] = float(np.std(values)) if values else None
    return result


def choose_regression_cv(
    y: np.ndarray,
    max_folds: int,
    *,
    groups: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[object | None, str]:
    """Choose a non-stratified validation strategy for regression probes."""
    n_samples = int(np.asarray(y).shape[0])
    if groups is not None:
        unique_groups = np.unique(groups)
        if unique_groups.shape[0] >= 2:
            return (
                GroupKFold(n_splits=min(max_folds, unique_groups.shape[0])),
                "group_k_fold",
            )
        return None, "group_split_unavailable"
    if n_samples >= 10:
        return (
            KFold(n_splits=min(max_folds, 5), shuffle=True, random_state=random_state),
            "k_fold",
        )
    if n_samples >= 4:
        return (
            KFold(n_splits=2, shuffle=True, random_state=random_state),
            "k_fold_low_sample",
        )
    return None, "resubstitution_low_reliability"


def evaluate_regression_estimator(
    estimator: Any,
    X: Any,
    Y: np.ndarray,
    *,
    cv: Any | None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return regression predictions with CV or a low-reliability fallback."""
    if cv is None:
        fitted = _prepared_estimator(estimator, Y.shape[0]).fit(X, Y)
        return np.asarray(fitted.predict(X)).reshape(Y.shape), (
            "resubstitution_low_reliability"
        )

    predictions = np.zeros_like(Y, dtype=float)
    split_iter = cv.split(X, Y, groups) if groups is not None else cv.split(X, Y)
    for train_idx, test_idx in split_iter:
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            _slice_rows(X, train_idx), Y[train_idx]
        )
        fold_pred = np.asarray(fitted.predict(_slice_rows(X, test_idx)))
        predictions[test_idx] = fold_pred.reshape(Y[test_idx].shape)
    return (
        predictions,
        "group_cross_validation" if groups is not None else "cross_validation",
    )


def summarize_regression_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    *,
    target_names: np.ndarray,
) -> dict[str, Any]:
    """Summarize single- or multi-target regression predictions."""
    true = np.asarray(Y_true, dtype=float)
    pred = np.asarray(Y_pred, dtype=float).reshape(true.shape)
    variances = np.var(true, axis=0)
    usable = variances > 1e-12
    per_target: list[dict[str, Any]] = []
    per_target_r2: list[float] = []
    per_target_nrmse: list[float] = []
    for idx in range(true.shape[1]):
        if not usable[idx] or true.shape[0] < 2:
            r2_value = None
            nrmse = None
        else:
            r2_value = float(r2_score(true[:, idx], pred[:, idx]))
            rmse = float(np.sqrt(mean_squared_error(true[:, idx], pred[:, idx])))
            nrmse = float(rmse / max(np.sqrt(variances[idx]), 1e-12))
            per_target_r2.append(r2_value)
            per_target_nrmse.append(nrmse)
        per_target.append(
            {
                "target": str(target_names[idx]),
                "variance": float(variances[idx]),
                "r2": r2_value,
                "normalized_rmse": nrmse,
            }
        )
    if np.any(usable) and true.shape[0] >= 2:
        r2_variance_weighted = float(
            r2_score(true[:, usable], pred[:, usable], multioutput="variance_weighted")
        )
        r2_uniform = float(
            r2_score(true[:, usable], pred[:, usable], multioutput="uniform_average")
        )
    else:
        r2_variance_weighted = 0.0
        r2_uniform = 0.0
    if not np.isfinite(r2_variance_weighted):
        r2_variance_weighted = 0.0
    if not np.isfinite(r2_uniform):
        r2_uniform = 0.0
    return {
        "r2_variance_weighted": r2_variance_weighted,
        "r2_uniform_average": r2_uniform,
        "normalized_rmse_mean": float(np.mean(per_target_nrmse))
        if per_target_nrmse
        else None,
        "primary_metrics": ["r2_variance_weighted", "r2_uniform_average"],
        "per_target_metrics": per_target,
        "per_target_r2_summary": {
            "min": float(np.min(per_target_r2)) if per_target_r2 else None,
            "median": float(np.median(per_target_r2)) if per_target_r2 else None,
            "max": float(np.max(per_target_r2)) if per_target_r2 else None,
        },
        "per_target_normalized_rmse_summary": {
            "min": float(np.min(per_target_nrmse)) if per_target_nrmse else None,
            "median": float(np.median(per_target_nrmse))
            if per_target_nrmse
            else None,
            "max": float(np.max(per_target_nrmse)) if per_target_nrmse else None,
        },
    }


class TargetMeanDummyRegressor(BaseEstimator, RegressorMixin):
    """Mean baseline that preserves two-dimensional regression predictions."""

    def fit(self, X: Any, y: Any) -> TargetMeanDummyRegressor:
        """Fit per-target means."""
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.mean_ = np.mean(Y, axis=0)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict fitted target means for every row."""
        return np.tile(self.mean_, (X.shape[0], 1))


def summarize_regression_stability(
    estimator: Any,
    X: Any,
    Y: np.ndarray,
    *,
    repeats: int,
    random_state: int | None,
    target_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate regression metric stability with repeated holdout splits."""
    if repeats <= 0 or Y.shape[0] < 4:
        return {
            "stability_repeats": 0,
            "stability_method": "disabled",
            "stability_r2_variance_weighted_std": None,
            "stability_r2_uniform_average_std": None,
        }
    if groups is not None:
        unique_groups = np.unique(groups)
        if unique_groups.shape[0] < 2:
            return {
                "stability_repeats": 0,
                "stability_method": "group_split_unavailable",
                "stability_r2_variance_weighted_std": None,
                "stability_r2_uniform_average_std": None,
            }
        splitter = GroupShuffleSplit(
            n_splits=repeats,
            test_size=0.25,
            random_state=random_state,
        )
        split_iter = splitter.split(X, Y, groups)
        method = "group_shuffle_split"
    else:
        splitter = ShuffleSplit(
            n_splits=repeats,
            test_size=0.25,
            random_state=random_state,
        )
        split_iter = splitter.split(X, Y)
        method = "shuffle_split"

    weighted_scores: list[float] = []
    uniform_scores: list[float] = []
    for train_idx, test_idx in split_iter:
        fitted = _prepared_estimator(estimator, len(train_idx)).fit(
            _slice_rows(X, train_idx), Y[train_idx]
        )
        preds = np.asarray(fitted.predict(_slice_rows(X, test_idx))).reshape(
            Y[test_idx].shape
        )
        summary = summarize_regression_predictions(
            Y[test_idx],
            preds,
            target_names=target_names,
        )
        weighted_scores.append(float(summary["r2_variance_weighted"]))
        uniform_scores.append(float(summary["r2_uniform_average"]))
    return {
        "stability_repeats": len(weighted_scores),
        "stability_method": method,
        "stability_r2_variance_weighted_mean": float(np.mean(weighted_scores))
        if weighted_scores
        else None,
        "stability_r2_variance_weighted_std": float(np.std(weighted_scores))
        if weighted_scores
        else None,
        "stability_r2_uniform_average_mean": float(np.mean(uniform_scores))
        if uniform_scores
        else None,
        "stability_r2_uniform_average_std": float(np.std(uniform_scores))
        if uniform_scores
        else None,
    }
