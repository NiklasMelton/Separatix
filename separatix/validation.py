"""Input validation for separatix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from separatix.grouping import (
    multilabel_group_support,
    singlelabel_group_support,
    summarize_groups,
    validate_groups,
)


@dataclass
class ValidatedInput:
    """Validated representation of the user input."""

    X: Any
    y_encoded: np.ndarray
    labels_original: np.ndarray
    classes_: np.ndarray
    is_sparse: bool
    n_samples: int
    n_features: int
    n_classes: int
    groups: np.ndarray | None
    grouping_summary: dict[str, Any]
    evaluable_y_encoded: np.ndarray
    evaluable_classes_: np.ndarray
    evaluable_groups: np.ndarray | None
    evaluable_mask: np.ndarray


@dataclass
class ValidatedMultilabelInput:
    """Validated representation of a multilabel diagnostic input."""

    X: Any
    Y: Any
    label_names: np.ndarray
    is_sparse_X: bool
    is_sparse_Y: bool
    n_samples: int
    n_features: int
    n_labels: int
    label_counts: np.ndarray
    label_prevalence: np.ndarray
    usable_label_mask: np.ndarray
    all_zero_sample_count: int
    warnings: list[str]
    groups: np.ndarray | None
    grouping_summary: dict[str, Any]


@dataclass
class ValidatedRegressionInput:
    """Validated representation of a regression diagnostic input."""

    X: Any
    Y: np.ndarray
    target_names: np.ndarray
    is_sparse_X: bool
    n_samples: int
    n_features: int
    n_targets: int
    target_variance: np.ndarray
    usable_target_mask: np.ndarray
    warnings: list[str]
    groups: np.ndarray | None
    grouping_summary: dict[str, Any]


def _coerce_pandas(X: Any) -> Any:
    """Convert pandas containers to their underlying arrays when available."""
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return X


def _label_names(y: Any, n_labels: int) -> np.ndarray:
    """Return label names, preserving pandas DataFrame columns when present."""
    columns = getattr(y, "columns", None)
    if columns is not None and len(columns) == n_labels:
        return np.asarray([str(item) for item in columns], dtype=object)
    return np.asarray([f"label_{idx}" for idx in range(n_labels)], dtype=object)


def _target_names(y: Any, n_targets: int) -> np.ndarray:
    """Return regression target names, preserving pandas column names."""
    name = getattr(y, "name", None)
    if n_targets == 1 and name is not None:
        return np.asarray([str(name)], dtype=object)
    columns = getattr(y, "columns", None)
    if columns is not None and len(columns) == n_targets:
        return np.asarray([str(item) for item in columns], dtype=object)
    return np.asarray([f"target_{idx}" for idx in range(n_targets)], dtype=object)


def _dense_is_binary(values: np.ndarray) -> bool:
    """Return whether a dense target array contains only finite binary values."""
    if values.dtype == bool:
        return True
    if not np.issubdtype(values.dtype, np.number):
        return False
    if not np.isfinite(values).all():
        return False
    unique = np.unique(values)
    return bool(np.all(np.isin(unique, [0, 1])))


def _sparse_is_binary(values: sparse.spmatrix) -> bool:
    """Return whether a sparse target matrix contains only finite binary values."""
    if values.data.size == 0:
        return True
    if values.dtype == bool:
        return True
    if not np.issubdtype(values.dtype, np.number):
        return False
    if not np.isfinite(values.data).all():
        return False
    return bool(np.all(np.isin(values.data, [0, 1])))


def is_multilabel_indicator(y: Any, *, allow_single_column: bool = False) -> bool:
    """Return whether y looks like a binary multilabel indicator matrix."""
    if sparse.issparse(y):
        matrix = y
    else:
        try:
            matrix = np.asarray(_coerce_pandas(y))
        except (TypeError, ValueError):
            return False
    if getattr(matrix, "ndim", None) != 2:
        return False
    if matrix.shape[1] < 1:
        return False
    if matrix.shape[1] == 1 and not allow_single_column:
        return False
    if sparse.issparse(matrix):
        return _sparse_is_binary(matrix)
    return _dense_is_binary(np.asarray(matrix))


def _validate_feature_matrix(X: Any) -> tuple[Any, bool]:
    """Validate and normalize the feature matrix."""
    X = _coerce_pandas(X)
    if sparse.issparse(X):
        X = X.tocsr()
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if np.isnan(X.data).any() or not np.isfinite(X.data).all():
            raise ValueError("X contains non-finite values.")
        return X, True

    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if not np.issubdtype(X_array.dtype, np.number):
        try:
            X_array = X_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X could not be converted to floating point.") from exc
    else:
        X_array = X_array.astype(float, copy=False)
    if not np.isfinite(X_array).all():
        raise ValueError("X contains non-finite values.")
    return X_array, False


def _validate_multilabel_target(y: Any) -> tuple[Any, np.ndarray, bool]:
    """Validate and normalize a multilabel target matrix."""
    if sparse.issparse(y):
        Y = y.tocsr()
        if Y.ndim != 2:
            raise ValueError("multilabel y must be two-dimensional.")
        if not _sparse_is_binary(Y):
            raise ValueError("multilabel y must contain only binary values.")
        return Y.astype(np.int8), _label_names(y, Y.shape[1]), True

    Y_raw = _coerce_pandas(y)
    try:
        Y = np.asarray(Y_raw)
    except ValueError as exc:
        raise ValueError("multilabel y must be a rectangular 2D matrix.") from exc
    if Y.ndim != 2:
        raise ValueError("multilabel y must be two-dimensional.")
    if not _dense_is_binary(Y):
        raise ValueError("multilabel y must contain only binary values.")
    return Y.astype(np.int8, copy=False), _label_names(y, Y.shape[1]), False


def validate_inputs(X: Any, y: Any, *, groups: Any = None) -> ValidatedInput:
    """Validate features and labels for classification diagnostics."""
    X = _coerce_pandas(X)
    if not sparse.issparse(X):
        X = np.asarray(X)
    y_array = _coerce_pandas(y)
    y_array = np.asarray(y_array)

    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if sparse.issparse(X):
        X = X.tocsr()
    elif X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if X.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have matching sample counts.")

    target_type = type_of_target(y_array)
    if target_type in {"continuous", "continuous-multioutput", "multilabel-indicator"}:
        raise ValueError(
            "Only categorical single-output classification targets are supported."
        )
    if target_type == "multiclass-multioutput":
        raise ValueError("Multioutput classification is not supported.")

    if sparse.issparse(X):
        if np.isnan(X.data).any() or not np.isfinite(X.data).all():
            raise ValueError("X contains non-finite values.")
    else:
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("X could not be converted to floating point.") from exc
        else:
            X = X.astype(float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

    if np.isnan(y_array).any() if np.issubdtype(y_array.dtype, np.number) else False:
        raise ValueError("y contains NaN values.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_array)
    classes_ = encoder.classes_
    if classes_.shape[0] < 2:
        raise ValueError("At least two classes are required.")
    group_info = validate_groups(groups, n_samples=int(X.shape[0]))
    grouping_summary = summarize_groups(group_info.encoded if group_info else None)

    evaluable_mask = np.ones_like(y_encoded, dtype=bool)
    evaluable_y_encoded = y_encoded
    evaluable_classes = classes_
    evaluable_groups = group_info.encoded if group_info else None
    if group_info is not None:
        support = singlelabel_group_support(y_encoded, group_info.encoded)
        evaluable_mask = np.asarray(support["evaluable_mask"], dtype=bool)
        supported_classes = np.asarray(support["supported_classes"], dtype=int)
        grouping_summary.update(
            {
                "supervised_evaluation_mode": "group_cross_validation",
                "groups_per_class": {
                    str(classes_[cls]): int(count)
                    for cls, count in support["groups_per_class"].items()
                },
                "supported_singlelabel_classes": [
                    str(classes_[cls]) for cls in supported_classes.tolist()
                ],
                "skipped_singlelabel_classes": [
                    str(classes_[cls]) for cls in support["skipped_classes"]
                ],
            }
        )
        if supported_classes.shape[0] < 2:
            raise ValueError(
                "At least two single-label classes must each appear in at least "
                "two distinct groups."
            )
        supported_inverse = {int(cls): idx for idx, cls in enumerate(supported_classes)}
        evaluable_original = y_encoded[evaluable_mask]
        evaluable_y_encoded = np.asarray(
            [supported_inverse[int(label)] for label in evaluable_original],
            dtype=int,
        )
        evaluable_classes = classes_[supported_classes]
        evaluable_groups = group_info.encoded[evaluable_mask]

    return ValidatedInput(
        X=X,
        y_encoded=y_encoded,
        labels_original=y_array,
        classes_=classes_,
        is_sparse=sparse.issparse(X),
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_classes=int(classes_.shape[0]),
        groups=group_info.encoded if group_info else None,
        grouping_summary=grouping_summary,
        evaluable_y_encoded=evaluable_y_encoded,
        evaluable_classes_=evaluable_classes,
        evaluable_groups=evaluable_groups,
        evaluable_mask=evaluable_mask,
    )


def validate_multilabel_inputs(
    X: Any, y: Any, *, groups: Any = None
) -> ValidatedMultilabelInput:
    """Validate features and multilabel indicator targets."""
    X_checked, is_sparse_X = _validate_feature_matrix(X)
    Y_checked, label_names, is_sparse_Y = _validate_multilabel_target(y)
    if X_checked.shape[0] != Y_checked.shape[0]:
        raise ValueError("X and multilabel y must have matching sample counts.")
    if Y_checked.shape[1] < 1:
        raise ValueError("multilabel y must contain at least one label column.")

    label_counts = np.asarray(Y_checked.sum(axis=0)).ravel().astype(int)
    n_samples = int(Y_checked.shape[0])
    label_prevalence = label_counts / max(1, n_samples)
    negatives = n_samples - label_counts
    usable_label_mask = (label_counts >= 2) & (negatives >= 2)
    group_info = validate_groups(groups, n_samples=n_samples)
    grouping_summary = summarize_groups(group_info.encoded if group_info else None)
    if group_info is not None:
        group_mask = multilabel_group_support(Y_checked, group_info.encoded)
        usable_label_mask = usable_label_mask & group_mask
    all_zero_rows = np.asarray(Y_checked.sum(axis=1)).ravel() == 0
    warnings: list[str] = []
    if Y_checked.shape[1] == 1:
        warnings.append(
            "A one-column multilabel indicator was treated as a degenerate "
            "multilabel target; ordinary binary single-label diagnostics may be "
            "more informative."
        )
    if np.any(all_zero_rows):
        warnings.append(
            "Some samples have no positive labels; they are included and audited."
        )
    if not bool(np.any(usable_label_mask)):
        raise ValueError(
            "At least one multilabel column needs at least two positive and two "
            "negative examples, and at least two distinct groups on both the "
            "positive and negative side when groups are provided."
        )
    if group_info is not None:
        grouping_summary.update(
            {
                "supervised_evaluation_mode": "group_cross_validation",
                "supported_multilabel_labels": [
                    str(label_names[idx])
                    for idx in np.flatnonzero(usable_label_mask).tolist()
                ],
                "skipped_multilabel_labels": [
                    str(label_names[idx])
                    for idx in np.flatnonzero(~usable_label_mask).tolist()
                ],
            }
        )

    return ValidatedMultilabelInput(
        X=X_checked,
        Y=Y_checked,
        label_names=label_names,
        is_sparse_X=is_sparse_X,
        is_sparse_Y=is_sparse_Y,
        n_samples=n_samples,
        n_features=int(X_checked.shape[1]),
        n_labels=int(Y_checked.shape[1]),
        label_counts=label_counts,
        label_prevalence=label_prevalence,
        usable_label_mask=usable_label_mask,
        all_zero_sample_count=int(np.sum(all_zero_rows)),
        warnings=warnings,
        groups=group_info.encoded if group_info else None,
        grouping_summary=grouping_summary,
    )


def validate_regression_inputs(
    X: Any, y: Any, *, groups: Any = None
) -> ValidatedRegressionInput:
    """Validate features and continuous targets for regression diagnostics."""
    X_checked, is_sparse_X = _validate_feature_matrix(X)
    if sparse.issparse(y):
        y_raw = y.toarray()
    else:
        y_raw = _coerce_pandas(y)
    try:
        Y = np.asarray(y_raw)
    except ValueError as exc:
        raise ValueError("regression y must be a rectangular numeric array.") from exc
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError("regression y must be one- or two-dimensional.")
    if X_checked.shape[0] != Y.shape[0]:
        raise ValueError("X and regression y must have matching sample counts.")
    if not np.issubdtype(Y.dtype, np.number):
        try:
            Y = Y.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("regression y must be numeric.") from exc
    else:
        Y = Y.astype(float, copy=False)
    if not np.isfinite(Y).all():
        raise ValueError("regression y contains non-finite values.")
    if Y.shape[0] < 3:
        raise ValueError("At least three samples are required for regression.")
    if Y.shape[1] < 1:
        raise ValueError("regression y must contain at least one target.")

    variance = np.var(Y, axis=0)
    usable_target_mask = variance > 1e-12
    if not np.any(usable_target_mask):
        raise ValueError("At least one non-constant regression target is required.")
    warnings: list[str] = []
    if not np.all(usable_target_mask):
        warnings.append(
            "Some regression targets are constant and will be excluded from "
            "probe-family scoring."
        )

    group_info = validate_groups(groups, n_samples=int(X_checked.shape[0]))
    grouping_summary = summarize_groups(group_info.encoded if group_info else None)
    return ValidatedRegressionInput(
        X=X_checked,
        Y=Y,
        target_names=_target_names(y, Y.shape[1]),
        is_sparse_X=is_sparse_X,
        n_samples=int(X_checked.shape[0]),
        n_features=int(X_checked.shape[1]),
        n_targets=int(Y.shape[1]),
        target_variance=variance,
        usable_target_mask=usable_target_mask,
        warnings=warnings,
        groups=group_info.encoded if group_info else None,
        grouping_summary=grouping_summary,
    )
