"""Input validation for separatix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


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


def validate_inputs(X: Any, y: Any) -> ValidatedInput:
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

    return ValidatedInput(
        X=X,
        y_encoded=y_encoded,
        labels_original=y_array,
        classes_=classes_,
        is_sparse=sparse.issparse(X),
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_classes=int(classes_.shape[0]),
    )


def validate_multilabel_inputs(X: Any, y: Any) -> ValidatedMultilabelInput:
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
            "negative examples."
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
    )
