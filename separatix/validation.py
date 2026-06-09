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


def _coerce_pandas(X: Any) -> Any:
    """Convert pandas containers to their underlying arrays when available."""
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return X


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
