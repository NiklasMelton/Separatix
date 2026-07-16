import numpy as np
import pytest
from scipy import sparse

from separatix import ProfilerConfig
from separatix.validation import validate_inputs


def test_validate_dense_numeric_labels() -> None:
    X = np.ones((10, 4))
    y = np.array([0, 1] * 5)
    validated = validate_inputs(X, y)
    assert validated.n_samples == 10
    assert validated.n_classes == 2


def test_validate_sparse_converts_to_csr() -> None:
    X = sparse.csc_matrix(np.eye(6))
    y = np.array(["a", "b", "a", "b", "a", "b"])
    validated = validate_inputs(X, y)
    assert validated.is_sparse
    assert sparse.isspmatrix_csr(validated.X)


def test_validate_rejects_one_class() -> None:
    with pytest.raises(ValueError):
        validate_inputs(np.ones((4, 2)), np.array([1, 1, 1, 1]))


def test_validate_rejects_multilabel() -> None:
    y = np.array([[1, 0], [0, 1], [1, 0]])
    with pytest.raises(ValueError):
        validate_inputs(np.ones((3, 2)), y)


def test_validate_rejects_mismatched_length() -> None:
    with pytest.raises(ValueError):
        validate_inputs(np.ones((4, 2)), np.array([0, 1, 0]))


def test_validate_rejects_nan_features() -> None:
    X = np.array([[1.0, np.nan], [2.0, 1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        validate_inputs(X, y)


def test_nonintegral_numeric_labels_are_categorical() -> None:
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0.25, 1.75] * 6)
    validated = validate_inputs(X, y)
    assert validated.classes_.tolist() == [0.25, 1.75]


@pytest.mark.parametrize(
    ("X", "y", "match"),
    [
        (np.empty((4, 0)), np.asarray([0, 1, 0, 1]), "feature column"),
        (
            np.asarray([[1 + 2j], [2 + 0j]]),
            np.asarray([0, 1]),
            "real-valued features",
        ),
        (
            np.asarray([[1.0], [2.0]]),
            np.asarray([0 + 1j, 1 + 0j]),
            "real-valued or string",
        ),
    ],
)
def test_rejects_zero_feature_and_complex_inputs(X, y, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_inputs(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": 0},
        {"random_state": -1},
        {"random_state": 2**32},
        {"random_state": 1.5},
        {"max_samples": 1.5},
    ],
)
def test_config_rejects_invalid_integer_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ProfilerConfig(**kwargs)
