import numpy as np
import pytest
from scipy import sparse

from separatix import diagnose
from separatix.multilabel.validation import validate_multilabel_inputs
from separatix.validation import validate_inputs


def test_single_label_validator_still_rejects_multilabel() -> None:
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        validate_inputs(np.ones((4, 2)), y)


def test_validate_multilabel_dense_binary() -> None:
    X = np.ones((6, 2))
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 1], [1, 0], [0, 0]])
    validated = validate_multilabel_inputs(X, Y)
    assert validated.n_labels == 2
    assert validated.usable_label_mask.tolist() == [True, True]


def test_validate_multilabel_boolean_and_sparse_targets() -> None:
    X = sparse.csr_matrix(np.eye(6))
    Y = sparse.csr_matrix(
        np.array(
            [
                [True, False],
                [True, True],
                [False, True],
                [False, True],
                [True, False],
                [False, False],
            ]
        )
    )
    validated = validate_multilabel_inputs(X, Y)
    assert validated.is_sparse_X
    assert validated.is_sparse_Y


def test_validate_multilabel_rejects_non_binary() -> None:
    X = np.ones((4, 2))
    Y = np.array([[0, 1], [1, 2], [0, 1], [1, 0]])
    with pytest.raises(ValueError):
        validate_multilabel_inputs(X, Y)


def test_validate_multilabel_rejects_mismatched_sample_count() -> None:
    with pytest.raises(ValueError):
        validate_multilabel_inputs(np.ones((4, 2)), np.ones((3, 2), dtype=int))


def test_validate_multilabel_warns_for_all_zero_rows() -> None:
    X = np.ones((5, 2))
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
    validated = validate_multilabel_inputs(X, Y)
    assert validated.all_zero_sample_count == 2
    assert any("no positive labels" in warning for warning in validated.warnings)


def test_auto_rejects_one_column_multilabel_indicator() -> None:
    X = np.ones((6, 2))
    Y = np.array([[0], [0], [1], [1], [0], [1]])
    with pytest.raises(ValueError):
        diagnose(X, Y, return_report=True, budget="fast")


def test_explicit_multilabel_allows_one_column_with_warning() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.array([[0], [0], [1], [1], [0], [1]])
    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    assert report.class_summary["target_type"] == "multilabel"
    assert any("one-column multilabel" in warning for warning in report.warnings)
