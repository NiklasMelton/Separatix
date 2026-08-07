"""Tests for effective probe training-size metadata."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_classification

from separatix import diagnose
from separatix.models.scoring import materialize_evaluation_plan

_SUMMARY_FIELDS = {
    "status",
    "basis",
    "min",
    "median",
    "mean",
    "max",
    "mean_fraction_of_evaluation_cohort",
}


class _UnevenSplitter:
    """Small deterministic splitter with deliberately uneven train folds."""

    def split(self, X, y=None, groups=None):
        del y, groups
        n_samples = len(X)
        assert n_samples == 8
        folds = (
            (np.arange(1, 8), np.asarray([0])),
            (np.asarray([0, 3, 4, 5, 6, 7]), np.asarray([1, 2])),
            (np.asarray([0, 1, 2]), np.arange(3, 8)),
        )
        yield from folds

    def get_n_splits(self, X=None, y=None, groups=None):
        del X, y, groups
        return 3


def _assert_summary_schema(summary: dict[str, object], *, basis: str | None) -> None:
    """Assert the stable envelope of effective train-size metadata."""
    assert set(summary) == _SUMMARY_FIELDS
    assert summary["status"] == ("available" if basis is not None else "unavailable")
    assert summary["basis"] == basis


def test_effective_train_size_summary_handles_uneven_held_out_folds() -> None:
    """Fold statistics use the actual train rows, including uneven folds."""
    X = np.zeros((8, 2), dtype=float)
    y = np.arange(8)
    _, evaluation = materialize_evaluation_plan(
        _UnevenSplitter(),
        X,
        y,
        method="custom_uneven",
        row_indices=np.arange(8),
    )

    summary = evaluation["effective_train_size_summary"]
    _assert_summary_schema(summary, basis="held_out_folds")
    assert summary["min"] == 3
    assert summary["median"] == 6.0
    assert summary["mean"] == pytest.approx(16.0 / 3.0)
    assert summary["max"] == 7
    assert summary["mean_fraction_of_evaluation_cohort"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("target_mode", ["singlelabel", "multilabel", "regression"])
def test_normal_target_modes_report_held_out_effective_train_sizes(
    target_mode: str,
) -> None:
    """Classification, multilabel, and regression reports expose the same schema."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(61, 5))
    if target_mode == "singlelabel":
        y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)
    elif target_mode == "multilabel":
        y = np.column_stack(
            [
                X[:, 0] > 0,
                X[:, 1] + X[:, 2] > 0,
            ]
        ).astype(int)
    else:
        y = X[:, 0] - 0.4 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])

    report = diagnose(
        X,
        y,
        target_mode=target_mode,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=13,
    )
    evaluation = report.metrics["probe_evaluation"]
    summary = evaluation["effective_train_size_summary"]
    _assert_summary_schema(summary, basis="held_out_folds")

    train_sizes = np.asarray(evaluation["train_sizes"], dtype=int)
    assert train_sizes.size >= 2
    assert summary["min"] == int(np.min(train_sizes))
    assert summary["median"] == pytest.approx(float(np.median(train_sizes)))
    assert summary["mean"] == pytest.approx(float(np.mean(train_sizes)))
    assert summary["max"] == int(np.max(train_sizes))
    assert summary["mean_fraction_of_evaluation_cohort"] == pytest.approx(
        float(np.mean(train_sizes) / evaluation["n_samples"])
    )


def test_small_ungrouped_regression_reports_resubstitution_train_size() -> None:
    """The low-sample regression fallback describes full-cohort fitting."""
    X = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    y = np.asarray([0.0, 1.0, 2.5])
    report = diagnose(
        X,
        y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        topology="off",
        random_state=0,
    )
    evaluation = report.metrics["probe_evaluation"]
    summary = evaluation["effective_train_size_summary"]
    _assert_summary_schema(summary, basis="resubstitution")
    assert evaluation["n_samples"] == 3
    assert summary["min"] == 3
    assert summary["median"] == 3.0
    assert summary["mean"] == 3.0
    assert summary["max"] == 3
    assert summary["mean_fraction_of_evaluation_cohort"] == 1.0


def test_skipped_evaluation_reports_unavailable_effective_train_size() -> None:
    """Group-disjoint regression with one group has no usable training basis."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(32, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=X.shape[0])
    report = diagnose(
        X,
        y,
        target_mode="regression",
        groups=np.zeros(X.shape[0], dtype=int),
        return_report=True,
        budget="fast",
        topology="off",
        random_state=17,
    )
    summary = report.metrics["probe_evaluation"]["effective_train_size_summary"]
    _assert_summary_schema(summary, basis=None)
    assert all(
        summary[field] is None for field in _SUMMARY_FIELDS - {"status", "basis"}
    )


def test_effective_train_size_summary_survives_terse_and_full_serialization() -> None:
    """Scalar effective-size metadata is retained in both report views and JSON."""
    X, y = make_classification(
        n_samples=61,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=21,
    )
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=21,
    )

    expected = report.metrics["probe_evaluation"]["effective_train_size_summary"]
    terse = report.to_dict()
    full = report.to_dict(terse=False)
    assert (
        terse["metrics"]["probe_evaluation"]["effective_train_size_summary"]
        == expected
    )
    assert (
        full["metrics"]["probe_evaluation"]["effective_train_size_summary"]
        == expected
    )

    terse_json = json.loads(report.to_json())
    full_json = json.loads(report.to_json(terse=False))
    assert (
        terse_json["metrics"]["probe_evaluation"]["effective_train_size_summary"]
        == expected
    )
    assert (
        full_json["metrics"]["probe_evaluation"]["effective_train_size_summary"]
        == expected
    )
