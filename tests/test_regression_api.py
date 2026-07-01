import json

import numpy as np
import pytest
from scipy import sparse

from separatix import DiagnosticReport, diagnose


def _linear_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 5))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.05, size=160)
    return X, y


def test_regression_requires_explicit_target_mode() -> None:
    X, y = _linear_regression_data()
    with pytest.raises(ValueError, match="categorical single-output"):
        diagnose(X, y, return_report=True, random_state=0)


def test_single_target_regression_returns_report_and_json() -> None:
    X, y = _linear_regression_data()
    report = diagnose(
        X,
        y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    assert isinstance(report, DiagnosticReport)
    assert report.class_summary["target_type"] == "regression"
    assert "regression_recommendation_evidence" in report.metrics
    assert "explicit regression diagnostic" in report.recommendation_text
    assert json.loads(report.to_json())["class_summary"]["target_type"] == "regression"


def test_multitarget_regression_reports_per_target_summary() -> None:
    X, y = _linear_regression_data()
    Y = np.column_stack([y, X[:, 2] ** 2])
    report = diagnose(
        X,
        Y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    assert report.class_summary["n_targets"] == 2
    assert report.metrics["audit"]["usable_target_count"] == 2
    assert (
        report.metrics["probes"]["linear"]["per_target_r2_summary"]["min"] is not None
    )


def test_sparse_regression_input_runs_without_global_densification() -> None:
    X, y = _linear_regression_data()
    report = diagnose(
        sparse.csr_matrix(X),
        y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    assert report.class_summary["target_type"] == "regression"
    assert report.preprocessing["is_sparse"] is True


def test_constant_regression_target_is_skipped_when_other_targets_work() -> None:
    X, y = _linear_regression_data()
    Y = np.column_stack([y, np.ones_like(y)])
    report = diagnose(
        X,
        Y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    assert report.class_summary["constant_target_count"] == 1
    assert any(
        item["name"] == "constant_regression_targets"
        for item in report.skipped_diagnostics
    )


def test_all_constant_regression_targets_are_rejected() -> None:
    X = np.ones((10, 2))
    y = np.ones(10)
    with pytest.raises(ValueError, match="non-constant"):
        diagnose(X, y, target_mode="regression", return_report=True)


def test_linear_regression_data_recommends_linear_or_inconclusive() -> None:
    X, y = _linear_regression_data()
    report = diagnose(
        X,
        y,
        target_mode="regression",
        return_report=True,
        budget="standard",
        random_state=0,
    )
    assert report.recommendation in {
        "linear_response_likely_sufficient",
        "inconclusive_regression_diagnostic",
    }
