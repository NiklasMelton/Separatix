import numpy as np
from scipy import sparse

from separatix import DiagnosticReport, diagnose


def _linear_multilabel_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    Y = np.column_stack(
        [
            X[:, 0] > 0,
            X[:, 1] + X[:, 2] > 0,
            X[:, 3] - X[:, 4] > 0,
        ]
    ).astype(int)
    return X, Y


def test_diagnose_auto_multilabel_returns_report() -> None:
    X, Y = _linear_multilabel_data()
    report = diagnose(X, Y, return_report=True, budget="fast", random_state=0)
    assert isinstance(report, DiagnosticReport)
    assert report.class_summary["target_type"] == "multilabel"
    assert "multilabel_recommendation_evidence" in report.metrics


def test_diagnose_multilabel_returns_text_by_default() -> None:
    X, Y = _linear_multilabel_data()
    text = diagnose(X, Y, target_mode="multilabel", budget="fast", random_state=0)
    assert isinstance(text, str)
    assert "multilabel diagnostic" in text


def test_multilabel_report_serializes_and_prunes_verbose_fields() -> None:
    X, Y = _linear_multilabel_data()
    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    terse = report.to_dict()
    full = report.to_dict(terse=False)
    assert "per_label_metrics" not in terse["metrics"]["probes"]["linear"]
    assert "predictions" not in terse["metrics"]["probes"]["linear"]
    assert "local_label_entropy" not in terse["metrics"]["neighborhood"]
    assert "per_label_metrics" in full["metrics"]["probes"]["linear"]
    assert "predictions" in full["metrics"]["probes"]["linear"]
    assert report.to_json()


def test_sparse_multilabel_input_runs_without_global_densification() -> None:
    X, Y = _linear_multilabel_data()
    sparse_X = sparse.csr_matrix(X)
    sparse_Y = sparse.csr_matrix(Y)
    report = diagnose(
        sparse_X,
        sparse_Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        densify_policy="skip",
        max_dense_mb=1,
        random_state=0,
    )
    assert report.preprocessing["is_sparse"] is True
    assert report.class_summary["target_type"] == "multilabel"


def test_multilabel_empty_union_jaccard_convention_is_reported() -> None:
    X, Y = _linear_multilabel_data()
    Y[:10] = 0
    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        random_state=0,
    )
    neighborhood = report.metrics["neighborhood"]
    assert neighborhood["empty_union_jaccard_convention"]
    assert "all_zero_neighbor_pair_fraction" in neighborhood
