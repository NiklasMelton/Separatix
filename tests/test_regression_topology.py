import builtins
import json

import numpy as np
from scipy import sparse

from separatix import diagnose


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(180, 5))
    y = (
        1.5 * X[:, 0]
        + 0.8 * X[:, 1] ** 2
        + np.where(X[:, 2] > 0.0, 1.0, -1.0)
        + rng.normal(scale=0.1, size=X.shape[0])
    )
    return X, y


def test_regression_topology_off_is_explicitly_skipped() -> None:
    X, y = _regression_data()
    report = diagnose(
        X,
        y,
        target_mode="regression",
        topology="off",
        return_report=True,
        random_state=0,
    )

    assert report.metrics["topology"] == {
        "target_type": "regression",
        "mode": "off",
        "skipped_reason": "topology disabled",
    }


def test_regression_topology_fast_auto_skips_work() -> None:
    X, y = _regression_data()
    report = diagnose(
        X,
        y,
        target_mode="regression",
        topology="auto",
        budget="fast",
        return_report=True,
        random_state=0,
    )

    assert report.metrics["topology"]["skipped_reason"] == (
        "topology disabled for this budget"
    )


def test_regression_graph_topology_reports_interpretable_hard_subsets() -> None:
    X, y = _regression_data()
    report = diagnose(
        X,
        np.column_stack([y, 0.5 * y + X[:, 3] ** 2]),
        target_mode="regression",
        topology="graph",
        return_report=True,
        random_state=0,
    )

    topology = report.metrics["topology"]
    assert topology["recommendation_role"] == "supporting_only"
    assert topology["topology_strength"] is not None
    assert {obj["object_type"] for obj in topology["objects"]} == {
        "high_residual_points",
        "high_discontinuity_points",
    }
    for obj in topology["objects"]:
        assert obj["candidate_count"] >= 30
        assert obj["selection_rule"].endswith("75th percentile")
        assert obj["graph"]["graph_rule"].startswith("mutual_")
        assert obj["persistent"]["skipped_reason"] == (
            "persistent topology not requested"
        )
    assert report.scores["topology_score"] == topology["topology_strength"]
    assert json.loads(report.to_json())["metrics"]["topology"]["target_type"] == (
        "regression"
    )


def test_regression_persistent_mode_keeps_graph_when_ripser_is_missing(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ripser":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X, y = _regression_data()
    report = diagnose(
        X,
        y,
        target_mode="regression",
        topology="persistent",
        return_report=True,
        random_state=0,
    )

    assert report.metrics["topology"]["topology_strength"] is not None
    assert all(
        obj["persistent"]["skipped_reason"] == "ripser is not installed"
        for obj in report.metrics["topology"]["objects"]
    )
    assert any(
        item["name"] == "regression_persistent_topology"
        for item in report.skipped_diagnostics
    )


def test_sparse_regression_graph_topology_avoids_densification() -> None:
    X, y = _regression_data()
    report = diagnose(
        sparse.csr_matrix(X),
        y,
        target_mode="regression",
        topology="graph",
        densify_policy="skip",
        return_report=True,
        random_state=0,
    )

    assert report.metrics["topology"]["topology_strength"] is not None
    assert not any(
        event.get("reason", "").startswith("regression_topology")
        for event in report.densification_events
    )


def test_regression_topology_does_not_change_recommendation_or_confidence() -> None:
    X, y = _regression_data()
    common = {
        "target_mode": "regression",
        "return_report": True,
        "random_state": 0,
    }
    without_topology = diagnose(X, y, topology="off", **common)
    with_topology = diagnose(X, y, topology="graph", **common)

    assert with_topology.recommendation == without_topology.recommendation
    assert with_topology.confidence == without_topology.confidence
