import json

from sklearn.datasets import make_blobs

from separatix import diagnose
from separatix.constants import HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
from separatix.recommendation.engine import (
    compute_multilabel_scores,
    compute_regression_scores,
    compute_scores,
    make_multilabel_recommendation,
)
from separatix.recommendation.text import render_recommendation


def _singlelabel_metrics(probes: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "probes": probes,
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 200, "1": 200},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
    }


def _multimetric_probe(
    first: float, second: float, third: float | None = None
) -> dict[str, float]:
    if third is None:
        return {
            "r2_variance_weighted": first,
            "r2_uniform_average": second,
        }
    return {
        "micro_f1": first,
        "macro_f1": second,
        "sample_jaccard": third,
    }


def test_singlelabel_frontier_uses_marginal_fallback_and_rank_floor() -> None:
    metrics = _singlelabel_metrics(
        {
            "dummy": {"balanced_accuracy": 0.50},
            "linear": {"balanced_accuracy": 0.60},
            "smooth_poly": {"balanced_accuracy": 0.92},
            "knn": {"balanced_accuracy": 0.93},
        }
    )

    compute_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["recommendation_evidence"]["plausible_family_set"]

    assert family_set["status"] == "available"
    assert family_set["minimum_recommended_family"] == "smooth_nonlinear"
    assert family_set["plausible_families"] == [
        "smooth_nonlinear",
        "local_kernel",
    ]
    assert family_set["decision_method"] == "marginal_standard_error_fallback"
    assert not family_set["assessments"]["linear"]["eligible_by_complexity"]


def test_singlelabel_frontier_is_not_applicable_without_signal() -> None:
    metrics = _singlelabel_metrics(
        {
            "dummy": {"balanced_accuracy": 0.50},
            "linear": {"balanced_accuracy": 0.51},
            "smooth_poly": {"balanced_accuracy": 0.52},
            "knn": {"balanced_accuracy": 0.51},
        }
    )

    compute_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["recommendation_evidence"]["plausible_family_set"]

    assert family_set["status"] == "not_applicable"
    assert family_set["minimum_recommended_family"] is None
    assert family_set["plausible_families"] == []


def test_singlelabel_frontier_reports_missing_required_family() -> None:
    metrics = _singlelabel_metrics(
        {
            "dummy": {"balanced_accuracy": 0.50},
            "linear": {"balanced_accuracy": 0.60},
            "smooth_poly": {"balanced_accuracy": 0.92},
        }
    )

    compute_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["recommendation_evidence"]["plausible_family_set"]

    assert family_set["status"] == "unavailable"
    assert family_set["minimum_recommended_family"] == "smooth_nonlinear"
    assert family_set["plausible_families"] == []
    assert not family_set["assessments"]["local_kernel"]["available"]


def test_multilabel_frontier_retains_competitive_smooth_and_local() -> None:
    metrics = {
        "probes": {
            "dummy": _multimetric_probe(0.10, 0.10, 0.10),
            "linear": _multimetric_probe(0.35, 0.35, 0.35),
            "smooth_poly": _multimetric_probe(0.75, 0.75, 0.75),
            "knn": _multimetric_probe(0.76, 0.74, 0.76),
        },
        "audit": {"n_samples": 120, "usable_label_count": 3},
    }

    compute_multilabel_scores(metrics, skipped_count=0, warning_count=0)
    evidence = metrics["multilabel_recommendation_evidence"]
    family_set = evidence["plausible_family_set"]

    assert evidence["recommended_family"] == "smooth_nonlinear"
    assert family_set["plausible_families"] == [
        "smooth_nonlinear",
        "local_kernel",
    ]
    assert family_set["decision_method"] == "marginal_standard_error_fallback"


def test_multilabel_unresolved_minimum_returns_nondominated_families() -> None:
    metrics = {
        "probes": {
            "dummy": _multimetric_probe(0.05, 0.05, 0.05),
            "linear": _multimetric_probe(0.30, 0.30, 0.30),
            "smooth_poly": _multimetric_probe(0.80, 0.30, 0.30),
            "knn": _multimetric_probe(0.30, 0.80, 0.30),
        },
        "audit": {"n_samples": 400, "usable_label_count": 3},
    }

    compute_multilabel_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["multilabel_recommendation_evidence"]["plausible_family_set"]

    assert family_set["status"] == "available"
    assert family_set["minimum_recommended_family"] is None
    assert family_set["plausible_families"] == [
        "linear",
        "smooth_nonlinear",
        "local_kernel",
    ]
    assert "nondominated" in family_set["reason"]


def test_multilabel_high_capacity_upgrade_stays_outside_core_family_set() -> None:
    metrics = {
        "probes": {
            "dummy": _multimetric_probe(0.10, 0.10, 0.10),
            "linear": _multimetric_probe(0.30, 0.30, 0.30),
            "smooth_poly": _multimetric_probe(0.50, 0.50, 0.50),
            "knn": _multimetric_probe(0.90, 0.90, 0.90),
            "mlp_shallow": _multimetric_probe(0.95, 0.95, 0.95),
        },
        "audit": {"n_samples": 400, "usable_label_count": 3},
        "graph": {"graph_fragmentation_score": 0.75},
        "boundary": {"boundary_sample_size": 20},
    }

    scores = compute_multilabel_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, _, _ = make_multilabel_recommendation(scores, metrics)
    family_set = metrics["multilabel_recommendation_evidence"]["plausible_family_set"]

    assert recommendation == HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
    assert family_set["plausible_families"] == ["local_kernel"]
    assert set(family_set["assessments"]) == {
        "linear",
        "smooth_nonlinear",
        "local_kernel",
    }


def test_regression_frontier_uses_pareto_dominance() -> None:
    metrics = {
        "probes": {
            "dummy": _multimetric_probe(0.00, 0.00),
            "linear": _multimetric_probe(0.30, 0.30),
            "smooth_poly": _multimetric_probe(0.75, 0.75),
            "knn": _multimetric_probe(0.77, 0.73),
        },
        "audit": {"n_samples": 400, "usable_target_count": 2},
    }

    compute_regression_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["regression_recommendation_evidence"]["plausible_family_set"]

    assert family_set["minimum_recommended_family"] == "smooth_nonlinear"
    assert family_set["plausible_families"] == [
        "smooth_nonlinear",
        "local_kernel",
    ]
    assert family_set["decision_method"] == "marginal_standard_error_fallback"


def test_regression_blocking_evidence_makes_set_not_applicable() -> None:
    metrics = {
        "probes": {
            "dummy": _multimetric_probe(0.00, 0.00),
            "linear": _multimetric_probe(0.60, 0.60),
            "smooth_poly": _multimetric_probe(0.70, 0.70),
            "knn": _multimetric_probe(0.72, 0.72),
        },
        "audit": {"n_samples": 120, "usable_target_count": 0},
    }

    compute_regression_scores(metrics, skipped_count=0, warning_count=0)
    family_set = metrics["regression_recommendation_evidence"]["plausible_family_set"]

    assert family_set["status"] == "not_applicable"
    assert family_set["plausible_families"] == []


def test_family_set_serializes_and_renders_in_default_text() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=0,
    )
    family_set = report.metrics["recommendation_evidence"]["plausible_family_set"]
    payload = json.loads(report.to_json())

    assert family_set["status"] == "available"
    assert (
        payload["metrics"]["recommendation_evidence"]["plausible_family_set"]
        == family_set
    )
    assert "Minimum recommended core family:" in report.recommendation_text
    assert "Statistically plausible core families:" in report.recommendation_text
    assert family_set["decision_method"] == "paired_oof_bootstrap"

    family_set["minimum_recommended_family"] = None
    family_set["plausible_families"] = ["smooth_nonlinear", "local_kernel"]
    unresolved_text = render_recommendation(report)
    assert "unresolved across primary metrics" in unresolved_text
    assert "core families (nondominated)" in unresolved_text

    family_set["status"] = "not_applicable"
    not_applicable_text = render_recommendation(report)
    assert "Minimum recommended core family" not in not_applicable_text
    assert "Statistically plausible core families" not in not_applicable_text
