import numpy as np
import pytest
from sklearn.datasets import make_blobs

from separatix import ProfilerConfig, diagnose
from separatix.constants import (
    FEEDFORWARD_MLP_RECOMMENDED,
    FEEDFORWARD_MLP_REGRESSION_RECOMMENDED,
)
from separatix.models.mlp import maybe_run_singlelabel_mlp_probes
from separatix.recommendation.engine import (
    compute_regression_scores,
    compute_scores,
    make_recommendation,
    make_regression_recommendation,
)


def test_profiler_config_validates_mlp_thresholds() -> None:
    with pytest.raises(ValueError, match="mlp_trigger_skill_threshold"):
        ProfilerConfig(mlp_trigger_skill_threshold=1.5)
    with pytest.raises(ValueError, match="mlp_min_improvement"):
        ProfilerConfig(mlp_min_improvement=-0.1)


def test_default_report_marks_mlp_not_requested() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)

    assert report.metrics["mlp_probes"]["status"] == "not_requested"
    assert report.metrics["mlp_trigger_evidence"]["status"] == "not_requested"
    assert report.metrics["mlp_recommendation_evidence"]["status"] == "not_requested"


def test_good_simple_probe_prevents_mlp_trigger_even_when_requested() -> None:
    X, y = make_blobs(n_samples=160, centers=2, cluster_std=0.5, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0, mlp_probes=True)

    assert report.metrics["mlp_probes"]["status"] == "not_triggered"
    assert report.metrics["mlp_trigger_evidence"]["good_enough"] is True


def test_missing_torch_is_reported_only_after_mlp_trigger() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0, 1, 0, 1], dtype=int)
    config = ProfilerConfig(mlp_probes=True, random_state=0)
    metrics = {
        "audit": {"class_counts": {"0": 2, "1": 2}},
        "probes": {
            "dummy": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "linear": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "smooth_poly": {
                "balanced_accuracy": 0.5,
                "per_class_recall": [0.5, 0.5],
            },
            "knn": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "kernel_approx": {
                "balanced_accuracy": 0.5,
                "per_class_recall": [0.5, 0.5],
            },
        },
    }

    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=config,
        metrics=metrics,
        report_context={
            "warnings": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        },
        class_labels=np.asarray([0, 1]),
    )

    assert result["trigger"]["status"] == "triggered"
    assert result["status"] == "dependency_unavailable"


def test_singlelabel_recommendation_can_be_overridden_by_mlp_evidence() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.7},
            "smooth_poly": {"balanced_accuracy": 0.79},
            "knn": {"balanced_accuracy": 0.78},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 60, "1": 60},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "smooth_poly"},
        "mlp_recommendation_evidence": {
            "recommendation_override": True,
            "override_reason": (
                "The best tested MLP clearly beat the aligned simpler probes."
            ),
            "best_architecture": {"probe_name": "mlp_one_layer_compact"},
        },
    }

    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, decision_path, interpretations = make_recommendation(
        scores, metrics
    )

    assert recommendation == FEEDFORWARD_MLP_RECOMMENDED
    assert any("mlp_one_layer_compact" in item for item in decision_path)
    assert "mlp_recommendation_evidence" in interpretations


def test_regression_recommendation_can_be_overridden_by_mlp_evidence() -> None:
    metrics = {
        "probes": {
            "dummy": {
                "r2_variance_weighted": 0.0,
                "r2_uniform_average": 0.0,
            },
            "linear": {
                "r2_variance_weighted": 0.55,
                "r2_uniform_average": 0.5,
            },
            "smooth_poly": {
                "r2_variance_weighted": 0.68,
                "r2_uniform_average": 0.62,
            },
            "knn": {
                "r2_variance_weighted": 0.66,
                "r2_uniform_average": 0.6,
            },
        },
        "neighborhood": {
            "target_smoothness_score": 0.75,
            "high_discontinuity_fraction": 0.1,
        },
        "topology": {},
        "audit": {"n_samples": 120},
        "mlp_recommendation_evidence": {
            "recommendation_override": True,
            "override_reason": (
                "The best tested MLP clearly beat the aligned simpler regressors."
            ),
            "best_architecture": {"probe_name": "mlp_two_layer_compact"},
        },
    }

    scores = compute_regression_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, decision_path, interpretations = make_regression_recommendation(
        scores, metrics
    )

    assert recommendation == FEEDFORWARD_MLP_REGRESSION_RECOMMENDED
    assert any("mlp_two_layer_compact" in item for item in decision_path)
    assert "mlp_recommendation_evidence" in interpretations
