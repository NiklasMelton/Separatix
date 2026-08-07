from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from separatix import ProfilerConfig, diagnose
from separatix.constants import (
    FEEDFORWARD_MLP_RECOMMENDED,
)
from separatix.models import mlp as mlp_module
from separatix.models.mlp import (
    TorchMLPClassifier,
    _architecture_candidates,
    _evaluate_multilabel_models,
    _evaluate_regression_models,
    _evaluate_singlelabel_models,
    _multilabel_override_evidence,
    _regression_override_evidence,
    _resolve_device,
    _safe_evaluate_models,
    _singlelabel_override_evidence,
    _singlelabel_validation_indices,
    maybe_run_multilabel_mlp_probes,
    maybe_run_regression_mlp_probes,
    maybe_run_singlelabel_mlp_probes,
)
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
    evidence = report.metrics["mlp_recommendation_evidence"]
    assert evidence["override_policy"] == "paired_improvement_and_dummy_signal"
    assert evidence["trigger_threshold_used_for_override"] is False
    assert evidence["minimum_improvement"] == 0.02
    audit = report.metrics["mlp_probes"]["pairwise_comparison_audit"]
    assert audit == {
        "status": "not_run",
        "method": "paired_oof_bootstrap",
        "scope": "dummy_and_metric_strongest_simpler",
        "resamples_requested": 500,
        "resamples_used": 0,
        "resample_plan_id": None,
        "comparators_by_metric": {},
        "reason": (
            "MLP paired comparisons were not run because MLP probes were disabled."
        ),
    }


def test_good_simple_probe_prevents_mlp_trigger_even_when_requested() -> None:
    X, y = make_blobs(n_samples=160, centers=2, cluster_std=0.5, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0, mlp_probes=True)

    assert report.metrics["mlp_probes"]["status"] == "not_triggered"
    assert report.metrics["mlp_trigger_evidence"]["good_enough"] is True
    audit = report.metrics["mlp_probes"]["pairwise_comparison_audit"]
    assert audit["status"] == "not_run"
    assert audit["method"] == "paired_oof_bootstrap"
    assert audit["scope"] == "dummy_and_metric_strongest_simpler"
    assert audit["resamples_used"] == 0
    assert audit["resample_plan_id"] is None
    assert audit["comparators_by_metric"] == {}


def test_missing_torch_is_reported_only_after_mlp_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: None)
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
    audit = result["pairwise_comparison_audit"]
    assert audit["status"] == "unavailable"
    assert audit["method"] == "paired_oof_bootstrap"
    assert audit["scope"] == "dummy_and_metric_strongest_simpler"
    assert audit["resamples_requested"] == 500
    assert audit["resamples_used"] == 0
    assert audit["resample_plan_id"] is None
    assert audit["comparators_by_metric"] == {}
    assert "optional torch" in audit["reason"]


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
    assert "optional MLP probe clearly improved" in decision_path[0]
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

    assert recommendation == FEEDFORWARD_MLP_RECOMMENDED
    assert "optional MLP probe clearly improved" in decision_path[0]
    assert any("mlp_two_layer_compact" in item for item in decision_path)
    assert "mlp_recommendation_evidence" in interpretations


def test_mlp_override_is_skipped_without_held_out_split(monkeypatch) -> None:
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    metrics = {
        "audit": {"class_counts": {"0": 6, "1": 6}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: object())
    monkeypatch.setattr(
        mlp_module, "_resolve_device", lambda torch, device: ("cpu", None)
    )
    monkeypatch.setattr(
        mlp_module,
        "_split_rows",
        lambda *args, **kwargs: (None, "group_split_unavailable"),
    )
    context = {
        "warnings": [],
        "errors": [],
        "skipped_diagnostics": [],
        "densification_events": [],
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, random_state=0),
        metrics=metrics,
        report_context=context,
        class_labels=np.asarray([0, 1]),
    )
    assert result["status"] == "skipped"
    assert "held-out" in result["reason"]
    assert result["recommendation_override"] is False
    assert result["pairwise_comparison_audit"]["status"] == "not_run"
    assert result["pairwise_comparison_audit"]["resamples_used"] == 0


def test_safe_mlp_evaluation_localizes_architecture_failure() -> None:
    def evaluator(*, estimators, **kwargs):
        name = next(iter(estimators))
        if name == "bad":
            raise RuntimeError("training failed")
        return {name: {"score": 1.0}}

    errors: list[str] = []
    results = _safe_evaluate_models(
        evaluator,
        {"good": object(), "bad": object()},
        errors=errors,
    )
    assert results["good"]["score"] == 1.0
    assert results["bad"]["status"] == "runtime_failed"
    assert len(errors) == 1


def test_all_failed_mlp_architectures_disable_override_cleanly() -> None:
    evidence = _singlelabel_override_evidence(
        {
            "mlp_one_layer_compact": {
                "status": "runtime_failed",
                "error": "training failed",
            }
        },
        {},
        y_true=np.asarray([0, 1]),
        config=ProfilerConfig(mlp_probes=True, budget="fast"),
        groups=None,
    )
    assert evidence["recommendation_override"] is False
    assert evidence["best_architecture"] is None


def test_mlp_override_requires_clear_minimum_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(mlp_module._MLP_BUDGETS["fast"], "bootstrap_repeats", 50)
    y = np.asarray([0, 1] * 20)
    perfect = y.tolist()
    inverse = (1 - y).tolist()
    dummy = np.zeros_like(y).tolist()
    mlp_results = {
        "mlp_one_layer_compact": {
            "balanced_accuracy": 1.0,
            "predictions": perfect,
        }
    }
    weak_comparators = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
        **{
            name: {"balanced_accuracy": 0.0, "predictions": inverse}
            for name in ("linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    successful = _singlelabel_override_evidence(
        mlp_results,
        weak_comparators,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )
    tied_comparators = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
        **{
            name: {"balanced_accuracy": 1.0, "predictions": perfect}
            for name in ("linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    rejected = _singlelabel_override_evidence(
        mlp_results,
        tied_comparators,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )
    assert successful["recommendation_override"] is True
    assert rejected["recommendation_override"] is False
    assert successful["pairwise_comparisons"]["linear"]["point_delta"] == 1.0


def _singlelabel_decoupling_case() -> tuple[
    np.ndarray, dict[str, dict[str, object]], dict[str, dict[str, object]]
]:
    """Return aligned evidence with clear gains but normalized skill below 0.75."""
    y = np.asarray([0, 1] * 50)
    dummy = np.zeros_like(y)
    simpler = dummy.copy()
    mlp = dummy.copy()
    positive_rows = np.flatnonzero(y == 1)
    simpler[positive_rows[:10]] = 1
    mlp[positive_rows[:20]] = 1
    mlp_results: dict[str, dict[str, object]] = {
        "mlp_one_layer_compact": {
            "balanced_accuracy": 0.7,
            "predictions": mlp.tolist(),
        }
    }
    comparator_results: dict[str, dict[str, object]] = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy.tolist()},
        **{
            name: {
                "balanced_accuracy": 0.6,
                "predictions": simpler.tolist(),
            }
            for name in ("linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    return y, mlp_results, comparator_results


def test_completed_override_is_independent_of_trigger_threshold() -> None:
    y, mlp_results, comparator_results = _singlelabel_decoupling_case()
    outcomes = [
        _singlelabel_override_evidence(
            mlp_results,
            comparator_results,
            y_true=y,
            config=ProfilerConfig(
                mlp_probes=True,
                budget="fast",
                random_state=0,
                mlp_trigger_skill_threshold=threshold,
            ),
            groups=None,
        )
        for threshold in (0.0, 1.0)
    ]

    assert [item["recommendation_override"] for item in outcomes] == [True, True]
    assert outcomes[0]["absolute_skill"] == pytest.approx(0.4)
    assert outcomes[0]["absolute_skill"] < 0.75
    assert outcomes[0]["trigger_threshold_used_for_override"] is False
    assert outcomes[0]["metrics_clearing_override"] == ["balanced_accuracy"]


def test_singlelabel_override_compares_against_point_strongest_probe() -> None:
    y = np.asarray([0, 1] * 20)
    perfect = y.tolist()
    inverse = (1 - y).tolist()
    dummy = np.zeros_like(y).tolist()
    comparators = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
        "linear": {"balanced_accuracy": 0.0, "predictions": inverse},
        "smooth_poly": {"balanced_accuracy": 0.0, "predictions": inverse},
        "knn": {"balanced_accuracy": 0.0, "predictions": inverse},
        "kernel_approx": {"balanced_accuracy": 1.0, "predictions": perfect},
    }
    evidence = _singlelabel_override_evidence(
        {
            "mlp_one_layer_compact": {
                "balanced_accuracy": 1.0,
                "predictions": perfect,
            }
        },
        comparators,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["strongest_simpler_probe_by_metric"] == {
        "balanced_accuracy": "kernel_approx"
    }
    assert set(evidence["pairwise_comparisons"]) == {"dummy", "kernel_approx"}
    assert evidence["recommendation_override"] is False
    assert evidence["metrics_beating_dummy"] == ["balanced_accuracy"]
    assert evidence["metrics_beating_strongest_simpler"] == []
    assert evidence["pairwise_comparison_audit"]["comparators_by_metric"] == {
        "balanced_accuracy": {
            "dummy": "dummy",
            "strongest_simpler": "kernel_approx",
        }
    }


def test_mlp_pairwise_audit_resample_plan_is_deterministic() -> None:
    y = np.asarray([0, 1] * 30)
    perfect = y.tolist()
    inverse = (1 - y).tolist()
    comparator_results = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": np.zeros_like(y).tolist()},
        **{
            name: {"balanced_accuracy": 0.0, "predictions": inverse}
            for name in ("linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    mlp_results = {
        "mlp_one_layer_compact": {
            "balanced_accuracy": 1.0,
            "predictions": perfect,
        }
    }
    config = ProfilerConfig(
        mlp_probes=True,
        budget="fast",
        random_state=7,
    )
    first = _singlelabel_override_evidence(
        mlp_results,
        comparator_results,
        y_true=y,
        config=config,
        groups=None,
    )["pairwise_comparison_audit"]
    second = _singlelabel_override_evidence(
        mlp_results,
        comparator_results,
        y_true=y,
        config=config,
        groups=None,
    )["pairwise_comparison_audit"]

    assert first["status"] == second["status"] == "available"
    assert first["method"] == "paired_oof_bootstrap"
    assert first["scope"] == "dummy_and_metric_strongest_simpler"
    assert first["resample_plan_id"] == second["resample_plan_id"]
    assert first["resamples_used"] == second["resamples_used"]


def test_singlelabel_override_requires_paired_dummy_signal() -> None:
    y = np.asarray([0, 1] * 20)
    dummy = np.zeros_like(y).tolist()
    inverse = (1 - y).tolist()
    evidence = _singlelabel_override_evidence(
        {
            "mlp_one_layer_compact": {
                "balanced_accuracy": 0.5,
                "predictions": dummy,
            }
        },
        {
            "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
            **{
                name: {"balanced_accuracy": 0.0, "predictions": inverse}
                for name in ("linear", "smooth_poly", "knn", "kernel_approx")
            },
        },
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["metrics_beating_strongest_simpler"] == ["balanced_accuracy"]
    assert evidence["metrics_beating_dummy"] == []
    assert evidence["recommendation_override"] is False
    assert "dummy baseline" in evidence["override_reason"]


def _patch_cached_pair_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Return deterministic cache stubs while recording target-aware inputs."""
    calls: list[dict[str, object]] = []

    def fake_build(y_true, predictions, **kwargs):
        calls.append(
            {
                "y_true": np.asarray(y_true),
                "predictions": dict(predictions),
                **kwargs,
            }
        )
        metric_names = {
            "singlelabel": ("balanced_accuracy",),
            "multilabel": ("micro_f1", "macro_f1", "sample_jaccard"),
            "regression": ("r2_variance_weighted", "r2_uniform_average"),
        }[kwargs["target_mode"]]
        return SimpleNamespace(
            status="available",
            reason=None,
            resamples_requested=int(kwargs["requested_resamples"]),
            resamples_used=12,
            resample_plan_id="shared-test-plan",
            metric_names=metric_names,
        )

    def fake_summary(cache, first_probe, second_probe, *, point_scores):
        first_points = point_scores[first_probe]
        second_points = point_scores[second_probe]
        return {
            "metrics": {
                metric: {
                    "point_delta": float(first_points[metric])
                    - float(second_points[metric]),
                    "mean_delta": float(first_points[metric])
                    - float(second_points[metric]),
                    "paired_standard_error": 0.0,
                    "lower_95": max(
                        0.0,
                        float(first_points[metric]) - float(second_points[metric]),
                    ),
                    "upper_95": float(first_points[metric])
                    - float(second_points[metric]),
                    "resamples_requested": cache.resamples_requested,
                    "resamples_used": cache.resamples_used,
                }
                for metric in cache.metric_names
            }
        }

    monkeypatch.setattr(mlp_module, "_build_paired_score_cache", fake_build)
    monkeypatch.setattr(mlp_module, "_summarize_cached_probe_pair", fake_summary)
    return calls


def test_multilabel_override_retains_only_dummy_and_metric_specific_strongest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_pair_summaries(monkeypatch)
    Y = np.zeros((20, 3), dtype=int)
    predictions = Y.tolist()
    mlp_results = {
        "mlp_one_layer_compact": {
            "micro_f1": 0.9,
            "macro_f1": 0.9,
            "sample_jaccard": 0.9,
            "predictions": predictions,
        }
    }
    comparator_results = {
        "dummy": {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "sample_jaccard": 0.0,
            "predictions": predictions,
        },
        "linear": {
            "micro_f1": 0.8,
            "macro_f1": 0.2,
            "sample_jaccard": 0.2,
            "predictions": predictions,
        },
        "smooth_poly": {
            "micro_f1": 0.7,
            "macro_f1": 0.8,
            "sample_jaccard": 0.3,
            "predictions": predictions,
        },
        "knn": {
            "micro_f1": 0.6,
            "macro_f1": 0.4,
            "sample_jaccard": 0.85,
            "predictions": predictions,
        },
        "kernel_approx": {
            "micro_f1": 0.5,
            "macro_f1": 0.3,
            "sample_jaccard": 0.5,
            "predictions": predictions,
        },
    }
    groups = np.repeat(np.arange(10), 2)
    labels = np.asarray(["a", "b", "c"])
    evidence = _multilabel_override_evidence(
        mlp_results,
        comparator_results,
        Y_true=Y,
        label_names=labels,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=groups,
    )

    assert evidence["strongest_simpler_probe_by_metric"] == {
        "micro_f1": "linear",
        "macro_f1": "smooth_poly",
        "sample_jaccard": "knn",
    }
    assert set(evidence["pairwise_comparisons"]) == {
        "dummy",
        "linear",
        "smooth_poly",
        "knn",
    }
    assert "kernel_approx" not in evidence["pairwise_comparisons"]
    assert evidence["pairwise_comparisons"]["linear"] == {
        "micro_f1": evidence["pairwise_comparisons"]["linear"]["micro_f1"]
    }
    assert evidence["pairwise_comparison_audit"] == {
        "status": "available",
        "method": "paired_oof_bootstrap",
        "scope": "dummy_and_metric_strongest_simpler",
        "resamples_requested": 200,
        "resamples_used": 12,
        "resample_plan_id": "shared-test-plan",
        "comparators_by_metric": {
            "micro_f1": {"dummy": "dummy", "strongest_simpler": "linear"},
            "macro_f1": {"dummy": "dummy", "strongest_simpler": "smooth_poly"},
            "sample_jaccard": {"dummy": "dummy", "strongest_simpler": "knn"},
        },
        "reason": None,
    }
    assert len(calls) == 1
    assert calls[0]["target_mode"] == "multilabel"
    assert np.array_equal(calls[0]["groups"], groups)
    assert np.array_equal(calls[0]["names"], labels)
    assert set(calls[0]["predictions"]) == {
        "mlp_one_layer_compact",
        "dummy",
        "linear",
        "smooth_poly",
        "knn",
    }


def test_regression_override_deduplicates_metric_specific_strongest_comparators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_pair_summaries(monkeypatch)
    Y = np.zeros((20, 2), dtype=float)
    predictions = Y.tolist()
    mlp_results = {
        "mlp_one_layer_compact": {
            "r2_variance_weighted": 0.9,
            "r2_uniform_average": 0.9,
            "predictions": predictions,
        }
    }
    comparator_results = {
        "dummy": {
            "r2_variance_weighted": 0.0,
            "r2_uniform_average": 0.0,
            "predictions": predictions,
        },
        "linear": {
            "r2_variance_weighted": 0.8,
            "r2_uniform_average": 0.85,
            "predictions": predictions,
        },
        "smooth_poly": {
            "r2_variance_weighted": 0.7,
            "r2_uniform_average": 0.7,
            "predictions": predictions,
        },
        "knn": {
            "r2_variance_weighted": 0.6,
            "r2_uniform_average": 0.4,
            "predictions": predictions,
        },
        "kernel_approx": {
            "r2_variance_weighted": 0.5,
            "r2_uniform_average": 0.3,
            "predictions": predictions,
        },
    }
    names = np.asarray(["first", "second"])
    evidence = _regression_override_evidence(
        mlp_results,
        comparator_results,
        Y_true=Y,
        target_names=names,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["strongest_simpler_probe_by_metric"] == {
        "r2_variance_weighted": "linear",
        "r2_uniform_average": "linear",
    }
    assert set(evidence["pairwise_comparisons"]) == {"dummy", "linear"}
    assert set(evidence["pairwise_comparisons"]["linear"]) == {
        "r2_variance_weighted",
        "r2_uniform_average",
    }
    assert "knn" not in evidence["pairwise_comparisons"]
    assert "kernel_approx" not in evidence["pairwise_comparisons"]
    assert evidence["pairwise_comparison_audit"]["comparators_by_metric"] == {
        "r2_variance_weighted": {
            "dummy": "dummy",
            "strongest_simpler": "linear",
        },
        "r2_uniform_average": {
            "dummy": "dummy",
            "strongest_simpler": "linear",
        },
    }
    assert len(calls) == 1
    assert calls[0]["target_mode"] == "regression"
    assert np.array_equal(calls[0]["names"], names)
    assert set(calls[0]["predictions"]) == {
        "mlp_one_layer_compact",
        "dummy",
        "linear",
    }


@pytest.mark.parametrize("failure", ["missing", "runtime", "misaligned", "metric"])
def test_singlelabel_override_requires_complete_comparator_evidence(
    failure: str,
) -> None:
    y, mlp_results, comparator_results = _singlelabel_decoupling_case()
    if failure == "missing":
        del comparator_results["kernel_approx"]
    elif failure == "runtime":
        comparator_results["kernel_approx"] = {"status": "runtime_failed"}
    elif failure == "misaligned":
        comparator_results["kernel_approx"]["predictions"] = [0]
    else:
        del comparator_results["kernel_approx"]["balanced_accuracy"]

    evidence = _singlelabel_override_evidence(
        mlp_results,
        comparator_results,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["required_comparators_complete"] is False
    assert evidence["missing_or_failed_comparators"] == ["kernel_approx"]
    assert evidence["recommendation_override"] is False


@pytest.mark.parametrize("qualifying_metrics, expected", [(2, True), (1, False)])
def test_multilabel_override_requires_two_jointly_qualifying_metrics(
    monkeypatch: pytest.MonkeyPatch,
    qualifying_metrics: int,
    expected: bool,
) -> None:
    _patch_cached_pair_summaries(monkeypatch)
    metric_names = ("micro_f1", "macro_f1", "sample_jaccard")
    Y = np.zeros((20, 3), dtype=int)
    mlp_scores = {metric: 0.6 for metric in metric_names}
    simple_scores = {
        metric: 0.5 if index < qualifying_metrics else 0.6
        for index, metric in enumerate(metric_names)
    }
    predictions = Y.tolist()
    evidence = _multilabel_override_evidence(
        {
            "mlp_one_layer_compact": {
                **mlp_scores,
                "predictions": predictions,
            }
        },
        {
            "dummy": {
                **{metric: 0.0 for metric in metric_names},
                "predictions": predictions,
            },
            **{
                name: {**simple_scores, "predictions": predictions}
                for name in ("linear", "smooth_poly", "knn", "kernel_approx")
            },
        },
        Y_true=Y,
        label_names=np.asarray(["a", "b", "c"]),
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["recommendation_override"] is expected
    assert len(evidence["metrics_clearing_override"]) == qualifying_metrics
    assert evidence["required_metrics_to_override"] == 2


@pytest.mark.parametrize("qualifying_metrics, expected", [(2, True), (1, False)])
def test_regression_override_requires_both_jointly_qualifying_metrics(
    monkeypatch: pytest.MonkeyPatch,
    qualifying_metrics: int,
    expected: bool,
) -> None:
    _patch_cached_pair_summaries(monkeypatch)
    metric_names = ("r2_variance_weighted", "r2_uniform_average")
    Y = np.zeros((20, 2), dtype=float)
    mlp_scores = {metric: 0.6 for metric in metric_names}
    simple_scores = {
        metric: 0.5 if index < qualifying_metrics else 0.6
        for index, metric in enumerate(metric_names)
    }
    predictions = Y.tolist()
    evidence = _regression_override_evidence(
        {
            "mlp_one_layer_compact": {
                **mlp_scores,
                "predictions": predictions,
            }
        },
        {
            "dummy": {
                **{metric: 0.0 for metric in metric_names},
                "predictions": predictions,
            },
            **{
                name: {**simple_scores, "predictions": predictions}
                for name in ("linear", "smooth_poly", "knn", "kernel_approx")
            },
        },
        Y_true=Y,
        target_names=np.asarray(["one", "two"]),
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )

    assert evidence["recommendation_override"] is expected
    assert len(evidence["metrics_clearing_override"]) == qualifying_metrics
    assert evidence["required_metrics_to_override"] == 2


def test_mlp_parameter_cap_and_device_fallbacks() -> None:
    assert _architecture_candidates(100, 10, max_parameters=1) == []

    unavailable = SimpleNamespace(is_available=lambda: False)
    fake_torch = SimpleNamespace(
        cuda=unavailable,
        backends=SimpleNamespace(mps=unavailable),
    )
    cuda_device, cuda_warning = _resolve_device(fake_torch, "cuda")
    mps_device, mps_warning = _resolve_device(fake_torch, "mps")
    assert cuda_device == mps_device == "cpu"
    assert "CUDA" in str(cuda_warning)
    assert "MPS" in str(mps_warning)


def test_aligned_mlp_evaluators_cover_resubstitution_and_held_out_modes() -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = np.asarray([0, 1] * 6)
    Y_multi = np.column_stack([y, 1 - y])
    Y_reg = np.column_stack([X[:, 0], X[:, 1]])
    first = np.arange(0, 6)
    second = np.arange(6, 12)
    splits = [(first, second), (second, first)]

    single = _evaluate_singlelabel_models(
        X,
        y,
        {"dummy": mlp_module.DummyClassifier(strategy="prior")},
        class_labels=np.asarray([0, 1]),
        splits=splits,
        evaluation_mode="cross_validation",
        groups=None,
    )
    multi = _evaluate_multilabel_models(
        X,
        Y_multi,
        {"dummy": mlp_module.MultilabelPriorDummy()},
        label_names=np.asarray(["a", "b"]),
        splits=None,
        evaluation_mode="descriptive_resubstitution",
        groups=None,
    )
    regression = _evaluate_regression_models(
        X,
        Y_reg,
        {"dummy": mlp_module.TargetMeanDummyRegressor()},
        target_names=np.asarray(["one", "two"]),
        splits=splits,
        evaluation_mode="cross_validation",
        groups=None,
    )

    assert single["dummy"]["evaluation_mode"] == "cross_validation"
    assert multi["dummy"]["evaluation_mode"] == "descriptive_resubstitution"
    assert regression["dummy"]["evaluation_mode"] == "cross_validation"


def test_mlp_inner_validation_keeps_every_class_on_both_sides() -> None:
    y = np.repeat(np.arange(5), 4)
    fit_idx, valid_idx = _singlelabel_validation_indices(y, random_state=0)
    assert set(y[fit_idx]) == set(range(5))
    assert set(y[valid_idx]) == set(range(5))


def _patch_mlp_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: object())
    monkeypatch.setattr(
        mlp_module, "_resolve_device", lambda torch, device: ("cpu", None)
    )
    monkeypatch.setitem(mlp_module._MLP_BUDGETS["fast"], "bootstrap_repeats", 5)


def test_singlelabel_mlp_completed_path_disables_override_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_mlp_runtime(monkeypatch)
    X, y = make_blobs(n_samples=60, centers=2, n_features=4, random_state=4)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        if names[0].startswith("mlp_"):
            results = {
                name: {
                    "balanced_accuracy": 1.0,
                    "predictions": y.tolist(),
                }
                for name in names
            }
            results[names[-1]] = {
                "status": "runtime_failed",
                "error": "architecture failed",
            }
            return results
        wrong = (1 - y).tolist()
        return {
            name: {"balanced_accuracy": 0.5, "predictions": wrong} for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "audit": {"class_counts": {"0": 30, "1": 30}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        class_labels=np.asarray([0, 1]),
    )
    assert result["status"] == "completed"
    assert result["recommendation_override"] is False
    assert result["required_comparators_complete"] is True
    assert set(result["aligned_comparators"]) == set(
        mlp_module._REQUIRED_MLP_COMPARATORS
    )
    assert result["architectures_complete"] is False
    assert any(
        item.get("status") == "runtime_failed" for item in result["architectures"]
    )


def test_multilabel_mlp_completed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlp_runtime(monkeypatch)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 4))
    Y = (rng.random((60, 3)) < 0.4).astype(int)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        is_mlp = names[0].startswith("mlp_")
        predictions = Y if is_mlp else np.zeros_like(Y)
        score = 1.0 if is_mlp else 0.0
        return {
            name: {
                "micro_f1": score,
                "macro_f1": score,
                "sample_jaccard": score,
                "predictions": predictions.tolist(),
            }
            for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "audit": {"n_samples": 60},
        "probes": {
            name: {"micro_f1": 0.0, "macro_f1": 0.0, "sample_jaccard": 0.0}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_multilabel_mlp_probes(
        X,
        Y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        label_names=np.asarray(["a", "b", "c"]),
    )
    assert result["status"] == "completed"
    assert result["best_architecture"] is not None
    assert set(result["aligned_comparators"]) == set(
        mlp_module._REQUIRED_MLP_COMPARATORS
    )


def test_regression_mlp_completed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlp_runtime(monkeypatch)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 4))
    Y = np.column_stack([X[:, 0], X[:, 1]])

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        is_mlp = names[0].startswith("mlp_")
        predictions = Y if is_mlp else np.zeros_like(Y)
        score = 1.0 if is_mlp else 0.0
        return {
            name: {
                "r2_variance_weighted": score,
                "r2_uniform_average": score,
                "predictions": predictions.tolist(),
            }
            for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "probes": {
            name: {"r2_variance_weighted": 0.0, "r2_uniform_average": 0.0}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_regression_mlp_probes(
        X,
        Y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        target_names=np.asarray(["one", "two"]),
    )
    assert result["status"] == "completed"
    assert result["best_architecture"] is not None
    assert set(result["aligned_comparators"]) == set(
        mlp_module._REQUIRED_MLP_COMPARATORS
    )


def test_torch_mlp_does_not_mutate_numpy_global_rng() -> None:
    torch = mlp_module._torch_module()
    if torch is None:
        pytest.skip("torch is not installed")
    X, y = make_blobs(n_samples=24, centers=2, n_features=3, random_state=0)
    np.random.seed(123)
    state = np.random.get_state()
    torch.manual_seed(321)
    torch_state = torch.random.get_rng_state().clone()
    estimator = TorchMLPClassifier(
        task="singlelabel",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    estimator.fit(X, y)
    multilabel = TorchMLPClassifier(
        task="multilabel",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    multilabel.fit(X, np.column_stack([y, 1 - y]))
    assert multilabel.predict(X).shape == (24, 2)
    regressor = mlp_module.TorchMLPRegressor(
        task="regression",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    regressor.fit(X, X[:, :2])
    assert regressor.predict(X).shape == (24, 2)
    observed = np.random.random(5)
    np.random.set_state(state)
    expected = np.random.random(5)
    assert np.array_equal(observed, expected)
    observed_torch = torch.rand(5)
    torch.random.set_rng_state(torch_state)
    expected_torch = torch.rand(5)
    assert torch.equal(observed_torch, expected_torch)


def test_torch_module_rejects_incomplete_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubSpec:
        loader = object()

    monkeypatch.setattr(mlp_module, "find_spec", lambda _name: StubSpec())
    monkeypatch.setattr(mlp_module.importlib, "import_module", lambda _name: object())

    assert mlp_module._torch_module() is None
