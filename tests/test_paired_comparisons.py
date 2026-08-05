from __future__ import annotations

import numpy as np
from scipy import sparse

from separatix import ComplexityProfiler, diagnose
from separatix.models.comparison import (
    bootstrap_indices,
    build_paired_probe_comparisons,
    lookup_paired_comparison,
)
from separatix.recommendation.engine import compute_scores, make_recommendation


def _singlelabel_probe_results(
    y: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "balanced_accuracy": float(
                np.mean(
                    [
                        np.mean(values[y == cls] == cls)
                        for cls in np.unique(y)
                    ]
                )
            ),
            "predictions": values.tolist(),
            "evaluation_plan_id": "plan",
        }
        for name, values in predictions.items()
    }


def test_paired_comparison_is_deterministic_and_oriented() -> None:
    y = np.asarray([0] * 50 + [1] * 50)
    dummy = np.zeros_like(y)
    linear = y.copy()
    probes = _singlelabel_probe_results(y, {"dummy": dummy, "linear": linear})

    first = build_paired_probe_comparisons(
        probes,
        y,
        target_mode="singlelabel",
        requested_resamples=100,
        random_state=7,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    second = build_paired_probe_comparisons(
        probes,
        y,
        target_mode="singlelabel",
        requested_resamples=100,
        random_state=7,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    comparison = lookup_paired_comparison(
        first, "linear", "dummy", "balanced_accuracy"
    )

    assert first == second
    assert comparison is not None
    assert comparison["point_delta"] == 0.5
    assert comparison["lower_95"] > 0.0
    assert first["resamples_used"] == 100


def test_identical_predictions_have_zero_paired_interval() -> None:
    y = np.asarray([0, 1] * 40)
    probes = _singlelabel_probe_results(
        y, {"dummy": np.zeros_like(y), "linear": np.zeros_like(y)}
    )
    payload = build_paired_probe_comparisons(
        probes,
        y,
        target_mode="singlelabel",
        requested_resamples=100,
        random_state=0,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    comparison = lookup_paired_comparison(
        payload, "linear", "dummy", "balanced_accuracy"
    )

    assert comparison is not None
    assert comparison["point_delta"] == 0.0
    assert comparison["lower_95"] == 0.0
    assert comparison["upper_95"] == 0.0


def test_group_bootstrap_keeps_group_rows_whole() -> None:
    groups = np.repeat(np.arange(5), 3)
    for indices in bootstrap_indices(
        groups.size,
        repeats=20,
        random_state=0,
        groups=groups,
    ):
        for group_id in np.unique(groups):
            count = int(np.sum(groups[indices] == group_id))
            assert count % 3 == 0


def test_real_probe_run_shares_rows_folds_and_plan_ids() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int)
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=3,
    )

    evaluation = report.metrics["probe_evaluation"]
    available = [
        result
        for result in report.metrics["probes"].values()
        if result.get("predictions") is not None
    ]
    assert evaluation["alignment_status"] == "aligned"
    assert len(evaluation["fold_assignments"]) == evaluation["n_samples"]
    assert {
        result["evaluation_plan_id"] for result in available
    } == {evaluation["evaluation_plan_id"]}
    assert {
        tuple(result["sample_info"]["indices"]) for result in available
    } == {tuple(evaluation["row_indices"])}
    assert report.metrics["paired_probe_comparisons"]["status"] == "available"


def test_multilabel_and_regression_runs_use_paired_evidence() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(90, 4))
    Y_labels = np.column_stack(
        [
            (X[:, 0] > 0).astype(int),
            (X[:, 1] + X[:, 2] > 0).astype(int),
        ]
    )
    Y_regression = np.column_stack(
        [X[:, 0] + 0.1 * rng.normal(size=90), X[:, 1] ** 2]
    )

    multilabel = diagnose(
        X,
        Y_labels,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        topology="off",
        random_state=8,
    )
    regression = diagnose(
        X,
        Y_regression,
        target_mode="regression",
        return_report=True,
        budget="fast",
        topology="off",
        random_state=8,
    )

    assert multilabel.metrics["paired_probe_comparisons"]["status"] == "available"
    assert regression.metrics["paired_probe_comparisons"]["status"] == "available"
    assert all(
        item["decision_method"] == "paired_oof_bootstrap"
        for item in multilabel.metrics["multilabel_recommendation_evidence"][
            "signal_comparisons"
        ].values()
    )
    assert all(
        item["decision_method"] == "paired_oof_bootstrap"
        for item in regression.metrics["regression_recommendation_evidence"][
            "signal_comparisons"
        ].values()
    )


def test_grouped_alignment_keeps_each_group_in_one_test_fold() -> None:
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(12), 6)
    X = rng.normal(size=(groups.size, 4))
    y = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 12)
    report = diagnose(
        X,
        y,
        groups=groups,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=9,
    )
    evaluation = report.metrics["probe_evaluation"]
    used_groups = groups[np.asarray(evaluation["row_indices"], dtype=int)]
    assignments = np.asarray(evaluation["fold_assignments"], dtype=int)

    assert evaluation["group_aware"] is True
    for group_id in np.unique(used_groups):
        assert np.unique(assignments[used_groups == group_id]).size == 1


def test_sparse_warn_and_sample_uses_one_shared_probe_cohort() -> None:
    rng = np.random.default_rng(4)
    X = sparse.csr_matrix(rng.normal(size=(300, 1000)))
    y = np.asarray([0, 1] * 150)
    profiler = ComplexityProfiler(
        budget="fast",
        topology="off",
        max_dense_mb=1,
        min_dense_samples=10,
        warn_on_densify=False,
        random_state=4,
    ).fit(X, y)
    report = profiler.report()
    available = [
        result
        for result in report.metrics["probes"].values()
        if result.get("predictions") is not None
    ]
    alignment_events = [
        event
        for event in report.densification_events
        if event["reason"] == "probe_family_alignment"
    ]

    assert len(alignment_events) == 1
    assert alignment_events[0]["sampling_used"] is True
    assert len({result["evaluation_support"]["n_samples"] for result in available}) == 1
    assert {
        tuple(result["sample_info"]["indices"]) for result in available
    } == {tuple(report.metrics["probe_evaluation"]["row_indices"])}


def test_paired_evidence_drives_singlelabel_family_decision() -> None:
    y = np.asarray([0] * 100 + [1] * 100)
    dummy = np.zeros_like(y)
    linear = y.copy()
    smooth = y.copy()
    local = y.copy()
    linear[np.r_[0:20, 100:120]] = 1 - linear[np.r_[0:20, 100:120]]
    smooth[np.r_[0:10, 100:110]] = 1 - smooth[np.r_[0:10, 100:110]]
    probes = _singlelabel_probe_results(
        y,
        {
            "dummy": dummy,
            "linear": linear,
            "smooth_poly": smooth,
            "knn": local,
        },
    )
    for result in probes.values():
        result["stability_balanced_accuracy_std"] = 0.5
    paired = build_paired_probe_comparisons(
        probes,
        y,
        target_mode="singlelabel",
        requested_resamples=200,
        random_state=0,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    metrics = {
        "probes": probes,
        "paired_probe_comparisons": paired,
        "audit": {"class_counts": {"0": 100, "1": 100}, "n_classes": 2},
        "geometry": {"distance_concentration_proxy": 0.2},
        "neighborhood": {"cross_class_neighbor_fraction": 0.1},
        "graph": {"graph_fragmentation_score": 0.0},
        "topology": {},
        "boundary": {},
    }

    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, _, _ = make_recommendation(scores, metrics)
    comparison = metrics["recommendation_evidence"]["family_comparisons"]

    assert recommendation == "kernel_or_local_recommended"
    assert comparison["local_kernel_vs_smooth_nonlinear"]["decision_method"] == (
        "paired_oof_bootstrap"
    )


def test_fold_assignments_are_pruned_only_from_terse_report() -> None:
    X = np.arange(240, dtype=float).reshape(120, 2)
    y = np.asarray([0, 1] * 60)
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="fast",
        topology="off",
        random_state=0,
    )

    assert "fold_assignments" not in report.to_dict()["metrics"]["probe_evaluation"]
    assert "fold_assignments" in report.to_dict(terse=False)["metrics"][
        "probe_evaluation"
    ]
