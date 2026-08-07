from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from separatix import ComplexityProfiler, diagnose
from separatix.models import mlp as mlp_module
from separatix.models import scoring as scoring_module
from separatix.models.comparison import (
    bootstrap_indices,
    build_paired_probe_comparisons,
    lookup_paired_comparison,
)
from separatix.models.mlp import (
    _balanced_accuracy_delta,
    _multilabel_metric_delta,
    _regression_metric_delta,
)
from separatix.models.scoring import (
    primary_metric_scores,
    summarize_multilabel_predictions,
    summarize_predictions,
    summarize_regression_predictions,
)
from separatix.recommendation.engine import compute_scores, make_recommendation


def _singlelabel_probe_results(
    y: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "balanced_accuracy": float(
                np.mean([np.mean(values[y == cls] == cls) for cls in np.unique(y)])
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
    comparison = lookup_paired_comparison(first, "linear", "dummy", "balanced_accuracy")

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


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (
            np.asarray([0, 1, 0, 1, 0, 1, 1, 0]),
            np.asarray([0, 1, 1, 1, 0, 0, 1, 0]),
        ),
        (
            np.asarray([0, 1, 2, 0, 1, 2, 2, 1, 0]),
            np.asarray([0, 2, 2, 1, 1, 2, 0, 1, 0]),
        ),
    ],
)
def test_primary_singlelabel_scores_match_full_summary(
    y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """The optimized single-label scorer preserves the full summary metric."""
    expected = summarize_predictions(y_true, y_pred)["balanced_accuracy"]

    all_metrics = primary_metric_scores(
        y_true,
        y_pred,
        target_mode="singlelabel",
    )
    selected = primary_metric_scores(
        y_true,
        y_pred,
        target_mode="singlelabel",
        metrics=("balanced_accuracy",),
    )

    assert all_metrics == selected
    assert all_metrics["balanced_accuracy"] == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("Y_true", "Y_pred"),
    [
        # One indicator column with no positive labels (empty label sets).
        (
            np.zeros((8, 1), dtype=int),
            np.zeros((8, 1), dtype=int),
        ),
        # One indicator column with both classes represented.
        (
            np.asarray([[0], [1], [0], [1], [1], [0], [1], [0]], dtype=int),
            np.asarray([[0], [1], [1], [0], [1], [0], [0], [0]], dtype=int),
        ),
        # Multiple columns where one label is constant and rows can be empty.
        (
            np.asarray(
                [[0, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 0]],
                dtype=int,
            ),
            np.asarray(
                [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
                dtype=int,
            ),
        ),
        # A constant positive multi-column target.
        (
            np.ones((7, 3), dtype=int),
            np.asarray(
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=int,
            ),
        ),
    ],
)
def test_primary_multilabel_scores_match_full_summary(
    Y_true: np.ndarray, Y_pred: np.ndarray
) -> None:
    """Primary multilabel scores agree for one- and multi-column edge cases."""
    names = np.asarray([f"label-{index}" for index in range(Y_true.shape[1])])
    expected = summarize_multilabel_predictions(
        Y_true,
        Y_pred,
        label_names=names,
    )

    all_metrics = primary_metric_scores(
        Y_true,
        Y_pred,
        target_mode="multilabel",
        names=names,
    )
    selected = primary_metric_scores(
        Y_true,
        Y_pred,
        target_mode="multilabel",
        metrics=("micro_f1", "macro_f1", "sample_jaccard"),
        names=names,
    )

    assert all_metrics == selected
    for metric in ("micro_f1", "macro_f1", "sample_jaccard"):
        assert all_metrics[metric] == pytest.approx(float(expected[metric]))


@pytest.mark.parametrize(
    ("Y_true", "Y_pred"),
    [
        (
            np.full((8, 1), 4.0),
            np.full((8, 1), 4.0),
        ),
        (
            np.asarray(
                [[0.0], [1.0], [1.0], [2.0], [2.0], [3.0], [3.0], [4.0]],
            ),
            np.asarray(
                [[0.0], [1.2], [0.8], [2.1], [1.7], [3.0], [3.4], [3.8]],
            ),
        ),
        # Duplicated rows and one constant target exercise multi-target handling.
        (
            np.asarray(
                [
                    [0.0, 2.0],
                    [0.0, 2.0],
                    [1.0, 2.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [3.0, 2.0],
                    [3.0, 2.0],
                ]
            ),
            np.asarray(
                [
                    [0.0, 2.0],
                    [0.2, 1.0],
                    [0.8, 2.0],
                    [1.1, 1.5],
                    [2.2, 2.0],
                    [1.8, 2.0],
                    [3.0, 2.0],
                    [3.2, 2.0],
                ]
            ),
        ),
    ],
)
def test_primary_regression_scores_match_full_summary(
    Y_true: np.ndarray, Y_pred: np.ndarray
) -> None:
    """Primary regression scores preserve constant and duplicated-target behavior."""
    names = np.asarray([f"target-{index}" for index in range(Y_true.shape[1])])
    expected = summarize_regression_predictions(
        Y_true,
        Y_pred,
        target_names=names,
    )

    all_metrics = primary_metric_scores(
        Y_true,
        Y_pred,
        target_mode="regression",
        names=names,
    )
    selected = primary_metric_scores(
        Y_true,
        Y_pred,
        target_mode="regression",
        metrics=("r2_variance_weighted", "r2_uniform_average"),
        names=names,
    )

    assert all_metrics == selected
    for metric in ("r2_variance_weighted", "r2_uniform_average"):
        assert all_metrics[metric] == pytest.approx(float(expected[metric]))


def test_primary_metric_scores_validate_target_mode_metrics_and_names() -> None:
    """Invalid modes, metrics, and target-name metadata fail clearly."""
    y = np.asarray([0, 1, 0, 1])
    Y = np.column_stack([y, 1 - y])
    names = np.asarray(["first", "second"])

    with pytest.raises(ValueError):
        primary_metric_scores(y, y, target_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        primary_metric_scores(
            y,
            y,
            target_mode="singlelabel",
            metrics=("not_a_metric",),
        )
    with pytest.raises(ValueError):
        primary_metric_scores(Y, Y, target_mode="multilabel")
    with pytest.raises(IndexError):
        primary_metric_scores(Y, Y, target_mode="multilabel", names=np.asarray(["one"]))
    with pytest.raises(IndexError):
        primary_metric_scores(
            Y.astype(float),
            Y.astype(float),
            target_mode="regression",
            names=np.asarray(["one"]),
        )
    # Correct metadata remains accepted for both matrix-valued target modes.
    assert set(primary_metric_scores(Y, Y, target_mode="multilabel", names=names)) == {
        "micro_f1",
        "macro_f1",
        "sample_jaccard",
    }


def test_mlp_singlelabel_delta_matches_primary_and_full_scorers() -> None:
    """The MLP single-label delta helper scores only its primary metric."""
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 1, 0])
    first_pred = np.asarray([0, 1, 1, 1, 0, 0, 1, 0])
    second_pred = np.asarray([0, 0, 0, 1, 1, 1, 0, 0])
    sample_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 1])
    expected = float(
        summarize_predictions(y_true[sample_idx], first_pred[sample_idx])[
            "balanced_accuracy"
        ]
    ) - float(
        summarize_predictions(y_true[sample_idx], second_pred[sample_idx])[
            "balanced_accuracy"
        ]
    )

    assert _balanced_accuracy_delta(
        y_true, first_pred, second_pred, sample_idx
    ) == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["micro_f1", "macro_f1", "sample_jaccard"])
def test_mlp_multilabel_delta_matches_primary_and_full_scorers(metric: str) -> None:
    """The MLP multilabel delta helper preserves each primary metric."""
    Y_true = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=int,
    )
    first_pred = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]],
        dtype=int,
    )
    second_pred = np.asarray(
        [[0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [1, 0], [0, 1], [0, 0]],
        dtype=int,
    )
    sample_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 3])
    names = np.asarray(["first", "second"])
    first = summarize_multilabel_predictions(
        Y_true[sample_idx], first_pred[sample_idx], label_names=names
    )
    second = summarize_multilabel_predictions(
        Y_true[sample_idx], second_pred[sample_idx], label_names=names
    )
    expected = float(first[metric]) - float(second[metric])

    assert _multilabel_metric_delta(
        Y_true,
        first_pred,
        second_pred,
        sample_idx,
        metric=metric,
        label_names=names,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["r2_variance_weighted", "r2_uniform_average"])
def test_mlp_regression_delta_matches_primary_and_full_scorers(metric: str) -> None:
    """The MLP regression delta helper preserves both primary R2 metrics."""
    Y_true = np.asarray(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 2.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]]
    )
    first_pred = np.asarray(
        [[0.0, 1.0], [0.1, 1.0], [0.9, 2.0], [1.1, 2.0], [2.0, 3.0], [2.1, 3.0]]
    )
    second_pred = np.asarray(
        [[0.5, 1.0], [0.5, 1.0], [0.5, 2.0], [0.5, 2.0], [0.5, 3.0], [0.5, 3.0]]
    )
    sample_idx = np.asarray([0, 1, 2, 3, 4, 5, 2])
    names = np.asarray(["varying", "constant-ish"])
    first = summarize_regression_predictions(
        Y_true[sample_idx], first_pred[sample_idx], target_names=names
    )
    second = summarize_regression_predictions(
        Y_true[sample_idx], second_pred[sample_idx], target_names=names
    )
    expected = float(first[metric]) - float(second[metric])

    assert _regression_metric_delta(
        Y_true,
        first_pred,
        second_pred,
        sample_idx,
        metric=metric,
        target_names=names,
    ) == pytest.approx(expected)


def test_primary_comparison_and_delta_paths_skip_full_summarizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimized paths do not construct detailed summary payloads."""

    def fail_summary(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("full summarizer should not run in primary scoring paths")

    for module in (mlp_module, scoring_module):
        monkeypatch.setattr(module, "summarize_predictions", fail_summary)
        monkeypatch.setattr(module, "summarize_multilabel_predictions", fail_summary)
        monkeypatch.setattr(module, "summarize_regression_predictions", fail_summary)

    y_true = np.asarray([0, 1, 0, 1] * 4)
    first_pred = y_true.copy()
    second_pred = np.zeros_like(y_true)
    score_first = primary_metric_scores(
        y_true,
        first_pred,
        target_mode="singlelabel",
    )
    score_second = primary_metric_scores(
        y_true,
        second_pred,
        target_mode="singlelabel",
    )
    probes = {
        name: {
            **scores,
            "predictions": predictions.tolist(),
            "evaluation_plan_id": "plan",
        }
        for name, predictions, scores in (
            ("dummy", second_pred, score_second),
            ("linear", first_pred, score_first),
        )
    }
    payload = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=4,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )

    assert payload["status"] == "available"
    sample_idx = np.arange(y_true.shape[0])
    assert _balanced_accuracy_delta(y_true, first_pred, second_pred, sample_idx) > 0.0


def test_multilabel_paired_payload_is_deterministic_with_primary_scores() -> None:
    """Aligned multilabel comparisons remain byte-for-byte deterministic."""
    Y_true = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1]] * 10,
        dtype=int,
    )
    first_pred = Y_true.copy()
    second_pred = np.zeros_like(Y_true)
    names = np.asarray(["left", "right"])
    probes = {
        name: {
            **{
                metric: float(
                    summarize_multilabel_predictions(
                        Y_true,
                        predictions,
                        label_names=names,
                    )[metric]
                )
                for metric in ("micro_f1", "macro_f1", "sample_jaccard")
            },
            "predictions": predictions.tolist(),
            "evaluation_plan_id": "plan",
        }
        for name, predictions in {
            "dummy": second_pred,
            "linear": first_pred,
        }.items()
    }

    first = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="multilabel",
        requested_resamples=75,
        random_state=12,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )
    second = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="multilabel",
        requested_resamples=75,
        random_state=12,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )

    assert first == second
    assert first["status"] == "available"
    assert first["resamples_used"] == 75


def test_regression_paired_payload_is_deterministic_with_primary_scores() -> None:
    """Aligned regression comparisons preserve deterministic paired resamples."""
    Y_true = np.asarray(
        [
            [0.0, 2.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, 2.0],
        ]
        * 5
    )
    first_pred = Y_true.copy()
    second_pred = np.column_stack(
        [np.full(Y_true.shape[0], 1.5), np.full(Y_true.shape[0], 2.0)]
    )
    names = np.asarray(["varying", "constant"])
    probes = {
        name: {
            **{
                metric: float(
                    summarize_regression_predictions(
                        Y_true,
                        predictions,
                        target_names=names,
                    )[metric]
                )
                for metric in ("r2_variance_weighted", "r2_uniform_average")
            },
            "predictions": predictions.tolist(),
            "evaluation_plan_id": "plan",
        }
        for name, predictions in {
            "dummy": second_pred,
            "linear": first_pred,
        }.items()
    }

    first = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="regression",
        requested_resamples=75,
        random_state=12,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )
    second = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="regression",
        requested_resamples=75,
        random_state=12,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )

    assert first == second
    assert first["status"] == "available"
    assert first["resamples_used"] == 75


@pytest.mark.parametrize("target_mode", ["multilabel", "regression"])
def test_matrix_paired_comparisons_require_target_names(target_mode: str) -> None:
    """Missing matrix-target names retain the unavailable comparison path."""
    if target_mode == "multilabel":
        y_true = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 4, dtype=int)
        pred_a = y_true.copy()
        pred_b = np.zeros_like(y_true)
        metrics = ("micro_f1", "macro_f1", "sample_jaccard")
        summary = summarize_multilabel_predictions(
            y_true, pred_a, label_names=np.asarray(["a", "b"])
        )
    else:
        y_true = np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]] * 4)
        pred_a = y_true.copy()
        pred_b = np.zeros_like(y_true)
        metrics = ("r2_variance_weighted", "r2_uniform_average")
        summary = summarize_regression_predictions(
            y_true, pred_a, target_names=np.asarray(["a", "b"])
        )
    probes = {
        name: {
            **{metric: float(summary[metric]) for metric in metrics},
            "predictions": predictions.tolist(),
            "evaluation_plan_id": "plan",
        }
        for name, predictions in {"dummy": pred_b, "linear": pred_a}.items()
    }

    payload = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode=target_mode,  # type: ignore[arg-type]
        requested_resamples=60,
        random_state=1,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )

    assert payload["status"] == "unavailable"
    assert payload["resamples_used"] == 0
    assert "too few valid" in payload["reason"]


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
    assert {result["evaluation_plan_id"] for result in available} == {
        evaluation["evaluation_plan_id"]
    }
    assert {tuple(result["sample_info"]["indices"]) for result in available} == {
        tuple(evaluation["row_indices"])
    }
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
    Y_regression = np.column_stack([X[:, 0] + 0.1 * rng.normal(size=90), X[:, 1] ** 2])

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
    assert {tuple(result["sample_info"]["indices"]) for result in available} == {
        tuple(report.metrics["probe_evaluation"]["row_indices"])
    }


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
    assert (
        "fold_assignments" in report.to_dict(terse=False)["metrics"]["probe_evaluation"]
    )
