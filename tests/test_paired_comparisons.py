from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from separatix import ComplexityProfiler, diagnose
from separatix.models import comparison as comparison_module
from separatix.models import mlp as mlp_module
from separatix.models import scoring as scoring_module
from separatix.models.comparison import (
    _resample_stream,
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
    _primary_metric_score_tensor,
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


_PRIMARY_METRICS = {
    "singlelabel": ("balanced_accuracy",),
    "multilabel": ("micro_f1", "macro_f1", "sample_jaccard"),
    "regression": ("r2_variance_weighted", "r2_uniform_average"),
}
_ORDERED_PROBES = ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")


def _primary_probe_results(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    target_mode: str,
    names: np.ndarray | None = None,
    evaluation_plan_id: str = "plan",
) -> dict[str, dict[str, object]]:
    """Build aligned probe payloads with only the configured primary metrics."""
    metrics = _PRIMARY_METRICS[target_mode]
    return {
        name: {
            **primary_metric_scores(
                y_true,
                values,
                target_mode=target_mode,  # type: ignore[arg-type]
                metrics=metrics,
                names=names,
            ),
            "predictions": values.tolist(),
            "evaluation_plan_id": evaluation_plan_id,
        }
        for name, values in predictions.items()
    }


def _scalar_reference_payload(
    probes: dict[str, dict[str, Any]],
    y_true: np.ndarray,
    *,
    target_mode: str,
    requested_resamples: int,
    random_state: int | None,
    evaluation_plan_id: str,
    evaluation_available: bool,
    groups: np.ndarray | None = None,
    names: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reference implementation of the pre-tensor paired bootstrap path."""
    base: dict[str, Any] = {
        "status": "unavailable",
        "method": "paired_oof_bootstrap",
        "evaluation_plan_id": evaluation_plan_id,
        "resamples_requested": int(requested_resamples),
        "resamples_used": 0,
        "resample_plan_id": None,
        "comparisons": {},
    }
    if not evaluation_available:
        return {**base, "reason": "held-out aligned predictions are unavailable"}

    n_rows = int(y_true.shape[0])
    prediction_arrays: dict[str, np.ndarray] = {}
    for name, result in probes.items():
        predictions = result.get("predictions")
        if (
            predictions is None
            or result.get("evaluation_plan_id") != evaluation_plan_id
        ):
            continue
        array = np.asarray(predictions)
        if array.shape[0] == n_rows:
            prediction_arrays[name] = array
    if len(prediction_arrays) < 2:
        return {
            **base,
            "reason": "fewer than two aligned prediction arrays are available",
        }

    ordered_names = [name for name in _ORDERED_PROBES if name in prediction_arrays]
    pairs = list(combinations(ordered_names, 2))
    metrics = _PRIMARY_METRICS[target_mode]
    deltas: dict[tuple[str, str, str], list[float]] = {
        (first, second, metric): [] for first, second in pairs for metric in metrics
    }
    digest = hashlib.sha256()
    used = 0
    for indices in _resample_stream(
        np.asarray(y_true),
        target_mode=target_mode,  # type: ignore[arg-type]
        requested=requested_resamples,
        random_state=random_state,
        groups=groups,
    ):
        try:
            scores = {
                name: primary_metric_scores(
                    np.asarray(y_true)[indices],
                    predictions[indices],
                    target_mode=target_mode,  # type: ignore[arg-type]
                    metrics=metrics,
                    names=names,
                )
                for name, predictions in prediction_arrays.items()
            }
        except (TypeError, ValueError, IndexError):
            continue
        if not all(
            np.isfinite(value) for score in scores.values() for value in score.values()
        ):
            continue
        digest.update(indices.tobytes())
        used += 1
        for first, second in pairs:
            for metric in metrics:
                deltas[(first, second, metric)].append(
                    scores[first][metric] - scores[second][metric]
                )

    minimum = max(50, requested_resamples // 2)
    if used < minimum:
        return {
            **base,
            "resamples_used": used,
            "reason": "too few valid paired bootstrap resamples",
        }

    comparisons: dict[str, Any] = {}
    for first, second in pairs:
        metric_payload: dict[str, Any] = {}
        for metric in metrics:
            values = np.asarray(deltas[(first, second, metric)], dtype=float)
            metric_payload[metric] = {
                "point_delta": float(probes[first][metric])
                - float(probes[second][metric]),
                "mean_delta": float(np.mean(values)),
                "paired_standard_error": float(np.std(values, ddof=1)),
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
                "resamples_requested": int(requested_resamples),
                "resamples_used": int(used),
            }
        comparisons[f"{first}__vs__{second}"] = {
            "first_probe": first,
            "second_probe": second,
            "metrics": metric_payload,
        }
    return {
        **base,
        "status": "available",
        "reason": None,
        "resamples_used": used,
        "resample_plan_id": digest.hexdigest()[:16],
        "comparisons": comparisons,
    }


def _weights_from_resamples(resamples: list[np.ndarray], n_rows: int) -> np.ndarray:
    """Convert bootstrap index rows into the kernel's multiplicity weights."""
    weights = np.zeros((len(resamples), n_rows), dtype=np.float64)
    for row, indices in enumerate(resamples):
        np.add.at(weights[row], indices, 1.0)
    return weights


def _assert_payload_equal_with_tight_float_tolerance(
    actual: Any, expected: Any
) -> None:
    """Assert payload schema exactly while allowing only roundoff in floats."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_payload_equal_with_tight_float_tolerance(actual[key], expected[key])
        return
    if isinstance(expected, float):
        assert isinstance(actual, (float, int))
        assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_payload_equal_with_tight_float_tolerance(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.parametrize(
    ("target_mode", "y_true", "predictions", "names"),
    [
        (
            "singlelabel",
            np.asarray([0, 1, 0, 1, 2, 2] * 4),
            [
                np.asarray([0, 0, 0, 1, 2, 1] * 4),
                np.asarray([0, 1, 0, 1, 2, 2] * 4),
            ],
            None,
        ),
        (
            "singlelabel",
            np.asarray(["left", "right", "middle"] * 8),
            [
                np.asarray(["left", "left", "middle"] * 8),
                np.asarray(["left", "right", "middle"] * 8),
            ],
            None,
        ),
        (
            "multilabel",
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 6, dtype=int),
            [
                np.asarray([[0, 0], [0, 0], [0, 1], [1, 0]] * 6, dtype=int),
                np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 6, dtype=int),
            ],
            np.asarray(["a", "b"]),
        ),
        (
            "regression",
            np.asarray([[0.0, 2.0], [1.0, 2.5], [2.0, 3.0]] * 8),
            [
                np.asarray([[0.2, 2.1], [0.8, 2.3], [2.1, 2.9]] * 8),
                np.asarray([[0.0, 2.0], [1.0, 2.5], [2.0, 3.0]] * 8),
            ],
            np.asarray(["x", "y"]),
        ),
    ],
)
def test_primary_metric_score_tensor_matches_scalar_resamples(
    target_mode: str,
    y_true: np.ndarray,
    predictions: list[np.ndarray],
    names: np.ndarray | None,
) -> None:
    """The weighted tensor kernel matches scalar scoring per resample/probe."""
    metrics = _PRIMARY_METRICS[target_mode]
    resamples = list(
        _resample_stream(
            y_true,
            target_mode=target_mode,  # type: ignore[arg-type]
            requested=9,
            random_state=59,
            groups=None,
        )
    )
    weights = _weights_from_resamples(resamples, y_true.shape[0])
    tensor = _primary_metric_score_tensor(
        y_true,
        predictions,
        weights,
        target_mode=target_mode,  # type: ignore[arg-type]
        metrics=metrics,
        names=names,
    )
    mapping = {f"probe-{index}": value for index, value in enumerate(predictions)}
    mapped_tensor = _primary_metric_score_tensor(
        y_true,
        mapping,
        weights,
        target_mode=target_mode,  # type: ignore[arg-type]
        metrics=metrics,
        names=names,
    )

    assert tensor.dtype == np.float64
    assert tensor.shape == (len(resamples), len(predictions), len(metrics))
    np.testing.assert_allclose(tensor, mapped_tensor, rtol=0.0, atol=0.0)
    for resample_index, indices in enumerate(resamples):
        for probe_index, prediction in enumerate(predictions):
            expected = primary_metric_scores(
                y_true[indices],
                prediction[indices],
                target_mode=target_mode,  # type: ignore[arg-type]
                metrics=metrics,
                names=names,
            )
            for metric_index, metric in enumerate(metrics):
                assert tensor[
                    resample_index, probe_index, metric_index
                ] == pytest.approx(expected[metric], rel=1e-12, abs=1e-12)


def test_primary_metric_score_tensor_validates_shapes_weights_and_names() -> None:
    """Kernel input validation remains explicit for malformed aligned data."""
    y_true = np.asarray([0, 1, 0, 1])
    predictions = [y_true.copy(), np.zeros_like(y_true)]
    weights = np.eye(4, dtype=float)

    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            np.ones(4),
            target_mode="singlelabel",
            metrics=("balanced_accuracy",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            np.ones((4, 3)),
            target_mode="singlelabel",
            metrics=("balanced_accuracy",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            [y_true[:-1], predictions[1]],
            weights,
            target_mode="singlelabel",
            metrics=("balanced_accuracy",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            np.asarray([[1.0, np.nan, 0.0, 0.0]]),
            target_mode="singlelabel",
            metrics=("balanced_accuracy",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            np.asarray([[1.0, -1.0, 0.0, 0.0]]),
            target_mode="singlelabel",
            metrics=("balanced_accuracy",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            weights,
            target_mode="singlelabel",
            metrics=("unknown_metric",),
        )
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            y_true,
            predictions,
            weights,
            target_mode="unknown",  # type: ignore[arg-type]
            metrics=("balanced_accuracy",),
        )

    multilabel = np.column_stack([y_true, 1 - y_true])
    with pytest.raises(ValueError):
        _primary_metric_score_tensor(
            multilabel,
            [multilabel],
            np.ones((1, 4)),
            target_mode="multilabel",
            metrics=_PRIMARY_METRICS["multilabel"],
        )
    with pytest.raises(IndexError):
        _primary_metric_score_tensor(
            multilabel,
            [multilabel],
            np.ones((1, 4)),
            target_mode="multilabel",
            metrics=_PRIMARY_METRICS["multilabel"],
            names=np.asarray(["one"]),
        )


def test_builder_falls_back_to_scalar_scoring_when_tensor_kernel_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tensor-kernel failure does not discard otherwise valid paired evidence."""
    y_true = np.asarray([0, 1] * 15)
    probes = _primary_probe_results(
        y_true,
        {"dummy": np.zeros_like(y_true), "linear": y_true.copy()},
        target_mode="singlelabel",
    )
    expected = _scalar_reference_payload(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=61,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )

    def fail_kernel(*args: Any, **kwargs: Any) -> np.ndarray:
        raise ValueError("forced tensor failure")

    monkeypatch.setattr(comparison_module, "_primary_metric_score_tensor", fail_kernel)
    actual = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=61,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )

    _assert_payload_equal_with_tight_float_tolerance(actual, expected)


@pytest.mark.parametrize("target_mode", ["singlelabel", "multilabel", "regression"])
def test_tiny_and_large_memory_chunks_are_deterministically_equivalent(
    target_mode: str,
) -> None:
    """Changing tensor chunk size preserves schema, IDs, and all float values."""
    if target_mode == "singlelabel":
        y_true = np.asarray([0, 1, 2] * 10)
        names = None
        predictions = {
            "dummy": np.zeros_like(y_true),
            "linear": y_true.copy(),
            "smooth_poly": np.roll(y_true, 1),
        }
    elif target_mode == "multilabel":
        y_true = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 8, dtype=int)
        names = np.asarray(["a", "b"])
        predictions = {
            "dummy": np.zeros_like(y_true),
            "linear": y_true.copy(),
            "smooth_poly": np.roll(y_true, 1, axis=0),
        }
    else:
        y_true = np.asarray([[0.0, 2.0], [1.0, 2.5], [2.0, 3.0]] * 10)
        names = np.asarray(["x", "y"])
        predictions = {
            "dummy": np.zeros_like(y_true),
            "linear": y_true.copy(),
            "smooth_poly": y_true + np.asarray([0.1, -0.1]),
        }
    probes = _primary_probe_results(
        y_true,
        predictions,
        target_mode=target_mode,
        names=names,
    )
    kwargs = {
        "target_mode": target_mode,
        "requested_resamples": 60,
        "random_state": 67,
        "evaluation_plan_id": "plan",
        "evaluation_available": True,
        "names": names,
    }
    tiny = build_paired_probe_comparisons(
        probes,
        y_true,
        max_working_memory_mb=0.001,
        **kwargs,  # type: ignore[arg-type]
    )
    large = build_paired_probe_comparisons(
        probes,
        y_true,
        max_working_memory_mb=128.0,
        **kwargs,  # type: ignore[arg-type]
    )

    _assert_payload_equal_with_tight_float_tolerance(tiny, large)
    assert tiny["resample_plan_id"] == large["resample_plan_id"]


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


def test_builder_matches_scalar_reference_for_primary_target_modes() -> None:
    """Tensor-backed paired scoring preserves scalar results and pair order."""
    binary = np.asarray([0, 1] * 15)
    multiclass = np.asarray([0, 1, 2] * 10)
    string_labels = np.asarray(["left", "right"] * 15)
    multilabel = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]] * 6,
        dtype=int,
    )
    regression = np.asarray(
        [[1_000_000.0, 2.0], [1_000_000.0, 2.0], [1_000_001.0, 3.0]] * 10
    )
    cases = [
        (
            "singlelabel",
            binary,
            {
                "dummy": np.zeros_like(binary),
                "linear": binary.copy(),
                "smooth_poly": np.roll(binary, 1),
            },
            None,
        ),
        (
            "singlelabel",
            multiclass,
            {
                "dummy": np.zeros_like(multiclass),
                "linear": multiclass.copy(),
                "smooth_poly": np.roll(multiclass, 1),
            },
            None,
        ),
        (
            "singlelabel",
            string_labels,
            {
                "dummy": np.full_like(string_labels, "left"),
                "linear": string_labels.copy(),
                "smooth_poly": np.roll(string_labels, 1),
            },
            None,
        ),
        (
            "multilabel",
            multilabel,
            {
                "dummy": np.zeros_like(multilabel),
                "linear": multilabel.copy(),
                "smooth_poly": np.roll(multilabel, 1, axis=0),
            },
            np.asarray(["first", "second"]),
        ),
        (
            "regression",
            regression,
            {
                "dummy": np.tile(np.asarray([[1_000_000.2, 2.5]]), (30, 1)),
                "linear": regression.copy(),
                "smooth_poly": regression + np.asarray([0.1, -0.1]),
            },
            np.asarray(["offset", "small_scale"]),
        ),
    ]

    for target_mode, y_true, predictions, names in cases:
        probes = _primary_probe_results(
            y_true,
            predictions,
            target_mode=target_mode,
            names=names,
        )
        actual = build_paired_probe_comparisons(
            probes,
            y_true,
            target_mode=target_mode,  # type: ignore[arg-type]
            requested_resamples=60,
            random_state=17,
            evaluation_plan_id="plan",
            evaluation_available=True,
            names=names,
        )
        expected = _scalar_reference_payload(
            probes,
            y_true,
            target_mode=target_mode,
            requested_resamples=60,
            random_state=17,
            evaluation_plan_id="plan",
            evaluation_available=True,
            names=names,
        )
        _assert_payload_equal_with_tight_float_tolerance(actual, expected)
        assert list(actual["comparisons"]) == [
            "dummy__vs__linear",
            "dummy__vs__smooth_poly",
            "linear__vs__smooth_poly",
        ]


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


def test_builder_matches_scalar_reference_for_unequal_whole_group_bootstrap() -> None:
    """Group-aware tensor scoring keeps unequal groups and exact resample IDs."""
    groups = np.repeat(np.arange(6), [2, 3, 4, 5, 6, 7])
    y_true = np.concatenate(
        [
            np.asarray([0, 1]),
            np.asarray([0, 1, 1]),
            np.asarray([0, 1, 0, 1]),
            np.asarray([1, 0, 1, 0, 1]),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.asarray([1, 0, 1, 0, 1, 0, 1]),
        ]
    )
    predictions = {
        "dummy": np.zeros_like(y_true),
        "linear": y_true.copy(),
    }
    probes = _primary_probe_results(y_true, predictions, target_mode="singlelabel")
    actual = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=23,
        evaluation_plan_id="plan",
        evaluation_available=True,
        groups=groups,
    )
    expected = _scalar_reference_payload(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=23,
        evaluation_plan_id="plan",
        evaluation_available=True,
        groups=groups,
    )

    _assert_payload_equal_with_tight_float_tolerance(actual, expected)
    assert actual["status"] == "available"
    assert actual["resamples_used"] == 60
    assert actual["resample_plan_id"] is not None


def test_group_class_rejection_matches_scalar_reference() -> None:
    """Group draws missing a class are rejected identically by both paths."""
    groups = np.repeat(np.arange(4), 3)
    y_true = np.repeat(np.asarray([0, 1, 2, 0]), 3)
    predictions = {
        "dummy": np.zeros_like(y_true),
        "linear": y_true.copy(),
    }
    probes = _primary_probe_results(y_true, predictions, target_mode="singlelabel")
    actual = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=75,
        random_state=29,
        evaluation_plan_id="plan",
        evaluation_available=True,
        groups=groups,
    )
    expected = _scalar_reference_payload(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=75,
        random_state=29,
        evaluation_plan_id="plan",
        evaluation_available=True,
        groups=groups,
    )

    _assert_payload_equal_with_tight_float_tolerance(actual, expected)
    assert actual["resamples_used"] == actual["resamples_requested"]


@pytest.mark.parametrize(
    ("Y_true", "Y_pred"),
    [
        (np.zeros((24, 1), dtype=int), np.zeros((24, 1), dtype=int)),
        (np.ones((24, 3), dtype=int), np.ones((24, 3), dtype=int)),
        (
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 6, dtype=int),
            np.asarray([[0, 0], [1, 0], [0, 0], [1, 1]] * 6, dtype=int),
        ),
    ],
)
def test_multilabel_edge_payloads_match_scalar_reference(
    Y_true: np.ndarray, Y_pred: np.ndarray
) -> None:
    """Empty and constant multilabel cohorts retain finite paired intervals."""
    names = np.asarray([f"label-{index}" for index in range(Y_true.shape[1])])
    probes = _primary_probe_results(
        Y_true,
        {"dummy": np.zeros_like(Y_true), "linear": Y_pred},
        target_mode="multilabel",
        names=names,
    )
    actual = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="multilabel",
        requested_resamples=60,
        random_state=31,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )
    expected = _scalar_reference_payload(
        probes,
        Y_true,
        target_mode="multilabel",
        requested_resamples=60,
        random_state=31,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )

    _assert_payload_equal_with_tight_float_tolerance(actual, expected)
    assert actual["status"] == "available"
    for metric in _PRIMARY_METRICS["multilabel"]:
        comparison = lookup_paired_comparison(actual, "linear", "dummy", metric)
        assert comparison is not None
        assert np.isfinite(float(comparison["point_delta"]))


@pytest.mark.parametrize(
    ("Y_true", "Y_pred"),
    [
        (np.full((24, 1), 7.0), np.full((24, 1), 7.0)),
        (
            1_000_000.0 + np.asarray([[0.0], [1e-7], [0.0], [1e-7]] * 6),
            1_000_000.0 + np.asarray([[0.0], [0.9e-7], [0.1e-7], [1.1e-7]] * 6),
        ),
        (
            1_000_000_000.0
            + np.asarray([[0.0, 0.0], [3e-6, 1.0], [0.0, 2.0], [3e-6, 3.0]] * 6),
            1_000_000_000.0
            + np.asarray([[0.1, 0.0], [2.8e-6, 1.1], [0.2, 1.9], [3.1e-6, 3.1]] * 6),
        ),
    ],
)
def test_regression_edge_payloads_match_scalar_reference(
    Y_true: np.ndarray, Y_pred: np.ndarray
) -> None:
    """Constant, near-variance-threshold, and offset regression remain stable."""
    names = np.asarray([f"target-{index}" for index in range(Y_true.shape[1])])
    probes = _primary_probe_results(
        Y_true,
        {"dummy": np.zeros_like(Y_true), "linear": Y_pred},
        target_mode="regression",
        names=names,
    )
    actual = build_paired_probe_comparisons(
        probes,
        Y_true,
        target_mode="regression",
        requested_resamples=60,
        random_state=37,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )
    expected = _scalar_reference_payload(
        probes,
        Y_true,
        target_mode="regression",
        requested_resamples=60,
        random_state=37,
        evaluation_plan_id="plan",
        evaluation_available=True,
        names=names,
    )

    _assert_payload_equal_with_tight_float_tolerance(actual, expected)
    assert actual["status"] == "available"


def test_extra_aligned_unrecognized_probe_has_no_pairwise_comparisons() -> None:
    """Unknown aligned probe names are scored but never become output pairs."""
    y_true = np.asarray([0, 1] * 15)
    probes = _primary_probe_results(
        y_true,
        {
            "dummy": np.zeros_like(y_true),
            "linear": y_true.copy(),
            "future_probe": np.roll(y_true, 1),
        },
        target_mode="singlelabel",
    )
    payload = build_paired_probe_comparisons(
        probes,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=41,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )

    assert payload["status"] == "available"
    assert list(payload["comparisons"]) == ["dummy__vs__linear"]
    assert all("future_probe" not in key for key in payload["comparisons"])


def test_unavailable_and_fewer_probe_payloads_keep_schema() -> None:
    """Unavailable evaluation and insufficient aligned probes stay explicit."""
    y_true = np.asarray([0, 1] * 15)
    probes = _primary_probe_results(
        y_true,
        {"dummy": np.zeros_like(y_true)},
        target_mode="singlelabel",
    )
    for evaluation_available, reason in (
        (False, "held-out aligned predictions are unavailable"),
        (True, "fewer than two aligned prediction arrays are available"),
    ):
        actual = build_paired_probe_comparisons(
            probes,
            y_true,
            target_mode="singlelabel",
            requested_resamples=60,
            random_state=43,
            evaluation_plan_id="plan",
            evaluation_available=evaluation_available,
        )
        expected = _scalar_reference_payload(
            probes,
            y_true,
            target_mode="singlelabel",
            requested_resamples=60,
            random_state=43,
            evaluation_plan_id="plan",
            evaluation_available=evaluation_available,
        )
        _assert_payload_equal_with_tight_float_tolerance(actual, expected)
        assert actual["status"] == "unavailable"
        assert actual["reason"] == reason


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_malformed_and_nonfinite_predictions_are_skipped_or_unavailable() -> None:
    """Bad aligned cohorts do not fabricate paired evidence."""
    y_true = np.asarray([0, 1] * 15)
    valid = _primary_probe_results(
        y_true,
        {"dummy": np.zeros_like(y_true), "linear": y_true.copy()},
        target_mode="singlelabel",
    )

    malformed = {name: dict(result) for name, result in valid.items()}
    malformed["linear"]["predictions"] = [0]
    malformed_payload = build_paired_probe_comparisons(
        malformed,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=47,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    assert malformed_payload["status"] == "unavailable"
    assert "fewer than two" in malformed_payload["reason"]

    nonfinite = {name: dict(result) for name, result in valid.items()}
    nonfinite["linear"]["predictions"] = [np.nan] * y_true.size
    nonfinite_payload = build_paired_probe_comparisons(
        nonfinite,
        y_true,
        target_mode="singlelabel",
        requested_resamples=60,
        random_state=47,
        evaluation_plan_id="plan",
        evaluation_available=True,
    )
    assert nonfinite_payload["status"] == "unavailable"
    assert "too few valid" in nonfinite_payload["reason"]


def test_short_or_missing_matrix_names_match_scalar_reference_failure() -> None:
    """Missing and short target names retain the unavailable comparison path."""
    Y_true = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]] * 6, dtype=int)
    probes = _primary_probe_results(
        Y_true,
        {"dummy": np.zeros_like(Y_true), "linear": Y_true.copy()},
        target_mode="multilabel",
        names=np.asarray(["a", "b"]),
    )
    for names in (None, np.asarray(["only-one"]), np.asarray([])):
        actual = build_paired_probe_comparisons(
            probes,
            Y_true,
            target_mode="multilabel",
            requested_resamples=60,
            random_state=53,
            evaluation_plan_id="plan",
            evaluation_available=True,
            names=names,
        )
        expected = _scalar_reference_payload(
            probes,
            Y_true,
            target_mode="multilabel",
            requested_resamples=60,
            random_state=53,
            evaluation_plan_id="plan",
            evaluation_available=True,
            names=names,
        )
        _assert_payload_equal_with_tight_float_tolerance(actual, expected)
        assert actual["status"] == "unavailable"
        assert actual["resamples_used"] == 0


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
