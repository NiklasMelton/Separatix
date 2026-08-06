"""Paired resampling utilities for aligned probe predictions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from itertools import combinations
from typing import Any, Literal, cast

import numpy as np

from separatix.models.scoring import (
    summarize_multilabel_predictions,
    summarize_predictions,
    summarize_regression_predictions,
)
from separatix.utils.random import make_rng

TargetMode = Literal["singlelabel", "multilabel", "regression"]


def bootstrap_indices(
    n_rows: int,
    *,
    repeats: int,
    random_state: int | None,
    groups: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return ordinary or whole-group bootstrap index sets."""
    rng = make_rng(random_state)
    if groups is None:
        return [
            np.sort(rng.choice(n_rows, size=n_rows, replace=True)).astype(int)
            for _ in range(repeats)
        ]
    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group_id) for group_id in unique_groups]
    samples: list[np.ndarray] = []
    for _ in range(repeats):
        chosen = rng.choice(len(group_rows), size=len(group_rows), replace=True)
        samples.append(
            np.sort(np.concatenate([group_rows[int(index)] for index in chosen]))
        )
    return samples


def bootstrap_comparison(
    delta_fn: Callable[[np.ndarray], float],
    *,
    repeats: int,
    random_state: int | None,
    n_rows: int,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """Return paired percentile-bootstrap delta summaries."""
    deltas = np.asarray(
        [
            delta_fn(sample_idx)
            for sample_idx in bootstrap_indices(
                n_rows,
                repeats=repeats,
                random_state=random_state,
                groups=groups,
            )
        ],
        dtype=float,
    )
    if deltas.size == 0:
        return {"mean_delta": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    return {
        "mean_delta": float(np.mean(deltas)),
        "lower_95": float(np.percentile(deltas, 2.5)),
        "upper_95": float(np.percentile(deltas, 97.5)),
    }


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap within each class so balanced accuracy remains defined."""
    sampled = [
        rng.choice(rows, size=rows.size, replace=True)
        for cls in np.unique(y)
        if (rows := np.flatnonzero(y == cls)).size
    ]
    return np.sort(np.concatenate(sampled)).astype(int)


def _resample_stream(
    y: np.ndarray,
    *,
    target_mode: TargetMode,
    requested: int,
    random_state: int | None,
    groups: np.ndarray | None,
) -> Iterator[np.ndarray]:
    """Yield target-appropriate paired bootstrap samples."""
    rng = make_rng(random_state)
    n_rows = int(y.shape[0])
    if groups is None:
        for _ in range(requested):
            if target_mode == "singlelabel":
                yield _stratified_indices(np.asarray(y), rng)
            else:
                yield np.sort(rng.choice(n_rows, size=n_rows, replace=True)).astype(int)
        return

    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group_id) for group_id in unique_groups]
    expected_classes = np.unique(y) if target_mode == "singlelabel" else None
    attempts = 0
    accepted = 0
    while accepted < requested and attempts < requested * 20:
        attempts += 1
        chosen = rng.choice(len(group_rows), size=len(group_rows), replace=True)
        sampled = np.sort(
            np.concatenate([group_rows[int(index)] for index in chosen])
        ).astype(int)
        if expected_classes is not None and not np.array_equal(
            np.unique(y[sampled]), expected_classes
        ):
            continue
        accepted += 1
        yield sampled


def _metric_names(target_mode: TargetMode) -> tuple[str, ...]:
    if target_mode == "singlelabel":
        return ("balanced_accuracy",)
    if target_mode == "multilabel":
        return ("micro_f1", "macro_f1", "sample_jaccard")
    return ("r2_variance_weighted", "r2_uniform_average")


def _score_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
    *,
    target_mode: TargetMode,
    names: np.ndarray | None,
) -> dict[str, float]:
    """Score one probe on one paired resample."""
    if target_mode == "singlelabel":
        summary = summarize_predictions(y_true[indices], predictions[indices])
    elif target_mode == "multilabel":
        if names is None:
            raise ValueError("Multilabel paired comparisons require label names.")
        summary = summarize_multilabel_predictions(
            y_true[indices], predictions[indices], label_names=names
        )
    else:
        if names is None:
            raise ValueError("Regression paired comparisons require target names.")
        summary = summarize_regression_predictions(
            y_true[indices], predictions[indices], target_names=names
        )
    return {
        metric: float(cast(Any, summary[metric]))
        for metric in _metric_names(target_mode)
    }


def build_paired_probe_comparisons(
    probes: dict[str, dict[str, Any]],
    y_true: np.ndarray,
    *,
    target_mode: TargetMode,
    requested_resamples: int,
    random_state: int | None,
    evaluation_plan_id: str,
    evaluation_available: bool,
    groups: np.ndarray | None = None,
    names: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build all pairwise metric-delta intervals from aligned OOF predictions."""
    base = {
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

    prediction_arrays: dict[str, np.ndarray] = {}
    for name, result in probes.items():
        predictions = result.get("predictions")
        if (
            predictions is None
            or result.get("evaluation_plan_id") != evaluation_plan_id
        ):
            continue
        array = np.asarray(predictions)
        if array.shape[0] == y_true.shape[0]:
            prediction_arrays[name] = array
    if len(prediction_arrays) < 2:
        return {
            **base,
            "reason": "fewer than two aligned prediction arrays are available",
        }

    ordered_names = [
        name
        for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        if name in prediction_arrays
    ]
    pairs = list(combinations(ordered_names, 2))
    metric_names = _metric_names(target_mode)
    deltas: dict[tuple[str, str, str], list[float]] = {
        (first, second, metric): []
        for first, second in pairs
        for metric in metric_names
    }
    resample_digest = hashlib.sha256()
    used = 0
    for indices in _resample_stream(
        np.asarray(y_true),
        target_mode=target_mode,
        requested=requested_resamples,
        random_state=random_state,
        groups=groups,
    ):
        try:
            scores = {
                probe_name: _score_predictions(
                    np.asarray(y_true),
                    predictions,
                    indices,
                    target_mode=target_mode,
                    names=names,
                )
                for probe_name, predictions in prediction_arrays.items()
            }
        except (TypeError, ValueError, IndexError):
            continue
        if not all(
            np.isfinite(value)
            for probe_scores in scores.values()
            for value in probe_scores.values()
        ):
            continue
        resample_digest.update(indices.tobytes())
        used += 1
        for first, second in pairs:
            for metric in metric_names:
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
        for metric in metric_names:
            values = np.asarray(deltas[(first, second, metric)], dtype=float)
            point_delta = float(probes[first][metric]) - float(probes[second][metric])
            metric_payload[metric] = {
                "point_delta": point_delta,
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
        "resamples_used": int(used),
        "resample_plan_id": resample_digest.hexdigest()[:16],
        "comparisons": comparisons,
    }


def lookup_paired_comparison(
    payload: dict[str, Any],
    first_probe: str | None,
    second_probe: str | None,
    metric: str,
) -> dict[str, Any] | None:
    """Return a comparison oriented as ``first_probe - second_probe``."""
    if (
        payload.get("status") != "available"
        or first_probe is None
        or second_probe is None
    ):
        return None
    direct = payload.get("comparisons", {}).get(f"{first_probe}__vs__{second_probe}")
    if direct is not None:
        item = direct.get("metrics", {}).get(metric)
        return None if item is None else dict(item)
    reverse = payload.get("comparisons", {}).get(f"{second_probe}__vs__{first_probe}")
    if reverse is None:
        return None
    item = reverse.get("metrics", {}).get(metric)
    if item is None:
        return None
    return {
        **item,
        "point_delta": -float(item["point_delta"]),
        "mean_delta": -float(item["mean_delta"]),
        "lower_95": -float(item["upper_95"]),
        "upper_95": -float(item["lower_95"]),
    }
