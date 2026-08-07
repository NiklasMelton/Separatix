"""Paired resampling utilities for aligned probe predictions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np

from separatix.models.scoring import (
    _primary_metric_score_tensor,
    primary_metric_scores,
)
from separatix.utils.random import make_rng

TargetMode = Literal["singlelabel", "multilabel", "regression"]

_TENSOR_CHUNK_CAP_BYTES = 32 * 1024**2
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


@dataclass
class _ResampleBatch:
    """Hold one chunk of accepted bootstrap indices and row weights."""

    indices: tuple[np.ndarray, ...]
    weights: np.ndarray | None


@dataclass
class _PairedScoreCache:
    """Finite primary scores from one paired bootstrap resample plan.

    ``scores`` is a dense ``[resample, probe, metric]`` tensor containing only
    accepted rows.  Keeping the accepted rows (rather than paired deltas) lets
    callers compare any two probes included in the cache without drawing
    another bootstrap plan.
    """

    scores: np.ndarray
    probe_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    status: str
    reason: str | None
    resamples_requested: int
    resamples_used: int
    resample_plan_id: str | None

    @property
    def requested(self) -> int:
        """Return the requested number of paired bootstrap resamples."""
        return self.resamples_requested

    @property
    def used(self) -> int:
        """Return the number of finite paired bootstrap rows retained."""
        return self.resamples_used

    @property
    def plan_id(self) -> str | None:
        """Return the stable identifier for the accepted resample plan."""
        return self.resample_plan_id


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


def _tensor_memory_budget_bytes(max_working_memory_mb: float | None) -> int:
    """Return the bounded memory budget for one tensor-score chunk."""
    if max_working_memory_mb is None:
        return _TENSOR_CHUNK_CAP_BYTES
    try:
        value = float(max_working_memory_mb)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not np.isfinite(value) or value <= 0.0:
        return 0
    if value >= (_TENSOR_CHUNK_CAP_BYTES / 1024**2):
        return _TENSOR_CHUNK_CAP_BYTES
    return max(0, int(value * 1024**2))


def _row_value_bytes(array: np.ndarray) -> int:
    """Estimate bytes occupied by one row of an aligned prediction array."""
    trailing = int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1
    return max(1, trailing) * max(1, int(array.dtype.itemsize))


def _resample_work_bytes(
    indices: np.ndarray,
    *,
    n_rows: int,
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    n_metrics: int,
) -> int:
    """Conservatively estimate one resample's tensor working-set bytes."""
    index_bytes = max(1, int(np.asarray(indices).size)) * np.dtype(np.int64).itemsize
    weight_bytes = max(1, n_rows) * _FLOAT64_BYTES
    target_bytes = _row_value_bytes(y_true)
    output_bytes = sum(_row_value_bytes(array) for array in predictions.values())
    score_bytes = max(1, len(predictions)) * max(1, n_metrics) * _FLOAT64_BYTES
    # Reserve one source-sized copy and one weighted intermediate for both the
    # target and every probe output, plus the score tensor row itself.
    return int(
        index_bytes
        + weight_bytes
        + (2 * target_bytes)
        + (2 * output_bytes)
        + score_bytes
    )


def _canonical_tensor_inputs(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    target_mode: TargetMode,
    names: np.ndarray | None,
) -> bool:
    """Return whether aligned data is safe for the vectorized scoring path."""
    if target_mode == "singlelabel":
        if y_true.ndim != 1:
            return False
    elif target_mode == "multilabel":
        if y_true.ndim != 2 or names is None or len(names) < y_true.shape[1]:
            return False
    elif target_mode == "regression":
        if y_true.ndim not in {1, 2} or names is None:
            return False
        n_targets = 1 if y_true.ndim == 1 else y_true.shape[1]
        if len(names) < n_targets:
            return False
    else:
        return False
    arrays = (y_true, *predictions.values())
    for array in arrays:
        if array.dtype.kind not in "biuf":
            return False
        try:
            if not bool(np.all(np.isfinite(array))):
                return False
        except (TypeError, ValueError):
            return False
    if target_mode == "regression":
        # Weighted R² uses centered second moments.  Near the report scorer's
        # variance usability cutoff, or at extreme target scales, those
        # moments can lose enough precision to change the paired interval;
        # retain the scalar scorer as the numerically safe compatibility path.
        regression_values = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
        variances = np.var(regression_values, axis=0)
        positive_variances = variances > 0.0
        if np.any(positive_variances & (variances <= 1e-10)):
            return False
        if np.any(np.abs(regression_values) > 1e12):
            return False
    expected_shape = y_true.shape
    for array in predictions.values():
        if target_mode == "singlelabel":
            if array.ndim != 1 or array.shape != expected_shape:
                return False
        elif array.shape != expected_shape:
            return False
    return True


def _build_resample_batch(
    indices: list[np.ndarray],
    *,
    n_rows: int,
) -> _ResampleBatch:
    """Build one tensor batch's multiplicity-weight matrix."""
    weights = np.zeros((len(indices), n_rows), dtype=np.float64)
    for row, sample_idx in enumerate(indices):
        values = np.asarray(sample_idx)
        if values.ndim != 1 or values.dtype.kind not in "biuf":
            raise ValueError(
                "Bootstrap indices must be one-dimensional numeric arrays."
            )
        if values.size and (
            np.any(values < 0)
            or np.any(values >= n_rows)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Bootstrap indices fall outside the evaluation cohort.")
        integer_values = values.astype(np.int64, copy=False)
        if np.any(integer_values != values):
            raise ValueError("Bootstrap indices must be integral.")
        counts = np.bincount(integer_values, minlength=n_rows)
        if counts.size != n_rows:
            raise ValueError("Bootstrap multiplicity weights have the wrong width.")
        weights[row] = counts
    return _ResampleBatch(tuple(indices), weights)


def _iter_resample_batches(
    resamples: Iterator[np.ndarray],
    *,
    n_rows: int,
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    n_metrics: int,
    max_working_memory_mb: float | None,
) -> Iterator[_ResampleBatch]:
    """Yield dynamically sized chunks bounded by the tensor working budget."""
    budget_bytes = _tensor_memory_budget_bytes(max_working_memory_mb)
    pending: list[np.ndarray] = []
    pending_bytes = 0

    def flush() -> _ResampleBatch | None:
        nonlocal pending, pending_bytes
        if not pending:
            return None
        batch_indices = pending
        pending = []
        pending_bytes = 0
        try:
            return _build_resample_batch(batch_indices, n_rows=n_rows)
        except (MemoryError, TypeError, ValueError, OverflowError):
            return _ResampleBatch(tuple(batch_indices), None)

    for indices in resamples:
        row_bytes = _resample_work_bytes(
            indices,
            n_rows=n_rows,
            y_true=y_true,
            predictions=predictions,
            n_metrics=n_metrics,
        )
        if budget_bytes <= 0 or row_bytes > budget_bytes:
            batch = flush()
            if batch is not None:
                yield batch
            yield _ResampleBatch((indices,), None)
            continue
        if pending and pending_bytes + row_bytes > budget_bytes:
            batch = flush()
            if batch is not None:
                yield batch
        pending.append(indices)
        pending_bytes += row_bytes
    batch = flush()
    if batch is not None:
        yield batch


def _finite_score_row(
    scores: Mapping[str, Mapping[str, float]],
    *,
    metric_names: tuple[str, ...],
) -> bool:
    """Return whether one scalar score row contains every finite metric."""
    return all(
        metric in probe_scores and bool(np.isfinite(float(probe_scores[metric])))
        for probe_scores in scores.values()
        for metric in metric_names
    )


def _score_row_array(
    scores: Mapping[str, Mapping[str, float]],
    *,
    probe_names: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> np.ndarray | None:
    """Convert one finite scalar-score mapping to cache tensor row form."""
    if not _finite_score_row(scores, metric_names=metric_names):
        return None
    try:
        row = np.asarray(
            [
                [float(scores[probe][metric]) for metric in metric_names]
                for probe in probe_names
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if row.shape != (len(probe_names), len(metric_names)) or not np.all(
        np.isfinite(row)
    ):
        return None
    return row


def _build_paired_score_cache(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    target_mode: TargetMode,
    requested_resamples: int,
    random_state: int | None,
    groups: np.ndarray | None = None,
    names: np.ndarray | None = None,
    max_working_memory_mb: float | None = None,
) -> _PairedScoreCache:
    """Build finite primary scores for one target-aware paired resample plan.

    The returned cache is deliberately independent of any pair selection.  A
    single target-aware resample stream is scored with the chunked tensor path
    whenever it is numerically safe; rows that produce a non-finite tensor are
    retried through the scalar compatibility scorer.  Only rows containing a
    finite score for every probe and requested metric are retained, so any
    subsequent pair summary uses exactly the same accepted resamples.
    """
    y_array = np.asarray(y_true)
    prediction_arrays = {name: np.asarray(array) for name, array in predictions.items()}
    probe_names = tuple(prediction_arrays)
    metric_names = _metric_names(target_mode)
    digest = hashlib.sha256()
    score_rows: list[np.ndarray] = []

    def scalar_scores(
        sample_idx: np.ndarray,
    ) -> Mapping[str, Mapping[str, float]] | None:
        """Score one sample through the scalar compatibility path."""
        try:
            scores = {
                probe_name: _score_predictions(
                    y_array,
                    probe_predictions,
                    sample_idx,
                    target_mode=target_mode,
                    names=names,
                )
                for probe_name, probe_predictions in prediction_arrays.items()
            }
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        return scores if _finite_score_row(scores, metric_names=metric_names) else None

    def record_scalar(sample_idx: np.ndarray) -> None:
        """Record one sample through the scalar compatibility path."""
        scores = scalar_scores(sample_idx)
        if scores is None:
            return
        row = _score_row_array(
            scores,
            probe_names=probe_names,
            metric_names=metric_names,
        )
        if row is None:
            return
        digest.update(sample_idx.tobytes())
        score_rows.append(row)

    resamples = _resample_stream(
        y_array,
        target_mode=target_mode,
        requested=requested_resamples,
        random_state=random_state,
        groups=groups,
    )
    tensor_enabled = _canonical_tensor_inputs(
        y_array,
        prediction_arrays,
        target_mode=target_mode,
        names=names,
    )
    if not tensor_enabled:
        for sample_idx in resamples:
            record_scalar(sample_idx)
    else:
        for batch in _iter_resample_batches(
            resamples,
            n_rows=y_array.shape[0],
            y_true=y_array,
            predictions=prediction_arrays,
            n_metrics=len(metric_names),
            max_working_memory_mb=max_working_memory_mb,
        ):
            tensor_scores: np.ndarray | None = None
            if batch.weights is not None:
                try:
                    candidate = _primary_metric_score_tensor(
                        y_array,
                        prediction_arrays,
                        batch.weights,
                        target_mode=target_mode,
                        metrics=metric_names,
                        names=names,
                    )
                    candidate_array = np.asarray(candidate, dtype=np.float64)
                    expected_shape = (
                        len(batch.indices),
                        len(prediction_arrays),
                        len(metric_names),
                    )
                    if candidate_array.shape == expected_shape:
                        tensor_scores = candidate_array
                except (MemoryError, TypeError, ValueError, IndexError, OverflowError):
                    tensor_scores = None
            if tensor_scores is None:
                for sample_idx in batch.indices:
                    record_scalar(sample_idx)
                continue

            finite_rows = np.all(np.isfinite(tensor_scores), axis=(1, 2))
            for row_index, sample_idx in enumerate(batch.indices):
                if bool(finite_rows[row_index]):
                    digest.update(sample_idx.tobytes())
                    score_rows.append(
                        np.asarray(tensor_scores[row_index], dtype=np.float64)
                    )
                    continue
                record_scalar(sample_idx)

    used = len(score_rows)
    if score_rows:
        score_array = np.stack(score_rows, axis=0).astype(np.float64, copy=False)
    else:
        score_array = np.empty(
            (0, len(probe_names), len(metric_names)),
            dtype=np.float64,
        )
    minimum = max(50, requested_resamples // 2)
    available = used >= minimum
    return _PairedScoreCache(
        scores=score_array,
        probe_names=probe_names,
        metric_names=metric_names,
        status="available" if available else "unavailable",
        reason=None if available else "too few valid paired bootstrap resamples",
        resamples_requested=int(requested_resamples),
        resamples_used=int(used),
        resample_plan_id=digest.hexdigest()[:16] if used else None,
    )


def _summarize_cached_probe_pair(
    cache: _PairedScoreCache,
    first_probe: str,
    second_probe: str,
    *,
    point_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any] | None:
    """Summarize one oriented probe pair from a finite score cache."""
    if cache.status != "available" or cache.resamples_used <= 0:
        return None
    try:
        first_index = cache.probe_names.index(first_probe)
        second_index = cache.probe_names.index(second_probe)
    except ValueError:
        return None
    if cache.scores.ndim != 3 or cache.scores.shape[0] != cache.resamples_used:
        return None
    if cache.scores.shape[1] != len(cache.probe_names):
        return None
    if cache.scores.shape[2] != len(cache.metric_names):
        return None
    if not np.all(np.isfinite(cache.scores)):
        return None
    try:
        first_points = point_scores[first_probe]
        second_points = point_scores[second_probe]
        metric_payload: dict[str, Any] = {}
        differences = cache.scores[:, first_index, :] - cache.scores[:, second_index, :]
        for metric_index, metric in enumerate(cache.metric_names):
            point_delta = float(first_points[metric]) - float(second_points[metric])
            values = np.asarray(differences[:, metric_index], dtype=float)
            if not np.all(np.isfinite(values)) or not np.isfinite(point_delta):
                return None
            metric_payload[metric] = {
                "point_delta": point_delta,
                "mean_delta": float(np.mean(values)),
                "paired_standard_error": float(np.std(values, ddof=1)),
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
                "resamples_requested": int(cache.resamples_requested),
                "resamples_used": int(cache.resamples_used),
            }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return {
        "first_probe": first_probe,
        "second_probe": second_probe,
        "metrics": metric_payload,
    }


def _score_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
    *,
    target_mode: TargetMode,
    names: np.ndarray | None,
) -> dict[str, float]:
    """Score one probe on one paired resample."""
    return primary_metric_scores(
        y_true[indices],
        predictions[indices],
        target_mode=target_mode,
        metrics=_metric_names(target_mode),
        names=names,
    )


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
    max_working_memory_mb: float | None = None,
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
    cache = _build_paired_score_cache(
        np.asarray(y_true),
        prediction_arrays,
        target_mode=target_mode,
        requested_resamples=requested_resamples,
        random_state=random_state,
        groups=groups,
        names=names,
        max_working_memory_mb=max_working_memory_mb,
    )
    if cache.status != "available":
        return {
            **base,
            "resamples_used": cache.resamples_used,
            "reason": cache.reason,
        }

    comparisons: dict[str, Any] = {}
    for first, second in pairs:
        summary = _summarize_cached_probe_pair(
            cache,
            first,
            second,
            point_scores=probes,
        )
        if summary is not None:
            comparisons[f"{first}__vs__{second}"] = summary
    return {
        **base,
        "status": "available",
        "reason": None,
        "resamples_used": int(cache.resamples_used),
        "resample_plan_id": cache.resample_plan_id,
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
