"""Optional topology diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.densify import ensure_dense_or_sample

_MIN_TOPOLOGY_SAMPLES = 30
_MAX_PERSISTENT_TOPOLOGY_SAMPLES = 2000
_MAX_MULTILABEL_TOPOLOGY_LABELS = 5
_MAX_REPORTED_SKIPPED_LABELS = 50


def _boundary_scale(X: np.ndarray) -> float:
    """Return a simple coordinate-scale proxy for normalized persistence."""
    if X.size == 0:
        return 0.0
    ranges = np.ptp(X, axis=0)
    positive_ranges = ranges[ranges > 0]
    if positive_ranges.size == 0:
        return 0.0
    return float(np.mean(positive_ranges))


def _topology_sample_cap(config: ProfilerConfig) -> int:
    """Return the persistent-topology row cap for the active budget."""
    budget = cast(dict[str, Any], BUDGETS[config.budget])
    max_allowed = min(
        int(budget["max_boundary_samples"]),
        _MAX_PERSISTENT_TOPOLOGY_SAMPLES,
    )
    if config.max_samples is not None:
        max_allowed = min(max_allowed, config.max_samples)
    return max_allowed


def _slice_rows(matrix: Any, indices: np.ndarray) -> Any:
    """Slice dense or sparse rows."""
    return matrix[indices, :] if sparse.issparse(matrix) else matrix[indices]


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _label_counts(Y: Any) -> np.ndarray:
    """Return positive counts for each multilabel column."""
    if sparse.issparse(Y):
        return np.asarray(Y.sum(axis=0)).ravel().astype(int)
    return np.asarray(Y).sum(axis=0).astype(int)


def _positive_indices_for_label(Y: Any, label_index: int) -> np.ndarray:
    """Return row indices where one multilabel column is positive."""
    if sparse.issparse(Y):
        return np.asarray(Y[:, label_index].nonzero()[0], dtype=int)
    return np.flatnonzero(np.asarray(Y)[:, label_index] > 0)


def _sample_indices(
    indices: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic sampled indices and sampling metadata."""
    if indices.shape[0] <= n_samples:
        return indices, {
            "sampled": False,
            "n_original": int(indices.shape[0]),
            "n_used": int(indices.shape[0]),
        }
    rng = np.random.default_rng(random_state)
    sampled = np.sort(rng.choice(indices, size=n_samples, replace=False))
    return sampled, {
        "sampled": True,
        "n_original": int(indices.shape[0]),
        "n_used": int(sampled.shape[0]),
        "method": "random_without_replacement",
    }


def _summarize_persistent_topology(X_dense: np.ndarray) -> dict[str, Any]:
    """Return compact persistent homology summaries for dense samples."""
    from ripser import ripser

    dgms = ripser(X_dense, maxdim=1)["dgms"]
    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h1_lifetimes = h1[:, 1] - h1[:, 0] if h1.size else np.array([])
    h0_lifetimes = h0[:, 1] - h0[:, 0] if h0.size else np.array([])
    finite_h0 = h0_lifetimes[np.isfinite(h0_lifetimes)]
    finite_h1 = h1_lifetimes[np.isfinite(h1_lifetimes)]
    max_h1 = float(np.max(finite_h1)) if finite_h1.size else 0.0
    total_h1 = float(np.sum(finite_h1)) if finite_h1.size else 0.0
    scale = _boundary_scale(X_dense)
    relative_h1 = float(max_h1 / max(scale, 1e-12)) if scale > 0 else 0.0
    topology_strength = float(np.clip((relative_h1 - 0.08) / 0.25, 0.0, 1.0))
    h0_threshold = np.median(finite_h0) if finite_h0.size else 0.0
    h1_threshold = np.median(finite_h1) if finite_h1.size else 0.0
    return {
        "h0_persistence_count": int(np.sum(finite_h0 > h0_threshold)),
        "h1_persistence_count": int(np.sum(finite_h1 > h1_threshold)),
        "total_h0_persistence": float(np.sum(finite_h0)) if finite_h0.size else 0.0,
        "total_h1_persistence": total_h1,
        "max_h1_persistence": max_h1,
        "boundary_scale": scale,
        "relative_h1_persistence": relative_h1,
        "topology_strength": topology_strength,
        "persistence_entropy": float(
            -np.sum(
                (finite_h1 / np.sum(finite_h1))
                * np.log(np.clip(finite_h1 / np.sum(finite_h1), 1e-12, None))
            )
        )
        if finite_h1.size and np.sum(finite_h1) > 0
        else 0.0,
    }


def _compute_topology_object(
    X_subset: Any,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Compute topology for one already-capped sample subset."""
    n_samples = int(X_subset.shape[0])
    if n_samples < _MIN_TOPOLOGY_SAMPLES:
        return {
            "sample_size": n_samples,
            "sampling": sampling,
            "skipped_reason": "too few samples",
        }

    y_proxy = np.ones(n_samples, dtype=int)
    dense_info = ensure_dense_or_sample(
        X_subset,
        y_proxy,
        reason=reason,
        config=config,
        report_context=report_context,
    )
    if dense_info["skipped"]:
        return {
            "sample_size": n_samples,
            "sampling": sampling,
            "skipped_reason": "dense conversion unavailable",
        }

    X_dense = np.asarray(dense_info["X"])
    return {
        "sample_size": int(X_dense.shape[0]),
        "sampling": sampling,
        **_summarize_persistent_topology(X_dense),
    }


def compute_topology_diagnostics(
    X: Any,
    y: np.ndarray,
    boundary: dict[str, Any],
    geometry: dict[str, Any],
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Run optional persistent topology diagnostics when available and feasible."""
    budget = cast(dict[str, Any], BUDGETS[config.budget])
    if config.topology == "off":
        return {"mode": "off", "skipped_reason": "topology disabled"}
    if config.topology == "graph":
        return {"mode": "graph", "skipped_reason": "persistent topology not requested"}
    if config.topology == "auto" and not budget["run_persistent_topology"]:
        return {
            "mode": config.topology,
            "skipped_reason": "persistent topology disabled for this budget",
        }

    indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    if indices.shape[0] < _MIN_TOPOLOGY_SAMPLES:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "too few boundary candidates"}
        )
        return {
            "mode": config.topology,
            "skipped_reason": "too few boundary candidates",
        }
    if indices.shape[0] > 2000:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "too many boundary candidates"}
        )
        return {
            "mode": config.topology,
            "skipped_reason": "too many boundary candidates",
        }
    if (
        geometry.get("distance_concentration_proxy") is not None
        and geometry["distance_concentration_proxy"] < 0.05
    ):
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "geometry reliability too low"}
        )
        return {
            "mode": config.topology,
            "skipped_reason": "geometry reliability too low",
        }

    try:
        from ripser import ripser as _ripser  # noqa: F401
    except ImportError:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "ripser is not installed"}
        )
        return {"mode": config.topology, "skipped_reason": "ripser is not installed"}

    dense_info = ensure_dense_or_sample(
        X[indices],
        y[indices],
        reason="persistent_topology",
        config=config,
        report_context=report_context,
    )
    if dense_info["skipped"]:
        return {
            "mode": config.topology,
            "skipped_reason": "dense conversion unavailable",
        }

    return {
        "mode": config.topology,
        **_summarize_persistent_topology(np.asarray(dense_info["X"])),
    }


def _multilabel_topology_skip(
    *,
    mode: str,
    reason: str,
    report_context: dict[str, Any] | None = None,
    record: bool = False,
) -> dict[str, Any]:
    """Return a skipped multilabel topology payload."""
    if record and report_context is not None:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "multilabel_persistent_topology", "reason": reason}
        )
    return {"target_type": "multilabel", "mode": mode, "skipped_reason": reason}


def _selected_multilabel_topology_labels(
    Y: Any,
    boundary_indices: np.ndarray,
    *,
    label_names: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]], np.ndarray]:
    """Choose high-support labels for per-label topology diagnostics."""
    counts = _label_counts(Y)
    boundary_counts = (
        _label_counts(_slice_rows(Y, boundary_indices))
        if boundary_indices.size
        else np.zeros_like(counts)
    )
    eligible = np.flatnonzero(counts >= _MIN_TOPOLOGY_SAMPLES)
    order = sorted(
        eligible.tolist(),
        key=lambda idx: (int(boundary_counts[idx]), int(counts[idx])),
        reverse=True,
    )
    selected = order[:_MAX_MULTILABEL_TOPOLOGY_LABELS]
    selected_set = set(selected)
    skipped: list[dict[str, Any]] = []
    for idx, count in enumerate(counts):
        if len(skipped) >= _MAX_REPORTED_SKIPPED_LABELS:
            break
        if count < _MIN_TOPOLOGY_SAMPLES:
            skipped.append(
                {
                    "label": str(label_names[idx]),
                    "reason": "too few positive samples",
                    "positive_count": int(count),
                }
            )
        elif idx not in selected_set:
            skipped.append(
                {
                    "label": str(label_names[idx]),
                    "reason": "not selected due to per-label topology cap",
                    "positive_count": int(count),
                    "boundary_positive_count": int(boundary_counts[idx]),
                }
            )
    return selected, skipped, counts


def compute_multilabel_topology_diagnostics(
    X: Any,
    Y: Any,
    boundary: dict[str, Any],
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    label_names: np.ndarray,
) -> dict[str, Any]:
    """Run optional persistent topology diagnostics for multilabel targets."""
    budget = cast(dict[str, Any], BUDGETS[config.budget])
    if config.topology == "off":
        return _multilabel_topology_skip(
            mode="off",
            reason="topology disabled",
        )
    if config.topology == "graph":
        return _multilabel_topology_skip(
            mode="graph",
            reason="persistent topology not requested",
        )
    if config.topology == "auto" and not budget["run_persistent_topology"]:
        return _multilabel_topology_skip(
            mode=config.topology,
            reason="persistent topology disabled for this budget",
        )

    try:
        from ripser import ripser as _ripser  # noqa: F401
    except ImportError:
        return _multilabel_topology_skip(
            mode=config.topology,
            reason="ripser is not installed",
            report_context=report_context,
            record=True,
        )

    cap = _topology_sample_cap(config)
    boundary_indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    strengths: list[float] = []
    boundary_topology: dict[str, Any]
    if boundary_indices.shape[0] < _MIN_TOPOLOGY_SAMPLES:
        boundary_topology = {
            "sample_size": int(boundary_indices.shape[0]),
            "skipped_reason": "too few boundary candidates",
        }
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "multilabel_boundary_topology",
                "reason": "too few boundary candidates",
            }
        )
    else:
        boundary_used, boundary_sampling = _sample_indices(
            boundary_indices,
            n_samples=cap,
            random_state=config.random_state,
        )
        boundary_topology = _compute_topology_object(
            _slice_rows(X, boundary_used),
            reason="multilabel_boundary_topology",
            config=config,
            report_context=report_context,
            sampling=boundary_sampling,
        )
        if "topology_strength" in boundary_topology:
            strengths.append(float(boundary_topology["topology_strength"]))

    selected_labels, skipped_labels, counts = _selected_multilabel_topology_labels(
        Y,
        boundary_indices,
        label_names=label_names,
    )
    per_label_topology = []
    for label_index in selected_labels:
        positive_indices = _positive_indices_for_label(Y, label_index)
        used_indices, sampling = _sample_indices(
            positive_indices,
            n_samples=cap,
            random_state=config.random_state,
        )
        label_topology = _compute_topology_object(
            _slice_rows(X, used_indices),
            reason="multilabel_per_label_topology",
            config=config,
            report_context=report_context,
            sampling=sampling,
        )
        label_topology.update(
            {
                "label": str(label_names[label_index]),
                "label_index": int(label_index),
                "positive_count": int(counts[label_index]),
            }
        )
        per_label_topology.append(label_topology)
        if "topology_strength" in label_topology:
            strengths.append(float(label_topology["topology_strength"]))

    result: dict[str, Any] = {
        "target_type": "multilabel",
        "mode": config.topology,
        "boundary_topology": boundary_topology,
        "per_label_topology": per_label_topology,
        "selected_label_count": len(per_label_topology),
        "skipped_labels": skipped_labels,
        "skipped_label_count": max(0, int(Y.shape[1]) - len(per_label_topology)),
        "topology_strength": float(max(strengths)) if strengths else 0.0,
    }
    if not strengths:
        result["skipped_reason"] = "no multilabel topology objects were computed"
    return result
