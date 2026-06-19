"""Boundary candidate diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _aligned_predictions(
    probe_result: dict[str, Any],
    row_indices: np.ndarray,
) -> np.ndarray | None:
    """Align probe predictions to one sampled row order when possible."""
    predictions = probe_result.get("predictions")
    sample_info = probe_result.get("sample_info", {})
    probe_indices = np.asarray(sample_info.get("indices", []), dtype=int)
    if predictions is None:
        return None
    predictions_array = np.asarray(predictions)
    if probe_indices.size == 0 and predictions_array.shape[0] == row_indices.shape[0]:
        return predictions_array
    if probe_indices.shape[0] != predictions_array.shape[0]:
        return None
    mapping = {
        int(idx): predictions_array[pos] for pos, idx in enumerate(probe_indices)
    }
    aligned: list[Any] = []
    for idx in row_indices.tolist():
        if idx not in mapping:
            return None
        aligned.append(mapping[idx])
    return np.asarray(aligned)


def compute_boundary_candidates(
    y: np.ndarray,
    neighborhood: dict[str, Any],
    probes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract boundary candidate indices from ambiguity and disagreement."""
    local_entropy = np.asarray(neighborhood.get("local_entropy", []), dtype=float)
    local_ambiguity = np.asarray(neighborhood.get("local_ambiguity", []), dtype=float)
    row_indices = np.asarray(
        neighborhood.get("row_indices", list(range(local_entropy.shape[0]))), dtype=int
    )
    if local_entropy.size == 0 or local_ambiguity.size == 0:
        return {
            "candidate_indices": [],
            "candidate_fraction": 0.0,
            "boundary_sample_size": 0,
            "class_composition": {},
            "warning": "Boundary diagnostics unavailable.",
        }
    entropy_threshold = float(np.quantile(local_entropy, 0.75))
    ambiguity_threshold = max(0.33, float(np.quantile(local_ambiguity, 0.75)))
    candidate_mask = (local_entropy >= entropy_threshold) | (
        local_ambiguity >= ambiguity_threshold
    )
    linear_preds = _aligned_predictions(probes.get("linear", {}), row_indices)
    knn_preds = _aligned_predictions(probes.get("knn", {}), row_indices)
    if linear_preds is not None and knn_preds is not None:
        disagreement = np.asarray(linear_preds) != np.asarray(knn_preds)
        candidate_mask = candidate_mask | disagreement
    sample_positions = np.flatnonzero(candidate_mask)
    indices = row_indices[sample_positions]
    counts = {
        str(cls): int(np.sum(y[indices] == cls))
        for cls in np.unique(y[indices])
        if indices.size
    }
    return {
        "candidate_indices": indices.tolist(),
        "sample_position_indices": sample_positions.tolist(),
        "candidate_fraction": float(indices.shape[0] / max(1, y.shape[0])),
        "boundary_sample_size": int(indices.shape[0]),
        "class_composition": counts,
        "warning": "Boundary sample is very small."
        if indices.shape[0] < max(10, len(np.unique(y)))
        else None,
    }


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _sample_prediction_jaccard(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return samplewise Jaccard similarity between two multilabel predictions."""
    first_bool = first.astype(bool)
    second_bool = second.astype(bool)
    intersection = np.logical_and(first_bool, second_bool).sum(axis=1).astype(float)
    union = np.logical_or(first_bool, second_bool).sum(axis=1).astype(float)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )


def _top_candidate_label_counts(
    Y_dense: np.ndarray,
    label_names: np.ndarray,
    indices: np.ndarray,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the top label counts within a candidate subset."""
    if indices.size == 0:
        return []
    counts = Y_dense[indices].sum(axis=0).astype(int)
    order = np.argsort(-counts)[:limit]
    return [
        {"label": str(label_names[idx]), "count": int(counts[idx])}
        for idx in order
        if counts[idx] > 0
    ]


def compute_multilabel_boundary_candidates(
    Y: Any,
    neighborhood: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    *,
    label_names: np.ndarray,
) -> dict[str, Any]:
    """Extract multilabel boundary candidates from local instability triggers."""
    local_jaccard = np.asarray(
        neighborhood.get("local_neighbor_jaccard", []),
        dtype=float,
    )
    local_hamming = np.asarray(
        neighborhood.get("local_neighbor_hamming_distance", []), dtype=float
    )
    local_entropy = np.asarray(neighborhood.get("local_label_entropy", []), dtype=float)
    local_cardinality_std = np.asarray(
        neighborhood.get("local_cardinality_std", []), dtype=float
    )
    row_indices = np.asarray(
        neighborhood.get("row_indices", list(range(local_jaccard.shape[0]))), dtype=int
    )
    if (
        local_jaccard.size == 0
        or local_hamming.size == 0
        or local_entropy.size == 0
        or local_cardinality_std.size == 0
    ):
        return {
            "candidate_indices": [],
            "strong_candidate_indices": [],
            "candidate_fraction": 0.0,
            "strong_candidate_fraction": 0.0,
            "boundary_sample_size": 0,
            "trigger_counts": {},
            "trigger_thresholds": {},
            "mean_trigger_count": 0.0,
            "label_cardinality_summary": {},
            "top_candidate_label_counts": [],
            "warning": "Multilabel boundary diagnostics unavailable.",
        }

    thresholds = {
        "low_neighbor_jaccard": float(np.quantile(local_jaccard, 0.25)),
        "high_neighbor_hamming": float(np.quantile(local_hamming, 0.75)),
        "high_local_label_entropy": float(np.quantile(local_entropy, 0.75)),
        "high_cardinality_variance": float(np.quantile(local_cardinality_std, 0.75)),
    }
    trigger_masks: dict[str, np.ndarray] = {
        "low_neighbor_jaccard": local_jaccard <= thresholds["low_neighbor_jaccard"],
        "high_neighbor_hamming": local_hamming >= thresholds["high_neighbor_hamming"],
        "high_local_label_entropy": (
            local_entropy >= thresholds["high_local_label_entropy"]
        ),
        "high_cardinality_variance": (
            local_cardinality_std >= thresholds["high_cardinality_variance"]
        ),
    }

    local_probe_names = ("knn", "kernel_approx")
    local_probe = next(
        (
            probes[name]
            for name in local_probe_names
            if probes.get(name, {}).get("predictions") is not None
        ),
        None,
    )
    probes.get("linear", {}).get("predictions")
    prediction_similarity_threshold = None
    aligned_linear = _aligned_predictions(probes.get("linear", {}), row_indices)
    aligned_local = (
        _aligned_predictions(local_probe, row_indices)
        if local_probe is not None
        else None
    )
    if aligned_linear is not None and aligned_local is not None:
        prediction_similarity = _sample_prediction_jaccard(
            np.asarray(aligned_linear, dtype=int),
            np.asarray(aligned_local, dtype=int),
        )
        prediction_similarity_threshold = float(
            np.quantile(prediction_similarity, 0.25)
        )
        trigger_masks["linear_vs_local_prediction_disagreement"] = (
            prediction_similarity <= prediction_similarity_threshold
        )
    else:
        trigger_masks["linear_vs_local_prediction_disagreement"] = np.zeros(
            local_jaccard.shape[0], dtype=bool
        )

    trigger_names_by_index: list[list[str]] = []
    trigger_count_array = np.zeros(local_jaccard.shape[0], dtype=int)
    trigger_counts: dict[str, int] = {}
    for name, mask in trigger_masks.items():
        trigger_counts[name] = int(np.sum(mask))
        trigger_count_array += mask.astype(int)
    for index in range(local_jaccard.shape[0]):
        trigger_names_by_index.append(
            [name for name, mask in trigger_masks.items() if bool(mask[index])]
        )

    candidate_mask = trigger_count_array >= 1
    strong_candidate_mask = trigger_count_array >= 2
    sample_positions = np.flatnonzero(candidate_mask)
    strong_positions = np.flatnonzero(strong_candidate_mask)
    indices = row_indices[sample_positions]
    strong_indices = row_indices[strong_positions]
    Y_dense = _dense_multilabel_matrix(Y)
    candidate_cardinality = (
        Y_dense[indices].sum(axis=1).astype(float) if indices.size else np.array([])
    )
    trigger_thresholds = {
        **thresholds,
        "linear_vs_local_prediction_disagreement": prediction_similarity_threshold,
    }
    return {
        "candidate_indices": indices.tolist(),
        "strong_candidate_indices": strong_indices.tolist(),
        "sample_position_indices": sample_positions.tolist(),
        "candidate_fraction": float(indices.shape[0] / max(1, Y_dense.shape[0])),
        "strong_candidate_fraction": float(
            strong_indices.shape[0] / max(1, Y_dense.shape[0])
        ),
        "boundary_sample_size": int(indices.shape[0]),
        "trigger_counts": trigger_counts,
        "trigger_thresholds": trigger_thresholds,
        "candidate_trigger_counts": trigger_count_array.tolist(),
        "trigger_names_by_index": trigger_names_by_index,
        "mean_trigger_count": float(np.mean(trigger_count_array)),
        "label_cardinality_summary": {
            "mean": (
                float(np.mean(candidate_cardinality))
                if candidate_cardinality.size
                else 0.0
            ),
            "std": (
                float(np.std(candidate_cardinality))
                if candidate_cardinality.size
                else 0.0
            ),
            "min": (
                float(np.min(candidate_cardinality))
                if candidate_cardinality.size
                else 0.0
            ),
            "max": (
                float(np.max(candidate_cardinality))
                if candidate_cardinality.size
                else 0.0
            ),
        },
        "top_candidate_label_counts": _top_candidate_label_counts(
            Y_dense, label_names, indices
        ),
        "warning": "Boundary sample is very small."
        if indices.shape[0] < max(10, Y_dense.shape[1])
        else None,
    }
