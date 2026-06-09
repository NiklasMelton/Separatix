"""Recommendation engine for separatix."""

from __future__ import annotations

from typing import Any

import numpy as np

from separatix.constants import (
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED,
    INCONCLUSIVE,
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY,
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)


def _clip(value: float | None) -> float | None:
    if value is None:
        return None
    return float(np.clip(value, 0.0, 1.0))


def compute_scores(
    metrics: dict[str, Any], skipped_count: int, warning_count: int
) -> dict[str, float | None]:
    """Compute transparent normalized scores."""
    probes = metrics["probes"]
    dummy = probes.get("dummy", {}).get("balanced_accuracy")
    linear = probes.get("linear", {}).get("balanced_accuracy")
    available_scores = [
        result["balanced_accuracy"]
        for result in probes.values()
        if "balanced_accuracy" in result
    ]
    best_available = max(available_scores) if available_scores else None
    nonlinear_scores = [
        result["balanced_accuracy"]
        for name, result in probes.items()
        if name in {"knn", "kernel_approx"} and "balanced_accuracy" in result
    ]
    best_nonlinear = (
        max(nonlinear_scores or ([linear] if linear is not None else []))
        if nonlinear_scores or linear is not None
        else None
    )
    signal = (
        _clip((best_available - dummy) / max(1e-9, 1.0 - dummy))
        if best_available is not None and dummy is not None
        else None
    )
    overlap = _clip(
        np.mean(
            [
                metrics["neighborhood"].get("mean_local_entropy", 0.0),
                metrics["neighborhood"].get("high_entropy_fraction", 0.0),
                metrics["neighborhood"].get("cross_class_neighbor_fraction", 0.0),
            ]
        )
    )
    linearity = (
        _clip(linear / max(best_available, 1e-9))
        if linear is not None and best_available is not None
        else None
    )
    nonlinear = (
        _clip((best_nonlinear - linear) / max(1e-9, 1.0 - linear))
        if linear is not None and best_nonlinear is not None
        else None
    )
    fragmentation = _clip(metrics["graph"].get("graph_fragmentation_score", 0.0))
    topology = None
    if "h1_persistence_count" in metrics["topology"]:
        topology = _clip(
            min(
                1.0,
                (
                    metrics["topology"].get("max_h1_persistence", 0.0)
                    + metrics["topology"].get("h1_persistence_count", 0)
                )
                / 5.0,
            )
        )
    min_class_count = min(metrics["audit"]["class_counts"].values())
    reliability = 1.0
    reliability -= min(0.4, skipped_count * 0.08)
    reliability -= min(0.2, warning_count * 0.05)
    reliability -= (
        0.15
        if metrics["geometry"].get("distance_concentration_proxy") is not None
        and metrics["geometry"]["distance_concentration_proxy"] < 0.08
        else 0.0
    )
    reliability -= 0.15 if min_class_count < 5 else 0.0
    reliability -= 0.1 if metrics["audit"]["imbalance_ratio"] > 10 else 0.0
    if metrics["boundary"]["boundary_sample_size"] < max(
        10, metrics["audit"]["n_classes"] * 2
    ):
        reliability -= 0.1
    linear_stability = probes.get("linear", {}).get("stability_balanced_accuracy_std")
    if linear_stability is not None:
        reliability -= min(0.15, float(linear_stability) * 0.5)
    if best_available is None or linear is None:
        reliability -= 0.3
    return {
        "signal_score": signal,
        "overlap_score": overlap,
        "linearity_score": linearity,
        "nonlinearity_score": nonlinear,
        "fragmentation_score": fragmentation,
        "topology_score": topology,
        "reliability_score": _clip(reliability),
    }


def make_recommendation(
    scores: dict[str, float | None], metrics: dict[str, Any]
) -> tuple[str, str, list[str], dict[str, str]]:
    """Generate a rule-based recommendation and decision path."""
    decision_path: list[str] = []
    interpretations: dict[str, str] = {}
    reliability = scores["reliability_score"] or 0.0
    signal = scores["signal_score"] or 0.0
    nonlinearity = scores["nonlinearity_score"] or 0.0
    linearity = scores["linearity_score"] or 0.0
    overlap = scores["overlap_score"] or 0.0
    fragmentation = scores["fragmentation_score"] or 0.0
    topology = scores["topology_score"] or 0.0

    if reliability < 0.35:
        recommendation = INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
        decision_path.append(
            "Reliability was too low to trust geometry-heavy guidance."
        )
    elif signal < 0.2:
        recommendation = FEATURE_OR_LABEL_BOTTLENECK_LIKELY
        decision_path.append(
            "All probes were close to the dummy baseline, "
            "suggesting weak usable signal."
        )
    elif linearity >= 0.93 and nonlinearity < 0.08:
        recommendation = LINEAR_LIKELY_SUFFICIENT
        decision_path.append(
            "The linear probe performed close to the best available probe."
        )
    elif overlap > 0.6 and nonlinearity < 0.12:
        recommendation = FEATURE_OR_LABEL_BOTTLENECK_LIKELY
        decision_path.append("Local overlap was high without much nonlinear gain.")
    elif nonlinearity >= 0.18:
        best_probe = metrics["baseline"]["best_probe"]
        decision_path.append(
            "Best nonlinear probe improved over the linear probe, "
            f"with {best_probe} performing best."
        )
        if fragmentation >= 0.55:
            recommendation = HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
            decision_path.append("Boundary graph fragmentation was high.")
        elif topology >= 0.4:
            recommendation = KERNEL_OR_LOCAL_RECOMMENDED
            decision_path.append(
                "Persistent topology suggested nontrivial local structure."
            )
        elif metrics["baseline"]["best_probe"] in {"knn", "kernel_approx"}:
            recommendation = KERNEL_OR_LOCAL_RECOMMENDED
            decision_path.append(
                "Local or kernel-style probes outperformed smoother global probes."
            )
        else:
            recommendation = SMOOTH_NONLINEAR_RECOMMENDED
            decision_path.append(
                "Nonlinear gain was present without strong fragmentation."
            )
    else:
        recommendation = INCONCLUSIVE
        decision_path.append(
            "The diagnostics showed mixed evidence without a strong dominant pattern."
        )

    confidence = (
        "high"
        if reliability >= 0.8 and signal >= 0.5
        else "medium"
        if reliability >= 0.55
        else "low"
    )
    interpretations["signal"] = (
        "Higher values mean the labels appear more predictable "
        "than a class-prior baseline."
    )
    interpretations["overlap"] = (
        "Higher values mean neighborhoods are more class-mixed and ambiguous."
    )
    interpretations["linearity"] = (
        "Higher values mean the linear probe is close to the best observed probe."
    )
    interpretations["nonlinearity"] = (
        "Higher values mean nonlinear probes improved noticeably over the linear probe."
    )
    interpretations["fragmentation"] = (
        "Higher values mean the estimated boundary looks more "
        "partitioned or locally broken up."
    )
    interpretations["reliability"] = (
        "Higher values mean the recommendation rests on more "
        "stable and complete diagnostics."
    )
    return recommendation, confidence, decision_path, interpretations
