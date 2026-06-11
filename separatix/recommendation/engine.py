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

# Lowest normalized score returned by recommendation score helpers.
MIN_NORMALIZED_SCORE = 0.0
# Highest normalized score returned by recommendation score helpers.
MAX_NORMALIZED_SCORE = 1.0
# Small denominator guard used when normalizing probe improvements.
NORMALIZATION_EPSILON = 1e-9
# Maximum reliability penalty assigned for skipped diagnostics.
MAX_SKIPPED_DIAGNOSTIC_PENALTY = 0.4
# Reliability penalty added for each skipped diagnostic.
SKIPPED_DIAGNOSTIC_PENALTY = 0.08
# Maximum reliability penalty assigned for reported warnings.
MAX_WARNING_PENALTY = 0.2
# Reliability penalty added for each reported warning.
WARNING_PENALTY = 0.05
# Reliability penalty for concentrated distances in high-dimensional geometry.
DISTANCE_CONCENTRATION_PENALTY = 0.15
# Distance concentration threshold below which geometry is less reliable.
DISTANCE_CONCENTRATION_THRESHOLD = 0.08
# Reliability penalty when the smallest class has too few samples.
SMALL_CLASS_PENALTY = 0.15
# Minimum per-class sample count before class-size reliability is penalized.
MIN_RELIABLE_CLASS_COUNT = 5
# Reliability penalty when class imbalance is substantial.
IMBALANCE_PENALTY = 0.1
# Imbalance ratio above which recommendation reliability is reduced.
HIGH_IMBALANCE_RATIO = 10
# Minimum absolute boundary sample count needed for boundary guidance.
MIN_BOUNDARY_SAMPLE_SIZE = 10
# Multiplier for the class-aware minimum boundary sample size.
BOUNDARY_SAMPLE_CLASS_MULTIPLIER = 2
# Reliability penalty when boundary sampling is too small.
SMALL_BOUNDARY_SAMPLE_PENALTY = 0.1
# Maximum reliability penalty assigned for unstable linear probe scores.
MAX_LINEAR_STABILITY_PENALTY = 0.15
# Converts linear probe standard deviation into a reliability penalty.
LINEAR_STABILITY_PENALTY_SCALE = 0.5
# Reliability penalty when essential probe scores are unavailable.
MISSING_ESSENTIAL_PROBE_PENALTY = 0.3
# Reliability threshold below which geometry-heavy guidance is not trusted.
LOW_RELIABILITY_THRESHOLD = 0.35
# Signal threshold below which usable label signal is considered weak.
WEAK_SIGNAL_THRESHOLD = 0.2
# Linearity threshold for treating a linear model as likely sufficient.
LINEAR_SUFFICIENCY_THRESHOLD = 0.93
# Nonlinearity threshold below which nonlinear gain is considered negligible.
LOW_NONLINEARITY_GAIN_THRESHOLD = 0.08
# Overlap threshold for flagging class mixing as a likely bottleneck.
HIGH_OVERLAP_THRESHOLD = 0.6
# Nonlinearity threshold used with high overlap to indicate limited gain.
OVERLAP_NONLINEARITY_GAIN_THRESHOLD = 0.12
# Nonlinearity threshold for entering the nonlinear recommendation branch.
MEANINGFUL_NONLINEARITY_GAIN_THRESHOLD = 0.18
# Fragmentation threshold for recommending higher-capacity partitioning.
HIGH_FRAGMENTATION_THRESHOLD = 0.55
# Smooth-probe margin treated as clearly better than local/kernel probes.
SMOOTH_MARGIN_TOLERANCE = 0.01
# Topology score threshold for emphasizing local/kernel-style models.
STRONG_TOPOLOGY_THRESHOLD = 0.4
# Reliability threshold used for high-confidence recommendations.
HIGH_CONFIDENCE_RELIABILITY_THRESHOLD = 0.8
# Signal threshold used for high-confidence recommendations.
HIGH_CONFIDENCE_SIGNAL_THRESHOLD = 0.5
# Reliability threshold used for medium-confidence recommendations.
MEDIUM_CONFIDENCE_RELIABILITY_THRESHOLD = 0.55


def _clip(value: float | None) -> float | None:
    if value is None:
        return None
    return float(np.clip(value, MIN_NORMALIZED_SCORE, MAX_NORMALIZED_SCORE))


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
        if name in {"smooth_poly", "knn", "kernel_approx"}
        and "balanced_accuracy" in result
    ]
    best_nonlinear = (
        max(nonlinear_scores or ([linear] if linear is not None else []))
        if nonlinear_scores or linear is not None
        else None
    )
    signal = (
        _clip(
            (best_available - dummy)
            / max(NORMALIZATION_EPSILON, MAX_NORMALIZED_SCORE - dummy)
        )
        if best_available is not None and dummy is not None
        else None
    )
    overlap = _clip(
        np.mean(
            [
                metrics["neighborhood"].get("mean_local_entropy", MIN_NORMALIZED_SCORE),
                metrics["neighborhood"].get(
                    "high_entropy_fraction", MIN_NORMALIZED_SCORE
                ),
                metrics["neighborhood"].get(
                    "cross_class_neighbor_fraction", MIN_NORMALIZED_SCORE
                ),
            ]
        )
    )
    linearity = (
        _clip(linear / max(best_available, NORMALIZATION_EPSILON))
        if linear is not None and best_available is not None
        else None
    )
    nonlinear = (
        _clip(
            (best_nonlinear - linear)
            / max(NORMALIZATION_EPSILON, MAX_NORMALIZED_SCORE - linear)
        )
        if linear is not None and best_nonlinear is not None
        else None
    )
    fragmentation = _clip(
        metrics["graph"].get("graph_fragmentation_score", MIN_NORMALIZED_SCORE)
    )
    topology = None
    if "topology_strength" in metrics["topology"]:
        topology = _clip(
            metrics["topology"].get("topology_strength", MIN_NORMALIZED_SCORE)
        )
    elif "h1_persistence_count" in metrics["topology"]:
        topology = _clip(
            min(
                MAX_NORMALIZED_SCORE,
                metrics["topology"].get("max_h1_persistence", MIN_NORMALIZED_SCORE),
            )
        )
    min_class_count = min(metrics["audit"]["class_counts"].values())
    reliability = MAX_NORMALIZED_SCORE
    reliability -= min(
        MAX_SKIPPED_DIAGNOSTIC_PENALTY,
        skipped_count * SKIPPED_DIAGNOSTIC_PENALTY,
    )
    reliability -= min(MAX_WARNING_PENALTY, warning_count * WARNING_PENALTY)
    reliability -= (
        DISTANCE_CONCENTRATION_PENALTY
        if metrics["geometry"].get("distance_concentration_proxy") is not None
        and metrics["geometry"]["distance_concentration_proxy"]
        < DISTANCE_CONCENTRATION_THRESHOLD
        else MIN_NORMALIZED_SCORE
    )
    reliability -= (
        SMALL_CLASS_PENALTY
        if min_class_count < MIN_RELIABLE_CLASS_COUNT
        else MIN_NORMALIZED_SCORE
    )
    reliability -= (
        IMBALANCE_PENALTY
        if metrics["audit"]["imbalance_ratio"] > HIGH_IMBALANCE_RATIO
        else MIN_NORMALIZED_SCORE
    )
    if metrics["boundary"]["boundary_sample_size"] < max(
        MIN_BOUNDARY_SAMPLE_SIZE,
        metrics["audit"]["n_classes"] * BOUNDARY_SAMPLE_CLASS_MULTIPLIER,
    ):
        reliability -= SMALL_BOUNDARY_SAMPLE_PENALTY
    linear_stability = probes.get("linear", {}).get("stability_balanced_accuracy_std")
    if linear_stability is not None:
        reliability -= min(
            MAX_LINEAR_STABILITY_PENALTY,
            float(linear_stability) * LINEAR_STABILITY_PENALTY_SCALE,
        )
    if best_available is None or linear is None:
        reliability -= MISSING_ESSENTIAL_PROBE_PENALTY
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
    probes = metrics["probes"]
    reliability = scores["reliability_score"] or MIN_NORMALIZED_SCORE
    signal = scores["signal_score"] or MIN_NORMALIZED_SCORE
    nonlinearity = scores["nonlinearity_score"] or MIN_NORMALIZED_SCORE
    linearity = scores["linearity_score"] or MIN_NORMALIZED_SCORE
    overlap = scores["overlap_score"] or MIN_NORMALIZED_SCORE
    fragmentation = scores["fragmentation_score"] or MIN_NORMALIZED_SCORE
    topology = scores["topology_score"] or MIN_NORMALIZED_SCORE

    if reliability < LOW_RELIABILITY_THRESHOLD:
        recommendation = INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
        decision_path.append(
            "Reliability was too low to trust geometry-heavy guidance."
        )
    elif signal < WEAK_SIGNAL_THRESHOLD:
        recommendation = FEATURE_OR_LABEL_BOTTLENECK_LIKELY
        decision_path.append(
            "All probes were close to the dummy baseline, "
            "suggesting weak usable signal."
        )
    elif (
        linearity >= LINEAR_SUFFICIENCY_THRESHOLD
        and nonlinearity < LOW_NONLINEARITY_GAIN_THRESHOLD
    ):
        recommendation = LINEAR_LIKELY_SUFFICIENT
        decision_path.append(
            "The linear probe performed close to the best available probe."
        )
    elif (
        overlap > HIGH_OVERLAP_THRESHOLD
        and nonlinearity < OVERLAP_NONLINEARITY_GAIN_THRESHOLD
    ):
        recommendation = FEATURE_OR_LABEL_BOTTLENECK_LIKELY
        decision_path.append("Local overlap was high without much nonlinear gain.")
    elif nonlinearity >= MEANINGFUL_NONLINEARITY_GAIN_THRESHOLD:
        best_probe = metrics["baseline"]["best_probe"]
        smooth_score = probes.get("smooth_poly", {}).get("balanced_accuracy")
        knn_score = probes.get("knn", {}).get("balanced_accuracy")
        kernel_score = probes.get("kernel_approx", {}).get("balanced_accuracy")
        best_local_kernel = max(
            [score for score in (knn_score, kernel_score) if score is not None],
            default=None,
        )
        smooth_margin = (
            float(smooth_score - best_local_kernel)
            if smooth_score is not None and best_local_kernel is not None
            else None
        )
        decision_path.append(
            "Best nonlinear probe improved over the linear probe, "
            f"with {best_probe} performing best."
        )
        if fragmentation >= HIGH_FRAGMENTATION_THRESHOLD:
            recommendation = HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
            decision_path.append("Boundary graph fragmentation was high.")
        elif smooth_score is not None and (
            best_local_kernel is None
            or smooth_margin is not None
            and smooth_margin >= SMOOTH_MARGIN_TOLERANCE
        ):
            recommendation = SMOOTH_NONLINEAR_RECOMMENDED
            decision_path.append(
                "A smooth global nonlinear probe clearly outperformed the "
                "local and kernel-style probes."
            )
        elif topology >= STRONG_TOPOLOGY_THRESHOLD and (
            smooth_margin is None or smooth_margin < SMOOTH_MARGIN_TOLERANCE
        ):
            recommendation = KERNEL_OR_LOCAL_RECOMMENDED
            decision_path.append(
                "Persistent topology suggested nontrivial local structure."
            )
        elif (
            smooth_score is not None
            and smooth_margin is not None
            and smooth_margin >= -SMOOTH_MARGIN_TOLERANCE
        ):
            recommendation = SMOOTH_NONLINEAR_RECOMMENDED
            decision_path.append(
                "A smooth global nonlinear probe was competitive with the "
                "local and kernel-style probes."
            )
        elif best_local_kernel is not None:
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
        if reliability >= HIGH_CONFIDENCE_RELIABILITY_THRESHOLD
        and signal >= HIGH_CONFIDENCE_SIGNAL_THRESHOLD
        else "medium"
        if reliability >= MEDIUM_CONFIDENCE_RELIABILITY_THRESHOLD
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
