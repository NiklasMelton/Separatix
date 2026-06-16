"""Recommendation logic for multilabel diagnostics."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from separatix.constants import (
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
    INCONCLUSIVE,
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY,
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)
from separatix.recommendation.engine import SIGNAL_CONFIDENCE_Z

PRIMARY_METRICS = ("micro_f1", "macro_f1", "sample_jaccard")
FAMILY_PROBES = {
    "dummy": ("dummy",),
    "linear": ("linear",),
    "smooth_nonlinear": ("smooth_poly",),
    "local_kernel": ("knn", "kernel_approx"),
}
FAMILY_ORDER = ("linear", "smooth_nonlinear", "local_kernel")


def _metric_error(result: dict[str, Any], metric: str, n_samples: int) -> float:
    """Return a conservative metric uncertainty estimate."""
    stability = result.get(f"stability_{metric}_std")
    if stability is not None:
        return float(stability)
    score = result.get(metric)
    if score is None:
        return 1.0
    return float(
        sqrt(max(0.0, float(score) * (1.0 - float(score))) / max(1, n_samples))
    )


def _best_probe_for_metric(
    probes: dict[str, dict[str, Any]],
    family: str,
    metric: str,
    n_samples: int,
) -> dict[str, Any]:
    """Return the best probe evidence for one family and metric."""
    candidates = []
    for probe_name in FAMILY_PROBES[family]:
        result = probes.get(probe_name, {})
        if metric in result:
            candidates.append(
                {
                    "probe": probe_name,
                    "score": float(result[metric]),
                    "standard_error": _metric_error(result, metric, n_samples),
                    "available": True,
                }
            )
    if not candidates:
        return {
            "probe": None,
            "score": None,
            "standard_error": None,
            "available": False,
        }
    return max(candidates, key=lambda item: item["score"])


def _family_metric_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build per-family, per-metric evidence tables."""
    probes = metrics.get("probes", {})
    n_samples = int(metrics.get("audit", {}).get("n_samples", 1))
    evidence: dict[str, Any] = {}
    for family in ("dummy", *FAMILY_ORDER):
        evidence[family] = {
            metric: _best_probe_for_metric(probes, family, metric, n_samples)
            for metric in PRIMARY_METRICS
        }
    return evidence


def _combined_error(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    """Return combined uncertainty for two metric evidence entries."""
    if first["standard_error"] is None or second["standard_error"] is None:
        return None
    return float(sqrt(first["standard_error"] ** 2 + second["standard_error"] ** 2))


def _clearly_better(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether first clearly beats second for higher-is-better metrics."""
    if first["score"] is None or second["score"] is None:
        return False
    error = _combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(first["score"] - second["score"] > SIGNAL_CONFIDENCE_Z * tolerance)


def _within_one_standard_error(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether first is within one standard error of second."""
    if first["score"] is None or second["score"] is None:
        return False
    error = _combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(second["score"] - first["score"] <= tolerance)


def _best_predictive_family_for_metric(
    evidence: dict[str, Any], metric: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the best predictive family for one metric."""
    candidates = [
        (family, evidence[family][metric])
        for family in FAMILY_ORDER
        if evidence[family][metric]["score"] is not None
    ]
    if not candidates:
        return None, None
    return max(
        candidates,
        key=lambda item: (
            item[1]["score"],
            -FAMILY_ORDER.index(item[0]),
        ),
    )


def _comparison_counts(
    evidence: dict[str, Any], first: str, second: str
) -> dict[str, Any]:
    """Count clear and within-error comparisons across primary metrics."""
    clear = []
    within = []
    worse = []
    for metric in PRIMARY_METRICS:
        first_item = evidence[first][metric]
        second_item = evidence[second][metric]
        if _clearly_better(first_item, second_item):
            clear.append(metric)
        if _within_one_standard_error(first_item, second_item):
            within.append(metric)
        if _clearly_better(second_item, first_item):
            worse.append(metric)
    return {
        "first_family": first,
        "second_family": second,
        "clear_metrics": clear,
        "within_one_standard_error_metrics": within,
        "clearly_worse_metrics": worse,
    }


def _quality_flags(
    metrics: dict[str, Any],
    family_metrics: dict[str, Any],
    skipped_count: int,
    warning_count: int,
) -> list[dict[str, Any]]:
    """Return multilabel evidence quality flags."""
    flags: list[dict[str, Any]] = []
    if metrics.get("audit", {}).get("usable_label_count", 0) <= 0:
        flags.append(
            {
                "name": "no_usable_labels",
                "severity": "blocking",
                "message": (
                    "No multilabel columns had enough positive and negative examples."
                ),
            }
        )
    for family in ("dummy", "linear"):
        if not any(
            family_metrics[family][metric]["available"] for metric in PRIMARY_METRICS
        ):
            flags.append(
                {
                    "name": f"missing_{family}_probe",
                    "severity": "blocking",
                    "message": f"The {family} probe did not produce primary metrics.",
                }
            )
    if not any(
        family_metrics[family][metric]["available"]
        for family in FAMILY_ORDER
        for metric in PRIMARY_METRICS
    ):
        flags.append(
            {
                "name": "all_primary_metrics_unavailable",
                "severity": "blocking",
                "message": (
                    "No predictive family produced any primary multilabel metric."
                ),
            }
        )
    if skipped_count:
        flags.append(
            {
                "name": "skipped_diagnostics_present",
                "severity": "caution",
                "count": int(skipped_count),
                "message": (
                    "Some diagnostics were skipped and are listed in the report."
                ),
            }
        )
    if warning_count:
        flags.append(
            {
                "name": "warnings_present",
                "severity": "caution",
                "count": int(warning_count),
                "message": "Warnings were recorded while computing diagnostics.",
            }
        )
    return flags


def _has_blocking(flags: list[dict[str, Any]]) -> bool:
    return any(flag.get("severity") == "blocking" for flag in flags)


def compute_multilabel_scores(
    metrics: dict[str, Any],
    *,
    skipped_count: int,
    warning_count: int,
) -> dict[str, float | None]:
    """Compute transparent multilabel compatibility scores and evidence."""
    family_metrics = _family_metric_evidence(metrics)
    signal_metrics = []
    best_by_metric: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        best_family, best_item = _best_predictive_family_for_metric(
            family_metrics, metric
        )
        best_by_metric[metric] = {
            "family": best_family,
            "probe": best_item["probe"] if best_item else None,
            "score": best_item["score"] if best_item else None,
        }
        if best_item is not None and _clearly_better(
            best_item, family_metrics["dummy"][metric]
        ):
            signal_metrics.append(metric)

    comparisons = {
        "linear_vs_best": {},
        "smooth_vs_linear": _comparison_counts(
            family_metrics, "smooth_nonlinear", "linear"
        ),
        "local_kernel_vs_smooth": _comparison_counts(
            family_metrics, "local_kernel", "smooth_nonlinear"
        ),
    }
    for metric in PRIMARY_METRICS:
        best_family = best_by_metric[metric]["family"]
        if best_family is None:
            continue
        comparisons["linear_vs_best"][metric] = {
            "best_family": best_family,
            "linear_within_one_standard_error": _within_one_standard_error(
                family_metrics["linear"][metric],
                family_metrics[best_family][metric],
            ),
            "linear_clearly_worse": _clearly_better(
                family_metrics[best_family][metric],
                family_metrics["linear"][metric],
            ),
        }

    flags = _quality_flags(metrics, family_metrics, skipped_count, warning_count)
    evidence = {
        "selection_rule": (
            "Compare probe families across micro F1, macro F1, and sample "
            "Jaccard without collapsing them into a weighted aggregate."
        ),
        "primary_metrics": list(PRIMARY_METRICS),
        "signal_confidence_z": SIGNAL_CONFIDENCE_Z,
        "families": family_metrics,
        "best_by_metric": best_by_metric,
        "signal_metrics_beating_dummy": signal_metrics,
        "best_clearly_beats_dummy_on_two_primary_metrics": len(signal_metrics) >= 2,
        "family_comparisons": comparisons,
        "quality_flags": flags,
        "quality_score": float(
            np.mean([flag.get("severity") != "blocking" for flag in flags])
        )
        if flags
        else 1.0,
    }
    metrics["multilabel_recommendation_evidence"] = evidence
    return {
        "signal_micro_f1": _score_gap(family_metrics, "micro_f1"),
        "signal_macro_f1": _score_gap(family_metrics, "macro_f1"),
        "signal_sample_jaccard": _score_gap(family_metrics, "sample_jaccard"),
        "linearity_micro_f1": _linearity_score(family_metrics, "micro_f1"),
        "linearity_macro_f1": _linearity_score(family_metrics, "macro_f1"),
        "linearity_sample_jaccard": _linearity_score(family_metrics, "sample_jaccard"),
        "neighborhood_coherence_score": metrics.get("neighborhood", {}).get(
            "mean_neighbor_jaccard"
        ),
        "reliability_score": evidence["quality_score"],
    }


def _score_gap(family_metrics: dict[str, Any], metric: str) -> float | None:
    """Return best predictive score improvement over dummy for one metric."""
    _, best = _best_predictive_family_for_metric(family_metrics, metric)
    dummy = family_metrics["dummy"][metric]["score"]
    if best is None or best["score"] is None or dummy is None:
        return None
    return float(max(0.0, best["score"] - dummy))


def _linearity_score(family_metrics: dict[str, Any], metric: str) -> float | None:
    """Return linear score as a fraction of the best predictive score."""
    _, best = _best_predictive_family_for_metric(family_metrics, metric)
    linear = family_metrics["linear"][metric]["score"]
    if best is None or best["score"] is None or linear is None:
        return None
    return float(np.clip(linear / max(best["score"], 1e-9), 0.0, 1.0))


def make_multilabel_recommendation(
    scores: dict[str, float | None],
    metrics: dict[str, Any],
) -> tuple[str, str, list[str], dict[str, str]]:
    """Generate a conservative multilabel recommendation and decision path."""
    evidence = metrics["multilabel_recommendation_evidence"]
    flags = evidence["quality_flags"]
    decision_path = [
        "This run used the multilabel diagnostic path; probe families were "
        "compared across micro F1, macro F1, and sample Jaccard."
    ]
    if _has_blocking(flags):
        recommendation = INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
        decision_path.append(
            "Essential multilabel probe evidence was unavailable, so the result "
            "is limited to data sufficiency and diagnostic reliability."
        )
    elif not evidence["best_clearly_beats_dummy_on_two_primary_metrics"]:
        neighborhood = metrics.get("neighborhood", {})
        if neighborhood.get("mean_neighbor_jaccard", 1.0) < 0.2:
            recommendation = FEATURE_OR_LABEL_BOTTLENECK_LIKELY
        else:
            recommendation = INCONCLUSIVE
        decision_path.append(
            "The best predictive family did not clearly beat the per-label "
            "prevalence baseline on at least two primary multilabel metrics."
        )
    else:
        comparisons = evidence["family_comparisons"]
        linear_entries = comparisons["linear_vs_best"].values()
        linear_within = sum(
            bool(item["linear_within_one_standard_error"]) for item in linear_entries
        )
        linear_worse = sum(
            bool(item["linear_clearly_worse"]) for item in linear_entries
        )
        smooth_clear = len(comparisons["smooth_vs_linear"]["clear_metrics"])
        local_clear = len(comparisons["local_kernel_vs_smooth"]["clear_metrics"])
        if linear_within >= 2 and linear_worse == 0:
            recommendation = LINEAR_LIKELY_SUFFICIENT
            decision_path.append(
                "The linear family was within uncertainty of the best family on "
                "at least two primary metrics and was not clearly worse on the third."
            )
        elif smooth_clear >= 2 and local_clear < 2:
            recommendation = SMOOTH_NONLINEAR_RECOMMENDED
            decision_path.append(
                "Smooth nonlinear probes clearly improved over linear on at "
                "least two primary multilabel metrics."
            )
        elif local_clear >= 2:
            recommendation = KERNEL_OR_LOCAL_RECOMMENDED
            decision_path.append(
                "Local or kernel-style probes clearly improved over smooth "
                "nonlinear probes on at least two primary multilabel metrics."
            )
        else:
            recommendation = INCONCLUSIVE
            decision_path.append(
                "Primary multilabel metrics disagreed across probe families, "
                "so no single model-family recommendation was forced."
            )

    if flags:
        decision_path.append(
            "Evidence quality flags: "
            + ", ".join(str(flag["name"]) for flag in flags)
            + "."
        )
    confidence = "low" if recommendation == INCONCLUSIVE else "medium"
    if recommendation == INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY:
        confidence = "low"
    elif not any(flag.get("severity") == "caution" for flag in flags):
        confidence = "high"
    interpretations = {
        "multilabel_recommendation_evidence": (
            "Multilabel probe families are compared across separate primary "
            "metrics; no weighted aggregate is used for the decision."
        ),
        "signal": "Higher values mean labels appear predictable beyond prevalence.",
        "linearity": "Higher values mean linear probes are close to the best family.",
        "neighborhood": "Higher values mean nearby samples share label sets.",
        "reliability": (
            "Higher values mean essential multilabel evidence was available."
        ),
    }
    decision_path.append(
        "Signal metrics beating dummy: "
        + ", ".join(evidence["signal_metrics_beating_dummy"] or ["none"])
        + "."
    )
    return recommendation, confidence, decision_path, interpretations
