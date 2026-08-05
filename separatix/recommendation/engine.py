"""Recommendation engine for separatix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np

from separatix.constants import (
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
    FEATURE_OR_TARGET_BOTTLENECK_LIKELY,
    FEEDFORWARD_MLP_RECOMMENDED,
    FEEDFORWARD_MLP_REGRESSION_RECOMMENDED,
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED,
    HIGH_CAPACITY_OR_PARTITIONING_REGRESSION_RECOMMENDED,
    INCONCLUSIVE,
    INCONCLUSIVE_REGRESSION_DIAGNOSTIC,
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY,
    INSUFFICIENT_DATA_OR_UNRELIABLE_REGRESSION_GEOMETRY,
    KERNEL_OR_LOCAL_RECOMMENDED,
    KERNEL_OR_LOCAL_REGRESSION_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    LINEAR_RESPONSE_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
    SMOOTH_NONLINEAR_RESPONSE_RECOMMENDED,
)
from separatix.models.comparison import lookup_paired_comparison

MIN_NORMALIZED_SCORE = 0.0
MAX_NORMALIZED_SCORE = 1.0
NORMALIZATION_EPSILON = 1e-9
ONE_STANDARD_ERROR = 1.0
# Standard normal two-sided 95% critical value used to decide whether any
# predictive probe has evidence above the class-prior dummy baseline.
SIGNAL_CONFIDENCE_Z = 1.96
MIN_SIGNAL_MARGIN = 0.05
MIN_FAMILY_IMPROVEMENT = 0.01

_FAMILY_ORDER = ("linear", "smooth_nonlinear", "local_kernel")
_FAMILY_PROBES = {
    "dummy": ("dummy",),
    "linear": ("linear",),
    "smooth_nonlinear": ("smooth_poly",),
    "local_kernel": ("knn", "kernel_approx"),
}
_FAMILY_RECOMMENDATIONS = {
    "linear": LINEAR_LIKELY_SUFFICIENT,
    "smooth_nonlinear": SMOOTH_NONLINEAR_RECOMMENDED,
    "local_kernel": KERNEL_OR_LOCAL_RECOMMENDED,
}
_PROBE_DISPLAY_NAMES = {
    "dummy": "class-prior dummy",
    "linear": "linear",
    "smooth_poly": "smooth nonlinear",
    "knn": "local k-nearest-neighbor",
    "kernel_approx": "kernel approximation",
}
_MULTILABEL_PRIMARY_METRICS = ("micro_f1", "macro_f1", "sample_jaccard")
_REGRESSION_PRIMARY_METRICS = ("r2_variance_weighted", "r2_uniform_average")
_REGRESSION_FAMILY_RECOMMENDATIONS = {
    "linear": LINEAR_RESPONSE_LIKELY_SUFFICIENT,
    "smooth_nonlinear": SMOOTH_NONLINEAR_RESPONSE_RECOMMENDED,
    "local_kernel": KERNEL_OR_LOCAL_REGRESSION_RECOMMENDED,
}


def _context_quality_flags(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate structured diagnostic outcomes into recommendation flags."""
    flags: list[dict[str, Any]] = []
    entries = metrics.get("quality_context", {}).get("skipped_diagnostics", [])
    for entry in entries:
        severity = entry.get("severity")
        if severity not in {"blocking", "caution"}:
            continue
        flags.append(
            {
                "name": str(entry.get("name", "diagnostic_failure")),
                "severity": severity,
                "message": str(entry.get("reason", "A diagnostic was unavailable.")),
            }
        )
    if metrics.get("quality_context", {}).get("errors"):
        flags.append(
            {
                "name": "diagnostic_errors",
                "severity": "caution",
                "message": "One or more optional diagnostics failed at runtime.",
            }
        )
    return flags


@dataclass(frozen=True)
class _ProbeEvidence:
    name: str
    family: str
    score: float
    standard_error: float
    evaluation_mode: str | None
    stability_standard_deviation: float | None


@dataclass(frozen=True)
class _FamilyEvidence:
    family: str
    available: bool
    complexity_rank: int | None
    best_probe: str | None
    probe_names: list[str]
    score: float | None
    standard_error: float | None
    score_lower_bound: float | None
    score_upper_bound: float | None


def _clip(value: float | None) -> float | None:
    if value is None:
        return None
    return float(np.clip(value, MIN_NORMALIZED_SCORE, MAX_NORMALIZED_SCORE))


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _class_counts(metrics: dict[str, Any]) -> list[int]:
    counts = metrics.get("audit", {}).get("class_counts", {})
    return [int(count) for count in counts.values()]


def _class_proportions(metrics: dict[str, Any]) -> list[float]:
    proportions = metrics.get("audit", {}).get("class_proportions")
    if proportions:
        return [float(value) for value in proportions.values()]
    counts = _class_counts(metrics)
    total = sum(counts)
    return [count / total for count in counts] if total else []


def _balanced_accuracy_standard_error(
    result: dict[str, Any],
    class_counts: list[int],
) -> float:
    score = float(result["balanced_accuracy"])
    support_counts = result.get("evaluation_support", {}).get("class_counts")
    if isinstance(support_counts, dict) and support_counts:
        class_counts = [int(count) for count in support_counts.values()]
    recalls = result.get("per_class_recall")
    if (
        isinstance(recalls, list)
        and len(recalls) == len(class_counts)
        and len(class_counts) > 0
    ):
        recall_values = [float(recall) for recall in recalls]
    else:
        recall_values = [score for _ in class_counts]

    if recall_values and class_counts:
        variance = sum(
            recall * (MAX_NORMALIZED_SCORE - recall) / max(1, class_counts[index])
            for index, recall in enumerate(recall_values)
        ) / (len(recall_values) ** 2)
        class_aware_error = sqrt(max(MIN_NORMALIZED_SCORE, variance))
    else:
        class_aware_error = MIN_NORMALIZED_SCORE

    stability_error = _numeric(result.get("stability_balanced_accuracy_std"))
    if stability_error is None:
        return float(class_aware_error)
    return float(max(class_aware_error, stability_error))


def _probe_evidence(
    metrics: dict[str, Any],
) -> dict[str, _ProbeEvidence]:
    class_counts = _class_counts(metrics)
    probes = metrics.get("probes", {})
    evidence: dict[str, _ProbeEvidence] = {}
    for family, probe_names in _FAMILY_PROBES.items():
        for name in probe_names:
            result = probes.get(name, {})
            if "balanced_accuracy" not in result:
                continue
            evidence[name] = _ProbeEvidence(
                name=name,
                family=family,
                score=float(result["balanced_accuracy"]),
                standard_error=_balanced_accuracy_standard_error(result, class_counts),
                evaluation_mode=result.get("evaluation_mode"),
                stability_standard_deviation=_numeric(
                    result.get("stability_balanced_accuracy_std")
                ),
            )
    return evidence


def _complexity_rank(family: str) -> int | None:
    if family == "dummy":
        return 0
    if family in _FAMILY_ORDER:
        return _FAMILY_ORDER.index(family) + 1
    return None


def _best_probe_for_family(
    family: str,
    probes: dict[str, _ProbeEvidence],
) -> _ProbeEvidence | None:
    candidates = [probes[name] for name in _FAMILY_PROBES[family] if name in probes]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.score)


def _family_evidence(
    probes: dict[str, _ProbeEvidence],
) -> dict[str, _FamilyEvidence]:
    evidence: dict[str, _FamilyEvidence] = {}
    for family, probe_names in _FAMILY_PROBES.items():
        best_probe = _best_probe_for_family(family, probes)
        if best_probe is None:
            evidence[family] = _FamilyEvidence(
                family=family,
                available=False,
                complexity_rank=_complexity_rank(family),
                best_probe=None,
                probe_names=list(probe_names),
                score=None,
                standard_error=None,
                score_lower_bound=None,
                score_upper_bound=None,
            )
            continue
        standard_error = best_probe.standard_error
        evidence[family] = _FamilyEvidence(
            family=family,
            available=True,
            complexity_rank=_complexity_rank(family),
            best_probe=best_probe.name,
            probe_names=list(probe_names),
            score=best_probe.score,
            standard_error=standard_error,
            score_lower_bound=_clip(best_probe.score - standard_error),
            score_upper_bound=_clip(best_probe.score + standard_error),
        )
    return evidence


def _best_predictive_family(
    families: dict[str, _FamilyEvidence],
) -> _FamilyEvidence | None:
    candidates = [
        family
        for family_name, family in families.items()
        if family_name in _FAMILY_ORDER and family.score is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda family: (
            family.score if family.score is not None else MIN_NORMALIZED_SCORE,
            -(_complexity_rank(family.family) or 0),
        ),
    )


def _combined_standard_error(
    first: _FamilyEvidence | None,
    second: _FamilyEvidence | None,
) -> float | None:
    if (
        first is None
        or second is None
        or first.standard_error is None
        or second.standard_error is None
    ):
        return None
    return float(sqrt(first.standard_error**2 + second.standard_error**2))


def _family_within_one_standard_error(
    family: _FamilyEvidence,
    best_family: _FamilyEvidence,
    paired_payload: dict[str, Any] | None = None,
) -> bool:
    if family.score is None or best_family.score is None:
        return False
    paired = lookup_paired_comparison(
        paired_payload or {},
        best_family.best_probe,
        family.best_probe,
        "balanced_accuracy",
    )
    if paired is not None:
        tolerance = float(paired["paired_standard_error"])
        return bool(
            float(paired["point_delta"]) <= max(tolerance, MIN_FAMILY_IMPROVEMENT)
        )
    combined_error = _combined_standard_error(family, best_family)
    tolerance = ONE_STANDARD_ERROR * (
        combined_error if combined_error is not None else MIN_NORMALIZED_SCORE
    )
    return bool(
        best_family.score - family.score <= max(tolerance, MIN_FAMILY_IMPROVEMENT)
    )


def _family_comparison(
    first: _FamilyEvidence,
    second: _FamilyEvidence,
    paired_payload: dict[str, Any] | None = None,
) -> dict[str, float | bool | str | None]:
    if first.score is None or second.score is None:
        return {
            "first_family": first.family,
            "second_family": second.family,
            "score_gap": None,
            "combined_standard_error": None,
            "z_score": None,
            "first_clearly_better": False,
            "second_clearly_better": False,
            "decision_method": "unavailable",
            "fallback_reason": "one or both family scores are unavailable",
        }
    paired = lookup_paired_comparison(
        paired_payload or {}, first.best_probe, second.best_probe, "balanced_accuracy"
    )
    combined_error = _combined_standard_error(first, second)
    score_gap = first.score - second.score
    if paired is not None:
        paired_error = float(paired["paired_standard_error"])
        point_delta = float(paired["point_delta"])
        lower = float(paired["lower_95"])
        upper = float(paired["upper_95"])
        return {
            "first_family": first.family,
            "second_family": second.family,
            "first_probe": first.best_probe,
            "second_probe": second.best_probe,
            "score_gap": point_delta,
            "combined_standard_error": combined_error,
            "paired_standard_error": paired_error,
            "lower_95": lower,
            "upper_95": upper,
            "z_score": point_delta / paired_error if paired_error > 0 else None,
            "clear_advantage_z": SIGNAL_CONFIDENCE_Z,
            "first_clearly_better": bool(
                point_delta > MIN_FAMILY_IMPROVEMENT and lower > 0.0
            ),
            "second_clearly_better": bool(
                -point_delta > MIN_FAMILY_IMPROVEMENT and upper < 0.0
            ),
            "decision_method": "paired_oof_bootstrap",
            "fallback_reason": None,
        }
    tolerance = combined_error or MIN_NORMALIZED_SCORE
    z_score = score_gap / tolerance if tolerance > 0 else None
    return {
        "first_family": first.family,
        "second_family": second.family,
        "score_gap": float(score_gap),
        "combined_standard_error": combined_error,
        "z_score": float(z_score) if z_score is not None else None,
        "clear_advantage_z": SIGNAL_CONFIDENCE_Z,
        "first_clearly_better": bool(
            score_gap > max(SIGNAL_CONFIDENCE_Z * tolerance, MIN_FAMILY_IMPROVEMENT)
        ),
        "second_clearly_better": bool(
            -score_gap > max(SIGNAL_CONFIDENCE_Z * tolerance, MIN_FAMILY_IMPROVEMENT)
        ),
        "decision_method": "marginal_standard_error_fallback",
        "fallback_reason": "paired aligned predictions were unavailable",
    }


def _recommended_family(
    families: dict[str, _FamilyEvidence],
    raw_best_family: _FamilyEvidence | None,
    paired_payload: dict[str, Any] | None = None,
) -> str | None:
    """Choose a family by conservative escalation from simpler to more complex."""
    if raw_best_family is None:
        return None
    linear = families["linear"]
    smooth = families["smooth_nonlinear"]
    local = families["local_kernel"]

    if _family_within_one_standard_error(linear, raw_best_family, paired_payload):
        return "linear"
    if smooth.score is None:
        return local.family if local.score is not None else raw_best_family.family
    if local.score is None:
        return smooth.family

    local_vs_smooth = _family_comparison(local, smooth, paired_payload)
    if local_vs_smooth["first_clearly_better"]:
        return local.family
    return smooth.family


def _topology_score(metrics: dict[str, Any]) -> float | None:
    topology = metrics.get("topology", {})
    if "topology_strength" in topology:
        return _clip(topology.get("topology_strength", MIN_NORMALIZED_SCORE))
    if "h1_persistence_count" in topology:
        return _clip(
            min(
                MAX_NORMALIZED_SCORE,
                topology.get("max_h1_persistence", MIN_NORMALIZED_SCORE),
            )
        )
    return None


def _geometry_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    proportions = _class_proportions(metrics)
    expected_cross_class_fraction = (
        MAX_NORMALIZED_SCORE - sum(proportion**2 for proportion in proportions)
        if proportions
        else None
    )
    observed_cross_class_fraction = _numeric(
        metrics.get("neighborhood", {}).get("cross_class_neighbor_fraction")
    )
    overlap_vs_label_shuffle = (
        observed_cross_class_fraction / expected_cross_class_fraction
        if observed_cross_class_fraction is not None
        and expected_cross_class_fraction is not None
        and expected_cross_class_fraction > 0
        else None
    )

    graph = metrics.get("graph", {})
    largest_component_fraction = _numeric(graph.get("largest_component_fraction"))
    component_count = graph.get("component_count")
    class_majority_fraction = max(proportions) if proportions else None
    boundary_fragmentation_supported = (
        bool(
            component_count is not None
            and int(component_count)
            > max(1, metrics.get("audit", {}).get("n_classes", 1))
            and largest_component_fraction is not None
            and class_majority_fraction is not None
            and largest_component_fraction < class_majority_fraction
        )
        if graph
        else False
    )

    topology = metrics.get("topology", {})
    return {
        "expected_cross_class_neighbor_fraction_under_label_shuffle": (
            float(expected_cross_class_fraction)
            if expected_cross_class_fraction is not None
            else None
        ),
        "observed_cross_class_neighbor_fraction": observed_cross_class_fraction,
        "overlap_vs_label_shuffle": (
            float(overlap_vs_label_shuffle)
            if overlap_vs_label_shuffle is not None
            else None
        ),
        "graph_fragmentation_score": _numeric(graph.get("graph_fragmentation_score")),
        "graph_fragmentation_bootstrap_repeats": graph.get(
            "graph_fragmentation_bootstrap_repeats"
        ),
        "graph_fragmentation_bootstrap_mean": _numeric(
            graph.get("graph_fragmentation_bootstrap_mean")
        ),
        "graph_fragmentation_bootstrap_std": _numeric(
            graph.get("graph_fragmentation_bootstrap_std")
        ),
        "largest_boundary_component_fraction": largest_component_fraction,
        "boundary_fragmentation_supported": boundary_fragmentation_supported,
        "topology_score": _topology_score(metrics),
        "topology_available": "topology_strength" in topology
        or "h1_persistence_count" in topology,
        "topology_note": topology.get("skipped_reason"),
    }


def _quality_flags(
    metrics: dict[str, Any],
    probes: dict[str, _ProbeEvidence],
    families: dict[str, _FamilyEvidence],
    *,
    skipped_count: int,
    warning_count: int,
    best_family: _FamilyEvidence | None,
    dummy_family: _FamilyEvidence,
    paired_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if not dummy_family.available:
        flags.append(
            {
                "name": "missing_dummy_probe",
                "severity": "blocking",
                "message": "The class-prior dummy probe is unavailable.",
            }
        )
    if not families["linear"].available:
        flags.append(
            {
                "name": "missing_linear_probe",
                "severity": "blocking",
                "message": "The linear probe is unavailable.",
            }
        )
    if best_family is None:
        flags.append(
            {
                "name": "missing_predictive_probe",
                "severity": "blocking",
                "message": "No predictive probe produced a balanced accuracy.",
            }
        )

    if any(
        probe.evaluation_mode == "resubstitution_low_reliability"
        for probe in probes.values()
    ):
        flags.append(
            {
                "name": "resubstitution_evaluation",
                "severity": "blocking",
                "message": (
                    "At least one probe used in-sample evaluation because there "
                    "were too few examples for stratified validation."
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

    geometry = metrics.get("geometry", {})
    if geometry.get("distance_concentration_proxy") is None:
        flags.append(
            {
                "name": "geometry_diagnostics_unavailable",
                "severity": "info",
                "message": "Geometry concentration diagnostics are unavailable.",
            }
        )
    if metrics.get("topology", {}).get("skipped_reason") is not None:
        flags.append(
            {
                "name": "topology_diagnostics_unavailable",
                "severity": "info",
                "message": (
                    "Persistent topology was unavailable or skipped; it will only "
                    "appear as supporting evidence when present."
                ),
            }
        )
    if metrics.get("graph", {}).get("warning") is not None:
        flags.append(
            {
                "name": "boundary_graph_limited",
                "severity": "info",
                "message": metrics["graph"]["warning"],
            }
        )

    signal_error = _combined_standard_error(best_family, dummy_family)
    paired_signal = (
        lookup_paired_comparison(
            paired_payload or {},
            best_family.best_probe,
            dummy_family.best_probe,
            "balanced_accuracy",
        )
        if best_family is not None
        else None
    )
    if (
        best_family is not None
        and best_family.score is not None
        and dummy_family.score is not None
        and (paired_signal is not None or signal_error is not None)
        and (
            (
                paired_signal is not None
                and (
                    float(paired_signal["point_delta"]) <= MIN_SIGNAL_MARGIN
                    or float(paired_signal["lower_95"]) <= 0.0
                )
            )
            or (
                paired_signal is None
                and signal_error is not None
                and (
                    best_family.score - dummy_family.score <= MIN_SIGNAL_MARGIN
                    or (SIGNAL_CONFIDENCE_Z * signal_error)
                    >= abs(best_family.score - dummy_family.score)
                )
            )
        )
    ):
        flags.append(
            {
                "name": "weak_signal_evidence",
                "severity": "caution",
                "message": (
                    "The best probe's advantage over the dummy baseline does not "
                    "clear the configured uncertainty-aware signal check."
                ),
            }
        )
    local_vs_smooth = _family_comparison(
        families["local_kernel"], families["smooth_nonlinear"], paired_payload
    )
    local_score = families["local_kernel"].score
    smooth_score = families["smooth_nonlinear"].score
    if (
        local_score is not None
        and smooth_score is not None
        and local_score > smooth_score
        and not local_vs_smooth["first_clearly_better"]
    ):
        flags.append(
            {
                "name": "borderline_family_difference",
                "severity": "caution",
                "message": (
                    "Local/kernel probes were numerically best but did not clearly "
                    "beat the smooth nonlinear family."
                ),
            }
        )
    existing = {flag["name"] for flag in flags}
    flags.extend(
        flag for flag in _context_quality_flags(metrics) if flag["name"] not in existing
    )
    return flags


def _quality_score(flags: list[dict[str, Any]]) -> float:
    checks = [
        "missing_dummy_probe",
        "missing_linear_probe",
        "missing_predictive_probe",
        "resubstitution_evaluation",
        "weak_signal_evidence",
        "borderline_family_difference",
        "geometry_diagnostics_unavailable",
    ]
    failed_checks = {flag["name"] for flag in flags}
    passed_count = sum(check not in failed_checks for check in checks)
    return float(passed_count / len(checks))


def _probe_table(probes: dict[str, _ProbeEvidence]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(probe),
            "display_name": _PROBE_DISPLAY_NAMES.get(probe.name, probe.name),
            "score_lower_bound": _clip(probe.score - probe.standard_error),
            "score_upper_bound": _clip(probe.score + probe.standard_error),
        }
        for probe in sorted(
            probes.values(),
            key=lambda item: (_complexity_rank(item.family) or 0, item.name),
        )
    ]


def _family_table(
    families: dict[str, _FamilyEvidence],
) -> dict[str, dict[str, Any]]:
    return {name: asdict(family) for name, family in families.items()}


def _build_recommendation_evidence(
    metrics: dict[str, Any],
    *,
    skipped_count: int,
    warning_count: int,
) -> dict[str, Any]:
    probes = _probe_evidence(metrics)
    families = _family_evidence(probes)
    paired_payload = metrics.get("paired_probe_comparisons", {})
    raw_best_family = _best_predictive_family(families)
    dummy_family = families["dummy"]
    candidate_family = _recommended_family(families, raw_best_family, paired_payload)
    signal_error = _combined_standard_error(raw_best_family, dummy_family)
    signal_paired = (
        lookup_paired_comparison(
            paired_payload,
            raw_best_family.best_probe,
            dummy_family.best_probe,
            "balanced_accuracy",
        )
        if raw_best_family is not None
        else None
    )
    best_clearly_beats_dummy = (
        bool(
            raw_best_family is not None
            and raw_best_family.score is not None
            and dummy_family.score is not None
            and (
                (
                    signal_paired is not None
                    and float(signal_paired["point_delta"]) > MIN_SIGNAL_MARGIN
                    and float(signal_paired["lower_95"]) > 0.0
                )
                or (
                    signal_paired is None
                    and signal_error is not None
                    and raw_best_family.score - dummy_family.score
                    > SIGNAL_CONFIDENCE_Z * signal_error
                    and raw_best_family.score - dummy_family.score > MIN_SIGNAL_MARGIN
                )
            )
        )
        if dummy_family.available
        else False
    )
    recommended_family = candidate_family if best_clearly_beats_dummy else None

    smooth_vs_local = _family_comparison(
        families["local_kernel"], families["smooth_nonlinear"], paired_payload
    )
    raw_best_vs_recommended = (
        _family_comparison(
            families[raw_best_family.family],
            families[recommended_family],
            paired_payload,
        )
        if raw_best_family is not None and recommended_family is not None
        else None
    )
    quality_flags = _quality_flags(
        metrics,
        probes,
        families,
        skipped_count=skipped_count,
        warning_count=warning_count,
        best_family=raw_best_family,
        dummy_family=dummy_family,
        paired_payload=paired_payload,
    )
    best_score = raw_best_family.score if raw_best_family is not None else None
    dummy_score = dummy_family.score
    signal_margin = (
        best_score - dummy_score
        if best_score is not None and dummy_score is not None
        else None
    )

    return {
        "selection_rule": (
            "Use conservative escalation: keep simpler probe families unless "
            "a more complex family has a clear uncertainty-adjusted advantage."
        ),
        "standard_error_multiplier": ONE_STANDARD_ERROR,
        "signal_confidence_z": SIGNAL_CONFIDENCE_Z,
        "minimum_signal_margin": MIN_SIGNAL_MARGIN,
        "probe_table": _probe_table(probes),
        "families": _family_table(families),
        "raw_best_family": raw_best_family.family
        if raw_best_family is not None
        else None,
        "raw_best_probe": raw_best_family.best_probe
        if raw_best_family is not None
        else None,
        "candidate_family": candidate_family,
        "recommended_family": recommended_family,
        "best_family": raw_best_family.family if raw_best_family is not None else None,
        "best_probe": raw_best_family.best_probe
        if raw_best_family is not None
        else None,
        "selected_family": recommended_family,
        "signal_margin_over_dummy": (
            float(signal_margin) if signal_margin is not None else None
        ),
        "signal_combined_standard_error": signal_error,
        "signal_comparison": {
            **signal_paired,
            "decision_method": "paired_oof_bootstrap",
            "fallback_reason": None,
        }
        if signal_paired is not None
        else {
            "decision_method": "marginal_standard_error_fallback",
            "fallback_reason": "paired aligned predictions were unavailable",
            "point_delta": signal_margin,
            "paired_standard_error": None,
            "lower_95": None,
            "upper_95": None,
        },
        "signal_z_score": (
            float(signal_margin / signal_error)
            if signal_margin is not None
            and signal_error is not None
            and signal_error > 0
            else None
        ),
        "best_clearly_beats_dummy": best_clearly_beats_dummy,
        "family_comparisons": {
            "local_kernel_vs_smooth_nonlinear": smooth_vs_local,
            "raw_best_vs_recommended": raw_best_vs_recommended,
        },
        "geometry": _geometry_evidence(metrics),
        "quality_flags": quality_flags,
        "quality_score": _quality_score(quality_flags),
    }


def _score_from_evidence(
    metrics: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, float | None]:
    families = evidence["families"]
    dummy = families["dummy"]["score"]
    linear = families["linear"]["score"]
    best = (
        families[evidence["best_family"]]["score"]
        if evidence["best_family"] is not None
        else None
    )
    nonlinear_scores = [
        families[name]["score"]
        for name in ("smooth_nonlinear", "local_kernel")
        if families[name]["score"] is not None
    ]
    best_nonlinear = max(nonlinear_scores) if nonlinear_scores else None
    overlap = _clip(
        np.mean(
            [
                metrics.get("neighborhood", {}).get(
                    "mean_local_entropy", MIN_NORMALIZED_SCORE
                ),
                metrics.get("neighborhood", {}).get(
                    "high_entropy_fraction", MIN_NORMALIZED_SCORE
                ),
                metrics.get("neighborhood", {}).get(
                    "cross_class_neighbor_fraction", MIN_NORMALIZED_SCORE
                ),
            ]
        )
    )
    signal = (
        _clip((best - dummy) / max(NORMALIZATION_EPSILON, MAX_NORMALIZED_SCORE - dummy))
        if best is not None and dummy is not None
        else None
    )
    linearity = (
        _clip(linear / max(best, NORMALIZATION_EPSILON))
        if linear is not None and best is not None
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
    return {
        "signal_score": signal,
        "overlap_score": overlap,
        "linearity_score": linearity,
        "nonlinearity_score": nonlinear,
        "fragmentation_score": _clip(
            metrics.get("graph", {}).get(
                "graph_fragmentation_score", MIN_NORMALIZED_SCORE
            )
        ),
        "topology_score": _topology_score(metrics),
        "reliability_score": _clip(evidence["quality_score"]),
    }


def compute_scores(
    metrics: dict[str, Any], skipped_count: int, warning_count: int
) -> dict[str, float | None]:
    """Compute transparent compatibility scores and attach recommendation evidence."""
    evidence = _build_recommendation_evidence(
        metrics,
        skipped_count=skipped_count,
        warning_count=warning_count,
    )
    metrics["recommendation_evidence"] = evidence
    return _score_from_evidence(metrics, evidence)


def _has_blocking_quality_flag(evidence: dict[str, Any]) -> bool:
    return any(
        flag.get("severity") == "blocking" for flag in evidence.get("quality_flags", [])
    )


def _has_caution_quality_flag(evidence: dict[str, Any]) -> bool:
    return any(
        flag.get("severity") == "caution" for flag in evidence.get("quality_flags", [])
    )


def _weak_signal_recommendation(evidence: dict[str, Any]) -> str:
    geometry = evidence["geometry"]
    overlap_vs_null = geometry.get("overlap_vs_label_shuffle")
    if overlap_vs_null is not None and overlap_vs_null >= MAX_NORMALIZED_SCORE:
        return FEATURE_OR_LABEL_BOTTLENECK_LIKELY
    return INCONCLUSIVE


def _recommend_selected_family(evidence: dict[str, Any]) -> str:
    selected_family = evidence.get("selected_family")
    if selected_family == "local_kernel":
        comparison = evidence["family_comparisons"]["local_kernel_vs_smooth_nonlinear"]
        local_clearly_better = bool(comparison.get("first_clearly_better"))
        if local_clearly_better and evidence["geometry"].get(
            "boundary_fragmentation_supported"
        ):
            return HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
    if selected_family in _FAMILY_RECOMMENDATIONS:
        return _FAMILY_RECOMMENDATIONS[selected_family]
    return INCONCLUSIVE


def _confidence_from_evidence(
    recommendation: str,
    evidence: dict[str, Any],
) -> str:
    if recommendation == INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY:
        return "low"
    if recommendation == INCONCLUSIVE:
        return "low" if _has_caution_quality_flag(evidence) else "medium"
    if _has_blocking_quality_flag(evidence):
        return "low"
    if _has_caution_quality_flag(evidence):
        return "medium"
    return "high"


def _format_family_score(evidence: dict[str, Any], family: str) -> str:
    family_evidence = evidence["families"][family]
    score = family_evidence["score"]
    standard_error = family_evidence["standard_error"]
    if score is None:
        return "unavailable"
    return f"{score:.3f} +/- {standard_error:.3f}"


def _mlp_override_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the optional MLP override payload."""
    payload = metrics.get("mlp_recommendation_evidence", {})
    return payload if isinstance(payload, dict) else {}


def _mlp_override_active(metrics: dict[str, Any]) -> bool:
    """Return whether the optional MLP should override simpler families."""
    return bool(_mlp_override_payload(metrics).get("recommendation_override"))


def _mlp_architecture_note(metrics: dict[str, Any]) -> str | None:
    """Return a compact architecture note for the decision path."""
    best = _mlp_override_payload(metrics).get("best_architecture")
    if not isinstance(best, dict):
        return None
    probe_name = best.get("probe_name")
    if probe_name is None:
        return None
    return f"Selected optional MLP architecture: {probe_name}."


def _comparison_method_note(metrics: dict[str, Any]) -> str:
    """Return a decision-path note describing family-comparison uncertainty."""
    paired = metrics.get("paired_probe_comparisons", {})
    if paired.get("status") == "available":
        return (
            "Probe-family decisions used paired bootstrap intervals over aligned "
            "out-of-fold predictions."
        )
    return (
        "Paired out-of-fold comparisons were unavailable, so affected decisions "
        "used marginal uncertainty estimates."
    )


def make_recommendation(
    scores: dict[str, float | None], metrics: dict[str, Any]
) -> tuple[str, str, list[str], dict[str, str]]:
    """Generate an uncertainty-aware recommendation and decision path."""
    if "recommendation_evidence" not in metrics:
        metrics["recommendation_evidence"] = _build_recommendation_evidence(
            metrics,
            skipped_count=0,
            warning_count=0,
        )
    evidence = metrics["recommendation_evidence"]
    decision_path: list[str] = []
    interpretations: dict[str, str] = {}

    if _has_blocking_quality_flag(evidence):
        recommendation = INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
        decision_path.append(
            "Essential probe evidence was unavailable, so the recommendation is "
            "limited to data sufficiency and diagnostic reliability."
        )
    elif not evidence["best_clearly_beats_dummy"]:
        recommendation = _weak_signal_recommendation(evidence)
        decision_path.append(
            "The best predictive probe did not clear the 95% signal check "
            "against the class-prior dummy baseline."
        )
    else:
        raw_best_family = evidence["raw_best_family"]
        selected_family = evidence["recommended_family"]
        recommendation = _recommend_selected_family(evidence)
        if selected_family == "linear":
            decision_path.append(
                "The linear probe was statistically indistinguishable from the "
                "best observed probe family."
            )
        elif selected_family == "smooth_nonlinear":
            decision_path.append(
                "Nonlinear probes improved over linear, and local/kernel probes "
                "did not clearly beat the smooth nonlinear family."
            )
        elif selected_family == "local_kernel":
            decision_path.append(
                "Local or kernel-style probes clearly outperformed the smooth "
                "nonlinear family after uncertainty adjustment."
            )
            if evidence["geometry"].get("topology_score") is not None:
                decision_path.append(
                    "Topology diagnostics were available as supporting evidence."
                )
            if recommendation == HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED:
                decision_path.append(
                    "Local/kernel probes clearly beat smooth probes and the "
                    "boundary graph was fragmented relative to class balance."
                )
        else:
            decision_path.append(
                "No predictive family satisfied the evidence-selection rule."
            )
        decision_path.append(
            "Probe families were compared with conservative escalation: "
            f"raw_best={raw_best_family}, recommended={selected_family}."
        )

    if _mlp_override_active(metrics):
        mlp_payload = _mlp_override_payload(metrics)
        recommendation = FEEDFORWARD_MLP_RECOMMENDED
        decision_path.insert(
            0,
            "The optional MLP probe clearly improved over the aligned simpler "
            "families and satisfied the configured override criteria.",
        )
        decision_path.append(
            "Conditional MLP probes were only run because simpler probes did not "
            "meet the configured absolute-skill threshold."
        )
        if mlp_payload.get("override_reason") is not None:
            decision_path.append(str(mlp_payload["override_reason"]))
        architecture_note = _mlp_architecture_note(metrics)
        if architecture_note is not None:
            decision_path.append(architecture_note)

    quality_flags = evidence.get("quality_flags", [])
    if quality_flags:
        flag_names = ", ".join(flag["name"] for flag in quality_flags)
        decision_path.append(f"Evidence quality flags: {flag_names}.")

    decision_path.append(_comparison_method_note(metrics))

    confidence = _confidence_from_evidence(recommendation, evidence)
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
        "Higher values mean nonlinear probes improved over the linear probe."
    )
    interpretations["fragmentation"] = (
        "Higher values mean the estimated boundary looks more "
        "partitioned or locally broken up."
    )
    interpretations["reliability"] = (
        "Higher values mean essential evidence was available, uncertainty was "
        "bounded, and diagnostics avoided low-reliability fallbacks."
    )
    interpretations["recommendation_evidence"] = (
        "Probe-family scores are shown as balanced accuracy plus or minus an "
        "estimated standard error; selection uses conservative escalation from "
        "simpler to more complex families."
    )
    if metrics.get("mlp_recommendation_evidence") is not None:
        interpretations["mlp_recommendation_evidence"] = (
            "Optional MLP probes run only after simpler models miss a configured "
            "absolute-skill threshold, and they override the simpler-family "
            "recommendation only when the best tested architecture clearly beats "
            "every aligned simpler probe."
        )

    decision_path.append(
        "Family evidence: "
        f"linear={_format_family_score(evidence, 'linear')}, "
        f"smooth={_format_family_score(evidence, 'smooth_nonlinear')}, "
        f"local/kernel={_format_family_score(evidence, 'local_kernel')}."
    )
    return recommendation, confidence, decision_path, interpretations


def _multilabel_metric_error(
    result: dict[str, Any], metric: str, n_samples: int
) -> float:
    """Return a conservative multilabel metric uncertainty estimate."""
    stability = result.get(f"stability_{metric}_std")
    if stability is not None:
        return float(stability)
    score = result.get(metric)
    if score is None:
        return 1.0
    return float(
        sqrt(max(0.0, float(score) * (1.0 - float(score))) / max(1, n_samples))
    )


def _best_multilabel_probe_for_metric(
    probes: dict[str, dict[str, Any]],
    family: str,
    metric: str,
    n_samples: int,
) -> dict[str, Any]:
    """Return the best multilabel probe evidence for one family and metric."""
    candidates = []
    for probe_name in _FAMILY_PROBES[family]:
        result = probes.get(probe_name, {})
        if metric in result:
            effective_n = int(
                result.get("evaluation_support", {}).get("n_samples", n_samples)
            )
            candidates.append(
                {
                    "probe": probe_name,
                    "score": float(result[metric]),
                    "standard_error": _multilabel_metric_error(
                        result, metric, effective_n
                    ),
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


def _multilabel_family_metric_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build per-family, per-metric multilabel evidence tables."""
    probes = metrics.get("probes", {})
    n_samples = int(metrics.get("audit", {}).get("n_samples", 1))
    evidence: dict[str, Any] = {}
    for family in ("dummy", *_FAMILY_ORDER):
        evidence[family] = {
            metric: _best_multilabel_probe_for_metric(probes, family, metric, n_samples)
            for metric in _MULTILABEL_PRIMARY_METRICS
        }
    return evidence


def _multilabel_combined_error(
    first: dict[str, Any], second: dict[str, Any]
) -> float | None:
    """Return combined uncertainty for two multilabel metric entries."""
    if first["standard_error"] is None or second["standard_error"] is None:
        return None
    return float(sqrt(first["standard_error"] ** 2 + second["standard_error"] ** 2))


def _paired_metric_entry(
    first: dict[str, Any],
    second: dict[str, Any],
    metric: str,
    paired_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return paired evidence for two per-metric family entries."""
    return lookup_paired_comparison(
        paired_payload or {}, first.get("probe"), second.get("probe"), metric
    )


def _multilabel_clearly_better(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metric: str,
    paired_payload: dict[str, Any] | None = None,
) -> bool:
    """Return whether first clearly beats second for higher-is-better metrics."""
    if first["score"] is None or second["score"] is None:
        return False
    paired = _paired_metric_entry(first, second, metric, paired_payload)
    if paired is not None:
        return bool(
            float(paired["point_delta"]) > MIN_FAMILY_IMPROVEMENT
            and float(paired["lower_95"]) > 0.0
        )
    error = _multilabel_combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(first["score"] - second["score"] > SIGNAL_CONFIDENCE_Z * tolerance)


def _multilabel_within_one_standard_error(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metric: str,
    paired_payload: dict[str, Any] | None = None,
) -> bool:
    """Return whether first is within one standard error of second."""
    if first["score"] is None or second["score"] is None:
        return False
    paired = _paired_metric_entry(second, first, metric, paired_payload)
    if paired is not None:
        return bool(
            float(paired["point_delta"])
            <= max(float(paired["paired_standard_error"]), MIN_FAMILY_IMPROVEMENT)
        )
    error = _multilabel_combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(second["score"] - first["score"] <= tolerance)


def _best_multilabel_family_for_metric(
    evidence: dict[str, Any], metric: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the best predictive family for one multilabel metric."""
    candidates = [
        (family, evidence[family][metric])
        for family in _FAMILY_ORDER
        if evidence[family][metric]["score"] is not None
    ]
    if not candidates:
        return None, None
    return max(
        candidates,
        key=lambda item: (
            item[1]["score"],
            -_FAMILY_ORDER.index(item[0]),
        ),
    )


def _multilabel_comparison_counts(
    evidence: dict[str, Any],
    first: str,
    second: str,
    paired_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count clear and within-error comparisons across primary metrics."""
    clear = []
    within = []
    worse = []
    metric_comparisons: dict[str, Any] = {}
    for metric in _MULTILABEL_PRIMARY_METRICS:
        first_item = evidence[first][metric]
        second_item = evidence[second][metric]
        paired = _paired_metric_entry(first_item, second_item, metric, paired_payload)
        if _multilabel_clearly_better(
            first_item,
            second_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            clear.append(metric)
        if _multilabel_within_one_standard_error(
            first_item,
            second_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            within.append(metric)
        if _multilabel_clearly_better(
            second_item,
            first_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            worse.append(metric)
        metric_comparisons[metric] = (
            {
                **paired,
                "decision_method": "paired_oof_bootstrap",
                "fallback_reason": None,
            }
            if paired is not None
            else {
                "decision_method": "marginal_standard_error_fallback",
                "fallback_reason": "paired aligned predictions were unavailable",
                "point_delta": None
                if first_item["score"] is None or second_item["score"] is None
                else float(first_item["score"] - second_item["score"]),
            }
        )
    return {
        "first_family": first,
        "second_family": second,
        "clear_metrics": clear,
        "within_one_standard_error_metrics": within,
        "clearly_worse_metrics": worse,
        "metric_comparisons": metric_comparisons,
    }


def _multilabel_quality_flags(
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
            family_metrics[family][metric]["available"]
            for metric in _MULTILABEL_PRIMARY_METRICS
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
        for family in _FAMILY_ORDER
        for metric in _MULTILABEL_PRIMARY_METRICS
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
    if metrics.get("graph", {}).get("warning") is not None:
        flags.append(
            {
                "name": "boundary_graph_limited",
                "severity": "info",
                "message": metrics["graph"]["warning"],
            }
        )
    if metrics.get("topology", {}).get("skipped_reason") is not None:
        flags.append(
            {
                "name": "topology_diagnostics_unavailable",
                "severity": "info",
                "message": (
                    "Persistent multilabel topology was unavailable or skipped; "
                    "it is only used as supporting evidence when present."
                ),
            }
        )
    existing = {flag["name"] for flag in flags}
    flags.extend(
        flag for flag in _context_quality_flags(metrics) if flag["name"] not in existing
    )
    return flags


def _score_gap_multilabel(family_metrics: dict[str, Any], metric: str) -> float | None:
    """Return best predictive score improvement over dummy for one metric."""
    _, best = _best_multilabel_family_for_metric(family_metrics, metric)
    dummy = family_metrics["dummy"][metric]["score"]
    if best is None or best["score"] is None or dummy is None:
        return None
    return float(max(0.0, best["score"] - dummy))


def _linearity_score_multilabel(
    family_metrics: dict[str, Any], metric: str
) -> float | None:
    """Return linear score as a fraction of the best predictive score."""
    _, best = _best_multilabel_family_for_metric(family_metrics, metric)
    linear = family_metrics["linear"][metric]["score"]
    if best is None or best["score"] is None or linear is None:
        return None
    return float(np.clip(linear / max(best["score"], 1e-9), 0.0, 1.0))


def compute_multilabel_scores(
    metrics: dict[str, Any],
    *,
    skipped_count: int,
    warning_count: int,
) -> dict[str, float | None]:
    """Compute transparent multilabel compatibility scores and evidence."""
    family_metrics = _multilabel_family_metric_evidence(metrics)
    paired_payload = metrics.get("paired_probe_comparisons", {})
    signal_metrics = []
    signal_comparisons: dict[str, Any] = {}
    best_by_metric: dict[str, Any] = {}
    for metric in _MULTILABEL_PRIMARY_METRICS:
        best_family, best_item = _best_multilabel_family_for_metric(
            family_metrics, metric
        )
        best_by_metric[metric] = {
            "family": best_family,
            "probe": best_item["probe"] if best_item else None,
            "score": best_item["score"] if best_item else None,
        }
        dummy_item = family_metrics["dummy"][metric]
        paired_signal = (
            _paired_metric_entry(best_item, dummy_item, metric, paired_payload)
            if best_item is not None
            else None
        )
        signal_comparisons[metric] = (
            {
                **paired_signal,
                "decision_method": "paired_oof_bootstrap",
                "fallback_reason": None,
            }
            if paired_signal is not None
            else {
                "decision_method": "marginal_standard_error_fallback",
                "fallback_reason": "paired aligned predictions were unavailable",
            }
        )
        if best_item is not None and _multilabel_clearly_better(
            best_item,
            dummy_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            signal_metrics.append(metric)

    comparisons = {
        "linear_vs_best": {},
        "smooth_vs_linear": _multilabel_comparison_counts(
            family_metrics, "smooth_nonlinear", "linear", paired_payload
        ),
        "local_kernel_vs_smooth": _multilabel_comparison_counts(
            family_metrics, "local_kernel", "smooth_nonlinear", paired_payload
        ),
    }
    for metric in _MULTILABEL_PRIMARY_METRICS:
        best_family = best_by_metric[metric]["family"]
        if best_family is None:
            continue
        comparisons["linear_vs_best"][metric] = {
            "best_family": best_family,
            "linear_within_one_standard_error": _multilabel_within_one_standard_error(
                family_metrics["linear"][metric],
                family_metrics[best_family][metric],
                metric=metric,
                paired_payload=paired_payload,
            ),
            "linear_clearly_worse": _multilabel_clearly_better(
                family_metrics[best_family][metric],
                family_metrics["linear"][metric],
                metric=metric,
                paired_payload=paired_payload,
            ),
        }

    flags = _multilabel_quality_flags(
        metrics, family_metrics, skipped_count, warning_count
    )
    topology = metrics.get("topology", {})
    topology_strength = _topology_score(metrics)
    evidence = {
        "selection_rule": (
            "Compare probe families across micro F1, macro F1, and sample "
            "Jaccard without collapsing them into a weighted aggregate."
        ),
        "primary_metrics": list(_MULTILABEL_PRIMARY_METRICS),
        "signal_confidence_z": SIGNAL_CONFIDENCE_Z,
        "families": family_metrics,
        "best_by_metric": best_by_metric,
        "signal_metrics_beating_dummy": signal_metrics,
        "signal_comparisons": signal_comparisons,
        "best_clearly_beats_dummy_on_two_primary_metrics": len(signal_metrics) >= 2,
        "family_comparisons": comparisons,
        "topology_available": bool(
            topology_strength is not None and topology.get("skipped_reason") is None
        ),
        "topology_strength": topology_strength,
        "quality_flags": flags,
        "quality_score": float(
            np.mean([flag.get("severity") != "blocking" for flag in flags])
        )
        if flags
        else 1.0,
    }
    metrics["multilabel_recommendation_evidence"] = evidence
    return {
        "signal_micro_f1": _score_gap_multilabel(family_metrics, "micro_f1"),
        "signal_macro_f1": _score_gap_multilabel(family_metrics, "macro_f1"),
        "signal_sample_jaccard": _score_gap_multilabel(
            family_metrics, "sample_jaccard"
        ),
        "linearity_micro_f1": _linearity_score_multilabel(family_metrics, "micro_f1"),
        "linearity_macro_f1": _linearity_score_multilabel(family_metrics, "macro_f1"),
        "linearity_sample_jaccard": _linearity_score_multilabel(
            family_metrics, "sample_jaccard"
        ),
        "neighborhood_coherence_score": metrics.get("neighborhood", {}).get(
            "mean_neighbor_jaccard"
        ),
        "fragmentation_score": _numeric(
            metrics.get("graph", {}).get("graph_fragmentation_score")
        ),
        "topology_score": topology_strength,
        "reliability_score": _numeric(evidence.get("quality_score")),
    }


def make_multilabel_recommendation(
    scores: dict[str, float | None],
    metrics: dict[str, Any],
) -> tuple[str, str, list[str], dict[str, str]]:
    """Generate a conservative multilabel recommendation and decision path."""
    evidence = metrics["multilabel_recommendation_evidence"]
    flags = evidence["quality_flags"]
    decision_path: list[str] = []
    if any(flag.get("severity") == "blocking" for flag in flags):
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
        graph = metrics.get("graph", {})
        multilabel_fragmentation_supported = bool(
            graph.get("graph_fragmentation_score") is not None
            and float(graph["graph_fragmentation_score"]) >= 0.5
            and int(metrics.get("boundary", {}).get("boundary_sample_size", 0)) >= 10
        )
        topology_available = bool(evidence.get("topology_available"))
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
            if multilabel_fragmentation_supported:
                recommendation = HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
                decision_path.append(
                    "Local or kernel-style probes clearly improved over smooth "
                    "nonlinear probes, and multilabel boundary candidates formed a "
                    "fragmented graph."
                )
            else:
                recommendation = KERNEL_OR_LOCAL_RECOMMENDED
                decision_path.append(
                    "Local or kernel-style probes clearly improved over smooth "
                    "nonlinear probes on at least two primary multilabel metrics."
                )
            if topology_available:
                decision_path.append(
                    "Persistent multilabel topology was available as supporting "
                    "structural evidence."
                )
        else:
            recommendation = INCONCLUSIVE
            decision_path.append(
                "Primary multilabel metrics disagreed across probe families, "
                "so no single model-family recommendation was forced."
            )

    if _mlp_override_active(metrics):
        mlp_payload = _mlp_override_payload(metrics)
        recommendation = FEEDFORWARD_MLP_RECOMMENDED
        decision_path.insert(
            0,
            "The optional MLP probe clearly improved over the aligned simpler "
            "multilabel families and satisfied the configured override criteria.",
        )
        decision_path.append(
            "Conditional MLP probes were only run because simpler multilabel "
            "probes did not meet the configured absolute-skill threshold."
        )
        if mlp_payload.get("override_reason") is not None:
            decision_path.append(str(mlp_payload["override_reason"]))
        architecture_note = _mlp_architecture_note(metrics)
        if architecture_note is not None:
            decision_path.append(architecture_note)

    if flags:
        decision_path.append(
            "Evidence quality flags: "
            + ", ".join(str(flag["name"]) for flag in flags)
            + "."
        )
    decision_path.append(
        "This run used the multilabel diagnostic path; probe families were "
        "compared across micro F1, macro F1, and sample Jaccard."
    )
    decision_path.append(_comparison_method_note(metrics))
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
        "fragmentation": (
            "Higher values mean multilabel boundary candidates form a more "
            "fragmented local graph; topology is supporting structural evidence "
            "when available."
        ),
        "reliability": (
            "Higher values mean essential multilabel evidence was available; "
            "optional topology is non-blocking."
        ),
    }
    if metrics.get("mlp_recommendation_evidence") is not None:
        interpretations["mlp_recommendation_evidence"] = (
            "Optional multilabel MLP probes run only after simpler models miss a "
            "configured absolute-skill threshold, and they override simpler probe "
            "families only when the best tested architecture clearly beats every "
            "aligned simpler probe on at least two primary metrics."
        )
    decision_path.append(
        "Signal metrics beating dummy: "
        + ", ".join(evidence["signal_metrics_beating_dummy"] or ["none"])
        + "."
    )
    return recommendation, confidence, decision_path, interpretations


def _regression_metric_error(
    result: dict[str, Any], metric: str, n_samples: int
) -> float:
    """Return a conservative regression metric uncertainty estimate."""
    stability = result.get(f"stability_{metric}_std")
    if stability is not None:
        return float(stability)
    if result.get(metric) is None:
        return 1.0
    return float(max(0.05, 1.0 / sqrt(max(1, n_samples))))


def _best_regression_probe_for_metric(
    probes: dict[str, dict[str, Any]],
    family: str,
    metric: str,
    n_samples: int,
) -> dict[str, Any]:
    """Return the best regression probe evidence for one family and metric."""
    candidates = []
    for probe_name in _FAMILY_PROBES[family]:
        result = probes.get(probe_name, {})
        if metric in result:
            effective_n = int(
                result.get("evaluation_support", {}).get("n_samples", n_samples)
            )
            candidates.append(
                {
                    "probe": probe_name,
                    "score": float(result[metric]),
                    "standard_error": _regression_metric_error(
                        result, metric, effective_n
                    ),
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


def _regression_family_metric_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build per-family, per-metric regression evidence tables."""
    probes = metrics.get("probes", {})
    n_samples = int(metrics.get("audit", {}).get("n_samples", 1))
    evidence: dict[str, Any] = {}
    for family in ("dummy", *_FAMILY_ORDER):
        evidence[family] = {
            metric: _best_regression_probe_for_metric(probes, family, metric, n_samples)
            for metric in _REGRESSION_PRIMARY_METRICS
        }
    return evidence


def _regression_combined_error(
    first: dict[str, Any], second: dict[str, Any]
) -> float | None:
    """Return combined uncertainty for two regression metric entries."""
    if first["standard_error"] is None or second["standard_error"] is None:
        return None
    return float(sqrt(first["standard_error"] ** 2 + second["standard_error"] ** 2))


def _regression_clearly_better(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metric: str,
    paired_payload: dict[str, Any] | None = None,
) -> bool:
    """Return whether first clearly beats second for higher-is-better metrics."""
    if first["score"] is None or second["score"] is None:
        return False
    paired = _paired_metric_entry(first, second, metric, paired_payload)
    if paired is not None:
        return bool(
            float(paired["point_delta"]) > MIN_FAMILY_IMPROVEMENT
            and float(paired["lower_95"]) > 0.0
        )
    error = _regression_combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(first["score"] - second["score"] > SIGNAL_CONFIDENCE_Z * tolerance)


def _regression_within_one_standard_error(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    metric: str,
    paired_payload: dict[str, Any] | None = None,
) -> bool:
    """Return whether first is within one standard error of second."""
    if first["score"] is None or second["score"] is None:
        return False
    paired = _paired_metric_entry(second, first, metric, paired_payload)
    if paired is not None:
        return bool(
            float(paired["point_delta"])
            <= max(float(paired["paired_standard_error"]), MIN_FAMILY_IMPROVEMENT)
        )
    error = _regression_combined_error(first, second)
    tolerance = error if error is not None else 0.0
    return bool(second["score"] - first["score"] <= tolerance)


def _best_regression_family_for_metric(
    evidence: dict[str, Any], metric: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the best predictive regression family for one metric."""
    candidates = [
        (family, evidence[family][metric])
        for family in _FAMILY_ORDER
        if evidence[family][metric]["score"] is not None
    ]
    if not candidates:
        return None, None
    return max(
        candidates,
        key=lambda item: (item[1]["score"], -_FAMILY_ORDER.index(item[0])),
    )


def _regression_comparison_counts(
    evidence: dict[str, Any],
    first: str,
    second: str,
    paired_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count clear and within-error comparisons across regression metrics."""
    clear = []
    within = []
    worse = []
    metric_comparisons: dict[str, Any] = {}
    for metric in _REGRESSION_PRIMARY_METRICS:
        first_item = evidence[first][metric]
        second_item = evidence[second][metric]
        paired = _paired_metric_entry(first_item, second_item, metric, paired_payload)
        if _regression_clearly_better(
            first_item,
            second_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            clear.append(metric)
        if _regression_within_one_standard_error(
            first_item,
            second_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            within.append(metric)
        if _regression_clearly_better(
            second_item,
            first_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            worse.append(metric)
        metric_comparisons[metric] = (
            {
                **paired,
                "decision_method": "paired_oof_bootstrap",
                "fallback_reason": None,
            }
            if paired is not None
            else {
                "decision_method": "marginal_standard_error_fallback",
                "fallback_reason": "paired aligned predictions were unavailable",
                "point_delta": None
                if first_item["score"] is None or second_item["score"] is None
                else float(first_item["score"] - second_item["score"]),
            }
        )
    return {
        "first_family": first,
        "second_family": second,
        "clear_metrics": clear,
        "within_one_standard_error_metrics": within,
        "clearly_worse_metrics": worse,
        "metric_comparisons": metric_comparisons,
    }


def _regression_quality_flags(
    metrics: dict[str, Any],
    family_metrics: dict[str, Any],
    skipped_count: int,
    warning_count: int,
) -> list[dict[str, Any]]:
    """Return regression evidence quality flags."""
    flags: list[dict[str, Any]] = []
    if metrics.get("audit", {}).get("usable_target_count", 0) <= 0:
        flags.append(
            {
                "name": "no_usable_targets",
                "severity": "blocking",
                "message": "No non-constant regression targets were available.",
            }
        )
    for family in ("dummy", "linear"):
        if not any(
            family_metrics[family][metric]["available"]
            for metric in _REGRESSION_PRIMARY_METRICS
        ):
            flags.append(
                {
                    "name": f"missing_{family}_probe",
                    "severity": "blocking",
                    "message": f"The {family} probe did not produce primary metrics.",
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
    if metrics.get("neighborhood", {}).get("skipped_reason") is not None:
        flags.append(
            {
                "name": "regression_neighborhood_unavailable",
                "severity": "info",
                "message": "Target-neighborhood smoothness diagnostics were skipped.",
            }
        )
    existing = {flag["name"] for flag in flags}
    flags.extend(
        flag for flag in _context_quality_flags(metrics) if flag["name"] not in existing
    )
    return flags


def compute_regression_scores(
    metrics: dict[str, Any],
    *,
    skipped_count: int,
    warning_count: int,
) -> dict[str, float | None]:
    """Compute transparent regression compatibility scores and evidence."""
    family_metrics = _regression_family_metric_evidence(metrics)
    paired_payload = metrics.get("paired_probe_comparisons", {})
    signal_metrics = []
    signal_comparisons: dict[str, Any] = {}
    best_by_metric: dict[str, Any] = {}
    for metric in _REGRESSION_PRIMARY_METRICS:
        best_family, best_item = _best_regression_family_for_metric(
            family_metrics, metric
        )
        best_by_metric[metric] = {
            "family": best_family,
            "probe": best_item["probe"] if best_item else None,
            "score": best_item["score"] if best_item else None,
        }
        dummy_item = family_metrics["dummy"][metric]
        paired_signal = (
            _paired_metric_entry(best_item, dummy_item, metric, paired_payload)
            if best_item is not None
            else None
        )
        signal_comparisons[metric] = (
            {
                **paired_signal,
                "decision_method": "paired_oof_bootstrap",
                "fallback_reason": None,
            }
            if paired_signal is not None
            else {
                "decision_method": "marginal_standard_error_fallback",
                "fallback_reason": "paired aligned predictions were unavailable",
            }
        )
        if best_item is not None and _regression_clearly_better(
            best_item,
            dummy_item,
            metric=metric,
            paired_payload=paired_payload,
        ):
            signal_metrics.append(metric)

    comparisons = {
        "linear_vs_best": {},
        "smooth_vs_linear": _regression_comparison_counts(
            family_metrics, "smooth_nonlinear", "linear", paired_payload
        ),
        "local_kernel_vs_smooth": _regression_comparison_counts(
            family_metrics, "local_kernel", "smooth_nonlinear", paired_payload
        ),
    }
    for metric in _REGRESSION_PRIMARY_METRICS:
        best_family = best_by_metric[metric]["family"]
        if best_family is None:
            continue
        comparisons["linear_vs_best"][metric] = {
            "best_family": best_family,
            "linear_within_one_standard_error": _regression_within_one_standard_error(
                family_metrics["linear"][metric],
                family_metrics[best_family][metric],
                metric=metric,
                paired_payload=paired_payload,
            ),
            "linear_clearly_worse": _regression_clearly_better(
                family_metrics[best_family][metric],
                family_metrics["linear"][metric],
                metric=metric,
                paired_payload=paired_payload,
            ),
        }

    flags = _regression_quality_flags(
        metrics, family_metrics, skipped_count, warning_count
    )
    topology = metrics.get("topology", {})
    topology_strength = _topology_score(metrics)
    evidence = {
        "selection_rule": (
            "Compare regression probe families across variance-weighted and "
            "uniform-average R2, using conservative escalation from simpler "
            "to more complex families."
        ),
        "primary_metrics": list(_REGRESSION_PRIMARY_METRICS),
        "signal_confidence_z": SIGNAL_CONFIDENCE_Z,
        "families": family_metrics,
        "best_by_metric": best_by_metric,
        "signal_metrics_beating_dummy": signal_metrics,
        "signal_comparisons": signal_comparisons,
        "best_clearly_beats_dummy_on_primary_metrics": len(signal_metrics) >= 1,
        "family_comparisons": comparisons,
        "topology_available": bool(
            topology_strength is not None and topology.get("skipped_reason") is None
        ),
        "topology_strength": topology_strength,
        "topology_role": "supporting_only",
        "quality_flags": flags,
        "quality_score": float(
            np.mean([flag.get("severity") != "blocking" for flag in flags])
        )
        if flags
        else 1.0,
    }
    metrics["regression_recommendation_evidence"] = evidence
    weighted_best = best_by_metric["r2_variance_weighted"]["score"]
    weighted_dummy = family_metrics["dummy"]["r2_variance_weighted"]["score"]
    linear_weighted = family_metrics["linear"]["r2_variance_weighted"]["score"]
    return {
        "signal_r2_variance_weighted": float(max(0.0, weighted_best - weighted_dummy))
        if weighted_best is not None and weighted_dummy is not None
        else None,
        "signal_r2_uniform_average": None
        if best_by_metric["r2_uniform_average"]["score"] is None
        or family_metrics["dummy"]["r2_uniform_average"]["score"] is None
        else float(
            max(
                0.0,
                best_by_metric["r2_uniform_average"]["score"]
                - family_metrics["dummy"]["r2_uniform_average"]["score"],
            )
        ),
        "linearity_score": float(
            np.clip(linear_weighted / max(weighted_best, 1e-9), 0.0, 1.0)
        )
        if (
            linear_weighted is not None
            and weighted_best is not None
            and weighted_best > 0
        )
        else None,
        "target_smoothness_score": _numeric(
            metrics.get("neighborhood", {}).get("target_smoothness_score")
        ),
        "high_discontinuity_fraction": _numeric(
            metrics.get("neighborhood", {}).get("high_discontinuity_fraction")
        ),
        "topology_score": topology_strength,
        "reliability_score": _numeric(evidence.get("quality_score")),
    }


def make_regression_recommendation(
    scores: dict[str, float | None],
    metrics: dict[str, Any],
) -> tuple[str, str, list[str], dict[str, str]]:
    """Generate a conservative regression recommendation and decision path."""
    evidence = metrics["regression_recommendation_evidence"]
    flags = evidence["quality_flags"]
    decision_path: list[str] = []
    if any(flag.get("severity") == "blocking" for flag in flags):
        recommendation = INSUFFICIENT_DATA_OR_UNRELIABLE_REGRESSION_GEOMETRY
        decision_path.append(
            "Essential regression probe evidence was unavailable, so the result "
            "is limited to data sufficiency and diagnostic reliability."
        )
    elif not evidence["best_clearly_beats_dummy_on_primary_metrics"]:
        smoothness = metrics.get("neighborhood", {}).get("target_smoothness_score")
        recommendation = (
            FEATURE_OR_TARGET_BOTTLENECK_LIKELY
            if smoothness is not None and float(smoothness) < 0.35
            else INCONCLUSIVE_REGRESSION_DIAGNOSTIC
        )
        decision_path.append(
            "The best predictive family did not clearly beat the target-mean "
            "dummy baseline on a primary regression metric."
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
        high_discontinuity = float(
            metrics.get("neighborhood", {}).get("high_discontinuity_fraction") or 0.0
        )
        if linear_within >= 1 and linear_worse == 0:
            recommendation = LINEAR_RESPONSE_LIKELY_SUFFICIENT
            decision_path.append(
                "The linear response family was within uncertainty of the best "
                "observed family."
            )
        elif smooth_clear >= 1 and local_clear < 1:
            recommendation = SMOOTH_NONLINEAR_RESPONSE_RECOMMENDED
            decision_path.append(
                "Smooth nonlinear probes clearly improved over the linear "
                "response family."
            )
        elif local_clear >= 1:
            if high_discontinuity >= 0.35:
                recommendation = HIGH_CAPACITY_OR_PARTITIONING_REGRESSION_RECOMMENDED
                decision_path.append(
                    "Local or kernel-style probes improved over smooth probes, "
                    "and target-neighborhood diagnostics showed discontinuity."
                )
            else:
                recommendation = KERNEL_OR_LOCAL_REGRESSION_RECOMMENDED
                decision_path.append(
                    "Local or kernel-style probes clearly improved over smooth "
                    "nonlinear regression probes."
                )
        else:
            recommendation = INCONCLUSIVE_REGRESSION_DIAGNOSTIC
            decision_path.append(
                "Primary regression metrics disagreed across probe families, "
                "so no model-family recommendation was forced."
            )

    if _mlp_override_active(metrics):
        mlp_payload = _mlp_override_payload(metrics)
        recommendation = FEEDFORWARD_MLP_REGRESSION_RECOMMENDED
        decision_path.insert(
            0,
            "The optional MLP probe clearly improved over the aligned simpler "
            "regression families and satisfied the configured override criteria.",
        )
        decision_path.append(
            "Conditional MLP probes were only run because simpler regression "
            "probes did not meet the configured absolute-skill threshold."
        )
        if mlp_payload.get("override_reason") is not None:
            decision_path.append(str(mlp_payload["override_reason"]))
        architecture_note = _mlp_architecture_note(metrics)
        if architecture_note is not None:
            decision_path.append(architecture_note)
    if flags:
        decision_path.append(
            "Evidence quality flags: "
            + ", ".join(str(flag["name"]) for flag in flags)
            + "."
        )
    if evidence["topology_available"]:
        decision_path.append(
            "Hard-subset topology strength was "
            f"{float(evidence['topology_strength']):.3f}; this is supporting "
            "evidence only and did not alter the recommendation or confidence."
        )
    else:
        decision_path.append(
            "Hard-subset topology was unavailable or skipped; optional topology "
            "did not alter the recommendation or confidence."
        )
    decision_path.append(
        "This run used the explicit regression diagnostic path; probe families "
        "were compared across variance-weighted and uniform-average R2."
    )
    decision_path.append(_comparison_method_note(metrics))
    confidence = (
        "low"
        if recommendation
        in {
            INCONCLUSIVE_REGRESSION_DIAGNOSTIC,
            INSUFFICIENT_DATA_OR_UNRELIABLE_REGRESSION_GEOMETRY,
        }
        else "medium"
    )
    if (
        not any(flag.get("severity") == "caution" for flag in flags)
        and confidence == "medium"
    ):
        confidence = "high"
    interpretations = {
        "regression_recommendation_evidence": (
            "Regression probe families are compared across separate R2 metrics; "
            "normalized RMSE and target-neighborhood smoothness are supporting "
            "diagnostics."
        ),
        "signal": "Higher values mean targets appear predictable beyond their mean.",
        "linearity": (
            "Higher values mean linear response probes are close to the best family."
        ),
        "target_smoothness": (
            "Higher values mean nearby samples in feature space have nearby "
            "continuous target values."
        ),
        "topology": (
            "Higher values indicate fragmentation or loop-like geometry among "
            "high-residual or high-discontinuity points. This evidence is "
            "descriptive and does not select the recommendation."
        ),
        "reliability": (
            "Higher values mean essential regression evidence was available."
        ),
    }
    if metrics.get("mlp_recommendation_evidence") is not None:
        interpretations["mlp_recommendation_evidence"] = (
            "Optional regression MLP probes run only after simpler models miss a "
            "configured absolute-skill threshold, and they override simpler probe "
            "families only when the best tested architecture clearly beats every "
            "aligned simpler regressor on both primary R2 metrics."
        )
    decision_path.append(
        "Signal metrics beating dummy: "
        + ", ".join(evidence["signal_metrics_beating_dummy"] or ["none"])
        + "."
    )
    return recommendation, confidence, decision_path, interpretations
