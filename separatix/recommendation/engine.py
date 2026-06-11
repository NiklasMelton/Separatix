"""Recommendation engine for separatix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
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

MIN_NORMALIZED_SCORE = 0.0
MAX_NORMALIZED_SCORE = 1.0
NORMALIZATION_EPSILON = 1e-9
ONE_STANDARD_ERROR = 1.0
# Standard normal two-sided 95% critical value used to decide whether any
# predictive probe has evidence above the class-prior dummy baseline.
SIGNAL_CONFIDENCE_Z = 1.96

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


def _one_standard_error_family(
    families: dict[str, _FamilyEvidence],
    best_family: _FamilyEvidence,
) -> str | None:
    if best_family.score is None:
        return None
    for family_name in _FAMILY_ORDER:
        family = families[family_name]
        if family.score is None:
            continue
        combined_error = _combined_standard_error(family, best_family)
        tolerance = ONE_STANDARD_ERROR * (
            combined_error if combined_error is not None else MIN_NORMALIZED_SCORE
        )
        if best_family.score - family.score <= tolerance:
            return family_name
    return None


def _family_comparison(
    first: _FamilyEvidence,
    second: _FamilyEvidence,
) -> dict[str, float | bool | str | None]:
    if first.score is None or second.score is None:
        return {
            "first_family": first.family,
            "second_family": second.family,
            "score_gap": None,
            "combined_standard_error": None,
            "first_clearly_better": False,
            "second_clearly_better": False,
        }
    combined_error = _combined_standard_error(first, second)
    score_gap = first.score - second.score
    tolerance = combined_error or MIN_NORMALIZED_SCORE
    return {
        "first_family": first.family,
        "second_family": second.family,
        "score_gap": float(score_gap),
        "combined_standard_error": combined_error,
        "first_clearly_better": bool(score_gap > tolerance),
        "second_clearly_better": bool(-score_gap > tolerance),
    }


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
                "severity": "caution",
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
    if (
        best_family is not None
        and best_family.score is not None
        and dummy_family.score is not None
        and signal_error is not None
        and (SIGNAL_CONFIDENCE_Z * signal_error)
        >= abs(best_family.score - dummy_family.score)
    ):
        flags.append(
            {
                "name": "weak_signal_evidence",
                "severity": "caution",
                "message": (
                    "The best probe's advantage over the dummy baseline does not "
                    "clear a 95% normal-approximation signal check."
                ),
            }
        )
    return flags


def _quality_score(flags: list[dict[str, Any]]) -> float:
    checks = [
        "missing_dummy_probe",
        "missing_linear_probe",
        "missing_predictive_probe",
        "resubstitution_evaluation",
        "weak_signal_evidence",
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
    best_family = _best_predictive_family(families)
    dummy_family = families["dummy"]
    selected_family = (
        _one_standard_error_family(families, best_family)
        if best_family is not None
        else None
    )
    signal_error = _combined_standard_error(best_family, dummy_family)
    best_clearly_beats_dummy = (
        bool(
            best_family is not None
            and best_family.score is not None
            and dummy_family.score is not None
            and signal_error is not None
            and best_family.score - dummy_family.score
            > SIGNAL_CONFIDENCE_Z * signal_error
        )
        if dummy_family.available
        else False
    )

    smooth_vs_local = _family_comparison(
        families["local_kernel"], families["smooth_nonlinear"]
    )
    quality_flags = _quality_flags(
        metrics,
        probes,
        families,
        skipped_count=skipped_count,
        warning_count=warning_count,
        best_family=best_family,
        dummy_family=dummy_family,
    )
    best_score = best_family.score if best_family is not None else None
    dummy_score = dummy_family.score
    signal_margin = (
        best_score - dummy_score
        if best_score is not None and dummy_score is not None
        else None
    )

    return {
        "selection_rule": (
            "Choose the simplest predictive probe family within one standard "
            "error of the best observed family using pairwise combined "
            "uncertainty."
        ),
        "standard_error_multiplier": ONE_STANDARD_ERROR,
        "signal_confidence_z": SIGNAL_CONFIDENCE_Z,
        "probe_table": _probe_table(probes),
        "families": _family_table(families),
        "best_family": best_family.family if best_family is not None else None,
        "best_probe": best_family.best_probe if best_family is not None else None,
        "selected_family": selected_family,
        "signal_margin_over_dummy": (
            float(signal_margin) if signal_margin is not None else None
        ),
        "signal_combined_standard_error": signal_error,
        "signal_z_score": (
            float(signal_margin / signal_error)
            if signal_margin is not None
            and signal_error is not None
            and signal_error > 0
            else None
        ),
        "best_clearly_beats_dummy": best_clearly_beats_dummy,
        "family_comparisons": {"local_kernel_vs_smooth_nonlinear": smooth_vs_local},
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
        best_family = evidence["best_family"]
        selected_family = evidence["selected_family"]
        decision_path.append(
            "Probe families were compared with a one-standard-error rule: "
            f"best={best_family}, selected={selected_family}."
        )
        recommendation = _recommend_selected_family(evidence)
        if selected_family == "linear":
            decision_path.append(
                "The linear probe was statistically indistinguishable from the "
                "best observed probe family."
            )
        elif selected_family == "smooth_nonlinear":
            decision_path.append(
                "Nonlinear probes improved over linear, and the smooth nonlinear "
                "family was the simplest family within one standard error of best."
            )
        elif selected_family == "local_kernel":
            decision_path.append(
                "Local or kernel-style probes were the simplest family within one "
                "standard error of best."
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

    quality_flags = evidence.get("quality_flags", [])
    if quality_flags:
        flag_names = ", ".join(flag["name"] for flag in quality_flags)
        decision_path.append(f"Evidence quality flags: {flag_names}.")

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
        "estimated standard error; selection uses the simplest family within "
        "one standard error of the best family."
    )

    decision_path.append(
        "Family evidence: "
        f"linear={_format_family_score(evidence, 'linear')}, "
        f"smooth={_format_family_score(evidence, 'smooth_nonlinear')}, "
        f"local/kernel={_format_family_score(evidence, 'local_kernel')}."
    )
    return recommendation, confidence, decision_path, interpretations
