"""Plain-text rendering for recommendations."""

from separatix.constants import RECOMMENDATION_LABELS
from separatix.report import DiagnosticReport

_SUGGESTIONS = {
    "linear_likely_sufficient": [
        "regularized logistic regression",
        "linear SVM or ridge classifier",
        "linear models with light feature engineering",
    ],
    "smooth_nonlinear_recommended": [
        "interaction or quadratic features with a linear classifier",
        "GAM-like or spline-based models",
        "small MLP",
    ],
    "kernel_or_local_recommended": [
        "RBF or approximate kernel classifier",
        "k-nearest neighbors or other local models",
        "moderate-capacity neural network",
    ],
    "high_capacity_or_partitioning_recommended": [
        "tree ensembles",
        "boosted trees",
        "higher-capacity neural network",
    ],
    "feedforward_mlp_recommended": [
        "a compact one-hidden-layer MLP as the first neural baseline",
        "a matched two-hidden-layer MLP if the compact network underfits",
        "retain a strong linear or kernel baseline for calibration",
    ],
    "feature_or_label_bottleneck_likely": [
        "feature cleanup or enrichment",
        "label audit",
        "embedding revision before classifier tuning",
    ],
    "insufficient_data_or_unreliable_geometry": [
        "collect more labeled examples",
        "reduce class imbalance",
        "revisit feature scaling and data quality",
    ],
    "inconclusive": [
        "compare a linear baseline and one nonlinear baseline",
        "inspect class overlap manually",
        "gather more data if feasible",
    ],
    "linear_response_likely_sufficient": [
        "ridge or elastic-net regression",
        "linear multioutput regression",
        "linear models with light feature engineering",
    ],
    "smooth_nonlinear_response_recommended": [
        "interaction or quadratic features with ridge regression",
        "GAM-like or spline-based regression",
        "small MLP regressor",
    ],
    "kernel_or_local_regression_recommended": [
        "RBF or approximate-kernel ridge regression",
        "k-nearest-neighbor regression",
        "moderate-capacity neural regressor",
    ],
    "higher_capacity_or_partitioning_regression_recommended": [
        "tree ensembles",
        "boosted-tree regression",
        "partitioned or mixture-of-experts regressors",
    ],
    "feedforward_mlp_regression_recommended": [
        "a compact one-hidden-layer MLP regressor as the first neural baseline",
        "a matched two-hidden-layer MLP regressor if the compact network underfits",
        "retain a strong ridge or kernel baseline for calibration",
    ],
    "feature_or_target_bottleneck_likely": [
        "target audit and noise review",
        "feature cleanup or enrichment",
        "embedding revision before regressor tuning",
    ],
    "insufficient_data_or_unreliable_regression_geometry": [
        "collect more target observations",
        "review target scaling and measurement reliability",
        "use simpler baselines until splits are reliable",
    ],
    "inconclusive_regression_diagnostic": [
        "compare a ridge baseline and one nonlinear baseline",
        "inspect per-target residuals",
        "gather more data if feasible",
    ],
}

_CORE_FAMILY_DISPLAY_NAMES = {
    "linear": "linear",
    "smooth_nonlinear": "smooth nonlinear",
    "local_kernel": "local/kernel",
}


def _render_plausible_family_set(report: DiagnosticReport) -> str:
    """Render the target-specific plausible core-family evidence when useful."""
    target_type = report.class_summary.get("target_type")
    if target_type == "multilabel":
        evidence_key = "multilabel_recommendation_evidence"
    elif target_type == "regression":
        evidence_key = "regression_recommendation_evidence"
    else:
        evidence_key = "recommendation_evidence"
    payload = report.metrics.get(evidence_key, {}).get("plausible_family_set", {})
    status = payload.get("status")
    if status == "not_applicable" or not status:
        return ""
    if status == "unavailable":
        reason = payload.get("reason") or "required probe comparisons were unavailable"
        return f"Plausible core-family set unavailable: {reason}\n\n"

    minimum = payload.get("minimum_recommended_family")
    plausible = [
        _CORE_FAMILY_DISPLAY_NAMES.get(family, family)
        for family in payload.get("plausible_families", [])
    ]
    plausible_text = ", ".join(plausible) if plausible else "none"
    if minimum is None:
        return (
            "Minimum recommended core family: unresolved across primary metrics.\n"
            "Statistically plausible core families (nondominated): "
            f"{plausible_text}.\n\n"
        )
    minimum_name = str(minimum)
    minimum_text = _CORE_FAMILY_DISPLAY_NAMES.get(minimum_name, minimum_name)
    return (
        f"Minimum recommended core family: {minimum_text}.\n"
        f"Statistically plausible core families: {plausible_text}.\n\n"
    )


def _render_multilabel_recommendation(report: DiagnosticReport) -> str:
    """Render a concise plain-text multilabel recommendation."""
    headline = RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    suggestions = _SUGGESTIONS.get(report.recommendation, _SUGGESTIONS["inconclusive"])
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    caveats = report.warnings[:2]
    if not caveats and report.skipped_diagnostics:
        caveats = [entry["reason"] for entry in report.skipped_diagnostics[:2]]
    caveat_text = "; ".join(caveats) if caveats else "no major caveats recorded"
    grouping_note = (
        "Grouped evaluation was used; cross-group evidence is primary.\n\n"
        if report.grouping.get("provided")
        else ""
    )
    family_set_text = _render_plausible_family_set(report)
    return (
        f"Recommendation: {headline}\n\n"
        f"{' '.join(report.decision_path[:3])}\n\n"
        f"{family_set_text}"
        f"{grouping_note}"
        "This is a multilabel diagnostic. Evidence is compared across micro F1, "
        "macro F1, and sample Jaccard rather than collapsed into one weighted "
        "score.\n\n"
        f"Suggested next models:\n{suggestion_text}\n\n"
        f"Confidence: {report.confidence}.\n"
        f"Main caveats: {caveat_text}."
    )


def render_recommendation(report: DiagnosticReport) -> str:
    """Render a concise plain-text recommendation from a diagnostic report."""
    if report.class_summary.get("target_type") == "multilabel":
        return _render_multilabel_recommendation(report)
    if report.class_summary.get("target_type") == "regression":
        return _render_regression_recommendation(report)
    headline = RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    suggestions = _SUGGESTIONS.get(report.recommendation, _SUGGESTIONS["inconclusive"])
    caveats = report.warnings[:2]
    if not caveats and report.skipped_diagnostics:
        caveats = [entry["reason"] for entry in report.skipped_diagnostics[:2]]
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    caveat_text = "; ".join(caveats) if caveats else "no major caveats recorded"
    grouping_note = (
        "Grouped evaluation was used; cross-group evidence is primary.\n\n"
        if report.grouping.get("provided")
        else ""
    )
    family_set_text = _render_plausible_family_set(report)
    return (
        f"Recommendation: {headline}\n\n"
        f"{' '.join(report.decision_path[:3])}\n\n"
        f"{family_set_text}"
        f"{grouping_note}"
        f"Suggested next models:\n{suggestion_text}\n\n"
        f"Confidence: {report.confidence}.\n"
        f"Main caveats: {caveat_text}."
    )


def _render_regression_recommendation(report: DiagnosticReport) -> str:
    """Render a concise plain-text regression recommendation."""
    headline = RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    suggestions = _SUGGESTIONS.get(report.recommendation, _SUGGESTIONS["inconclusive"])
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    caveats = report.warnings[:2]
    if not caveats and report.skipped_diagnostics:
        caveats = [entry["reason"] for entry in report.skipped_diagnostics[:2]]
    caveat_text = "; ".join(caveats) if caveats else "no major caveats recorded"
    grouping_note = (
        "Grouped evaluation was used; cross-group evidence is primary.\n\n"
        if report.grouping.get("provided")
        else ""
    )
    family_set_text = _render_plausible_family_set(report)
    return (
        f"Recommendation: {headline}\n\n"
        f"{' '.join(report.decision_path[:3])}\n\n"
        f"{family_set_text}"
        f"{grouping_note}"
        "This is an explicit regression diagnostic. Evidence is compared across "
        "variance-weighted and uniform-average R2, with normalized RMSE and "
        "target-neighborhood smoothness as supporting diagnostics.\n\n"
        f"Suggested next models:\n{suggestion_text}\n\n"
        f"Confidence: {report.confidence}.\n"
        f"Main caveats: {caveat_text}."
    )
