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
}


def render_recommendation(report: DiagnosticReport) -> str:
    """Render a concise plain-text recommendation from a diagnostic report."""
    headline = RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    suggestions = _SUGGESTIONS.get(report.recommendation, _SUGGESTIONS["inconclusive"])
    caveats = report.warnings[:2]
    if not caveats and report.skipped_diagnostics:
        caveats = [entry["reason"] for entry in report.skipped_diagnostics[:2]]
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    caveat_text = "; ".join(caveats) if caveats else "no major caveats recorded"
    return (
        f"Recommendation: {headline}\n\n"
        f"{' '.join(report.decision_path[:3])}\n\n"
        f"Suggested next models:\n{suggestion_text}\n\n"
        f"Confidence: {report.confidence}.\n"
        f"Main caveats: {caveat_text}."
    )
