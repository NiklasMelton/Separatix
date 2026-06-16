"""Plain-text rendering for multilabel recommendations."""

from __future__ import annotations

from separatix.constants import RECOMMENDATION_LABELS
from separatix.recommendation.text import _SUGGESTIONS
from separatix.report import DiagnosticReport


def render_multilabel_recommendation(report: DiagnosticReport) -> str:
    """Render a concise plain-text multilabel recommendation."""
    headline = RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    suggestions = _SUGGESTIONS.get(report.recommendation, _SUGGESTIONS["inconclusive"])
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    caveats = report.warnings[:2]
    if not caveats and report.skipped_diagnostics:
        caveats = [entry["reason"] for entry in report.skipped_diagnostics[:2]]
    caveat_text = "; ".join(caveats) if caveats else "no major caveats recorded"
    return (
        f"Recommendation: {headline}\n\n"
        f"{' '.join(report.decision_path[:3])}\n\n"
        "This is a multilabel diagnostic. Evidence is compared across micro F1, "
        "macro F1, and sample Jaccard rather than collapsed into one weighted "
        "score.\n\n"
        f"Suggested next models:\n{suggestion_text}\n\n"
        f"Confidence: {report.confidence}.\n"
        f"Main caveats: {caveat_text}."
    )
