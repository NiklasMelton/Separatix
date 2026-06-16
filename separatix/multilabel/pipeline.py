"""Top-level multilabel diagnostic pipeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.multilabel.audit import compute_multilabel_audit
from separatix.multilabel.neighborhood import (
    compute_multilabel_neighborhood_diagnostics,
)
from separatix.multilabel.probes import run_multilabel_model_probes
from separatix.multilabel.recommendation import (
    compute_multilabel_scores,
    make_multilabel_recommendation,
)
from separatix.multilabel.text import render_multilabel_recommendation
from separatix.multilabel.validation import validate_multilabel_inputs
from separatix.preprocessing import build_preprocessing_summary
from separatix.report import DiagnosticReport


def _slice_columns(Y: Any, mask: np.ndarray) -> Any:
    """Slice dense or sparse target columns."""
    return Y[:, mask] if sparse.issparse(Y) else np.asarray(Y)[:, mask]


def _baseline_summary(probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return compact best-probe summaries for multilabel primary metrics."""
    primary = ("micro_f1", "macro_f1", "sample_jaccard")
    summary: dict[str, Any] = {"primary_metrics": list(primary), "best_by_metric": {}}
    for metric in primary:
        candidates = [
            (name, result[metric])
            for name, result in probes.items()
            if metric in result and name != "dummy"
        ]
        if not candidates:
            summary["best_by_metric"][metric] = None
        else:
            name, score = max(candidates, key=lambda item: item[1])
            summary["best_by_metric"][metric] = {
                "probe": name,
                "score": float(score),
            }
    return summary


def fit_multilabel(X: Any, y: Any, *, config: ProfilerConfig) -> DiagnosticReport:
    """Run the multilabel diagnostic pipeline and return a report."""
    start = time.perf_counter()
    validated = validate_multilabel_inputs(X, y)
    report_context: dict[str, Any] = {
        "warnings": list(validated.warnings),
        "skipped_diagnostics": [],
        "densification_events": [],
    }

    skipped_label_indices = np.flatnonzero(~validated.usable_label_mask)
    if skipped_label_indices.size:
        report_context["skipped_diagnostics"].append(
            {
                "name": "unsupported_multilabel_columns",
                "reason": (
                    "Some labels were constant or had fewer than two positive "
                    "or negative examples."
                ),
                "count": int(skipped_label_indices.size),
                "labels": [
                    str(validated.label_names[idx])
                    for idx in skipped_label_indices[:20]
                ],
            }
        )

    Y_usable = _slice_columns(validated.Y, validated.usable_label_mask)
    label_names = validated.label_names[validated.usable_label_mask]
    audit = compute_multilabel_audit(validated)
    probes = run_multilabel_model_probes(
        validated.X,
        Y_usable,
        config=config,
        report_context=report_context,
        label_names=label_names,
    )
    neighborhood = compute_multilabel_neighborhood_diagnostics(
        validated.X,
        Y_usable,
        config=config,
        report_context=report_context,
    )
    metrics = {
        "audit": audit,
        "probes": probes,
        "baseline": _baseline_summary(probes),
        "neighborhood": neighborhood,
        "boundary": {
            "skipped_reason": (
                "multilabel boundary diagnostics require label-set disagreement "
                "semantics and are not implemented in this phase"
            )
        },
        "graph": {
            "skipped_reason": (
                "multilabel graph fragmentation is not implemented until "
                "multilabel boundary candidates are available"
            )
        },
        "topology": {
            "skipped_reason": (
                "topology is skipped for the initial multilabel diagnostic path"
            )
        },
    }
    report_context["skipped_diagnostics"].extend(
        [
            {
                "name": "multilabel_boundary",
                "reason": metrics["boundary"]["skipped_reason"],
            },
            {
                "name": "multilabel_graph",
                "reason": metrics["graph"]["skipped_reason"],
            },
            {
                "name": "multilabel_topology",
                "reason": metrics["topology"]["skipped_reason"],
            },
        ]
    )
    scores = compute_multilabel_scores(
        metrics,
        skipped_count=len(report_context["skipped_diagnostics"]),
        warning_count=len(report_context["warnings"]),
    )
    recommendation, confidence, decision_path, interpretations = (
        make_multilabel_recommendation(scores, metrics)
    )
    class_summary = {
        "target_type": "multilabel",
        "n_labels": validated.n_labels,
        "usable_label_count": int(np.sum(validated.usable_label_mask)),
        "label_names": [str(item) for item in validated.label_names.tolist()],
        "usable_label_names": [str(item) for item in label_names.tolist()],
        "label_counts": [int(item) for item in validated.label_counts.tolist()],
        "label_cardinality_mean": audit["label_cardinality_mean"],
        "label_density": audit["label_density"],
        "all_zero_sample_fraction": audit["all_zero_sample_fraction"],
    }
    report = DiagnosticReport(
        recommendation=recommendation,
        recommendation_text="",
        confidence=confidence,
        metrics=metrics,
        scores=scores,
        interpretations=interpretations,
        decision_path=decision_path,
        warnings=report_context["warnings"],
        errors=[],
        skipped_diagnostics=report_context["skipped_diagnostics"],
        preprocessing=build_preprocessing_summary(
            validated.X,
            is_sparse=validated.is_sparse_X,
        ),
        sampling={
            "probe": probes["linear"].get("sample_info"),
            "neighbors": neighborhood.get("sampling"),
            "boundary": None,
        },
        densification_events=report_context["densification_events"],
        class_summary=class_summary,
        runtime={"total_seconds": float(time.perf_counter() - start)},
        config=config.to_dict(),
    )
    report.recommendation_text = render_multilabel_recommendation(report)
    return report
