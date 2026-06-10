"""Estimator-style profiler API."""

from __future__ import annotations

import time
from typing import Any, Literal

from separatix.config import ProfilerConfig
from separatix.metrics.audit import compute_dataset_audit
from separatix.metrics.baseline import summarize_probe_family
from separatix.metrics.boundary import compute_boundary_candidates
from separatix.metrics.geometry import compute_geometry_diagnostics
from separatix.metrics.graph import compute_graph_fragmentation
from separatix.metrics.neighborhood import compute_neighborhood_diagnostics
from separatix.metrics.topology import compute_topology_diagnostics
from separatix.models.probes import run_model_probes
from separatix.preprocessing import build_preprocessing_summary
from separatix.recommendation.engine import compute_scores, make_recommendation
from separatix.recommendation.text import render_recommendation
from separatix.report import DiagnosticReport
from separatix.validation import validate_inputs


class ComplexityProfiler:
    """Diagnostic profiler for labeled embedding classification problems."""

    def __init__(
        self,
        *,
        budget: Literal["fast", "standard", "extended"] = "standard",
        topology: Literal["off", "auto", "graph", "persistent"] = "auto",
        densify_policy: Literal["fail", "warn_and_sample", "skip"] = (
            "warn_and_sample"
        ),
        max_dense_mb: int = 512,
        max_samples: int | None = None,
        min_dense_samples: int = 200,
        random_state: int | None = None,
        warn_on_densify: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        """Initialize the profiler with validated configuration."""
        self.config = ProfilerConfig(
            budget=budget,
            topology=topology,
            densify_policy=densify_policy,
            max_dense_mb=max_dense_mb,
            max_samples=max_samples,
            min_dense_samples=min_dense_samples,
            random_state=random_state,
            warn_on_densify=warn_on_densify,
            n_jobs=n_jobs,
        )
        self.report_: DiagnosticReport | None = None

    def fit(self, X: Any, y: Any) -> ComplexityProfiler:
        """Run diagnostics and store the resulting report in report_."""
        start = time.perf_counter()
        validated = validate_inputs(X, y)
        report_context: dict[str, Any] = {
            "warnings": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        }

        audit = compute_dataset_audit(
            validated.X,
            validated.y_encoded,
            classes=validated.classes_,
            is_sparse=validated.is_sparse,
        )
        geometry = compute_geometry_diagnostics(
            validated.X,
            validated.y_encoded,
            config=self.config,
            report_context=report_context,
        )
        probes = run_model_probes(
            validated.X,
            validated.y_encoded,
            config=self.config,
            report_context=report_context,
            class_labels=validated.classes_,
        )
        baseline = summarize_probe_family(probes)
        neighborhood = compute_neighborhood_diagnostics(
            validated.X,
            validated.y_encoded,
            config=self.config,
            report_context=report_context,
        )
        boundary = compute_boundary_candidates(
            validated.y_encoded, neighborhood, probes
        )
        graph = compute_graph_fragmentation(
            validated.X,
            validated.y_encoded,
            boundary,
            config=self.config,
        )
        topology = compute_topology_diagnostics(
            validated.X,
            validated.y_encoded,
            boundary,
            geometry,
            config=self.config,
            report_context=report_context,
        )
        metrics = {
            "audit": audit,
            "geometry": geometry,
            "probes": probes,
            "baseline": baseline,
            "neighborhood": neighborhood,
            "boundary": boundary,
            "graph": graph,
            "topology": topology,
        }
        scores = compute_scores(
            metrics,
            skipped_count=len(report_context["skipped_diagnostics"]),
            warning_count=len(report_context["warnings"]),
        )
        recommendation, confidence, decision_path, interpretations = (
            make_recommendation(scores, metrics)
        )
        class_summary = {
            "n_classes": validated.n_classes,
            "classes": validated.classes_.tolist(),
            "class_counts": audit["class_counts"],
            "imbalance_ratio": audit["imbalance_ratio"],
            "min_class_count": min(audit["class_counts"].values()),
            "max_class_count": max(audit["class_counts"].values()),
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
                validated.X, is_sparse=validated.is_sparse
            ),
            sampling={
                "probe": probes["linear"].get("sample_info"),
                "neighbors": neighborhood.get("sampling"),
                "boundary": graph.get("sampling"),
            },
            densification_events=report_context["densification_events"],
            class_summary=class_summary,
            runtime={"total_seconds": float(time.perf_counter() - start)},
            config=self.config.to_dict(),
        )
        report.recommendation_text = render_recommendation(report)
        self.report_ = report
        return self

    def report(self) -> DiagnosticReport:
        """Return the fitted DiagnosticReport."""
        if self.report_ is None:
            raise ValueError("Profiler has not been fit yet.")
        return self.report_

    def recommendation(self) -> str:
        """Return the plain-text recommendation."""
        return self.report().recommendation_text
