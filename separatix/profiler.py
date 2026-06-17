"""Estimator-style profiler API."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.metrics.audit import compute_dataset_audit, compute_multilabel_audit
from separatix.metrics.baseline import summarize_probe_family
from separatix.metrics.boundary import (
    compute_boundary_candidates,
    compute_multilabel_boundary_candidates,
)
from separatix.metrics.geometry import compute_geometry_diagnostics
from separatix.metrics.graph import (
    compute_graph_fragmentation,
    compute_multilabel_graph_fragmentation,
)
from separatix.metrics.neighborhood import (
    compute_multilabel_neighborhood_diagnostics,
    compute_neighborhood_diagnostics,
)
from separatix.metrics.topology import (
    compute_multilabel_topology_diagnostics,
    compute_topology_diagnostics,
)
from separatix.models.probes import run_model_probes, run_multilabel_model_probes
from separatix.preprocessing import build_preprocessing_summary
from separatix.recommendation.engine import (
    compute_multilabel_scores,
    compute_scores,
    make_multilabel_recommendation,
    make_recommendation,
)
from separatix.recommendation.text import render_recommendation
from separatix.report import DiagnosticReport
from separatix.validation import (
    is_multilabel_indicator,
    validate_inputs,
    validate_multilabel_inputs,
)


class ComplexityProfiler:
    """Diagnostic profiler for labeled embedding classification problems."""

    def __init__(
        self,
        *,
        target_mode: Literal["auto", "singlelabel", "multilabel"] = "auto",
        multilabel_stratification: Literal["auto", "iterative", "heuristic"] = "auto",
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
            target_mode=target_mode,
            multilabel_stratification=multilabel_stratification,
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
        if self.config.target_mode == "multilabel":
            return self._fit_multilabel(X, y)
        if self.config.target_mode == "auto":
            if is_multilabel_indicator(y, allow_single_column=False):
                return self._fit_multilabel(X, y)
        return self._fit_singlelabel(X, y)

    def _fit_singlelabel(self, X: Any, y: Any) -> ComplexityProfiler:
        """Run single-label diagnostics and store the resulting report."""
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

    def _fit_multilabel(self, X: Any, y: Any) -> ComplexityProfiler:
        """Run multilabel diagnostics and store the resulting report."""
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

        Y_usable = (
            validated.Y[:, validated.usable_label_mask]
            if sparse.issparse(validated.Y)
            else np.asarray(validated.Y)[:, validated.usable_label_mask]
        )
        label_names = validated.label_names[validated.usable_label_mask]
        audit = compute_multilabel_audit(validated)
        probes = run_multilabel_model_probes(
            validated.X,
            Y_usable,
            config=self.config,
            report_context=report_context,
            label_names=label_names,
        )
        neighborhood = compute_multilabel_neighborhood_diagnostics(
            validated.X,
            Y_usable,
            config=self.config,
            report_context=report_context,
        )
        boundary = compute_multilabel_boundary_candidates(
            Y_usable,
            neighborhood,
            probes,
            label_names=label_names,
        )
        graph = compute_multilabel_graph_fragmentation(
            validated.X,
            Y_usable,
            boundary,
            config=self.config,
        )
        topology = compute_multilabel_topology_diagnostics(
            validated.X,
            Y_usable,
            boundary,
            config=self.config,
            report_context=report_context,
            label_names=label_names,
        )
        metrics = {
            "audit": audit,
            "probes": probes,
            "baseline": self._multilabel_baseline_summary(probes),
            "neighborhood": neighborhood,
            "boundary": boundary,
            "graph": graph,
            "topology": topology,
        }
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

    @staticmethod
    def _multilabel_baseline_summary(
        probes: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Return compact best-probe summaries for multilabel primary metrics."""
        primary = ("micro_f1", "macro_f1", "sample_jaccard")
        summary: dict[str, Any] = {
            "primary_metrics": list(primary),
            "best_by_metric": {},
        }
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
