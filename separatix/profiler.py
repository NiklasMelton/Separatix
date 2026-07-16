"""Estimator-style profiler API."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.metrics.audit import (
    compute_dataset_audit,
    compute_multilabel_audit,
    compute_regression_audit,
)
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
    compute_regression_neighborhood_diagnostics,
)
from separatix.metrics.topology import (
    compute_multilabel_topology_diagnostics,
    compute_regression_topology_diagnostics,
    compute_topology_diagnostics,
)
from separatix.models.mlp import (
    maybe_run_multilabel_mlp_probes,
    maybe_run_regression_mlp_probes,
    maybe_run_singlelabel_mlp_probes,
)
from separatix.models.probes import (
    run_model_probes,
    run_multilabel_model_probes,
    run_regression_model_probes,
)
from separatix.preprocessing import build_preprocessing_summary
from separatix.recommendation.engine import (
    compute_multilabel_scores,
    compute_regression_scores,
    compute_scores,
    make_multilabel_recommendation,
    make_recommendation,
    make_regression_recommendation,
)
from separatix.recommendation.text import render_recommendation
from separatix.report import DiagnosticReport
from separatix.validation import (
    is_multilabel_indicator,
    validate_inputs,
    validate_multilabel_inputs,
    validate_regression_inputs,
)


def _deduplicate_messages(messages: list[str]) -> list[str]:
    """Return messages in first-seen order without duplicates."""
    return list(dict.fromkeys(messages))


def _reliability_skip_count(entries: list[dict[str, Any]]) -> int:
    """Count only skips that can affect recommendation reliability."""
    return sum(
        entry.get("status") not in {"not_applicable", "informational"}
        for entry in entries
    )


class ComplexityProfiler:
    """Estimator-style diagnostic profiler for supervised feature spaces.

    Diagnostics run during :meth:`fit` and are stored in :attr:`report_`. The
    class intentionally has no ``predict`` method because it recommends coarse
    model-family complexity rather than fitting a production predictor.

    Args:
        target_mode: Target routing mode. Regression is explicit opt-in.
        multilabel_stratification: Multilabel split strategy.
        budget: Diagnostic effort level.
        topology: Optional topology behavior.
        densify_policy: Behavior for dense-only diagnostics.
        max_dense_mb: Hard estimated memory limit for dense operations.
        max_samples: Optional hard row cap for diagnostic sampling.
        min_dense_samples: Minimum useful dense sample size after subsampling.
        random_state: Seed for deterministic sampling and randomized probes.
        warn_on_densify: Emit runtime warnings for densification events.
        n_jobs: Optional parallel job count forwarded to supported estimators.
        mlp_probes: Enable conditional feed-forward MLP probes.
        mlp_device: Device policy for optional MLP probes.
        mlp_trigger_skill_threshold: Minimum simpler-probe skill for the MLP
            trigger policy.
        mlp_min_improvement: Minimum held-out MLP gain required for an override.
        mlp_max_parameters: Optional hard MLP parameter cap.

    Attributes:
        config: Validated profiler configuration.
        report_: Most recently fitted report, or ``None`` before :meth:`fit`.
    """

    def __init__(
        self,
        *,
        target_mode: Literal[
            "auto", "singlelabel", "multilabel", "regression"
        ] = "auto",
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
        mlp_probes: bool = False,
        mlp_device: Literal["cpu", "auto", "cuda", "mps"] = "cpu",
        mlp_trigger_skill_threshold: float = 0.75,
        mlp_min_improvement: float = 0.02,
        mlp_max_parameters: int | None = None,
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
            mlp_probes=mlp_probes,
            mlp_device=mlp_device,
            mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
            mlp_min_improvement=mlp_min_improvement,
            mlp_max_parameters=mlp_max_parameters,
        )
        self.report_: DiagnosticReport | None = None

    def fit(self, X: Any, y: Any, *, groups: Any = None) -> ComplexityProfiler:
        """Run diagnostics and store the resulting report.

        Args:
            X: Numeric feature matrix with one row per sample.
            y: Targets accepted by the configured target mode.
            groups: Optional group identifier for every sample.

        Returns:
            This fitted profiler instance.

        Raises:
            ValueError: If inputs are invalid or incompatible with the target
                mode.
        """
        if self.config.target_mode == "regression":
            return self._fit_regression(X, y, groups=groups)
        if self.config.target_mode == "multilabel":
            return self._fit_multilabel(X, y, groups=groups)
        if self.config.target_mode == "auto":
            if is_multilabel_indicator(y, allow_single_column=False):
                return self._fit_multilabel(X, y, groups=groups)
        return self._fit_singlelabel(X, y, groups=groups)

    def _fit_singlelabel(
        self, X: Any, y: Any, *, groups: Any = None
    ) -> ComplexityProfiler:
        """Run single-label diagnostics and store the resulting report."""
        start = time.perf_counter()
        validated = validate_inputs(X, y, groups=groups)
        report_context: dict[str, Any] = {
            "warnings": list(validated.warnings),
            "errors": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        }

        audit = compute_dataset_audit(
            validated.X,
            validated.y_encoded,
            classes=validated.classes_,
            is_sparse=validated.is_sparse,
        )
        X_evaluable = (
            validated.X[validated.evaluable_mask]
            if not sparse.issparse(validated.X)
            else validated.X[validated.evaluable_mask, :]
        )
        if not np.all(validated.evaluable_mask):
            report_context["skipped_diagnostics"].append(
                {
                    "name": "unsupported_group_classes",
                    "reason": (
                        "At least one class cannot be evaluated across distinct "
                        "groups, so model-family guidance for the full task is unsafe."
                    ),
                    "severity": "blocking",
                    "count": len(
                        validated.grouping_summary.get(
                            "skipped_singlelabel_classes", []
                        )
                    ),
                }
            )
        geometry = compute_geometry_diagnostics(
            X_evaluable,
            validated.evaluable_y_encoded,
            config=self.config,
            report_context=report_context,
            groups=validated.evaluable_groups,
        )
        probes = run_model_probes(
            X_evaluable,
            validated.evaluable_y_encoded,
            config=self.config,
            report_context=report_context,
            class_labels=validated.evaluable_classes_,
            groups=validated.evaluable_groups,
        )
        baseline = summarize_probe_family(probes)
        neighborhood = compute_neighborhood_diagnostics(
            X_evaluable,
            validated.evaluable_y_encoded,
            config=self.config,
            report_context=report_context,
            groups=validated.evaluable_groups,
        )
        boundary = compute_boundary_candidates(
            validated.evaluable_y_encoded, neighborhood, probes
        )
        graph = compute_graph_fragmentation(
            X_evaluable,
            validated.evaluable_y_encoded,
            boundary,
            config=self.config,
            groups=validated.evaluable_groups,
        )
        topology = compute_topology_diagnostics(
            X_evaluable,
            validated.evaluable_y_encoded,
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
        mlp_evidence = maybe_run_singlelabel_mlp_probes(
            X_evaluable,
            validated.evaluable_y_encoded,
            config=self.config,
            metrics=metrics,
            report_context=report_context,
            class_labels=validated.evaluable_classes_,
            groups=validated.evaluable_groups,
        )
        metrics["mlp_trigger_evidence"] = mlp_evidence.get("trigger", {})
        metrics["mlp_probes"] = mlp_evidence
        metrics["mlp_recommendation_evidence"] = {
            key: value for key, value in mlp_evidence.items() if key != "trigger"
        }
        report_context["warnings"] = _deduplicate_messages(report_context["warnings"])
        metrics["quality_context"] = {
            "skipped_diagnostics": report_context["skipped_diagnostics"],
            "errors": report_context["errors"],
        }
        scores = compute_scores(
            metrics,
            skipped_count=_reliability_skip_count(
                report_context["skipped_diagnostics"]
            ),
            warning_count=len(report_context["warnings"]),
        )
        recommendation, confidence, decision_path, interpretations = (
            make_recommendation(scores, metrics)
        )
        class_summary = {
            "n_classes": validated.n_classes,
            "classes": validated.classes_.tolist(),
            "evaluable_classes": validated.evaluable_classes_.tolist(),
            "class_counts": audit["class_counts"],
            "imbalance_ratio": audit["imbalance_ratio"],
            "min_class_count": min(audit["class_counts"].values()),
            "max_class_count": max(audit["class_counts"].values()),
        }
        validated.grouping_summary["effective_supervised_evaluation_mode"] = probes[
            "linear"
        ].get("evaluation_mode")
        report = DiagnosticReport(
            recommendation=recommendation,
            recommendation_text="",
            confidence=confidence,
            metrics=metrics,
            scores=scores,
            interpretations=interpretations,
            decision_path=decision_path,
            warnings=report_context["warnings"],
            errors=report_context["errors"],
            skipped_diagnostics=report_context["skipped_diagnostics"],
            preprocessing=build_preprocessing_summary(
                validated.X, is_sparse=validated.is_sparse
            ),
            sampling={
                "probe": probes["linear"].get("sample_info"),
                "neighbors": neighborhood.get("sampling"),
                "boundary": graph.get("sampling"),
                "mlp": mlp_evidence.get("sample_info"),
            },
            densification_events=report_context["densification_events"],
            class_summary=class_summary,
            grouping=validated.grouping_summary,
            runtime={"total_seconds": float(time.perf_counter() - start)},
            config=self.config.to_dict(),
        )
        report.recommendation_text = render_recommendation(report)
        self.report_ = report
        return self

    def _fit_regression(
        self, X: Any, y: Any, *, groups: Any = None
    ) -> ComplexityProfiler:
        """Run regression diagnostics and store the resulting report."""
        start = time.perf_counter()
        validated = validate_regression_inputs(X, y, groups=groups)
        report_context: dict[str, Any] = {
            "warnings": list(validated.warnings),
            "errors": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        }
        skipped_target_indices = np.flatnonzero(~validated.usable_target_mask)
        if skipped_target_indices.size:
            report_context["skipped_diagnostics"].append(
                {
                    "name": "constant_regression_targets",
                    "reason": (
                        "Some regression targets were constant and excluded from "
                        "probe-family scoring."
                    ),
                    "count": int(skipped_target_indices.size),
                    "targets": [
                        str(validated.target_names[idx])
                        for idx in skipped_target_indices[:20]
                    ],
                }
            )

        Y_usable = validated.Y[:, validated.usable_target_mask]
        target_names = validated.target_names[validated.usable_target_mask]
        audit = compute_regression_audit(validated)
        geometry = compute_geometry_diagnostics(
            validated.X,
            np.arange(validated.n_samples),
            config=self.config,
            report_context=report_context,
            groups=validated.groups,
        )
        probes = run_regression_model_probes(
            validated.X,
            Y_usable,
            config=self.config,
            report_context=report_context,
            target_names=target_names,
            groups=validated.groups,
        )
        neighborhood = compute_regression_neighborhood_diagnostics(
            validated.X,
            Y_usable,
            config=self.config,
            report_context=report_context,
            groups=validated.groups,
        )
        topology_context: dict[str, Any] = {
            "warnings": [],
            "errors": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        }
        topology = compute_regression_topology_diagnostics(
            validated.X,
            Y_usable,
            probes,
            neighborhood,
            config=self.config,
            report_context=topology_context,
        )
        skipped_common = {
            "status": "not_applicable",
            "skipped_reason": (
                "Classification boundary diagnostics are not used for continuous "
                "regression targets."
            ),
        }
        metrics = {
            "audit": audit,
            "geometry": geometry,
            "probes": probes,
            "baseline": self._regression_baseline_summary(probes),
            "neighborhood": neighborhood,
            "boundary": skipped_common,
            "graph": skipped_common,
            "topology": topology,
        }
        mlp_evidence = maybe_run_regression_mlp_probes(
            validated.X,
            Y_usable,
            config=self.config,
            metrics=metrics,
            report_context=report_context,
            target_names=target_names,
            groups=validated.groups,
        )
        metrics["mlp_trigger_evidence"] = mlp_evidence.get("trigger", {})
        metrics["mlp_probes"] = mlp_evidence
        metrics["mlp_recommendation_evidence"] = {
            key: value for key, value in mlp_evidence.items() if key != "trigger"
        }
        report_context["warnings"] = _deduplicate_messages(report_context["warnings"])
        metrics["quality_context"] = {
            "skipped_diagnostics": report_context["skipped_diagnostics"],
            "errors": report_context["errors"],
        }
        scores = compute_regression_scores(
            metrics,
            skipped_count=_reliability_skip_count(
                report_context["skipped_diagnostics"]
            ),
            warning_count=len(report_context["warnings"]),
        )
        recommendation, confidence, decision_path, interpretations = (
            make_regression_recommendation(scores, metrics)
        )
        report_context["warnings"].extend(topology_context["warnings"])
        report_context["skipped_diagnostics"].extend(
            topology_context["skipped_diagnostics"]
        )
        report_context["densification_events"].extend(
            topology_context["densification_events"]
        )
        report_context["errors"].extend(topology_context["errors"])
        target_summary = {
            "target_type": "regression",
            "n_targets": validated.n_targets,
            "target_names": validated.target_names.tolist(),
            "usable_target_names": target_names.tolist(),
            "constant_target_count": int(np.sum(~validated.usable_target_mask)),
            "target_variance_summary": audit["target_variance_summary"],
        }
        validated.grouping_summary["effective_supervised_evaluation_mode"] = probes[
            "linear"
        ].get("evaluation_mode")
        report = DiagnosticReport(
            recommendation=recommendation,
            recommendation_text="",
            confidence=confidence,
            metrics=metrics,
            scores=scores,
            interpretations=interpretations,
            decision_path=decision_path,
            warnings=report_context["warnings"],
            errors=report_context["errors"],
            skipped_diagnostics=report_context["skipped_diagnostics"],
            preprocessing=build_preprocessing_summary(
                validated.X,
                is_sparse=validated.is_sparse_X,
            ),
            sampling={
                "probe": probes["linear"].get("sample_info"),
                "neighbors": neighborhood.get("sampling"),
                "boundary": None,
                "mlp": mlp_evidence.get("sample_info"),
                "topology": [
                    obj.get("sampling")
                    for obj in topology.get("objects", [])
                    if obj.get("sampling") is not None
                ],
            },
            densification_events=report_context["densification_events"],
            class_summary=target_summary,
            grouping=validated.grouping_summary,
            runtime={"total_seconds": float(time.perf_counter() - start)},
            config=self.config.to_dict(),
        )
        report.recommendation_text = render_recommendation(report)
        self.report_ = report
        return self

    def _fit_multilabel(
        self, X: Any, y: Any, *, groups: Any = None
    ) -> ComplexityProfiler:
        """Run multilabel diagnostics and store the resulting report."""
        start = time.perf_counter()
        validated = validate_multilabel_inputs(X, y, groups=groups)
        report_context: dict[str, Any] = {
            "warnings": list(validated.warnings),
            "errors": [],
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
            groups=validated.groups,
        )
        neighborhood = compute_multilabel_neighborhood_diagnostics(
            validated.X,
            Y_usable,
            config=self.config,
            report_context=report_context,
            groups=validated.groups,
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
            report_context=report_context,
            groups=validated.groups,
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
        mlp_evidence = maybe_run_multilabel_mlp_probes(
            validated.X,
            Y_usable,
            config=self.config,
            metrics=metrics,
            report_context=report_context,
            label_names=label_names,
            groups=validated.groups,
        )
        metrics["mlp_trigger_evidence"] = mlp_evidence.get("trigger", {})
        metrics["mlp_probes"] = mlp_evidence
        metrics["mlp_recommendation_evidence"] = {
            key: value for key, value in mlp_evidence.items() if key != "trigger"
        }
        report_context["warnings"] = _deduplicate_messages(report_context["warnings"])
        metrics["quality_context"] = {
            "skipped_diagnostics": report_context["skipped_diagnostics"],
            "errors": report_context["errors"],
        }
        scores = compute_multilabel_scores(
            metrics,
            skipped_count=_reliability_skip_count(
                report_context["skipped_diagnostics"]
            ),
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
        validated.grouping_summary["effective_supervised_evaluation_mode"] = probes[
            "linear"
        ].get("evaluation_mode")
        report = DiagnosticReport(
            recommendation=recommendation,
            recommendation_text="",
            confidence=confidence,
            metrics=metrics,
            scores=scores,
            interpretations=interpretations,
            decision_path=decision_path,
            warnings=report_context["warnings"],
            errors=report_context["errors"],
            skipped_diagnostics=report_context["skipped_diagnostics"],
            preprocessing=build_preprocessing_summary(
                validated.X,
                is_sparse=validated.is_sparse_X,
            ),
            sampling={
                "probe": probes["linear"].get("sample_info"),
                "neighbors": neighborhood.get("sampling"),
                "boundary": graph.get("sampling"),
                "mlp": mlp_evidence.get("sample_info"),
            },
            densification_events=report_context["densification_events"],
            class_summary=class_summary,
            grouping=validated.grouping_summary,
            runtime={"total_seconds": float(time.perf_counter() - start)},
            config=self.config.to_dict(),
        )
        report.recommendation_text = render_recommendation(report)
        self.report_ = report
        return self

    def report(self) -> DiagnosticReport:
        """Return the fitted diagnostic report.

        Raises:
            ValueError: If :meth:`fit` has not been called successfully.
        """
        if self.report_ is None:
            raise ValueError("Profiler has not been fit yet.")
        return self.report_

    def recommendation(self) -> str:
        """Return the fitted report's plain-text recommendation.

        Raises:
            ValueError: If :meth:`fit` has not been called successfully.
        """
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

    @staticmethod
    def _regression_baseline_summary(
        probes: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Return compact best-probe summaries for regression primary metrics."""
        primary = ("r2_variance_weighted", "r2_uniform_average")
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
