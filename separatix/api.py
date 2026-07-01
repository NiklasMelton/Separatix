"""Functional API for separatix."""

from __future__ import annotations

from typing import Any, Literal

from separatix.profiler import ComplexityProfiler
from separatix.report import DiagnosticReport


def diagnose(
    X: Any,
    y: Any,
    *,
    groups: Any = None,
    return_report: bool = False,
    target_mode: Literal["auto", "singlelabel", "multilabel", "regression"] = "auto",
    multilabel_stratification: Literal["auto", "iterative", "heuristic"] = "auto",
    budget: Literal["fast", "standard", "extended"] = "standard",
    topology: Literal["off", "auto", "graph", "persistent"] = "auto",
    densify_policy: Literal["fail", "warn_and_sample", "skip"] = ("warn_and_sample"),
    max_dense_mb: int = 512,
    max_samples: int | None = None,
    random_state: int | None = None,
    warn_on_densify: bool = True,
    mlp_probes: bool = False,
    mlp_device: Literal["cpu", "auto", "cuda", "mps"] = "cpu",
    mlp_trigger_skill_threshold: float = 0.75,
    mlp_min_improvement: float = 0.02,
    mlp_max_parameters: int | None = None,
) -> str | DiagnosticReport:
    """Diagnose apparent supervised complexity from embeddings and targets."""
    profiler = ComplexityProfiler(
        target_mode=target_mode,
        multilabel_stratification=multilabel_stratification,
        budget=budget,
        topology=topology,
        densify_policy=densify_policy,
        max_dense_mb=max_dense_mb,
        max_samples=max_samples,
        random_state=random_state,
        warn_on_densify=warn_on_densify,
        mlp_probes=mlp_probes,
        mlp_device=mlp_device,
        mlp_trigger_skill_threshold=mlp_trigger_skill_threshold,
        mlp_min_improvement=mlp_min_improvement,
        mlp_max_parameters=mlp_max_parameters,
    )
    report = profiler.fit(X, y, groups=groups).report_
    if report is None:
        raise RuntimeError("Profiler did not produce a report.")
    return report if return_report else report.recommendation_text
