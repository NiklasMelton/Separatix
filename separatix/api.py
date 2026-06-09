"""Functional API for separatix."""

from __future__ import annotations

from typing import Any, Literal

from separatix.profiler import ComplexityProfiler
from separatix.report import DiagnosticReport


def diagnose(
    X: Any,
    y: Any,
    *,
    return_report: bool = False,
    budget: Literal["fast", "standard", "extended"] = "standard",
    topology: Literal["off", "auto", "graph", "persistent"] = "auto",
    densify_policy: Literal["fail", "warn_and_sample", "skip"] = ("warn_and_sample"),
    max_dense_mb: int = 512,
    max_samples: int | None = None,
    random_state: int | None = None,
    warn_on_densify: bool = True,
) -> str | DiagnosticReport:
    """Diagnose apparent classification complexity from embeddings and labels."""
    profiler = ComplexityProfiler(
        budget=budget,
        topology=topology,
        densify_policy=densify_policy,
        max_dense_mb=max_dense_mb,
        max_samples=max_samples,
        random_state=random_state,
        warn_on_densify=warn_on_densify,
    )
    report = profiler.fit(X, y).report_
    if report is None:
        raise RuntimeError("Profiler did not produce a report.")
    return report if return_report else report.recommendation_text
