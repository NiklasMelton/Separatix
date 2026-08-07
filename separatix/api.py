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
    """Diagnose apparent supervised complexity from features and targets.

    Args:
        X: Numeric feature matrix with shape ``(n_samples, n_features)``. Dense
            NumPy arrays, SciPy sparse matrices, and pandas DataFrames are
            accepted.
        y: Target values with one row per sample. The accepted shape and values
            depend on ``target_mode``.
        groups: Optional group identifier for every row. Groups are kept intact
            during sampling and held-out evaluation.
        return_report: Return a structured :class:`DiagnosticReport` instead of
            the plain-text recommendation.
        target_mode: Target routing mode. Regression is explicit opt-in, while
            auto mode detects unambiguous two-dimensional multilabel indicators.
        multilabel_stratification: Multilabel split strategy. Auto mode uses
            iterative stratification when installed and otherwise falls back to
            deterministic heuristic stratification.
        budget: Diagnostic effort level.
        topology: Optional topology behavior. Graph summaries do not require the
            persistent-homology extra.
        densify_policy: Behavior when a dense-only diagnostic meets sparse or
            over-budget data.
        max_dense_mb: Hard estimated memory limit for dense operations.
        max_samples: Optional hard row cap for diagnostic sampling.
        random_state: Seed used by sampling, splits, and randomized probes.
        warn_on_densify: Emit runtime warnings when densification or dense
            subsampling occurs.
        mlp_probes: Enable conditional feed-forward MLP probes. PyTorch must be
            installed through the ``mlp`` extra.
        mlp_device: Device policy for optional MLP probes.
        mlp_trigger_skill_threshold: Minimum simpler-probe skill used only by the
            MLP compute-trigger policy. It does not gate a completed MLP override.
        mlp_min_improvement: Minimum held-out MLP improvement required for an
            override against both dummy and the strongest simpler probe.
        mlp_max_parameters: Optional hard cap on the MLP parameter count.

    Returns:
        A plain-text recommendation by default, or a
        :class:`DiagnosticReport` when ``return_report=True``.

    Raises:
        ValueError: If inputs or configuration are invalid.
        MemoryError: If a required dense operation exceeds ``max_dense_mb``
            under the ``"fail"`` densification policy.

    Note:
        This function provides coarse diagnostic guidance. It does not select
        or fit a final predictive model.
    """
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
