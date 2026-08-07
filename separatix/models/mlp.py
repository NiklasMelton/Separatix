"""Optional conditional feed-forward MLP probes."""

# ruff: noqa: E501

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from separatix.config import ProfilerConfig
from separatix.densify import (
    ensure_dense_multilabel_target,
    ensure_dense_or_sample,
    ensure_dense_or_sample_regression,
)
from separatix.models.comparison import (
    _build_paired_score_cache,
    _summarize_cached_probe_pair,
)
from separatix.models.probes import (
    _choose_sketch_components,
    _dense_multilabel_matrix,
    _ensure_dense_X_for_multilabel,
    _estimate_dense_mb,
    _full_multilabel_quadratic_classifier,
    _full_quadratic_classifier,
    _linear_classifier,
    _low_rank_multilabel_quadratic_classifier,
    _low_rank_quadratic_classifier,
    _multilabel_linear_classifier,
    _quadratic_feature_count,
    _regression_smooth_estimator,
    _scaled_pipeline,
)
from separatix.models.scoring import (
    MultilabelPriorDummy,
    TargetMeanDummyRegressor,
    choose_cv,
    choose_regression_cv,
    primary_metric_scores,
    summarize_multilabel_predictions,
    summarize_predictions,
    summarize_regression_predictions,
)
from separatix.recipes import build_probe_recipe
from separatix.sampling import (
    cap_multilabel_samples_for_budget,
    cap_regression_samples_for_budget,
    cap_samples_for_budget,
    choose_multilabel_cv,
    choose_multilabel_holdout,
)
from separatix.utils.warnings import record_warning

_MLP_BUDGETS: dict[str, dict[str, int]] = {
    "fast": {
        "max_samples": 2000,
        "cv_folds": 3,
        "bootstrap_repeats": 200,
        "epochs": 64,
        "patience": 8,
        "max_parameters": 250_000,
    },
    "standard": {
        "max_samples": 5000,
        "cv_folds": 3,
        "bootstrap_repeats": 500,
        "epochs": 128,
        "patience": 16,
        "max_parameters": 1_000_000,
    },
    "extended": {
        "max_samples": 10000,
        "cv_folds": 5,
        "bootstrap_repeats": 1000,
        "epochs": 256,
        "patience": 24,
        "max_parameters": 5_000_000,
    },
}
_MLP_HIDDEN_LABELS = (
    "one_layer_compact",
    "two_layer_compact",
    "one_layer_wide",
    "two_layer_wide",
)
_MLP_RUNTIME_DIM_WARNING = 2048
_MLP_RUNTIME_WORK_WARNING = 1e12
_REQUIRED_MLP_COMPARATORS = (
    "dummy",
    "linear",
    "smooth_poly",
    "knn",
    "kernel_approx",
)
_SIMPLER_MLP_COMPARATORS = _REQUIRED_MLP_COMPARATORS[1:]

# Keep every Torch training choice in one JSON-compatible policy.  The policy
# is copied per estimator below, then used by ``_TorchMLPBase.fit`` and passed
# to ``build_probe_recipe``.  This prevents an audited recipe from drifting
# away from the training implementation when a default changes.
_MLP_TRAINING_POLICY: dict[str, Any] = {
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.95],
        "weight_decay": 1e-4,
    },
    "schedule": {
        "name": "warmup_cosine",
        "warmup_epochs": 5,
        "warmup_start_factor": 0.2,
        "cosine_min_factor": 0.0,
    },
    "initialization": {
        "hidden": {"method": "kaiming_normal", "nonlinearity": "relu"},
        "output": {"method": "xavier_uniform"},
        "bias": {"method": "zeros"},
    },
    "early_stopping": {
        "monitor": "validation_loss",
        "min_delta": 1e-6,
        "restore_best": True,
    },
    "gradient_clip": {"method": "clip_grad_norm", "max_norm": 5.0},
    "loss": {
        "singlelabel": {
            "name": "CrossEntropyLoss",
            "class_weight": "inverse_frequency_balanced",
        },
        "multilabel": {
            "name": "BCEWithLogitsLoss",
            "positive_weight": "negative_over_positive",
            "positive_weight_clip": [0.05, 20.0],
        },
        "regression": {"name": "MSELoss"},
    },
}


def _mlp_training_policy(estimator: Any) -> dict[str, Any]:
    """Return the resolved JSON-compatible Torch policy for one estimator."""
    policy = deepcopy(_MLP_TRAINING_POLICY)
    policy["fit"] = {
        "epochs": int(estimator.epochs),
        "patience": int(estimator.patience),
        "batch_size": int(estimator.batch_size),
        "device": str(estimator.device),
        "random_state": (
            None if estimator.random_state is None else int(estimator.random_state)
        ),
    }
    # Explicitly state the task so a recipe remains self-describing even when
    # a consumer only inspects its training policy.
    policy["task"] = str(estimator.task)
    return policy


def _pairwise_comparison_audit(
    config: ProfilerConfig,
    *,
    status: Literal["available", "unavailable", "not_run"] = "not_run",
    reason: str | None = None,
    resamples_used: int = 0,
    resample_plan_id: str | None = None,
    comparators_by_metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compact audit record for MLP paired comparisons.

    MLP comparisons run on a capped, dense cohort that is intentionally
    independent from the ordinary probe-comparison cache.  Keep this metadata
    explicit so a report can distinguish an unavailable paired comparison from
    one that simply was not requested (or was not triggered).
    """
    return {
        "status": status,
        "method": "paired_oof_bootstrap",
        "scope": "dummy_and_metric_strongest_simpler",
        "resamples_requested": int(_MLP_BUDGETS[config.budget]["bootstrap_repeats"]),
        "resamples_used": int(resamples_used),
        "resample_plan_id": resample_plan_id,
        "comparators_by_metric": dict(comparators_by_metric or {}),
        "reason": reason,
    }


def _set_pairwise_audit(
    payload: dict[str, Any],
    config: ProfilerConfig,
    *,
    status: Literal["available", "unavailable", "not_run"],
    reason: str | None,
) -> None:
    """Set paired-comparison audit metadata on an in-progress MLP payload."""
    payload["pairwise_comparison_audit"] = _pairwise_comparison_audit(
        config,
        status=status,
        reason=reason,
    )


def _torch_module() -> Any | None:
    """Return a usable torch installation, otherwise None."""
    try:
        spec = find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.loader is None:
        return None
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None
    required_api = (
        "Tensor",
        "cuda",
        "manual_seed",
        "nn",
        "no_grad",
        "optim",
        "random",
        "tensor",
        "utils",
    )
    if any(not hasattr(torch, attribute) for attribute in required_api):
        return None
    return torch


def _default_mlp_artifacts(config: ProfilerConfig) -> dict[str, Any]:
    """Return the default MLP evidence payload."""
    return {
        "status": "not_requested" if not config.mlp_probes else "not_triggered",
        "reason": (
            "MLP probes were disabled."
            if not config.mlp_probes
            else "A simpler probe already met the configured skill threshold."
        ),
        "sample_info": None,
        "backend": {"requested_device": config.mlp_device, "resolved_device": None},
        "architectures": [],
        "aligned_comparators": {},
        "best_architecture": None,
        "pairwise_comparisons": {},
        "pairwise_comparison_audit": _pairwise_comparison_audit(
            config,
            status="not_run",
            reason=(
                "MLP paired comparisons were not run because MLP probes were disabled."
                if not config.mlp_probes
                else "MLP paired comparisons have not been run."
            ),
        ),
        "required_comparators_complete": False,
        "architectures_complete": None,
        "recommendation_override": False,
        "override_reason": None,
        "override_policy": "paired_improvement_and_dummy_signal",
        "trigger_threshold_used_for_override": False,
        "minimum_improvement": float(config.mlp_min_improvement),
        "required_comparators": list(_REQUIRED_MLP_COMPARATORS),
        "missing_or_failed_comparators": [],
        "strongest_simpler_probe_by_metric": {},
        "metrics_beating_strongest_simpler": [],
        "metrics_beating_dummy": [],
        "metrics_clearing_override": [],
        "required_metrics_to_override": None,
        "absolute_skill_by_metric": {},
    }


def _mlp_budget(config: ProfilerConfig) -> dict[str, int]:
    """Return the per-budget MLP limits."""
    budget = dict(_MLP_BUDGETS[config.budget])
    if config.mlp_max_parameters is not None:
        budget["max_parameters"] = int(config.mlp_max_parameters)
    return budget


_MLP_COMPARATOR_FAMILIES = {
    "dummy": "dummy",
    "linear": "linear",
    "smooth_poly": "smooth_nonlinear",
    "knn": "local_kernel",
    "kernel_approx": "local_kernel",
}
_MLP_COMPARATOR_IMPLEMENTATION_SUFFIXES = {
    "dummy": "dummy",
    "linear": "linear",
    "smooth_poly": "smooth_poly",
    "knn": "knn",
    "kernel_approx": "kernel_approx",
}


def _probe_input_contract(
    X: np.ndarray,
    *,
    n_outputs: int,
    sample_info: dict[str, Any],
) -> dict[str, Any]:
    """Describe the dense representation and resolution used by an MLP probe."""
    resolution_n_samples = sample_info.get("n_used")
    if resolution_n_samples is None:
        resolution_n_samples = X.shape[0]
    return {
        "representation": "dense",
        "n_features": int(X.shape[1]),
        "n_outputs": int(n_outputs),
        "resolution_n_samples": int(resolution_n_samples),
    }


def _comparator_training_policy(
    estimator: Any, config: ProfilerConfig
) -> dict[str, Any]:
    """Describe deterministic settings and scoring-time comparator adjustments."""
    adjustments: dict[str, Any] = {}
    try:
        params = estimator.get_params(deep=True)
    except (AttributeError, TypeError, ValueError):
        params = {}
    for key, value in params.items():
        if key == "n_neighbors" or key.endswith("__n_neighbors"):
            try:
                configured = int(value)
            except (TypeError, ValueError):
                continue
            adjustments["knn_n_neighbors"] = {
                "parameter": key,
                "configured_n_neighbors": configured,
                "rule": "max(1, min(configured_n_neighbors, fold_train_size))",
                "source": "separatix.models.scoring._prepared_estimator",
            }
            break
    return {
        "evaluation_random_state": config.random_state,
        "evaluation_deterministic": config.random_state is not None,
        "scoring_time_estimator_adjustments": adjustments,
    }


def _attach_mlp_probe_recipe(
    result: dict[str, Any],
    estimator: Any,
    *,
    probe_name: str,
    target_mode: Literal["singlelabel", "multilabel", "regression"],
    family: str,
    role: str,
    X: np.ndarray,
    n_outputs: int,
    sample_info: dict[str, Any],
    config: ProfilerConfig,
    variant: str | None = None,
    training_policy: dict[str, Any] | None = None,
    implementation_key: str | None = None,
) -> dict[str, Any]:
    """Attach a recipe and available status to one constructed estimator result."""
    result["probe_recipe"] = build_probe_recipe(
        estimator,
        probe_name=probe_name,
        family=family,
        target_mode=target_mode,
        role=role,
        variant=variant,
        input_contract=_probe_input_contract(
            X,
            n_outputs=n_outputs,
            sample_info=sample_info,
        ),
        training_policy=training_policy,
        implementation_key=implementation_key,
        implementation_version=1,
    )
    result["probe_recipe_status"] = {"status": "available", "reason": None}
    return result


def _mark_mlp_recipe_unavailable(
    result: dict[str, Any], reason: str | None = None
) -> dict[str, Any]:
    """Mark an unconstructed or skipped MLP result without inventing a recipe."""
    result["probe_recipe"] = None
    result["probe_recipe_status"] = {
        "status": "unavailable",
        "reason": str(reason) if reason else "estimator was not constructed",
    }
    return result


def _skill_from_bounds(
    probe_lower: float | None,
    dummy_upper: float | None,
) -> float | None:
    """Return normalized predictive skill above dummy."""
    if probe_lower is None or dummy_upper is None:
        return None
    return float(
        np.clip(
            (probe_lower - dummy_upper) / max(1e-9, 1.0 - dummy_upper),
            0.0,
            1.0,
        )
    )


def _balanced_accuracy_error(
    score: float,
    per_class_recall: list[float] | None,
    class_counts: list[int],
    stability_std: float | None,
) -> float:
    """Return a conservative balanced-accuracy error estimate."""
    if per_class_recall and len(per_class_recall) == len(class_counts):
        recalls = [float(item) for item in per_class_recall]
    else:
        recalls = [float(score) for _ in class_counts]
    variance = sum(
        recall * (1.0 - recall) / max(1, class_counts[index])
        for index, recall in enumerate(recalls)
    ) / max(1, len(recalls) ** 2)
    error = math.sqrt(max(0.0, variance))
    if stability_std is not None:
        error = max(error, float(stability_std))
    return float(error)


def _simple_singlelabel_trigger(
    metrics: dict[str, Any], config: ProfilerConfig
) -> dict[str, Any]:
    """Return trigger evidence for single-label MLP probes."""
    probes = metrics.get("probes", {})
    counts = [
        int(value)
        for value in metrics.get("audit", {}).get("class_counts", {}).values()
    ]
    dummy = probes.get("dummy", {})
    if "balanced_accuracy" not in dummy:
        return {
            "status": "triggered",
            "reason": "The dummy probe was unavailable, so no simple probe met the threshold.",
            "good_enough": False,
            "threshold": config.mlp_trigger_skill_threshold,
            "best_simple_probe": None,
            "best_simple_skill": None,
            "per_probe_skill": {},
        }
    dummy_error = _balanced_accuracy_error(
        float(dummy["balanced_accuracy"]),
        dummy.get("per_class_recall"),
        counts,
        dummy.get("stability_balanced_accuracy_std"),
    )
    dummy_upper = min(1.0, float(dummy["balanced_accuracy"]) + dummy_error)
    per_probe_skill: dict[str, float | None] = {}
    best_name: str | None = None
    best_skill: float | None = None
    for name in ("linear", "smooth_poly", "knn", "kernel_approx"):
        probe = probes.get(name, {})
        if "balanced_accuracy" not in probe:
            per_probe_skill[name] = None
            continue
        error = _balanced_accuracy_error(
            float(probe["balanced_accuracy"]),
            probe.get("per_class_recall"),
            counts,
            probe.get("stability_balanced_accuracy_std"),
        )
        skill = _skill_from_bounds(
            max(0.0, float(probe["balanced_accuracy"]) - error),
            dummy_upper,
        )
        per_probe_skill[name] = skill
        if skill is not None and (best_skill is None or skill > best_skill):
            best_name = name
            best_skill = skill
    good_enough = bool(
        best_skill is not None and best_skill >= config.mlp_trigger_skill_threshold
    )
    return {
        "status": "not_triggered" if good_enough else "triggered",
        "reason": (
            f"The {best_name} probe already recovered {best_skill:.3f} normalized skill above dummy."
            if good_enough and best_name is not None and best_skill is not None
            else "No simpler single-label probe met the configured skill threshold."
        ),
        "good_enough": good_enough,
        "threshold": config.mlp_trigger_skill_threshold,
        "best_simple_probe": best_name,
        "best_simple_skill": best_skill,
        "per_probe_skill": per_probe_skill,
    }


def _multilabel_metric_error(
    result: dict[str, Any], metric: str, n_samples: int
) -> float:
    """Return a conservative multilabel metric error estimate."""
    stability = result.get(f"stability_{metric}_std")
    if stability is not None:
        return float(stability)
    score = float(result.get(metric, 0.0))
    return float(math.sqrt(max(0.0, score * (1.0 - score)) / max(1, n_samples)))


def _simple_multilabel_trigger(
    metrics: dict[str, Any], config: ProfilerConfig
) -> dict[str, Any]:
    """Return trigger evidence for multilabel MLP probes."""
    probes = metrics.get("probes", {})
    n_samples = int(metrics.get("audit", {}).get("n_samples", 1))
    metric_names = ("micro_f1", "macro_f1", "sample_jaccard")
    skill_table: dict[str, dict[str, Any]] = {}
    strong_metrics = 0
    safe_third_metric = 0
    for metric in metric_names:
        dummy = probes.get("dummy", {})
        dummy_score = dummy.get(metric)
        if dummy_score is None:
            skill_table[metric] = {"best_probe": None, "skill": None}
            continue
        dummy_error = _multilabel_metric_error(dummy, metric, n_samples)
        dummy_upper = min(1.0, float(dummy_score) + dummy_error)
        best_probe: str | None = None
        best_skill: float | None = None
        best_lower: float | None = None
        for name in ("linear", "smooth_poly", "knn", "kernel_approx"):
            probe = probes.get(name, {})
            if metric not in probe:
                continue
            error = _multilabel_metric_error(probe, metric, n_samples)
            lower = max(0.0, float(probe[metric]) - error)
            skill = _skill_from_bounds(lower, dummy_upper)
            if skill is not None and (best_skill is None or skill > best_skill):
                best_probe = name
                best_skill = skill
                best_lower = lower
        if best_skill is not None and best_skill >= config.mlp_trigger_skill_threshold:
            strong_metrics += 1
        if best_lower is None or best_lower >= max(
            0.0, float(dummy_score) - dummy_error
        ):
            safe_third_metric += 1
        skill_table[metric] = {"best_probe": best_probe, "skill": best_skill}
    good_enough = strong_metrics >= 2 and safe_third_metric >= 3
    return {
        "status": "not_triggered" if good_enough else "triggered",
        "reason": (
            "At least two primary multilabel metrics already met the configured skill threshold."
            if good_enough
            else "No simpler multilabel probe family met the configured skill threshold on enough primary metrics."
        ),
        "good_enough": good_enough,
        "threshold": config.mlp_trigger_skill_threshold,
        "metric_skill": skill_table,
    }


def _simple_regression_trigger(
    metrics: dict[str, Any], config: ProfilerConfig
) -> dict[str, Any]:
    """Return trigger evidence for regression MLP probes."""
    probes = metrics.get("probes", {})
    metric_names = ("r2_variance_weighted", "r2_uniform_average")
    metric_skill: dict[str, dict[str, Any]] = {}
    good_metrics = 0
    for metric in metric_names:
        dummy = probes.get("dummy", {})
        dummy_score = dummy.get(metric)
        if dummy_score is None:
            metric_skill[metric] = {"best_probe": None, "skill": None}
            continue
        dummy_error = float(dummy.get(f"stability_{metric}_std") or 0.0)
        dummy_upper = min(1.0, float(dummy_score) + dummy_error)
        best_probe: str | None = None
        best_skill: float | None = None
        for name in ("linear", "smooth_poly", "knn", "kernel_approx"):
            probe = probes.get(name, {})
            if metric not in probe:
                continue
            error = float(probe.get(f"stability_{metric}_std") or 0.0)
            lower = max(-1.0, float(probe[metric]) - error)
            skill = _skill_from_bounds(min(1.0, lower), dummy_upper)
            if skill is not None and (best_skill is None or skill > best_skill):
                best_probe = name
                best_skill = skill
        if best_skill is not None and best_skill >= config.mlp_trigger_skill_threshold:
            good_metrics += 1
        metric_skill[metric] = {"best_probe": best_probe, "skill": best_skill}
    good_enough = good_metrics == len(metric_names)
    return {
        "status": "not_triggered" if good_enough else "triggered",
        "reason": (
            "Simpler regression probes already met the configured skill threshold on both primary metrics."
            if good_enough
            else "No simpler regression probe family met the configured skill threshold on both primary metrics."
        ),
        "good_enough": good_enough,
        "threshold": config.mlp_trigger_skill_threshold,
        "metric_skill": metric_skill,
    }


def _resolve_device(torch: Any, requested: str) -> tuple[str, str | None]:
    """Resolve the torch device from the user request."""
    if requested == "auto":
        if bool(torch.cuda.is_available()):
            return "cuda", None
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps", None
        return "cpu", None
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        return (
            "cpu",
            "CUDA was requested for MLP probes but is unavailable; falling back to CPU.",
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not bool(mps.is_available()):
            return (
                "cpu",
                "MPS was requested for MLP probes but is unavailable; falling back to CPU.",
            )
    return requested, None


def _next_power_of_two(value: float) -> int:
    """Return the next power of two at or above value."""
    return 1 if value <= 1 else 1 << int(math.ceil(math.log2(value)))


def _parameter_count(input_dim: int, output_dim: int, hidden: tuple[int, ...]) -> int:
    """Return the parameter count for a dense ReLU MLP."""
    layer_dims = (input_dim, *hidden, output_dim)
    return int(
        sum(
            (layer_dims[index] + 1) * layer_dims[index + 1]
            for index in range(len(layer_dims) - 1)
        )
    )


def _two_layer_width_for_budget(input_dim: int, output_dim: int, budget: int) -> int:
    """Return the largest equal hidden width that fits the matched budget."""
    linear_term = input_dim + output_dim + 2
    discriminant = (linear_term**2) - 4 * (output_dim - budget)
    if discriminant <= 0:
        return 0
    return max(0, int(math.floor((-linear_term + math.sqrt(discriminant)) / 2.0)))


def _architecture_candidates(
    input_dim: int,
    output_dim: int,
    *,
    max_parameters: int,
) -> list[dict[str, Any]]:
    """Return the architecture ladder filtered to the available budget."""
    compact_width = int(
        np.clip(_next_power_of_two(math.sqrt(max(1, input_dim * output_dim))), 16, 256)
    )
    wide_width = min(512, 4 * compact_width)
    one_compact = (compact_width,)
    one_wide = (wide_width,)
    one_compact_params = _parameter_count(input_dim, output_dim, one_compact)
    one_wide_params = _parameter_count(input_dim, output_dim, one_wide)
    two_compact_width = _two_layer_width_for_budget(
        input_dim, output_dim, one_compact_params
    )
    two_wide_width = _two_layer_width_for_budget(input_dim, output_dim, one_wide_params)
    candidates: list[dict[str, Any]] = [
        {
            "label": "one_layer_compact",
            "hidden_layer_sizes": one_compact,
            "tier": "compact",
            "depth": 1,
        },
        {
            "label": "two_layer_compact",
            "hidden_layer_sizes": (two_compact_width, two_compact_width),
            "tier": "compact",
            "depth": 2,
        },
        {
            "label": "one_layer_wide",
            "hidden_layer_sizes": one_wide,
            "tier": "wide",
            "depth": 1,
        },
        {
            "label": "two_layer_wide",
            "hidden_layer_sizes": (two_wide_width, two_wide_width),
            "tier": "wide",
            "depth": 2,
        },
    ]
    filtered: list[dict[str, Any]] = []
    for item in candidates:
        widths = tuple(int(value) for value in item["hidden_layer_sizes"])
        if any(value <= 0 for value in widths):
            continue
        param_count = _parameter_count(input_dim, output_dim, widths)
        if param_count > max_parameters:
            continue
        filtered.append(
            {
                **item,
                "hidden_layer_sizes": widths,
                "parameter_count": param_count,
            }
        )
    return filtered


def _slice_rows(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Slice dense row arrays."""
    return np.asarray(X[indices], dtype=np.float32)


def _split_rows(
    X: np.ndarray,
    y: Any,
    *,
    target_mode: Literal["singlelabel", "multilabel", "regression"],
    config: ProfilerConfig,
    max_folds: int,
    groups: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, str]:
    """Materialize outer evaluation splits."""
    if target_mode == "singlelabel":
        y_array = np.asarray(y, dtype=int)
        cv, method = choose_cv(
            y_array,
            max_folds,
            groups=groups,
            random_state=config.random_state,
        )
        if cv is None:
            return None, method
        singlelabel_splitter: Any = cv
        split_iter = (
            singlelabel_splitter.split(X, y_array, groups)
            if groups is not None
            else singlelabel_splitter.split(X, y_array)
        )
        return [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in split_iter
        ], ("group_cross_validation" if groups is not None else "cross_validation")
    if target_mode == "multilabel":
        y_dense = _dense_multilabel_matrix(y)
        cv, method = choose_multilabel_cv(
            y_dense,
            max_folds=max_folds,
            config=config,
            groups=groups,
        )
        if cv is None:
            return None, method
        split_y = (
            y_dense.ravel()
            if y_dense.shape[1] == 1 and method == "binary_stratified"
            else y_dense
        )
        multilabel_splitter: Any = cv
        split_iter = (
            multilabel_splitter.split(X, split_y, groups)
            if groups is not None
            else multilabel_splitter.split(X, split_y)
        )
        return [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in split_iter
        ], ("group_cross_validation" if groups is not None else "cross_validation")
    y_array = np.asarray(y, dtype=float)
    cv, method = choose_regression_cv(
        y_array,
        max_folds,
        groups=groups,
        random_state=config.random_state,
    )
    if cv is None:
        return None, method
    regression_splitter: Any = cv
    split_iter = (
        regression_splitter.split(X, y_array, groups)
        if groups is not None
        else regression_splitter.split(X, y_array)
    )
    return [
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in split_iter
    ], ("group_cross_validation" if groups is not None else "cross_validation")


def _singlelabel_validation_indices(
    y: np.ndarray,
    *,
    random_state: int | None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into fit and validation subsets."""
    n_rows = y.shape[0]
    if n_rows < 8:
        rows = np.arange(n_rows, dtype=int)
        return rows, rows[:0]
    if groups is not None and np.unique(groups).shape[0] >= 2:
        from sklearn.model_selection import GroupShuffleSplit

        for attempt in range(32):
            seed = None if random_state is None else random_state + attempt
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            fit_idx, valid_idx = next(splitter.split(np.zeros((n_rows, 1)), y, groups))
            if (
                np.unique(y[fit_idx]).shape[0] == np.unique(y).shape[0]
                and np.unique(y[valid_idx]).shape[0] == np.unique(y).shape[0]
            ):
                return fit_idx, valid_idx
        rows = np.arange(n_rows, dtype=int)
        return rows, rows[:0]
    min_count = int(np.min(np.bincount(y)))
    if min_count >= 2:
        from sklearn.model_selection import StratifiedShuffleSplit

        n_classes = int(np.unique(y).shape[0])
        valid_size = max(n_classes, int(math.ceil(0.2 * n_rows)))
        if n_rows - valid_size < n_classes:
            rows = np.arange(n_rows, dtype=int)
            return rows, rows[:0]
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=valid_size, random_state=random_state
        )
        fit_idx, valid_idx = next(splitter.split(np.zeros((n_rows, 1)), y))
        if (
            np.unique(y[fit_idx]).shape[0] == n_classes
            and np.unique(y[valid_idx]).shape[0] == n_classes
        ):
            return fit_idx, valid_idx
        rows = np.arange(n_rows, dtype=int)
        return rows, rows[:0]
    rows = np.arange(n_rows, dtype=int)
    split_at = max(1, int(math.floor(0.8 * n_rows)))
    return rows[:split_at], rows[split_at:]


def _multilabel_validation_indices(
    Y: np.ndarray,
    *,
    config: ProfilerConfig,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split multilabel training rows into fit and validation subsets."""
    n_rows = Y.shape[0]
    if n_rows < 8:
        rows = np.arange(n_rows, dtype=int)
        return rows, rows[:0]
    holdout, method = choose_multilabel_holdout(
        Y,
        repeats=1,
        config=config,
        groups=groups,
    )
    if holdout is None:
        rows = np.arange(n_rows, dtype=int)
        return rows, rows[:0]
    split_y = Y.ravel() if Y.shape[1] == 1 and method == "binary_stratified" else Y
    split_iter = (
        holdout.split(np.zeros((n_rows, 1)), split_y, groups)
        if groups is not None
        else holdout.split(np.zeros((n_rows, 1)), split_y)
    )
    train_idx, valid_idx = next(split_iter)
    return np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int)


def _regression_validation_indices(
    n_rows: int,
    *,
    random_state: int | None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split regression training rows into fit and validation subsets."""
    rows = np.arange(n_rows, dtype=int)
    if n_rows < 8:
        return rows, rows[:0]
    if groups is not None and np.unique(groups).shape[0] >= 2:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        return next(splitter.split(np.zeros((n_rows, 1)), rows, groups))
    from sklearn.model_selection import ShuffleSplit

    splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    return next(splitter.split(np.zeros((n_rows, 1)), rows))


class _TorchMLPBase(BaseEstimator):
    """Minimal torch-backed MLP used only when the optional extra is installed."""

    def __init__(
        self,
        *,
        task: Literal["singlelabel", "multilabel", "regression"],
        hidden_layer_sizes: tuple[int, ...],
        epochs: int,
        patience: int,
        batch_size: int,
        device: str,
        random_state: int | None,
        multilabel_stratification: Literal["auto", "iterative", "heuristic"] = "auto",
    ) -> None:
        """Store MLP hyperparameters for sklearn-style cloning."""
        self.task = task
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.multilabel_stratification = multilabel_stratification

    def _init_model(
        self,
        torch: Any,
        input_dim: int,
        output_dim: int,
        *,
        policy: dict[str, Any],
    ) -> Any:
        """Initialize the torch module and apply explicit weight initialization."""
        modules: list[Any] = []
        layer_dims = [input_dim, *self.hidden_layer_sizes, output_dim]
        initialization = policy["initialization"]
        for index in range(len(layer_dims) - 1):
            linear = torch.nn.Linear(layer_dims[index], layer_dims[index + 1])
            if index < len(layer_dims) - 2:
                hidden_init = initialization["hidden"]
                if hidden_init["method"] != "kaiming_normal":
                    raise ValueError(
                        "Unsupported hidden MLP initialization method: "
                        f"{hidden_init['method']}"
                    )
                torch.nn.init.kaiming_normal_(
                    linear.weight,
                    nonlinearity=str(hidden_init["nonlinearity"]),
                )
            else:
                output_init = initialization["output"]
                if output_init["method"] != "xavier_uniform":
                    raise ValueError(
                        "Unsupported output MLP initialization method: "
                        f"{output_init['method']}"
                    )
                torch.nn.init.xavier_uniform_(linear.weight)
            bias_init = initialization["bias"]
            if bias_init["method"] != "zeros":
                raise ValueError(
                    f"Unsupported MLP bias initialization method: {bias_init['method']}"
                )
            torch.nn.init.zeros_(linear.bias)
            modules.append(linear)
            if index < len(layer_dims) - 2:
                modules.append(torch.nn.ReLU())
        return torch.nn.Sequential(*modules)

    def _predict_tensor(self, torch: Any, logits: Any) -> np.ndarray:
        """Convert network outputs to numpy predictions."""
        if self.task == "singlelabel":
            return torch.argmax(logits, dim=1).detach().cpu().numpy()
        if self.task == "multilabel":
            return (torch.sigmoid(logits) >= 0.5).to(torch.int8).detach().cpu().numpy()
        values = logits.detach().cpu().numpy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def fit(self, X: Any, y: Any, *, groups: np.ndarray | None = None) -> _TorchMLPBase:
        """Fit the feed-forward MLP with early stopping on an inner holdout."""
        torch = _torch_module()
        if torch is None:
            raise RuntimeError("torch is required for MLP probes but is not installed.")
        policy = _mlp_training_policy(self)
        self.training_policy_ = deepcopy(policy)
        seed = int(self.random_state) if self.random_state is not None else 0
        X_array = np.asarray(X, dtype=np.float32)
        if self.task == "singlelabel":
            y_array = np.asarray(y, dtype=np.int64)
            output_dim = int(np.max(y_array)) + 1
            fit_idx, valid_idx = _singlelabel_validation_indices(
                y_array,
                random_state=self.random_state,
                groups=groups,
            )
        elif self.task == "multilabel":
            y_array = _dense_multilabel_matrix(y).astype(np.float32, copy=False)
            output_dim = int(y_array.shape[1])
            fit_idx, valid_idx = _multilabel_validation_indices(
                y_array,
                config=ProfilerConfig(
                    multilabel_stratification=self.multilabel_stratification
                ),
                groups=groups,
            )
        else:
            y_array = np.asarray(y, dtype=np.float32)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            output_dim = int(y_array.shape[1])
            fit_idx, valid_idx = _regression_validation_indices(
                X_array.shape[0],
                random_state=self.random_state,
                groups=groups,
            )

        self.x_scaler_ = StandardScaler().fit(X_array[fit_idx])
        X_fit = self.x_scaler_.transform(X_array[fit_idx]).astype(
            np.float32, copy=False
        )
        X_valid = (
            self.x_scaler_.transform(X_array[valid_idx]).astype(np.float32, copy=False)
            if valid_idx.size
            else np.empty((0, X_array.shape[1]), dtype=np.float32)
        )
        if self.task == "regression":
            self.y_mean_ = np.mean(y_array[fit_idx], axis=0, keepdims=True)
            self.y_scale_ = np.std(y_array[fit_idx], axis=0, keepdims=True)
            self.y_scale_[self.y_scale_ < 1e-6] = 1.0
            y_fit = ((y_array[fit_idx] - self.y_mean_) / self.y_scale_).astype(
                np.float32, copy=False
            )
            y_valid = ((y_array[valid_idx] - self.y_mean_) / self.y_scale_).astype(
                np.float32, copy=False
            )
        else:
            y_fit = y_array[fit_idx]
            y_valid = y_array[valid_idx]

        # fork_rng restores the caller's torch RNG after deterministic model
        # initialization. Batch ordering uses a dedicated CPU generator below.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = self._init_model(
                torch,
                X_array.shape[1],
                output_dim,
                policy=policy,
            ).to(self.device)
        batch_generator = torch.Generator(device="cpu")
        batch_generator.manual_seed(seed)
        optimizer_policy = policy["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_policy["learning_rate"]),
            betas=tuple(float(value) for value in optimizer_policy["betas"]),
            weight_decay=float(optimizer_policy["weight_decay"]),
        )

        def schedule(epoch: int) -> float:
            schedule_policy = policy["schedule"]
            warmup_epochs = int(schedule_policy["warmup_epochs"])
            if epoch < warmup_epochs:
                start_factor = float(schedule_policy["warmup_start_factor"])
                progress = float(epoch + 1) / max(1, warmup_epochs)
                return start_factor + (1.0 - start_factor) * progress
            progress = (epoch - warmup_epochs) / max(1, self.epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            minimum = float(schedule_policy["cosine_min_factor"])
            return minimum + (1.0 - minimum) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        loss_policy = policy["loss"][self.task]
        if self.task == "singlelabel":
            counts = np.bincount(y_fit.astype(np.int64))
            weights = np.sum(counts) / np.maximum(counts.shape[0] * counts, 1)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
            )
        elif self.task == "multilabel":
            positive = np.sum(y_fit, axis=0)
            negative = y_fit.shape[0] - positive
            positive_weight_clip = tuple(
                float(value) for value in loss_policy["positive_weight_clip"]
            )
            pos_weight = np.clip(
                negative / np.maximum(positive, 1.0),
                positive_weight_clip[0],
                positive_weight_clip[1],
            )
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    pos_weight, dtype=torch.float32, device=self.device
                )
            )
        else:
            loss_fn = torch.nn.MSELoss()

        X_fit_tensor = torch.tensor(X_fit, dtype=torch.float32, device=self.device)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
        if self.task == "singlelabel":
            y_fit_tensor = torch.tensor(y_fit, dtype=torch.long, device=self.device)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long, device=self.device)
        else:
            y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32, device=self.device)
            y_valid_tensor = torch.tensor(
                y_valid, dtype=torch.float32, device=self.device
            )

        best_state = None
        best_loss = float("inf")
        bad_epochs = 0
        epochs_trained = 0
        batch_size = max(1, min(self.batch_size, X_fit.shape[0]))
        clip_policy = policy["gradient_clip"]
        early_stopping_policy = policy["early_stopping"]
        for epoch in range(self.epochs):
            model.train()
            permutation = torch.randperm(
                X_fit_tensor.shape[0], generator=batch_generator, device="cpu"
            ).to(self.device)
            try:
                for start in range(0, X_fit_tensor.shape[0], batch_size):
                    batch = permutation[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(X_fit_tensor[batch])
                    loss = loss_fn(logits, y_fit_tensor[batch])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(clip_policy["max_norm"])
                    )
                    optimizer.step()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 8:
                    raise
                batch_size = max(8, batch_size // 2)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            scheduler.step()
            epochs_trained = epoch + 1
            if X_valid_tensor.shape[0] == 0:
                continue
            model.eval()
            with torch.no_grad():
                valid_loss = float(
                    loss_fn(model(X_valid_tensor), y_valid_tensor).item()
                )
            if valid_loss + float(early_stopping_policy["min_delta"]) < best_loss:
                best_loss = valid_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(policy["fit"]["patience"]):
                    break
        if best_state is not None and bool(early_stopping_policy["restore_best"]):
            model.load_state_dict(best_state)
        self.model_ = model
        self.training_summary_ = {
            "epochs_trained": int(epochs_trained),
            "best_validation_loss": None
            if best_loss == float("inf")
            else float(best_loss),
            "used_validation_split": bool(X_valid.shape[0] > 0),
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict discrete labels or continuous outputs."""
        torch = _torch_module()
        if torch is None:
            raise RuntimeError("torch is required for MLP probes but is not installed.")
        X_array = np.asarray(X, dtype=np.float32)
        X_scaled = self.x_scaler_.transform(X_array).astype(np.float32, copy=False)
        tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            prediction = self._predict_tensor(torch, self.model_(tensor))
        if self.task == "regression":
            values = (
                np.asarray(prediction, dtype=np.float32) * self.y_scale_ + self.y_mean_
            )
            return values
        return prediction


class TorchMLPClassifier(_TorchMLPBase, ClassifierMixin):
    """Torch-backed classifier used for single-label and multilabel MLP probes."""


class TorchMLPRegressor(_TorchMLPBase, RegressorMixin):
    """Torch-backed regressor used for regression MLP probes."""


def _fit_estimator(
    estimator: Any,
    X_train: np.ndarray,
    y_train: Any,
    *,
    groups: np.ndarray | None,
) -> Any:
    """Fit an estimator, passing groups only when supported."""
    fitted = clone(estimator)
    if isinstance(fitted, _TorchMLPBase):
        return fitted.fit(X_train, y_train, groups=groups)
    return fitted.fit(X_train, y_train)


def _evaluate_singlelabel_models(
    X: np.ndarray,
    y: np.ndarray,
    estimators: dict[str, Any],
    *,
    class_labels: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
    evaluation_mode: str,
    groups: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate aligned single-label models on shared outer splits."""
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        start = time.perf_counter()
        predictions = np.empty_like(y)
        training_summaries: list[dict[str, Any]] = []
        if splits is None:
            fitted = _fit_estimator(estimator, X, y, groups=groups)
            predictions = np.asarray(fitted.predict(X), dtype=int)
            if isinstance(fitted, _TorchMLPBase):
                training_summaries.append(dict(fitted.training_summary_))
        else:
            for train_idx, test_idx in splits:
                fitted = _fit_estimator(
                    estimator,
                    _slice_rows(X, train_idx),
                    y[train_idx],
                    groups=None if groups is None else groups[train_idx],
                )
                predictions[test_idx] = np.asarray(
                    fitted.predict(_slice_rows(X, test_idx)), dtype=int
                )
                if isinstance(fitted, _TorchMLPBase):
                    training_summaries.append(dict(fitted.training_summary_))
        summary = summarize_predictions(y, predictions, class_labels=class_labels)
        summary.update(
            {
                "runtime_seconds": float(time.perf_counter() - start),
                "predictions": predictions.tolist(),
                "evaluation_mode": evaluation_mode,
            }
        )
        if training_summaries:
            summary["training_summaries"] = training_summaries
        results[name] = summary
    return results


def _evaluate_multilabel_models(
    X: np.ndarray,
    Y: np.ndarray,
    estimators: dict[str, Any],
    *,
    label_names: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
    evaluation_mode: str,
    groups: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate aligned multilabel models on shared outer splits."""
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        start = time.perf_counter()
        predictions = np.zeros_like(Y, dtype=np.int8)
        training_summaries: list[dict[str, Any]] = []
        if splits is None:
            fitted = _fit_estimator(estimator, X, Y, groups=groups)
            predictions = _dense_multilabel_matrix(fitted.predict(X))
            if isinstance(fitted, _TorchMLPBase):
                training_summaries.append(dict(fitted.training_summary_))
        else:
            for train_idx, test_idx in splits:
                fitted = _fit_estimator(
                    estimator,
                    _slice_rows(X, train_idx),
                    Y[train_idx],
                    groups=None if groups is None else groups[train_idx],
                )
                predictions[test_idx] = _dense_multilabel_matrix(
                    fitted.predict(_slice_rows(X, test_idx))
                )
                if isinstance(fitted, _TorchMLPBase):
                    training_summaries.append(dict(fitted.training_summary_))
        summary = summarize_multilabel_predictions(
            Y,
            predictions,
            label_names=label_names,
        )
        summary.update(
            {
                "runtime_seconds": float(time.perf_counter() - start),
                "predictions": predictions.tolist(),
                "evaluation_mode": evaluation_mode,
            }
        )
        if training_summaries:
            summary["training_summaries"] = training_summaries
        results[name] = summary
    return results


def _evaluate_regression_models(
    X: np.ndarray,
    Y: np.ndarray,
    estimators: dict[str, Any],
    *,
    target_names: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
    evaluation_mode: str,
    groups: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate aligned regression models on shared outer splits."""
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        start = time.perf_counter()
        predictions = np.zeros_like(Y, dtype=float)
        training_summaries: list[dict[str, Any]] = []
        if splits is None:
            fitted = _fit_estimator(estimator, X, Y, groups=groups)
            predictions = np.asarray(fitted.predict(X), dtype=float).reshape(Y.shape)
            if isinstance(fitted, _TorchMLPBase):
                training_summaries.append(dict(fitted.training_summary_))
        else:
            for train_idx, test_idx in splits:
                fitted = _fit_estimator(
                    estimator,
                    _slice_rows(X, train_idx),
                    Y[train_idx],
                    groups=None if groups is None else groups[train_idx],
                )
                predictions[test_idx] = np.asarray(
                    fitted.predict(_slice_rows(X, test_idx)),
                    dtype=float,
                ).reshape(Y[test_idx].shape)
                if isinstance(fitted, _TorchMLPBase):
                    training_summaries.append(dict(fitted.training_summary_))
        summary = summarize_regression_predictions(
            Y,
            predictions,
            target_names=target_names,
        )
        summary.update(
            {
                "runtime_seconds": float(time.perf_counter() - start),
                "predictions": predictions.tolist(),
                "evaluation_mode": evaluation_mode,
            }
        )
        if training_summaries:
            summary["training_summaries"] = training_summaries
        results[name] = summary
    return results


def _safe_evaluate_models(
    evaluator: Callable[..., dict[str, dict[str, Any]]],
    estimators: dict[str, Any],
    *,
    errors: list[str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate models independently so one runtime failure is localized."""
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        try:
            results.update(evaluator(estimators={name: estimator}, **kwargs))
        except Exception as exc:  # optional backend failures must remain local
            message = f"MLP probe {name!r} failed: {type(exc).__name__}: {exc}"
            errors.append(message)
            results[name] = {
                "status": "runtime_failed",
                "error": message,
                "evaluation_mode": kwargs.get("evaluation_mode"),
            }
    return results


def _attach_aligned_comparator_recipes(
    results: dict[str, dict[str, Any]],
    estimators: dict[str, Any],
    *,
    target_mode: Literal["singlelabel", "multilabel", "regression"],
    X: np.ndarray,
    n_outputs: int,
    sample_info: dict[str, Any],
    config: ProfilerConfig,
    variants: dict[str, str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attach recipes to constructed aligned comparator results.

    A comparator omitted by a memory or budget gate is represented explicitly
    with an unavailable recipe status.  This keeps the aligned result schema
    auditable without fabricating an estimator that was never constructed.
    """
    variants = variants or {}
    for name, estimator in estimators.items():
        result = results.setdefault(name, {})
        family = _MLP_COMPARATOR_FAMILIES[name]
        implementation_suffix = _MLP_COMPARATOR_IMPLEMENTATION_SUFFIXES[name]
        _attach_mlp_probe_recipe(
            result,
            estimator,
            probe_name=name,
            family=family,
            target_mode=target_mode,
            role="mlp_aligned_comparator",
            X=X,
            n_outputs=n_outputs,
            sample_info=sample_info,
            config=config,
            variant=variants.get(name),
            training_policy=_comparator_training_policy(estimator, config),
            implementation_key=(
                f"separatix.probe.{target_mode}.mlp_comparator.{implementation_suffix}"
            ),
        )
    for name in _REQUIRED_MLP_COMPARATORS:
        if name not in results:
            result = results.setdefault(name, {})
            _mark_mlp_recipe_unavailable(result, "estimator was not constructed")
            result["status"] = "skipped"
    return results


def _attach_architecture_recipes(
    results: dict[str, dict[str, Any]],
    estimators: dict[str, Any],
    *,
    target_mode: Literal["singlelabel", "multilabel", "regression"],
    X: np.ndarray,
    n_outputs: int,
    sample_info: dict[str, Any],
    config: ProfilerConfig,
    variants: dict[str, str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attach exact recipes to each constructed MLP architecture result."""
    variants = variants or {}
    for name, estimator in estimators.items():
        result = results.setdefault(name, {})
        architecture_name = name.removeprefix("mlp_")
        _attach_mlp_probe_recipe(
            result,
            estimator,
            probe_name=name,
            family="mlp",
            target_mode=target_mode,
            role="mlp_architecture",
            X=X,
            n_outputs=n_outputs,
            sample_info=sample_info,
            config=config,
            variant=variants.get(name, architecture_name),
            training_policy=_mlp_training_policy(estimator),
            implementation_key=(
                f"separatix.probe.{target_mode}.mlp.{architecture_name}"
            ),
        )
    return results


def _balanced_accuracy_delta(
    y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    sample_idx: np.ndarray,
) -> float:
    """Return a balanced-accuracy delta on a bootstrap sample."""
    first_score = primary_metric_scores(
        y_true[sample_idx],
        first_pred[sample_idx],
        target_mode="singlelabel",
        metrics=("balanced_accuracy",),
    )["balanced_accuracy"]
    second_score = primary_metric_scores(
        y_true[sample_idx],
        second_pred[sample_idx],
        target_mode="singlelabel",
        metrics=("balanced_accuracy",),
    )["balanced_accuracy"]
    return float(first_score - second_score)


def _multilabel_metric_delta(
    Y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    sample_idx: np.ndarray,
    *,
    metric: str,
    label_names: np.ndarray,
) -> float:
    """Return a multilabel metric delta on a bootstrap sample."""
    first = primary_metric_scores(
        Y_true[sample_idx],
        first_pred[sample_idx],
        target_mode="multilabel",
        metrics=(metric,),
        names=label_names,
    )
    second = primary_metric_scores(
        Y_true[sample_idx],
        second_pred[sample_idx],
        target_mode="multilabel",
        metrics=(metric,),
        names=label_names,
    )
    return float(first[metric] - second[metric])


def _regression_metric_delta(
    Y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    sample_idx: np.ndarray,
    *,
    metric: str,
    target_names: np.ndarray,
) -> float:
    """Return a regression metric delta on a bootstrap sample."""
    first = primary_metric_scores(
        Y_true[sample_idx],
        first_pred[sample_idx],
        target_mode="regression",
        metrics=(metric,),
        names=target_names,
    )
    second = primary_metric_scores(
        Y_true[sample_idx],
        second_pred[sample_idx],
        target_mode="regression",
        metrics=(metric,),
        names=target_names,
    )
    return float(first[metric] - second[metric])


def _objective_score(result: dict[str, Any], *, metrics: tuple[str, ...]) -> float:
    """Return a simple mean objective across primary metrics."""
    values = [float(result[name]) for name in metrics if name in result]
    return float(np.mean(values)) if values else float("-inf")


def _finite_metric(result: dict[str, Any], metric: str) -> bool:
    """Return whether a result contains one finite numeric metric."""
    try:
        return metric in result and bool(np.isfinite(float(result[metric])))
    except (TypeError, ValueError):
        return False


def _select_best_architecture(
    architecture_results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    n_rows: int,
) -> tuple[str | None, dict[str, Any] | None]:
    """Select the MLP architecture using conservative ties."""
    usable_results = {
        name: result
        for name, result in architecture_results.items()
        if result.get("status") != "runtime_failed"
        and result.get("predictions") is not None
        and np.asarray(result["predictions"]).shape[0] == n_rows
        and all(_finite_metric(result, metric) for metric in metrics)
        and _objective_score(result, metrics=metrics) != float("-inf")
    }
    if not usable_results:
        return None, None
    ordered = sorted(
        usable_results.items(),
        key=lambda item: (
            _objective_score(item[1], metrics=metrics),
            -_MLP_HIDDEN_LABELS.index(item[0].replace("mlp_", "")),
        ),
        reverse=True,
    )
    name, result = ordered[0]
    return name, result


def _missing_or_failed_comparators(
    comparator_results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    n_rows: int,
) -> list[str]:
    """Return required comparators lacking complete aligned held-out evidence."""
    missing: list[str] = []
    for name in _REQUIRED_MLP_COMPARATORS:
        result = comparator_results.get(name)
        if result is None or result.get("status") == "runtime_failed":
            missing.append(name)
            continue
        predictions = result.get("predictions")
        if predictions is None or np.asarray(predictions).shape[0] != n_rows:
            missing.append(name)
            continue
        if any(not _finite_metric(result, metric) for metric in metrics):
            missing.append(name)
    return missing


def _strongest_simpler_by_metric(
    comparator_results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...],
) -> dict[str, str]:
    """Return the point-best non-dummy comparator for each primary metric."""
    strongest: dict[str, str] = {}
    for metric in metrics:
        candidates = [
            name
            for name in _SIMPLER_MLP_COMPARATORS
            if _finite_metric(comparator_results.get(name, {}), metric)
        ]
        if candidates:
            # ``max`` preserves the declared simple-to-complex order for exact ties.
            strongest[metric] = max(
                candidates,
                key=lambda name: float(comparator_results[name][metric]),
            )
    return strongest


def _absolute_skill_by_metric(
    best_result: dict[str, Any],
    comparator_results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...],
) -> dict[str, float | None]:
    """Return descriptive normalized MLP skill above dummy by metric."""
    dummy = comparator_results.get("dummy", {})
    return {
        metric: _skill_from_bounds(
            float(best_result[metric]) if _finite_metric(best_result, metric) else None,
            float(dummy[metric]) if _finite_metric(dummy, metric) else None,
        )
        for metric in metrics
    }


def _override_report_fields(
    *,
    config: ProfilerConfig,
    strongest: dict[str, str],
    beating_strongest: list[str],
    beating_dummy: list[str],
    clearing: list[str],
    required_metrics: int,
    missing_comparators: list[str],
    absolute_skill: dict[str, float | None],
) -> dict[str, Any]:
    """Return common transparent fields for an MLP override decision."""
    return {
        "override_policy": "paired_improvement_and_dummy_signal",
        "trigger_threshold_used_for_override": False,
        "minimum_improvement": float(config.mlp_min_improvement),
        "required_comparators": list(_REQUIRED_MLP_COMPARATORS),
        "missing_or_failed_comparators": missing_comparators,
        "strongest_simpler_probe_by_metric": strongest,
        "metrics_beating_strongest_simpler": beating_strongest,
        "metrics_beating_dummy": beating_dummy,
        "metrics_clearing_override": clearing,
        "required_metrics_to_override": int(required_metrics),
        "absolute_skill_by_metric": absolute_skill,
    }


def _failed_override_reason(
    *,
    missing_comparators: list[str],
    beating_strongest: list[str],
    beating_dummy: list[str],
    clearing: list[str],
    required_metrics: int,
) -> str:
    """Explain the first decisive MLP override gate that did not clear."""
    if missing_comparators:
        return (
            "MLP override was disabled because complete aligned held-out evidence "
            "was unavailable for required comparators: "
            + ", ".join(missing_comparators)
            + "."
        )
    if len(beating_dummy) < required_metrics:
        return (
            "The best MLP architecture did not show sufficient paired signal "
            "above the dummy baseline on the required primary metrics."
        )
    if len(beating_strongest) < required_metrics:
        return (
            "The best MLP architecture did not clearly improve over the strongest "
            "simpler probe on the required primary metrics."
        )
    if len(clearing) < required_metrics:
        return (
            "The best MLP architecture did not clear both the paired dummy-signal "
            "and strongest-simpler gates on enough of the same primary metrics."
        )
    return "The MLP override criteria were not satisfied."


def _pairwise_unavailable_override_reason(
    *,
    required_complete: bool,
    audit: Mapping[str, Any],
    fallback: str,
) -> str:
    """Prefer an explicit paired-resampling failure over a misleading gate reason."""
    if required_complete and audit.get("status") == "unavailable":
        detail = audit.get("reason")
        return (
            "Paired MLP resampling evidence was unavailable, so the override was "
            "disabled."
            + (f" {detail}" if detail else "")
        )
    return fallback


def _mlp_pairwise_cached_comparisons(
    *,
    best_name: str,
    best_result: Mapping[str, Any],
    comparator_results: Mapping[str, Mapping[str, Any]],
    y_true: np.ndarray,
    target_mode: Literal["singlelabel", "multilabel", "regression"],
    metrics: tuple[str, ...],
    config: ProfilerConfig,
    groups: np.ndarray | None,
    names: np.ndarray | None,
    strongest: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build retained MLP pairwise intervals from one target-aware score cache.

    The MLP cohort is capped and aligned independently of the ordinary probe
    cohort.  The best MLP, dummy baseline, and union of metric-specific
    strongest simpler predictions are supplied to one cache so every retained
    comparison uses exactly the same paired resampling plan; only the dummy
    baseline and metric-specific strongest simpler probe are retained in the
    report payload.
    """
    requested = int(_mlp_budget(config)["bootstrap_repeats"])
    comparators_by_metric = {
        metric: {
            "dummy": "dummy",
            "strongest_simpler": strongest.get(metric),
        }
        for metric in metrics
    }
    prediction_arrays: dict[str, np.ndarray] = {}
    point_scores: dict[str, Mapping[str, Any]] = {best_name: best_result}
    best_predictions = best_result.get("predictions")
    if best_predictions is not None:
        best_array = np.asarray(best_predictions)
        if best_array.shape[0] == y_true.shape[0]:
            prediction_arrays[best_name] = best_array
    # The cache need only contain probes that feed a retained comparison.  The
    # required-comparator completeness gate is evaluated separately, so a
    # missing required comparator cannot force unrelated cached scores to be
    # fabricated or retained.
    cache_comparator_names = {
        "dummy",
        *(name for name in strongest.values() if name is not None),
    }
    for name, result in comparator_results.items():
        if name not in cache_comparator_names:
            continue
        predictions = result.get("predictions")
        if predictions is None:
            continue
        array = np.asarray(predictions)
        if array.shape[0] != y_true.shape[0]:
            continue
        # Keep point scores alongside every prediction array.  The shared
        # summarizer uses these values for point deltas while the cache stores
        # only resample scores.
        prediction_arrays[name] = array
        point_scores[name] = result

    if best_name not in prediction_arrays or "dummy" not in prediction_arrays:
        audit = _pairwise_comparison_audit(
            config,
            status="unavailable",
            reason="best MLP and dummy predictions were not both aligned on the MLP cohort",
            comparators_by_metric=comparators_by_metric,
        )
        return {}, audit

    try:
        cache = _build_paired_score_cache(
            np.asarray(y_true),
            prediction_arrays,
            target_mode=target_mode,
            requested_resamples=requested,
            random_state=config.random_state,
            groups=groups,
            names=names,
            max_working_memory_mb=float(config.max_dense_mb),
        )
    except Exception as exc:  # cache failures must not mask recommendation evidence
        audit = _pairwise_comparison_audit(
            config,
            status="unavailable",
            reason=f"MLP paired score cache failed: {type(exc).__name__}: {exc}",
            comparators_by_metric=comparators_by_metric,
        )
        return {}, audit

    cache_status = str(getattr(cache, "status", "unavailable"))
    cache_reason = getattr(cache, "reason", None)
    audit = _pairwise_comparison_audit(
        config,
        status="available" if cache_status == "available" else "unavailable",
        reason=None if cache_status == "available" else str(cache_reason or "paired score cache was unavailable"),
        resamples_used=int(getattr(cache, "resamples_used", 0) or 0),
        resample_plan_id=getattr(cache, "resample_plan_id", None),
        comparators_by_metric=comparators_by_metric,
    )
    if cache_status != "available":
        return {}, audit

    pairwise: dict[str, Any] = {}
    # Preserve only the comparisons needed for the override decision.  A
    # comparator can be strongest for more than one metric; each such metric is
    # retained beneath the same comparator key.
    selected: dict[str, set[str]] = {"dummy": set(metrics)}
    for metric, comparator in strongest.items():
        if comparator is not None:
            selected.setdefault(comparator, set()).add(metric)
    for comparator, selected_metrics in selected.items():
        if comparator not in prediction_arrays:
            continue
        try:
            summary = _summarize_cached_probe_pair(
                cache,
                best_name,
                comparator,
                point_scores=point_scores,
            )
        except Exception:
            summary = None
        if not isinstance(summary, Mapping):
            continue
        metric_payload = summary.get("metrics", {})
        if not isinstance(metric_payload, Mapping):
            continue
        if target_mode == "singlelabel":
            metric = "balanced_accuracy"
            item = metric_payload.get(metric)
            if metric not in selected_metrics or not isinstance(item, Mapping):
                continue
            retained = dict(item)
            for key in ("paired_standard_error", "resamples_requested", "resamples_used"):
                retained.pop(key, None)
            retained["clear_advantage"] = bool(
                float(retained.get("point_delta", 0.0))
                >= config.mlp_min_improvement
                and float(retained.get("lower_95", 0.0)) > 0.0
            )
            pairwise[comparator] = retained
            continue
        retained_metrics: dict[str, Any] = {}
        for metric in metrics:
            if metric not in selected_metrics:
                continue
            item = metric_payload.get(metric)
            if not isinstance(item, Mapping):
                continue
            retained = dict(item)
            for key in ("paired_standard_error", "resamples_requested", "resamples_used"):
                retained.pop(key, None)
            retained["clear_advantage"] = bool(
                float(retained.get("point_delta", 0.0))
                >= config.mlp_min_improvement
                and float(retained.get("lower_95", 0.0)) > 0.0
            )
            retained_metrics[metric] = retained
        if retained_metrics:
            pairwise[comparator] = retained_metrics
    if not pairwise:
        audit["status"] = "unavailable"
        audit["reason"] = "No retained paired MLP comparison summaries were available."
    return pairwise, audit


def _singlelabel_override_evidence(
    mlp_results: dict[str, dict[str, Any]],
    comparator_results: dict[str, dict[str, Any]],
    *,
    y_true: np.ndarray,
    config: ProfilerConfig,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    """Return single-label MLP recommendation evidence."""
    metric_names = ("balanced_accuracy",)
    required_metrics = 1
    missing_comparators = _missing_or_failed_comparators(
        comparator_results,
        metrics=metric_names,
        n_rows=y_true.shape[0],
    )
    required_complete = not missing_comparators
    strongest = (
        _strongest_simpler_by_metric(comparator_results, metrics=metric_names)
        if required_complete
        else {}
    )
    best_name, best_result = _select_best_architecture(
        mlp_results,
        metrics=metric_names,
        n_rows=y_true.shape[0],
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": (
                "No MLP architecture produced complete aligned held-out evidence."
            ),
            "pairwise_comparisons": {},
            "pairwise_comparison_audit": _pairwise_comparison_audit(
                config,
                status="not_run",
                reason="No MLP architecture produced complete aligned held-out evidence.",
            ),
            "best_architecture": None,
            "required_comparators_complete": required_complete,
            "architectures_complete": False,
            **_override_report_fields(
                config=config,
                strongest=strongest,
                beating_strongest=[],
                beating_dummy=[],
                clearing=[],
                required_metrics=required_metrics,
                missing_comparators=missing_comparators,
                absolute_skill={},
            ),
        }
    pairwise, pairwise_audit = _mlp_pairwise_cached_comparisons(
        best_name=best_name,
        best_result=best_result,
        comparator_results=comparator_results,
        y_true=y_true,
        target_mode="singlelabel",
        metrics=metric_names,
        config=config,
        groups=groups,
        names=None,
        strongest=strongest,
    )
    beating_strongest = [
        metric
        for metric, comparator in strongest.items()
        if pairwise.get(comparator, {}).get("clear_advantage")
    ]
    beating_dummy = (
        ["balanced_accuracy"]
        if pairwise.get("dummy", {}).get("clear_advantage")
        else []
    )
    clearing = sorted(set(beating_strongest) & set(beating_dummy))
    absolute_skill = _absolute_skill_by_metric(
        best_result,
        comparator_results,
        metrics=metric_names,
    )
    override = bool(required_complete and len(clearing) >= required_metrics)
    override_reason = (
        "The best MLP architecture showed paired signal above dummy and clearly "
        "improved over the strongest aligned simpler probe."
        if override
        else _failed_override_reason(
            missing_comparators=missing_comparators,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
        )
    )
    override_reason = _pairwise_unavailable_override_reason(
        required_complete=required_complete,
        audit=pairwise_audit,
        fallback=override_reason,
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": override_reason,
        "pairwise_comparisons": pairwise,
        "pairwise_comparison_audit": pairwise_audit,
        "best_architecture": {
            "probe_name": best_name,
            "balanced_accuracy": float(best_result["balanced_accuracy"]),
            "probe_recipe_id": (
                best_result.get("probe_recipe", {}).get("recipe_id")
                if isinstance(best_result.get("probe_recipe"), dict)
                else None
            ),
        },
        "required_comparators_complete": required_complete,
        "architectures_complete": True,
        "absolute_skill": absolute_skill["balanced_accuracy"],
        **_override_report_fields(
            config=config,
            strongest=strongest,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
            missing_comparators=missing_comparators,
            absolute_skill=absolute_skill,
        ),
    }


def _multilabel_override_evidence(
    mlp_results: dict[str, dict[str, Any]],
    comparator_results: dict[str, dict[str, Any]],
    *,
    Y_true: np.ndarray,
    label_names: np.ndarray,
    config: ProfilerConfig,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    """Return multilabel MLP recommendation evidence."""
    metric_names = ("micro_f1", "macro_f1", "sample_jaccard")
    required_metrics = 2
    missing_comparators = _missing_or_failed_comparators(
        comparator_results,
        metrics=metric_names,
        n_rows=Y_true.shape[0],
    )
    required_complete = not missing_comparators
    strongest = (
        _strongest_simpler_by_metric(comparator_results, metrics=metric_names)
        if required_complete
        else {}
    )
    best_name, best_result = _select_best_architecture(
        mlp_results,
        metrics=metric_names,
        n_rows=Y_true.shape[0],
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": (
                "No MLP architecture produced complete aligned held-out evidence."
            ),
            "pairwise_comparisons": {},
            "pairwise_comparison_audit": _pairwise_comparison_audit(
                config,
                status="not_run",
                reason="No MLP architecture produced complete aligned held-out evidence.",
            ),
            "best_architecture": None,
            "required_comparators_complete": required_complete,
            "architectures_complete": False,
            **_override_report_fields(
                config=config,
                strongest=strongest,
                beating_strongest=[],
                beating_dummy=[],
                clearing=[],
                required_metrics=required_metrics,
                missing_comparators=missing_comparators,
                absolute_skill={},
            ),
        }
    pairwise, pairwise_audit = _mlp_pairwise_cached_comparisons(
        best_name=best_name,
        best_result=best_result,
        comparator_results=comparator_results,
        y_true=Y_true,
        target_mode="multilabel",
        metrics=metric_names,
        config=config,
        groups=groups,
        names=label_names,
        strongest=strongest,
    )
    beating_strongest = [
        metric
        for metric, comparator in strongest.items()
        if pairwise.get(comparator, {}).get(metric, {}).get("clear_advantage")
    ]
    beating_dummy = [
        metric
        for metric in metric_names
        if pairwise.get("dummy", {}).get(metric, {}).get("clear_advantage")
    ]
    clearing = [
        metric
        for metric in metric_names
        if metric in beating_strongest and metric in beating_dummy
    ]
    absolute_skill = _absolute_skill_by_metric(
        best_result,
        comparator_results,
        metrics=metric_names,
    )
    override = bool(required_complete and len(clearing) >= required_metrics)
    override_reason = (
        "The best MLP architecture showed paired signal above dummy and clearly "
        "improved over the strongest aligned simpler probe on at least two "
        "primary multilabel metrics."
        if override
        else _failed_override_reason(
            missing_comparators=missing_comparators,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
        )
    )
    override_reason = _pairwise_unavailable_override_reason(
        required_complete=required_complete,
        audit=pairwise_audit,
        fallback=override_reason,
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": override_reason,
        "pairwise_comparisons": pairwise,
        "pairwise_comparison_audit": pairwise_audit,
        "best_architecture": {
            "probe_name": best_name,
            "micro_f1": float(best_result["micro_f1"]),
            "macro_f1": float(best_result["macro_f1"]),
            "sample_jaccard": float(best_result["sample_jaccard"]),
            "probe_recipe_id": (
                best_result.get("probe_recipe", {}).get("recipe_id")
                if isinstance(best_result.get("probe_recipe"), dict)
                else None
            ),
        },
        "required_comparators_complete": required_complete,
        "architectures_complete": True,
        **_override_report_fields(
            config=config,
            strongest=strongest,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
            missing_comparators=missing_comparators,
            absolute_skill=absolute_skill,
        ),
    }


def _regression_override_evidence(
    mlp_results: dict[str, dict[str, Any]],
    comparator_results: dict[str, dict[str, Any]],
    *,
    Y_true: np.ndarray,
    target_names: np.ndarray,
    config: ProfilerConfig,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    """Return regression MLP recommendation evidence."""
    metric_names = ("r2_variance_weighted", "r2_uniform_average")
    required_metrics = len(metric_names)
    missing_comparators = _missing_or_failed_comparators(
        comparator_results,
        metrics=metric_names,
        n_rows=Y_true.shape[0],
    )
    required_complete = not missing_comparators
    strongest = (
        _strongest_simpler_by_metric(comparator_results, metrics=metric_names)
        if required_complete
        else {}
    )
    best_name, best_result = _select_best_architecture(
        mlp_results,
        metrics=metric_names,
        n_rows=Y_true.shape[0],
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": (
                "No MLP architecture produced complete aligned held-out evidence."
            ),
            "pairwise_comparisons": {},
            "pairwise_comparison_audit": _pairwise_comparison_audit(
                config,
                status="not_run",
                reason="No MLP architecture produced complete aligned held-out evidence.",
            ),
            "best_architecture": None,
            "required_comparators_complete": required_complete,
            "architectures_complete": False,
            **_override_report_fields(
                config=config,
                strongest=strongest,
                beating_strongest=[],
                beating_dummy=[],
                clearing=[],
                required_metrics=required_metrics,
                missing_comparators=missing_comparators,
                absolute_skill={},
            ),
        }
    pairwise, pairwise_audit = _mlp_pairwise_cached_comparisons(
        best_name=best_name,
        best_result=best_result,
        comparator_results=comparator_results,
        y_true=Y_true,
        target_mode="regression",
        metrics=metric_names,
        config=config,
        groups=groups,
        names=target_names,
        strongest=strongest,
    )
    beating_strongest = [
        metric
        for metric, comparator in strongest.items()
        if pairwise.get(comparator, {}).get(metric, {}).get("clear_advantage")
    ]
    beating_dummy = [
        metric
        for metric in metric_names
        if pairwise.get("dummy", {}).get(metric, {}).get("clear_advantage")
    ]
    clearing = [
        metric
        for metric in metric_names
        if metric in beating_strongest and metric in beating_dummy
    ]
    absolute_skill = _absolute_skill_by_metric(
        best_result,
        comparator_results,
        metrics=metric_names,
    )
    override = bool(required_complete and len(clearing) >= required_metrics)
    override_reason = (
        "The best MLP architecture showed paired signal above dummy and clearly "
        "improved over the strongest aligned simpler regressor on both primary "
        "R2 metrics."
        if override
        else _failed_override_reason(
            missing_comparators=missing_comparators,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
        )
    )
    override_reason = _pairwise_unavailable_override_reason(
        required_complete=required_complete,
        audit=pairwise_audit,
        fallback=override_reason,
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": override_reason,
        "pairwise_comparisons": pairwise,
        "pairwise_comparison_audit": pairwise_audit,
        "best_architecture": {
            "probe_name": best_name,
            "r2_variance_weighted": float(best_result["r2_variance_weighted"]),
            "r2_uniform_average": float(best_result["r2_uniform_average"]),
            "probe_recipe_id": (
                best_result.get("probe_recipe", {}).get("recipe_id")
                if isinstance(best_result.get("probe_recipe"), dict)
                else None
            ),
        },
        "required_comparators_complete": required_complete,
        "architectures_complete": True,
        **_override_report_fields(
            config=config,
            strongest=strongest,
            beating_strongest=beating_strongest,
            beating_dummy=beating_dummy,
            clearing=clearing,
            required_metrics=required_metrics,
            missing_comparators=missing_comparators,
            absolute_skill=absolute_skill,
        ),
    }


def _append_runtime_warnings(
    *,
    input_dim: int,
    output_dim: int,
    architectures: list[dict[str, Any]],
    config: ProfilerConfig,
    report_context: dict[str, Any],
    epochs: int,
    folds: int,
) -> None:
    """Record runtime warnings before training MLP probes."""
    warnings_list = report_context.setdefault("warnings", [])
    max_parameters = _mlp_budget(config)["max_parameters"]
    if input_dim >= _MLP_RUNTIME_DIM_WARNING:
        record_warning(
            "High input dimensionality may make optional MLP probes slow even after sample capping.",
            warnings_list,
            UserWarning,
        )
    for architecture in architectures:
        if architecture["parameter_count"] >= 0.5 * max_parameters:
            record_warning(
                "An optional MLP candidate used at least half of the configured parameter budget.",
                warnings_list,
                UserWarning,
            )
    if architectures:
        estimated_work = (
            6.0
            * max(architecture["parameter_count"] for architecture in architectures)
            * epochs
            * folds
        )
        if estimated_work >= _MLP_RUNTIME_WORK_WARNING:
            record_warning(
                "Optional MLP probe work estimate is very large; results may take noticeably longer than simpler probes.",
                warnings_list,
                UserWarning,
            )


def maybe_run_singlelabel_mlp_probes(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    metrics: dict[str, Any],
    report_context: dict[str, Any],
    class_labels: np.ndarray | None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run conditional single-label MLP probes when simpler probes are not good enough."""
    if not config.mlp_probes:
        payload = _default_mlp_artifacts(config)
        payload["trigger"] = {
            "status": "not_requested",
            "reason": "MLP probes were disabled.",
            "good_enough": None,
            "threshold": config.mlp_trigger_skill_threshold,
        }
        return payload
    trigger = _simple_singlelabel_trigger(metrics, config)
    payload = _default_mlp_artifacts(config)
    payload["trigger"] = trigger
    if trigger["good_enough"]:
        payload["status"] = "not_triggered"
        payload["reason"] = trigger["reason"]
        return payload
    torch = _torch_module()
    if torch is None:
        payload["status"] = "dependency_unavailable"
        payload["reason"] = "MLP probes require the optional torch extra."
        payload["pairwise_comparison_audit"] = _pairwise_comparison_audit(
            config,
            status="unavailable",
            reason=payload["reason"],
        )
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    resolved_device, fallback_warning = _resolve_device(torch, config.mlp_device)
    payload["backend"] = {
        "requested_device": config.mlp_device,
        "resolved_device": resolved_device,
        "torch_available": True,
    }
    if fallback_warning is not None:
        record_warning(
            fallback_warning, report_context.setdefault("warnings", []), UserWarning
        )
    budget = _mlp_budget(config)
    sample_config = replace(
        config,
        max_samples=min(
            budget["max_samples"],
            config.max_samples
            if config.max_samples is not None
            else budget["max_samples"],
        ),
    )
    X_used, y_used, sample_info = cap_samples_for_budget(
        X,
        y,
        config=sample_config,
        reason="probe",
        groups=groups,
    )
    if sample_info.get("support_preserved") is False:
        payload["status"] = "skipped"
        payload["reason"] = sample_info.get("skip_reason")
        _set_pairwise_audit(
            payload,
            config,
            status="not_run",
            reason=str(payload["reason"] or "MLP cohort support was not preserved."),
        )
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    dense_info = ensure_dense_or_sample(
        X_used,
        y_used,
        reason="mlp_probe",
        config=config,
        report_context=report_context,
        groups=None
        if groups is None
        else groups[np.asarray(sample_info["indices"], dtype=int)],
    )
    if dense_info["skipped"]:
        payload["status"] = "skipped"
        payload["reason"] = "Dense conversion unavailable under the current policy."
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    dense_X = np.asarray(dense_info["X"], dtype=np.float32)
    dense_y = np.asarray(dense_info["y"], dtype=int)
    dense_groups = dense_info.get("groups")
    architectures = _architecture_candidates(
        dense_X.shape[1],
        int(np.max(dense_y)) + 1,
        max_parameters=budget["max_parameters"],
    )
    if not architectures:
        payload["status"] = "skipped"
        payload["reason"] = (
            "No MLP architecture fit within the configured parameter budget."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    splits, evaluation_mode = _split_rows(
        dense_X,
        dense_y,
        target_mode="singlelabel",
        config=config,
        max_folds=budget["cv_folds"],
        groups=dense_groups,
    )
    if splits is None:
        payload["status"] = "skipped"
        payload["reason"] = (
            "MLP override requires a valid held-out split; no such split was available."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "mlp_held_out_evaluation",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    _append_runtime_warnings(
        input_dim=dense_X.shape[1],
        output_dim=int(np.max(dense_y)) + 1,
        architectures=architectures,
        config=config,
        report_context=report_context,
        epochs=budget["epochs"],
        folds=budget["cv_folds"],
    )
    comparators: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior"),
        "linear": _linear_classifier(dense_X),
        "knn": _scaled_pipeline(
            dense_X,
            KNeighborsClassifier(
                n_neighbors=min(
                    max(1, dense_y.shape[0] - 1),
                    min(15, max(3, int(np.sqrt(dense_y.shape[0])))),
                )
            ),
            name="knn",
        ),
    }
    comparator_variants: dict[str, str | None] = {}
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _full_quadratic_classifier(config.random_state)
        comparator_variants["smooth_poly"] = "full_quadratic"
    else:
        sketch = _choose_sketch_components(
            dense_X.shape[0],
            dense_X.shape[1],
            dense_X.dtype,
            max_dense_mb=config.max_dense_mb,
        )
        if sketch is not None:
            comparators["smooth_poly"] = _low_rank_quadratic_classifier(
                sketch,
                config.random_state,
            )
            comparator_variants["smooth_poly"] = "low_rank_quadratic"
    comparators["kernel_approx"] = Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "rff",
                RBFSampler(
                    gamma=1.0 / max(1, dense_X.shape[1]),
                    n_components=min(256, max(32, dense_X.shape[1] * 2)),
                    random_state=config.random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    random_state=config.random_state,
                    max_iter=3000,
                    tol=1e-4,
                ),
            ),
        ]
    )
    errors = report_context.setdefault("errors", [])
    comparator_results = _safe_evaluate_models(
        _evaluate_singlelabel_models,
        comparators,
        errors=errors,
        X=dense_X,
        y=dense_y,
        class_labels=class_labels,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    mlp_estimators = {
        f"mlp_{item['label']}": TorchMLPClassifier(
            task="singlelabel",
            hidden_layer_sizes=item["hidden_layer_sizes"],
            epochs=budget["epochs"],
            patience=budget["patience"],
            batch_size=min(256, dense_X.shape[0]),
            device=resolved_device,
            random_state=config.random_state,
            multilabel_stratification=config.multilabel_stratification,
        )
        for item in architectures
    }
    mlp_results = _safe_evaluate_models(
        _evaluate_singlelabel_models,
        mlp_estimators,
        errors=errors,
        X=dense_X,
        y=dense_y,
        class_labels=class_labels,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    comparator_results = _attach_aligned_comparator_recipes(
        comparator_results,
        comparators,
        target_mode="singlelabel",
        X=dense_X,
        n_outputs=1,
        sample_info=sample_info,
        config=config,
        variants=comparator_variants,
    )
    mlp_results = _attach_architecture_recipes(
        mlp_results,
        mlp_estimators,
        target_mode="singlelabel",
        X=dense_X,
        n_outputs=1,
        sample_info=sample_info,
        config=config,
    )
    recommendation = _singlelabel_override_evidence(
        mlp_results,
        comparator_results,
        y_true=dense_y,
        config=config,
        groups=dense_groups,
    )
    if any(result.get("status") == "runtime_failed" for result in mlp_results.values()):
        recommendation["recommendation_override"] = False
        recommendation["architectures_complete"] = False
        recommendation["override_reason"] = (
            "At least one requested MLP architecture failed, so override evidence "
            "was incomplete."
        )
    payload.update(
        {
            "status": "completed",
            "reason": recommendation["override_reason"],
            "sample_info": sample_info,
            "architectures": [
                {
                    **item,
                    **mlp_results.get(
                        f"mlp_{item['label']}",
                        _mark_mlp_recipe_unavailable(
                            {}, "estimator was not constructed"
                        ),
                    ),
                }
                for item in architectures
            ],
            "aligned_comparators": comparator_results,
            **recommendation,
        }
    )
    return payload


def maybe_run_multilabel_mlp_probes(
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    metrics: dict[str, Any],
    report_context: dict[str, Any],
    label_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run conditional multilabel MLP probes when simpler probes are not good enough."""
    if not config.mlp_probes:
        payload = _default_mlp_artifacts(config)
        payload["trigger"] = {
            "status": "not_requested",
            "reason": "MLP probes were disabled.",
            "good_enough": None,
            "threshold": config.mlp_trigger_skill_threshold,
        }
        return payload
    trigger = _simple_multilabel_trigger(metrics, config)
    payload = _default_mlp_artifacts(config)
    payload["trigger"] = trigger
    if trigger["good_enough"]:
        payload["status"] = "not_triggered"
        payload["reason"] = trigger["reason"]
        return payload
    torch = _torch_module()
    if torch is None:
        payload["status"] = "dependency_unavailable"
        payload["reason"] = "MLP probes require the optional torch extra."
        payload["pairwise_comparison_audit"] = _pairwise_comparison_audit(
            config,
            status="unavailable",
            reason=payload["reason"],
        )
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "multilabel_mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    resolved_device, fallback_warning = _resolve_device(torch, config.mlp_device)
    payload["backend"] = {
        "requested_device": config.mlp_device,
        "resolved_device": resolved_device,
        "torch_available": True,
    }
    if fallback_warning is not None:
        record_warning(
            fallback_warning, report_context.setdefault("warnings", []), UserWarning
        )
    budget = _mlp_budget(config)
    sample_config = replace(
        config,
        max_samples=min(
            budget["max_samples"],
            config.max_samples
            if config.max_samples is not None
            else budget["max_samples"],
        ),
    )
    X_used, Y_used, sample_info = cap_multilabel_samples_for_budget(
        X,
        Y,
        config=sample_config,
        reason="probe",
        groups=groups,
    )
    if sample_info.get("support_preserved") is False:
        payload["status"] = "skipped"
        payload["reason"] = sample_info.get("skip_reason")
        _set_pairwise_audit(
            payload,
            config,
            status="not_run",
            reason=str(payload["reason"] or "MLP cohort support was not preserved."),
        )
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "multilabel_mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    groups_used = (
        None
        if groups is None
        else groups[np.asarray(sample_info["indices"], dtype=int)]
    )
    target_info = ensure_dense_multilabel_target(
        X_used,
        Y_used,
        reason="multilabel_mlp_target",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if target_info["skipped"]:
        payload["status"] = "skipped"
        payload["reason"] = "Multilabel target exceeds the dense-memory budget."
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    X_used = target_info["X"]
    Y_used = target_info["Y"]
    groups_used = target_info.get("groups")
    dense_info = _ensure_dense_X_for_multilabel(
        X_used,
        Y_used,
        reason="mlp_probe",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if dense_info["skipped"]:
        payload["status"] = "skipped"
        payload["reason"] = "Dense conversion unavailable under the current policy."
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    dense_X = np.asarray(dense_info["X"], dtype=np.float32)
    dense_Y = _dense_multilabel_matrix(dense_info["Y"])
    dense_groups = dense_info.get("groups")
    architectures = _architecture_candidates(
        dense_X.shape[1],
        dense_Y.shape[1],
        max_parameters=budget["max_parameters"],
    )
    if not architectures:
        payload["status"] = "skipped"
        payload["reason"] = (
            "No MLP architecture fit within the configured parameter budget."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    splits, evaluation_mode = _split_rows(
        dense_X,
        dense_Y,
        target_mode="multilabel",
        config=config,
        max_folds=budget["cv_folds"],
        groups=dense_groups,
    )
    if splits is None:
        payload["status"] = "skipped"
        payload["reason"] = (
            "MLP override requires a valid held-out split; no such split was available."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "multilabel_mlp_held_out_evaluation",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    _append_runtime_warnings(
        input_dim=dense_X.shape[1],
        output_dim=dense_Y.shape[1],
        architectures=architectures,
        config=config,
        report_context=report_context,
        epochs=budget["epochs"],
        folds=budget["cv_folds"],
    )
    comparators: dict[str, Any] = {
        "dummy": MultilabelPriorDummy(threshold=0.5),
        "linear": _multilabel_linear_classifier(dense_X, config),
        "knn": _scaled_pipeline(
            dense_X,
            KNeighborsClassifier(
                n_neighbors=min(
                    max(1, dense_Y.shape[0] - 1),
                    min(15, max(3, int(np.sqrt(dense_Y.shape[0])))),
                )
            ),
            name="knn",
        ),
    }
    comparator_variants: dict[str, str | None] = {}
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _full_multilabel_quadratic_classifier(config)
        comparator_variants["smooth_poly"] = "full_quadratic"
    else:
        sketch = _choose_sketch_components(
            dense_X.shape[0],
            dense_X.shape[1],
            dense_X.dtype,
            max_dense_mb=config.max_dense_mb,
        )
        if sketch is not None:
            comparators["smooth_poly"] = _low_rank_multilabel_quadratic_classifier(
                sketch,
                config,
            )
            comparator_variants["smooth_poly"] = "low_rank_quadratic"
    comparators["kernel_approx"] = Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "rff",
                RBFSampler(
                    gamma=1.0 / max(1, dense_X.shape[1]),
                    n_components=min(256, max(32, dense_X.shape[1] * 2)),
                    random_state=config.random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            (
                "clf",
                OneVsRestClassifier(
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        random_state=config.random_state,
                        max_iter=3000,
                        tol=1e-4,
                    ),
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )
    errors = report_context.setdefault("errors", [])
    comparator_results = _safe_evaluate_models(
        _evaluate_multilabel_models,
        comparators,
        errors=errors,
        X=dense_X,
        Y=dense_Y,
        label_names=label_names,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    mlp_estimators = {
        f"mlp_{item['label']}": TorchMLPClassifier(
            task="multilabel",
            hidden_layer_sizes=item["hidden_layer_sizes"],
            epochs=budget["epochs"],
            patience=budget["patience"],
            batch_size=min(256, dense_X.shape[0]),
            device=resolved_device,
            random_state=config.random_state,
            multilabel_stratification=config.multilabel_stratification,
        )
        for item in architectures
    }
    mlp_results = _safe_evaluate_models(
        _evaluate_multilabel_models,
        mlp_estimators,
        errors=errors,
        X=dense_X,
        Y=dense_Y,
        label_names=label_names,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    comparator_results = _attach_aligned_comparator_recipes(
        comparator_results,
        comparators,
        target_mode="multilabel",
        X=dense_X,
        n_outputs=dense_Y.shape[1],
        sample_info=sample_info,
        config=config,
        variants=comparator_variants,
    )
    mlp_results = _attach_architecture_recipes(
        mlp_results,
        mlp_estimators,
        target_mode="multilabel",
        X=dense_X,
        n_outputs=dense_Y.shape[1],
        sample_info=sample_info,
        config=config,
    )
    recommendation = _multilabel_override_evidence(
        mlp_results,
        comparator_results,
        Y_true=dense_Y,
        label_names=label_names,
        config=config,
        groups=dense_groups,
    )
    if any(result.get("status") == "runtime_failed" for result in mlp_results.values()):
        recommendation["recommendation_override"] = False
        recommendation["architectures_complete"] = False
        recommendation["override_reason"] = (
            "At least one requested MLP architecture failed, so override evidence "
            "was incomplete."
        )
    payload.update(
        {
            "status": "completed",
            "reason": recommendation["override_reason"],
            "sample_info": sample_info,
            "architectures": [
                {
                    **item,
                    **mlp_results.get(
                        f"mlp_{item['label']}",
                        _mark_mlp_recipe_unavailable(
                            {}, "estimator was not constructed"
                        ),
                    ),
                }
                for item in architectures
            ],
            "aligned_comparators": comparator_results,
            **recommendation,
        }
    )
    return payload


def maybe_run_regression_mlp_probes(
    X: Any,
    Y: np.ndarray,
    *,
    config: ProfilerConfig,
    metrics: dict[str, Any],
    report_context: dict[str, Any],
    target_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run conditional regression MLP probes when simpler probes are not good enough."""
    if not config.mlp_probes:
        payload = _default_mlp_artifacts(config)
        payload["trigger"] = {
            "status": "not_requested",
            "reason": "MLP probes were disabled.",
            "good_enough": None,
            "threshold": config.mlp_trigger_skill_threshold,
        }
        return payload
    trigger = _simple_regression_trigger(metrics, config)
    payload = _default_mlp_artifacts(config)
    payload["trigger"] = trigger
    if trigger["good_enough"]:
        payload["status"] = "not_triggered"
        payload["reason"] = trigger["reason"]
        return payload
    torch = _torch_module()
    if torch is None:
        payload["status"] = "dependency_unavailable"
        payload["reason"] = "MLP probes require the optional torch extra."
        payload["pairwise_comparison_audit"] = _pairwise_comparison_audit(
            config,
            status="unavailable",
            reason=payload["reason"],
        )
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "regression_mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    resolved_device, fallback_warning = _resolve_device(torch, config.mlp_device)
    payload["backend"] = {
        "requested_device": config.mlp_device,
        "resolved_device": resolved_device,
        "torch_available": True,
    }
    if fallback_warning is not None:
        record_warning(
            fallback_warning, report_context.setdefault("warnings", []), UserWarning
        )
    budget = _mlp_budget(config)
    sample_config = replace(
        config,
        max_samples=min(
            budget["max_samples"],
            config.max_samples
            if config.max_samples is not None
            else budget["max_samples"],
        ),
    )
    X_used, Y_used, sample_info = cap_regression_samples_for_budget(
        X,
        Y,
        config=sample_config,
        reason="probe",
        groups=groups,
    )
    if sample_info.get("support_preserved") is False:
        payload["status"] = "skipped"
        payload["reason"] = sample_info.get("skip_reason")
        _set_pairwise_audit(
            payload,
            config,
            status="not_run",
            reason=str(payload["reason"] or "MLP cohort support was not preserved."),
        )
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "regression_mlp_probes",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    groups_used = (
        None
        if groups is None
        else groups[np.asarray(sample_info["indices"], dtype=int)]
    )
    dense_info = ensure_dense_or_sample_regression(
        X_used,
        Y_used,
        reason="mlp_probe",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if dense_info["skipped"]:
        payload["status"] = "skipped"
        payload["reason"] = "Dense conversion unavailable under the current policy."
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    dense_X = np.asarray(dense_info["X"], dtype=np.float32)
    dense_Y = np.asarray(dense_info["y"], dtype=float)
    if dense_Y.ndim == 1:
        dense_Y = dense_Y.reshape(-1, 1)
    dense_groups = dense_info.get("groups")
    architectures = _architecture_candidates(
        dense_X.shape[1],
        dense_Y.shape[1],
        max_parameters=budget["max_parameters"],
    )
    if not architectures:
        payload["status"] = "skipped"
        payload["reason"] = (
            "No MLP architecture fit within the configured parameter budget."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        return payload
    splits, evaluation_mode = _split_rows(
        dense_X,
        dense_Y,
        target_mode="regression",
        config=config,
        max_folds=budget["cv_folds"],
        groups=dense_groups,
    )
    if splits is None:
        payload["status"] = "skipped"
        payload["reason"] = (
            "MLP override requires a valid held-out split; no such split was available."
        )
        _set_pairwise_audit(payload, config, status="not_run", reason=payload["reason"])
        payload["sample_info"] = sample_info
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "regression_mlp_held_out_evaluation",
                "reason": payload["reason"],
                "severity": "caution",
            }
        )
        return payload
    _append_runtime_warnings(
        input_dim=dense_X.shape[1],
        output_dim=dense_Y.shape[1],
        architectures=architectures,
        config=config,
        report_context=report_context,
        epochs=budget["epochs"],
        folds=budget["cv_folds"],
    )
    comparators: dict[str, Any] = {
        "dummy": TargetMeanDummyRegressor(),
        "linear": _scaled_pipeline(
            dense_X,
            Ridge(alpha=1.0, random_state=config.random_state),
            name="reg",
        ),
        "knn": _scaled_pipeline(
            dense_X,
            KNeighborsRegressor(
                n_neighbors=min(
                    max(1, dense_Y.shape[0] - 1),
                    min(15, max(3, int(np.sqrt(dense_Y.shape[0])))),
                )
            ),
            name="knn",
        ),
    }
    comparator_variants: dict[str, str | None] = {}
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _regression_smooth_estimator(config)
        comparator_variants["smooth_poly"] = "full_quadratic"
    else:
        sketch = _choose_sketch_components(
            dense_X.shape[0],
            dense_X.shape[1],
            dense_X.dtype,
            max_dense_mb=config.max_dense_mb,
        )
        if sketch is not None:
            comparators["smooth_poly"] = _regression_smooth_estimator(
                config,
                low_rank_components=sketch,
            )
            comparator_variants["smooth_poly"] = "low_rank_quadratic"
    comparators["kernel_approx"] = Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "rff",
                RBFSampler(
                    gamma=1.0 / max(1, dense_X.shape[1]),
                    n_components=min(256, max(32, dense_X.shape[1] * 2)),
                    random_state=config.random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=config.random_state)),
        ]
    )
    errors = report_context.setdefault("errors", [])
    comparator_results = _safe_evaluate_models(
        _evaluate_regression_models,
        comparators,
        errors=errors,
        X=dense_X,
        Y=dense_Y,
        target_names=target_names,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    mlp_estimators = {
        f"mlp_{item['label']}": TorchMLPRegressor(
            task="regression",
            hidden_layer_sizes=item["hidden_layer_sizes"],
            epochs=budget["epochs"],
            patience=budget["patience"],
            batch_size=min(256, dense_X.shape[0]),
            device=resolved_device,
            random_state=config.random_state,
            multilabel_stratification=config.multilabel_stratification,
        )
        for item in architectures
    }
    mlp_results = _safe_evaluate_models(
        _evaluate_regression_models,
        mlp_estimators,
        errors=errors,
        X=dense_X,
        Y=dense_Y,
        target_names=target_names,
        splits=splits,
        evaluation_mode=evaluation_mode,
        groups=dense_groups,
    )
    comparator_results = _attach_aligned_comparator_recipes(
        comparator_results,
        comparators,
        target_mode="regression",
        X=dense_X,
        n_outputs=dense_Y.shape[1],
        sample_info=sample_info,
        config=config,
        variants=comparator_variants,
    )
    mlp_results = _attach_architecture_recipes(
        mlp_results,
        mlp_estimators,
        target_mode="regression",
        X=dense_X,
        n_outputs=dense_Y.shape[1],
        sample_info=sample_info,
        config=config,
    )
    recommendation = _regression_override_evidence(
        mlp_results,
        comparator_results,
        Y_true=dense_Y,
        target_names=target_names,
        config=config,
        groups=dense_groups,
    )
    if any(result.get("status") == "runtime_failed" for result in mlp_results.values()):
        recommendation["recommendation_override"] = False
        recommendation["architectures_complete"] = False
        recommendation["override_reason"] = (
            "At least one requested MLP architecture failed, so override evidence "
            "was incomplete."
        )
    payload.update(
        {
            "status": "completed",
            "reason": recommendation["override_reason"],
            "sample_info": sample_info,
            "architectures": [
                {
                    **item,
                    **mlp_results.get(
                        f"mlp_{item['label']}",
                        _mark_mlp_recipe_unavailable(
                            {}, "estimator was not constructed"
                        ),
                    ),
                }
                for item in architectures
            ],
            "aligned_comparators": comparator_results,
            **recommendation,
        }
    )
    return payload
