"""Optional conditional feed-forward MLP probes."""

# ruff: noqa: E501

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, Literal, cast

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
    summarize_multilabel_predictions,
    summarize_predictions,
    summarize_regression_predictions,
)
from separatix.sampling import (
    cap_multilabel_samples_for_budget,
    cap_regression_samples_for_budget,
    cap_samples_for_budget,
    choose_multilabel_cv,
    choose_multilabel_holdout,
)
from separatix.utils.random import make_rng
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
        "required_comparators_complete": False,
        "recommendation_override": False,
        "override_reason": None,
    }


def _mlp_budget(config: ProfilerConfig) -> dict[str, int]:
    """Return the per-budget MLP limits."""
    budget = dict(_MLP_BUDGETS[config.budget])
    if config.mlp_max_parameters is not None:
        budget["max_parameters"] = int(config.mlp_max_parameters)
    return budget


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

    def _init_model(self, torch: Any, input_dim: int, output_dim: int) -> Any:
        """Initialize the torch module and apply explicit weight initialization."""
        modules: list[Any] = []
        layer_dims = [input_dim, *self.hidden_layer_sizes, output_dim]
        for index in range(len(layer_dims) - 1):
            linear = torch.nn.Linear(layer_dims[index], layer_dims[index + 1])
            if index < len(layer_dims) - 2:
                torch.nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            else:
                torch.nn.init.xavier_uniform_(linear.weight)
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
            model = self._init_model(torch, X_array.shape[1], output_dim).to(
                self.device
            )
        batch_generator = torch.Generator(device="cpu")
        batch_generator.manual_seed(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )

        def schedule(epoch: int) -> float:
            if epoch < 5:
                return float(epoch + 1) / 5.0
            progress = (epoch - 5) / max(1, self.epochs - 5)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        if self.task == "singlelabel":
            counts = np.bincount(y_fit.astype(np.int64))
            weights = np.sum(counts) / np.maximum(counts.shape[0] * counts, 1)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
            )
        elif self.task == "multilabel":
            positive = np.sum(y_fit, axis=0)
            negative = y_fit.shape[0] - positive
            pos_weight = np.clip(negative / np.maximum(positive, 1.0), 0.05, 20.0)
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            if valid_loss + 1e-6 < best_loss:
                best_loss = valid_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
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


def _bootstrap_indices(
    n_rows: int,
    *,
    repeats: int,
    random_state: int | None,
    groups: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return bootstrap index sets, preserving groups when requested."""
    rng = make_rng(random_state)
    if groups is None:
        return [
            np.sort(rng.choice(np.arange(n_rows), size=n_rows, replace=True)).astype(
                int
            )
            for _ in range(repeats)
        ]
    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group_id) for group_id in unique_groups]
    samples: list[np.ndarray] = []
    for _ in range(repeats):
        chosen = rng.choice(
            np.arange(unique_groups.shape[0]), size=unique_groups.shape[0], replace=True
        )
        sampled = np.concatenate([group_rows[int(index)] for index in chosen]).astype(
            int
        )
        samples.append(np.sort(sampled))
    return samples


def _balanced_accuracy_delta(
    y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    sample_idx: np.ndarray,
) -> float:
    """Return a balanced-accuracy delta on a bootstrap sample."""
    first_score = cast(
        float,
        summarize_predictions(y_true[sample_idx], first_pred[sample_idx])[
            "balanced_accuracy"
        ],
    )
    second_score = cast(
        float,
        summarize_predictions(y_true[sample_idx], second_pred[sample_idx])[
            "balanced_accuracy"
        ],
    )
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
    first = summarize_multilabel_predictions(
        Y_true[sample_idx],
        first_pred[sample_idx],
        label_names=label_names,
    )
    second = summarize_multilabel_predictions(
        Y_true[sample_idx],
        second_pred[sample_idx],
        label_names=label_names,
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
    first = summarize_regression_predictions(
        Y_true[sample_idx],
        first_pred[sample_idx],
        target_names=target_names,
    )
    second = summarize_regression_predictions(
        Y_true[sample_idx],
        second_pred[sample_idx],
        target_names=target_names,
    )
    return float(first[metric] - second[metric])


def _bootstrap_comparison(
    delta_fn: Callable[[np.ndarray], float],
    *,
    repeats: int,
    random_state: int | None,
    n_rows: int,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """Return paired bootstrap delta summaries."""
    deltas = np.asarray(
        [
            delta_fn(sample_idx)
            for sample_idx in _bootstrap_indices(
                n_rows,
                repeats=repeats,
                random_state=random_state,
                groups=groups,
            )
        ],
        dtype=float,
    )
    if deltas.size == 0:
        return {"mean_delta": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    return {
        "mean_delta": float(np.mean(deltas)),
        "lower_95": float(np.percentile(deltas, 2.5)),
        "upper_95": float(np.percentile(deltas, 97.5)),
    }


def _objective_score(result: dict[str, Any], *, metrics: tuple[str, ...]) -> float:
    """Return a simple mean objective across primary metrics."""
    values = [float(result[name]) for name in metrics if name in result]
    return float(np.mean(values)) if values else float("-inf")


def _select_best_architecture(
    architecture_results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...],
) -> tuple[str | None, dict[str, Any] | None]:
    """Select the MLP architecture using conservative ties."""
    usable_results = {
        name: result
        for name, result in architecture_results.items()
        if result.get("status") != "runtime_failed"
        and result.get("predictions") is not None
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


def _singlelabel_override_evidence(
    mlp_results: dict[str, dict[str, Any]],
    comparator_results: dict[str, dict[str, Any]],
    *,
    y_true: np.ndarray,
    config: ProfilerConfig,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    """Return single-label MLP recommendation evidence."""
    best_name, best_result = _select_best_architecture(
        mlp_results,
        metrics=("balanced_accuracy",),
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": "No MLP architecture completed.",
            "pairwise_comparisons": {},
            "best_architecture": None,
            "required_comparators_complete": False,
        }
    pairwise: dict[str, Any] = {}
    required_complete = all(
        name in comparator_results for name in ("linear", "smooth_poly", "knn")
    )
    best_predictions = np.asarray(best_result["predictions"], dtype=int)
    for name, result in comparator_results.items():
        if "predictions" not in result:
            continue
        comparator_predictions = np.asarray(result["predictions"], dtype=int)

        def delta(
            sample_idx: np.ndarray, second: np.ndarray = comparator_predictions
        ) -> float:
            return _balanced_accuracy_delta(
                y_true,
                best_predictions,
                second,
                sample_idx,
            )

        comparison = _bootstrap_comparison(
            delta,
            repeats=_mlp_budget(config)["bootstrap_repeats"],
            random_state=config.random_state,
            n_rows=y_true.shape[0],
            groups=groups,
        )
        comparison["clear_advantage"] = bool(
            comparison["mean_delta"] >= config.mlp_min_improvement
            and comparison["lower_95"] > 0.0
        )
        pairwise[name] = comparison
    absolute_skill = _skill_from_bounds(
        float(best_result["balanced_accuracy"]),
        float(comparator_results["dummy"]["balanced_accuracy"]),
    )
    override = bool(
        absolute_skill is not None
        and absolute_skill >= config.mlp_trigger_skill_threshold
        and required_complete
        and pairwise
        and all(
            item["clear_advantage"] for item in pairwise.values() if item is not None
        )
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": (
            "The best MLP architecture clearly improved over every aligned simpler probe."
            if override
            else "No MLP architecture cleared the configured absolute-skill and pairwise-improvement thresholds."
        ),
        "pairwise_comparisons": pairwise,
        "best_architecture": {
            "probe_name": best_name,
            "balanced_accuracy": float(best_result["balanced_accuracy"]),
        },
        "required_comparators_complete": required_complete,
        "absolute_skill": absolute_skill,
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
    best_name, best_result = _select_best_architecture(
        mlp_results, metrics=metric_names
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": "No MLP architecture completed.",
            "pairwise_comparisons": {},
            "best_architecture": None,
            "required_comparators_complete": False,
        }
    pairwise: dict[str, Any] = {}
    required_complete = all(
        name in comparator_results for name in ("linear", "smooth_poly", "knn")
    )
    best_predictions = np.asarray(best_result["predictions"], dtype=np.int8)
    for name, result in comparator_results.items():
        if "predictions" not in result:
            continue
        comparator_predictions = np.asarray(result["predictions"], dtype=np.int8)
        metric_comparisons: dict[str, Any] = {}
        for metric in metric_names:

            def delta(
                sample_idx: np.ndarray,
                metric_name: str = metric,
                second: np.ndarray = comparator_predictions,
            ) -> float:
                return _multilabel_metric_delta(
                    Y_true,
                    best_predictions,
                    second,
                    sample_idx,
                    metric=metric_name,
                    label_names=label_names,
                )

            comparison = _bootstrap_comparison(
                delta,
                repeats=_mlp_budget(config)["bootstrap_repeats"],
                random_state=config.random_state,
                n_rows=Y_true.shape[0],
                groups=groups,
            )
            comparison["clear_advantage"] = bool(
                comparison["mean_delta"] >= config.mlp_min_improvement
                and comparison["lower_95"] > 0.0
            )
            metric_comparisons[metric] = comparison
        pairwise[name] = metric_comparisons
    dummy = comparator_results["dummy"]
    threshold_hits = 0
    for metric in metric_names:
        skill = _skill_from_bounds(float(best_result[metric]), float(dummy[metric]))
        if skill is not None and skill >= config.mlp_trigger_skill_threshold:
            threshold_hits += 1
    override = bool(
        threshold_hits >= 2
        and required_complete
        and pairwise
        and all(
            sum(
                1 for metric in metric_names if metric_result[metric]["clear_advantage"]
            )
            >= 2
            for metric_result in pairwise.values()
        )
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": (
            "The best MLP architecture clearly improved over every aligned simpler probe on at least two primary multilabel metrics."
            if override
            else "No MLP architecture cleared the configured multilabel skill and pairwise-improvement thresholds."
        ),
        "pairwise_comparisons": pairwise,
        "best_architecture": {
            "probe_name": best_name,
            "micro_f1": float(best_result["micro_f1"]),
            "macro_f1": float(best_result["macro_f1"]),
            "sample_jaccard": float(best_result["sample_jaccard"]),
        },
        "required_comparators_complete": required_complete,
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
    best_name, best_result = _select_best_architecture(
        mlp_results, metrics=metric_names
    )
    if best_name is None or best_result is None:
        return {
            "status": "completed",
            "recommendation_override": False,
            "override_reason": "No MLP architecture completed.",
            "pairwise_comparisons": {},
            "best_architecture": None,
            "required_comparators_complete": False,
        }
    pairwise: dict[str, Any] = {}
    required_complete = all(
        name in comparator_results for name in ("linear", "smooth_poly", "knn")
    )
    best_predictions = np.asarray(best_result["predictions"], dtype=float)
    for name, result in comparator_results.items():
        if "predictions" not in result:
            continue
        comparator_predictions = np.asarray(result["predictions"], dtype=float)
        metric_comparisons: dict[str, Any] = {}
        for metric in metric_names:

            def delta(
                sample_idx: np.ndarray,
                metric_name: str = metric,
                second: np.ndarray = comparator_predictions,
            ) -> float:
                return _regression_metric_delta(
                    Y_true,
                    best_predictions,
                    second,
                    sample_idx,
                    metric=metric_name,
                    target_names=target_names,
                )

            comparison = _bootstrap_comparison(
                delta,
                repeats=_mlp_budget(config)["bootstrap_repeats"],
                random_state=config.random_state,
                n_rows=Y_true.shape[0],
                groups=groups,
            )
            comparison["clear_advantage"] = bool(
                comparison["mean_delta"] >= config.mlp_min_improvement
                and comparison["lower_95"] > 0.0
            )
            metric_comparisons[metric] = comparison
        pairwise[name] = metric_comparisons
    dummy = comparator_results["dummy"]
    threshold_hits = 0
    for metric in metric_names:
        skill = _skill_from_bounds(
            min(1.0, float(best_result[metric])),
            min(1.0, float(dummy[metric])),
        )
        if skill is not None and skill >= config.mlp_trigger_skill_threshold:
            threshold_hits += 1
    override = bool(
        threshold_hits == len(metric_names)
        and required_complete
        and pairwise
        and all(
            all(metric_result[metric]["clear_advantage"] for metric in metric_names)
            for metric_result in pairwise.values()
        )
    )
    return {
        "status": "completed",
        "recommendation_override": override,
        "override_reason": (
            "The best MLP architecture clearly improved over every aligned simpler regressor on both primary R2 metrics."
            if override
            else "No MLP architecture cleared the configured regression skill and pairwise-improvement thresholds."
        ),
        "pairwise_comparisons": pairwise,
        "best_architecture": {
            "probe_name": best_name,
            "r2_variance_weighted": float(best_result["r2_variance_weighted"]),
            "r2_uniform_average": float(best_result["r2_uniform_average"]),
        },
        "required_comparators_complete": required_complete,
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
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _full_quadratic_classifier(config.random_state)
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
    recommendation = _singlelabel_override_evidence(
        mlp_results,
        comparator_results,
        y_true=dense_y,
        config=config,
        groups=dense_groups,
    )
    if any(result.get("status") == "runtime_failed" for result in mlp_results.values()):
        recommendation["recommendation_override"] = False
        recommendation["required_comparators_complete"] = False
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
                {**item, **mlp_results.get(f"mlp_{item['label']}", {})}
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
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _full_multilabel_quadratic_classifier(config)
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
        recommendation["required_comparators_complete"] = False
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
                {**item, **mlp_results.get(f"mlp_{item['label']}", {})}
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
    expanded_features = _quadratic_feature_count(dense_X.shape[1])
    estimated_expanded_mb = _estimate_dense_mb(
        dense_X.shape[0], expanded_features, dense_X.dtype
    )
    if expanded_features <= 50_000 and estimated_expanded_mb <= config.max_dense_mb:
        comparators["smooth_poly"] = _regression_smooth_estimator(config)
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
        recommendation["required_comparators_complete"] = False
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
                {**item, **mlp_results.get(f"mlp_{item['label']}", {})}
                for item in architectures
            ],
            "aligned_comparators": comparator_results,
            **recommendation,
        }
    )
    return payload
