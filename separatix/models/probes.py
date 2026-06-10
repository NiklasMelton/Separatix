"""Model probe execution."""

from __future__ import annotations

import time
from math import floor
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.densify import ensure_dense_or_sample
from separatix.models.scoring import (
    choose_cv,
    evaluate_estimator,
    summarize_predictions,
    summarize_stability,
)
from separatix.sampling import BudgetConfig, cap_samples_for_budget
from separatix.utils.warnings import record_warning

_QUADRATIC_DEGREE = 2
_MAX_FULL_QUADRATIC_FEATURES = 50_000
_MAX_SKETCH_COMPONENTS = 2048
_MIN_SKETCH_COMPONENTS = 128
_SMOOTH_PROBE_SKIP_REASON = (
    "quadratic expansion and low-rank sketch exceed configured memory budget"
)


def _linear_classifier(X: Any) -> LogisticRegression:
    solver = "saga" if sparse.issparse(X) else "lbfgs"
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver=solver,
    )


def _quadratic_feature_count(n_features: int) -> int:
    return n_features + (n_features * (n_features + 1)) // 2


def _estimate_dense_mb(n_rows: int, n_features: int, dtype: np.dtype[Any]) -> float:
    return float(n_rows * n_features * dtype.itemsize / 1024**2)


def _full_quadratic_classifier(random_state: int | None) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=_QUADRATIC_DEGREE, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _low_rank_quadratic_classifier(
    n_components: int,
    random_state: int | None,
) -> Pipeline:
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "poly_sketch",
                PolynomialCountSketch(
                    degree=_QUADRATIC_DEGREE,
                    coef0=1.0,
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _smooth_probe_metadata(
    X: np.ndarray,
    sample_info: dict[str, Any],
) -> dict[str, int | float | str | dict[str, Any]]:
    dtype = np.asarray(X).dtype if np.asarray(X).dtype is not None else np.dtype(float)
    n_rows, n_features = X.shape
    expanded_features = _quadratic_feature_count(n_features)
    return {
        "probe_degree": _QUADRATIC_DEGREE,
        "original_feature_count": int(n_features),
        "estimated_expanded_feature_count": int(expanded_features),
        "estimated_expanded_mb": _estimate_dense_mb(n_rows, expanded_features, dtype),
        "sample_info": sample_info,
    }


def _choose_sketch_components(
    n_rows: int,
    n_features: int,
    dtype: np.dtype[Any],
    *,
    max_dense_mb: int,
) -> int | None:
    preferred = min(_MAX_SKETCH_COMPONENTS, max(_MIN_SKETCH_COMPONENTS, 4 * n_features))
    max_by_memory = floor((max_dense_mb * 1024**2) / max(1, n_rows * dtype.itemsize))
    if max_by_memory < _MIN_SKETCH_COMPONENTS:
        return None
    return int(min(preferred, max_by_memory))


def run_model_probes(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    class_labels: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Run lightweight baseline and probe classifiers."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    X_used, y_used, sample_info = cap_samples_for_budget(
        X, y, config=config, reason="probe"
    )
    cv = choose_cv(y_used, budget["cv_folds"])
    warnings_list = report_context.setdefault("warnings", [])
    if cv is None:
        record_warning(
            (
                "Very small class counts forced low-reliability "
                "in-sample probe evaluation."
            ),
            warnings_list,
            UserWarning,
        )
    probes: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior"),
        "linear": _linear_classifier(X_used),
        "knn": KNeighborsClassifier(
            n_neighbors=min(
                max(1, len(y_used) - 1),
                min(15, max(3, int(np.sqrt(len(y_used))))),
            )
        ),
    }
    results: dict[str, dict[str, object]] = {}
    for name, estimator in probes.items():
        start = time.perf_counter()
        preds, evaluation_mode = evaluate_estimator(
            estimator,
            X_used,
            y_used,
            cv=cv,
        )
        metrics = summarize_predictions(y_used, preds, class_labels=class_labels)
        metrics.update(
            summarize_stability(
                estimator,
                X_used,
                y_used,
                repeats=budget["bootstrap_repeats"],
                random_state=config.random_state,
            )
        )
        metrics.update(
            {
                "model_name": estimator.__class__.__name__,
                "runtime_seconds": float(time.perf_counter() - start),
                "sample_info": sample_info,
                "evaluation_mode": evaluation_mode,
                "predictions": preds.tolist(),
            }
        )
        results[name] = metrics

    smooth_info = ensure_dense_or_sample(
        X_used,
        y_used,
        reason="smooth_nonlinear_probe",
        config=config,
        report_context=report_context,
    )
    if smooth_info["skipped"]:
        results["smooth_poly"] = {
            "skipped_reason": "dense conversion unavailable under current policy",
            "model_name": "PolynomialFeatures+LogisticRegression",
            "sample_info": sample_info,
        }
    else:
        dense_X = np.asarray(smooth_info["X"])
        dense_y = np.asarray(smooth_info["y"])
        dtype = dense_X.dtype if dense_X.dtype is not None else np.dtype(float)
        expanded_features = _quadratic_feature_count(dense_X.shape[1])
        estimated_expanded_mb = _estimate_dense_mb(
            dense_X.shape[0], expanded_features, dtype
        )
        metadata = _smooth_probe_metadata(dense_X, sample_info)
        cv_smooth = choose_cv(dense_y, budget["cv_folds"])
        if (
            expanded_features <= _MAX_FULL_QUADRATIC_FEATURES
            and estimated_expanded_mb <= config.max_dense_mb
        ):
            estimator = _full_quadratic_classifier(config.random_state)
            start = time.perf_counter()
            preds, evaluation_mode = evaluate_estimator(
                estimator,
                dense_X,
                dense_y,
                cv=cv_smooth,
            )
            metrics = summarize_predictions(dense_y, preds, class_labels=class_labels)
            metrics.update(
                summarize_stability(
                    estimator,
                    dense_X,
                    dense_y,
                    repeats=budget["bootstrap_repeats"],
                    random_state=config.random_state,
                )
            )
            metrics.update(
                {
                    **metadata,
                    "probe_variant": "full_quadratic",
                    "model_name": "PolynomialFeatures+LogisticRegression",
                    "runtime_seconds": float(time.perf_counter() - start),
                    "evaluation_mode": evaluation_mode,
                    "predictions": preds.tolist(),
                }
            )
            results["smooth_poly"] = metrics
        else:
            sketch_components = _choose_sketch_components(
                dense_X.shape[0],
                dense_X.shape[1],
                dtype,
                max_dense_mb=config.max_dense_mb,
            )
            if sketch_components is None:
                report_context.setdefault("skipped_diagnostics", []).append(
                    {
                        "name": "smooth_nonlinear_probe",
                        "reason": _SMOOTH_PROBE_SKIP_REASON,
                    }
                )
                results["smooth_poly"] = {
                    **metadata,
                    "skipped_reason": _SMOOTH_PROBE_SKIP_REASON,
                    "model_name": "PolynomialCountSketch+LogisticRegression",
                }
            else:
                estimator = _low_rank_quadratic_classifier(
                    sketch_components, config.random_state
                )
                start = time.perf_counter()
                preds, evaluation_mode = evaluate_estimator(
                    estimator,
                    dense_X,
                    dense_y,
                    cv=cv_smooth,
                )
                metrics = summarize_predictions(
                    dense_y, preds, class_labels=class_labels
                )
                metrics.update(
                    summarize_stability(
                        estimator,
                        dense_X,
                        dense_y,
                        repeats=budget["bootstrap_repeats"],
                        random_state=config.random_state,
                    )
                )
                metrics.update(
                    {
                        **metadata,
                        "probe_variant": "low_rank_quadratic",
                        "sketch_n_components": int(sketch_components),
                        "estimated_sketch_mb": _estimate_dense_mb(
                            dense_X.shape[0], sketch_components, dtype
                        ),
                        "model_name": "PolynomialCountSketch+LogisticRegression",
                        "runtime_seconds": float(time.perf_counter() - start),
                        "evaluation_mode": evaluation_mode,
                        "predictions": preds.tolist(),
                    }
                )
                results["smooth_poly"] = metrics

    if budget["run_kernel_probe"]:
        dense_info = ensure_dense_or_sample(
            X_used,
            y_used,
            reason="kernel_approximation_probe",
            config=config,
            report_context=report_context,
        )
        if dense_info["skipped"]:
            results["kernel_approx"] = {
                "skipped_reason": "dense conversion unavailable under current policy",
                "model_name": "RBFSampler+SGDClassifier",
                "sample_info": sample_info,
            }
        else:
            estimator = Pipeline(
                [
                    (
                        "rff",
                        RBFSampler(
                            gamma=1.0 / max(1, dense_info["X"].shape[1]),
                            n_components=min(
                                256, max(32, dense_info["X"].shape[1] * 2)
                            ),
                            random_state=config.random_state,
                        ),
                    ),
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
            start = time.perf_counter()
            preds, evaluation_mode = evaluate_estimator(
                estimator,
                dense_info["X"],
                dense_info["y"],
                cv=choose_cv(dense_info["y"], budget["cv_folds"]),
            )
            metrics = summarize_predictions(
                dense_info["y"], preds, class_labels=class_labels
            )
            metrics.update(
                summarize_stability(
                    estimator,
                    dense_info["X"],
                    dense_info["y"],
                    repeats=budget["bootstrap_repeats"],
                    random_state=config.random_state,
                )
            )
            metrics.update(
                {
                    "model_name": "RBFSampler+SGDClassifier",
                    "runtime_seconds": float(time.perf_counter() - start),
                    "sample_info": sample_info,
                    "evaluation_mode": evaluation_mode,
                    "predictions": preds.tolist(),
                }
            )
            results["kernel_approx"] = metrics
    else:
        results["kernel_approx"] = {
            "skipped_reason": "kernel probe disabled for this budget",
            "model_name": "RBFSampler+SGDClassifier",
            "sample_info": sample_info,
        }
    return results
