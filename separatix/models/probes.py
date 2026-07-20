"""Model probe execution."""

from __future__ import annotations

import time
from math import floor
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from separatix.config import ProfilerConfig
from separatix.constants import BUDGETS
from separatix.densify import (
    ensure_dense_multilabel_target,
    ensure_dense_or_sample,
    ensure_dense_or_sample_multilabel,
    ensure_dense_or_sample_regression,
)
from separatix.models.scoring import (
    MultilabelPriorDummy,
    TargetMeanDummyRegressor,
    choose_cv,
    choose_regression_cv,
    evaluate_estimator,
    evaluate_multilabel_estimator,
    evaluate_regression_estimator,
    summarize_multilabel_predictions,
    summarize_multilabel_stability,
    summarize_predictions,
    summarize_regression_predictions,
    summarize_regression_stability,
    summarize_stability,
)
from separatix.sampling import (
    BudgetConfig,
    cap_multilabel_samples_for_budget,
    cap_regression_samples_for_budget,
    cap_samples_for_budget,
    choose_multilabel_cv,
)

_QUADRATIC_DEGREE = 2
_MAX_FULL_QUADRATIC_FEATURES = 50_000
_MAX_SKETCH_COMPONENTS = 2048
_MIN_SKETCH_COMPONENTS = 128
_FULL_QUADRATIC_VARIANT = "full_quadratic"
_LOW_RANK_QUADRATIC_VARIANT = "low_rank_quadratic"
_SMOOTH_PROBE_VARIANTS = (
    _FULL_QUADRATIC_VARIANT,
    _LOW_RANK_QUADRATIC_VARIANT,
)
_SMOOTH_PROBE_SKIP_REASON = (
    "quadratic expansion and low-rank sketch exceed configured memory budget"
)


def _feature_scaler(X: Any) -> StandardScaler:
    """Return fold-local scaling compatible with the feature representation."""
    return StandardScaler(with_mean=not sparse.issparse(X))


def _scaled_pipeline(X: Any, estimator: Any, *, name: str) -> Pipeline:
    """Wrap an estimator in fold-local feature scaling."""
    return Pipeline([("scale", _feature_scaler(X)), (name, estimator)])


def _prediction_evidence(predictions: Any, config: ProfilerConfig) -> dict[str, Any]:
    """Retain row-level predictions only when they fit the dense-memory budget."""
    prediction_array = np.asarray(predictions)
    limit_bytes = int(config.max_dense_mb * 1024**2)
    if prediction_array.nbytes > limit_bytes:
        return {
            "predictions": None,
            "predictions_omitted_reason": (
                "row-level prediction evidence exceeds max_dense_mb"
            ),
            "prediction_evidence_bytes": int(prediction_array.nbytes),
        }
    return {
        "predictions": prediction_array.tolist(),
        "prediction_evidence_bytes": int(prediction_array.nbytes),
    }


def _linear_classifier(X: Any) -> Pipeline:
    solver = "saga" if sparse.issparse(X) else "lbfgs"
    return _scaled_pipeline(
        X,
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver=solver,
        ),
        name="clf",
    )


def _dense_multilabel_matrix(Y: Any) -> np.ndarray:
    """Return a dense multilabel indicator matrix."""
    dense = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense.astype(np.int8, copy=False)


def _multilabel_linear_classifier(
    X: Any, config: ProfilerConfig
) -> OneVsRestClassifier:
    """Return a sparse-aware one-vs-rest logistic probe."""
    solver = "saga" if sparse.issparse(X) else "lbfgs"
    return OneVsRestClassifier(
        _scaled_pipeline(
            X,
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver=solver,
                random_state=config.random_state,
            ),
            name="clf",
        ),
        n_jobs=config.n_jobs,
    )


def _quadratic_feature_count(n_features: int) -> int:
    return n_features + (n_features * (n_features + 1)) // 2


def _estimate_dense_mb(n_rows: int, n_features: int, dtype: np.dtype[Any]) -> float:
    return float(n_rows * n_features * dtype.itemsize / 1024**2)


def _full_quadratic_classifier(random_state: int | None) -> Pipeline:
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            ("poly", PolynomialFeatures(degree=_QUADRATIC_DEGREE, include_bias=False)),
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


def _full_multilabel_quadratic_classifier(config: ProfilerConfig) -> Pipeline:
    """Return a full quadratic multilabel probe pipeline."""
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            ("poly", PolynomialFeatures(degree=_QUADRATIC_DEGREE, include_bias=False)),
            ("scale_out", StandardScaler()),
            ("clf", _multilabel_linear_classifier(np.zeros((1, 1)), config)),
        ]
    )


def _low_rank_multilabel_quadratic_classifier(
    n_components: int,
    config: ProfilerConfig,
) -> Pipeline:
    """Return a low-rank quadratic multilabel probe pipeline."""
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "poly_sketch",
                PolynomialCountSketch(
                    degree=_QUADRATIC_DEGREE,
                    coef0=1.0,
                    n_components=n_components,
                    random_state=config.random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            ("clf", _multilabel_linear_classifier(np.zeros((1, 1)), config)),
        ]
    )


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


def _fold_count(cv: Any | None) -> int:
    """Return the configured number of validation folds when available."""
    if cv is None:
        return 0
    try:
        return int(cv.get_n_splits())
    except (AttributeError, TypeError):
        return 0


def _classification_evaluation_support(y: np.ndarray, cv: Any | None) -> dict[str, Any]:
    """Summarize the exact rows and class support used by a probe."""
    classes, counts = np.unique(y, return_counts=True)
    return {
        "n_samples": int(y.shape[0]),
        "cv_folds": _fold_count(cv),
        "class_counts": {
            str(cls): int(count)
            for cls, count in zip(classes, counts)  # noqa: B905
        },
    }


def _multilabel_evaluation_support(Y: Any, cv: Any | None) -> dict[str, Any]:
    """Summarize the exact rows and label support used by a probe."""
    dense = _dense_multilabel_matrix(Y)
    positives = dense.sum(axis=0).astype(int)
    return {
        "n_samples": int(dense.shape[0]),
        "cv_folds": _fold_count(cv),
        "positive_counts": positives.tolist(),
        "negative_counts": (dense.shape[0] - positives).tolist(),
    }


def _regression_evaluation_support(Y: np.ndarray, cv: Any | None) -> dict[str, Any]:
    """Summarize the exact rows and target support used by a probe."""
    return {
        "n_samples": int(Y.shape[0]),
        "cv_folds": _fold_count(cv),
        "target_variance": np.var(Y, axis=0).astype(float).tolist(),
    }


def run_model_probes(
    X: Any,
    y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    class_labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Run lightweight baseline and probe classifiers."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    X_used, y_used, sample_info = cap_samples_for_budget(
        X, y, config=config, reason="probe", groups=groups
    )
    groups_used = (
        groups[np.asarray(sample_info["indices"], dtype=int)]
        if groups is not None
        else None
    )
    if sample_info.get("support_preserved") is False:
        reason = str(sample_info.get("skip_reason"))
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "model_probes", "reason": reason, "severity": "blocking"}
        )
        return {
            name: {
                "skipped_reason": reason,
                "model_name": model_name,
                "sample_info": sample_info,
                "evaluation_mode": "insufficient_support",
            }
            for name, model_name in {
                "dummy": "DummyClassifier",
                "linear": "LogisticRegression",
                "knn": "KNeighborsClassifier",
                "smooth_poly": "PolynomialFeatures+LogisticRegression",
                "kernel_approx": "RBFSampler+SGDClassifier",
            }.items()
        }
    cv, cv_method = choose_cv(
        y_used,
        budget["cv_folds"],
        groups=groups_used,
        random_state=config.random_state,
    )
    if cv is None:
        skipped = report_context.setdefault("skipped_diagnostics", [])
        reason = (
            "No valid group-disjoint supervised split was available after "
            "filtering unsupported classes."
            if groups is not None
            else "At least two examples per class are required for held-out probes."
        )
        skipped.append(
            {
                "name": "supervised_cross_validation",
                "reason": reason,
                "severity": "blocking",
            }
        )
        evaluation_mode = (
            "group_split_unavailable" if groups is not None else "insufficient_support"
        )
        return {
            name: {
                "skipped_reason": reason,
                "model_name": estimator.__class__.__name__,
                "sample_info": sample_info,
                "evaluation_mode": evaluation_mode,
                "cv_stratification_method": cv_method,
            }
            for name, estimator in {
                "dummy": DummyClassifier(strategy="prior"),
                "linear": _linear_classifier(X_used),
                "knn": KNeighborsClassifier(
                    n_neighbors=min(
                        max(1, len(y_used) - 1),
                        min(15, max(3, int(np.sqrt(len(y_used))))),
                    )
                ),
            }.items()
        } | {
            "smooth_poly": {
                "skipped_reason": reason,
                "model_name": "PolynomialFeatures+LogisticRegression",
                "sample_info": sample_info,
                "evaluation_mode": evaluation_mode,
                "cv_stratification_method": cv_method,
            },
            "kernel_approx": {
                "skipped_reason": reason,
                "model_name": "RBFSampler+SGDClassifier",
                "sample_info": sample_info,
                "evaluation_mode": evaluation_mode,
                "cv_stratification_method": cv_method,
            },
        }
    probes: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior"),
        "linear": _linear_classifier(X_used),
        "knn": _scaled_pipeline(
            X_used,
            KNeighborsClassifier(
                n_neighbors=min(
                    max(1, len(y_used) - 1),
                    min(15, max(3, int(np.sqrt(len(y_used))))),
                )
            ),
            name="knn",
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
            groups=groups_used,
        )
        metrics = summarize_predictions(y_used, preds, class_labels=class_labels)
        metrics.update(
            summarize_stability(
                estimator,
                X_used,
                y_used,
                repeats=budget["bootstrap_repeats"],
                random_state=config.random_state,
                groups=groups_used,
            )
        )
        metrics.update(
            {
                "model_name": estimator.__class__.__name__,
                "runtime_seconds": float(time.perf_counter() - start),
                "sample_info": sample_info,
                "evaluation_mode": evaluation_mode,
                "cv_stratification_method": cv_method,
                **_prediction_evidence(preds, config),
                "evaluation_support": _classification_evaluation_support(y_used, cv),
            }
        )
        results[name] = metrics

    smooth_info = ensure_dense_or_sample(
        X_used,
        y_used,
        reason="smooth_nonlinear_probe",
        config=config,
        report_context=report_context,
        groups=groups_used,
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
        dense_groups = smooth_info.get("groups")
        dtype = dense_X.dtype if dense_X.dtype is not None else np.dtype(float)
        expanded_features = _quadratic_feature_count(dense_X.shape[1])
        estimated_expanded_mb = _estimate_dense_mb(
            dense_X.shape[0], expanded_features, dtype
        )
        metadata = _smooth_probe_metadata(dense_X, sample_info)
        cv_smooth, cv_smooth_method = choose_cv(
            dense_y,
            budget["cv_folds"],
            groups=dense_groups,
            random_state=config.random_state,
        )
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
                groups=dense_groups,
            )
            metrics = summarize_predictions(dense_y, preds, class_labels=class_labels)
            metrics.update(
                summarize_stability(
                    estimator,
                    dense_X,
                    dense_y,
                    repeats=budget["bootstrap_repeats"],
                    random_state=config.random_state,
                    groups=dense_groups,
                )
            )
            metrics.update(
                {
                    **metadata,
                    "probe_variant": _FULL_QUADRATIC_VARIANT,
                    "model_name": "PolynomialFeatures+LogisticRegression",
                    "runtime_seconds": float(time.perf_counter() - start),
                    "evaluation_mode": evaluation_mode,
                    "cv_stratification_method": cv_smooth_method,
                    **_prediction_evidence(preds, config),
                    "evaluation_support": _classification_evaluation_support(
                        dense_y, cv_smooth
                    ),
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
                    groups=dense_groups,
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
                        groups=dense_groups,
                    )
                )
                metrics.update(
                    {
                        **metadata,
                        "probe_variant": _LOW_RANK_QUADRATIC_VARIANT,
                        "sketch_n_components": int(sketch_components),
                        "estimated_sketch_mb": _estimate_dense_mb(
                            dense_X.shape[0], sketch_components, dtype
                        ),
                        "model_name": "PolynomialCountSketch+LogisticRegression",
                        "runtime_seconds": float(time.perf_counter() - start),
                        "evaluation_mode": evaluation_mode,
                        "cv_stratification_method": cv_smooth_method,
                        **_prediction_evidence(preds, config),
                        "evaluation_support": _classification_evaluation_support(
                            dense_y, cv_smooth
                        ),
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
            groups=groups_used,
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
                    ("scale_in", StandardScaler()),
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
            start = time.perf_counter()
            preds, evaluation_mode = evaluate_estimator(
                estimator,
                dense_info["X"],
                dense_info["y"],
                cv=choose_cv(
                    dense_info["y"],
                    budget["cv_folds"],
                    groups=dense_info.get("groups"),
                    random_state=config.random_state,
                )[0],
                groups=dense_info.get("groups"),
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
                    groups=dense_info.get("groups"),
                )
            )
            metrics.update(
                {
                    "model_name": "RBFSampler+SGDClassifier",
                    "runtime_seconds": float(time.perf_counter() - start),
                    "sample_info": sample_info,
                    "evaluation_mode": evaluation_mode,
                    "cv_stratification_method": choose_cv(
                        dense_info["y"],
                        budget["cv_folds"],
                        groups=dense_info.get("groups"),
                        random_state=config.random_state,
                    )[1],
                    **_prediction_evidence(preds, config),
                    "evaluation_support": _classification_evaluation_support(
                        np.asarray(dense_info["y"]),
                        choose_cv(
                            dense_info["y"],
                            budget["cv_folds"],
                            groups=dense_info.get("groups"),
                            random_state=config.random_state,
                        )[0],
                    ),
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


def _regression_smooth_estimator(
    config: ProfilerConfig,
    *,
    low_rank_components: int | None = None,
) -> Pipeline:
    """Return a smooth nonlinear regression probe."""
    if low_rank_components is None:
        return Pipeline(
            [
                ("scale_in", StandardScaler()),
                (
                    "poly",
                    PolynomialFeatures(degree=_QUADRATIC_DEGREE, include_bias=False),
                ),
                ("scale_out", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=config.random_state)),
            ]
        )
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "poly_sketch",
                PolynomialCountSketch(
                    degree=_QUADRATIC_DEGREE,
                    coef0=1.0,
                    n_components=low_rank_components,
                    random_state=config.random_state,
                ),
            ),
            ("scale_out", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=config.random_state)),
        ]
    )


def _record_regression_probe(
    estimator: Any,
    X: Any,
    Y: np.ndarray,
    *,
    name: str,
    model_name: str,
    config: ProfilerConfig,
    budget: BudgetConfig,
    sample_info: dict[str, Any],
    target_names: np.ndarray,
    groups: np.ndarray | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Evaluate and summarize one regression probe."""
    cv, cv_method = choose_regression_cv(
        Y,
        budget["cv_folds"],
        groups=groups,
        random_state=config.random_state,
    )
    if cv is None and groups is not None:
        return {
            "skipped_reason": "group-disjoint supervised split unavailable",
            "model_name": model_name,
            "probe_name": name,
            "sample_info": sample_info,
            "evaluation_mode": "group_split_unavailable",
            "cv_method": cv_method,
        }
    start = time.perf_counter()
    preds, evaluation_mode = evaluate_regression_estimator(
        estimator,
        X,
        Y,
        cv=cv,
        groups=groups,
    )
    metrics = summarize_regression_predictions(Y, preds, target_names=target_names)
    metrics.update(
        summarize_regression_stability(
            estimator,
            X,
            Y,
            repeats=budget["bootstrap_repeats"],
            random_state=config.random_state,
            target_names=target_names,
            groups=groups,
        )
    )
    metrics.update(
        {
            "model_name": model_name,
            "probe_name": name,
            "runtime_seconds": float(time.perf_counter() - start),
            "sample_info": sample_info,
            "evaluation_mode": evaluation_mode,
            "cv_method": cv_method,
            **_prediction_evidence(preds, config),
            "evaluation_support": _regression_evaluation_support(Y, cv),
        }
    )
    if extra:
        metrics.update(extra)
    return metrics


def run_regression_model_probes(
    X: Any,
    Y: np.ndarray,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    target_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Run lightweight baseline and probe regressors."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    X_used, Y_used, sample_info = cap_regression_samples_for_budget(
        X, Y, config=config, reason="probe", groups=groups
    )
    groups_used = (
        groups[np.asarray(sample_info["indices"], dtype=int)]
        if groups is not None
        else None
    )
    if sample_info.get("support_preserved") is False:
        reason = str(sample_info.get("skip_reason"))
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "regression_model_probes",
                "reason": reason,
                "severity": "blocking",
            }
        )
        return {
            name: {
                "skipped_reason": reason,
                "model_name": model_name,
                "sample_info": sample_info,
                "evaluation_mode": "insufficient_support",
            }
            for name, model_name in {
                "dummy": "TargetMeanDummyRegressor",
                "linear": "Ridge",
                "knn": "KNeighborsRegressor",
                "smooth_poly": "PolynomialFeatures+Ridge",
                "kernel_approx": "RBFSampler+Ridge",
            }.items()
        }
    n_neighbors = min(
        max(1, Y_used.shape[0] - 1),
        min(15, max(3, int(np.sqrt(Y_used.shape[0])))),
    )
    probes: dict[str, Any] = {
        "dummy": TargetMeanDummyRegressor(),
        "linear": _scaled_pipeline(
            X_used,
            Ridge(alpha=1.0, random_state=config.random_state),
            name="reg",
        ),
        "knn": _scaled_pipeline(
            X_used,
            KNeighborsRegressor(n_neighbors=n_neighbors),
            name="knn",
        ),
    }
    results: dict[str, dict[str, object]] = {}
    for name, estimator in probes.items():
        results[name] = _record_regression_probe(
            estimator,
            X_used,
            Y_used,
            name=name,
            model_name=estimator.__class__.__name__,
            config=config,
            budget=budget,
            sample_info=sample_info,
            target_names=target_names,
            groups=groups_used,
        )

    smooth_info = ensure_dense_or_sample_regression(
        X_used,
        Y_used,
        reason="regression_smooth_nonlinear_probe",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if smooth_info["skipped"]:
        results["smooth_poly"] = {
            "skipped_reason": "dense conversion unavailable under current policy",
            "model_name": "PolynomialFeatures+Ridge",
            "sample_info": sample_info,
        }
    else:
        dense_X = np.asarray(smooth_info["X"])
        dense_Y = np.asarray(smooth_info["y"], dtype=float)
        dense_groups = smooth_info.get("groups")
        dtype = dense_X.dtype if dense_X.dtype is not None else np.dtype(float)
        expanded_features = _quadratic_feature_count(dense_X.shape[1])
        estimated_expanded_mb = _estimate_dense_mb(
            dense_X.shape[0], expanded_features, dtype
        )
        metadata = _smooth_probe_metadata(dense_X, sample_info)
        if (
            expanded_features <= _MAX_FULL_QUADRATIC_FEATURES
            and estimated_expanded_mb <= config.max_dense_mb
        ):
            estimator = _regression_smooth_estimator(config)
            results["smooth_poly"] = _record_regression_probe(
                estimator,
                dense_X,
                dense_Y,
                name="smooth_poly",
                model_name="PolynomialFeatures+Ridge",
                config=config,
                budget=budget,
                sample_info=sample_info,
                target_names=target_names,
                groups=dense_groups,
                extra={**metadata, "probe_variant": _FULL_QUADRATIC_VARIANT},
            )
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
                        "name": "regression_smooth_nonlinear_probe",
                        "reason": _SMOOTH_PROBE_SKIP_REASON,
                    }
                )
                results["smooth_poly"] = {
                    **metadata,
                    "skipped_reason": _SMOOTH_PROBE_SKIP_REASON,
                    "model_name": "PolynomialCountSketch+Ridge",
                }
            else:
                estimator = _regression_smooth_estimator(
                    config, low_rank_components=sketch_components
                )
                results["smooth_poly"] = _record_regression_probe(
                    estimator,
                    dense_X,
                    dense_Y,
                    name="smooth_poly",
                    model_name="PolynomialCountSketch+Ridge",
                    config=config,
                    budget=budget,
                    sample_info=sample_info,
                    target_names=target_names,
                    groups=dense_groups,
                    extra={
                        **metadata,
                        "probe_variant": _LOW_RANK_QUADRATIC_VARIANT,
                        "sketch_n_components": int(sketch_components),
                        "estimated_sketch_mb": _estimate_dense_mb(
                            dense_X.shape[0], sketch_components, dtype
                        ),
                    },
                )

    if budget["run_kernel_probe"]:
        dense_info = ensure_dense_or_sample_regression(
            X_used,
            Y_used,
            reason="regression_kernel_approximation_probe",
            config=config,
            report_context=report_context,
            groups=groups_used,
        )
        if dense_info["skipped"]:
            results["kernel_approx"] = {
                "skipped_reason": "dense conversion unavailable under current policy",
                "model_name": "RBFSampler+Ridge",
                "sample_info": sample_info,
            }
        else:
            estimator = Pipeline(
                [
                    ("scale_in", StandardScaler()),
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
                    ("scale_out", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=config.random_state)),
                ]
            )
            results["kernel_approx"] = _record_regression_probe(
                estimator,
                np.asarray(dense_info["X"]),
                np.asarray(dense_info["y"], dtype=float),
                name="kernel_approx",
                model_name="RBFSampler+Ridge",
                config=config,
                budget=budget,
                sample_info=sample_info,
                target_names=target_names,
                groups=dense_info.get("groups"),
            )
    else:
        results["kernel_approx"] = {
            "skipped_reason": "kernel probe disabled for this budget",
            "model_name": "RBFSampler+Ridge",
            "sample_info": sample_info,
        }
    return results


def _ensure_dense_X_for_multilabel(
    X: Any,
    Y: Any,
    *,
    reason: str,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Densify X for multilabel probes while preserving aligned grouping."""
    return ensure_dense_or_sample_multilabel(
        X,
        Y,
        reason=reason,
        config=config,
        report_context=report_context,
        groups=groups,
    )


def _evaluate_multilabel_probe(
    name: str,
    estimator: Any,
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    budget: BudgetConfig,
    label_names: np.ndarray,
    sample_info: dict[str, Any],
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one multilabel probe and return report metrics."""
    start = time.perf_counter()
    cv, cv_method = choose_multilabel_cv(
        Y,
        max_folds=budget["cv_folds"],
        config=config,
        groups=groups,
    )
    if cv is None:
        unavailable = (
            "group-disjoint supervised split unavailable"
            if groups is not None
            else "support-preserving supervised split unavailable"
        )
        return {
            "skipped_reason": unavailable,
            "model_name": name,
            "sample_info": sample_info,
            "evaluation_mode": cv_method,
            "cv_stratification_method": cv_method,
        }
    preds, evaluation_mode = evaluate_multilabel_estimator(
        estimator,
        X,
        Y,
        cv=cv,
        groups=groups,
    )
    metrics = summarize_multilabel_predictions(Y, preds, label_names=label_names)
    metrics.update(
        summarize_multilabel_stability(
            estimator,
            X,
            Y,
            repeats=budget["bootstrap_repeats"],
            random_state=config.random_state,
            config=config,
            label_names=label_names,
            groups=groups,
        )
    )
    metrics.update(
        {
            "model_name": name,
            "runtime_seconds": float(time.perf_counter() - start),
            "sample_info": sample_info,
            "evaluation_mode": evaluation_mode,
            "cv_stratification_method": cv_method,
            **_prediction_evidence(preds, config),
            "evaluation_support": _multilabel_evaluation_support(Y, cv),
        }
    )
    return metrics


def run_multilabel_model_probes(
    X: Any,
    Y: Any,
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
    label_names: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Run lightweight multilabel baseline and probe classifiers."""
    budget = cast(BudgetConfig, BUDGETS[config.budget])
    X_used, Y_used, sample_info = cap_multilabel_samples_for_budget(
        X,
        Y,
        config=config,
        reason="probe",
        groups=groups,
    )
    groups_used = (
        groups[np.asarray(sample_info["indices"], dtype=int)]
        if groups is not None
        else None
    )
    if sample_info.get("support_preserved") is False:
        reason = str(sample_info.get("skip_reason"))
        report_context.setdefault("skipped_diagnostics", []).append(
            {
                "name": "multilabel_model_probes",
                "reason": reason,
                "severity": "blocking",
            }
        )
        return {
            name: {
                "skipped_reason": reason,
                "model_name": model_name,
                "sample_info": sample_info,
                "evaluation_mode": "insufficient_support",
            }
            for name, model_name in {
                "dummy": "MultilabelPriorDummy",
                "linear": "OneVsRestLogisticRegression",
                "knn": "KNeighborsClassifier",
                "smooth_poly": "PolynomialFeatures+OneVsRestLogisticRegression",
                "kernel_approx": "RBFSampler+OneVsRestSGDClassifier",
            }.items()
        }
    target_info = ensure_dense_multilabel_target(
        X_used,
        Y_used,
        reason="multilabel_probe_target",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if target_info["skipped"]:
        reason = "multilabel target exceeds configured dense-memory budget"
        return {
            name: {
                "skipped_reason": reason,
                "model_name": model_name,
                "sample_info": sample_info,
                "evaluation_mode": "memory_budget_unavailable",
            }
            for name, model_name in {
                "dummy": "MultilabelPriorDummy",
                "linear": "OneVsRestLogisticRegression",
                "knn": "KNeighborsClassifier",
                "smooth_poly": "PolynomialFeatures+OneVsRestLogisticRegression",
                "kernel_approx": "RBFSampler+OneVsRestSGDClassifier",
            }.items()
        }
    relative_indices = np.asarray(target_info["indices"], dtype=int)
    original_indices = np.asarray(sample_info["indices"], dtype=int)
    X_used = target_info["X"]
    Y_used = target_info["Y"]
    groups_used = target_info.get("groups")
    if relative_indices.shape[0] != original_indices.shape[0] or not np.array_equal(
        relative_indices, np.arange(original_indices.shape[0])
    ):
        sample_info = {
            **sample_info,
            "sampled": True,
            "n_used": int(relative_indices.size),
            "indices": original_indices[relative_indices].tolist(),
            "target_memory_sampling": True,
        }
    y_dense = _dense_multilabel_matrix(Y_used)
    n_neighbors = min(
        max(1, y_dense.shape[0] - 1),
        min(15, max(3, int(np.sqrt(y_dense.shape[0])))),
    )
    probes: dict[str, Any] = {
        "dummy": MultilabelPriorDummy(threshold=0.5),
        "linear": _multilabel_linear_classifier(X_used, config),
        "knn": _scaled_pipeline(
            X_used,
            KNeighborsClassifier(n_neighbors=n_neighbors),
            name="knn",
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in probes.items():
        results[name] = _evaluate_multilabel_probe(
            name,
            estimator,
            X_used,
            Y_used,
            config=config,
            budget=budget,
            label_names=label_names,
            sample_info=sample_info,
            groups=groups_used,
        )

    dense_info = _ensure_dense_X_for_multilabel(
        X_used,
        Y_used,
        reason="smooth_nonlinear_probe",
        config=config,
        report_context=report_context,
        groups=groups_used,
    )
    if dense_info["skipped"]:
        results["smooth_poly"] = {
            "skipped_reason": "dense conversion unavailable under current policy",
            "model_name": "PolynomialFeatures+OneVsRestLogisticRegression",
            "sample_info": sample_info,
        }
    else:
        dense_X = np.asarray(dense_info["X"])
        dense_Y = dense_info["Y"]
        dtype = dense_X.dtype if dense_X.dtype is not None else np.dtype(float)
        expanded_features = _quadratic_feature_count(dense_X.shape[1])
        expanded_mb = _estimate_dense_mb(dense_X.shape[0], expanded_features, dtype)
        metadata = {
            "probe_degree": _QUADRATIC_DEGREE,
            "original_feature_count": int(dense_X.shape[1]),
            "estimated_expanded_feature_count": int(expanded_features),
            "estimated_expanded_mb": expanded_mb,
            "sample_info": sample_info,
        }
        if (
            expanded_features <= _MAX_FULL_QUADRATIC_FEATURES
            and expanded_mb <= config.max_dense_mb
        ):
            results["smooth_poly"] = _evaluate_multilabel_probe(
                "PolynomialFeatures+OneVsRestLogisticRegression",
                _full_multilabel_quadratic_classifier(config),
                dense_X,
                dense_Y,
                config=config,
                budget=budget,
                label_names=label_names,
                sample_info=sample_info,
                groups=dense_info.get("groups"),
            )
            results["smooth_poly"].update(metadata)
            results["smooth_poly"]["probe_variant"] = _FULL_QUADRATIC_VARIANT
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
                    "model_name": "PolynomialCountSketch+OneVsRestLogisticRegression",
                }
            else:
                results["smooth_poly"] = _evaluate_multilabel_probe(
                    "PolynomialCountSketch+OneVsRestLogisticRegression",
                    _low_rank_multilabel_quadratic_classifier(
                        sketch_components, config
                    ),
                    dense_X,
                    dense_Y,
                    config=config,
                    budget=budget,
                    label_names=label_names,
                    sample_info=sample_info,
                    groups=dense_info.get("groups"),
                )
                results["smooth_poly"].update(metadata)
                results["smooth_poly"]["probe_variant"] = _LOW_RANK_QUADRATIC_VARIANT
                results["smooth_poly"]["sketch_n_components"] = int(sketch_components)

    if budget["run_kernel_probe"]:
        kernel_info = _ensure_dense_X_for_multilabel(
            X_used,
            Y_used,
            reason="kernel_approximation_probe",
            config=config,
            report_context=report_context,
            groups=groups_used,
        )
        if kernel_info["skipped"]:
            results["kernel_approx"] = {
                "skipped_reason": "dense conversion unavailable under current policy",
                "model_name": "RBFSampler+OneVsRestSGDClassifier",
                "sample_info": sample_info,
            }
        else:
            estimator = Pipeline(
                [
                    ("scale_in", StandardScaler()),
                    (
                        "rff",
                        RBFSampler(
                            gamma=1.0 / max(1, kernel_info["X"].shape[1]),
                            n_components=min(
                                256, max(32, kernel_info["X"].shape[1] * 2)
                            ),
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
            results["kernel_approx"] = _evaluate_multilabel_probe(
                "RBFSampler+OneVsRestSGDClassifier",
                estimator,
                kernel_info["X"],
                kernel_info["Y"],
                config=config,
                budget=budget,
                label_names=label_names,
                sample_info=sample_info,
                groups=kernel_info.get("groups"),
            )
    else:
        results["kernel_approx"] = {
            "skipped_reason": "kernel probe disabled for this budget",
            "model_name": "RBFSampler+OneVsRestSGDClassifier",
            "sample_info": sample_info,
        }
    return results
