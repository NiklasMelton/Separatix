"""Model probe execution."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

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


def _linear_classifier(X: Any) -> LogisticRegression:
    solver = "saga" if sparse.issparse(X) else "lbfgs"
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver=solver,
    )


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
