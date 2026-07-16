from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from separatix import ProfilerConfig, diagnose
from separatix.constants import (
    FEEDFORWARD_MLP_RECOMMENDED,
    FEEDFORWARD_MLP_REGRESSION_RECOMMENDED,
)
from separatix.models import mlp as mlp_module
from separatix.models.mlp import (
    TorchMLPClassifier,
    _architecture_candidates,
    _evaluate_multilabel_models,
    _evaluate_regression_models,
    _evaluate_singlelabel_models,
    _resolve_device,
    _safe_evaluate_models,
    _singlelabel_override_evidence,
    _singlelabel_validation_indices,
    maybe_run_multilabel_mlp_probes,
    maybe_run_regression_mlp_probes,
    maybe_run_singlelabel_mlp_probes,
)
from separatix.recommendation.engine import (
    compute_regression_scores,
    compute_scores,
    make_recommendation,
    make_regression_recommendation,
)


def test_profiler_config_validates_mlp_thresholds() -> None:
    with pytest.raises(ValueError, match="mlp_trigger_skill_threshold"):
        ProfilerConfig(mlp_trigger_skill_threshold=1.5)
    with pytest.raises(ValueError, match="mlp_min_improvement"):
        ProfilerConfig(mlp_min_improvement=-0.1)


def test_default_report_marks_mlp_not_requested() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)

    assert report.metrics["mlp_probes"]["status"] == "not_requested"
    assert report.metrics["mlp_trigger_evidence"]["status"] == "not_requested"
    assert report.metrics["mlp_recommendation_evidence"]["status"] == "not_requested"


def test_good_simple_probe_prevents_mlp_trigger_even_when_requested() -> None:
    X, y = make_blobs(n_samples=160, centers=2, cluster_std=0.5, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0, mlp_probes=True)

    assert report.metrics["mlp_probes"]["status"] == "not_triggered"
    assert report.metrics["mlp_trigger_evidence"]["good_enough"] is True


def test_missing_torch_is_reported_only_after_mlp_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: None)
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0, 1, 0, 1], dtype=int)
    config = ProfilerConfig(mlp_probes=True, random_state=0)
    metrics = {
        "audit": {"class_counts": {"0": 2, "1": 2}},
        "probes": {
            "dummy": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "linear": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "smooth_poly": {
                "balanced_accuracy": 0.5,
                "per_class_recall": [0.5, 0.5],
            },
            "knn": {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]},
            "kernel_approx": {
                "balanced_accuracy": 0.5,
                "per_class_recall": [0.5, 0.5],
            },
        },
    }

    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=config,
        metrics=metrics,
        report_context={
            "warnings": [],
            "skipped_diagnostics": [],
            "densification_events": [],
        },
        class_labels=np.asarray([0, 1]),
    )

    assert result["trigger"]["status"] == "triggered"
    assert result["status"] == "dependency_unavailable"


def test_singlelabel_recommendation_can_be_overridden_by_mlp_evidence() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.7},
            "smooth_poly": {"balanced_accuracy": 0.79},
            "knn": {"balanced_accuracy": 0.78},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 60, "1": 60},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "smooth_poly"},
        "mlp_recommendation_evidence": {
            "recommendation_override": True,
            "override_reason": (
                "The best tested MLP clearly beat the aligned simpler probes."
            ),
            "best_architecture": {"probe_name": "mlp_one_layer_compact"},
        },
    }

    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, decision_path, interpretations = make_recommendation(
        scores, metrics
    )

    assert recommendation == FEEDFORWARD_MLP_RECOMMENDED
    assert any("mlp_one_layer_compact" in item for item in decision_path)
    assert "mlp_recommendation_evidence" in interpretations


def test_regression_recommendation_can_be_overridden_by_mlp_evidence() -> None:
    metrics = {
        "probes": {
            "dummy": {
                "r2_variance_weighted": 0.0,
                "r2_uniform_average": 0.0,
            },
            "linear": {
                "r2_variance_weighted": 0.55,
                "r2_uniform_average": 0.5,
            },
            "smooth_poly": {
                "r2_variance_weighted": 0.68,
                "r2_uniform_average": 0.62,
            },
            "knn": {
                "r2_variance_weighted": 0.66,
                "r2_uniform_average": 0.6,
            },
        },
        "neighborhood": {
            "target_smoothness_score": 0.75,
            "high_discontinuity_fraction": 0.1,
        },
        "topology": {},
        "audit": {"n_samples": 120},
        "mlp_recommendation_evidence": {
            "recommendation_override": True,
            "override_reason": (
                "The best tested MLP clearly beat the aligned simpler regressors."
            ),
            "best_architecture": {"probe_name": "mlp_two_layer_compact"},
        },
    }

    scores = compute_regression_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, decision_path, interpretations = make_regression_recommendation(
        scores, metrics
    )

    assert recommendation == FEEDFORWARD_MLP_REGRESSION_RECOMMENDED
    assert any("mlp_two_layer_compact" in item for item in decision_path)
    assert "mlp_recommendation_evidence" in interpretations


def test_mlp_override_is_skipped_without_held_out_split(monkeypatch) -> None:
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    metrics = {
        "audit": {"class_counts": {"0": 6, "1": 6}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: object())
    monkeypatch.setattr(
        mlp_module, "_resolve_device", lambda torch, device: ("cpu", None)
    )
    monkeypatch.setattr(
        mlp_module,
        "_split_rows",
        lambda *args, **kwargs: (None, "group_split_unavailable"),
    )
    context = {
        "warnings": [],
        "errors": [],
        "skipped_diagnostics": [],
        "densification_events": [],
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, random_state=0),
        metrics=metrics,
        report_context=context,
        class_labels=np.asarray([0, 1]),
    )
    assert result["status"] == "skipped"
    assert "held-out" in result["reason"]
    assert result["recommendation_override"] is False


def test_safe_mlp_evaluation_localizes_architecture_failure() -> None:
    def evaluator(*, estimators, **kwargs):
        name = next(iter(estimators))
        if name == "bad":
            raise RuntimeError("training failed")
        return {name: {"score": 1.0}}

    errors: list[str] = []
    results = _safe_evaluate_models(
        evaluator,
        {"good": object(), "bad": object()},
        errors=errors,
    )
    assert results["good"]["score"] == 1.0
    assert results["bad"]["status"] == "runtime_failed"
    assert len(errors) == 1


def test_all_failed_mlp_architectures_disable_override_cleanly() -> None:
    evidence = _singlelabel_override_evidence(
        {
            "mlp_one_layer_compact": {
                "status": "runtime_failed",
                "error": "training failed",
            }
        },
        {},
        y_true=np.asarray([0, 1]),
        config=ProfilerConfig(mlp_probes=True, budget="fast"),
        groups=None,
    )
    assert evidence["recommendation_override"] is False
    assert evidence["best_architecture"] is None


def test_mlp_override_requires_clear_minimum_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(mlp_module._MLP_BUDGETS["fast"], "bootstrap_repeats", 5)
    y = np.asarray([0, 1] * 20)
    perfect = y.tolist()
    inverse = (1 - y).tolist()
    dummy = np.zeros_like(y).tolist()
    mlp_results = {
        "mlp_one_layer_compact": {
            "balanced_accuracy": 1.0,
            "predictions": perfect,
        }
    }
    weak_comparators = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
        **{
            name: {"balanced_accuracy": 0.0, "predictions": inverse}
            for name in ("linear", "smooth_poly", "knn")
        },
    }
    successful = _singlelabel_override_evidence(
        mlp_results,
        weak_comparators,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )
    tied_comparators = {
        "dummy": {"balanced_accuracy": 0.5, "predictions": dummy},
        **{
            name: {"balanced_accuracy": 1.0, "predictions": perfect}
            for name in ("linear", "smooth_poly", "knn")
        },
    }
    rejected = _singlelabel_override_evidence(
        mlp_results,
        tied_comparators,
        y_true=y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        groups=None,
    )
    assert successful["recommendation_override"] is True
    assert rejected["recommendation_override"] is False


def test_mlp_parameter_cap_and_device_fallbacks() -> None:
    assert _architecture_candidates(100, 10, max_parameters=1) == []

    unavailable = SimpleNamespace(is_available=lambda: False)
    fake_torch = SimpleNamespace(
        cuda=unavailable,
        backends=SimpleNamespace(mps=unavailable),
    )
    cuda_device, cuda_warning = _resolve_device(fake_torch, "cuda")
    mps_device, mps_warning = _resolve_device(fake_torch, "mps")
    assert cuda_device == mps_device == "cpu"
    assert "CUDA" in str(cuda_warning)
    assert "MPS" in str(mps_warning)


def test_aligned_mlp_evaluators_cover_resubstitution_and_held_out_modes() -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = np.asarray([0, 1] * 6)
    Y_multi = np.column_stack([y, 1 - y])
    Y_reg = np.column_stack([X[:, 0], X[:, 1]])
    first = np.arange(0, 6)
    second = np.arange(6, 12)
    splits = [(first, second), (second, first)]

    single = _evaluate_singlelabel_models(
        X,
        y,
        {"dummy": mlp_module.DummyClassifier(strategy="prior")},
        class_labels=np.asarray([0, 1]),
        splits=splits,
        evaluation_mode="cross_validation",
        groups=None,
    )
    multi = _evaluate_multilabel_models(
        X,
        Y_multi,
        {"dummy": mlp_module.MultilabelPriorDummy()},
        label_names=np.asarray(["a", "b"]),
        splits=None,
        evaluation_mode="descriptive_resubstitution",
        groups=None,
    )
    regression = _evaluate_regression_models(
        X,
        Y_reg,
        {"dummy": mlp_module.TargetMeanDummyRegressor()},
        target_names=np.asarray(["one", "two"]),
        splits=splits,
        evaluation_mode="cross_validation",
        groups=None,
    )

    assert single["dummy"]["evaluation_mode"] == "cross_validation"
    assert multi["dummy"]["evaluation_mode"] == "descriptive_resubstitution"
    assert regression["dummy"]["evaluation_mode"] == "cross_validation"


def test_mlp_inner_validation_keeps_every_class_on_both_sides() -> None:
    y = np.repeat(np.arange(5), 4)
    fit_idx, valid_idx = _singlelabel_validation_indices(y, random_state=0)
    assert set(y[fit_idx]) == set(range(5))
    assert set(y[valid_idx]) == set(range(5))


def _patch_mlp_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: object())
    monkeypatch.setattr(
        mlp_module, "_resolve_device", lambda torch, device: ("cpu", None)
    )
    monkeypatch.setitem(mlp_module._MLP_BUDGETS["fast"], "bootstrap_repeats", 5)


def test_singlelabel_mlp_completed_path_disables_override_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_mlp_runtime(monkeypatch)
    X, y = make_blobs(n_samples=60, centers=2, n_features=4, random_state=4)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        if names[0].startswith("mlp_"):
            results = {
                name: {
                    "balanced_accuracy": 1.0,
                    "predictions": y.tolist(),
                }
                for name in names
            }
            results[names[-1]] = {
                "status": "runtime_failed",
                "error": "architecture failed",
            }
            return results
        wrong = (1 - y).tolist()
        return {
            name: {"balanced_accuracy": 0.5, "predictions": wrong} for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "audit": {"class_counts": {"0": 30, "1": 30}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        class_labels=np.asarray([0, 1]),
    )
    assert result["status"] == "completed"
    assert result["recommendation_override"] is False
    assert result["required_comparators_complete"] is False
    assert any(
        item.get("status") == "runtime_failed" for item in result["architectures"]
    )


def test_multilabel_mlp_completed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlp_runtime(monkeypatch)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 4))
    Y = (rng.random((60, 3)) < 0.4).astype(int)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        is_mlp = names[0].startswith("mlp_")
        predictions = Y if is_mlp else np.zeros_like(Y)
        score = 1.0 if is_mlp else 0.0
        return {
            name: {
                "micro_f1": score,
                "macro_f1": score,
                "sample_jaccard": score,
                "predictions": predictions.tolist(),
            }
            for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "audit": {"n_samples": 60},
        "probes": {
            name: {"micro_f1": 0.0, "macro_f1": 0.0, "sample_jaccard": 0.0}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_multilabel_mlp_probes(
        X,
        Y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        label_names=np.asarray(["a", "b", "c"]),
    )
    assert result["status"] == "completed"
    assert result["best_architecture"] is not None


def test_regression_mlp_completed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlp_runtime(monkeypatch)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 4))
    Y = np.column_stack([X[:, 0], X[:, 1]])

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        names = list(estimators)
        is_mlp = names[0].startswith("mlp_")
        predictions = Y if is_mlp else np.zeros_like(Y)
        score = 1.0 if is_mlp else 0.0
        return {
            name: {
                "r2_variance_weighted": score,
                "r2_uniform_average": score,
                "predictions": predictions.tolist(),
            }
            for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "probes": {
            name: {"r2_variance_weighted": 0.0, "r2_uniform_average": 0.0}
            for name in ("dummy", "linear", "smooth_poly", "knn", "kernel_approx")
        },
    }
    result = maybe_run_regression_mlp_probes(
        X,
        Y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context={"warnings": [], "errors": [], "skipped_diagnostics": []},
        target_names=np.asarray(["one", "two"]),
    )
    assert result["status"] == "completed"
    assert result["best_architecture"] is not None


def test_torch_mlp_does_not_mutate_numpy_global_rng() -> None:
    torch = mlp_module._torch_module()
    if torch is None:
        pytest.skip("torch is not installed")
    X, y = make_blobs(n_samples=24, centers=2, n_features=3, random_state=0)
    np.random.seed(123)
    state = np.random.get_state()
    torch.manual_seed(321)
    torch_state = torch.random.get_rng_state().clone()
    estimator = TorchMLPClassifier(
        task="singlelabel",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    estimator.fit(X, y)
    multilabel = TorchMLPClassifier(
        task="multilabel",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    multilabel.fit(X, np.column_stack([y, 1 - y]))
    assert multilabel.predict(X).shape == (24, 2)
    regressor = mlp_module.TorchMLPRegressor(
        task="regression",
        hidden_layer_sizes=(8,),
        epochs=1,
        patience=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    regressor.fit(X, X[:, :2])
    assert regressor.predict(X).shape == (24, 2)
    observed = np.random.random(5)
    np.random.set_state(state)
    expected = np.random.random(5)
    assert np.array_equal(observed, expected)
    observed_torch = torch.rand(5)
    torch.random.set_rng_state(torch_state)
    expected_torch = torch.rand(5)
    assert torch.equal(observed_torch, expected_torch)


def test_torch_module_rejects_incomplete_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubSpec:
        loader = object()

    monkeypatch.setattr(mlp_module, "find_spec", lambda _name: StubSpec())
    monkeypatch.setattr(mlp_module.importlib, "import_module", lambda _name: object())

    assert mlp_module._torch_module() is None
