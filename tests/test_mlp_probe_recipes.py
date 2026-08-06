"""Recipe metadata emitted by optional MLP probes."""

import numpy as np

from separatix import ProfilerConfig, make_probe_estimator
from separatix.models import mlp as mlp_module
from separatix.models.mlp import (
    maybe_run_multilabel_mlp_probes,
    maybe_run_regression_mlp_probes,
    maybe_run_singlelabel_mlp_probes,
)


def _runtime_patch(monkeypatch) -> None:
    """Avoid importing or training Torch while exercising recipe assembly."""
    monkeypatch.setattr(mlp_module, "_torch_module", lambda: object())
    monkeypatch.setattr(
        mlp_module, "_resolve_device", lambda torch, device: ("cpu", None)
    )
    monkeypatch.setitem(mlp_module._MLP_BUDGETS["fast"], "bootstrap_repeats", 5)


def _context() -> dict[str, object]:
    return {"warnings": [], "errors": [], "skipped_diagnostics": []}


def test_singlelabel_mlp_results_include_exact_architecture_and_comparator_recipes(
    monkeypatch,
) -> None:
    """Recipes reflect live estimator parameters and integration roles."""
    _runtime_patch(monkeypatch)
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 4))
    y = np.asarray([0, 1] * 30)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        del evaluator, errors
        names = list(estimators)
        values = kwargs["y"]
        is_mlp = names[0].startswith("mlp_")
        predictions = values if is_mlp else (1 - values)
        return {
            name: {
                "balanced_accuracy": 1.0 if is_mlp else 0.5,
                "predictions": predictions.tolist(),
            }
            for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    low_skill = {
        "audit": {"class_counts": {"0": 30, "1": 30}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in mlp_module._REQUIRED_MLP_COMPARATORS
        },
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=low_skill,
        report_context=_context(),
        class_labels=np.asarray([0, 1]),
    )

    assert result["status"] == "completed"
    assert result["best_architecture"]["probe_recipe_id"]
    for architecture in result["architectures"]:
        recipe = architecture["probe_recipe"]
        assert architecture["probe_recipe_status"] == {
            "status": "available",
            "reason": None,
        }
        assert recipe["probe"]["family"] == "mlp"
        assert recipe["probe"]["role"] == "mlp_architecture"
        assert recipe["estimator"]["params"]["hidden_layer_sizes"] == list(
            architecture["hidden_layer_sizes"]
        )
        assert recipe["training_policy"]["task"] == "singlelabel"
        restored = make_probe_estimator(recipe, version_policy="ignore")
        assert tuple(restored.hidden_layer_sizes) == tuple(
            architecture["hidden_layer_sizes"]
        )
        assert not hasattr(restored, "model_")
        if result["best_architecture"]["probe_name"] == (
            "mlp_" + architecture["label"]
        ):
            assert result["best_architecture"]["probe_recipe_id"] == recipe["recipe_id"]
    for name, comparator in result["aligned_comparators"].items():
        assert comparator["probe_recipe_status"] == {
            "status": "available",
            "reason": None,
        }
        assert comparator["probe_recipe"]["probe"]["role"] == ("mlp_aligned_comparator")
        assert comparator["probe_recipe"]["probe"]["name"] == name


def test_unconstructed_smooth_comparator_has_unavailable_recipe_status(
    monkeypatch,
) -> None:
    """Memory-gated comparators expose a stable unavailable status."""
    _runtime_patch(monkeypatch)
    monkeypatch.setattr(
        mlp_module,
        "_choose_sketch_components",
        lambda *args, **kwargs: None,
    )
    X = np.zeros((60, 400), dtype=float)
    y = np.asarray([0, 1] * 30)

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        del evaluator, errors
        values = kwargs["y"]
        return {
            name: {"balanced_accuracy": 0.5, "predictions": values.tolist()}
            for name in estimators
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    metrics = {
        "audit": {"class_counts": {"0": 30, "1": 30}},
        "probes": {
            name: {"balanced_accuracy": 0.5, "per_class_recall": [0.5, 0.5]}
            for name in mlp_module._REQUIRED_MLP_COMPARATORS
        },
    }
    result = maybe_run_singlelabel_mlp_probes(
        X,
        y,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=metrics,
        report_context=_context(),
        class_labels=np.asarray([0, 1]),
    )

    smooth = result["aligned_comparators"]["smooth_poly"]
    assert smooth["probe_recipe"] is None
    assert smooth["probe_recipe_status"] == {
        "status": "unavailable",
        "reason": "estimator was not constructed",
    }


def test_multilabel_and_regression_architecture_recipes_have_target_modes(
    monkeypatch,
) -> None:
    """Both non-single-label paths carry the same recipe contract."""
    _runtime_patch(monkeypatch)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 3))
    Y_multi = (rng.random((48, 2)) > 0.5).astype(int)
    Y_reg = np.column_stack([X[:, 0], X[:, 1]])

    def fake_safe(evaluator, estimators, *, errors, **kwargs):
        del evaluator, errors
        names = list(estimators)
        is_mlp = names[0].startswith("mlp_")
        if "Y" in kwargs:
            values = np.asarray(kwargs["Y"])
            predictions = values if is_mlp else np.zeros_like(values)
            if values.dtype.kind in "iu":
                score = 1.0 if is_mlp else 0.0
                metrics = {
                    "micro_f1": score,
                    "macro_f1": score,
                    "sample_jaccard": score,
                }
            else:
                score = 1.0 if is_mlp else 0.0
                metrics = {
                    "r2_variance_weighted": score,
                    "r2_uniform_average": score,
                }
        else:  # pragma: no cover - both paths above use Y
            raise AssertionError("expected Y evaluator argument")
        return {
            name: {**metrics, "predictions": predictions.tolist()} for name in names
        }

    monkeypatch.setattr(mlp_module, "_safe_evaluate_models", fake_safe)
    multi_metrics = {
        "audit": {"n_samples": 48},
        "probes": {
            name: {"micro_f1": 0.0, "macro_f1": 0.0, "sample_jaccard": 0.0}
            for name in mlp_module._REQUIRED_MLP_COMPARATORS
        },
    }
    multi = maybe_run_multilabel_mlp_probes(
        X,
        Y_multi,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=multi_metrics,
        report_context=_context(),
        label_names=np.asarray(["a", "b"]),
    )
    reg_metrics = {
        "probes": {
            name: {"r2_variance_weighted": 0.0, "r2_uniform_average": 0.0}
            for name in mlp_module._REQUIRED_MLP_COMPARATORS
        },
    }
    regression = maybe_run_regression_mlp_probes(
        X,
        Y_reg,
        config=ProfilerConfig(mlp_probes=True, budget="fast", random_state=0),
        metrics=reg_metrics,
        report_context=_context(),
        target_names=np.asarray(["x", "y"]),
    )

    assert all(
        item["probe_recipe"]["probe"]["target_mode"] == "multilabel"
        for item in multi["architectures"]
    )
    assert all(
        item["probe_recipe"]["probe"]["target_mode"] == "regression"
        for item in regression["architectures"]
    )
