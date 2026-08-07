"""Recipes and availability metadata for built-in model probes."""

import numpy as np
from sklearn.datasets import make_classification

from separatix import diagnose, make_probe_estimator

_CORE_PROBES = ("dummy", "linear", "knn", "smooth_poly", "kernel_approx")


def _assert_available_recipe(
    result: dict[str, object], *, target_mode: str, n_outputs: int
) -> None:
    """Check the stable envelope shared by constructed probe recipes."""
    assert result["probe_recipe_status"] == {"status": "available", "reason": None}
    recipe = result["probe_recipe"]
    assert isinstance(recipe, dict)
    assert {
        "schema",
        "schema_version",
        "recipe_id",
        "probe",
        "implementation",
        "input_contract",
        "estimator",
        "training_policy",
        "created_with",
    } <= recipe.keys()
    assert recipe["schema_version"] == 1
    contract = recipe["input_contract"]
    assert isinstance(contract, dict)
    assert contract["n_outputs"] == n_outputs
    assert contract["n_features"] > 0
    assert contract["resolution_n_samples"] > 0
    probe = recipe["probe"]
    assert isinstance(probe, dict)
    assert probe["target_mode"] == target_mode
    assert probe["role"] == "core_probe"
    restored = make_probe_estimator(recipe, version_policy="ignore")
    assert hasattr(restored, "fit")
    assert not hasattr(restored, "n_features_in_")


def test_singlelabel_core_probe_recipes_and_knn_policy() -> None:
    X, y = make_classification(
        n_samples=72,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="standard",
        topology="off",
        random_state=7,
    )
    probes = report.metrics["probes"]
    for name in _CORE_PROBES:
        result = probes[name]
        _assert_available_recipe(result, target_mode="singlelabel", n_outputs=1)
    knn_policy = probes["knn"]["probe_recipe"]["training_policy"]
    assert knn_policy["evaluation_random_state"] == 7
    assert "knn_n_neighbors" in knn_policy["scoring_time_estimator_adjustments"]
    assert (
        knn_policy["scoring_time_estimator_adjustments"]["knn_n_neighbors"]["source"]
        == "separatix.models.scoring._prepared_estimator"
    )


def test_multilabel_core_probe_recipes_record_target_width() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(72, 4))
    Y = np.column_stack(
        [
            (X[:, 0] > 0).astype(np.int8),
            (X[:, 1] > 0).astype(np.int8),
        ]
    )
    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        topology="off",
        random_state=0,
    )
    probes = report.metrics["probes"]
    for name in ("dummy", "linear", "knn", "smooth_poly"):
        _assert_available_recipe(probes[name], target_mode="multilabel", n_outputs=2)
    kernel = probes["kernel_approx"]
    assert kernel["probe_recipe"] is None
    assert kernel["probe_recipe_status"]["status"] == "unavailable"
    assert kernel["probe_recipe_status"]["reason"] == (
        "kernel probe disabled for this budget"
    )


def test_regression_core_probe_recipes_record_multioutput_contract() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(72, 3))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] ** 2])
    report = diagnose(
        X,
        Y,
        target_mode="regression",
        return_report=True,
        budget="fast",
        topology="off",
        random_state=2,
    )
    probes = report.metrics["probes"]
    for name in ("dummy", "linear", "knn", "smooth_poly"):
        _assert_available_recipe(probes[name], target_mode="regression", n_outputs=2)
    assert probes["kernel_approx"]["probe_recipe"] is None
