"""Focused tests for versioned probe recipes and the safe estimator factory."""

import copy
import json
import warnings

import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from separatix import (
    ProbeRecipe,
    ProbeRecipeCompatibilityError,
    ProbeRecipeError,
    UnsupportedProbeRecipeVersion,
    build_probe_recipe,
    make_probe_estimator,
)
from separatix import recipes as recipe_module
from separatix.models.scoring import MultilabelPriorDummy, TargetMeanDummyRegressor


def _recipe(estimator):
    return build_probe_recipe(
        estimator,
        probe_name="test_probe",
        family="linear",
        target_mode="singlelabel",
        role="probe",
        input_contract={"n_features": 2, "sparse": False},
        training_policy={"cv": 3},
        implementation_key="tests.recipe",
    )


def test_pipeline_and_nested_one_vs_rest_round_trip_unfitted() -> None:
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                OneVsRestClassifier(LogisticRegression(max_iter=17, random_state=3)),
            ),
        ]
    )
    recipe = _recipe(estimator)
    assert recipe["schema"] == "separatix.probe_recipe"
    assert recipe["schema_version"] == 1
    assert recipe["created_with"].keys() == {
        "separatix",
        "python",
        "numpy",
        "scipy",
        "scikit_learn",
        "torch",
    }
    restored = make_probe_estimator(recipe, version_policy="ignore")
    assert isinstance(restored, Pipeline)
    assert [name for name, _ in restored.steps] == ["scale", "clf"]
    assert isinstance(restored.named_steps["clf"], OneVsRestClassifier)
    assert not hasattr(restored, "classes_")


def test_recipe_id_is_structural_and_excludes_runtime_provenance() -> None:
    first = _recipe(Ridge(alpha=1.5))
    second = copy.deepcopy(first)
    second["created_with"] = {key: "different" for key in second["created_with"]}
    assert second["recipe_id"] == first["recipe_id"]
    assert ProbeRecipe.from_dict(first).recipe_id == first["recipe_id"]
    assert make_probe_estimator(second, version_policy="ignore") is not None


def test_created_with_is_resolved_from_the_active_runtime(monkeypatch) -> None:
    versions = {
        "separatix": "0.1.runtime",
        "numpy": "2.runtime",
        "scipy": "1.runtime",
        "scikit-learn": "1.9.runtime",
    }

    def fake_version(distribution: str) -> str:
        if distribution == "torch":
            raise recipe_module.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(recipe_module.metadata, "version", fake_version)
    monkeypatch.setattr(
        recipe_module.platform, "python_version", lambda: "3.12.runtime"
    )

    assert _recipe(Ridge())["created_with"] == {
        "separatix": "0.1.runtime",
        "python": "3.12.runtime",
        "numpy": "2.runtime",
        "scipy": "1.runtime",
        "scikit_learn": "1.9.runtime",
        "torch": None,
    }


def test_version_policy_warn_and_error() -> None:
    recipe = _recipe(Ridge())
    recipe["created_with"]["numpy"] = "0.0.invalid"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(make_probe_estimator(recipe, version_policy="warn"), Ridge)
    assert any("compatibility mismatch" in str(item.message) for item in caught)
    with pytest.raises(ProbeRecipeCompatibilityError):
        make_probe_estimator(recipe, version_policy="error")
    assert isinstance(make_probe_estimator(recipe, version_policy="ignore"), Ridge)


def test_schema_and_allowlist_errors_are_not_ignored() -> None:
    recipe = _recipe(Ridge())
    unsupported = copy.deepcopy(recipe)
    unsupported["schema_version"] = 99
    with pytest.raises(UnsupportedProbeRecipeVersion):
        make_probe_estimator(unsupported, version_policy="ignore")

    unknown = copy.deepcopy(recipe)
    unknown["estimator"]["key"] = "evil.module.Class"
    # The id is intentionally stale; schema validation must fail before any
    # class lookup or arbitrary import is attempted.
    with pytest.raises(ProbeRecipeError):
        make_probe_estimator(unknown, version_policy="ignore")


@pytest.mark.parametrize(
    "estimator", [MultilabelPriorDummy(), TargetMeanDummyRegressor()]
)
def test_separatix_dummy_estimators_round_trip(estimator) -> None:
    recipe = _recipe(estimator)
    restored = make_probe_estimator(
        json.loads(json.dumps(recipe)), version_policy="ignore"
    )
    assert type(restored) is type(estimator)
