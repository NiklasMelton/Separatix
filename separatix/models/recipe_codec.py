"""Safe, allowlisted encoding and decoding of probe estimators.

Probe recipes are deliberately data-only.  This module converts the small set
of estimators used by :mod:`separatix` to a recursive JSON-compatible
description and reconstructs a fresh (unfitted) estimator from that
description.  Recipe data never controls an import path; every supported
component is registered in the explicit allowlist below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from separatix.exceptions import ProbeRecipeError
from separatix.models.scoring import MultilabelPriorDummy, TargetMeanDummyRegressor

# Keep keys stable.  They are identifiers in persisted data, not import paths
# supplied by a caller.  The values are imported eagerly only for the light
# sklearn and separatix components.  The optional torch components are handled
# by lazy factories below so importing separatix never imports torch.
_ALLOWLIST: dict[str, type[BaseEstimator]] = {
    "sklearn.dummy.DummyClassifier": DummyClassifier,
    "sklearn.kernel_approximation.PolynomialCountSketch": PolynomialCountSketch,
    "sklearn.kernel_approximation.RBFSampler": RBFSampler,
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    "sklearn.linear_model.Ridge": Ridge,
    "sklearn.linear_model.SGDClassifier": SGDClassifier,
    "sklearn.multiclass.OneVsRestClassifier": OneVsRestClassifier,
    "sklearn.neighbors.KNeighborsClassifier": KNeighborsClassifier,
    "sklearn.neighbors.KNeighborsRegressor": KNeighborsRegressor,
    "sklearn.pipeline.Pipeline": Pipeline,
    "sklearn.preprocessing.PolynomialFeatures": PolynomialFeatures,
    "sklearn.preprocessing.StandardScaler": StandardScaler,
    "separatix.models.scoring.MultilabelPriorDummy": MultilabelPriorDummy,
    "separatix.models.scoring.TargetMeanDummyRegressor": TargetMeanDummyRegressor,
}

_TORCH_KEYS = {
    "separatix.models.mlp.TorchMLPClassifier",
    "separatix.models.mlp.TorchMLPRegressor",
}


def _class_key(estimator: Any) -> str | None:
    """Return an allowlist key for ``estimator`` without importing torch."""
    for key, estimator_type in _ALLOWLIST.items():
        if type(estimator) is estimator_type:
            return key
    module = type(estimator).__module__
    qualname = type(estimator).__qualname__
    candidate = f"{module}.{qualname}"
    if candidate in _TORCH_KEYS:
        return candidate
    return None


def _primitive(value: Any) -> Any:
    """Convert a scalar numpy value to its Python equivalent."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _encode_value(value: Any) -> Any:
    """Recursively encode estimator parameters as JSON-compatible values."""
    value = _primitive(value)
    if isinstance(value, BaseEstimator):
        return encode_estimator(value)
    if isinstance(value, np.ndarray):
        return [_encode_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProbeRecipeError("Estimator parameters must contain finite numbers.")
        return value
    # An unknown estimator-like object must not be serialized as an arbitrary
    # class path.  Failing here also makes malformed recipes fail early.
    if hasattr(value, "get_params"):
        raise ProbeRecipeError(
            "Estimator component is not in the separatix probe allowlist: "
            f"{type(value).__module__}.{type(value).__qualname__}."
        )
    raise ProbeRecipeError(
        "Estimator parameter is not JSON-compatible: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def encode_estimator(estimator: Any) -> dict[str, Any]:
    """Encode an allowlisted unfitted or fitted estimator recursively.

    Args:
        estimator: sklearn-compatible estimator to encode.

    Returns:
        A JSON-compatible estimator specification with ``kind``, ``key`` and
        recursively encoded ``params`` fields.

    Raises:
        ProbeRecipeError: If a component is not explicitly allowlisted or has
            a non-serializable parameter.
    """
    key = _class_key(estimator)
    if key is None:
        raise ProbeRecipeError(
            "Estimator is not in the separatix probe allowlist: "
            f"{type(estimator).__module__}.{type(estimator).__qualname__}."
        )
    get_params = getattr(estimator, "get_params", None)
    if not callable(get_params):
        raise ProbeRecipeError(f"Estimator {key!r} does not expose get_params().")
    try:
        params = get_params(deep=False)
    except Exception as exc:  # pragma: no cover - defensive third-party guard
        raise ProbeRecipeError(f"Could not inspect estimator {key!r}: {exc}") from exc
    if not isinstance(params, Mapping):
        raise ProbeRecipeError(f"Estimator {key!r} returned invalid parameters.")
    return {
        "kind": "estimator",
        "key": key,
        "params": {str(name): _encode_value(value) for name, value in params.items()},
    }


def _decode_value(value: Any) -> Any:
    """Recursively decode estimator markers nested in a recipe."""
    if isinstance(value, Mapping):
        if value.get("kind") == "estimator":
            return decode_estimator(value)
        return {str(key): _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_value(item) for item in value)
    return value


def _lazy_torch_factory(key: str) -> Callable[..., BaseEstimator]:
    """Return a decoder that imports the optional MLP module only on demand."""
    def factory(**params: Any) -> BaseEstimator:
        try:
            from separatix.models.mlp import TorchMLPClassifier, TorchMLPRegressor
        except Exception as exc:  # pragma: no cover - depends on optional module
            raise ProbeRecipeError(
                "Torch MLP probe recipes require the optional torch extra."
            ) from exc
        estimator_type: type[BaseEstimator]
        if key.endswith("TorchMLPClassifier"):
            estimator_type = TorchMLPClassifier
        else:
            estimator_type = TorchMLPRegressor
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        try:
            return estimator_type(**params)
        except Exception as exc:
            raise ProbeRecipeError(
                f"Could not construct allowlisted estimator {key!r}: {exc}"
            ) from exc

    return factory


def decode_estimator(spec: Mapping[str, Any]) -> BaseEstimator:
    """Construct a fresh estimator from an allowlisted recursive spec.

    Args:
        spec: Mapping produced by :func:`encode_estimator`.

    Returns:
        An unfitted estimator instance.

    Raises:
        ProbeRecipeError: If the specification is malformed or names a
            component outside the explicit allowlist.
    """
    if not isinstance(spec, Mapping):
        raise ProbeRecipeError("Estimator specification must be a mapping.")
    if spec.get("kind") != "estimator":
        raise ProbeRecipeError("Estimator specification has unsupported kind.")
    key = spec.get("key")
    if not isinstance(key, str):
        raise ProbeRecipeError("Estimator specification key must be a string.")
    params_raw = spec.get("params", {})
    if not isinstance(params_raw, Mapping):
        raise ProbeRecipeError(f"Estimator parameters for {key!r} must be a mapping.")
    if key in _TORCH_KEYS:
        factory = _lazy_torch_factory(key)
    else:
        estimator_type = _ALLOWLIST.get(key)
        if estimator_type is None:
            raise ProbeRecipeError(f"Unsupported estimator component key: {key!r}.")
        factory = estimator_type
    params = {str(name): _decode_value(value) for name, value in params_raw.items()}
    # Pipeline steps are encoded as JSON lists.  Keep their order and names,
    # while giving sklearn the tuple shape expected by older releases too.
    if key == "sklearn.pipeline.Pipeline" and "steps" in params:
        steps = params["steps"]
        if not isinstance(steps, list):
            raise ProbeRecipeError("Pipeline estimator steps must be a list.")
        normalized_steps: list[tuple[str, BaseEstimator]] = []
        for step in steps:
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                raise ProbeRecipeError(
                    "Pipeline steps must be [name, estimator] pairs."
                )
            name, component = step
            if not isinstance(name, str) or not isinstance(component, BaseEstimator):
                raise ProbeRecipeError("Pipeline steps must contain named estimators.")
            normalized_steps.append((name, component))
        params["steps"] = normalized_steps
    try:
        return factory(**params)
    except ProbeRecipeError:
        raise
    except TypeError as exc:
        # ``transform_input`` was added to sklearn Pipeline after the minimum
        # supported sklearn version.  It is safe to omit that known optional
        # parameter when decoding a recipe made on a newer sklearn release.
        if key == "sklearn.pipeline.Pipeline" and "transform_input" in params:
            compatible = dict(params)
            compatible.pop("transform_input")
            try:
                return factory(**compatible)
            except Exception as retry_exc:
                raise ProbeRecipeError(
                    f"Could not construct allowlisted estimator {key!r}: {retry_exc}"
                ) from retry_exc
        raise ProbeRecipeError(
            f"Could not construct allowlisted estimator {key!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise ProbeRecipeError(
            f"Could not construct allowlisted estimator {key!r}: {exc}"
        ) from exc


__all__ = ["decode_estimator", "encode_estimator"]
