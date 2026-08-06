"""Versioned, portable recipes for separatix probe estimators.

Recipes are intentionally transparent dictionaries.  They contain the probe
metadata needed by integrations, a recursive allowlisted estimator
specification, runtime provenance, and a deterministic structural identifier.
The :func:`make_probe_estimator` factory always returns a new unfitted
estimator; it never imports a class named by recipe data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import warnings
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import numpy as np

from separatix.exceptions import (
    ProbeRecipeCompatibilityError,
    ProbeRecipeError,
    UnsupportedProbeRecipeVersion,
)
from separatix.models.recipe_codec import decode_estimator, encode_estimator

_SCHEMA = "separatix.probe_recipe"
_SCHEMA_VERSION = 1
_VERSION_POLICIES = frozenset({"warn", "error", "ignore"})
_RUNTIME_DISTRIBUTIONS = {
    "separatix": "separatix",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "torch": "torch",
}
_CREATED_WITH_KEYS = (
    "separatix",
    "python",
    "numpy",
    "scipy",
    "scikit_learn",
    "torch",
)


def _runtime_versions() -> dict[str, str | None]:
    """Return runtime versions without importing optional torch."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for key, distribution in _RUNTIME_DISTRIBUTIONS.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
        except Exception:
            # Metadata lookup should not make recipe creation unusable in an
            # unusual embedded environment.  ``None`` is explicit provenance.
            versions[key] = None
    return {key: versions.get(key) for key in _CREATED_WITH_KEYS}


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Normalize metadata to JSON-compatible values and reject unsafe values."""
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return [_json_safe(item, path=f"{path}[]") for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ProbeRecipeError(f"{path} must contain only finite numbers.")
        return value
    raise ProbeRecipeError(
        f"{path} contains a value that is not JSON-compatible: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _canonical_structural_payload(payload: Mapping[str, Any]) -> str:
    """Serialize structural recipe fields in a deterministic canonical form."""
    structural = {
        key: value
        for key, value in payload.items()
        if key not in {"created_with", "recipe_id"}
    }
    try:
        normalized = _json_safe(structural, path="recipe")
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProbeRecipeError(
            f"Recipe cannot be canonically serialized: {exc}"
        ) from exc


def _recipe_id(payload: Mapping[str, Any]) -> str:
    """Compute the stable SHA-256 identifier for structural recipe contents."""
    canonical = _canonical_structural_payload(payload).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_recipe_payload(
    payload: Mapping[str, Any], *, verify_id: bool = True
) -> None:
    """Validate schema and component shape before compatibility checks."""
    if not isinstance(payload, Mapping):
        raise ProbeRecipeError("Probe recipe must be a mapping.")
    schema = payload.get("schema")
    if schema != _SCHEMA:
        raise ProbeRecipeError(
            f"Unsupported probe recipe schema: expected {_SCHEMA!r}, got {schema!r}."
        )
    schema_version = payload.get("schema_version")
    if schema_version != _SCHEMA_VERSION or isinstance(schema_version, bool):
        raise UnsupportedProbeRecipeVersion(
            f"Unsupported probe recipe schema version: {schema_version!r}."
        )
    required = {
        "schema",
        "schema_version",
        "recipe_id",
        "probe",
        "implementation",
        "input_contract",
        "estimator",
        "training_policy",
        "created_with",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ProbeRecipeError(f"Probe recipe is missing required fields: {missing}.")
    recipe_id = payload.get("recipe_id")
    if not isinstance(recipe_id, str) or not recipe_id:
        raise ProbeRecipeError("Probe recipe recipe_id must be a non-empty string.")

    probe = payload.get("probe")
    if not isinstance(probe, Mapping):
        raise ProbeRecipeError("Probe recipe probe metadata must be a mapping.")
    for key in ("name", "family", "target_mode", "role"):
        value = probe.get(key)
        if not isinstance(value, str) or not value:
            raise ProbeRecipeError(
                f"Probe metadata field {key!r} must be a non-empty string."
            )
    if "variant" in probe and probe["variant"] is not None and not isinstance(
        probe["variant"], str
    ):
        raise ProbeRecipeError("Probe metadata variant must be a string or null.")

    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ProbeRecipeError("Probe recipe implementation must be a mapping.")
    implementation_key = implementation.get("key")
    implementation_version = implementation.get("version")
    if not isinstance(implementation_key, str) or not implementation_key:
        raise ProbeRecipeError("Implementation key must be a non-empty string.")
    if (
        not isinstance(implementation_version, int)
        or isinstance(implementation_version, bool)
        or implementation_version <= 0
    ):
        raise ProbeRecipeError("Implementation version must be a positive integer.")

    for key in ("input_contract", "training_policy", "created_with"):
        if not isinstance(payload.get(key), Mapping):
            raise ProbeRecipeError(f"Probe recipe {key} must be a mapping.")
    estimator = payload.get("estimator")
    if not isinstance(estimator, Mapping):
        raise ProbeRecipeError("Probe recipe estimator must be a mapping.")
    if estimator.get("kind") != "estimator":
        raise ProbeRecipeError("Probe recipe estimator has unsupported kind.")
    if not isinstance(estimator.get("key"), str) or not estimator.get("key"):
        raise ProbeRecipeError("Probe recipe estimator key must be a non-empty string.")
    if not isinstance(estimator.get("params", {}), Mapping):
        raise ProbeRecipeError("Probe recipe estimator params must be a mapping.")

    created_with = payload.get("created_with")
    assert isinstance(created_with, Mapping)  # narrowed above
    missing_runtime = sorted(set(_CREATED_WITH_KEYS).difference(created_with.keys()))
    if missing_runtime:
        raise ProbeRecipeError(
            f"created_with is missing required runtime fields: {missing_runtime}."
        )
    for key in _CREATED_WITH_KEYS:
        value = created_with.get(key)
        if value is not None and not isinstance(value, str):
            raise ProbeRecipeError(
                f"created_with field {key!r} must be a string or null."
            )

    if verify_id and recipe_id != _recipe_id(payload):
        raise ProbeRecipeError(
            "Probe recipe recipe_id does not match its structural contents."
        )


def _compatibility_mismatches(
    created_with: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return runtime provenance fields that differ from the current process."""
    current = _runtime_versions()
    return {
        key: (created_with.get(key), current.get(key))
        for key in _CREATED_WITH_KEYS
        if created_with.get(key) != current.get(key)
    }


def _check_version_policy(payload: Mapping[str, Any], version_policy: str) -> None:
    """Apply warn/error/ignore behavior to runtime provenance mismatches."""
    if version_policy not in _VERSION_POLICIES:
        raise ProbeRecipeError(
            "version_policy must be one of 'warn', 'error', or 'ignore'."
        )
    if version_policy == "ignore":
        return
    mismatches = _compatibility_mismatches(payload["created_with"])
    if not mismatches:
        return
    details = ", ".join(
        f"{key}: recipe={expected!r}, runtime={actual!r}"
        for key, (expected, actual) in mismatches.items()
    )
    message = f"Probe recipe runtime version compatibility mismatch ({details})."
    if version_policy == "error":
        raise ProbeRecipeCompatibilityError(message)
    warnings.warn(message, UserWarning, stacklevel=3)


class ProbeRecipe(dict[str, Any]):
    """Dictionary-backed versioned recipe with validation and serialization helpers.

    ``ProbeRecipe`` accepts a complete recipe mapping.  For convenience it can
    also be constructed with estimator metadata keyword arguments, matching
    :func:`build_probe_recipe`.  It remains a normal mapping so integrations
    can pass it directly to JSON encoders and :func:`make_probe_estimator`.
    """

    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        estimator: Any | None = None,
        probe_name: str | None = None,
        family: str | None = None,
        target_mode: str | None = None,
        role: str | None = None,
        input_contract: Mapping[str, Any] | None = None,
        variant: str | None = None,
        training_policy: Mapping[str, Any] | None = None,
        implementation_key: str | None = None,
        implementation_version: int = 1,
    ) -> None:
        """Create and validate a recipe mapping or semantic recipe arguments."""
        semantic = estimator is not None or any(
            value is not None
            for value in (probe_name, family, target_mode, role, input_contract)
        )
        if payload is not None and semantic:
            raise ProbeRecipeError(
                "Pass either a complete recipe payload or semantic recipe "
                "arguments, not both."
            )
        if payload is None:
            if estimator is None or any(
                value is None
                for value in (probe_name, family, target_mode, role, input_contract)
            ):
                raise ProbeRecipeError(
                    "Semantic ProbeRecipe construction requires estimator, "
                    "probe metadata, and input_contract."
                )
            assert probe_name is not None
            assert family is not None
            assert target_mode is not None
            assert role is not None
            assert input_contract is not None
            payload = build_probe_recipe(
                estimator,
                probe_name=probe_name,
                family=family,
                target_mode=target_mode,
                role=role,
                input_contract=input_contract,
                variant=variant,
                training_policy=training_policy,
                implementation_key=implementation_key,
                implementation_version=implementation_version,
            )
        normalized = _json_safe(dict(payload), path="recipe")
        if not isinstance(normalized, dict):  # pragma: no cover - dict(payload) guard
            raise ProbeRecipeError("Probe recipe payload must be a mapping.")
        _validate_recipe_payload(normalized)
        super().__init__(normalized)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeRecipe:
        """Create a validated :class:`ProbeRecipe` from a dictionary."""
        return cls(payload)

    @classmethod
    def from_json(cls, value: str) -> ProbeRecipe:
        """Create a validated recipe from JSON text."""
        try:
            payload = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ProbeRecipeError(f"Invalid probe recipe JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ProbeRecipeError("Probe recipe JSON must contain an object.")
        return cls(payload)

    @classmethod
    def from_estimator(cls, estimator: Any, **kwargs: Any) -> ProbeRecipe:
        """Build a recipe from an allowlisted estimator and metadata."""
        return cls(build_probe_recipe(estimator, **kwargs))

    @property
    def recipe_id(self) -> str:
        """Return the deterministic structural recipe identifier."""
        return str(self["recipe_id"])

    @property
    def schema(self) -> str:
        """Return the recipe schema identifier."""
        return str(self["schema"])

    @property
    def schema_version(self) -> int:
        """Return the supported schema version."""
        return int(self["schema_version"])

    @property
    def probe(self) -> dict[str, Any]:
        """Return probe metadata as an independent dictionary."""
        return copy.deepcopy(self["probe"])

    @property
    def implementation(self) -> dict[str, Any]:
        """Return implementation metadata as an independent dictionary."""
        return copy.deepcopy(self["implementation"])

    @property
    def estimator_spec(self) -> dict[str, Any]:
        """Return the recursive estimator specification as a dictionary."""
        return copy.deepcopy(self["estimator"])

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible recipe dictionary."""
        return copy.deepcopy(dict(self))

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_dict` used by some serialization integrations."""
        return self.to_dict()

    def to_json(self, *, indent: int | None = None) -> str:
        """Return canonical JSON text for this recipe."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            allow_nan=False,
        )


def build_probe_recipe(
    estimator: Any,
    *,
    probe_name: str,
    family: str,
    target_mode: str,
    role: str,
    input_contract: Mapping[str, Any],
    variant: str | None = None,
    training_policy: Mapping[str, Any] | None = None,
    implementation_key: str | None = None,
    implementation_version: int = 1,
) -> dict[str, Any]:
    """Build a versioned JSON-compatible recipe for a probe estimator.

    Args:
        estimator: Allowlisted, sklearn-compatible estimator instance.
        probe_name: Stable name used by the calling diagnostic.
        family: Probe family (for example ``"linear"`` or ``"kernel"``).
        target_mode: Target mode used by the probe.
        role: Integration role, such as ``"probe"`` or ``"comparator"``.
        input_contract: JSON-compatible feature/target contract metadata.
        variant: Optional family variant identifier.
        training_policy: Optional JSON-compatible training metadata.
        implementation_key: Stable implementation identifier; defaults to
            ``"separatix.probe"``.
        implementation_version: Positive implementation version integer.

    Returns:
        A plain dictionary containing schema, provenance, estimator, and
        deterministic identifier fields.
    """
    for name, value in {
        "probe_name": probe_name,
        "family": family,
        "target_mode": target_mode,
        "role": role,
    }.items():
        if not isinstance(value, str) or not value:
            raise ProbeRecipeError(f"{name} must be a non-empty string.")
    if not isinstance(input_contract, Mapping):
        raise ProbeRecipeError("input_contract must be a mapping.")
    if training_policy is not None and not isinstance(training_policy, Mapping):
        raise ProbeRecipeError("training_policy must be a mapping when provided.")
    if implementation_key is not None and (
        not isinstance(implementation_key, str) or not implementation_key
    ):
        raise ProbeRecipeError("implementation_key must be a non-empty string.")
    if (
        not isinstance(implementation_version, int)
        or isinstance(implementation_version, bool)
        or implementation_version <= 0
    ):
        raise ProbeRecipeError("implementation_version must be a positive integer.")
    if variant is not None and not isinstance(variant, str):
        raise ProbeRecipeError("variant must be a string or None.")

    estimator_spec = encode_estimator(estimator)
    payload: dict[str, Any] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "recipe_id": "",
        "probe": {
            "name": probe_name,
            "family": family,
            "target_mode": target_mode,
            "role": role,
            "variant": variant,
        },
        "implementation": {
            "key": implementation_key or "separatix.probe",
            "version": implementation_version,
        },
        "input_contract": _json_safe(dict(input_contract), path="input_contract"),
        "estimator": estimator_spec,
        "training_policy": _json_safe(
            {} if training_policy is None else dict(training_policy),
            path="training_policy",
        ),
        "created_with": _runtime_versions(),
    }
    payload["recipe_id"] = _recipe_id(payload)
    _validate_recipe_payload(payload)
    return payload


def make_probe_estimator(
    recipe: Mapping[str, Any] | ProbeRecipe | str,
    *,
    version_policy: str = "warn",
) -> Any:
    """Construct a fresh unfitted estimator from a validated probe recipe.

    Args:
        recipe: Probe recipe mapping, :class:`ProbeRecipe`, or JSON text.
        version_policy: Runtime provenance policy: ``"warn"`` (default),
            ``"error"``, or ``"ignore"``.

    Returns:
        An unfitted estimator reconstructed through the explicit allowlist.

    Raises:
        ProbeRecipeError: For malformed schema or unsupported components.
        UnsupportedProbeRecipeVersion: For schema versions other than v1.
        ProbeRecipeCompatibilityError: When ``version_policy="error"`` and
            runtime provenance differs.
    """
    if isinstance(recipe, str):
        try:
            loaded = json.loads(recipe)
        except (TypeError, ValueError) as exc:
            raise ProbeRecipeError(f"Invalid probe recipe JSON: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ProbeRecipeError("Probe recipe JSON must contain an object.")
        payload: Mapping[str, Any] = loaded
    elif isinstance(recipe, Mapping):
        payload = recipe
    else:
        raise ProbeRecipeError("recipe must be a mapping, ProbeRecipe, or JSON string.")
    normalized = _json_safe(dict(payload), path="recipe")
    if not isinstance(normalized, dict):  # pragma: no cover
        raise ProbeRecipeError("Probe recipe payload must be a mapping.")
    _validate_recipe_payload(normalized)
    _check_version_policy(normalized, version_policy)
    try:
        return decode_estimator(normalized["estimator"])
    except ProbeRecipeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive codec guard
        raise ProbeRecipeError(f"Could not decode probe estimator: {exc}") from exc


__all__ = [
    "ProbeRecipe",
    "build_probe_recipe",
    "make_probe_estimator",
]
