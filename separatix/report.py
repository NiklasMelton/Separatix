"""Report objects returned by separatix."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np

_TERSE_PRUNED_KEYS = frozenset(
    {
        "candidate_indices",
        "candidate_trigger_counts",
        "local_ambiguity",
        "local_entropy",
        "local_cardinality_std",
        "local_neighbor_hamming_distance",
        "local_neighbor_jaccard",
        "local_normalized_target_distance",
        "local_target_distance",
        "local_label_entropy",
        "per_label_metrics",
        "per_target_metrics",
        "predictions",
        "sample_position_indices",
        "strong_candidate_indices",
        "trigger_names_by_index",
        "indices",
        "row_indices",
    }
)


def _serialize_value(value: Any, *, terse: bool) -> Any:
    """Convert report values without copying fields pruned from terse output."""
    if isinstance(value, dict):
        return {
            key: _serialize_value(item, terse=terse)
            for key, item in value.items()
            if not terse or key not in _TERSE_PRUNED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item, terse=terse) for item in value]
    if isinstance(value, np.ndarray):
        return _serialize_value(value.tolist(), terse=terse)
    if isinstance(value, np.generic):
        return _serialize_value(value.item(), terse=terse)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize_value(getattr(value, field.name), terse=terse)
            for field in fields(value)
        }
    return value


@dataclass
class DiagnosticReport:
    """Structured diagnostic report produced by separatix."""

    recommendation: str
    recommendation_text: str
    confidence: str
    metrics: dict[str, Any]
    scores: dict[str, float | None]
    interpretations: dict[str, str]
    decision_path: list[str]
    warnings: list[str]
    errors: list[str]
    skipped_diagnostics: list[dict[str, Any]]
    preprocessing: dict[str, Any]
    sampling: dict[str, Any]
    densification_events: list[dict[str, Any]]
    class_summary: dict[str, Any]
    grouping: dict[str, Any]
    runtime: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self, *, terse: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            field.name: _serialize_value(getattr(self, field.name), terse=terse)
            for field in fields(self)
        }

    def to_json(self, *, indent: int = 2, terse: bool = True) -> str:
        """Return a JSON string representation of the report."""
        return json.dumps(
            self.to_dict(terse=terse),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
