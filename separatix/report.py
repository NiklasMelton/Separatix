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
        "fold_assignments",
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
    """Structured, JSON-serializable result produced by separatix.

    Attributes:
        recommendation: Stable machine-readable recommendation label.
        recommendation_text: Plain-language recommendation and rationale.
        confidence: Coarse evidence-quality level.
        metrics: Raw and derived diagnostic-family evidence.
        scores: Normalized summary scores where applicable.
        interpretations: Human-readable descriptions of report evidence.
        decision_path: Ordered recommendation gates and decisions.
        warnings: Non-fatal warnings raised during the run.
        errors: Captured diagnostic errors.
        skipped_diagnostics: Diagnostics omitted with structured reasons.
        preprocessing: Probe and geometry preprocessing metadata.
        sampling: Sampling metadata by diagnostic family.
        densification_events: Dense conversion, sampling, and skip events.
        class_summary: Classification, multilabel, or regression target summary.
        grouping: Group-validation and group-split metadata.
        runtime: Runtime measurements.
        config: Effective profiler configuration.
    """

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
        """Return a JSON-serializable dictionary representation.

        Args:
            terse: Prune large row-level arrays before copying the report.

        Returns:
            A dictionary containing only standard JSON-compatible values.
        """
        return {
            field.name: _serialize_value(getattr(self, field.name), terse=terse)
            for field in fields(self)
        }

    def to_json(self, *, indent: int = 2, terse: bool = True) -> str:
        """Return a standards-compliant JSON representation.

        Args:
            indent: Number of spaces used for indentation.
            terse: Prune large row-level arrays before serialization.

        Returns:
            JSON text with non-finite numeric values represented as ``null``.
        """
        return json.dumps(
            self.to_dict(terse=terse),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
