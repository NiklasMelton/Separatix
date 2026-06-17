"""Report objects returned by separatix."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

_TERSE_PRUNED_KEYS = frozenset(
    {
        "candidate_indices",
        "local_ambiguity",
        "local_entropy",
        "local_label_entropy",
        "per_label_metrics",
        "predictions",
    }
)


def _prune_verbose_items(value: Any) -> Any:
    """Recursively remove verbose fields from a serialized report payload."""
    if isinstance(value, dict):
        return {
            key: _prune_verbose_items(item)
            for key, item in value.items()
            if key not in _TERSE_PRUNED_KEYS
        }
    if isinstance(value, list):
        return [_prune_verbose_items(item) for item in value]
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
    runtime: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self, *, terse: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        payload = asdict(self)
        return _prune_verbose_items(payload) if terse else payload

    def to_json(self, *, indent: int = 2, terse: bool = True) -> str:
        """Return a JSON string representation of the report."""
        return json.dumps(self.to_dict(terse=terse), indent=indent, sort_keys=True)
