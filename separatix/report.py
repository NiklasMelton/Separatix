"""Report objects returned by separatix."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON string representation of the report."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
