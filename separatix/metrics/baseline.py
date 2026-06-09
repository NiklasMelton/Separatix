"""Baseline metric helpers."""

from __future__ import annotations

from typing import Any


def summarize_probe_family(probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize high-level probe behavior."""
    available = {
        name: result["balanced_accuracy"]
        for name, result in probes.items()
        if "balanced_accuracy" in result
    }
    best_name = (
        max(available.items(), key=lambda item: item[1])[0] if available else None
    )
    return {
        "best_probe": best_name,
        "best_probe_score": available.get(best_name) if best_name is not None else None,
    }
