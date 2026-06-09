"""Optional topology diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from separatix.config import ProfilerConfig
from separatix.densify import ensure_dense_or_sample


def compute_topology_diagnostics(
    X: Any,
    y: np.ndarray,
    boundary: dict[str, Any],
    geometry: dict[str, Any],
    *,
    config: ProfilerConfig,
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Run optional persistent topology diagnostics when available and feasible."""
    if config.topology == "off":
        return {"mode": "off", "skipped_reason": "topology disabled"}
    if config.topology == "graph":
        return {"mode": "graph", "skipped_reason": "persistent topology not requested"}

    indices = np.asarray(boundary.get("candidate_indices", []), dtype=int)
    if indices.shape[0] < 30:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "too few boundary candidates"}
        )
        return {
            "mode": config.topology,
            "skipped_reason": "too few boundary candidates",
        }
    if indices.shape[0] > 2000:
        return {
            "mode": config.topology,
            "skipped_reason": "too many boundary candidates",
        }
    if (
        geometry.get("distance_concentration_proxy") is not None
        and geometry["distance_concentration_proxy"] < 0.05
    ):
        return {
            "mode": config.topology,
            "skipped_reason": "geometry reliability too low",
        }

    try:
        from ripser import ripser
    except ImportError:
        report_context.setdefault("skipped_diagnostics", []).append(
            {"name": "persistent_topology", "reason": "ripser is not installed"}
        )
        return {"mode": config.topology, "skipped_reason": "ripser is not installed"}

    dense_info = ensure_dense_or_sample(
        X[indices],
        y[indices],
        reason="persistent_topology",
        config=config,
        report_context=report_context,
    )
    if dense_info["skipped"]:
        return {
            "mode": config.topology,
            "skipped_reason": "dense conversion unavailable",
        }

    dgms = ripser(dense_info["X"], maxdim=1)["dgms"]
    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h1_lifetimes = h1[:, 1] - h1[:, 0] if h1.size else np.array([])
    h0_lifetimes = h0[:, 1] - h0[:, 0] if h0.size else np.array([])
    finite_h0 = h0_lifetimes[np.isfinite(h0_lifetimes)]
    finite_h1 = h1_lifetimes[np.isfinite(h1_lifetimes)]
    return {
        "mode": config.topology,
        "h0_persistence_count": int(
            np.sum(finite_h0 > np.median(finite_h0) if finite_h0.size else 0)
        ),
        "h1_persistence_count": int(
            np.sum(finite_h1 > np.median(finite_h1) if finite_h1.size else 0)
        ),
        "total_h0_persistence": float(np.sum(finite_h0)) if finite_h0.size else 0.0,
        "total_h1_persistence": float(np.sum(finite_h1)) if finite_h1.size else 0.0,
        "max_h1_persistence": float(np.max(finite_h1)) if finite_h1.size else 0.0,
        "persistence_entropy": float(
            -np.sum(
                (finite_h1 / np.sum(finite_h1))
                * np.log(np.clip(finite_h1 / np.sum(finite_h1), 1e-12, None))
            )
        )
        if finite_h1.size and np.sum(finite_h1) > 0
        else 0.0,
    }
