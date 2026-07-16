from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from separatix import diagnose
from separatix.config import ProfilerConfig
from separatix.metrics import topology as topology_module
from tests.test_synthetic_model_family_matrix import (
    _binary_smooth,
    _binary_topological,
    _pad_noise,
)


def test_topology_off_runs() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, topology="off", random_state=0)
    assert report.metrics["topology"]["mode"] == "off"


def test_topology_persistent_without_dependency_is_safe() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, topology="persistent", random_state=0)
    assert "mode" in report.metrics["topology"]


def test_topology_strength_is_not_saturated_for_smooth_curve() -> None:
    X, y = _binary_smooth(2)
    report = diagnose(X, y, return_report=True, topology="persistent", random_state=0)
    if "topology_strength" not in report.metrics["topology"]:
        pytest.skip("persistent topology strength is unavailable in this environment")
    assert report.metrics["topology"]["topology_strength"] < 0.2
    assert report.scores["topology_score"] < 0.2


def test_topology_strength_is_higher_for_spiral_than_smooth_curve() -> None:
    X_smooth, y_smooth = _binary_smooth(2)
    smooth = diagnose(
        X_smooth,
        y_smooth,
        return_report=True,
        topology="persistent",
        random_state=0,
    )
    X_spiral, y_spiral = _binary_topological(20)
    spiral = diagnose(
        X_spiral,
        y_spiral,
        return_report=True,
        topology="persistent",
        random_state=0,
    )
    if "topology_strength" not in smooth.metrics["topology"]:
        pytest.skip("persistent topology strength is unavailable in this environment")
    assert (
        spiral.metrics["topology"]["topology_strength"]
        > smooth.metrics["topology"]["topology_strength"]
    )
    assert (
        spiral.metrics["topology"]["relative_h1_persistence"]
        > smooth.metrics["topology"]["relative_h1_persistence"]
    )


def test_topology_strength_does_not_saturate_from_sample_count_alone() -> None:
    X_base, y_base = _binary_smooth(2)
    X = _pad_noise(X_base, 5, seed=123, noise_scale=0.05)
    report = diagnose(
        X,
        y_base,
        return_report=True,
        topology="persistent",
        random_state=0,
    )
    if "topology_strength" not in report.metrics["topology"]:
        pytest.skip("persistent topology strength is unavailable in this environment")
    assert report.metrics["boundary"]["boundary_sample_size"] >= 100
    assert report.metrics["topology"]["topology_strength"] < 0.4


def test_persistent_topology_runtime_failure_is_localized(monkeypatch) -> None:
    monkeypatch.setattr(
        topology_module,
        "_summarize_persistent_topology",
        lambda X: (_ for _ in ()).throw(RuntimeError("ripser failed")),
    )
    context = {
        "warnings": [],
        "errors": [],
        "skipped_diagnostics": [],
        "densification_events": [],
    }
    result = topology_module._compute_topology_object(
        np.arange(120, dtype=float).reshape(40, 3),
        reason="runtime_test",
        config=ProfilerConfig(topology="persistent"),
        report_context=context,
        sampling={"sampled": False},
    )
    assert result["skipped_reason"] == "persistent topology runtime failure"
    assert context["errors"]
    assert context["skipped_diagnostics"][0]["severity"] == "caution"
