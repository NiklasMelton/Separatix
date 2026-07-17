import json

import numpy as np
from sklearn.datasets import make_blobs

from separatix import diagnose


def test_report_serialization() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    as_dict = report.to_dict()
    assert "recommendation" in as_dict
    assert "scores" in as_dict
    assert "decision_path" in as_dict
    assert isinstance(report.to_json(), str)


def test_report_serializes_recommendation_evidence() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    evidence = report.to_dict()["metrics"]["recommendation_evidence"]
    assert evidence["selection_rule"]
    assert evidence["probe_table"]
    assert "raw_best_family" in evidence
    assert "recommended_family" in evidence
    assert "linear" in evidence["families"]
    assert "quality_flags" in evidence
    assert "graph_fragmentation_bootstrap_repeats" in evidence["geometry"]
    assert "recommendation_evidence" in report.interpretations


def test_report_serialization_is_terse_by_default() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)

    terse = report.to_dict()
    full = report.to_dict(terse=False)

    neighborhood_terse = terse["metrics"]["neighborhood"]
    neighborhood_full = full["metrics"]["neighborhood"]
    boundary_terse = terse["metrics"]["boundary"]
    boundary_full = full["metrics"]["boundary"]
    linear_terse = terse["metrics"]["probes"]["linear"]
    linear_full = full["metrics"]["probes"]["linear"]

    assert "local_entropy" not in neighborhood_terse
    assert "local_ambiguity" not in neighborhood_terse
    assert "candidate_indices" not in boundary_terse
    assert "sample_position_indices" not in boundary_terse
    assert "predictions" not in linear_terse
    assert "local_entropy" in neighborhood_full
    assert "local_ambiguity" in neighborhood_full
    assert "candidate_indices" in boundary_full
    assert "sample_position_indices" in boundary_full
    assert "predictions" in linear_full


def test_report_json_full_mode_preserves_verbose_fields() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)

    terse_json = report.to_json()
    full_json = report.to_json(terse=False)

    assert '"local_entropy"' not in terse_json
    assert '"local_ambiguity"' not in terse_json
    assert '"candidate_indices"' not in terse_json
    assert '"sample_position_indices"' not in terse_json
    assert '"predictions"' not in terse_json
    assert '"local_entropy"' in full_json
    assert '"local_ambiguity"' in full_json
    assert '"candidate_indices"' in full_json
    assert '"sample_position_indices"' in full_json
    assert '"predictions"' in full_json


def test_skipped_diagnostics_appear() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, topology="persistent", random_state=0)
    assert isinstance(report.skipped_diagnostics, list)


def test_fast_budget_skips_persistent_topology() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, budget="fast", random_state=0)
    assert report.metrics["topology"]["skipped_reason"] in {
        "persistent topology disabled for this budget",
        "too few boundary candidates",
    }


def test_probe_stability_fields_are_present() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    linear = report.metrics["probes"]["linear"]
    assert "stability_repeats" in linear
    assert "stability_balanced_accuracy_std" in linear


def test_smooth_probe_skip_is_serialized() -> None:
    X, y = make_blobs(n_samples=2000, centers=2, n_features=500, random_state=0)
    report = diagnose(
        X,
        y,
        return_report=True,
        random_state=0,
        topology="off",
        max_dense_mb=1,
    )
    smooth = report.metrics["probes"]["smooth_poly"]
    assert smooth["skipped_reason"]
    assert any(
        item["name"] == "smooth_nonlinear_probe" for item in report.skipped_diagnostics
    )
    assert isinstance(report.to_json(), str)


def test_report_json_replaces_nonfinite_values_with_null() -> None:
    X = np.ones((20, 1), dtype=float)
    y = np.asarray([0, 1] * 10)
    report = diagnose(X, y, return_report=True, topology="off", random_state=0)
    report.metrics["injected_nonfinite"] = {
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "zero_dimensional_array": np.asarray(float("-inf")),
        "tuple_value": (1.0, float("nan")),
    }
    payload = report.to_json(terse=False)
    parsed = json.loads(payload)
    assert parsed["metrics"]["injected_nonfinite"] == {
        "nan": None,
        "positive_infinity": None,
        "zero_dimensional_array": None,
        "tuple_value": [1.0, None],
    }
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_constant_one_feature_geometry_is_finite_and_degenerate() -> None:
    X = np.ones((30, 1), dtype=float)
    y = np.asarray([0, 1] * 15)
    report = diagnose(X, y, return_report=True, topology="off", random_state=0)
    geometry = report.metrics["geometry"]
    assert geometry["effective_rank_estimate"] == 0.0
    assert geometry["intrinsic_dimension_proxy"] == 0.0
    assert geometry["degenerate_geometry"] is True
    json.loads(report.to_json())
