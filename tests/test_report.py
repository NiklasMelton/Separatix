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
