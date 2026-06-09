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
