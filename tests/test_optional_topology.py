from sklearn.datasets import make_blobs

from separatix import diagnose


def test_topology_off_runs() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, topology="off", random_state=0)
    assert report.metrics["topology"]["mode"] == "off"


def test_topology_persistent_without_dependency_is_safe() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    report = diagnose(X, y, return_report=True, topology="persistent", random_state=0)
    assert "mode" in report.metrics["topology"]
