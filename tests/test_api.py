from sklearn.datasets import make_blobs

from separatix import ComplexityProfiler, DiagnosticReport, diagnose


def test_diagnose_returns_string() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    result = diagnose(X, y, random_state=0)
    assert isinstance(result, str)
    assert "Recommendation:" in result


def test_diagnose_return_report() -> None:
    X, y = make_blobs(n_samples=120, centers=3, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert isinstance(report, DiagnosticReport)
    assert report.recommendation


def test_profiler_fit_sets_report() -> None:
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    profiler = ComplexityProfiler(random_state=0).fit(X, y)
    assert profiler.report_ is not None
    assert profiler.recommendation() == profiler.report_.recommendation_text


def test_deterministic_report() -> None:
    X, y = make_blobs(n_samples=150, centers=3, random_state=3)
    report_a = diagnose(X, y, return_report=True, random_state=7)
    report_b = diagnose(X, y, return_report=True, random_state=7)
    assert report_a.to_dict()["recommendation"] == report_b.to_dict()["recommendation"]
    assert report_a.scores == report_b.scores
