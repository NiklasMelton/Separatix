"""OpenML-backed multilabel baseline example using yeast v4."""

from __future__ import annotations

import ssl
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from separatix import diagnose


@contextmanager
def _openml_https_context() -> object:
    """Temporarily prefer certifi CA roots when available."""
    original = ssl._create_default_https_context
    try:
        import certifi
    except ImportError:
        yield
        return
    ssl._create_default_https_context = lambda *args, **kwargs: (  # noqa: E731
        ssl.create_default_context(cafile=certifi.where())
    )
    try:
        yield
    finally:
        ssl._create_default_https_context = original


with _openml_https_context():
    dataset = fetch_openml(
        name="yeast",
        version=4,
        as_frame=False,
        parser="liac-arff",
        data_home=str(Path(tempfile.gettempdir()) / "separatix_sklearn_data"),
    )

X = np.asarray(dataset.data, dtype=float)
Y = (np.asarray(dataset.target) == "TRUE").astype(int)

report = diagnose(
    X,
    Y,
    target_mode="multilabel",
    return_report=True,
    budget="standard",
    topology="off",
    random_state=0,
)

evidence = report.metrics["multilabel_recommendation_evidence"]
comparison = evidence["family_comparisons"]["local_kernel_vs_smooth"]

print(f"dataset: OpenML yeast v4 {X.shape=} {Y.shape=}")
print(f"recommendation: {report.recommendation} ({report.confidence})")
print("decision path:")
for step in report.decision_path:
    print(f"- {step}")
print("signal metrics beating dummy:", evidence["signal_metrics_beating_dummy"])
print("local_kernel_vs_smooth clear metrics:", comparison["clear_metrics"])
