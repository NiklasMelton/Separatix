import numpy as np
from sklearn.datasets import make_blobs

from separatix import diagnose
from separatix.constants import (
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY,
    LINEAR_LIKELY_SUFFICIENT,
)


def test_multiclass_blobs() -> None:
    X, y = make_blobs(n_samples=180, centers=3, n_features=6, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.class_summary["n_classes"] == 3
    assert report.recommendation in {
        LINEAR_LIKELY_SUFFICIENT,
        "smooth_nonlinear_recommended",
    }


def test_multiclass_string_labels() -> None:
    X, y = make_blobs(n_samples=150, centers=3, n_features=4, random_state=3)
    labels = np.array(["cat", "dog", "fox"])[y]
    report = diagnose(X, labels, return_report=True, random_state=0)
    assert report.class_summary["classes"] == ["cat", "dog", "fox"]


def test_multiclass_imbalance_runs() -> None:
    centers = [
        [-4.0, -4.0, -4.0, -4.0],
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0, 4.0],
    ]
    X, y = make_blobs(
        n_samples=[80, 40, 12],
        centers=centers,
        n_features=4,
        random_state=1,
    )
    report = diagnose(X, y, return_report=True, random_state=0)
    assert (
        report.recommendation != INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
        or report.confidence in {"low", "medium", "high"}
    )
