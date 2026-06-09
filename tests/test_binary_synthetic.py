import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons

from separatix import diagnose
from separatix.constants import (
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)


def test_linear_blobs_recommend_linearish() -> None:
    X, y = make_blobs(n_samples=180, centers=2, cluster_std=1.2, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.recommendation in {
        LINEAR_LIKELY_SUFFICIENT,
        SMOOTH_NONLINEAR_RECOMMENDED,
    }


def test_moons_recommend_nonlinear() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.recommendation in {
        SMOOTH_NONLINEAR_RECOMMENDED,
        KERNEL_OR_LOCAL_RECOMMENDED,
    }


def test_circles_recommend_local_or_kernel() -> None:
    X, y = make_circles(n_samples=220, noise=0.08, factor=0.4, random_state=0)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.recommendation in {
        KERNEL_OR_LOCAL_RECOMMENDED,
        SMOOTH_NONLINEAR_RECOMMENDED,
    }


def test_random_labels_are_bottleneck_or_inconclusive() -> None:
    X, y = make_classification(
        n_samples=180, n_features=12, n_informative=10, random_state=0
    )
    y = np.random.default_rng(0).permutation(y)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.recommendation in {
        FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
        "inconclusive",
        "linear_likely_sufficient",
    }
