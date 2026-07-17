from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_moons

from separatix import diagnose
from separatix.constants import (
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
    INCONCLUSIVE,
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)
from separatix.report import DiagnosticReport

DatasetBuilder = Callable[[int], tuple[np.ndarray, np.ndarray]]


def format_evidence_table(report: DiagnosticReport) -> str:
    """Return a compact recommendation-evidence table for calibration failures."""
    evidence = report.metrics["recommendation_evidence"]
    rows = [
        (
            row["family"],
            row["name"],
            f"{row['score']:.3f}",
            f"{row['standard_error']:.3f}",
        )
        for row in evidence["probe_table"]
    ]
    rendered = ["family | probe | score | se"]
    rendered.extend(" | ".join(row) for row in rows)
    rendered.append(f"raw_best={evidence['raw_best_family']}")
    rendered.append(f"recommended={evidence['recommended_family']}")
    rendered.append(f"recommendation={report.recommendation}")
    return "\n".join(rendered)


def _pad_noise(
    X: np.ndarray,
    target_dim: int,
    *,
    seed: int,
    scale: float = 0.2,
) -> np.ndarray:
    if X.shape[1] >= target_dim:
        return X[:, :target_dim]
    # Constant padding keeps the designed geometry invariant under the
    # production fold-local StandardScaler while still exercising extra columns.
    padding = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=float)
    return np.hstack([X, padding])


def _linear(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=260,
        centers=[(-2.5, -2.5), (2.5, 2.5)],
        cluster_std=0.9,
        random_state=seed,
    )
    return _pad_noise(X, 6, seed=seed), y


def _smooth(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.5, 2.5, size=(360, 2))
    signal = X[:, 1] - 0.65 * (X[:, 0] ** 2) + 0.08 * X[:, 0]
    y = (signal + 0.08 * rng.normal(size=X.shape[0]) > 0.0).astype(int)
    return X, y


def _local_kernel(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=360, noise=0.16, random_state=seed)
    return _pad_noise(X, 8, seed=seed + 200), y


def _topological(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.3, 4.0 * np.pi, 180)
    radius = t
    first = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
    second = np.column_stack([-radius * np.cos(t), -radius * np.sin(t)])
    X = np.vstack([first, second]) + rng.normal(scale=0.2, size=(360, 2))
    y = np.array([0] * 180 + [1] * 180)
    return _pad_noise(X, 8, seed=seed + 300), y


def _random_labels(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=260,
        n_features=10,
        n_informative=8,
        random_state=seed,
    )
    return X, np.random.default_rng(seed).permutation(y)


def _permuted_blob_labels(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=600,
        centers=[(-2.5, -2.5), (2.5, 2.5)],
        cluster_std=0.75,
        random_state=seed,
    )
    return X, np.random.default_rng(seed).permutation(y)


@pytest.mark.parametrize("budget", ["fast", "standard"])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (_linear, {LINEAR_LIKELY_SUFFICIENT}),
        (_smooth, {SMOOTH_NONLINEAR_RECOMMENDED}),
        (_local_kernel, {KERNEL_OR_LOCAL_RECOMMENDED}),
        (_topological, {KERNEL_OR_LOCAL_RECOMMENDED}),
        (
            _random_labels,
            {
                FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
                INCONCLUSIVE,
            },
        ),
        (
            _permuted_blob_labels,
            {
                FEATURE_OR_LABEL_BOTTLENECK_LIKELY,
                INCONCLUSIVE,
            },
        ),
    ],
)
def test_recommendation_calibration_archetypes(
    builder: DatasetBuilder,
    expected: set[str],
    seed: int,
    budget: str,
) -> None:
    X, y = builder(seed)
    report = diagnose(
        X,
        y,
        return_report=True,
        random_state=seed,
        budget=budget,
        topology="off",
    )
    assert report.recommendation in expected, format_evidence_table(report)
    assert report.metrics["recommendation_evidence"]["probe_table"]
