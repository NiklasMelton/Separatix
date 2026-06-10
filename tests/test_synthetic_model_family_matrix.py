from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from separatix import diagnose
from separatix.constants import (
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)
from separatix.report import DiagnosticReport

DatasetBuilder = Callable[[int], tuple[np.ndarray, np.ndarray]]


def _pad_noise(
    X: np.ndarray,
    target_dim: int,
    *,
    seed: int,
    noise_scale: float = 0.3,
) -> np.ndarray:
    if X.shape[1] >= target_dim:
        return X[:, :target_dim]
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=noise_scale, size=(X.shape[0], target_dim - X.shape[1]))
    return np.hstack([X, noise])


def _binary_linear(dim: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=360,
        centers=[(-2.0, -2.0), (2.0, 2.0)],
        cluster_std=0.8,
        random_state=0,
    )
    return _pad_noise(X, dim, seed=dim), y


def _multiclass_linear(dim: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=450,
        centers=[(-4.0, -1.0), (0.0, 3.0), (4.0, -1.0)],
        cluster_std=0.9,
        random_state=1,
    )
    return _pad_noise(X, dim, seed=dim + 10), y


def _binary_smooth(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.5, 2.5, size=(500, 2))
    signal = X[:, 1] - 0.18 * (X[:, 0] ** 2) + 0.08 * X[:, 0]
    y = (signal + 0.10 * rng.normal(size=500) > 0.0).astype(int)
    return _pad_noise(X, dim, seed=dim + 20), y


def _multiclass_smooth(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(25)
    X = rng.uniform(-3.0, 3.0, size=(700, 2))
    curve = 0.22 * X[:, 0] ** 2 - 0.2 * X[:, 0]
    signed_distance = X[:, 1] - curve + 0.08 * rng.normal(size=X.shape[0])
    y = np.digitize(signed_distance, [-0.75, 0.75])
    return _pad_noise(X, dim, seed=dim + 30, noise_scale=0.25), y


def _binary_kernel(dim: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(
        n_samples=500,
        noise=0.18,
        random_state=0,
    )
    return _pad_noise(X, dim, seed=dim + 40), y


def _multiclass_kernel(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    n_samples = 750
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    y = rng.integers(0, 3, size=n_samples)
    radii = np.choose(y, [1.0, 2.0, 3.2]) + rng.normal(scale=0.12, size=n_samples)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _pad_noise(X, dim, seed=dim + 50), y


def _binary_topological(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    n_samples = 500
    t = np.linspace(0.3, 4.0 * np.pi, n_samples // 2)
    radius = t
    x0 = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
    x1 = np.column_stack([-radius * np.cos(t), -radius * np.sin(t)])
    X = np.vstack([x0, x1]) + rng.normal(scale=0.2, size=(n_samples, 2))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return _pad_noise(X, dim, seed=dim + 60), y


def _multiclass_topological(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(14)
    points_per_class = 180
    t = np.linspace(0.4, 4.0 * np.pi, points_per_class)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for class_id, phase in enumerate([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]):
        radius = t
        X_class = np.column_stack(
            [radius * np.cos(t + phase), radius * np.sin(t + phase)]
        )
        X_class = X_class + rng.normal(scale=0.18, size=(points_per_class, 2))
        X_parts.append(X_class)
        y_parts.append(np.full(points_per_class, class_id))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return _pad_noise(X, dim, seed=dim + 70), y


def _local_kernel_margin(report: DiagnosticReport) -> float | None:
    probes = report.metrics["probes"]
    smooth = probes["smooth_poly"].get("balanced_accuracy")
    local_scores = [
        probes[name].get("balanced_accuracy")
        for name in ("knn", "kernel_approx")
        if probes[name].get("balanced_accuracy") is not None
    ]
    if smooth is None or not local_scores:
        return None
    return float(max(local_scores) - smooth)


def _assert_supporting_evidence(
    name: str, expected: str, report: DiagnosticReport
) -> None:
    probes = report.metrics["probes"]
    topology_strength = report.scores.get("topology_score") or 0.0
    if expected == LINEAR_LIKELY_SUFFICIENT:
        assert report.scores["linearity_score"] is not None
        assert report.scores["linearity_score"] >= 0.93
    elif expected == SMOOTH_NONLINEAR_RECOMMENDED:
        margin = _local_kernel_margin(report)
        assert probes["smooth_poly"].get("balanced_accuracy") is not None
        assert margin is None or margin <= 0.02
        assert topology_strength < 0.4
    elif "topological" in name:
        assert topology_strength >= 0.4
    elif expected == KERNEL_OR_LOCAL_RECOMMENDED:
        margin = _local_kernel_margin(report)
        assert margin is not None
        assert margin >= 0.02 or topology_strength >= 0.4


@pytest.mark.parametrize(
    ("name", "builder", "dim", "expected"),
    [
        ("binary_linear_2d", _binary_linear, 2, LINEAR_LIKELY_SUFFICIENT),
        ("binary_linear_10d", _binary_linear, 10, LINEAR_LIKELY_SUFFICIENT),
        ("binary_smooth_2d", _binary_smooth, 2, SMOOTH_NONLINEAR_RECOMMENDED),
        ("binary_kernel_10d", _binary_kernel, 10, KERNEL_OR_LOCAL_RECOMMENDED),
        (
            "binary_topological_20d",
            _binary_topological,
            20,
            KERNEL_OR_LOCAL_RECOMMENDED,
        ),
    ],
)
def test_binary_synthetic_model_family_matrix(
    name: str,
    builder: DatasetBuilder,
    dim: int,
    expected: str,
) -> None:
    X, y = builder(dim)
    topology: Literal["auto", "persistent"] = (
        "persistent" if "kernel" in name or "topological" in name else "auto"
    )
    report = diagnose(X, y, return_report=True, random_state=0, topology=topology)
    assert not isinstance(report, str)
    assert report.recommendation == expected
    _assert_supporting_evidence(name, expected, report)


@pytest.mark.parametrize(
    ("name", "builder", "dim", "expected"),
    [
        ("multiclass_linear_5d", _multiclass_linear, 5, LINEAR_LIKELY_SUFFICIENT),
        ("multiclass_linear_20d", _multiclass_linear, 20, LINEAR_LIKELY_SUFFICIENT),
        ("multiclass_smooth_5d", _multiclass_smooth, 5, SMOOTH_NONLINEAR_RECOMMENDED),
        ("multiclass_kernel_20d", _multiclass_kernel, 20, KERNEL_OR_LOCAL_RECOMMENDED),
        (
            "multiclass_topological_5d",
            _multiclass_topological,
            5,
            KERNEL_OR_LOCAL_RECOMMENDED,
        ),
    ],
)
def test_multiclass_synthetic_model_family_matrix(
    name: str,
    builder: DatasetBuilder,
    dim: int,
    expected: str,
) -> None:
    X, y = builder(dim)
    topology: Literal["auto", "persistent"] = (
        "persistent" if "kernel" in name or "topological" in name else "auto"
    )
    report = diagnose(X, y, return_report=True, random_state=0, topology=topology)
    assert not isinstance(report, str)
    assert report.recommendation == expected
    _assert_supporting_evidence(name, expected, report)
