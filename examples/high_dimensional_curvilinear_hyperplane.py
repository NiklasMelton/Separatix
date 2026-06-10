"""High-dimensional synthetic example with a curved class boundary."""

from __future__ import annotations

import os

import numpy as np

from separatix import diagnose

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def main() -> None:
    """Create a high-dimensional nonlinear split and print the diagnostic report."""
    rng = np.random.default_rng(1)
    n_samples = 1000
    n_features = 20

    X = rng.normal(size=(n_samples, n_features))
    nonlinear_signal = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] ** 2 - 0.3 * X[:, 3]
    noisy_signal = nonlinear_signal + 0.15 * rng.normal(size=n_samples)
    y = np.where(noisy_signal > 0.35, "inside curve", "outside curve")

    report = diagnose(X, y, return_report=True, random_state=0)

    print("High-dimensional curvilinear boundary example")
    print(report.recommendation_text)
    print("Decision path:")
    for step in report.decision_path:
        print(f"- {step}")
    print("Key scores:")
    for name in (
        "signal_score",
        "linearity_score",
        "nonlinearity_score",
        "reliability_score",
    ):
        print(f"- {name}: {report.scores[name]}")


if __name__ == "__main__":
    main()
