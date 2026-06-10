"""High-dimensional synthetic example with a linear separating hyperplane."""

from __future__ import annotations

import os

import numpy as np

from separatix import diagnose

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def main() -> None:
    """Create a high-dimensional linear split and print the diagnostic report."""
    rng = np.random.default_rng(0)
    n_samples = 800
    n_features = 20

    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    margin = X @ weights + 0.35 * rng.normal(size=n_samples)
    y = np.where(margin > 0.0, "positive side", "negative side")

    report = diagnose(X, y, return_report=True, random_state=0)

    print("High-dimensional linear hyperplane example")
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
