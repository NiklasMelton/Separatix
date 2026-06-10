"""Visualize a curved decision boundary with the recommendation overlaid."""

from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from separatix import diagnose

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def main() -> None:
    """Generate a nonlinear dataset and plot a fitted curved decision boundary."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.5, 2.5, size=(400, 2))
    signal = X[:, 1] - 0.18 * (X[:, 0] ** 2) + 0.08 * X[:, 0]
    y = (signal + 0.10 * rng.normal(size=400) > 0.0).astype(int)
    class_names = np.array(["class A", "class B"])[y]

    recommendation = diagnose(X, class_names, random_state=0)

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    model.fit(X, class_names)

    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    scores = model.decision_function(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.contour(
        grid_x,
        grid_y,
        scores,
        levels=[0.0],
        colors="black",
        linewidths=2.2,
    )

    ax.set_title("Curvilinear synthetic split")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(*scatter.legend_elements(), title="class", loc="upper right")
    ax.text(
        0.02,
        0.02,
        textwrap.fill(recommendation, width=48),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
