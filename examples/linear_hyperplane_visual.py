"""Visualize a linear split with the fitted separating line and recommendation."""

from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from separatix import diagnose

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def main() -> None:
    """Generate a linearly separable dataset and visualize its fitted boundary."""
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    class_names = np.array(["class A", "class B"])[y]

    recommendation = diagnose(X, class_names, random_state=0)

    model = LogisticRegression(random_state=0)
    model.fit(X, class_names)

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx = np.linspace(x_min, x_max, 300)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    yy = -(coef[0] * xx + intercept) / coef[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.plot(xx, yy, color="black", linewidth=2.2, label="logistic boundary")

    ax.set_title("Linear synthetic split")
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
