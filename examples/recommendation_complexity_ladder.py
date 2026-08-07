"""Plot Separatix recommendations against a designed synthetic complexity ladder."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "separatix-matplotlib"),
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons

from separatix import DiagnosticReport, diagnose
from separatix.constants import (
    FEATURE_OR_TARGET_BOTTLENECK_LIKELY,
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED,
    INCONCLUSIVE,
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY,
    KERNEL_OR_LOCAL_RECOMMENDED,
    LINEAR_LIKELY_SUFFICIENT,
    SMOOTH_NONLINEAR_RECOMMENDED,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "img"
    / "separatix_recommendation_complexity_ladder.png"
)

RECOMMENDATION_LEVELS = {
    LINEAR_LIKELY_SUFFICIENT: 0,
    SMOOTH_NONLINEAR_RECOMMENDED: 1,
    KERNEL_OR_LOCAL_RECOMMENDED: 2,
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED: 3,
    FEATURE_OR_TARGET_BOTTLENECK_LIKELY: 4,
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY: 4,
    INCONCLUSIVE: 4,
}

LEVEL_LABELS = {
    0: "linear",
    1: "smooth nonlinear",
    2: "kernel or local",
    3: "high-capacity / partitioning",
    4: "bottleneck / unreliable / inconclusive",
}

LEVEL_COLORS = {
    0: "#1b9e77",
    1: "#4daf4a",
    2: "#377eb8",
    3: "#e41a1c",
    4: "#b35806",
}

ANNOTATION_OFFSETS = {
    "two moons": (-7, 9, "right", "bottom"),
    "radial rings": (7, -11, "left", "top"),
    "fragmented islands 3x3": (-7, 9, "right", "bottom"),
    "fragmented islands 5x5": (0, -11, "center", "top"),
    "random labels": (-7, 7, "right", "bottom"),
}


DatasetBuilder = Callable[[], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SyntheticCase:
    """Synthetic dataset case used in the complexity ladder plot."""

    name: str
    designed_complexity: float
    builder: DatasetBuilder


def _clean_linear() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=420,
        centers=[(-2.5, -2.5), (2.5, 2.5)],
        cluster_std=0.7,
        random_state=0,
    )


def _overlapping_linear() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=500,
        centers=[(-0.8, 0.0), (0.8, 0.0)],
        cluster_std=1.15,
        random_state=1,
    )


def _smooth_quadratic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.5, 2.5, size=(500, 2))
    signal = X[:, 1] - 0.18 * (X[:, 0] ** 2) + 0.08 * X[:, 0]
    y = (signal + 0.10 * rng.normal(size=500) > 0.0).astype(int)
    return X, y


def _radial_rings() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    n_samples = 700
    ring_ids = rng.integers(0, 4, size=n_samples)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    radii = (ring_ids + 1) * 0.45 + rng.normal(scale=0.04, size=n_samples)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    y = (ring_ids % 2).astype(int)
    return X, y


def _fragmented_islands(grid_size: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points_per_island = 45
    spacing = 3.0
    island_std = 0.22
    offset = (grid_size - 1) * spacing / 2.0
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for row in range(grid_size):
        for col in range(grid_size):
            center = (row * spacing - offset, col * spacing - offset)
            label = (row + col) % 2
            X_parts.append(
                rng.normal(loc=center, scale=island_std, size=(points_per_island, 2))
            )
            y_parts.append(np.full(points_per_island, label))

    return np.vstack(X_parts), np.concatenate(y_parts)


def _random_labels() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=600,
        centers=[(-2.5, -2.5), (2.5, 2.5)],
        cluster_std=0.75,
        random_state=7,
    )
    rng = np.random.default_rng(7)
    return X, rng.permutation(y)


def build_cases() -> list[SyntheticCase]:
    """Return the ordered synthetic dataset ladder."""
    return [
        SyntheticCase("clean linear", 1.0, _clean_linear),
        SyntheticCase("overlap/noise", 1.7, _overlapping_linear),
        SyntheticCase("smooth quadratic", 2.3, _smooth_quadratic),
        SyntheticCase(
            "two moons",
            3.0,
            lambda: make_moons(n_samples=600, noise=0.18, random_state=0),
        ),
        SyntheticCase("radial rings", 3.4, _radial_rings),
        SyntheticCase(
            "fragmented islands 3x3",
            4.2,
            lambda: _fragmented_islands(3, seed=8),
        ),
        SyntheticCase(
            "fragmented islands 5x5",
            4.7,
            lambda: _fragmented_islands(5, seed=9),
        ),
        SyntheticCase("random labels", 5.0, _random_labels),
    ]


def render_case_summary(case: SyntheticCase, report: DiagnosticReport) -> None:
    """Print a compact textual summary for one dataset."""
    level = RECOMMENDATION_LEVELS[report.recommendation]
    scores = report.scores
    lead_reason = (
        report.decision_path[0] if report.decision_path else "No decision path."
    )
    print(
        f"{case.designed_complexity:>4.1f} | "
        f"{case.name:<24} | "
        f"level={level} | "
        f"{report.recommendation:<44} | "
        f"confidence={report.confidence:<6} | "
        f"signal={scores['signal_score']:.3f} | "
        f"linearity={scores['linearity_score']:.3f} | "
        f"nonlinearity={scores['nonlinearity_score']:.3f} | "
        f"fragmentation={scores['fragmentation_score']:.3f}"
    )
    print(f"      decision: {lead_reason}")


def plot_ladder(cases: list[SyntheticCase], reports: list[DiagnosticReport]) -> None:
    """Render and save the recommendation complexity ladder plot."""
    fig, ax = plt.subplots(figsize=(11, 7))
    x_min, x_max = 0.7, 5.3

    for level, label in LEVEL_LABELS.items():
        ax.axhspan(
            level - 0.5,
            level + 0.5,
            facecolor=LEVEL_COLORS[level],
            alpha=0.12,
            zorder=0,
        )
        ax.text(
            x_max - 0.05,
            level + 0.28,
            label,
            ha="right",
            va="top",
            fontsize=8,
            color=LEVEL_COLORS[level],
            alpha=0.95,
        )

    for case, report in zip(cases, reports, strict=True):
        x = case.designed_complexity
        y = RECOMMENDATION_LEVELS[report.recommendation]
        color = LEVEL_COLORS[y]
        marker = "o" if report.confidence == "high" else "s"
        size = 140 if report.confidence == "high" else 110

        ax.scatter(
            x,
            y,
            s=size,
            c=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
            zorder=3,
        )
        ax.annotate(
            case.name,
            (x, y),
            xytext=ANNOTATION_OFFSETS.get(case.name, (7, 7, "left", "bottom"))[:2],
            textcoords="offset points",
            fontsize=9,
            ha=ANNOTATION_OFFSETS.get(case.name, (7, 7, "left", "bottom"))[2],
            va=ANNOTATION_OFFSETS.get(case.name, (7, 7, "left", "bottom"))[3],
        )

    ax.plot(
        [case.designed_complexity for case in cases],
        [RECOMMENDATION_LEVELS[report.recommendation] for report in reports],
        color="#4d4d4d",
        linewidth=1.2,
        alpha=0.5,
        zorder=2,
    )

    ax.set_title("Separatix recommendation complexity ladder")
    ax.set_xlabel("Designed synthetic dataset complexity")
    ax.set_ylabel("Separatix recommendation")
    ax.set_yticks(sorted(LEVEL_LABELS))
    ax.set_yticklabels([LEVEL_LABELS[level] for level in sorted(LEVEL_LABELS)])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.4, 4.4)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)

    ax.text(
        0.02,
        0.02,
        "Circle = high confidence, square = medium/low confidence",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"\nSaved plot to: {OUTPUT_PATH}")
    plt.show()


def main() -> None:
    """Run the synthetic ladder and visualize recommendation complexity."""
    cases = build_cases()
    reports: list[DiagnosticReport] = []

    print("Designed complexity | dataset                  | summary")
    for case in cases:
        X, y = case.builder()
        report = diagnose(X, y, return_report=True, random_state=0, topology="off")
        reports.append(report)
        render_case_summary(case, report)

    plot_ladder(cases, reports)


if __name__ == "__main__":
    main()
