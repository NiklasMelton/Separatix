"""Generate the combined probe-family gallery used in the README."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "separatix-matplotlib"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from separatix import DiagnosticReport, diagnose
from separatix.constants import FEEDFORWARD_MLP_RECOMMENDED
from separatix.models.mlp import (
    _MLP_BUDGETS,
    _MLP_HIDDEN_LABELS,
    TorchMLPClassifier,
    _torch_module,
)
from separatix.models.probes import (
    _SMOOTH_PROBE_VARIANTS,
    _full_quadratic_classifier,
    _linear_classifier,
    _scaled_pipeline,
)
from separatix.recommendation.engine import _FAMILY_PROBES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "img" / "separatix_probe_family_gallery.png"

BACKGROUND_COLORS = ("#e7f0f7", "#fbe9dd")
CLASS_COLORS = ("#2f6f9f", "#dd6b3d")
FAMILY_COLORS = {
    "Baseline": "#69747c",
    "Linear": "#16857b",
    "Smooth nonlinear": "#54a24b",
    "Local / kernel": "#3f67b1",
    "Conditional MLP": "#8156a7",
}
SHORT_RECOMMENDATIONS = {
    "linear_likely_sufficient": "linear likely sufficient",
    "smooth_nonlinear_recommended": "smooth nonlinear",
    "kernel_or_local_recommended": "kernel or local",
    "high_capacity_or_partitioning_recommended": "higher capacity",
    "feedforward_mlp_recommended": "feed-forward MLP",
    "feature_or_label_bottleneck_likely": "feature / label bottleneck",
    "insufficient_data_or_unreliable_geometry": "unreliable geometry",
    "inconclusive": "inconclusive",
}

DatasetBuilder = Callable[[], tuple[np.ndarray, np.ndarray]]
EstimatorBuilder = Callable[[np.ndarray, np.ndarray], BaseEstimator]


@dataclass(frozen=True)
class SurfaceExample:
    """One representative dataset and probe to render in the gallery."""

    probe_name: str
    title: str
    family: str
    description: str
    dataset_builder: DatasetBuilder
    estimator_builder: EstimatorBuilder


@dataclass(frozen=True)
class MLPDatasetSpec:
    """One independently calibrated task for an optional MLP subtype."""

    label: str
    task_name: str
    description: str
    teacher_seed: int
    sample_seed: int
    random_state: int
    max_parameters: int | None
    trigger_skill_threshold: float
    min_improvement: float
    min_architecture_margin: float


@dataclass(frozen=True)
class MLPSurfaceExample:
    """One calibrated MLP task, fitted winner, and held-out evidence."""

    spec: MLPDatasetSpec
    X: np.ndarray
    y: np.ndarray
    report: DiagnosticReport
    hidden_layer_sizes: tuple[int, ...]
    tier: str
    depth: int
    parameter_count: int
    held_out_score: float
    estimator: TorchMLPClassifier


def _random_label_data() -> tuple[np.ndarray, np.ndarray]:
    """Return structured features with deliberately uninformative labels."""
    X, y = make_blobs(
        n_samples=360,
        centers=[(-2.1, -1.8), (2.1, 1.8)],
        cluster_std=0.85,
        random_state=7,
    )
    return X, np.random.default_rng(7).permutation(y)


def _linear_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a clean, approximately linear binary split."""
    return make_classification(
        n_samples=360,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.85,
        flip_y=0.025,
        random_state=4,
    )


def _quadratic_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth parabolic class boundary."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.6, 2.6, size=(420, 2))
    signal = X[:, 1] - 0.42 * X[:, 0] ** 2 + 0.72
    y = (signal + 0.12 * rng.normal(size=X.shape[0]) > 0.0).astype(int)
    return X, y


def _checkerboard_data() -> tuple[np.ndarray, np.ndarray]:
    """Return alternating compact islands suited to a local probe."""
    rng = np.random.default_rng(8)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    grid_size = 4
    spacing = 1.8
    offset = (grid_size - 1) * spacing / 2.0
    for row in range(grid_size):
        for column in range(grid_size):
            center = (column * spacing - offset, row * spacing - offset)
            X_parts.append(rng.normal(center, 0.24, size=(24, 2)))
            y_parts.append(np.full(24, (row + column) % 2, dtype=int))
    return np.vstack(X_parts), np.concatenate(y_parts)


def _moon_data() -> tuple[np.ndarray, np.ndarray]:
    """Return curved classes suited to an RBF feature map."""
    return make_moons(
        n_samples=440,
        noise=0.15,
        random_state=6,
    )


def _mlp_teacher_labels(
    X_visible: np.ndarray,
    *,
    teacher_seed: int,
) -> np.ndarray:
    """Return one deterministic piecewise-linear nonlinear target."""
    rng = np.random.default_rng(teacher_seed)
    width = 24
    first_weights = rng.normal(scale=1.2, size=(width, 2))
    first_bias = rng.uniform(-2.0, 2.0, size=width)
    second_weights = rng.normal(
        scale=1.0 / np.sqrt(width),
        size=(width, width),
    )
    second_bias = rng.uniform(-0.8, 0.8, size=width)
    output_weights = rng.normal(size=width)
    first_hidden = np.maximum(0.0, X_visible @ first_weights.T + first_bias)
    second_hidden = np.maximum(
        0.0,
        first_hidden @ second_weights.T + second_bias,
    )
    score = second_hidden @ output_weights
    return (score > np.median(score)).astype(int)


def _mlp_data(spec: MLPDatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return one independently sampled, visualizable MLP calibration task."""
    rng = np.random.default_rng(spec.sample_seed)
    visible = rng.uniform(-3.0, 3.0, size=(5_000, 2))
    nuisance = rng.normal(size=(visible.shape[0], 5))
    y = _mlp_teacher_labels(visible, teacher_seed=spec.teacher_seed)
    return np.column_stack([visible, nuisance]), y


def build_mlp_dataset_specs() -> list[MLPDatasetSpec]:
    """Return the four independently calibrated optional-MLP tasks."""
    return [
        MLPDatasetSpec(
            label="one_layer_compact",
            task_name="Compact piecewise task A",
            description="Selected under a compact parameter budget.",
            teacher_seed=439,
            sample_seed=41_439,
            random_state=3,
            max_parameters=400,
            trigger_skill_threshold=0.68,
            min_improvement=0.01,
            min_architecture_margin=0.01,
        ),
        MLPDatasetSpec(
            label="two_layer_compact",
            task_name="Compact piecewise task B",
            description="Added depth wins within the compact budget.",
            teacher_seed=63,
            sample_seed=11,
            random_state=0,
            max_parameters=400,
            trigger_skill_threshold=0.75,
            min_improvement=0.01,
            min_architecture_margin=0.01,
        ),
        MLPDatasetSpec(
            label="one_layer_wide",
            task_name="Wide piecewise task C",
            description="A wider single hidden layer wins all four candidates.",
            teacher_seed=60,
            sample_seed=11_060,
            random_state=5,
            max_parameters=None,
            trigger_skill_threshold=0.69,
            min_improvement=0.01,
            min_architecture_margin=0.01,
        ),
        MLPDatasetSpec(
            label="two_layer_wide",
            task_name="Deep-wide piecewise task D",
            description="Depth and width together recover the hardest partition.",
            teacher_seed=6,
            sample_seed=11_006,
            random_state=5,
            max_parameters=None,
            trigger_skill_threshold=0.75,
            min_improvement=0.02,
            min_architecture_margin=0.01,
        ),
    ]


def _dummy_estimator(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Return the class-prior baseline used by Separatix."""
    del X, y
    return DummyClassifier(strategy="prior")


def _linear_estimator(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Return the exact single-label linear probe pipeline."""
    del y
    return _linear_classifier(X)


def _quadratic_estimator(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Return the exact full-quadratic single-label probe pipeline."""
    del X, y
    return _full_quadratic_classifier(random_state=0)


def _knn_estimator(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Return the same scaled neighborhood probe used by Separatix."""
    n_neighbors = min(15, max(3, int(np.sqrt(y.shape[0]))))
    return _scaled_pipeline(
        X,
        KNeighborsClassifier(n_neighbors=n_neighbors),
        name="knn",
    )


def _kernel_estimator(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Return the same RBF-approximation classifier used by Separatix."""
    del y
    return Pipeline(
        [
            ("scale_in", StandardScaler()),
            (
                "rff",
                RBFSampler(
                    gamma=1.0 / max(1, X.shape[1]),
                    n_components=min(256, max(32, X.shape[1] * 2)),
                    random_state=0,
                ),
            ),
            ("scale_out", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    random_state=0,
                    max_iter=3000,
                    tol=1e-4,
                ),
            ),
        ]
    )


def build_surface_examples() -> list[SurfaceExample]:
    """Return representative examples for every always-available probe."""
    return [
        SurfaceExample(
            probe_name="dummy",
            title="Class-prior baseline",
            family="Baseline",
            description="No feature geometry: predicts the training-fold prior.",
            dataset_builder=_random_label_data,
            estimator_builder=_dummy_estimator,
        ),
        SurfaceExample(
            probe_name="linear",
            title="Linear probe",
            family="Linear",
            description="A single global separating hyperplane.",
            dataset_builder=_linear_data,
            estimator_builder=_linear_estimator,
        ),
        SurfaceExample(
            probe_name="smooth_poly",
            title="Quadratic probe",
            family="Smooth nonlinear",
            description="A smooth global curve; full and sketched variants shown.",
            dataset_builder=_quadratic_data,
            estimator_builder=_quadratic_estimator,
        ),
        SurfaceExample(
            probe_name="knn",
            title="k-nearest neighbors",
            family="Local / kernel",
            description="Local neighborhoods can follow disconnected regions.",
            dataset_builder=_checkerboard_data,
            estimator_builder=_knn_estimator,
        ),
        SurfaceExample(
            probe_name="kernel_approx",
            title="RBF approximation",
            family="Local / kernel",
            description="Random Fourier features expose kernel-like structure.",
            dataset_builder=_moon_data,
            estimator_builder=_kernel_estimator,
        ),
    ]


def _assert_complete_probe_coverage(
    examples: list[SurfaceExample],
    mlp_specs: list[MLPDatasetSpec],
) -> None:
    """Fail if the gallery falls behind the implemented probe registry."""
    expected_base = {
        probe_name
        for probe_names in _FAMILY_PROBES.values()
        for probe_name in probe_names
    }
    rendered_base = {example.probe_name for example in examples}
    if rendered_base != expected_base:
        missing = sorted(expected_base - rendered_base)
        extra = sorted(rendered_base - expected_base)
        raise RuntimeError(
            f"Probe gallery coverage mismatch: missing={missing}, extra={extra}"
        )

    rendered_smooth_variants = {"full_quadratic", "low_rank_quadratic"}
    expected_smooth_variants = set(_SMOOTH_PROBE_VARIANTS)
    if rendered_smooth_variants != expected_smooth_variants:
        raise RuntimeError("The gallery must show both quadratic probe variants.")

    rendered_mlp_variants = {spec.label for spec in mlp_specs}
    expected_mlp_variants = set(_MLP_HIDDEN_LABELS)
    if rendered_mlp_variants != expected_mlp_variants:
        missing = sorted(expected_mlp_variants - rendered_mlp_variants)
        extra = sorted(rendered_mlp_variants - expected_mlp_variants)
        raise RuntimeError(
            f"MLP gallery coverage mismatch: missing={missing}, extra={extra}"
        )


def _diagnose_example(X: np.ndarray, y: np.ndarray) -> DiagnosticReport:
    """Run the public API with the budget that enables every base probe."""
    report = diagnose(
        X,
        y,
        return_report=True,
        budget="standard",
        topology="off",
        random_state=0,
    )
    if not isinstance(report, DiagnosticReport):
        raise RuntimeError(
            "Expected diagnose(..., return_report=True) to return a report."
        )
    return report


def _mesh_for(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a bounded plotting mesh for a two-dimensional dataset."""
    padding = max(0.3, 0.08 * float(np.ptp(X, axis=0).max()))
    x_values = np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, 280)
    y_values = np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, 280)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_x, grid_y, grid


def _style_surface_axis(ax: Axes) -> None:
    """Apply common styling to a decision-surface panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d5dce1")
        spine.set_linewidth(0.8)


def _family_badge(ax: Axes, family: str) -> None:
    """Draw a small model-family badge."""
    ax.text(
        0.025,
        0.97,
        family.upper(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7.5,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": FAMILY_COLORS[family],
            "edgecolor": "none",
        },
        zorder=8,
    )


def _report_badge(ax: Axes, report: DiagnosticReport) -> None:
    """Annotate a panel with the overall diagnostic recommendation."""
    recommendation = SHORT_RECOMMENDATIONS.get(
        report.recommendation,
        report.recommendation.replace("_", " "),
    )
    ax.text(
        0.025,
        0.025,
        f"Separatix → {recommendation}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#27333b",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#d5dce1",
            "alpha": 0.94,
        },
        zorder=8,
    )


def _draw_surface(
    ax: Axes,
    example: SurfaceExample,
) -> None:
    """Draw one fitted probe surface and its Separatix evidence."""
    X, y = example.dataset_builder()
    report = _diagnose_example(X, y)
    estimator = example.estimator_builder(X, y)
    estimator.fit(X, y)

    grid_x, grid_y, grid = _mesh_for(X)
    grid_predictions = np.asarray(estimator.predict(grid), dtype=int).reshape(
        grid_x.shape
    )
    ax.contourf(
        grid_x,
        grid_y,
        grid_predictions,
        levels=(-0.5, 0.5, 1.5),
        colors=BACKGROUND_COLORS,
        alpha=0.92,
    )
    if np.unique(grid_predictions).shape[0] > 1:
        ax.contour(
            grid_x,
            grid_y,
            grid_predictions,
            levels=(0.5,),
            colors=(FAMILY_COLORS[example.family],),
            linewidths=1.8,
        )
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=[CLASS_COLORS[int(label)] for label in y],
        s=19,
        edgecolors="white",
        linewidths=0.45,
        alpha=0.88,
        zorder=4,
    )

    probe_metrics = report.metrics["probes"][example.probe_name]
    cv_score = float(probe_metrics["balanced_accuracy"])
    fitted_score = balanced_accuracy_score(y, estimator.predict(X))
    ax.set_title(
        f"{example.title}\n{example.probe_name} · held-out BA {cv_score:.2f}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#24313a",
        pad=10,
    )
    ax.text(
        0.99,
        0.97,
        f"fit {fitted_score:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#66727a",
        zorder=8,
    )
    ax.text(
        0.0,
        -0.08,
        example.description,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#52616a",
    )
    _family_badge(ax, example.family)
    _report_badge(ax, report)
    _style_surface_axis(ax)


def _draw_quadratic_variant_key(ax: Axes) -> None:
    """Show how the two quadratic subtypes construct their feature maps."""
    ax.text(
        0.975,
        0.08,
        "full_quadratic\n"
        "x → exact degree-2 map → linear\n\n"
        "low_rank_quadratic\n"
        "x → sketched degree-2 map → linear",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.1,
        color="#26343d",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#d5dce1",
            "alpha": 0.93,
        },
        zorder=8,
    )


def _fit_mlp_surface_examples(
    specs: list[MLPDatasetSpec],
) -> list[MLPSurfaceExample]:
    """Fit each MLP subtype on its own validated calibration task."""
    if _torch_module() is None:
        raise RuntimeError(
            "The probe gallery requires the optional MLP dependency. "
            "Install it with `poetry install -E mlp -E examples`."
        )
    budget = _MLP_BUDGETS["standard"]
    examples: list[MLPSurfaceExample] = []
    for spec in specs:
        X, y = _mlp_data(spec)
        report = diagnose(
            X,
            y,
            return_report=True,
            budget="standard",
            topology="off",
            random_state=spec.random_state,
            mlp_probes=True,
            mlp_trigger_skill_threshold=spec.trigger_skill_threshold,
            mlp_min_improvement=spec.min_improvement,
            mlp_max_parameters=spec.max_parameters,
        )
        if not isinstance(report, DiagnosticReport):
            raise RuntimeError("Expected the MLP diagnostic to return a report.")
        payload: dict[str, Any] = report.metrics["mlp_probes"]
        expected_probe_name = f"mlp_{spec.label}"
        selected_probe_name = str(
            payload.get("best_architecture", {}).get("probe_name")
        )
        if (
            report.recommendation != FEEDFORWARD_MLP_RECOMMENDED
            or payload.get("recommendation_override") is not True
            or selected_probe_name != expected_probe_name
        ):
            raise RuntimeError(
                f"The {spec.label} calibration no longer selects "
                f"{expected_probe_name} with a validated MLP override."
            )

        evidence_by_label = {
            str(item["label"]): item for item in payload.get("architectures", [])
        }
        if spec.label not in evidence_by_label:
            raise RuntimeError(
                f"The {spec.label} calibration did not evaluate its intended subtype."
            )
        evidence = evidence_by_label[spec.label]
        other_scores = [
            float(item["balanced_accuracy"])
            for label, item in evidence_by_label.items()
            if label != spec.label and "balanced_accuracy" in item
        ]
        architecture_margin = float(evidence["balanced_accuracy"]) - max(
            other_scores
        )
        if architecture_margin < spec.min_architecture_margin:
            raise RuntimeError(
                f"The {spec.label} calibration margin fell to "
                f"{architecture_margin:.3f}; expected at least "
                f"{spec.min_architecture_margin:.3f}."
            )
        hidden_layer_sizes = tuple(
            int(value) for value in evidence["hidden_layer_sizes"]
        )
        estimator = TorchMLPClassifier(
            task="singlelabel",
            hidden_layer_sizes=hidden_layer_sizes,
            epochs=budget["epochs"],
            patience=budget["patience"],
            batch_size=min(256, X.shape[0]),
            device="cpu",
            random_state=spec.random_state,
        )
        estimator.fit(X, y)
        examples.append(
            MLPSurfaceExample(
                spec=spec,
                X=X,
                y=y,
                report=report,
                hidden_layer_sizes=hidden_layer_sizes,
                tier=str(evidence["tier"]),
                depth=int(evidence["depth"]),
                parameter_count=int(evidence["parameter_count"]),
                held_out_score=float(evidence["balanced_accuracy"]),
                estimator=estimator,
            )
        )
    return examples


def _draw_mlp_surface(
    ax: Axes,
    example: MLPSurfaceExample,
) -> None:
    """Draw the selected MLP boundary on its own calibration task."""
    X = example.X
    y = example.y
    report = example.report
    grid_x, grid_y, visible_grid = _mesh_for(X[:, :2])
    reference_row = np.median(X, axis=0)
    full_grid = np.tile(reference_row, (visible_grid.shape[0], 1))
    full_grid[:, :2] = visible_grid
    grid_predictions = np.asarray(
        example.estimator.predict(full_grid),
        dtype=int,
    ).reshape(grid_x.shape)
    ax.contourf(
        grid_x,
        grid_y,
        grid_predictions,
        levels=(-0.5, 0.5, 1.5),
        colors=BACKGROUND_COLORS,
        alpha=0.92,
    )
    if np.unique(grid_predictions).shape[0] > 1:
        ax.contour(
            grid_x,
            grid_y,
            grid_predictions,
            levels=(0.5,),
            colors=(FAMILY_COLORS["Conditional MLP"],),
            linewidths=1.8,
        )

    displayed_rows = np.random.default_rng(0).choice(
        X.shape[0],
        size=800,
        replace=False,
    )
    ax.scatter(
        X[displayed_rows, 0],
        X[displayed_rows, 1],
        c=[CLASS_COLORS[int(label)] for label in y[displayed_rows]],
        s=13,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.82,
        zorder=4,
    )
    title = example.spec.label.replace("_", " ").capitalize()
    ax.set_title(
        f"{title}\nmlp_{example.spec.label} · held-out BA "
        f"{example.held_out_score:.2f}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#24313a",
        pad=10,
    )
    _family_badge(ax, "Conditional MLP")
    aligned_comparators: dict[str, dict[str, Any]] = report.metrics["mlp_probes"][
        "aligned_comparators"
    ]
    best_simpler_score = max(
        float(result["balanced_accuracy"])
        for name, result in aligned_comparators.items()
        if name != "dummy" and "balanced_accuracy" in result
    )
    ax.text(
        0.975,
        0.025,
        f"best simpler {best_simpler_score:.2f} → MLP {example.held_out_score:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#44334f",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d5dce1",
            "alpha": 0.94,
        },
        zorder=8,
    )
    _report_badge(ax, report)
    widths = " × ".join(str(value) for value in example.hidden_layer_sizes)
    budget_note = (
        "compact budget"
        if example.spec.max_parameters is not None
        else "all candidates"
    )
    ax.text(
        0.0,
        -0.08,
        f"{widths} hidden units · {example.parameter_count:,} parameters · "
        f"{budget_note}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#52616a",
    )
    _style_surface_axis(ax)


def create_gallery() -> Figure:
    """Build and return the complete probe-family gallery figure."""
    examples = build_surface_examples()
    mlp_specs = build_mlp_dataset_specs()
    _assert_complete_probe_coverage(examples, mlp_specs)
    mlp_examples = _fit_mlp_surface_examples(mlp_specs)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15.2, 14.0),
        facecolor="#ffffff",
    )
    flat_axes = axes.ravel()
    for ax, base_example in zip(flat_axes, examples):
        _draw_surface(ax, base_example)
        if base_example.probe_name == "smooth_poly":
            _draw_quadratic_variant_key(ax)
    for ax, mlp_example in zip(flat_axes[len(examples) :], mlp_examples):
        _draw_mlp_surface(ax, mlp_example)

    fig.suptitle(
        "What Separatix probe families can see",
        x=0.055,
        y=0.99,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="#1f2d35",
    )
    fig.text(
        0.055,
        0.965,
        "Illustrative fitted surfaces. Separatix compares held-out probe evidence "
        "to the dummy baseline and escalates only when added complexity clearly helps.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#52616a",
    )
    fig.text(
        0.945,
        0.02,
        "Point color = true class  ·  background = fitted prediction  ·  "
        "BA = balanced accuracy",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#66727a",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.945,
        top=0.91,
        bottom=0.065,
        wspace=0.14,
        hspace=0.36,
    )
    return fig


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for gallery generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output resolution.")
    return parser.parse_args()


def main() -> None:
    """Generate and save the README probe-family gallery."""
    args = _parse_args()
    figure = create_gallery()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    print(f"Saved probe-family gallery to: {args.output}")
    plt.close(figure)


if __name__ == "__main__":
    main()
