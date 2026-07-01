"""Constants used across the separatix package."""

from __future__ import annotations

LINEAR_LIKELY_SUFFICIENT = "linear_likely_sufficient"
SMOOTH_NONLINEAR_RECOMMENDED = "smooth_nonlinear_recommended"
KERNEL_OR_LOCAL_RECOMMENDED = "kernel_or_local_recommended"
HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED = "high_capacity_or_partitioning_recommended"
FEATURE_OR_LABEL_BOTTLENECK_LIKELY = "feature_or_label_bottleneck_likely"
INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY = "insufficient_data_or_unreliable_geometry"
INCONCLUSIVE = "inconclusive"
LINEAR_RESPONSE_LIKELY_SUFFICIENT = "linear_response_likely_sufficient"
SMOOTH_NONLINEAR_RESPONSE_RECOMMENDED = "smooth_nonlinear_response_recommended"
KERNEL_OR_LOCAL_REGRESSION_RECOMMENDED = "kernel_or_local_regression_recommended"
HIGH_CAPACITY_OR_PARTITIONING_REGRESSION_RECOMMENDED = (
    "higher_capacity_or_partitioning_regression_recommended"
)
FEATURE_OR_TARGET_BOTTLENECK_LIKELY = "feature_or_target_bottleneck_likely"
INSUFFICIENT_DATA_OR_UNRELIABLE_REGRESSION_GEOMETRY = (
    "insufficient_data_or_unreliable_regression_geometry"
)
INCONCLUSIVE_REGRESSION_DIAGNOSTIC = "inconclusive_regression_diagnostic"

RECOMMENDATION_LABELS = {
    LINEAR_LIKELY_SUFFICIENT: "Linear model likely sufficient.",
    SMOOTH_NONLINEAR_RECOMMENDED: "Smooth nonlinear model likely useful.",
    KERNEL_OR_LOCAL_RECOMMENDED: "Kernel or local model likely useful.",
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED: (
        "Higher-capacity or partitioning model likely useful."
    ),
    FEATURE_OR_LABEL_BOTTLENECK_LIKELY: "Feature or label bottleneck likely.",
    INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY: (
        "Insufficient data or unreliable geometry."
    ),
    INCONCLUSIVE: "Diagnostic result is inconclusive.",
    LINEAR_RESPONSE_LIKELY_SUFFICIENT: "Linear response model likely sufficient.",
    SMOOTH_NONLINEAR_RESPONSE_RECOMMENDED: (
        "Smooth nonlinear response model likely useful."
    ),
    KERNEL_OR_LOCAL_REGRESSION_RECOMMENDED: (
        "Kernel or local regression model likely useful."
    ),
    HIGH_CAPACITY_OR_PARTITIONING_REGRESSION_RECOMMENDED: (
        "Higher-capacity or partitioning regression model likely useful."
    ),
    FEATURE_OR_TARGET_BOTTLENECK_LIKELY: "Feature or target bottleneck likely.",
    INSUFFICIENT_DATA_OR_UNRELIABLE_REGRESSION_GEOMETRY: (
        "Insufficient data or unreliable regression geometry."
    ),
    INCONCLUSIVE_REGRESSION_DIAGNOSTIC: "Regression diagnostic result is inconclusive.",
}

BUDGETS = {
    "fast": {
        "max_probe_samples": 5000,
        "max_neighbor_samples": 5000,
        "max_boundary_samples": 2000,
        "cv_folds": 3,
        "bootstrap_repeats": 0,
        "run_kernel_probe": False,
        "run_persistent_topology": False,
    },
    "standard": {
        "max_probe_samples": 20000,
        "max_neighbor_samples": 10000,
        "max_boundary_samples": 3000,
        "cv_folds": 5,
        "bootstrap_repeats": 3,
        "run_kernel_probe": True,
        "run_persistent_topology": "auto",
    },
    "extended": {
        "max_probe_samples": 50000,
        "max_neighbor_samples": 20000,
        "max_boundary_samples": 5000,
        "cv_folds": 5,
        "bootstrap_repeats": 10,
        "run_kernel_probe": True,
        "run_persistent_topology": "auto",
    },
}

CONFIDENCE_LEVELS = ("low", "medium", "high")
