from separatix.constants import (
    HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED,
    KERNEL_OR_LOCAL_RECOMMENDED,
)
from separatix.recommendation.engine import (
    compute_multilabel_scores,
    compute_scores,
    make_multilabel_recommendation,
    make_recommendation,
)


def test_recommendation_engine_linear_branch() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.95},
            "knn": {"balanced_accuracy": 0.94},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 50, "1": 50},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "linear"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, confidence, path, _ = make_recommendation(scores, metrics)
    assert recommendation == "linear_likely_sufficient"
    assert confidence in {"medium", "high"}
    assert path


def test_recommendation_engine_smooth_nonlinear_branch() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.84},
            "smooth_poly": {"balanced_accuracy": 0.95},
            "knn": {"balanced_accuracy": 0.94},
            "kernel_approx": {"balanced_accuracy": 0.93},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {"h1_persistence_count": 4, "max_h1_persistence": 0.8},
        "audit": {
            "class_counts": {"0": 50, "1": 50},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "smooth_poly"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, confidence, path, _ = make_recommendation(scores, metrics)
    assert recommendation == "smooth_nonlinear_recommended"
    assert confidence in {"medium", "high"}
    assert path


def test_recommendation_engine_prefers_simpler_smooth_when_scores_tie() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.6},
            "smooth_poly": {"balanced_accuracy": 0.95},
            "knn": {"balanced_accuracy": 0.95},
            "kernel_approx": {"balanced_accuracy": 0.94},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {"h1_persistence_count": 4, "max_h1_persistence": 0.8},
        "audit": {
            "class_counts": {"0": 50, "1": 50},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "knn"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, confidence, path, _ = make_recommendation(scores, metrics)
    assert recommendation == "smooth_nonlinear_recommended"
    assert confidence in {"medium", "high"}
    assert path
    assert metrics["recommendation_evidence"]["selected_family"] == ("smooth_nonlinear")


def test_recommendation_engine_prefers_kernel_local_when_it_clearly_wins() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.6},
            "smooth_poly": {"balanced_accuracy": 0.78},
            "knn": {"balanced_accuracy": 0.94},
            "kernel_approx": {"balanced_accuracy": 0.92},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {"h1_persistence_count": 4, "max_h1_persistence": 0.8},
        "audit": {
            "class_counts": {"0": 120, "1": 120},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "knn"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, confidence, path, _ = make_recommendation(scores, metrics)
    assert recommendation == "kernel_or_local_recommended"
    assert confidence in {"medium", "high"}
    assert path
    assert metrics["recommendation_evidence"]["selected_family"] == "local_kernel"


def test_recommendation_engine_keeps_smooth_when_kernel_margin_is_borderline() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {
                "balanced_accuracy": 0.822,
                "stability_balanced_accuracy_std": 0.056,
            },
            "smooth_poly": {
                "balanced_accuracy": 0.971,
                "stability_balanced_accuracy_std": 0.009,
            },
            "kernel_approx": {
                "balanced_accuracy": 0.986,
                "stability_balanced_accuracy_std": 0.007,
            },
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 180, "1": 180},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "kernel_approx"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, confidence, path, _ = make_recommendation(scores, metrics)
    evidence = metrics["recommendation_evidence"]

    assert recommendation == "smooth_nonlinear_recommended"
    assert confidence == "medium"
    assert path
    assert evidence["raw_best_family"] == "local_kernel"
    assert evidence["recommended_family"] == "smooth_nonlinear"
    assert not evidence["family_comparisons"]["local_kernel_vs_smooth_nonlinear"][
        "first_clearly_better"
    ]
    assert any(
        flag["name"] == "borderline_family_difference"
        for flag in evidence["quality_flags"]
    )


def test_recommendation_engine_keeps_linear_within_one_standard_error() -> None:
    metrics = {
        "probes": {
            "dummy": {"balanced_accuracy": 0.5},
            "linear": {"balanced_accuracy": 0.88},
            "smooth_poly": {"balanced_accuracy": 0.90},
            "knn": {"balanced_accuracy": 0.89},
        },
        "neighborhood": {
            "mean_local_entropy": 0.1,
            "high_entropy_fraction": 0.05,
            "cross_class_neighbor_fraction": 0.08,
        },
        "graph": {"graph_fragmentation_score": 0.1},
        "topology": {},
        "audit": {
            "class_counts": {"0": 25, "1": 25},
            "imbalance_ratio": 1.0,
            "n_classes": 2,
        },
        "boundary": {"boundary_sample_size": 30},
        "geometry": {"distance_concentration_proxy": 0.2},
        "baseline": {"best_probe": "smooth_poly"},
    }
    scores = compute_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, _, _ = make_recommendation(scores, metrics)
    assert recommendation == "linear_likely_sufficient"
    assert (
        "conservative escalation"
        in metrics["recommendation_evidence"]["selection_rule"]
    )


def test_multilabel_recommendation_engine_uses_fragmentation_for_high_capacity(
) -> None:
    metrics = {
        "probes": {
            "dummy": {
                "micro_f1": 0.30,
                "macro_f1": 0.25,
                "sample_jaccard": 0.20,
            },
            "linear": {
                "micro_f1": 0.55,
                "macro_f1": 0.48,
                "sample_jaccard": 0.42,
            },
            "smooth_poly": {
                "micro_f1": 0.62,
                "macro_f1": 0.58,
                "sample_jaccard": 0.50,
            },
            "knn": {
                "micro_f1": 0.82,
                "macro_f1": 0.60,
                "sample_jaccard": 0.76,
            },
            "kernel_approx": {
                "micro_f1": 0.78,
                "macro_f1": 0.57,
                "sample_jaccard": 0.72,
            },
        },
        "neighborhood": {"mean_neighbor_jaccard": 0.45},
        "graph": {"graph_fragmentation_score": 0.62},
        "boundary": {"boundary_sample_size": 18},
        "audit": {"n_samples": 120, "usable_label_count": 4},
    }
    scores = compute_multilabel_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, path, _ = make_multilabel_recommendation(scores, metrics)

    assert recommendation == HIGH_CAPACITY_OR_PARTITIONING_RECOMMENDED
    assert any("fragmented graph" in step for step in path)


def test_multilabel_topology_does_not_replace_fragmentation_gate() -> None:
    metrics = {
        "probes": {
            "dummy": {
                "micro_f1": 0.30,
                "macro_f1": 0.25,
                "sample_jaccard": 0.20,
            },
            "linear": {
                "micro_f1": 0.55,
                "macro_f1": 0.48,
                "sample_jaccard": 0.42,
            },
            "smooth_poly": {
                "micro_f1": 0.62,
                "macro_f1": 0.58,
                "sample_jaccard": 0.50,
            },
            "knn": {
                "micro_f1": 0.82,
                "macro_f1": 0.60,
                "sample_jaccard": 0.76,
            },
        },
        "neighborhood": {"mean_neighbor_jaccard": 0.45},
        "graph": {"graph_fragmentation_score": 0.10},
        "topology": {"target_type": "multilabel", "topology_strength": 1.0},
        "boundary": {"boundary_sample_size": 18},
        "audit": {"n_samples": 120, "usable_label_count": 4},
    }
    scores = compute_multilabel_scores(metrics, skipped_count=0, warning_count=0)
    recommendation, _, path, _ = make_multilabel_recommendation(scores, metrics)

    assert recommendation == KERNEL_OR_LOCAL_RECOMMENDED
    assert any("topology was available" in step for step in path)


def test_multilabel_skipped_topology_is_not_blocking() -> None:
    metrics = {
        "probes": {
            "dummy": {
                "micro_f1": 0.30,
                "macro_f1": 0.25,
                "sample_jaccard": 0.20,
            },
            "linear": {
                "micro_f1": 0.55,
                "macro_f1": 0.48,
                "sample_jaccard": 0.42,
            },
            "smooth_poly": {
                "micro_f1": 0.62,
                "macro_f1": 0.58,
                "sample_jaccard": 0.50,
            },
            "knn": {
                "micro_f1": 0.82,
                "macro_f1": 0.60,
                "sample_jaccard": 0.76,
            },
        },
        "neighborhood": {"mean_neighbor_jaccard": 0.45},
        "graph": {"graph_fragmentation_score": 0.10},
        "topology": {"target_type": "multilabel", "skipped_reason": "topology off"},
        "boundary": {"boundary_sample_size": 18},
        "audit": {"n_samples": 120, "usable_label_count": 4},
    }
    scores = compute_multilabel_scores(metrics, skipped_count=1, warning_count=0)
    recommendation, confidence, _, _ = make_multilabel_recommendation(scores, metrics)
    flags = metrics["multilabel_recommendation_evidence"]["quality_flags"]

    assert recommendation == KERNEL_OR_LOCAL_RECOMMENDED
    assert confidence == "medium"
    assert not any(flag["severity"] == "blocking" for flag in flags)
