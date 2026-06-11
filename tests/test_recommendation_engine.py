from separatix.recommendation.engine import compute_scores, make_recommendation


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
    assert metrics["recommendation_evidence"]["selected_family"] == (
        "smooth_nonlinear"
    )


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
    assert metrics["recommendation_evidence"]["selection_rule"].startswith("Choose")
