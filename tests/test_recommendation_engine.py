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


def test_recommendation_engine_prefers_kernel_local_when_topology_wins_tie() -> None:
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
    assert recommendation == "kernel_or_local_recommended"
    assert confidence in {"medium", "high"}
    assert path
