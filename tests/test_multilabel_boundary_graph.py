import numpy as np

from separatix.config import ProfilerConfig
from separatix.metrics.boundary import compute_multilabel_boundary_candidates
from separatix.metrics.graph import compute_multilabel_graph_fragmentation


def test_multilabel_boundary_candidates_record_trigger_reasons() -> None:
    Y = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    neighborhood = {
        "local_neighbor_jaccard": [0.90, 0.10, 0.85, 1.0],
        "local_neighbor_hamming_distance": [0.10, 0.90, 0.15, 0.10],
        "local_label_entropy": [0.10, 0.80, 0.20, 0.10],
        "local_cardinality_std": [0.10, 0.70, 0.20, 0.10],
    }
    probes = {
        "linear": {
            "predictions": [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        },
        "knn": {
            "predictions": [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        },
    }
    boundary = compute_multilabel_boundary_candidates(
        Y,
        neighborhood,
        probes,
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert boundary["candidate_indices"] == [1, 3]
    assert boundary["strong_candidate_indices"] == [1]
    assert boundary["trigger_counts"]["high_neighbor_hamming"] == 1
    assert boundary["trigger_counts"]["linear_vs_local_prediction_disagreement"] == 1
    assert "high_local_label_entropy" in boundary["trigger_names_by_index"][1]
    assert "linear_vs_local_prediction_disagreement" in (
        boundary["trigger_names_by_index"][3]
    )


def test_multilabel_graph_fragmentation_reports_edge_and_component_metrics() -> None:
    cluster_a = np.column_stack(
        [
            np.linspace(0.0, 0.6, 12),
            np.zeros(12),
        ]
    )
    cluster_b = np.column_stack(
        [
            10.0 + np.linspace(0.0, 0.6, 12),
            np.zeros(12),
        ]
    )
    X = np.vstack([cluster_a, cluster_b])
    Y = np.zeros((24, 4), dtype=int)
    Y[:12, 0] = 1
    Y[:12, 1] = np.arange(12) % 2
    Y[12:, 2] = 1
    Y[12:, 3] = np.arange(12) % 2
    boundary = {"candidate_indices": list(range(24))}
    graph = compute_multilabel_graph_fragmentation(
        X,
        Y,
        boundary,
        config=ProfilerConfig(budget="fast", random_state=0),
    )

    assert graph["component_count"] >= 2
    assert 0.0 <= graph["mean_edge_label_jaccard"] <= 1.0
    assert 0.0 <= graph["mean_edge_hamming_distance"] <= 1.0
    assert 0.0 <= graph["low_label_overlap_edge_fraction"] <= 1.0
    assert "component_label_diversity" in graph
    assert "component_cardinality_variance" in graph
