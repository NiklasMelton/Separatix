import numpy as np
import pytest
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.metrics.boundary import (
    compute_boundary_candidates,
    compute_multilabel_boundary_candidates,
)
from separatix.metrics.graph import (
    _multilabel_edge_metrics,
    compute_multilabel_graph_fragmentation,
)
from separatix.metrics.neighborhood import _label_entropy


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
    assert (
        "linear_vs_local_prediction_disagreement"
        in (boundary["trigger_names_by_index"][3])
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


def test_flat_easy_neighborhoods_do_not_make_every_row_a_boundary() -> None:
    y = np.asarray([0, 1] * 10)
    neighborhood = {
        "local_entropy": [0.0] * 20,
        "local_ambiguity": [0.0] * 20,
        "row_indices": list(range(20)),
    }
    probes = {
        "linear": {"predictions": y.tolist()},
        "knn": {"predictions": y.tolist()},
    }
    boundary = compute_boundary_candidates(y, neighborhood, probes)
    assert boundary["candidate_indices"] == []
    assert boundary["candidate_fraction"] == 0.0


def test_multilabel_empty_predictions_agree_and_flat_rows_are_not_boundaries() -> None:
    Y = np.zeros((12, 3), dtype=int)
    neighborhood = {
        "local_neighbor_jaccard": [1.0] * 12,
        "local_neighbor_hamming_distance": [0.0] * 12,
        "local_label_entropy": [0.0] * 12,
        "local_cardinality_std": [0.0] * 12,
    }
    predictions = np.zeros_like(Y).tolist()
    boundary = compute_multilabel_boundary_candidates(
        Y,
        neighborhood,
        {"linear": {"predictions": predictions}, "knn": {"predictions": predictions}},
        label_names=np.asarray(["a", "b", "c"]),
    )
    assert boundary["candidate_indices"] == []
    assert boundary["trigger_counts"]["linear_vs_local_prediction_disagreement"] == 0


def test_empty_label_sets_have_perfect_graph_edge_jaccard() -> None:
    metrics = _multilabel_edge_metrics(
        np.zeros((2, 3), dtype=int),
        np.asarray([0]),
        np.asarray([1]),
    )
    assert metrics["mean_edge_label_jaccard"] == 1.0
    assert metrics["low_label_overlap_edge_fraction"] == 0.0


def test_local_multilabel_entropy_averages_constant_labels_as_zero() -> None:
    values = np.column_stack([np.asarray([0, 1, 0, 1]), np.zeros(4), np.ones(4)])
    assert _label_entropy(values) == pytest.approx(1.0 / 3.0)


def test_sparse_boundary_summary_does_not_densify_full_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Y = sparse.csr_matrix(np.eye(12, 3, dtype=int))
    neighborhood = {
        "local_neighbor_jaccard": [0.0] + [1.0] * 11,
        "local_neighbor_hamming_distance": [1.0] + [0.0] * 11,
        "local_label_entropy": [1.0] + [0.0] * 11,
        "local_cardinality_std": [1.0] + [0.0] * 11,
    }

    def fail_toarray(self: object, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("full sparse target must not be densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    boundary = compute_multilabel_boundary_candidates(
        Y,
        neighborhood,
        {},
        label_names=np.asarray(["a", "b", "c"]),
    )
    assert boundary["candidate_indices"] == [0]
    assert boundary["top_candidate_label_counts"] == [{"label": "a", "count": 1}]
