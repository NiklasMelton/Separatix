from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from separatix import diagnose
from separatix.constants import INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY
from separatix.metrics.neighborhood import _singlelabel_summary
from separatix.models.scoring import choose_cv


def test_grouped_singlelabel_report_skips_unsupported_classes() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ]
    )
    y = np.array(["cat", "cat", "dog", "dog", "fox", "fox"], dtype=object)
    groups = np.array(
        ["patient_a", "patient_b", "patient_c", "patient_d", "patient_e", "patient_e"],
        dtype=object,
    )

    report = diagnose(X, y, groups=groups, return_report=True, random_state=0)

    assert report.grouping["provided"] is True
    assert report.grouping["skipped_singlelabel_classes"] == ["fox"]
    assert report.class_summary["evaluable_classes"] == ["cat", "dog"]
    assert report.recommendation == INSUFFICIENT_DATA_OR_UNRELIABLE_GEOMETRY


def test_grouped_singlelabel_requires_two_supported_classes() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ]
    )
    y = np.array(["cat", "cat", "dog", "dog", "fox", "fox"], dtype=object)
    groups = np.array(
        ["g1", "g1", "g2", "g3", "g4", "g4"],
        dtype=object,
    )

    with pytest.raises(ValueError, match="At least two single-label classes"):
        diagnose(X, y, groups=groups, return_report=True, random_state=0)


def test_grouped_choose_cv_keeps_groups_disjoint() -> None:
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)

    cv, method = choose_cv(y, 5, groups=groups, random_state=0)

    assert cv is not None
    assert method in {"stratified_group", "group_kfold", "group_heuristic"}
    for train_idx, test_idx in cv.split(np.zeros((len(y), 1)), y, groups):
        assert np.intersect1d(groups[train_idx], groups[test_idx]).size == 0
        assert set(np.unique(y[train_idx])) == {0, 1}


def test_grouped_multilabel_report_skips_labels_without_group_support() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ]
    )
    Y = np.array(
        [
            [1, 1],
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
        ],
        dtype=int,
    )
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p4"], dtype=object)

    report = diagnose(
        X,
        Y,
        groups=groups,
        target_mode="multilabel",
        return_report=True,
        random_state=0,
    )

    assert report.grouping["supported_multilabel_labels"] == ["label_0"]
    assert report.grouping["skipped_multilabel_labels"] == ["label_1"]


def test_grouped_report_serialization_omits_raw_group_ids() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ]
    )
    y = np.array(["cat", "cat", "cat", "cat", "dog", "dog", "dog", "dog"], dtype=object)
    groups = np.array(
        [
            "patient_a",
            "patient_a",
            "patient_b",
            "patient_b",
            "patient_c",
            "patient_c",
            "patient_d",
            "patient_d",
        ],
        dtype=object,
    )

    report = diagnose(X, y, groups=groups, return_report=True, random_state=0)
    payload = report.to_json()

    assert "patient_a" not in payload
    assert "patient_b" not in payload
    assert report.grouping["provided"] is True
    assert "Grouped evaluation was used" in report.recommendation_text


def test_cross_group_neighbors_never_request_all_rows(monkeypatch) -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 5))
    y = np.asarray([0, 1] * 100)
    groups = np.repeat(np.arange(10), 20)
    requested: list[int] = []
    original = NearestNeighbors.kneighbors

    def recording_kneighbors(self, *args, **kwargs):
        requested.append(int(kwargs.get("n_neighbors") or self.n_neighbors))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NearestNeighbors, "kneighbors", recording_kneighbors)
    result = _singlelabel_summary(X, y, groups=groups, cross_group=True)
    assert result["local_entropy"]
    assert requested
    assert max(requested) <= 15
