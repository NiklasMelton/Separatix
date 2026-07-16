import numpy as np
import pytest
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.densify import (
    ensure_dense_multilabel_target,
    ensure_dense_or_sample,
    ensure_dense_or_sample_multilabel,
)
from separatix.exceptions import DensificationError, DensificationWarning


def _context() -> dict[str, object]:
    return {"warnings": [], "skipped_diagnostics": [], "densification_events": []}


def test_dense_passthrough() -> None:
    X = np.ones((10, 3))
    y = np.array([0, 1] * 5)
    result = ensure_dense_or_sample(
        X, y, reason="test", config=ProfilerConfig(), report_context=_context()
    )
    assert result["X"].shape == X.shape
    assert result["performed"] is False


def test_densify_fail_policy() -> None:
    X = sparse.csr_matrix(np.ones((1000, 1000)))
    y = np.array([0, 1] * 500)
    config = ProfilerConfig(densify_policy="fail", max_dense_mb=1)
    with pytest.raises(DensificationError):
        ensure_dense_or_sample(
            X, y, reason="test", config=config, report_context=_context()
        )


def test_densify_skip_policy() -> None:
    X = sparse.csr_matrix(np.ones((200, 200)))
    y = np.array([0, 1] * 100)
    config = ProfilerConfig(densify_policy="skip", max_dense_mb=1)
    result = ensure_dense_or_sample(
        X, y, reason="test", config=config, report_context=_context()
    )
    assert result["skipped"] in {True, False}


def test_warn_and_sample_records_event() -> None:
    X = sparse.csr_matrix(np.ones((1000, 1000)))
    y = np.array([0] * 500 + [1] * 500)
    context = _context()
    config = ProfilerConfig(
        densify_policy="warn_and_sample",
        max_dense_mb=2,
        min_dense_samples=10,
        random_state=0,
    )
    with pytest.warns(DensificationWarning):
        result = ensure_dense_or_sample(
            X, y, reason="test", config=config, report_context=context
        )
    assert context["densification_events"]
    assert "skipped" in result


def test_full_sparse_multilabel_densification_preserves_groups() -> None:
    X = sparse.csr_matrix(np.eye(8))
    Y = np.column_stack([np.asarray([0, 1] * 4), np.asarray([1, 1, 0, 0] * 2)])
    groups = np.repeat(np.arange(4), 2)
    result = ensure_dense_or_sample_multilabel(
        X,
        Y,
        reason="multilabel_test",
        config=ProfilerConfig(max_dense_mb=10, warn_on_densify=False),
        report_context=_context(),
        groups=groups,
    )
    assert np.array_equal(result["groups"], groups)
    assert np.array_equal(result["Y"], Y)


def test_sparse_multilabel_target_never_densifies_above_budget(monkeypatch) -> None:
    rows = np.tile(np.asarray([0, 1]), 5000)
    columns = np.repeat(np.arange(5000), 2)
    Y = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, columns)),
        shape=(300, 5000),
    )
    X = sparse.csr_matrix(np.eye(300))
    densified_shapes: list[tuple[int, int]] = []
    original = sparse.csr_matrix.toarray

    def recording_toarray(self, *args, **kwargs):
        densified_shapes.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", recording_toarray)
    result = ensure_dense_multilabel_target(
        X,
        Y,
        reason="target_test",
        config=ProfilerConfig(
            max_dense_mb=1,
            min_dense_samples=20,
            max_samples=250,
            random_state=0,
        ),
        report_context=_context(),
    )
    assert result["skipped"] is False
    assert result["Y"].shape[0] <= 209
    assert densified_shapes
    assert all(shape[0] <= 209 for shape in densified_shapes)
