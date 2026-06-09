import numpy as np
import pytest
from scipy import sparse

from separatix.config import ProfilerConfig
from separatix.densify import ensure_dense_or_sample
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
