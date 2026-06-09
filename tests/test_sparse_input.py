import numpy as np
import pytest
from scipy import sparse

from separatix import diagnose
from separatix.exceptions import DensificationError


def test_sparse_input_runs() -> None:
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(120, 20)))
    y = np.array([0] * 60 + [1] * 60)
    report = diagnose(X, y, return_report=True, random_state=0)
    assert report.preprocessing["is_sparse"] is True


def test_sparse_fail_policy_raises() -> None:
    X = sparse.csr_matrix(np.ones((1000, 1000)))
    y = np.array([0, 1] * 500)
    with pytest.raises(DensificationError):
        diagnose(X, y, densify_policy="fail", max_dense_mb=1, random_state=0)


def test_sparse_skip_policy_records_skip() -> None:
    X = sparse.csr_matrix(np.ones((1000, 1000)))
    y = np.array([0, 1] * 500)
    report = diagnose(
        X,
        y,
        return_report=True,
        densify_policy="skip",
        max_dense_mb=1,
        random_state=0,
    )
    assert report.skipped_diagnostics
