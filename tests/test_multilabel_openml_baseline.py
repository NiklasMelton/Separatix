"""Optional OpenML-backed multilabel recommendation baseline tests."""

from __future__ import annotations

import os
import ssl
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError

from separatix import diagnose
from separatix.constants import KERNEL_OR_LOCAL_RECOMMENDED

RUN_OPENML_TESTS = os.environ.get("SEPARATIX_RUN_OPENML_TESTS") == "1"


@contextmanager
def _openml_https_context() -> object:
    """Temporarily prefer certifi CA roots when available."""
    original = ssl._create_default_https_context
    try:
        import certifi
    except ImportError:
        yield
        return
    ssl._create_default_https_context = lambda *args, **kwargs: (  # noqa: E731
        ssl.create_default_context(cafile=certifi.where())
    )
    try:
        yield
    finally:
        ssl._create_default_https_context = original


def _openml_data_home() -> str:
    """Return the cache directory used for optional OpenML baseline downloads."""
    override = os.environ.get("SEPARATIX_OPENML_DATA_HOME")
    if override:
        return override
    return str(Path(tempfile.gettempdir()) / "separatix_sklearn_data")


def _fetch_yeast_multilabel() -> tuple[np.ndarray, np.ndarray]:
    """Return the OpenML yeast v4 multilabel benchmark as dense arrays."""
    with _openml_https_context():
        dataset = fetch_openml(
            name="yeast",
            version=4,
            as_frame=False,
            parser="liac-arff",
            data_home=_openml_data_home(),
        )
    X = np.asarray(dataset.data, dtype=float)
    Y = (np.asarray(dataset.target) == "TRUE").astype(int)
    return X, Y


@pytest.mark.skipif(
    not RUN_OPENML_TESTS,
    reason="set SEPARATIX_RUN_OPENML_TESTS=1 to run OpenML baseline tests",
)
@pytest.mark.parametrize("seed", [0, 1])
def test_multilabel_openml_yeast_baseline_recommends_kernel_local(seed: int) -> None:
    """Yeast v4 should exercise the kernel/local multilabel recommendation path."""
    try:
        X, Y = _fetch_yeast_multilabel()
    except (OpenMLError, OSError, TimeoutError, URLError) as exc:
        pytest.skip(f"unable to fetch OpenML yeast baseline: {exc}")

    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        return_report=True,
        budget="fast",
        random_state=seed,
        topology="off",
    )

    evidence = report.metrics["multilabel_recommendation_evidence"]
    local_vs_smooth = evidence["family_comparisons"]["local_kernel_vs_smooth"]

    assert report.recommendation == KERNEL_OR_LOCAL_RECOMMENDED
    assert evidence["signal_metrics_beating_dummy"] == [
        "micro_f1",
        "macro_f1",
        "sample_jaccard",
    ]
    assert local_vs_smooth["clear_metrics"] == ["micro_f1", "sample_jaccard"]
