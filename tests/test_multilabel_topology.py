import builtins

import numpy as np
from scipy import sparse

from separatix import diagnose
from separatix.config import ProfilerConfig
from separatix.metrics.topology import compute_multilabel_topology_diagnostics


def _topology_data(n_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    X = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            rng.normal(scale=0.05, size=n_samples),
        ]
    )
    Y = np.column_stack(
        [
            np.cos(theta) > -0.2,
            np.sin(theta) > -0.2,
            np.arange(n_samples) % 2 == 0,
        ]
    ).astype(int)
    return X, Y


def test_multilabel_topology_off_skips_without_recording_diagnostic() -> None:
    X, Y = _topology_data()
    context: dict[str, list[dict[str, str]]] = {"skipped_diagnostics": []}
    topology = compute_multilabel_topology_diagnostics(
        X,
        Y,
        {"candidate_indices": list(range(40))},
        config=ProfilerConfig(topology="off"),
        report_context=context,
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert topology["target_type"] == "multilabel"
    assert topology["mode"] == "off"
    assert topology["skipped_reason"] == "topology disabled"
    assert context["skipped_diagnostics"] == []


def test_multilabel_topology_fast_auto_skips_persistent_work() -> None:
    X, Y = _topology_data()
    topology = compute_multilabel_topology_diagnostics(
        X,
        Y,
        {"candidate_indices": list(range(40))},
        config=ProfilerConfig(budget="fast", topology="auto"),
        report_context={"skipped_diagnostics": []},
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert topology["skipped_reason"] == "persistent topology disabled for this budget"


def test_multilabel_topology_too_few_samples_skips_cleanly() -> None:
    X, Y = _topology_data(n_samples=20)
    topology = compute_multilabel_topology_diagnostics(
        X,
        Y,
        {"candidate_indices": list(range(10))},
        config=ProfilerConfig(topology="persistent"),
        report_context={"skipped_diagnostics": []},
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert topology["skipped_reason"] == "no multilabel topology objects were computed"
    assert topology["boundary_topology"]["skipped_reason"] == (
        "too few boundary candidates"
    )
    assert topology["per_label_topology"] == []


def test_multilabel_topology_samples_large_candidate_sets() -> None:
    X, Y = _topology_data()
    topology = compute_multilabel_topology_diagnostics(
        X,
        Y,
        {"candidate_indices": list(range(80))},
        config=ProfilerConfig(topology="persistent", max_samples=35),
        report_context={"skipped_diagnostics": [], "densification_events": []},
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert topology["boundary_topology"]["sampling"]["sampled"] is True
    assert topology["boundary_topology"]["sample_size"] == 35
    assert topology["selected_label_count"] <= 3
    assert "topology_strength" in topology


def test_multilabel_topology_missing_ripser_is_safe(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ripser":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X, Y = _topology_data()
    context: dict[str, list[dict[str, str]]] = {"skipped_diagnostics": []}
    topology = compute_multilabel_topology_diagnostics(
        X,
        Y,
        {"candidate_indices": list(range(40))},
        config=ProfilerConfig(topology="persistent"),
        report_context=context,
        label_names=np.array(["a", "b", "c"], dtype=object),
    )

    assert topology["skipped_reason"] == "ripser is not installed"
    assert context["skipped_diagnostics"] == [
        {
            "name": "multilabel_persistent_topology",
            "reason": "ripser is not installed",
        }
    ]


def test_diagnose_multilabel_persistent_topology_serializes() -> None:
    X, Y = _topology_data()
    report = diagnose(
        X,
        Y,
        target_mode="multilabel",
        topology="persistent",
        return_report=True,
        random_state=0,
    )

    assert report.metrics["topology"]["target_type"] == "multilabel"
    assert "boundary_topology" in report.metrics["topology"]
    assert "per_label_topology" in report.metrics["topology"]
    assert report.to_json()


def test_sparse_multilabel_persistent_topology_avoids_global_densification() -> None:
    X, Y = _topology_data()
    report = diagnose(
        sparse.csr_matrix(X),
        sparse.csr_matrix(Y),
        target_mode="multilabel",
        topology="persistent",
        return_report=True,
        random_state=0,
        densify_policy="skip",
        max_dense_mb=1,
    )

    assert report.preprocessing["is_sparse"] is True
    assert report.metrics["topology"]["target_type"] == "multilabel"
