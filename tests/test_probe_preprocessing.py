import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from separatix.config import ProfilerConfig
from separatix.models.probes import _linear_classifier, _prediction_evidence
from separatix.models.scoring import choose_cv, evaluate_estimator


def test_probe_scaler_is_fit_only_on_training_folds(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.asarray([0, 1] * 50)
    fit_sizes: list[int] = []
    original_fit = StandardScaler.fit

    def recording_fit(self, values, y=None, sample_weight=None):
        fit_sizes.append(int(values.shape[0]))
        return original_fit(self, values, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    cv, _ = choose_cv(y, 5, random_state=0)
    evaluate_estimator(_linear_classifier(X), X, y, cv=cv)

    assert fit_sizes
    assert max(fit_sizes) < X.shape[0]


def test_sparse_probe_scaling_does_not_center() -> None:
    X = sparse.csr_matrix(np.eye(8))
    estimator = _linear_classifier(X)
    scaler = estimator.named_steps["scale"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.with_mean is False


def test_oversized_row_prediction_evidence_is_not_retained() -> None:
    predictions = np.zeros((140_000, 1), dtype=float)
    evidence = _prediction_evidence(
        predictions,
        ProfilerConfig(max_dense_mb=1),
    )
    assert evidence["predictions"] is None
    assert evidence["prediction_evidence_bytes"] > 1024**2
    assert "max_dense_mb" in evidence["predictions_omitted_reason"]
