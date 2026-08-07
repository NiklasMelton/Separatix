"""Tests for the shared, target-neutral recommendation vocabulary."""

from separatix.constants import RECOMMENDATION_LABELS

_CANONICAL_RECOMMENDATION_LABELS = {
    "linear_likely_sufficient",
    "smooth_nonlinear_recommended",
    "kernel_or_local_recommended",
    "high_capacity_or_partitioning_recommended",
    "feedforward_mlp_recommended",
    "feature_or_target_bottleneck_likely",
    "insufficient_data_or_unreliable_geometry",
    "inconclusive",
}


def test_recommendation_label_vocabulary_is_exactly_canonical() -> None:
    """Expose exactly one shared label for each recommendation outcome."""
    assert set(RECOMMENDATION_LABELS) == _CANONICAL_RECOMMENDATION_LABELS
    assert len(RECOMMENDATION_LABELS) == 8
