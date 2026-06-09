# separatix

`separatix` profiles labeled embedding spaces before classifier training and
returns transparent, confidence-aware guidance about apparent classification
complexity.

It does not choose the optimal classifier. It provides interpretable
pretraining diagnostics that help you decide whether a problem looks linear,
smoothly nonlinear, local or kernel-like, fragmented, bottlenecked, or simply
too unreliable to trust yet.

## Quick start

```python
from separatix import diagnose

recommendation = diagnose(X, y)
print(recommendation)
```

For a structured audit:

```python
from separatix import diagnose

report = diagnose(X, y, return_report=True)
print(report.recommendation_text)
print(report.to_json())
```

## Main behaviors

- Accepts dense NumPy arrays, SciPy sparse matrices, and pandas inputs.
- Supports binary and multiclass classification with string or numeric labels.
- Records warnings, skipped diagnostics, sampling, and densification events.
- Keeps optional topology features optional; the package works without `ripser`.

## Sparse inputs

Sparse matrices are accepted directly. Diagnostics that need dense data use a
shared densification policy that can fail, skip, or warn and sample before
densifying.

## Recommendation categories

- `linear_likely_sufficient`
- `smooth_nonlinear_recommended`
- `kernel_or_local_recommended`
- `high_capacity_or_partitioning_recommended`
- `feature_or_label_bottleneck_likely`
- `insufficient_data_or_unreliable_geometry`
- `inconclusive`

## Development

```bash
poetry install --with dev
poetry run pytest
poetry run ruff check separatix tests
poetry run mypy separatix
```
