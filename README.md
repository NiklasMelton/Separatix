[![separatix logo](https://raw.githubusercontent.com/NiklasMelton/Separatix/develop/img/separatix_logo.png)](https://github.com/NiklasMelton/Separatix)

# separatix

`separatix` profiles labeled feature spaces before classifier training and
returns transparent, confidence-aware guidance about apparent classification
complexity.

The intended use case includes learned embeddings, but the package is not
restricted to embeddings. It also works on raw feature matrices when you want a
coarse diagnostic of whether the observed class geometry looks mostly linear,
smoothly nonlinear, local or kernel-like, fragmented, bottlenecked, or too
unreliable to trust.

`separatix` does not claim to pick the optimal classifier. It is a pretraining
diagnostic and auditing tool designed to make its reasoning visible.

## Installation

```bash
pip install separatix
```

To install the latest development version directly from GitHub:

```bash
pip install "git+https://github.com/NiklasMelton/Separatix.git@develop"
```

## Quick start

```python
from separatix import diagnose

recommendation = diagnose(X, y, random_state=0)
print(recommendation)
```

For a structured audit:

```python
from separatix import diagnose

report = diagnose(X, y, return_report=True, random_state=0)
print(report.recommendation_text)
print(report.decision_path)
print(report.scores)
print(report.to_json())
```

## What It Accepts

- Dense NumPy arrays
- SciPy sparse matrices
- pandas DataFrames and Series when pandas is installed
- Binary and multiclass classification targets
- String or numeric labels treated as categorical class identifiers

Regression, multilabel classification, and multioutput classification are not
supported.

## What It Returns

By default, `diagnose(...)` returns a plain-text recommendation. With
`return_report=True`, it returns a `DiagnosticReport` that includes:

- the recommendation label
- plain-text recommendation text
- confidence level
- underlying metric groups
- normalized summary scores
- a visible decision path
- warnings and skipped diagnostics
- sampling and densification events
- preprocessing and runtime metadata

The report is JSON-serializable through `report.to_dict()` and `report.to_json()`.
The full report structure is documented in
[docs/report.md](/Users/niklasmelton/code/Separatix/docs/report.md).

## Recommendation Categories

- `linear_likely_sufficient`
- `smooth_nonlinear_recommended`
- `kernel_or_local_recommended`
- `high_capacity_or_partitioning_recommended`
- `feature_or_label_bottleneck_likely`
- `insufficient_data_or_unreliable_geometry`
- `inconclusive`

These categories are intentionally coarse. They describe the apparent geometry
and difficulty of the labeled feature space, not a guaranteed best model choice.

## Decision Pipeline

The recommendation is produced by a fixed, inspectable pipeline:

1. Validate inputs and encode labels.
2. Audit class counts, imbalance, sparsity, and basic dataset conditions.
3. Compute geometry, neighborhood, and boundary-related diagnostics.
4. Run simple probe models and compare them to a dummy baseline.
5. Aggregate the raw metrics into normalized scores such as signal,
   linearity, nonlinearity, overlap, fragmentation, and reliability.
6. Apply explicit rule-based branching to map those scores to a recommendation
   category and confidence level.
7. Render both a plain-language summary and a structured report.

The full rationale and decision rules are documented in
[docs/decision_pipeline.md](/Users/niklasmelton/code/Separatix/docs/decision_pipeline.md).

## Sparse Inputs And Memory Behavior

Sparse matrices are accepted directly. Diagnostics that need dense data use a
shared densification policy rather than a separate dense-only code path. When a
step would require densification, `separatix` can fail, skip, or warn and
subsample before densifying, depending on configuration. These events are
recorded in the report.

## Examples

- [examples/basic_breast_cancer.py](/Users/niklasmelton/code/Separatix/examples/basic_breast_cancer.py)
- [examples/linear_hyperplane_visual.py](/Users/niklasmelton/code/Separatix/examples/linear_hyperplane_visual.py)
- [examples/curvilinear_boundary_visual.py](/Users/niklasmelton/code/Separatix/examples/curvilinear_boundary_visual.py)
- [examples/high_dimensional_linear_hyperplane.py](/Users/niklasmelton/code/Separatix/examples/high_dimensional_linear_hyperplane.py)
- [examples/high_dimensional_curvilinear_hyperplane.py](/Users/niklasmelton/code/Separatix/examples/high_dimensional_curvilinear_hyperplane.py)
- [examples/moons_vs_linear.py](/Users/niklasmelton/code/Separatix/examples/moons_vs_linear.py)
- [examples/circles_kernel_signal.py](/Users/niklasmelton/code/Separatix/examples/circles_kernel_signal.py)
- [examples/multiclass_wine.py](/Users/niklasmelton/code/Separatix/examples/multiclass_wine.py)
- [examples/sparse_text_like_embeddings.py](/Users/niklasmelton/code/Separatix/examples/sparse_text_like_embeddings.py)

## Related Work

This package is not an implementation of a published dataset-complexity
procedure, but the project is adjacent to and inspired by prior work on
classification complexity and data geometry. In particular, would like to acknowledge:

- Ho and Basu, "Complexity Measures of Supervised Classification Problems"
  ([PDF](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2002-IEEE-TPAMI-Ho-DC.pdf))
- Lorena, Garcia, Lehmann, Souto, and Ho, "How Complex Is Your
  Classification Problem? A Survey on Measuring Classification Complexity"
  ([DOI](https://doi.org/10.1145/3347711),
  [PDF](https://dl.acm.org/doi/epdf/10.1145/3347711))

We do not follow those procedures directly, but they are relevant background
for why geometry-aware pretraining diagnostics are useful.
