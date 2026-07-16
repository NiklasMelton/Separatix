[![separatix logo](https://raw.githubusercontent.com/NiklasMelton/Separatix/develop/img/separatix_logo.png)](https://github.com/NiklasMelton/Separatix)

# Separatix

`separatix` profiles labeled feature spaces before supervised model training
and returns transparent, confidence-aware guidance about apparent classification
or regression complexity.

The intended use case includes learned embeddings, but the package is not
restricted to embeddings. It also works on raw feature matrices when you want a
coarse diagnostic of whether the observed supervised geometry looks mostly
linear, smoothly nonlinear, local or kernel-like, fragmented or discontinuous,
bottlenecked, or too unreliable to trust.

`separatix` does not claim to pick the optimal classifier or regressor. It is a
pretraining diagnostic and auditing tool designed to make its reasoning visible.

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
- Multilabel binary indicator targets with `target_mode="multilabel"` or
  auto-detection for unambiguous 2D indicators
- Continuous single- or multi-target regression with explicit
  `target_mode="regression"`
- String or numeric labels treated as categorical class identifiers

Regression is opt-in so numeric class identifiers remain categorical by
default. General multioutput classification is not supported.

## What It Returns

By default, `diagnose(...)` returns a plain-text recommendation. With
`return_report=True`, it returns a `DiagnosticReport` that includes:

- the recommendation label
- plain-text recommendation text
- confidence level
- underlying metric groups
- probe-family evidence, including uncertainty-aware family comparisons
- normalized summary scores
- a visible decision path
- warnings and skipped diagnostics
- sampling and densification events
- preprocessing and runtime metadata

The report is JSON-serializable through `report.to_dict()` and `report.to_json()`.
Non-finite diagnostic values are represented as JSON `null`; `to_json()` never
emits non-standard `NaN` or infinity literals. The default terse form removes
large row-level arrays before copying them.

For multilabel targets, `separatix` compares probe families across micro F1,
macro F1, and sample Jaccard rather than collapsing the evidence into a single
weighted score. Optional iterative multilabel stratification can be installed
with:

```bash
pip install "separatix[multilabel]"
```

For regression targets, call `diagnose(X, y, target_mode="regression")`.
Regression evidence is compared across variance-weighted R2 and uniform-average
R2, with normalized RMSE and target-neighborhood smoothness as supporting
diagnostics. Classification-only boundary and fragmentation diagnostics are
marked not applicable and do not reduce regression confidence.

All non-dummy probe families learn feature scaling inside each validation
training fold. Sparse probes use non-centering scaling. Geometry and topology
continue to describe the supplied, unscaled coordinate space, and the report
records both choices under `preprocessing`.

Optional feed-forward MLP probes can be installed and enabled explicitly:

```bash
pip install "separatix[mlp]"
```

Set `mlp_probes=True` and use `mlp_device`, `mlp_trigger_skill_threshold`,
`mlp_min_improvement`, and `mlp_max_parameters` to control them. An MLP can
override simpler-family guidance only with complete held-out evidence; failed
or infeasible group splits never fall back to in-sample override evidence.

Optional persistent-topology diagnostics can be installed with:

```bash
pip install "separatix[tda]"
```

For multilabel targets, persistent topology is supporting evidence only. When
enabled, it is computed on capped boundary-candidate subsets and a small capped
set of high-support label-positive subsets.

For regression targets, optional topology is computed only on capped
high-residual and high-local-discontinuity subsets. `topology="graph"` uses a
sparse-compatible mutual-nearest-neighbor component summary;
`topology="persistent"` adds persistent homology when `ripser` is installed.
`topology="auto"` skips topology under the fast budget and otherwise attempts
both summaries. Regression topology is descriptive supporting evidence: it is
included in the report but never changes the recommendation label or confidence.

## Recommendation Categories

- `linear_likely_sufficient`
- `smooth_nonlinear_recommended`
- `kernel_or_local_recommended`
- `high_capacity_or_partitioning_recommended`
- `feedforward_mlp_recommended`
- `feature_or_label_bottleneck_likely`
- `insufficient_data_or_unreliable_geometry`
- `inconclusive`
- `linear_response_likely_sufficient`
- `smooth_nonlinear_response_recommended`
- `kernel_or_local_regression_recommended`
- `higher_capacity_or_partitioning_regression_recommended`
- `feedforward_mlp_regression_recommended`
- `feature_or_target_bottleneck_likely`
- `insufficient_data_or_unreliable_regression_geometry`
- `inconclusive_regression_diagnostic`

These categories are intentionally coarse. They describe the apparent geometry
and difficulty of the labeled feature space, not a guaranteed best model choice.

The synthetic recommendation ladder below shows how `separatix` responds as the
designed dataset geometry moves from simple linear structure toward smoother
nonlinearity, local or kernel-like structure, fragmented boundaries, and
finally weak-signal or random-label bottlenecks. The x-axis is the intended
dataset complexity, while the y-axis is the coarse recommendation level
reported by `separatix`.

![separatix recommendation complexity ladder](https://raw.githubusercontent.com/NiklasMelton/Separatix/develop/img/separatix_recommendation_complexity_ladder.png)

## Decision Pipeline

The recommendation is produced by a fixed, inspectable pipeline:

1. Validate inputs and encode labels.
2. Audit class counts, imbalance, sparsity, and basic dataset conditions.
3. Compute geometry, neighborhood, boundary, fragmentation, and optional
   topology diagnostics, using a distinct multilabel path for binary indicator
   targets.
4. Run simple probe models and compare them to a dummy baseline.
5. Build probe-family evidence with uncertainty estimates for `linear`,
   `smooth_nonlinear`, and `local_kernel`.
6. Apply a 95% signal-vs-dummy gate before making any model-family
   recommendation for single-label targets, or a two-of-three primary-metric
   signal gate for multilabel targets.
7. Use conservative escalation: keep the simpler family unless a more complex
   family has a clear uncertainty-adjusted advantage.
8. Treat fragmentation and optional topology as supporting structural evidence,
   not as shortcuts around weak probe evidence.
9. Render both a plain-language summary and a structured report, including
   `raw_best_family` and `recommended_family` when a report is requested.

The full rationale and decision rules are documented in
[docs/decision_pipeline.md](docs/decision_pipeline.md).

## Sparse Inputs And Memory Behavior

Sparse matrices are accepted directly. Diagnostics that need dense data use a
shared densification policy rather than a separate dense-only code path. When a
step would require densification, `separatix` can fail, skip, or warn and
subsample before densifying, depending on configuration. These events are
recorded in the report.

`max_samples` and `max_dense_mb` are hard limits. Group-aware sampling never
splits a group or exceeds the row cap. If no support-preserving sample fits,
the affected supervised diagnostic is skipped and reliability is marked
insufficient instead of silently dropping classes or labels. The dense-memory
budget applies to sparse multilabel targets as well as feature matrices.

When `groups` are supplied, sampling keeps groups whole and predictive evidence
must come from group-disjoint held-out splits. Each evaluated class or label
side needs support in both training and test partitions. A single group, an
oversized group, or a class confined to too few groups therefore causes the
affected supervised evidence to be skipped instead of evaluated on its training
rows. Geometry and topology remain descriptive in those cases.

Numeric one-dimensional targets—including non-integral values—remain
categorical unless `target_mode="regression"` is explicit. High-cardinality
numeric classification targets produce a warning to make accidental routing
visible.

## Examples

- [examples/basic_breast_cancer.py](examples/basic_breast_cancer.py)
- [examples/linear_hyperplane_visual.py](examples/linear_hyperplane_visual.py)
- [examples/curvilinear_boundary_visual.py](examples/curvilinear_boundary_visual.py)
- [examples/high_dimensional_linear_hyperplane.py](examples/high_dimensional_linear_hyperplane.py)
- [examples/high_dimensional_curvilinear_hyperplane.py](examples/high_dimensional_curvilinear_hyperplane.py)
- [examples/moons_vs_linear.py](examples/moons_vs_linear.py)
- [examples/circles_kernel_signal.py](examples/circles_kernel_signal.py)
- [examples/recommendation_complexity_ladder.py](examples/recommendation_complexity_ladder.py)
- [examples/multiclass_wine.py](examples/multiclass_wine.py)
- [examples/openml_multilabel_yeast.py](examples/openml_multilabel_yeast.py)
- [examples/sparse_text_like_embeddings.py](examples/sparse_text_like_embeddings.py)

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
