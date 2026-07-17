# Inputs and target modes

## Feature matrices

`X` must be a numeric, finite, two-dimensional matrix with at least one feature.
Supported containers are dense NumPy arrays, SciPy sparse matrices, and pandas
DataFrames when pandas is installed. Sparse matrices are accepted directly;
dense-only diagnostics follow the configured densification policy.

## Single-label classification

One-dimensional targets route to single-label classification by default.
Binary, multiclass, string, and numeric labels are supported:

```python
report = diagnose(X, y, target_mode="singlelabel", return_report=True)
```

Numeric labels remain categorical even when they are non-integral. This avoids
silently interpreting class identifiers as regression values.

## Multilabel classification

An unambiguous two-dimensional binary indicator matrix is detected as
multilabel in `target_mode="auto"`. Specify the mode explicitly for clarity or
for a one-column indicator target:

```python
report = diagnose(
    X,
    Y,
    target_mode="multilabel",
    return_report=True,
    random_state=0,
)
```

Each target column represents one label. Dense and sparse indicator matrices
are supported. General multioutput classification, where each column is a
separate multiclass task, is not supported.

## Regression

Regression is always opt-in:

```python
report = diagnose(
    X,
    y_continuous,
    target_mode="regression",
    return_report=True,
    random_state=0,
)
```

Single-target and multi-target continuous regression are supported. Constant
target columns are reported and excluded from probe-family scoring. Regression
topology is descriptive only and cannot alter the recommendation label or
confidence.

## Group identifiers

Pass one group identifier per row when related samples must not cross an
evaluation boundary:

```python
report = diagnose(X, y, groups=subject_ids, return_report=True)
```

See [group-aware evaluation and memory controls](groups_and_memory.md) for the
failure and fallback behavior.
