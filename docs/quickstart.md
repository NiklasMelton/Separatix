# Quickstart

## A plain-text recommendation

Pass a two-dimensional feature matrix and a target with one row per sample:

```python
from sklearn.datasets import load_breast_cancer

from separatix import diagnose

dataset = load_breast_cancer()
recommendation = diagnose(dataset.data, dataset.target, random_state=0)
print(recommendation)
```

The default return value is a concise recommendation intended for interactive
use. Set `random_state` when you want repeatable sampling and probe evaluation.

## A structured audit report

Request a {class}`~separatix.DiagnosticReport` when the result will be audited,
stored, or processed:

```python
report = diagnose(
    dataset.data,
    dataset.target,
    return_report=True,
    random_state=0,
)

print(report.recommendation)
print(report.confidence)
print(report.recommendation_text)
print(report.decision_path)
print(report.scores)

payload = report.to_json()
```

The default serialized representation is terse and removes large row-level
arrays. Use `report.to_dict(terse=False)` or `report.to_json(terse=False)` only
when those detailed arrays are required.

The target-specific recommendation evidence exposes both the conservative
minimum family and the broader competitive frontier. For a single-label run:

```python
family_set = report.metrics["recommendation_evidence"]["plausible_family_set"]
print(family_set["minimum_recommended_family"])
print(family_set["plausible_families"])
```

Constructed probe entries also contain versioned estimator recipes. The public
factory reconstructs a fresh unfitted estimator for an audit; see
[Reading a diagnostic report](reports.md#reconstructing-an-audited-probe) for
the reproducibility boundaries.

```python
from separatix import make_probe_estimator

recipe = report.metrics["probes"]["linear"]["probe_recipe"]
estimator = make_probe_estimator(recipe)
```

## Estimator-style use

Use {class}`~separatix.ComplexityProfiler` when configuration should be reused
or kept alongside the fitted result:

```python
from separatix import ComplexityProfiler

profiler = ComplexityProfiler(
    budget="standard",
    topology="off",
    random_state=0,
)
profiler.fit(dataset.data, dataset.target)

print(profiler.recommendation())
report = profiler.report()
```

The profiler performs diagnostics during `fit`; it is not a predictive
estimator and does not expose `predict`.

## Two-stage experimental protocol

Separatix recommendations can depend on the amount of labeled training data.
In an experiment with a final held-out test set, diagnose the feature space once
at selection time and again at the training size used for the final fit.

Assume `X_train`, `X_validation`, and `X_test` are disjoint NumPy arrays, with
matching target arrays:

```python
import numpy as np

from separatix import diagnose

# Stage 1: guide candidate-family selection using only the training cohort.
selection_report = diagnose(
    X_train,
    y_train,
    return_report=True,
    random_state=0,
)

# Tune and compare actual candidate models using X_validation/y_validation.
# Separatix is diagnostic evidence, not a replacement for that validation.

# Stage 2: after validation-based tuning, match the cohort for the final fit.
X_development = np.concatenate([X_train, X_validation], axis=0)
y_development = np.concatenate([y_train, y_validation], axis=0)
deployment_report = diagnose(
    X_development,
    y_development,
    return_report=True,
    random_state=0,
)

# Record deployment_report, settle the final model specification, fit it on
# X_development/y_development, and evaluate it once on X_test/y_test.
```

Use the same Separatix configuration in both stages when you want the reports to
be directly comparable. The first report describes the evidence available when
candidate families are selected. The second describes the enlarged cohort on
which the final model will train. A changed recommendation is useful evidence
of training-size sensitivity; it is not, by itself, proof that either family
will perform best.

The held-out folds used internally by Separatix compare diagnostic probes. They
are not an independent estimate of the performance of the final tuned model and
do not replace the final test set.

Do not pass test rows or test labels to either diagnostic run, preprocessing
fit, hyperparameter search, stopping decision, or model-family decision. If the
second report changes the final specification, make and record that decision
before inspecting test performance. The untouched test set then evaluates the
entire development procedure, including the second diagnostic.

With group-dependent observations, split by group before both stages and pass
the matching `groups` arrays to `diagnose`. When the cohorts are sparse matrices
or pandas objects, use their native row-stacking operation instead of the NumPy
concatenation shown above.
