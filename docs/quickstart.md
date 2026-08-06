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
