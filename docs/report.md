# Diagnostic Report Reference

This document describes the structured report returned by:

```python
from separatix import diagnose

report = diagnose(X, y, return_report=True)
```

The returned object is a `DiagnosticReport`. It can be inspected as Python
attributes, converted to a JSON-serializable dictionary with `report.to_dict()`,
or serialized with `report.to_json()`.

The report is designed for two audiences:

- people reading the diagnostic result and deciding what to try next
- software that parses reports for auditing, dashboards, or experiment tracking

The report is intentionally transparent, but it is not a promise that
`separatix` has selected an optimal classifier. It records evidence about the
observed labeled geometry and explains how that evidence led to a coarse model
complexity recommendation.

## Stability And Parsing Contract

Every `DiagnosticReport` has the same top-level fields documented below. Parser
authors should treat those top-level field names as the most stable part of the
report.

Nested metric dictionaries are more diagnostic-specific. They may gain new keys
over time as new measurements are added. Existing parsers should therefore:

- require only the documented top-level fields
- check whether optional nested keys exist before reading them
- treat `null` values as "not computed", "not applicable", or "unavailable"
- treat a `skipped_reason` key as an explicit signal that a diagnostic did not
  run
- preserve unknown keys when storing or forwarding reports

The JSON representation contains only JSON-compatible values: objects, arrays,
strings, numbers, booleans, and `null`.

## Top-Level Structure

The top-level report dictionary contains these fields:

```text
recommendation
recommendation_text
confidence
metrics
scores
interpretations
decision_path
warnings
errors
skipped_diagnostics
preprocessing
sampling
densification_events
class_summary
runtime
config
```

All of these fields are always present on a successful run.

## Recommendation Fields

### `recommendation`

Always present. A machine-readable string category.

Current possible values are:

- `linear_likely_sufficient`
- `smooth_nonlinear_recommended`
- `kernel_or_local_recommended`
- `high_capacity_or_partitioning_recommended`
- `feature_or_label_bottleneck_likely`
- `insufficient_data_or_unreliable_geometry`
- `inconclusive`

These labels are intentionally coarse. They describe the apparent structure of
the classification problem, not a guaranteed best estimator.

### `recommendation_text`

Always present. A human-readable plain-text rendering of the recommendation,
including a headline, a short rationale, confidence, suggestions, and caveats
when available.

This field is meant for display. Parsers should prefer `recommendation`,
`confidence`, `scores`, and `decision_path` for structured logic.

### `confidence`

Always present. One of:

- `low`
- `medium`
- `high`

Confidence summarizes how reliable the diagnostic process appeared, based on
factors such as skipped diagnostics, warnings, class counts, imbalance, distance
concentration, boundary sample size, and probe stability.

## Decision Explanation

### `decision_path`

Always present. A list of human-readable strings describing the branch or
branches that led to the final recommendation.

The list is ordered. Earlier entries usually describe the main decision gate,
and later entries may add supporting evidence. The exact wording is
human-facing, so parsers should not depend on exact sentence text. For automated
logic, use `recommendation`, `scores`, and selected `metrics`.

### `interpretations`

Always present. A dictionary mapping score families to short explanations.

Current keys include:

- `signal`
- `overlap`
- `linearity`
- `nonlinearity`
- `fragmentation`
- `reliability`

This section is explanatory. It is useful for displaying tooltips or report
legends, but it should not be treated as raw evidence.

## Scores

### `scores`

Always present. A dictionary of normalized summary scores. Values are floats in
the `[0, 1]` range when available, or `null` when a score could not be computed.

Current keys are:

| Key | Meaning |
| --- | --- |
| `signal_score` | How far the best available probe rises above the dummy baseline. Higher means labels appear more predictable from the features. |
| `overlap_score` | A summary of neighborhood class mixing. Higher means more local ambiguity or overlap. |
| `linearity_score` | How close the linear probe is to the best available probe. Higher means linear behavior explains most observed probe performance. |
| `nonlinearity_score` | How much the best nonlinear probe improves over the linear probe. Higher means nonlinear structure appears useful. |
| `fragmentation_score` | Boundary graph fragmentation. Higher means the estimated boundary looks more partitioned or locally broken up. |
| `topology_score` | Optional topology strength when persistent topology diagnostics ran. May be `null`. |
| `reliability_score` | Trust support for the diagnostic process itself. Higher means fewer reliability penalties. |

Scores are compact summaries used by the recommendation engine. They should be
read together with `metrics`, `warnings`, and `skipped_diagnostics`.

## Metrics

### `metrics`

Always present. A dictionary containing diagnostic families:

```text
audit
geometry
probes
baseline
neighborhood
boundary
graph
topology
```

Each family is always present, but individual keys inside a family may be
conditional.

### `metrics["audit"]`

Always present. Cheap dataset-level facts.

Always included:

- `n_samples`
- `n_features`
- `n_classes`
- `class_counts`
- `class_proportions`
- `imbalance_ratio`
- `is_sparse`
- `dtype`
- `estimated_dense_memory_mb`

Included for sparse inputs:

- `nnz`
- `density`
- `sparsity_fraction`

Included for dense inputs:

- `constant_feature_fraction`

`class_counts` and `class_proportions` use stringified original class labels as
keys. This keeps JSON object keys valid and stable across numeric or string
labels.

### `metrics["geometry"]`

Always present. Measures basic geometric reliability and dimensionality.

Current keys:

- `feature_scale_range_estimate`
- `effective_rank_estimate`
- `intrinsic_dimension_proxy`
- `distance_concentration_proxy`
- `high_dimensionality_flag`
- `sample_to_feature_ratio`
- `sampling`

`feature_scale_range_estimate` is `null` for sparse inputs. `distance_concentration_proxy`
may be `null` if dense distance computation was skipped under the configured
densification policy.

### `metrics["probes"]`

Always present. A dictionary of lightweight model probes. Current probe keys
are:

- `dummy`
- `linear`
- `knn`
- `smooth_poly`
- `kernel_approx`

The `dummy`, `linear`, and `knn` probes are normally present with evaluated
metrics. Evaluated probes include:

- `balanced_accuracy`
- `macro_f1`
- `accuracy`
- `per_class_recall`
- `model_name`
- `runtime_seconds`
- `sample_info`
- `evaluation_mode`
- `predictions`
- `stability_repeats`
- `stability_balanced_accuracy_mean`
- `stability_balanced_accuracy_std`

For multiclass problems, evaluated probes may also include:

- `most_confused_pairs`

`evaluation_mode` is usually `cross_validation`. For very small class counts it
may be `resubstitution_low_reliability`, and a warning is recorded.

`smooth_poly` may contain evaluated metrics or a skip entry. When evaluated, it
also includes metadata such as:

- `probe_degree`
- `original_feature_count`
- `estimated_expanded_feature_count`
- `estimated_expanded_mb`
- `probe_variant`

Depending on dimensionality and memory budget, `probe_variant` may be
`full_quadratic` or `low_rank_quadratic`. The low-rank variant also includes:

- `sketch_n_components`
- `estimated_sketch_mb`

When `smooth_poly` is not run, it includes:

- `skipped_reason`
- `model_name`
- usually `sample_info`

`kernel_approx` may be evaluated or skipped. It is skipped in the `fast` budget
and may also be skipped if required dense conversion is unavailable. Skipped
probe entries include `skipped_reason`.

The `predictions` arrays are primarily for audit and debugging. They are encoded
class ids, not necessarily the original label values.

### `metrics["baseline"]`

Always present. Summarizes the probe family.

Current keys:

- `best_probe`
- `best_probe_score`

Values may be `null` if no evaluated probe score is available.

### `metrics["neighborhood"]`

Always present. Measures local class mixing and ambiguity.

When computed, current keys include:

- `mean_local_entropy`
- `high_entropy_fraction`
- `same_class_neighbor_fraction`
- `cross_class_neighbor_fraction`
- `nearest_enemy_distance_estimate`
- `mean_local_ambiguity`
- `local_entropy`
- `local_ambiguity`
- `sampling`

For very small inputs, summary values are still returned and list-valued local
details may be absent. If sparse nearest-neighbor computation requires dense
conversion and that conversion is unavailable, this section may contain:

- `sampling`
- `skipped_reason`

### `metrics["boundary"]`

Always present. Describes candidate samples near likely class boundaries.

Current keys:

- `candidate_indices`
- `candidate_fraction`
- `boundary_sample_size`
- `class_composition`
- `warning`

`candidate_indices` are integer row positions in the data used by the boundary
diagnostic. When upstream neighborhood details were unavailable, this section
contains an empty candidate list and a warning.

`warning` is either a string or `null`.

### `metrics["graph"]`

Always present. Measures fragmentation over boundary candidates.

Current keys:

- `component_count`
- `largest_component_fraction`
- `component_size_entropy`
- `small_component_count`
- `cross_class_edge_density`
- `graph_fragmentation_score`
- `sampling`
- `warning`

When there are too few boundary candidates, graph diagnostics return default
low-fragmentation values and include `warning`. In that case `sampling` may be
absent.

### `metrics["topology"]`

Always present. Persistent topology is optional and may not run.

This section always includes at least:

- `mode`

When topology is disabled, not requested, infeasible, or unavailable, it also
includes:

- `skipped_reason`

Common skip reasons include:

- `topology disabled`
- `persistent topology not requested`
- `persistent topology disabled for this budget`
- `too few boundary candidates`
- `too many boundary candidates`
- `geometry reliability too low`
- `ripser is not installed`
- `dense conversion unavailable`

When persistent topology runs successfully, current keys include:

- `h0_persistence_count`
- `h1_persistence_count`
- `total_h0_persistence`
- `total_h1_persistence`
- `max_h1_persistence`
- `boundary_scale`
- `relative_h1_persistence`
- `topology_strength`
- `persistence_entropy`

Because topology is optional supporting evidence, parser logic should be robust
to this section containing only `mode` and `skipped_reason`.

## Warnings, Errors, And Skips

### `warnings`

Always present. A list of warning strings emitted during the run and recorded in
the report.

Examples include low-reliability probe evaluation and densification notices.
The list may be empty.

### `errors`

Always present. A list reserved for recoverable errors. It is currently empty
on successful reports.

Input validation failures and unrecoverable densification failures raise Python
exceptions instead of returning a report.

### `skipped_diagnostics`

Always present. A list of dictionaries describing diagnostics that did not run.
The list may be empty.

Each entry currently has:

- `name`
- `reason`

Examples:

```json
{"name": "persistent_topology", "reason": "ripser is not installed"}
```

```json
{"name": "smooth_nonlinear_probe", "reason": "quadratic expansion and low-rank sketch exceed configured memory budget"}
```

Skips can also be visible inside the relevant metric family through
`skipped_reason`. Parser authors should check both places: `skipped_diagnostics`
is the run-level index, while nested `skipped_reason` explains the state of a
specific metric entry.

## Preprocessing, Sampling, And Densification

### `preprocessing`

Always present. Summarizes the accepted input representation.

Current keys:

- `input_type`
- `is_sparse`

`input_type` reflects the internal validated matrix object type, not necessarily
the exact user object originally passed to `diagnose`.

### `sampling`

Always present. A dictionary collecting sampling summaries for major diagnostic
families.

Current keys:

- `probe`
- `neighbors`
- `boundary`

Each value is either a sampling dictionary or `null`. Sampling dictionaries
currently include:

- `reason`
- `sampled`
- `n_original`
- `n_used`

`sampled` is `true` when the diagnostic used a stratified subsample because of
budget or `max_samples` limits.

The same sampling dictionaries may also appear nested under specific metric
families, such as `metrics["geometry"]["sampling"]` or individual probe
`sample_info`.

### `densification_events`

Always present. A list describing dense conversions attempted for sparse input.
The list is empty when no sparse-to-dense conversion was needed.

Current event keys:

- `operation`
- `reason`
- `input_shape`
- `estimated_full_dense_mb`
- `max_dense_mb`
- `policy`
- `sampling_used`
- `n_original`
- `n_used`
- `status`

Current `status` values include:

- `performed`
- `performed_on_subsample`
- `skipped`
- `skipped_too_small`

Dense-only diagnostics use this shared mechanism. This makes sparse behavior
auditable without maintaining a separate sparse-only report shape.

## Class Summary

### `class_summary`

Always present. A compact label summary.

Current keys:

- `n_classes`
- `classes`
- `class_counts`
- `imbalance_ratio`
- `min_class_count`
- `max_class_count`

`classes` contains original class labels as JSON-compatible values. Numeric
labels are treated as categorical identifiers, not regression targets.

`class_counts` mirrors the audit section and uses stringified label keys.

## Runtime And Configuration

### `runtime`

Always present.

Current keys:

- `total_seconds`

This is wall-clock elapsed time for the full diagnostic run.

### `config`

Always present. The profiler configuration used for the run.

Current keys:

- `budget`
- `topology`
- `densify_policy`
- `max_dense_mb`
- `max_samples`
- `min_dense_samples`
- `random_state`
- `warn_on_densify`
- `n_jobs`

This section is useful for reproducibility and for explaining why diagnostics
were skipped or sampled.

## Minimal Parser Example

This example reads only stable top-level fields and handles optional nested
fields defensively:

```python
data = report.to_dict()

recommendation = data["recommendation"]
confidence = data["confidence"]
scores = data["scores"]

topology = data["metrics"]["topology"]
if "skipped_reason" in topology:
    topology_state = f"skipped: {topology['skipped_reason']}"
else:
    topology_state = f"strength: {topology.get('topology_strength')}"

skipped_names = [
    item.get("name")
    for item in data["skipped_diagnostics"]
    if isinstance(item, dict)
]
```

## Practical Reading Guide

For a quick human read, start with:

1. `recommendation`
2. `confidence`
3. `decision_path`
4. `scores["reliability_score"]`
5. `warnings` and `skipped_diagnostics`

For automated experiment tracking, store at least:

- `recommendation`
- `confidence`
- all `scores`
- `class_summary`
- `config`
- `warnings`
- `skipped_diagnostics`
- `sampling`
- `densification_events`

For detailed audits, preserve the full `metrics` dictionary, including unknown
future keys.
