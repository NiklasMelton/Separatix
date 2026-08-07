# Reading a diagnostic report

A {class}`~separatix.DiagnosticReport` is intended to make both the conclusion
and its limitations visible.

## Start with these fields

`recommendation`
: Stable machine-readable recommendation label.

`recommendation_text`
: Plain-language summary suitable for display.

`confidence`
: Coarse `high`, `medium`, or `low` evidence-quality label.

`decision_path`
: Ordered explanation of the gates and comparisons used to reach the result.

`scores`
: Normalized summaries such as signal, linearity, overlap, fragmentation, and
  reliability where applicable.

## Audit the evidence

The `metrics` mapping retains diagnostic-family outputs and recommendation
evidence. Depending on the target path, inspect `recommendation_evidence`,
`multilabel_recommendation_evidence`, or `regression_recommendation_evidence`.
These objects distinguish the numerically strongest `raw_best_family` from the
conservatively selected `recommended_family`.

Each target-specific evidence object also contains `plausible_family_set`. Its
main fields are:

- `status`: `available`, `not_applicable`, or `unavailable`
- `minimum_recommended_family`: the simplest family supported by the ordinary
  conservative recommendation rule, or `null` when primary metrics disagree
- `plausible_families`: the ordered, uncertainty-aware competitive frontier
- `decision_method`: `paired_oof_bootstrap`,
  `marginal_standard_error_fallback`, `mixed`, or `null` when no eligible
  pairwise comparison was needed
- `assessments`: per-family availability, complexity eligibility, dominance,
  and inclusion reasons

The scope is deliberately limited to the core `linear`, `smooth_nonlinear`, and
`local_kernel` probes. An MLP override or a high-capacity/partitioning upgrade
does not enter this set. A `not_applicable` result means the signal gate failed
or blocking evidence prevents a family interpretation. `unavailable` means a
required core probe or primary metric was missing, so the frontier could not be
completed.

This plausible set is a heuristic diagnostic frontier, not a formal confidence
set, equivalence test, or assertion that all retained families have equal
performance. It answers the narrower question: which tested core families have
not been clearly ruled out under the target-specific comparison policy?

`probe_evaluation` describes the shared evaluation cohort and materialized fold
plan. `paired_probe_comparisons` contains probe-level paired bootstrap deltas;
the target-specific recommendation evidence identifies whether each decision
used that paired evidence or the marginal fallback. Full reports retain
row-to-fold assignments for audits, while terse serialization prunes them. The
paired comparisons are conditional on the best probe selected to represent each
family; they do not adjust for selecting that representative from the same OOF
evidence.

The same `probe_evaluation` object contains
`effective_train_size_summary`, a per-run summary of rows used to fit the shared
ordinary probes:

```python
evaluation = report.metrics["probe_evaluation"]
summary = evaluation["effective_train_size_summary"]
```

Its schema is fixed:

- `status`: `"available"` when the fit row counts are known, otherwise
  `"unavailable"`.
- `basis`: `"held_out_folds"` for cross-validation fold fits,
  `"resubstitution"` for an ungrouped no-split fallback, or `None` when the
  summary is unavailable.
- `min`, `median`, `mean`, and `max`: the smallest, median, arithmetic mean,
  and largest fit-row counts. The first and last are integers; the median and
  mean are floats. They are `None` (JSON `null`) when `status` is
  `"unavailable"`.
- `mean_fraction_of_evaluation_cohort`: `mean` divided by
  `evaluation["n_samples"]`, or `None` (JSON `null`) when unavailable.

`evaluation["n_samples"]` is the denominator for this fraction and is the
number of rows in the shared evaluation cohort after any memory-aware sampling
or densification. It is therefore the post-sampling evaluation denominator,
not necessarily the original input row count. For `"held_out_folds"`, the
summary is derived from the existing `train_sizes`; for `"resubstitution"`,
the full post-sampling cohort is the fit size. If no ordinary probe fit or
usable evaluation basis can provide counts, all numeric fields are `None`.
Optional MLP probes are outside this summary. The summary is descriptive
metadata for one run, not an implemented learning curve or size-sensitivity
analysis.

The package compares simple families first and requires clear evidence before
escalating. Geometry and topology support the explanation; they do not bypass a
weak predictive-signal gate.

## Audit limitations

Always inspect:

- `warnings` and `errors`
- `skipped_diagnostics`
- `densification_events`
- `sampling`
- `grouping`
- `preprocessing`
- `config` and `runtime`

A recommendation can be technically valid while still carrying low confidence
because support-preserving evaluation was infeasible, important diagnostics
were skipped, or probe improvements were borderline.

## Serialization

```python
terse_dict = report.to_dict()
terse_json = report.to_json(indent=2)

full_dict = report.to_dict(terse=False)
full_json = report.to_json(indent=2, terse=False)
```

The terse form is the recommended storage format. It prunes large row-level
arrays before copying them. Non-finite values are serialized as JSON `null`, so
the JSON output never contains non-standard `NaN` or infinity literals.

## Reconstructing an audited probe

Constructed probe results include a versioned `probe_recipe` and a
`probe_recipe_status`. Recipes record the resolved preprocessing and estimator
graph, hyperparameters, data-dependent dimensions, training policy, and the
runtime environment that created them. Environment versions are populated from
the installed runtime rather than copied from static package metadata.

```python
from separatix import make_probe_estimator

recipe = report.metrics["probes"]["linear"]["probe_recipe"]
estimator = make_probe_estimator(recipe)
estimator.fit(X_train, y_train)
```

The factory uses a fixed allowlist of supported scikit-learn and Separatix probe
components and returns an unfitted estimator. It never imports an arbitrary
class named by serialized input. Its `version_policy` argument controls whether
differences between the recorded and current Python/library environments warn,
raise an error, or are ignored:

```python
estimator = make_probe_estimator(recipe, version_policy="error")
```

A recipe describes the resolved unfitted estimator configuration and records
the diagnostic's fit-policy metadata. The factory reconstructs estimator
parameters, but it does not replay the evaluation cohort, validation split plan,
or scoring-time orchestration. The training policy is audit metadata rather than
instructions automatically applied by the factory. Consumers must inspect and
honor any `training_policy.scoring_time_estimator_adjustments`, such as
fold-local kNN neighbor reduction, when reproducing a diagnostic evaluation.
Recipes do not contain fitted coefficients or claim bit-for-bit reproducibility
across library versions. Skipped probes expose an unavailable recipe status and
a reason rather than a recipe for an estimator that was never constructed.
