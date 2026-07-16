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
