# Frequently asked questions

## Does separatix choose the best model?

No. It gives coarse, confidence-aware guidance about apparent supervised
difficulty and model-family complexity. Candidate models still need proper
task-specific validation and tuning.

## Why are numeric labels treated as classification?

Numeric class identifiers are common. Regression is therefore opt-in with
`target_mode="regression"`, preventing identifiers such as `0`, `1`, and `2`
from being interpreted as ordered continuous values.

## Why is the recommendation simpler than the best raw probe?

The policy is intentionally conservative. A more complex family must show a
clear uncertainty-adjusted advantage; a small numerical lead is not enough.
The report exposes both `raw_best_family` and `recommended_family`.

## Why was a diagnostic skipped?

Common reasons include insufficient class or label support, infeasible
group-disjoint splits, a dense-memory limit, a sample cap, a fast budget, or a
missing optional dependency. Inspect `skipped_diagnostics`, `warnings`,
`sampling`, and `densification_events` for the run-specific reason.

## Is topology required?

No. Graph and persistent-topology summaries are supporting structural evidence.
Set `topology="off"` to disable them. Persistent homology requires the `tda`
extra; graph summaries do not.

## Can I pass a pandas object?

Yes, when pandas is installed. DataFrame column names are retained where they
serve as regression target or multilabel names.

## Can I store the entire report?

Yes. `report.to_json()` is standards-compliant and terse by default. Full mode
retains large row-level diagnostic arrays and should be used only when needed.
