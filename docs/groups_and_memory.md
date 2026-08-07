# Group-aware evaluation and memory controls

## Leakage-aware groups

Use `groups` for repeated measurements, users, subjects, source documents, or
any other unit that must remain intact. Separatix then keeps groups whole during
sampling and requires predictive evidence to come from group-disjoint held-out
splits.

Every evaluated class or label side needs support in both training and test
partitions. A single group, an oversized group, or a class confined to too few
groups can therefore make supervised evaluation infeasible. In that case the
affected evidence is skipped and reported; it is not replaced by a misleading
in-sample result. Geometry and topology may remain as descriptive evidence.

Ordinary core-probe comparisons also respect the independence unit. Their
paired bootstrap resamples whole groups, rather than individual rows, and
single-label resamples that lose required class support are rejected. The
report records how many paired resamples were requested and successfully used.

Optional MLP probes use a separately capped, dense cohort, so their paired
bootstrap cache is local to that cohort rather than reused from the ordinary
probe comparison. Group-aware MLP sampling keeps groups whole, and MLP
held-out folds remain group-disjoint. The local cache also resamples whole
groups; single-label draws that lose class support are rejected, while any
non-finite target-aware score row is discarded. The full required comparator
results remain under `report.metrics["mlp_probes"]["aligned_comparators"]` even
though `pairwise_comparisons` retains only the best MLP versus dummy and the
metric-specific strongest simpler comparator. If support-preserving sampling or
fold construction fails, paired MLP comparison is marked `not_run`; if the
cache cannot retain enough valid paired rows, it is `unavailable`. Neither
condition is replaced with in-sample override evidence.

## Sparse inputs

Sparse feature matrices are accepted directly, and sparse-compatible operations
are used where practical. Probe preprocessing scales sparse features without
centering. Geometry and topology continue to describe the original coordinate
space rather than the fold-scaled probe representation.

## Densification policies

Some diagnostics require a dense representation. `densify_policy` controls
what happens when such a diagnostic is reached:

- `"fail"` raises instead of densifying unexpectedly.
- `"warn_and_sample"` attempts support-preserving subsampling, records the
  event, and warns when configured to do so.
- `"skip"` omits the dense-only diagnostic and records why.

Ordinary probe families must be compared on matching evidence. If
`"warn_and_sample"` selects a smaller dense-compatible cohort, all active
ordinary probes use those same rows and folds, although sparse-compatible
estimators continue to consume a sparse view. Under `"skip"`, dense-only probes
are omitted and the remaining sparse-compatible probes stay aligned on the
full capped cohort.

`max_dense_mb` is a hard dense-memory estimate, and `max_samples` is a hard row
cap. Group-aware sampling never splits a group or exceeds the row cap. If no
support-preserving sample fits, the supervised diagnostic is skipped rather
than silently dropping required classes, labels, or target structure.

Inspect `report.densification_events`, `report.sampling`, and
`report.skipped_diagnostics` to audit the effective behavior.
