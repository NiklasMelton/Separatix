# Decision Pipeline And Recommendation Logic

This document explains how `separatix` turns a feature matrix `X` and supervised
targets `y` into a recommendation. It is intentionally written as a methods note
for users who want to understand or justify the package's behavior.

`separatix` is designed for transparency rather than hidden optimization. It
does not claim to identify the best classifier or regressor. Instead, it
summarizes whether the observed supervised geometry looks more compatible with
simple linear models, smooth nonlinear models, local or kernel methods, more
partitioned higher-capacity models, or whether the data appear bottlenecked or
unreliable.

The intended setting includes learned embeddings, but the same logic applies to
raw tabular or vectorized features. Regression diagnostics are explicit opt-in
with `target_mode="regression"` so numeric class identifiers remain categorical
by default.

## High-Level Flow

The current implementation follows this sequence:

1. Validate `X` and `y`, reject unsupported target structures, and encode class
   labels or multilabel indicators.
2. Dispatch to single-label, multilabel, or explicit regression diagnostics.
3. Record dataset audit information such as class counts, imbalance, sparsity,
   number of classes, or multilabel support statistics.
4. Compute diagnostic families:
   - single-label: geometry, probe models, neighborhood overlap, boundary
     candidates, graph fragmentation, optional topology
   - multilabel: probe models, neighborhood coherence, boundary candidates,
     graph fragmentation, optional topology
   - regression: probe models, target-neighborhood smoothness, and optional
     hard-subset topology; classification boundary and graph-fragmentation
     diagnostics are reported as not applicable
5. Build probe-family evidence and uncertainty estimates from the probe-model
   results.
6. Convert the raw diagnostic outputs into a smaller set of normalized summary
   scores for reporting.
7. Apply an uncertainty-aware recommendation policy to produce:
   - a recommendation category
   - a confidence level
   - a visible decision path
8. Return either a plain-text recommendation or a structured
   `DiagnosticReport`.

For single-label targets, the recommendation path is anchored in balanced
accuracy plus geometry-aware supporting diagnostics. For multilabel targets, the
recommendation path is anchored in `micro_f1`, `macro_f1`, and `sample_jaccard`,
without collapsing those primary metrics into a weighted aggregate. For
regression targets, the recommendation path is anchored in variance-weighted and
uniform-average R2, with normalized RMSE and target-neighborhood smoothness as
supporting diagnostics. Optional regression topology is descriptive evidence
only and cannot alter the recommendation or confidence.

The profiler implementation that wires these stages together lives in
[separatix/profiler.py](../separatix/profiler.py),
and the final score aggregation and recommendation logic lives in
[separatix/recommendation/engine.py](../separatix/recommendation/engine.py).

## Diagnostic Families

The recommendation engine does not act on raw coordinates alone. It combines
several different views of the labeled feature space.

`separatix` now has three related but distinct recommendation paths:

- a single-label path for binary and multiclass classification
- a multilabel path for binary indicator targets
- an explicit regression path for continuous single- or multi-target problems

The evidence objects and recommendation labels are tailored to the target
structure.

### Dataset Audit

The audit step captures dataset conditions that affect trustworthiness:

- number of classes or labels
- class counts or label supports
- class imbalance ratio or label density
- all-zero multilabel rows
- small-class or too-rare-label edge cases

These signals are used later when estimating recommendation reliability.

### Probe Models

`separatix` runs a small family of simple probes, including a dummy baseline,
a linear probe, a smooth quadratic/global interaction probe, and depending on
the configured budget, local or kernel-approximation style probes.

The smooth probe uses explicit quadratic features when that expansion is still
small and transparent. When dimensionality makes full quadratic expansion too
expensive, it can fall back to a low-rank polynomial-kernel approximation
instead of materializing every pairwise interaction.

The probe family is not treated as a model-selection tournament. Instead, probe
performance is used as evidence about the shape of the decision surface:

All non-dummy probe preprocessing is fitted within each validation fold.
Dense inputs are centered and scaled, sparse inputs are scaled without
centering, and random-feature maps run after scaling. Geometry and topology
remain descriptions of the original input coordinates.

All active ordinary probes use the same evaluation rows and a single
materialized validation split plan. When dense-only probes require memory-aware
subsampling, that support-preserving cohort becomes the shared cohort for the
other ordinary probes as well. Each probe records the same
`evaluation_plan_id`.

- If the linear probe is close to the strongest observed family, that supports
  a linear recommendation.
- If nonlinear probes clearly improve over linear, that supports a nonlinear
  recommendation.
- If all probes are only slightly better than the dummy or prevalence
  baseline, that suggests weak usable signal or a feature/label bottleneck.

### Neighborhood Diagnostics

Neighborhood metrics estimate how mixed the labels are locally.

For single-label targets, the current summary logic uses signals such as:

- mean local entropy
- fraction of high-entropy neighborhoods
- cross-class neighbor fraction

For multilabel targets, the current summary logic uses signals such as:

- mean neighbor Jaccard similarity
- mean neighbor Hamming distance
- local label entropy
- local label-cardinality variation

Local label entropy is the arithmetic mean across every evaluated label column;
columns with zero entropy remain in the denominator instead of being dropped.

High local mixing is interpreted as overlap or ambiguity in the labeled feature
space.

### Boundary And Fragmentation Diagnostics

Boundary-related computations identify candidate points near class transitions,
then estimate whether the boundary appears relatively smooth or more fragmented.

For multilabel targets, the current boundary candidates come from several
transparent triggers rather than from one weighted score. Those triggers
include:

- low local neighbor Jaccard
- high local neighbor Hamming distance
- high local label entropy
- high local label-cardinality variation
- disagreement between linear and local probe predictions when those
  predictions are directly comparable

High fragmentation does not directly force a recommendation on its own. Instead,
it acts as supporting evidence when the probe-model comparisons already suggest
that smoother global structure may not be enough.

### Optional Topology Diagnostics

Topology is optional and intentionally not the first-class driver of the
package. When topology is available, it acts as supporting evidence for
nontrivial local structure rather than as the primary decision source.

For single-label targets, persistent topology is computed over boundary
candidate subsets when feasible.

For multilabel targets, persistent topology is computed on two capped object
types when feasible:

- the multilabel boundary-candidate cloud
- a small capped set of high-support label-positive subsets

This is intentionally narrower than "topology over all observed label sets."
That broader construction is harder to explain, more expensive, and more
combinatorial.

## Probe-Family Evidence

The recommendation decision is anchored in probe-family evidence rather than in
raw score thresholds alone.

For each available single-label probe, `separatix` records:

- balanced accuracy
- a per-probe uncertainty estimate
- repeated-fit stability when available
- the family that probe belongs to

Probe uncertainty combines:

- a class-aware balanced-accuracy variance estimate
- repeated holdout stability when the budget includes repeated fits

Recommendation comparisons primarily use paired bootstrap deltas computed from
the shared out-of-fold predictions. The report records point deltas, paired
standard errors, 95% percentile bounds, resample counts, and the decision
method. Marginal uncertainty remains visible and is used as a per-comparison
fallback when aligned prediction evidence is unavailable. Paired OOF bootstrap
intervals account for correlated probe errors, but they are not confidence
intervals from an independent test set.

For multilabel probes, `separatix` records the corresponding evidence across
three primary metrics:

- `micro_f1`
- `macro_f1`
- `sample_jaccard`

Each metric keeps its own uncertainty estimate. 

The report then aggregates probes into three predictive families:

- `linear`
- `smooth_nonlinear`
- `local_kernel`

For each family, the report records:

- the best probe within that family
- the observed family score or per-metric family evidence
- a family-level uncertainty estimate

The report also distinguishes between:

- `raw_best_family`: the family with the highest observed probe score
- `recommended_family`: the family actually recommended after conservative
  escalation

This distinction matters because `separatix` is intentionally biased toward
simpler explanations unless a more complex family has a clear,
uncertainty-adjusted advantage.

## Normalized Summary Scores

The recommendation engine still compresses raw diagnostics into a small set of
scores in the `[0, 1]` range when possible. These scores are mainly descriptive
report summaries rather than the sole driver of the final recommendation.

The exact score names differ slightly between the single-label and multilabel
paths.

### Signal Score

`signal_score` measures how far the best available probe rises above the dummy
baseline, normalized by the remaining headroom to perfect performance.

Interpretation:

- higher means the labels appear more predictable than a class-prior baseline
- lower means the feature space may contain weak usable signal

For multilabel runs, this evidence is reported separately as:

- `signal_micro_f1`
- `signal_macro_f1`
- `signal_sample_jaccard`

### Overlap Score

`overlap_score` averages several neighborhood-mixing statistics.

Interpretation:

- higher means nearby examples are more class-mixed and ambiguous
- lower means neighborhoods are more class-consistent

For multilabel runs, the analogous summary score is
`neighborhood_coherence_score`, which is currently based on mean neighbor
Jaccard similarity.

### Linearity Score

`linearity_score` compares the linear probe to the strongest observed family.

Interpretation:

- higher means the linear probe already captures most of the observed probe
  performance
- lower means some nonlinearity appears useful

For multilabel runs, this evidence is again split across the three primary
metrics:

- `linearity_micro_f1`
- `linearity_macro_f1`
- `linearity_sample_jaccard`

### Nonlinearity Score

`nonlinearity_score` measures how much the best nonlinear family improves over
the linear probe, normalized by the linear probe's remaining headroom.

Interpretation:

- higher means nonlinear structure appears materially useful
- near zero means nonlinear probes did not add much beyond linear

The multilabel path does not currently report a single `nonlinearity_score`.
Instead, it keeps the family-comparison logic directly in the
`multilabel_recommendation_evidence` object.

### Fragmentation Score

`fragmentation_score` comes from the boundary graph diagnostics.

Interpretation:

- higher means the class boundary looks more partitioned, irregular, or locally
  broken up
- lower means the observed boundary structure looks less fragmented

### Topology Score

When topology diagnostics are available, `topology_score` summarizes whether
there is evidence of nontrivial local structure. For regression, the diagnostic
examines only points at or above the 75th percentile of normalized linear-probe
residual norm or normalized local target discontinuity. It reports
mutual-nearest-neighbor fragmentation and, when requested and available,
persistent homology.

Interpretation:

- higher means topology contributed supporting evidence for more structured
  local geometry
- missing means topology was unavailable or skipped

Regression topology is intentionally non-prescriptive. Its score and object
summaries appear in the report and decision path, but they are excluded from
recommendation selection, confidence, and reliability calculations.

### Reliability Score

`reliability_score` is a confidence support score for the diagnostic process
itself rather than a measure of class separability alone.

It is derived from evidence-quality flags rather than from a single threshold
ladder. The current implementation reduces reliability when the run shows signs
that the diagnostic evidence may be unstable, incomplete, or underpowered, for
example:

- many skipped diagnostics
- many warnings
- weak signal relative to the dummy baseline
- resubstitution fallback instead of stratified validation
- unavailable geometry diagnostics
- borderline family differences where a more complex family is numerically best
  but not clearly better

This score is important because `separatix` prefers to say "the geometry is not
reliable enough to trust" rather than overstate a model recommendation.

## Recommendation Policy

The final recommendation is not a simple threshold cascade over the summary
scores. Instead, it follows a conservative escalation policy:

1. Decide whether there is usable label signal at all.
2. If there is signal, compare probe families from simpler to more complex.
3. Escalate to a more complex family only when the evidence clearly supports
   that move.

### 1. Reliability Gate

If essential evidence is missing or the diagnostic run is too incomplete to
support a family recommendation, the result is:

- `insufficient_data_or_unreliable_geometry`

Reasoning:

- the package avoids geometry-heavy conclusions when the diagnostic process
  itself does not look trustworthy enough

### 2. Weak-Signal Gate

Before choosing any model family, `separatix` checks whether the strongest
observed probe clears a signal check against the dummy or prevalence baseline.

For single-label runs, this primarily requires the paired 95% bootstrap
interval against the dummy to exclude zero and the point gain to clear the
minimum signal margin. If that paired comparison is unavailable, the existing
95% normal-approximation check is used for that comparison only.

For multilabel runs, this requires the best predictive family to clearly beat
the dummy or prevalence baseline on at least two of:

- `micro_f1`
- `macro_f1`
- `sample_jaccard`

If that signal test fails, the result is:

- `feature_or_label_bottleneck_likely` when the neighborhoods already look as
  mixed as a label-shuffled baseline would suggest
- `inconclusive` otherwise

Reasoning:

- if even the best probe does not clearly beat the class-prior baseline, the
  package should not claim that any model family is strongly indicated

### 3. Conservative Family Escalation

If signal is present, `separatix` compares probe families in complexity order:

- `linear`
- `smooth_nonlinear`
- `local_kernel`

#### Linear Recommendation

If the linear family is statistically close enough to the strongest observed
family, the result is:

- `linear_likely_sufficient`

Reasoning:

- the package prefers not to escalate complexity when a simpler linear view is
  already competitive
- in the multilabel path, "close enough" means linear is within one standard
  error of the best family on at least two primary metrics and not clearly
  worse on the third

#### Smooth Nonlinear Recommendation

If linear is no longer sufficient, the default nonlinear recommendation is:

- `smooth_nonlinear_recommended`

Reasoning:

- smooth nonlinear structure is the default next step once linear evidence is
  no longer enough
- a small numeric lead from a local/kernel probe is not treated as decisive on
  its own
- this is one of the main situations where `raw_best_family` and
  `recommended_family` may differ
- in the multilabel path, the smooth family must clearly improve over linear on
  at least two primary metrics

#### Local Or Kernel Recommendation

`kernel_or_local_recommended` is reserved for cases where the local/kernel
family clearly beats the smooth nonlinear family after uncertainty adjustment.

Reasoning:

- this reduces brittleness from noisy probe-level wins, especially when one
  local/kernel probe edges out smooth by only a small amount
- in the multilabel path, the local/kernel family must clearly beat the smooth
  family on at least two primary metrics

### 4. High-Capacity Or Partitioning Upgrade

If the local/kernel family clearly wins and the boundary evidence also suggests
fragmented structure, the result can be upgraded to:

- `high_capacity_or_partitioning_recommended`

Reasoning:

- not all local structure implies highly partitioned boundaries
- in the multilabel path, this upgrade currently requires both a clear
  local/kernel win and sufficiently strong boundary graph fragmentation
- topology can reinforce the explanation, but it does not replace the
  fragmentation gate

### 5. Mixed-Evidence Fallback

If none of the above branches dominate, the result is:

- `inconclusive`

Reasoning:

- the package prefers an explicit inconclusive result over inventing certainty

## Confidence Level

The user-facing confidence label is derived from evidence quality:

- `high` when the signal gate clears and no cautionary evidence flags dominate
- `medium` when a recommendation is still made but cautionary flags are present
- `low` when the result is inconclusive or the diagnostics are too unreliable

This confidence label is deliberately coarse. It communicates how much trust to
place in the recommendation, not how likely a future classifier is to achieve a
specific accuracy.

## Why This Logic Exists

The logic reflects a few design choices:

- Recommendations should be understandable from the report alone.
- The package should degrade gracefully when data are sparse, small, imbalanced,
  or too expensive to analyze fully.
- Weak-signal and low-reliability states deserve their own outcomes.
- The output should suggest a rough model-family direction, not a single
  algorithm prescription.
- More complex families should need clear evidence, not just a tiny raw-score
  edge.

This is why the implementation uses a visible evidence-and-escalation pipeline
instead of a hidden meta-model trained to predict the "best classifier."

## Relation To Prior Work

`separatix` is related to the literature on classification complexity, dataset
geometry, and problem-characterization measures. Two particularly relevant
references are:

- Tin Kam Ho and Mitra Basu, "Complexity Measures of Supervised Classification
  Problems" ([PDF](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2002-IEEE-TPAMI-Ho-DC.pdf))
- Ana C. Lorena, Lu\'is P. F. Garcia, Jens Lehmann, Marcilio C. P. Souto, and
  Tin Kam Ho, "How Complex Is Your Classification Problem? A Survey on
  Measuring Classification Complexity"
  ([DOI](https://doi.org/10.1145/3347711),
  [PDF](https://dl.acm.org/doi/epdf/10.1145/3347711))

`separatix` does not implement those procedures directly. They are better
understood here as related work and inspiration for the broader idea that data
geometry can provide useful pretraining guidance about classification
difficulty.
