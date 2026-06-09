# Decision Pipeline And Recommendation Logic

This document explains how `separatix` turns a labeled feature matrix `X` and
labels `y` into a recommendation. It is intentionally written as a methods note
for users who want to understand or justify the package's behavior.

`separatix` is designed for transparency rather than hidden optimization. It
does not claim to identify the best classifier. Instead, it summarizes whether
the observed labeled geometry looks more compatible with simple linear models,
smooth nonlinear models, local or kernel methods, more partitioned
higher-capacity models, or whether the data appear bottlenecked or unreliable.

The intended setting includes learned embeddings, but the same logic applies to
raw tabular or vectorized features as long as the task is classification.

## High-Level Flow

The current implementation follows this sequence:

1. Validate `X` and `y`, reject unsupported target structures, and encode class
   labels.
2. Record dataset audit information such as class counts, imbalance, sparsity,
   and number of classes.
3. Compute diagnostic families:
   - geometry diagnostics
   - simple probe-model diagnostics
   - neighborhood overlap diagnostics
   - boundary candidate diagnostics
   - graph fragmentation diagnostics
   - optional topology diagnostics
4. Convert the raw diagnostic outputs into a smaller set of normalized summary
   scores.
5. Apply explicit rule-based branching to produce:
   - a recommendation category
   - a confidence level
   - a visible decision path
6. Return either a plain-text recommendation or a structured
   `DiagnosticReport`.

The profiler implementation that wires these stages together lives in
[separatix/profiler.py](/Users/niklasmelton/code/Separatix/separatix/profiler.py),
and the final score aggregation and branching logic lives in
[separatix/recommendation/engine.py](/Users/niklasmelton/code/Separatix/separatix/recommendation/engine.py).

## Diagnostic Families

The recommendation engine does not act on raw coordinates alone. It combines
several different views of the labeled feature space.

### Dataset Audit

The audit step captures dataset conditions that affect trustworthiness:

- number of classes
- class counts
- class imbalance ratio
- small-class edge cases

These signals are used later when estimating recommendation reliability.

### Probe Models

`separatix` runs a small family of simple probes, including a dummy baseline
and at least a linear probe. Depending on the configured budget, it may also
run local or kernel-approximation style probes.

The probe family is not treated as a model-selection tournament. Instead, probe
performance is used as evidence about the shape of the class boundary:

- If the linear probe is close to the best observed probe, that supports a
  linear recommendation.
- If nonlinear probes improve materially over the linear probe, that supports a
  nonlinear recommendation.
- If all probes are only slightly better than the dummy baseline, that suggests
  weak usable signal or a feature/label bottleneck.

### Neighborhood Diagnostics

Neighborhood metrics estimate how mixed the classes are locally. The current
summary logic uses signals such as:

- mean local entropy
- fraction of high-entropy neighborhoods
- cross-class neighbor fraction

High local mixing is interpreted as overlap or ambiguity in the labeled feature
space.

### Boundary And Fragmentation Diagnostics

Boundary-related computations identify candidate points near class transitions,
then estimate whether the boundary appears relatively smooth or more fragmented.

High fragmentation pushes the recommendation away from smooth global models and
toward more partitioning-oriented or higher-capacity model families.

### Optional Topology Diagnostics

Topology is optional and intentionally not the first-class driver of the
package. When topology is available, it acts as supporting evidence for
nontrivial local structure rather than as the primary decision source.

## Normalized Summary Scores

The recommendation engine compresses the raw diagnostics into a small set of
scores in the `[0, 1]` range when possible.

### Signal Score

`signal_score` measures how far the best available probe rises above the dummy
baseline, normalized by the remaining headroom to perfect performance.

Interpretation:

- higher means the labels appear more predictable than a class-prior baseline
- lower means the feature space may contain weak usable signal

### Overlap Score

`overlap_score` averages several neighborhood-mixing statistics.

Interpretation:

- higher means nearby examples are more class-mixed and ambiguous
- lower means neighborhoods are more class-consistent

### Linearity Score

`linearity_score` compares the linear probe to the best available probe.

Interpretation:

- higher means the linear probe already captures most of the observed probe
  performance
- lower means some nonlinearity appears useful

### Nonlinearity Score

`nonlinearity_score` measures how much the best nonlinear probe improves over
the linear probe, normalized by the linear probe's remaining headroom.

Interpretation:

- higher means nonlinear structure appears materially useful
- near zero means nonlinear probes did not add much beyond linear

### Fragmentation Score

`fragmentation_score` comes from the boundary graph diagnostics.

Interpretation:

- higher means the class boundary looks more partitioned, irregular, or locally
  broken up
- lower means the observed boundary structure looks less fragmented

### Topology Score

When persistent-topology diagnostics are available, `topology_score` summarizes
whether there is evidence of nontrivial local structure.

Interpretation:

- higher means topology contributed evidence for more structured local geometry
- missing means topology was unavailable or skipped

### Reliability Score

`reliability_score` is a confidence support score for the diagnostic process
itself rather than a measure of class separability alone.

It starts from a high-trust default and is reduced when the run shows signs
that geometric conclusions may be unstable, incomplete, or underpowered. The
current implementation subtracts reliability for conditions such as:

- many skipped diagnostics
- many warnings
- extreme distance concentration
- very small classes
- severe class imbalance
- too few boundary samples
- unstable linear-probe estimates
- missing core probe results

This score is important because `separatix` prefers to say "the geometry is not
reliable enough to trust" rather than overstate a model recommendation.

## Recommendation Branches

The final recommendation is produced by explicit rules, in order. This ordering
matters.

### 1. Reliability Gate

If reliability is too low, the result is:

- `insufficient_data_or_unreliable_geometry`

Reasoning:

- the package avoids geometry-heavy conclusions when the diagnostics themselves
  do not look trustworthy enough

### 2. Weak-Signal Gate

If reliability is acceptable but the overall signal score is very low, the
result is:

- `feature_or_label_bottleneck_likely`

Reasoning:

- if even the best simple probe barely improves over the dummy baseline, the
  limiting factor may be the features, the labels, or irreducible overlap

### 3. Linear Sufficiency Branch

If the linearity score is very high and the nonlinearity score stays small, the
result is:

- `linear_likely_sufficient`

Reasoning:

- the linear probe already matches the best observed probe closely, so more
  complex model families may add little

### 4. High-Overlap Bottleneck Branch

If overlap is high but nonlinear gain remains limited, the result is:

- `feature_or_label_bottleneck_likely`

Reasoning:

- local class mixing is already high, but the observed nonlinear probes do not
  appear to rescue the problem much

### 5. Nonlinear Branch

If nonlinear gain is clearly present, `separatix` looks for what kind of
nonlinearity seems most plausible.

If fragmentation is high:

- `high_capacity_or_partitioning_recommended`

If topology suggests nontrivial local structure:

- `kernel_or_local_recommended`

If the best probe among the nonlinear probes is a local or kernel-style probe:

- `kernel_or_local_recommended`

Otherwise:

- `smooth_nonlinear_recommended`

Reasoning:

- not all nonlinearity points to the same modeling family
- smoother gains suggest smooth nonlinear models
- stronger local structure or fragmented boundaries suggest more local,
  kernel-like, or partitioning-oriented approaches

### 6. Mixed-Evidence Fallback

If none of the above branches dominate, the result is:

- `inconclusive`

Reasoning:

- the package prefers an explicit inconclusive result over inventing certainty

## Confidence Level

The user-facing confidence label is derived from reliability and signal:

- `high` when reliability is high and signal is reasonably strong
- `medium` when reliability is moderate
- `low` otherwise

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

This is why the implementation uses a visible score-and-rule pipeline instead
of a hidden meta-model trained to predict the "best classifier."

## Relation To Prior Work

`separatix` is related to the literature on classification complexity, dataset
geometry, and problem-characterization measures. Two particularly relevant
references are:

- Tin Kam Ho and Mitra Basu, "Complexity Measures of Supervised Classification
  Problems" ([PDF](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2002-IEEE-TPAMI-Ho-DC.pdf))
- the ACM paper at [https://dl.acm.org/doi/10.1145/3347711](https://dl.acm.org/doi/10.1145/3347711)

`separatix` does not implement those procedures directly. They are better
understood here as related work and inspiration for the broader idea that data
geometry can provide useful pretraining guidance about classification
difficulty.
